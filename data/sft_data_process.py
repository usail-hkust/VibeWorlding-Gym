"""
sft_data_process.py — 采样 log 目录 → SFT 数据集（JSON + images + parquet）一步到位

在整条流水线里的位置：

    main.py 采样  →  eval.py 打分（每个 case 产出 sft_trajectory_verified.json）
                  →  本脚本（reward 过滤 + sys_prompt 对齐 + messages 拼接
                              + 图片落盘 + 打包 parquet）
                  →  verl/run_map_gen_sft.sh 训练

★ sys_prompt 对齐（关键）：
  按 query.json / sft_trajectory.json 里的 task_setting，用
  utils/prompt.py::get_system_prompt(task_setting) 选取 system prompt，与 RL
  训练时使用的完全一致，避免 SFT/RL 之间的 prompt 漂移：
    - task_setting == "generate" → 3D world construction 的 system prompt
    -否则 (refine)              → 3D world refinement 的 system prompt

★ reward 过滤（读 sft_trajectory_verified.json 的 reward_info）：
  只保留通过 verifier 的轨迹。三条评测路由分别对应：
    - onestep_scene (construction)    → hard_pass == True
    - verified      (precise edit)    → total_reward == 1
    - unverified    (其余 refinement) → hard_pass == True

输出 parquet 三列，与 run_map_gen_sft.sh 期望的格式一致：
  messages(list<{role,content}>) / images(list<{bytes}>) / tools

用法：
  cd data
  python sft_data_process.py \
      --log_dir ../log/sample_train \
      --out_json        sft_packed/sft.json \
      --out_image_dir   sft_packed/images \
      --out_parquet_dir sft_packed \
      --val_ratio 0.05
  # 跳过 reward 过滤（全量保留，不要求 verify pass）：加 --skip_reward_filter

  接着训练：
  cd ../verl && DATA_DIR=../data/sft_packed bash run_map_gen_sft.sh
"""
import os
import re
import sys
import json
import shutil
import argparse
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# ---- 复用与 RL 训练一致的 sys_prompt（<repo-root>/utils/prompt.py）----
_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)
from prompt import get_system_prompt  # noqa: E402


# ==================== Tools Schema（Qwen3 原生 tool calling）====================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_assets",
            "description": "从 qy 轻游资产库中检索 top-K 候选资产（基于 Qwen3-Embedding-4B）。必填 entity_name（中文实体名）；可选 top_k（默认5，上限100）、size_class（大/中/小尺寸物体）、scene_limit（室内/沙地/雪地等）。返回 [{type_id, name, score, category_minor, type, size_class, native_bbox_m, description, color, ...}]。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "要检索的中文实体名"},
                    "top_k": {"type": "integer", "description": "返回候选数，默认5，上限100"},
                    "size_class": {"type": "string", "description": "大/中/小尺寸物体（慎用）"},
                    "scene_limit": {"type": "string", "description": "室内/沙地/雪地等场景限定（慎用）"},
                },
                "required": ["entity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotation_and_translation",
            "description": "用于旋转以及平移场景中已有的元件。提供现有元件的 name、pos、Extend 进行匹配定位，然后输入修改后的 pos、Extend、rotate 和 reason。",
            "parameters": {
                "type": "object",
                "properties": {
                    "corrections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original_data": {"type": "object", "description": "待修改元件的原始信息"},
                                "modified_data": {"type": "object", "description": "修改后的元件信息"},
                            },
                            "required": ["original_data", "modified_data"],
                        },
                    }
                },
                "required": ["corrections"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete",
            "description": "用于删除场景中不合理的元件。提供待删除元件的 name、pos、Extend，并给出 reason。",
            "parameters": {
                "type": "object",
                "properties": {
                    "modified_data": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["modified_data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "用于在场景中添加新元件。提供新元件的 name、pos、Extend、rotate 和 reason。",
            "parameters": {
                "type": "object",
                "properties": {
                    "modified_data": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["modified_data"],
            },
        },
    },
]


# ==================== 工具函数 ====================

def extract_reasoning(content: str) -> str:
    """从含 <think>...</think> 的文本里抽出 reasoning。"""
    m = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_final_content(content: str) -> str:
    """抽出 </think> 之后的最终文本（无 think 则返回全文）。"""
    if "</think>" in content:
        return content.split("</think>", 1)[1].strip()
    return content.strip()


def _get_task_setting(sample_dir: str, log_data: dict) -> str:
    """确定 task_setting：优先 trajectory，其次 query.json，兜底 refine。"""
    ts = log_data.get("task_setting", "")
    if ts:
        return ts
    query_file = os.path.join(sample_dir, "query.json")
    if os.path.exists(query_file):
        try:
            with open(query_file, "r", encoding="utf-8") as qf:
                return json.load(qf).get("task_setting", "refine")
        except Exception:
            pass
    return "refine"


def _resolve_image_path(img_path: str, sample_dir: str) -> str:
    """把 trajectory 里存的（可能失效的相对/临时）图片路径重定位到 sample_dir 下。

    trajectory 里的图片路径来源多样（运行时 cwd 相对路径 / data 软链目录 / 临时目录），
    但图片都已实际落在 sample_dir 下（init_image/ 、turn_N/image/ 、final_image/）。
    策略：
      1. 原路径存在 → 直接用
      2. 取 `{sample_id}/` 之后的相对段拼到 sample_dir（覆盖 turn_N/image/ 情形）
      3. init 图特例：原路径的 `.../image/xxx` 在 log 里是 `init_image/xxx`
      4. 兜底：按 basename 在 sample_dir 下递归找（唯一命中才用）
    """
    if os.path.exists(img_path):
        return img_path

    sid = os.path.basename(sample_dir.rstrip("/"))
    base = os.path.basename(img_path)

    # 2. {sid}/ 之后的相对段
    if f"/{sid}/" in img_path:
        rel = img_path.split(f"/{sid}/", 1)[1]
        cand = os.path.join(sample_dir, rel)
        if os.path.exists(cand):
            return cand
        # 3. init 图：xxx/image/yyy → init_image/yyy
        if rel.startswith("image/"):
            cand2 = os.path.join(sample_dir, "init_image", base)
            if os.path.exists(cand2):
                return cand2

    # 4. 兜底：递归找同名文件（唯一命中）
    hits = []
    for root, _dirs, files in os.walk(sample_dir):
        if base in files:
            hits.append(os.path.join(root, base))
        if len(hits) > 1:
            break
    if len(hits) == 1:
        return hits[0]

    return img_path  # 仍失败，返回原值（后续 exists 判定会过滤）


# ==================== Reward 过滤 ====================

def _check_reward_pass(sample_dir: str) -> tuple:
    """读 eval.py 写的 sft_trajectory_verified.json，判断该轨迹是否通过 verifier。

    判断：onestep_scene→hard_pass / verified→total_reward==1 / unverified→hard_pass。
    """
    verified_file = os.path.join(sample_dir, "sft_trajectory_verified.json")
    if not os.path.exists(verified_file):
        return False, "无 verified 文件（该 case 还没跑 eval.py？）"

    try:
        with open(verified_file, "r", encoding="utf-8") as f:
            verified_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False, "verified 文件解析失败"

    reward_info = verified_data.get("reward_info", {})
    vtype = reward_info.get("verifier_type", "")

    if vtype == "onestep_scene":
        if reward_info.get("hard_pass", False):
            return True, "onestep_scene hard_pass=True"
        return False, f"onestep_scene hard_pass=False (reward={reward_info.get('total_reward', 0)})"
    elif vtype == "verified":
        if reward_info.get("total_reward", 0) == 1:
            return True, "verified total_reward=1"
        return False, f"verified total_reward={reward_info.get('total_reward', 0)}"
    elif vtype == "unverified":
        if reward_info.get("hard_pass", False):
            return True, "unverified hard_pass=True"
        return False, "unverified hard_pass=False"
    else:
        if reward_info.get("hard_pass", False):
            return True, f"verifier_type={vtype!r} hard_pass=True"
        return False, f"verifier_type={vtype!r} hard_pass=False"


# ==================== 样本校验 ====================

def validate_sample(target_sample: dict, label: str) -> tuple:
    """校验样本合法性（<image> 数匹配 + assistant 轮次结构）。"""
    messages = target_sample["messages"]

    image_tag_count = sum(
        msg.get("content", "").count("<image>")
        for msg in messages
        if msg["role"] in ("user", "tool") and isinstance(msg.get("content"), str)
    )
    actual_image_len = len(target_sample.get("images", []))
    if image_tag_count != actual_image_len:
        return False, f"<image> 数 ({image_tag_count}) != 图片数 ({actual_image_len})"

    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    if not assistant_msgs:
        return False, "没有 assistant 消息"

    for pos, msg in enumerate(assistant_msgs):
        is_last = (pos == len(assistant_msgs) - 1)
        content = msg.get("content") or ""
        has_tool_calls = "<tool_call>" in content
        rm = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        reasoning = rm.group(1).strip() if rm else ""

        if is_last:
            if has_tool_calls:
                return False, "最后一轮 assistant 仍有 tool_calls（超最大轮数）"
            if not content and not reasoning:
                return False, "最后一轮 assistant 为空"
            continue
        if not reasoning and not has_tool_calls:
            return False, f"第 {pos+1} 轮 assistant reasoning 和 tool_calls 均为空"
        if not has_tool_calls:
            return False, f"第 {pos+1} 轮 assistant 缺少 tool_calls"

    return True, ""


# ==================== log → JSON 样本 ====================

def build_target_samples(log_dir: str, output_image_dir: str,
                         skip_reward_filter: bool = False) -> list:
    """遍历 log_dir，reward 过滤 + sys_prompt 对齐 + messages 拼接 + 图片复制。"""
    os.makedirs(output_image_dir, exist_ok=True)

    target_data = []
    kept = 0
    filtered = 0
    filter_reasons = {"reward_filter": 0, "json_error": 0, "validate": 0}

    for sample_id in sorted(os.listdir(log_dir)):
        sample_dir = os.path.join(log_dir, sample_id)
        if not os.path.isdir(sample_dir):
            continue
        json_file = os.path.join(sample_dir, "sft_trajectory.json")
        if not os.path.exists(json_file):
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ JSON 解析失败: {json_file}")
            filtered += 1
            filter_reasons["json_error"] += 1
            continue

        # ---- reward 过滤 ----
        reward_reason = "no_filter"
        if not skip_reward_filter:
            reward_pass, reward_reason = _check_reward_pass(sample_dir)
            if not reward_pass:
                filtered += 1
                filter_reasons["reward_filter"] += 1
                continue

        # ---- sys_prompt 对齐 RL（按 task_setting）----
        task_setting = _get_task_setting(sample_dir, log_data)
        sys_instr = get_system_prompt("generate" if task_setting == "generate" else "refine")

        target_sample = {"images": [], "tools": TOOLS_SCHEMA, "messages": []}
        target_sample["messages"].append({"role": "system", "content": sys_instr})
        label = f"{sample_id}"

        # ---- 对话轮次拼接 ----
        conversations = log_data.get("conversations", [])
        assistant_indices = [
            i for i, t in enumerate(conversations)
            if t.get("role", "").lower() == "assistant"
        ]
        first_user_idx = next(
            (i for i, t in enumerate(conversations)
             if t.get("role", "").lower() == "user"), -1
        )

        for idx, turn in enumerate(conversations):
            role = turn.get("role", "").lower()
            turn_n = turn.get("turn_n", idx + 1)

            if role == "user":
                raw_content = turn.get("content", "").replace("<image>", "<image> ")
                for img_path in turn.get("images", []):
                    resolved = _resolve_image_path(img_path, sample_dir)
                    if os.path.exists(resolved):
                        img_name = f"s{sample_id}_t{turn_n}_{os.path.basename(resolved)}"
                        new_path = os.path.join(output_image_dir, img_name)
                        # 断点续拷：已存在且非空则跳过（机器易重启，避免重拷 30G）
                        if not (os.path.exists(new_path) and os.path.getsize(new_path) > 0):
                            shutil.copy2(resolved, new_path)
                        target_sample["images"].append(os.path.abspath(new_path))
                    else:
                        print(f"⚠️ 图片不存在(重定位失败): {img_path}")

                if idx == first_user_idx:
                    target_sample["messages"].append({"role": "user", "content": raw_content})
                else:
                    # 非首轮 user = tool response → role=tool，剥去 <tool_response> 壳
                    tool_content = re.sub(
                        r'<tool_response>(.*?)</tool_response>',
                        lambda m: m.group(1).replace("\n", ""),
                        raw_content, flags=re.DOTALL,
                    ).strip()
                    target_sample["messages"].append({"role": "tool", "content": tool_content})

            elif role == "assistant":
                content = turn.get("content", "") or ""
                function_calls = turn.get("function_calls", []) or []
                content = re.sub(r'<plan>', '<think>', content)
                content = re.sub(r'</plan>', '</think>', content)
                reasoning_text = extract_reasoning(content)
                is_last_assistant = (idx == assistant_indices[-1]) if assistant_indices else True

                if function_calls:
                    tool_calls_text = "\n".join(
                        f'<tool_call>\n{json.dumps({"name": fc["name"], "arguments": fc["arguments"]}, ensure_ascii=False)}\n</tool_call>'
                        for fc in function_calls
                    )
                    content_str = (f"<think>\n{reasoning_text}\n</think>\n{tool_calls_text}"
                                   if reasoning_text else tool_calls_text)
                elif is_last_assistant:
                    final_content = extract_final_content(content)
                    if not final_content:
                        final_content = content.strip() or "场景改造已完成。"
                    content_str = (f"<think>\n{reasoning_text}\n</think>\n{final_content}"
                                   if reasoning_text else final_content)
                else:
                    content_str = content.strip()

                target_sample["messages"].append({"role": "assistant", "content": content_str})

        # ---- 校验 ----
        is_valid, reason = validate_sample(target_sample, label)
        if not is_valid:
            print(f"🗑️ [{label}] 过滤: {reason}")
            filtered += 1
            filter_reasons["validate"] += 1
            continue

        target_data.append(target_sample)
        kept += 1
        print(f"  ✅ [{label}] 保留 (task={task_setting}, {reward_reason})")

    print(f"\n✨ log→样本 完成: 保留 {kept}  过滤 {filtered}  明细 {filter_reasons}")
    return target_data


# ==================== JSON → parquet ====================

def read_image_as_bytes(image_path: str, max_pixels: int = 401408) -> bytes:
    """读图并按 max_pixels 等比缩放（Qwen3-VL 显存友好），返回 JPEG bytes。"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        new_w = max(28, (int(w * scale) // 28) * 28)
        new_h = max(28, (int(h * scale) // 28) * 28)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _write_parquet(records: list, path: str):
    if not records:
        print(f"⚠️ 无记录，跳过 {path}")
        return
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, row_group_size=len(records), compression="snappy")
    print(f"💾 {len(records)} 条 → {path}")


def json_to_parquet(target_data: list, output_parquet_dir: str,
                    val_ratio: float = 0.05, seed: int = 42):
    """内存中的 target_data（含 images 为磁盘路径）→ parquet（images 转 bytes）。"""
    os.makedirs(output_parquet_dir, exist_ok=True)

    records = []
    img_errors = 0
    for idx, sample in enumerate(target_data):
        image_dicts = []
        has_error = False
        for img_path in sample.get("images", []):
            try:
                image_dicts.append({"bytes": read_image_as_bytes(img_path)})
            except Exception as e:
                print(f"  ⚠️ [样本 {idx}] 图片失败: {img_path}: {e}")
                has_error = True
                break
        if has_error:
            img_errors += 1
            continue
        rec = {"messages": sample["messages"], "images": image_dicts}
        if "tools" in sample:
            rec["tools"] = sample["tools"]
        records.append(rec)

    if img_errors:
        print(f"  ⚠️ 图片读取失败跳过: {img_errors}")
    print(f"  📦 parquet 有效记录: {len(records)}")

    if val_ratio > 0 and len(records) > 1:
        import numpy as np
        np.random.seed(seed)
        idxs = np.random.permutation(len(records))
        val_size = max(1, int(len(records) * val_ratio))
        val_recs = [records[i] for i in idxs[:val_size].tolist()]
        train_recs = [records[i] for i in idxs[val_size:].tolist()]
        _write_parquet(train_recs, os.path.join(output_parquet_dir, "train.parquet"))
        _write_parquet(val_recs, os.path.join(output_parquet_dir, "val.parquet"))
    else:
        _write_parquet(records, os.path.join(output_parquet_dir, "train.parquet"))

    return records


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="log 目录 → SFT 数据集（JSON + images + parquet），sys_prompt 对齐 RL")
    parser.add_argument("--log_dir", required=True,
                        help="main.py 采样输出的 log 目录（含各 case 的 sft_trajectory.json + eval.py 写的 sft_trajectory_verified.json）")
    parser.add_argument("--out_json", required=True, help="输出 JSON 路径")
    parser.add_argument("--out_image_dir", required=True, help="输出图片目录（磁盘副本）")
    parser.add_argument("--out_parquet_dir", required=True, help="输出 parquet 目录")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例（0=不划分）")
    parser.add_argument("--skip_reward_filter", action="store_true",
                        help="跳过 reward 过滤（全量保留，不要求 verify pass）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"📂 log_dir: {args.log_dir}")
    print(f"   sys_prompt 来源: utils/prompt.py::get_system_prompt (与 RL 训练一致)")

    # 1. log → JSON 样本 + 图片
    target_data = build_target_samples(
        args.log_dir, args.out_image_dir, skip_reward_filter=args.skip_reward_filter)

    # 2. 写 JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(target_data, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON → {args.out_json} ({len(target_data)} 条)")

    # 3. JSON → parquet
    json_to_parquet(target_data, args.out_parquet_dir,
                    val_ratio=args.val_ratio, seed=args.seed)

    # 4. 示例预览
    if target_data:
        s = target_data[0]
        roles = " -> ".join(m["role"] for m in s["messages"])
        print(f"\n📝 示例 (第1条): system={len(s['messages'][0]['content'])}c, "
              f"图片 {len(s['images'])}, 角色: {roles}")

    print(f"\n✅ 完成：JSON={args.out_json}  images={args.out_image_dir}  parquet={args.out_parquet_dir}")


if __name__ == "__main__":
    main()
