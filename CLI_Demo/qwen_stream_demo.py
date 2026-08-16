import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    sys.exit("需要 openai SDK：pip install openai")

try:
    import httpx
except ImportError:
    httpx = None


_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


DIM     = lambda s: _c("2", s)
ITALIC  = lambda s: _c("3;2", s)
BOLD    = lambda s: _c("1", s)
BLUE    = lambda s: _c("34", s)
GREEN   = lambda s: _c("32", s)
RED     = lambda s: _c("31", s)
MAGENTA = lambda s: _c("35", s)
CYAN    = lambda s: _c("36", s)
GREY    = lambda s: _c("90", s)

TOOL_COLOR = {"retrieve_assets": BLUE, "add": GREEN,
              "delete": RED, "rotation_and_translation": MAGENTA}


def bullet():
    """一轮的起始标记：空行 + 蓝色 ●（与 CLI_Demo on_turn_start 一致）。"""
    print()
    print(BLUE("●"), flush=True)


def branch(msg: str, color=None):
    """工具调用 / 渲染结果行：  ⎿ <msg>"""
    body = color(msg) if color else msg
    print(f"  {GREY('⎿')} {body}", flush=True)


# ── 工具调用摘要（复刻 cli.py::_summarize_tool_call）─────────────────────────────
def _names_from(items, limit=4):
    names = [str(it.get("name", "?")) for it in items if isinstance(it, dict)]
    if not names:
        return ""
    tail = f" …+{len(names) - limit} more" if len(names) > limit else ""
    return ", ".join(names[:limit]) + tail


def summarize_tool_call(name: str, args: dict) -> str:
    args = args or {}
    if name == "retrieve_assets":
        ent = args.get("entity_name", "?")
        tk = args.get("top_k")
        return f"🔍 retrieve: {ent}" + (f"  (top_k={tk})" if tk else "")
    if name == "add":
        items = args.get("modified_data", []) or []
        return f"➕ add {len(items)}: {_names_from(items)}"
    if name == "delete":
        items = args.get("modified_data", []) or []
        return f"➖ delete {len(items)}: {_names_from(items)}"
    if name == "rotation_and_translation":
        corr = args.get("corrections", []) or []
        names = _names_from([c.get("original_data", {}) for c in corr if isinstance(c, dict)])
        return f"🔄 adjust {len(corr)}: {names}"
    return f"{name}: {json.dumps(args, ensure_ascii=False)[:80]}"


# ── 工具 schema：与 main.py TOOLS_SCHEMA_GENERATE 对齐 ──────────────────────────
TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "retrieve_assets",
        "description": "从资产库检索 top-K 候选资产。必填 entity_name（中文）。",
        "parameters": {"type": "object", "properties": {
            "entity_name": {"type": "string"},
            "top_k": {"type": "integer"},
            "size_class": {"type": "string"},
            "scene_limit": {"type": "string"},
        }, "required": ["entity_name"]}}},
    {"type": "function", "function": {
        "name": "add",
        "description": ("添加新元件到场景。modified_data 为元件数组，每项需含 "
                        "name(中文名) / type_id(必须来自 retrieve_assets 返回) / "
                        "position{x,y,z} / rotation{z}。"),
        "parameters": {"type": "object", "properties": {
            "modified_data": {"type": "array", "items": {"type": "object"}},
        }, "required": ["modified_data"]}}},
    {"type": "function", "function": {
        "name": "rotation_and_translation",
        "description": "旋转/平移场景中已有元件。corrections 每项含 original_data 与 modified_data。",
        "parameters": {"type": "object", "properties": {
            "corrections": {"type": "array", "items": {"type": "object"}},
        }, "required": ["corrections"]}}},
    {"type": "function", "function": {
        "name": "delete",
        "description": "删除场景中不合理的元件。modified_data 为待删除元件数组。",
        "parameters": {"type": "object", "properties": {
            "modified_data": {"type": "array", "items": {"type": "object"}},
        }, "required": ["modified_data"]}}},
]

SYSTEM_PROMPT = """你是一个 3D 场景构建 agent。目标：根据用户描述搭出一个合理的 3D 场景。

工作流程（严格遵守）：
1. 先用 retrieve_assets 检索你需要的每一类资产（一次可以并行发多个 retrieve_assets）。
2. 拿到候选资产后，用 add 把选中的元件放进场景。**type_id 必须来自 retrieve_assets 的返回**，
   不允许自己编造。每个元件需给出 name / type_id / position{x,y,z} / rotation{z}。
3. 看到渲染图后，如发现穿模、朝向错误、间距不合理，用 rotation_and_translation 修正，
   或用 delete 删掉不合理元件。
4. 场景合理即停止调用工具，用一段话总结你搭建的场景。

布局要求：单位是米，地面为 z=0 平面。元件之间不要互相穿模，保持合理间距与朝向。
每一步都先简短说明你的思路，再调用工具。"""


# ── 资产检索（真实服务）──────────────────────────────────────────────────────
def retrieve_assets(entity_name: str, top_k: int = 5, server: str = "", timeout: float = 30.0):
    """调 /recommend/single_slot，返回 [{type_id,name,score}, ...]。"""
    if httpx is None:
        return {"error": "httpx 未安装，无法检索"}
    try:
        r = httpx.post(f"{server.rstrip('/')}/recommend/single_slot",
                       json={"entity_name": entity_name, "top_k": top_k},
                       timeout=timeout)
        r.raise_for_status()
        data = r.json()
        per = (data.get("per_entity_results") or {}).get(entity_name)
        if per is None:
            combos = data.get("combinations") or []
            per = combos[0].get("combination_list", []) if combos else []
        return [{"type_id": it.get("type_id"), "name": it.get("name"),
                 "score": round(float(it.get("score") or 0), 4)} for it in (per or [])]
    except Exception as e:
        return {"error": f"检索失败: {type(e).__name__}: {e}"}


# ── 场景状态：add / delete / rotation_and_translation 的本地应用 ─────────────────
class Scene:
    """极简场景状态：一个 element 列表（name/type_id/position/rotation）。"""

    def __init__(self):
        self.elements = []

    def add(self, items):
        added = []
        for it in items or []:
            if not isinstance(it, dict):
                continue
            el = {
                "name": it.get("name") or it.get("entity_name") or "?",
                "type_id": str(it.get("type_id") or it.get("typeId") or ""),
                "position": it.get("position") or {"x": 0, "y": 0, "z": 0},
                "rotation": it.get("rotation") or {"z": 0},
            }
            self.elements.append(el)
            added.append(el)
        return added

    def delete(self, items):
        want = {(it.get("name"), str(it.get("type_id") or "")) for it in (items or [])
                if isinstance(it, dict)}
        before = len(self.elements)
        self.elements = [e for e in self.elements
                         if (e["name"], e["type_id"]) not in want
                         and e["name"] not in {n for n, _ in want}]
        return before - len(self.elements)

    def adjust(self, corrections):
        n = 0
        for c in corrections or []:
            if not isinstance(c, dict):
                continue
            orig, mod = c.get("original_data") or {}, c.get("modified_data") or {}
            key = orig.get("name")
            for e in self.elements:
                if e["name"] == key:
                    if mod.get("position"):
                        e["position"] = mod["position"]
                    if mod.get("rotation"):
                        e["rotation"] = mod["rotation"]
                    n += 1
                    break
        return n

    def summary(self):
        if not self.elements:
            return "（场景为空）"
        return "\n".join(
            f"- {e['name']} (type_id={e['type_id']}) @ "
            f"({e['position'].get('x')}, {e['position'].get('y')}, {e['position'].get('z')}) "
            f"rot_z={e['rotation'].get('z')}"
            for e in self.elements)


# ── 渲染（真实 PCG 服务，可选）────────────────────────────────────────────────
def _scene_to_actors(scene: "Scene"):
    """Scene.elements → PCG actors。

    actor 的字段约定来自真实管线产物（turn_N/pcg_render.json），**不是**通用 glTF 风格：
      pos    [x, y, z]，单位 **厘米**（LLM 用米思考，pcg_render.actors_meter_to_cm 会 ×100）
      rot    **四元数** [x, y, z, w]（不是欧拉角）
      sca    [sx, sy, sz]
      typeId 5 位资产 id 字符串
      另需 c / col / gname / id / m 几个渲染服务必需的常量字段。
    早期版本误用 position/rotation/scale + 米制，渲染服务读不到位置，
    整个场景被挤在原点附近只剩几个可见物体。
    """
    actors = []
    for e in scene.elements:
        if not e["type_id"]:
            continue
        p = e["position"]
        # 米 → 厘米
        pos = [float(p.get("x", 0)) * 100, float(p.get("y", 0)) * 100, float(p.get("z", 0)) * 100]
        # 绕 z 轴欧拉角(度) → 四元数 [x, y, z, w]
        half = math.radians(float(e["rotation"].get("z", 0) or 0)) / 2.0
        rot = [0.0, 0.0, math.sin(half), math.cos(half)]
        actors.append({
            "c": 1, "name": e["name"], "pos": pos, "rot": rot,
            "gname": "asset0", "id": 0, "m": 0, "col": [[0, 0, 0, 255]],
            "typeId": e["type_id"], "sca": [1.0, 1.0, 1.0],
        })
    return actors


def render_scene(scene: Scene, out_dir: str, server: str, quality: str):
    """走 VibeWorlding 的 gradio_render。返回 (images, err)。失败不致命。"""
    if not scene.elements:
        return [], "场景为空，跳过渲染"
    try:
        # pcg_render 在 VibeWorlding/utils/ 下，且被当作**顶层模块**导入
        # （main.py 就是先把 utils/ 注入 sys.path 再 `from pcg_render import ...`）。
        here = os.path.dirname(os.path.abspath(__file__))
        vw = os.path.dirname(here)                      # VibeWorlding/
        for p in (os.path.join(vw, "utils"), vw):
            if p not in sys.path:
                sys.path.insert(0, p)
        from pcg_render import gradio_render            # noqa
    except Exception as e:
        return [], f"渲染模块导入失败: {type(e).__name__}: {e}"

    actors = _scene_to_actors(scene)
    if not actors:
        return [], "没有带 type_id 的元件，跳过渲染"

    os.makedirs(out_dir, exist_ok=True)
    # 落一份 actors，便于和真实管线的 turn_N/pcg_render.json 对照排查
    try:
        with open(os.path.join(os.path.dirname(out_dir), "pcg_render.json"),
                  "w", encoding="utf-8") as f:
            json.dump([{"actors": actors}], f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        images, err = gradio_render(None, actors, out_dir, quality=quality,
                                    lens=31, pcg_timeout=120, server_url=server)
        return images or [], err
    except Exception as e:
        return [], f"渲染异常: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 流式 LLM 一轮：边收边打
# ══════════════════════════════════════════════════════════════════════════════
def stream_turn(client, model, history, tools, show_raw=False):
    """跑一轮流式 chat completion，实时打印 reasoning / tool_call。

    返回 (reasoning_text, content_text, tool_calls)，tool_calls 为
    [{id,name,arguments(dict)}]。

    要点：tool_call 的 arguments 是**逐片到达**的（实测 qwen3.8-max 会把
    '{"entity_name": ' / '"木屋' / '"' / ', "top_k": ' … 分成多个 delta），
    必须按 tc.index 累积成完整 JSON 才能解析。name 通常只在该 index 的
    第一个 delta 出现，后续 delta 的 name 是 None。
    """
    stream = client.chat.completions.create(
        model=model, messages=history, tools=tools,
        temperature=0.2, top_p=1, stream=True,
    )

    reasoning_parts, content_parts = [], []
    acc = {}                 # index -> {"id","name","args"(str)}
    printed = set()          # 已打印摘要的 index（参数收全即打，不等整轮结束）
    order = []               # index 到达顺序
    in_reasoning = False

    def _try_print(idx):
        """参数 JSON 一旦完整就立刻打印这条 tool_call。

        注意：**不能**把空 args 当成「完整的 {}」。qwen3.8-max 的第一个 delta 形如
        (name='retrieve_assets', arguments='')，参数分片在后续 delta 才到；若此刻就
        判定完成，会打出 `🔍 retrieve: ?` 并把该 index 锁死，真参数全部丢失。
        所以这里只在 args 能解析成**非空** JSON 时才落地；真正无参的调用留到流结束时
        由收尾逻辑兜底。
        """
        if idx in printed:
            return
        slot = acc[idx]
        if not slot["name"]:
            return
        raw = slot["args"].strip()
        if not raw:
            return                         # 参数还没开始到
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            return                         # 还没收全，等后续 delta
        printed.add(idx)
        slot["parsed"] = args
        nonlocal in_reasoning
        if in_reasoning:                    # reasoning 段落收尾，换行再打工具
            print("\n", end="", flush=True)
            in_reasoning = False
        branch(summarize_tool_call(slot["name"], args), TOOL_COLOR.get(slot["name"]))

    for chunk in stream:
        if show_raw:
            print(GREY(f"[raw] {chunk}"), file=sys.stderr)
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta

        # 1) reasoning（qwen 走 reasoning_content）：逐 token 打字机
        rc = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
        if rc:
            if not in_reasoning:
                print("    ", end="", flush=True)
                in_reasoning = True
            reasoning_parts.append(rc)
            print(ITALIC(rc.replace("\n", "\n    ")), end="", flush=True)

        # 2) content：模型的正式回复（总结轮全在这里）
        if getattr(d, "content", None):
            if in_reasoning:
                print("\n", end="", flush=True)
                in_reasoning = False
            content_parts.append(d.content)
            print(d.content, end="", flush=True)

        # 3) tool_calls：按 index 累积 name + 分片 arguments
        for tc in (getattr(d, "tool_calls", None) or []):
            idx = tc.index if tc.index is not None else 0
            if idx not in acc:
                acc[idx] = {"id": None, "name": None, "args": ""}
                order.append(idx)
            slot = acc[idx]
            if tc.id:
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["args"] += fn.arguments
            _try_print(idx)

    if in_reasoning or content_parts:
        print(flush=True)

    # 收尾：流结束后仍未成功解析的（参数截断/非法 JSON）也要交出去
    tool_calls = []
    for idx in order:
        slot = acc[idx]
        if not slot["name"]:
            continue
        if "parsed" not in slot:
            try:
                slot["parsed"] = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                branch(f"⚠️  {slot['name']}: 参数 JSON 不完整，已跳过 "
                       f"({slot['args'][:60]}…)", RED)
                continue
            if idx not in printed:
                branch(summarize_tool_call(slot["name"], slot["parsed"]),
                       TOOL_COLOR.get(slot["name"]))
        tool_calls.append({"id": slot["id"] or f"call_{idx}",
                           "name": slot["name"], "arguments": slot["parsed"]})

    return "".join(reasoning_parts), "".join(content_parts), tool_calls


# ══════════════════════════════════════════════════════════════════════════════
# Agent loop
# ══════════════════════════════════════════════════════════════════════════════
def run(args):
    api_key = (os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
               or args.api_key)
    if not api_key:
        sys.exit("缺少 API key：export BAILIAN_API_KEY=sk-xxx  (或 --api-key)")

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    session_dir = args.session_dir or os.path.join(
        "/tmp/vibeworld_sessions", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)

    print(DIM(f"model    {args.model}"))
    print(DIM(f"session  {session_dir}"))
    print(DIM(f"retrieve {args.retrieve_server}"))
    print(DIM(f"render   {'(skipped: --no-render)' if args.no_render else args.render_server}"))
    print(DIM("─" * 100))
    print(f"{BOLD(GREEN('>'))} {args.query}")

    scene = Scene()
    history = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": args.query}]

    t0 = time.time()
    for turn in range(1, args.max_turns + 1):
        bullet()
        try:
            reasoning, content, tool_calls = stream_turn(
                client, args.model, history, TOOLS_SCHEMA, show_raw=args.raw)
        except Exception as e:
            branch(f"LLM 请求失败: {type(e).__name__}: {e}", RED)
            break

        # 记录 assistant 轮
        assistant = {"role": "assistant", "content": content or ""}
        if tool_calls:
            assistant["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"],
                              "arguments": json.dumps(tc["arguments"], ensure_ascii=False)}}
                for tc in tool_calls]
        history.append(assistant)

        if not tool_calls:
            branch("✅ 无工具调用，场景完成", GREEN)
            break

        # ── 执行工具 ──────────────────────────────────────────────────────────
        results, scene_changed = [], False
        for tc in tool_calls:
            name, a = tc["name"], tc["arguments"]
            if name == "retrieve_assets":
                out = retrieve_assets(a.get("entity_name", ""), int(a.get("top_k") or 5),
                                      args.retrieve_server)
                if isinstance(out, dict) and out.get("error"):
                    branch(f"   ↳ {out['error']}", RED)
                results.append((tc["id"], out))
            elif name == "add":
                added = scene.add(a.get("modified_data"))
                scene_changed = bool(added)
                results.append((tc["id"], {"ok": True, "added": len(added),
                                           "scene_size": len(scene.elements)}))
            elif name == "delete":
                n = scene.delete(a.get("modified_data"))
                scene_changed = bool(n)
                results.append((tc["id"], {"ok": True, "deleted": n,
                                           "scene_size": len(scene.elements)}))
            elif name == "rotation_and_translation":
                n = scene.adjust(a.get("corrections"))
                scene_changed = bool(n)
                results.append((tc["id"], {"ok": True, "adjusted": n}))
            else:
                results.append((tc["id"], {"error": f"未知工具 {name}"}))

        # ── 渲染（有场景变更时）──────────────────────────────────────────────
        images = []
        if scene_changed and not args.no_render:
            turn_dir = os.path.join(session_dir, f"turn_{turn}")
            images, err = render_scene(scene, os.path.join(turn_dir, "image"),
                                       args.render_server, args.quality)
            if images:
                branch(f"🖼  rendered · {len(images)} views", GREEN)
                for i, p in enumerate(images[:5]):
                    label = ["left", "right", "front", "back", "top"][i] if i < 5 else f"view{i+1}"
                    print(DIM(f"       {label:<6}{p}"), flush=True)
            elif err:
                branch(f"🖼  渲染跳过: {err}", RED)

        # ── 回填 tool 结果（OpenAI 严格要求每个 tool_call_id 都有 role=tool 回应）──
        for tc_id, out in results:
            history.append({"role": "tool", "tool_call_id": tc_id,
                            "content": json.dumps(out, ensure_ascii=False)[:4000]})

        # 场景现状 + 渲染图提示，作为下一轮输入
        nudge = f"当前场景（{len(scene.elements)} 个元件）：\n{scene.summary()}"
        if images:
            nudge += (f"\n\n本轮已渲染 5 视角图：{', '.join(os.path.basename(p) for p in images[:5])}。"
                      "（本 demo 为纯文本流式，未回传图片）")
        nudge += "\n\n若场景已合理，请不要再调用工具，直接用一段话总结。"
        history.append({"role": "user", "content": nudge})

    # ── 落盘 ─────────────────────────────────────────────────────────────────
    out_json = os.path.join(session_dir, "final_scene.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"query": args.query, "model": args.model,
                   "elements": scene.elements}, f, ensure_ascii=False, indent=2)

    print()
    print(DIM("─" * 100))
    print(f"{CYAN('✓')} {len(scene.elements)} elements · {time.time() - t0:.1f}s · {out_json}")


def main():
    p = argparse.ArgumentParser(
        description="qwen3.8-max 流式 reasoning + tool_call demo",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-q", "--query", default="搭建一个乡村农舍场景，有木屋、木栅栏、干草堆、水井和苹果树",
                   help="场景描述")
    p.add_argument("--model", default="qwen3.8-max", help="模型 id（默认 qwen3.8-max）")
    p.add_argument("--max-turns", type=int, default=6, help="最大推理轮数（默认 6）")
    p.add_argument("--api-key", default=None, help="百炼 key（默认读 BAILIAN_API_KEY/DASHSCOPE_API_KEY）")
    p.add_argument("--base-url", default=os.getenv(
        "BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    p.add_argument("--retrieve-server", default=os.getenv(
        "VIBEWORLD_RETRIEVE_SERVER", "http://localhost:8081"))
    p.add_argument("--render-server", default=os.getenv(
        "VIBEWORLD_RENDER_SERVER", "http://localhost:8080"))
    p.add_argument("--quality", default="低质量 (快速预览)", help="渲染质量")
    p.add_argument("--no-render", action="store_true", help="跳过 PCG 渲染（只跑 LLM + 检索）")
    p.add_argument("--session-dir", default=None, help="输出目录（默认 /tmp/vibeworld_sessions/<ts>）")
    p.add_argument("--raw", action="store_true", help="把原始 delta 流打到 stderr（调试用）")
    args = p.parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        print(f"\n{DIM('中断')}")
        sys.exit(130)


if __name__ == "__main__":
    main()
