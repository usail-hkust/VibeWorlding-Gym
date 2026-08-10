"""common.py — 三个 agent-scaffold baseline 的共享驱动。

设计思路（关键）：不重写 main.py 已验证的 sandbox / 工具执行 / 渲染 / 输出格式，
而是用一个 ScaffoldBot 包裹真实 LLM 客户端，把每个方法的 scaffolding 注入进去：
  - 方法专属 system prompt（构造 bot 时传入）
  - 每轮在把观测交给 policy 之前，先用一个 critic/reflect LLM 对当前渲染图打分/给反馈，
    再把反馈拼进本轮 user message（复现 reason-act-reflect / generator-critic / 视觉反馈）
  - 方法专属 max_turns 与终止引导（"满足即停手，不再调用工具"）

这样每个 case 的输出（query.json / final_map.json / sft_trajectory.json / final_image/…）
与 main.py 完全一致，可直接被 verifier/eval.py 打分。

policy 与 critic 用同一个 backbone（--model_type / --model_name 指定）。

参考文献：
  SceneWeaver    arXiv:2509.20414  (reason-act-reflect, T=10, memory l=1)
  SAGE           arXiv:2602.10116  (generator + visual/physics critic, self-refine)
  SceneAssistant arXiv:2603.12238  (visual-feedback agent, atomic ops, T_M=20)
"""
import os
import sys
import json
import logging
import argparse

# baseline/ 位于仓库根目录下一层：把仓库根与 utils/ 都加入 sys.path，
# 以便复用根目录的 main.py 以及 utils/ 下的 llm / tools / prompt 等模块。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_UTILS_DIR = os.path.join(_REPO_ROOT, "utils")
for _p in (_REPO_ROOT, _UTILS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 复用 main.py 的一切（工具 schema / 两个 runner / item_infos / 检索客户端 / prompt）
import main as M
from llm import MODEL_TYPE_MAP


# ══════════════════════════════════════════════════════════════════════════════
# ScaffoldBot：包裹真实 policy 客户端，注入 critic 反馈
# ══════════════════════════════════════════════════════════════════════════════

def _last_assistant_text(bot) -> str:
    """从一个无工具 critic bot 里取最后一条 assistant 文本。"""
    for msg in reversed(getattr(bot, "history", []) or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            c = msg.get("content")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            return (c or "").strip()
    return ""


class Controller:
    """方法控制器基类。子类覆写 system_prompt / critic_* 等即可。

    model_type / model_name 由 run_baseline 解析 argv 后注入，供 make_critic 用同一 backbone。
    """
    name = "base"
    max_turns = 8
    critic_system_prompt = None          # None → 不跑 critic，退化 vanilla loop
    feedback_header = "【评审反馈】"
    model_type = "gemini"
    model_name = "gemini-2.5-pro"

    def system_prompt(self, task_setting):
        # 默认沿用 sandbox 基础 prompt（含工具格式），子类在其上叠加方法 scaffolding
        return M.get_system_prompt("generate" if task_setting == "generate" else "refine")

    def critic_user(self, prompt):
        return ("以下是当前场景的多视角渲染图与状态，请按你的评审标准给出反馈：\n\n" + prompt)

    def make_critic(self):
        return MODEL_TYPE_MAP[self.model_type](
            model_name=self.model_name, system_instruction=self.critic_system_prompt, tools=None)


class ScaffoldBot:
    """与普通 bot 同接口（mllm / reset / history），但每轮把 critic 反馈注入观测。

    controller 提供：
      critic_system_prompt : str | None   （None=不跑 critic，退化为 vanilla loop）
      critic_user(prompt)  -> str          critic 的 user 提问模板（针对当前观测）
      feedback_header       : str          反馈在下一轮 user message 里的引导语
      make_critic()         -> bot         构造 critic 客户端（一般同 backbone、无工具）
    """

    def __init__(self, inner, controller):
        self.inner = inner
        self.controller = controller
        self._critic = None
        if getattr(controller, "critic_system_prompt", None):
            self._critic = controller.make_critic()

    # 让 run_sample_* 能读到底层 history
    @property
    def history(self):
        return self.inner.history

    @history.setter
    def history(self, v):
        self.inner.history = v

    def reset(self):
        self.inner.reset()

    def _reflect(self, prompt, image_list) -> str:
        if not self._critic or not image_list:
            return ""
        try:
            self._critic.reset()
            self._critic.mllm(self.controller.critic_user(prompt), image_list)
            return _last_assistant_text(self._critic)
        except Exception as e:
            logging.warning(f"[{self.controller.name}] critic 失败: {e}")
            return ""

    def mllm(self, prompt, image_list, role="user"):
        aug = prompt
        fb = self._reflect(prompt, image_list)
        if fb:
            aug = f"{prompt}\n\n{self.controller.feedback_header}\n{fb}"
        try:
            return self.inner.mllm(aug, image_list, role=role)
        except TypeError:
            return self.inner.mllm(aug, image_list)


# ══════════════════════════════════════════════════════════════════════════════
# 驱动：仿 main.main() 的 case 遍历，但用 ScaffoldBot + 方法专属 sys prompt / max_turns
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline(controller, argv=None):
    ap = argparse.ArgumentParser(description=f"{controller.name} baseline on VWE-Bench sandbox")
    ap.add_argument("--base_data_dir", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--model_type", default="gemini", choices=list(MODEL_TYPE_MAP.keys()))
    ap.add_argument("--model_name", default="gemini-2.5-pro")
    ap.add_argument("--server", default=None, help="Gradio 渲染服务地址")
    ap.add_argument("--retrieve_server", default=M.RETRIEVE_SERVER_DEFAULT)
    ap.add_argument("--quality", default="低质量 (快速预览)")
    ap.add_argument("--max_turns", type=int, default=controller.max_turns)
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--cases", default=None)
    ap.add_argument("--task_setting", default=None, choices=["refine", "generate"])
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    # 让 critic 与 policy 共用同一 backbone
    controller.model_type = args.model_type
    controller.model_name = args.model_name

    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler()], force=True)
    logging.info(f"===== baseline={controller.name}  backbone={args.model_name}  "
                 f"max_turns={args.max_turns} =====")

    retrieve_client = M.AssetRetrievalClient(base_url=args.retrieve_server)
    case_filter = [c.strip() for c in args.cases.split(",")] if args.cases else None

    n_done = 0
    for folder_name in sorted(os.listdir(args.base_data_dir)):
        sample_dir = os.path.join(args.base_data_dir, folder_name)
        if not os.path.isdir(sample_dir):
            continue
        if case_filter and folder_name not in case_filter:
            continue
        if args.max_cases > 0 and n_done >= args.max_cases:
            break

        case_log_dir = os.path.join(args.log_dir, folder_name)
        if os.path.exists(os.path.join(case_log_dir, "final_map.json")):
            logging.info(f"⏩ 已处理，跳过: {folder_name}")
            continue
        os.makedirs(case_log_dir, exist_ok=True)

        q_path = os.path.join(sample_dir, "query.json")
        detected = args.task_setting
        if detected is None and os.path.exists(q_path):
            try:
                detected = json.load(open(q_path, encoding="utf-8")).get("task_setting", "refine")
            except Exception:
                detected = "refine"

        sys_prompt = controller.system_prompt(detected)
        logging.info(f"\n{'='*50}\n[{controller.name}] {folder_name} (task={detected})\n{'='*50}")

        try:
            if detected == "generate":
                inner = MODEL_TYPE_MAP[args.model_type](
                    model_name=args.model_name, system_instruction=sys_prompt,
                    tools=M.TOOLS_SCHEMA_GENERATE)
                bot = ScaffoldBot(inner, controller)
                ok = M.run_sample_generate(
                    sample_dir, case_log_dir, bot=bot, retrieve_client=retrieve_client,
                    max_turns=args.max_turns, quality=args.quality, debug=args.debug,
                    sys_prompt=sys_prompt, server_url=args.server)
            else:
                inner = MODEL_TYPE_MAP[args.model_type](
                    model_name=args.model_name, system_instruction=sys_prompt,
                    tools=M.TOOLS_SCHEMA_REFINE)
                bot = ScaffoldBot(inner, controller)
                ok = M.run_sample_refine(
                    sample_dir, case_log_dir, bot=bot,
                    max_turns=args.max_turns, quality=args.quality, debug=args.debug,
                    sys_prompt=sys_prompt, server_url=args.server)
        except Exception as e:
            logging.error(f"❌ {folder_name} 异常: {e}", exc_info=True)
            continue

        if ok:
            n_done += 1
            logging.info(f"✅ [{n_done}] {folder_name} ({detected}) done")

    logging.info(f"\n{controller.name} 全部完成: {n_done} 个 case")


# ══════════════════════════════════════════════════════════════════════════════
# 共享工具：给 critic 构造一个同 backbone、无工具的客户端
# ══════════════════════════════════════════════════════════════════════════════

def make_critic_factory(model_type, model_name, system_prompt):
    def _make():
        return MODEL_TYPE_MAP[model_type](model_name=model_name,
                                          system_instruction=system_prompt, tools=None)
    return _make
