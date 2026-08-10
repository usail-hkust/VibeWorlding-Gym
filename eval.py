#!/usr/bin/env python3
"""eval.py — 评测入口（转发到 verifier/eval.py）

对main.py（或 baseline/*.py）采样出的 log 目录做评测，按每个 case 的
(task_setting, verifier_type) 自动分派到三条 verifier 路线：

| 路线 | 判定 | verifier |
|------|------|----------|
| generate          | task_setting == "generate"                             | 场景级 rubric (H1~H5) |
| refine-unverified | task_setting == "refine" & verifier_type=="unverified" | rubric (H1~H4) |
| refine-verified   | task_setting == "refine" & verifier_type=="verified"   | 规则匹配 gt_map，无需 LLM |

每个 case 目录下生成 sft_trajectory_verified.json（含reward_info / reward_instruction）。

用法：
    python eval.py \
      --result_dir log/eval_test \
      --model_type gemini \
      --model_name gemini-2.5-flash

完整参数见 `python eval.py --help`。
"""

import importlib.util
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_VERIFIER_DIR = os.path.join(_REPO_ROOT, "verifier")
_UTILS_DIR = os.path.join(_REPO_ROOT, "utils")
for _p in (_VERIFIER_DIR, _UTILS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 按文件路径显式加载 verifier/eval.py，避免与本文件（同名 eval.py）互相遮蔽。
_spec = importlib.util.spec_from_file_location(
    "vibeworld_verifier_eval", os.path.join(_VERIFIER_DIR, "eval.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main()
