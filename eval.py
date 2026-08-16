
"""
usage：
    python eval.py \
      --result_dir log/eval_test \
      --model_type gemini \
      --model_name gemini-2.5-flash

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
