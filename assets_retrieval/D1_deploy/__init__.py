"""D1_deploy package init —— **NPU 兜底引导**。

为什么放在这里：
  生产部署应当通过 ``scripts/_npu_env.sh`` 把 ``_npu_site/sitecustomize.py``
  挂到 PYTHONPATH 上，让 ``site.main()`` 在解释器最早期完成 ``import torch_npu``
  + ``safetensors.safe_open`` patch + ``torch.cuda._lazy_init`` no-op。

  但开发期 / 排错时常常会直接 ``python -m D1_deploy.main`` 不走 shell 引导，
  这时如果跑在 NPU 机器（+cpu torch wheel + torch_npu）上，``transformers``
  在加载 4B ckpt 时会触发：

      AssertionError: Torch not compiled with CUDA enabled
      NotImplementedError: 'aten::empty_strided' on the 'CUDA' backend.

  这里做一次「import 包就尝试引导」的兜底：

    1. 检测当前是不是 +cpu torch wheel（``torch.version.cuda is None``）。
    2. 是的话尝试 ``import torch_npu`` 注册 npu backend。
    3. 同时跑 _npu_site/sitecustomize.py 里的 patch（safetensors + lazy_init），
       即便 PYTHONPATH 没指向它。

  这一步是**幂等**的：sitecustomize 本身已经 import 过的话，重跑完全无害。
  非 NPU 机器（CUDA / CPU only / 没装 torch_npu）会安静跳过，绝不抛错。
"""

import os as _os
import sys as _sys
from pathlib import Path as _Path


def _bootstrap_npu_runtime() -> None:
    """开发期兜底：若 sitecustomize 没生效，再跑一次。"""
    try:
        import torch  # noqa: F401
    except Exception:
        return  # 没装 torch 就让真正的报错由业务模块抛
    cuda_compiled_in = getattr(torch.version, "cuda", None) is not None
    if cuda_compiled_in:
        return  # 真 CUDA wheel：完全不动 npu 路径
    # CPU-only torch wheel —— 多半是 NPU 机器，尝试拉起 torch_npu。
    try:
        import torch_npu  # noqa: F401  # registers the 'npu' backend on torch
    except Exception:
        # 没装 torch_npu 也不阻断：可能是真 CPU 机器（demo / CI）。
        return

    # 把 _npu_site/sitecustomize.py 里的 patch 跑一遍（idempotent）。
    site_dir = _Path(__file__).resolve().parent / "_npu_site"
    site_py = site_dir / "sitecustomize.py"
    if not site_py.exists():
        return
    if str(site_dir) not in _sys.path:
        _sys.path.insert(0, str(site_dir))
    try:
        # 直接 import：如果已经被 PYTHONPATH 引导跑过，不会重复（module cached）；
        # 没跑过的话此时执行 _patch_safetensors_for_npu / _neutralize_cuda_lazy_init。
        import sitecustomize  # type: ignore  # noqa: F401
    except Exception as _e:
        # 兜底失败也不抛 —— 真正的错会在 InferenceEncoder 加载 ckpt 时报出来。
        if _os.environ.get("D1_DEBUG_NPU_BOOTSTRAP"):
            print(f"[D1_deploy] NPU bootstrap fallback failed: {_e!r}", flush=True)


_bootstrap_npu_runtime()
