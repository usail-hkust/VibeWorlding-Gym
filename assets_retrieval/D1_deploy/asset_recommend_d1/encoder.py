"""推理 encoder — 加载 phase4_training_v2 训出的 ckpt 并提供 encode() 接口。

设计要点（与训练侧 phase4_training_v2/encoder_qwen3.py 严格对齐）：
  - tokenizer.padding_side = "left"
  - last_token_pool（兼容左/右 padding）
  - F.normalize(emb.float(), p=2, dim=1)
  - bf16 + flash_attention_2（CUDA 上自动启用，CPU/NPU 自动 fallback）
  - 全参 ckpt（如 best/）→ 直接 AutoModel.from_pretrained
  - LoRA ckpt（含 adapter_config.json）→ 加载 base + merge_and_unload

> 不引入 peft 之外的训练依赖（不需要 deepspeed / ms-swift / accelerate）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..common.logger import get_logger

log = get_logger(__name__)


@dataclass
class InferenceEncoderConfig:
    ckpt_dir: str
    backbone_name: str = "Qwen/Qwen3-Embedding-4B"  # LoRA ckpt fallback 用
    max_q_len: int = 1024
    max_d_len: int = 1024
    bf16: bool = True
    flash_attn: bool = True


def last_token_pool(last_hidden_states, attention_mask):
    """官方实现，兼容左 / 右 padding（与训练侧严格一致）。"""
    import torch
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.size(0), device=last_hidden_states.device),
        sequence_lengths,
    ]


class InferenceEncoder:
    """推理专用 encoder（不暴露 trainable_params / save_pretrained 等训练 API）。"""

    def __init__(self, cfg: InferenceEncoderConfig):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from transformers.utils import is_flash_attn_2_available
        except ImportError as e:
            raise ImportError(
                "InferenceEncoder 需要 torch + transformers; "
                "pip install torch transformers"
            ) from e

        self.cfg = cfg
        self._torch = torch

        # ── 设备探测：NPU > CUDA > CPU ──────────────────────────────────
        # 这个项目环境里同时存在 NPU（Ascend）部署和 CUDA 部署，逻辑必须兼容两条线。
        # CPU 上跑 4B 模型只能 demo 用（编码 6560 个 doc 大约 30+ min）。
        self._has_npu = bool(getattr(torch, "npu", None) and torch.npu.is_available())
        self._has_cuda = bool(torch.cuda.is_available())
        if self._has_npu:
            self.device = "npu:0"
        elif self._has_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
            log.warning("未检测到 GPU/NPU，将在 CPU 上推理（4B 模型 + 6560 资产 ≈ 30+min 预编码）")

        # bf16 可用性探测：
        #   - CUDA：基本所有 sm_80+（A100/3090/4090/...）都支持，sm_75 及以下退到 fp16/fp32
        #   - NPU：910B / 910Pro 支持，老型号（310B 部分）不完全支持
        #     -> 优先用 torch_npu.npu.is_bf16_supported() 探测
        #   - CPU：False（避免 cpu 上 bf16 forward 走慢路径）
        self._can_bf16 = False
        if self._has_npu:
            try:
                import torch_npu  # type: ignore  # noqa: F401
                # torch_npu 新版本暴露 is_bf16_supported；老版本可能没有
                if hasattr(torch.npu, "is_bf16_supported"):
                    self._can_bf16 = bool(torch.npu.is_bf16_supported())
                else:
                    # 没探测到的老版本：保守默认开启（与历史行为一致）
                    self._can_bf16 = True
                if not self._can_bf16:
                    log.warning("当前 NPU 不支持 bf16，自动 fallback 到 fp32")
            except Exception as e:
                log.warning("NPU bf16 探测失败 (%r)，保守开启 bf16", e)
                self._can_bf16 = True
        elif self._has_cuda:
            try:
                self._can_bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                self._can_bf16 = True
            if not self._can_bf16:
                log.warning("当前 CUDA 卡不支持 bf16，自动 fallback 到 fp32")

        # ── 加载 ckpt（自动判别全参 / LoRA） ────────────────────────────
        ckpt_dir = Path(cfg.ckpt_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"ckpt_dir 不存在: {ckpt_dir}")

        is_lora = (ckpt_dir / "adapter_config.json").exists()
        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if cfg.bf16 and self._can_bf16:
            load_kwargs["torch_dtype"] = torch.bfloat16
        if cfg.flash_attn and self._has_cuda and is_flash_attn_2_available():
            load_kwargs["attn_implementation"] = "flash_attention_2"

        # tokenizer 优先从 ckpt_dir 加载（保留训练时的特殊 token）；缺失时用 backbone 兜底。
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(ckpt_dir), trust_remote_code=True, padding_side="left",
            )
        except Exception as e:
            log.info("ckpt_dir 无 tokenizer，回退到 backbone=%s", cfg.backbone_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.backbone_name, trust_remote_code=True, padding_side="left",
            )

        if is_lora:
            log.info("检测到 LoRA ckpt，加载 base=%s 后 merge adapter", cfg.backbone_name)
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError("LoRA ckpt 需要安装 peft: pip install peft") from e
            base = AutoModel.from_pretrained(
                cfg.backbone_name,
                device_map={"": self.device},
                **load_kwargs,
            )
            self.model = PeftModel.from_pretrained(base, str(ckpt_dir)).merge_and_unload()
        else:
            log.info("加载全参 ckpt=%s (device=%s, bf16=%s, fa2=%s)",
                     ckpt_dir, self.device,
                     cfg.bf16 and self._can_bf16,
                     load_kwargs.get("attn_implementation") == "flash_attention_2")
            self.model = AutoModel.from_pretrained(
                str(ckpt_dir),
                device_map={"": self.device},
                **load_kwargs,
            )
        self.model.eval()

    # ── encode：与训练侧 encode() 一致的 last-token + L2 ──────────────
    def encode(
        self,
        sentences: Sequence[str],
        is_query: bool = False,
    ) -> "Any":
        """前向一个 batch；不切 batch（调用方需要自己切 batch）。

        sentences 必须已经 wrap 过 instruction（即对 query 端：
            "Instruct: {task}\\n{q}"），与训练侧 encode(is_query=True, instruction=...)
        的内部拼接结果保持一致。

        is_query 仅用于决定 max_length（query 端默认更短，doc 端更长）。
        """
        torch = self._torch
        max_length = self.cfg.max_q_len if is_query else self.cfg.max_d_len
        inputs = self.tokenizer(
            list(sentences), padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(self.device)
        out = self.model(**inputs)
        emb = last_token_pool(out.last_hidden_state, inputs["attention_mask"])
        # 用 fp32 做 normalize，避免 bf16 数值损失（与训练完全一致）
        return torch.nn.functional.normalize(emb.float(), p=2, dim=1)

    def encode_corpus(
        self,
        sentences: Sequence[str],
        is_query: bool = False,
        batch_size: int = 32,
    ) -> "Any":
        """批量 encode（自动切 batch + no_grad）。返回 [N, D] tensor。"""
        torch = self._torch
        outs: List[Any] = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                chunk = list(sentences[i: i + batch_size])
                outs.append(self.encode(chunk, is_query=is_query).detach())
        if not outs:
            return torch.zeros((0, 1), device=self.device)
        return torch.cat(outs, dim=0)
