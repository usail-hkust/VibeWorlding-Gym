#!/usr/bin/env python3
"""
Convert a float32 safetensors checkpoint to bfloat16.
BF16 = top 2 bytes of F32 (same sign + exponent bits).
Simple truncation is acceptable for weight conversion.

Usage:
    python convert_f32_to_bf16.py <hf_ckpt_dir> [out_dir]
"""
import os, sys, json, struct, shutil, numpy as np
from pathlib import Path
from safetensors import safe_open
from safetensors.numpy import save_file

if len(sys.argv) < 2:
    sys.exit(f"usage: {sys.argv[0]} <hf_ckpt_dir> [out_dir]\n"
             "  <hf_ckpt_dir>: float32 HuggingFace checkpoint directory\n"
             "  [out_dir]:     defaults to <hf_ckpt_dir>/../huggingface_bf16")

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC.parent / "huggingface_bf16"

def f32_arr_to_bf16(arr: np.ndarray) -> np.ndarray:
    """Truncate float32 → bfloat16 (upper 16 bits of each f32)."""
    f32 = arr.astype(np.float32, copy=False)
    u32 = f32.view(np.uint32)
    # round-to-nearest-even (add 0x8000 + half of bit 16)
    rounding_bias = (u32 >> 16) & 1
    u32_rounded = u32 + 0x7FFF + rounding_bias
    # Handle NaN: preserve NaN
    nan_mask = ((u32 & 0x7F800000) == 0x7F800000) & ((u32 & 0x007FFFFF) != 0)
    u32_rounded[nan_mask] = u32[nan_mask] | 0x00010000  # keep NaN payload
    bf16 = (u32_rounded >> 16).astype(np.uint16)
    return bf16

def convert_shard(src_path: Path, dst_path: Path):
    tensors = {}
    with safe_open(str(src_path), framework="numpy") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if t.dtype == np.float32:
                tensors[key] = f32_arr_to_bf16(t)
            else:
                tensors[key] = t
    save_file(tensors, str(dst_path))
    print(f"  saved {dst_path.name}")

def main():
    DST.mkdir(parents=True, exist_ok=True)

    # Find all safetensors shards
    shards = sorted(SRC.glob("model-*.safetensors"))
    if not shards:
        print("No safetensors shards found, trying model.safetensors")
        shards = list(SRC.glob("model.safetensors"))
    print(f"Converting {len(shards)} shard(s) from {SRC} → {DST}")

    for shard in shards:
        print(f"  processing {shard.name} ...")
        convert_shard(shard, DST / shard.name)

    # Copy + patch config.json
    with open(SRC / "config.json") as f:
        cfg = json.load(f)
    # Update dtype fields
    for key in ("dtype",):
        if cfg.get(key) == "float32":
            cfg[key] = "bfloat16"
    for sub in ("text_config", "vision_config"):
        if sub in cfg and cfg[sub].get("dtype") == "float32":
            cfg[sub]["dtype"] = "bfloat16"
    with open(DST / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("  patched config.json dtype → bfloat16")

    # Copy remaining files
    for fname in SRC.iterdir():
        if fname.suffix in (".safetensors",) or fname.name == "config.json":
            continue
        shutil.copy2(fname, DST / fname.name)
    print("  copied tokenizer/preprocessor files")

    # Patch model.safetensors.index.json if present (no content change needed, just copy)
    idx_src = SRC / "model.safetensors.index.json"
    if idx_src.exists():
        shutil.copy2(idx_src, DST / "model.safetensors.index.json")

    print(f"\nDone! BF16 checkpoint at:\n  {DST}")
    print("Update HF_MODEL_PATH in run_map_gen_grpo.sh to point to this directory.")

if __name__ == "__main__":
    main()
