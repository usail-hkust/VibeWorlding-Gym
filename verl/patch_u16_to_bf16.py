#!/usr/bin/env python3
"""
Patch safetensors files: change dtype "U16" → "BF16" in file headers.
Data bytes are identical (both types are 2-byte); only the metadata tag changes.
data_offsets in safetensors are relative to the data section, so they stay valid
as long as we correctly update the 8-byte header-length field.

Usage:
    python patch_u16_to_bf16.py <bf16_ckpt_dir>
"""
import struct, json, os, sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit(f"usage: {sys.argv[0]} <bf16_ckpt_dir>")

BF16_DIR = Path(sys.argv[1])

def patch_shard(path: Path):
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_bytes = f.read(header_len)
        data_bytes = f.read()  # rest is tensor data

    header = json.loads(header_bytes.decode('utf-8'))
    changed = 0
    for name, info in header.items():
        if name == '__metadata__':
            continue
        if isinstance(info, dict) and info.get('dtype') == 'U16':
            info['dtype'] = 'BF16'
            changed += 1

    if changed == 0:
        print(f"  {path.name}: no U16 tensors, skipping")
        return

    new_header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    # safetensors spec: header should be padded to multiple of 8 bytes (optional but safe)
    pad = (8 - len(new_header_bytes) % 8) % 8
    new_header_bytes += b' ' * pad
    new_header_len = struct.pack('<Q', len(new_header_bytes))

    # Write back in-place (atomic: write to .tmp then rename)
    tmp = path.with_suffix('.safetensors.tmp')
    with open(tmp, 'wb') as f:
        f.write(new_header_len)
        f.write(new_header_bytes)
        f.write(data_bytes)
    os.replace(tmp, path)
    print(f"  {path.name}: patched {changed} tensors U16→BF16")

def main():
    shards = sorted(BF16_DIR.glob("model-*.safetensors"))
    print(f"Patching {len(shards)} shards in {BF16_DIR}")
    for shard in shards:
        patch_shard(shard)
    print("Done. Verify:")
    # Quick check on first shard
    import struct as st, json as js
    with open(shards[0], 'rb') as f:
        n = st.unpack('<Q', f.read(8))[0]
        h = js.loads(f.read(n))
    sample = [(k,v['dtype']) for k,v in h.items() if k != '__metadata__'][:3]
    for name, dtype in sample:
        print(f"  {name}: {dtype}")

if __name__ == '__main__':
    main()
