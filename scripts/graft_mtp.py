"""Graft an MTP (next-token-prediction) head onto a model that shipped without one.

Bonsai-2-27B is Qwen3.8-27B made ternary, but it drops Qwen3.8's trained MTP layer. unsloth
publishes that layer as a sidecar (mtp-Qwen3.8-27B-*.gguf: blk.64.* plus its own copies of
token_embd/output). We write base tensors + ONLY the sidecar's blk.<n_layer>.* tensors, bump
block_count and set nextn_predict_layers, so llama.cpp's draft-mtp runs the head on the base
model's (ternary) embeddings and output head. Cost: ~266 MB instead of the sidecar's 1.37 GB.
The head was trained on full-precision hidden states; it still drafts well from the ternary
model's (see the README's Numbers).

    python -I graft_mtp.py --gguf-py <llama.cpp>/gguf-py base.gguf mtp-sidecar.gguf out.gguf
"""
import argparse
import os
from pathlib import Path
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("sidecar")
    p.add_argument("out")
    p.add_argument("--gguf-py", required=True, help="path to a gguf-py that knows the base's quant types")
    a = p.parse_args()
    output = Path(a.out)
    for source in (Path(a.base), Path(a.sidecar)):
        if output.resolve() == source.resolve() or (output.exists() and os.path.samefile(output, source)):
            p.error("output must be different from both input files")
    sys.path.insert(0, a.gguf_py)
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGUFValueType

    base, side = GGUFReader(a.base), GGUFReader(a.sidecar)
    arch = bytes(base.fields["general.architecture"].parts[-1]).decode()
    n_layer = int(base.fields[f"{arch}.block_count"].parts[-1][0])
    graft = [t for t in side.tensors if t.name.startswith(f"blk.{n_layer}.")]
    if not graft:
        sys.exit(f"sidecar has no blk.{n_layer}.* tensors")

    w = GGUFWriter(a.out, arch)
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for name, f in base.fields.items():
        if name in skip:
            continue
        vtype = f.types[0]
        if name == "general.alignment":
            w.add_custom_alignment(int(f.parts[-1][0]))
            continue
        if name == f"{arch}.block_count":
            w.add_uint32(name, n_layer + 1)
            continue
        if vtype == GGUFValueType.ARRAY:
            sub = f.types[-1]
            if sub == GGUFValueType.STRING:
                values = [bytes(f.parts[i]) for i in f.data]
            else:
                values = [f.parts[i][0].item() for i in f.data]
            if values:
                w.add_key_value(name, values, vtype, sub_type=sub)
        elif vtype == GGUFValueType.STRING:
            w.add_key_value(name, bytes(f.parts[-1]), vtype)
        else:
            w.add_key_value(name, f.parts[-1][0].item(), vtype)
    w.add_uint32(f"{arch}.nextn_predict_layers", 1)

    for t in list(base.tensors) + graft:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.n_bytes, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in list(base.tensors) + graft:
        w.write_tensor_data(t.data)
    w.close()
    print(f"wrote {a.out}: {len(base.tensors)} base + {len(graft)} grafted tensors "
          f"({sum(int(t.n_bytes) for t in graft) / 1e6:.0f} MB), block_count {n_layer} -> {n_layer + 1}")


if __name__ == "__main__":
    main()
