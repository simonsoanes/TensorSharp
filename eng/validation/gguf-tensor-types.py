#!/usr/bin/env python3
"""Summarize the tensor types of a (split) GGUF from its headers alone: no numpy, no gguf package,
no tensor data read. Layer indices are folded (blk.N.*), so the output reads as the model's recipe.

    gguf-tensor-types.py FIRST_SHARD.gguf [--match SUBSTRING]

Lists every shard of a split file (the -00001-of-0000N naming), then one line per folded tensor
name with its type counts, then the bytes per type.
"""
import argparse
import collections
import os
import re
import struct
import sys

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
    34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}


def read_string(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def skip_value(f, vtype):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in sizes:
        f.seek(sizes[vtype], 1)
    elif vtype == 8:
        read_string(f)
    elif vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if etype in sizes:
            f.seek(sizes[etype] * count, 1)
        else:
            for _ in range(count):
                skip_value(f, etype)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def tensors(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            skip_value(f, vtype)
        for _ in range(n_tensors):
            name = read_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            (ttype,) = struct.unpack("<I", f.read(4))
            f.read(8)  # offset
            yield name, dims, ttype


def shards(first):
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", first)
    if not m:
        return [first]
    count = int(m.group(3))
    return [f"{m.group(1)}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--match", default=None)
    args = ap.parse_args()
    folded = collections.defaultdict(collections.Counter)
    shapes = {}
    for shard in shards(args.path):
        if not os.path.exists(shard):
            print(f"missing shard {shard}", file=sys.stderr)
            return 1
        for name, dims, ttype in tensors(shard):
            if args.match and args.match not in name:
                continue
            key = re.sub(r"^blk\.\d+\.", "blk.N.", name)
            folded[key][GGML_TYPES.get(ttype, str(ttype))] += 1
            shapes.setdefault(key, dims)
    for key in sorted(folded):
        print(f"{key:48s} {dict(folded[key])} {list(shapes[key])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
