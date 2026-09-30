#!/usr/bin/env python3
"""Diff two DeepSeek V4 / V4.1 tensor traces layer by layer.

The direct-CUDA engine (TS_DSV4_CUDA_TRACE_DIR) and the pure-C# CPU executor (TS_DSV4_CPU_TRACE_DIR)
write the same stage tensors under the same names, p<first position>_<stage>.f32 (raw little-endian
float32). The CPU executor is held to the PyTorch reference, so running both on the same prompt and
diffing the directories shows the first stage where an executor departs from it, and by how much.

    compare-dsv4-traces.py REFERENCE_DIR CANDIDATE_DIR [--threshold 1e-2]

Prints one line per stage, in forward order: the relative error (max |a - b| over max |b|), the
cosine similarity, and a flag on the first stage whose relative error exceeds the threshold.
"""
import argparse
import os
import re
import sys

import numpy as np

STAGE_ORDER = ["embedding", "attn_input", "q", "raw_k", "attn_out", "ffn_input", "ffn_out", "hidden"]


def stage_key(name):
    m = re.match(r"p(\d+)_(?:blk(\d+)_)?(.+)\.f32$", name)
    if not m:
        return (sys.maxsize, sys.maxsize, sys.maxsize, name)
    pos, layer, stage = int(m.group(1)), m.group(2), m.group(3)
    layer = -1 if layer is None else int(layer)
    order = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
    return (pos, layer, order, stage)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--threshold", type=float, default=1e-2)
    args = ap.parse_args()

    names = sorted(set(os.listdir(args.reference)) & set(os.listdir(args.candidate)), key=stage_key)
    names = [n for n in names if n.endswith(".f32")]
    if not names:
        print("no stage both traces wrote")
        return 2
    first_bad = None
    for name in names:
        ref = np.fromfile(os.path.join(args.reference, name), dtype=np.float32).astype(np.float64)
        cand = np.fromfile(os.path.join(args.candidate, name), dtype=np.float32).astype(np.float64)
        if ref.shape != cand.shape:
            print(f"{name:40s} shape {cand.shape} vs reference {ref.shape}")
            first_bad = first_bad or name
            continue
        scale = max(np.abs(ref).max(), 1e-30)
        rel = float(np.abs(cand - ref).max() / scale)
        denom = np.linalg.norm(ref) * np.linalg.norm(cand)
        cos = float(ref @ cand / denom) if denom > 0 else 1.0
        flag = ""
        if rel > args.threshold and first_bad is None:
            first_bad = name
            flag = "  <-- first above threshold"
        print(f"{name:40s} rel {rel:10.3e}  cos {cos:.6f}{flag}")
    only_ref = sorted(set(os.listdir(args.reference)) - set(os.listdir(args.candidate)))
    only_cand = sorted(set(os.listdir(args.candidate)) - set(os.listdir(args.reference)))
    if only_ref or only_cand:
        print(f"stages only in the reference: {len(only_ref)}, only in the candidate: {len(only_cand)}")
    return 1 if first_bad else 0


if __name__ == "__main__":
    sys.exit(main())
