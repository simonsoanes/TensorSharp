#!/usr/bin/env python3
"""Compare TS_Q4E_DRIVER_DUMP residuals captured with equal debug span cuts."""
import argparse
import json
from pathlib import Path
import re
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("--position", type=int, default=0)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
rows = []
for source in args.reference.glob("*.f32"):
    match = re.fullmatch(r"span(\d+) \[(\d+),(\d+)\) T=(\d+) pos=(\d+)\.f32", source.name)
    if not match or int(match[5]) != args.position:
        continue
    target = args.candidate / source.name
    if not target.exists():
        continue
    x = np.fromfile(source, dtype="<f4").astype(np.float64)
    y = np.fromfile(target, dtype="<f4").astype(np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Different shapes: {source.name}")
    delta = x - y
    rows.append(dict(span=int(match[1]), begin=int(match[2]), end=int(match[3]),
        tokens=int(match[4]), position=args.position, count=x.size,
        finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),
        max_abs=float(np.max(np.abs(delta))),
        relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(x), 1e-300)),
        cosine=float(np.dot(x,y) / max(np.linalg.norm(x)*np.linalg.norm(y), 1e-300))))
rows.sort(key=lambda x: x["span"])
for row in rows:
    print(json.dumps(row))
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
