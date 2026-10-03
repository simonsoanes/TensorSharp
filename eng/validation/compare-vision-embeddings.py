#!/usr/bin/env python3
"""Compare VisionEncoderBench's length-prefixed float32 output dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np


def read_dump(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 4:
        raise ValueError(f"{path}: missing element count")
    count = struct.unpack_from("<i", raw)[0]
    if count <= 0 or len(raw) != 4 + count * 4:
        raise ValueError(f"{path}: invalid dump length or element count")
    values = np.frombuffer(raw, dtype="<f4", offset=4).astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: nonfinite embedding values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--width", type=int, required=True, help="Projector output dimension")
    parser.add_argument("--max-relative-l2", type=float, default=0.003)
    parser.add_argument("--min-row-cosine", type=float, default=0.9999)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference, candidate = read_dump(args.reference), read_dump(args.candidate)
    if args.width <= 0 or reference.shape != candidate.shape or len(reference) % args.width:
        raise ValueError("Embedding shapes differ or do not match --width")
    reference = reference.reshape(-1, args.width)
    candidate = candidate.reshape(reference.shape)
    delta = candidate - reference
    norm_reference = np.linalg.norm(reference, axis=1)
    norm_candidate = np.linalg.norm(candidate, axis=1)
    denominator = norm_reference * norm_candidate
    if np.any(denominator == 0):
        raise ValueError("Zero-norm embedding rows cannot be compared by cosine")
    cosine = np.clip(np.sum(reference * candidate, axis=1) / denominator, -1, 1)
    relative_l2 = float(np.linalg.norm(delta) / np.linalg.norm(reference))
    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "shape": list(reference.shape),
        "max_absolute_error": float(np.max(np.abs(delta))),
        "relative_l2": relative_l2,
        "minimum_row_cosine": float(np.min(cosine)),
        "mean_row_cosine": float(np.mean(cosine)),
        "worst_row_index": int(np.argmin(cosine)),
        "max_relative_l2_bound": args.max_relative_l2,
        "min_row_cosine_bound": args.min_row_cosine,
        "passed": relative_l2 <= args.max_relative_l2 and float(np.min(cosine)) >= args.min_row_cosine,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
