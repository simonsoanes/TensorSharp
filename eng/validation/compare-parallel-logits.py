#!/usr/bin/env python3
"""Compare TensorSharp TS_DUMP_LOGITS F32 vectors without requiring numpy."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    value = array("f")
    value.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        value.byteswap()
    if not value or not all(math.isfinite(x) for x in value):
        raise ValueError(f"Empty or nonfinite logits: {path}")
    return value


def compare(reference, candidate, max_absolute_error=None, max_rmse=None, min_cosine=None,
            require_same_argmax=False, max_relative_l2=None):
    if not reference or len(reference) != len(candidate):
        raise ValueError("Logit vector sizes differ or are empty")
    if not all(math.isfinite(x) for x in (*reference, *candidate)):
        raise ValueError("Nonfinite logits")
    delta = [x - y for x, y in zip(reference, candidate)]
    dot = sum(x * y for x, y in zip(reference, candidate))
    norm = math.sqrt(sum(x * x for x in reference) * sum(y * y for y in candidate))
    reference_l2 = math.sqrt(sum(x * x for x in reference))
    error_l2 = math.sqrt(sum(x * x for x in delta))
    ref_max = max(range(len(reference)), key=reference.__getitem__)
    cand_max = max(range(len(candidate)), key=candidate.__getitem__)
    absolute = sorted(abs(x) for x in delta)
    top_reference = sorted(range(len(reference)), key=reference.__getitem__, reverse=True)[:10]
    top_candidate = sorted(range(len(candidate)), key=candidate.__getitem__, reverse=True)[:10]
    report = {"count": len(reference), "all_finite": True,
              "max_absolute_error": max(abs(x) for x in delta),
              "rmse": math.sqrt(sum(x * x for x in delta) / len(delta)),
              "relative_l2_error": error_l2 / reference_l2 if reference_l2 else (0. if not error_l2 else None),
              "cosine_similarity": dot / norm if norm else None,
              "reference_argmax": ref_max, "candidate_argmax": cand_max,
              "same_argmax": ref_max == cand_max,
              "absolute_error_percentiles": {str(p): absolute[round((len(absolute) - 1) * p / 100)]
                                              for p in (50, 90, 95, 99, 100)},
              "reference_top10": [{"token": i, "logit": reference[i]} for i in top_reference],
              "candidate_top10": [{"token": i, "logit": candidate[i]} for i in top_candidate],
              "reference_top1_margin": reference[top_reference[0]] - reference[top_reference[1]] if len(reference) > 1 else None,
              "candidate_top1_margin": candidate[top_candidate[0]] - candidate[top_candidate[1]] if len(candidate) > 1 else None,
              "top10_overlap": len(set(top_reference) & set(top_candidate)),
              "thresholds": {"max_absolute_error": max_absolute_error, "max_rmse": max_rmse,
                             "min_cosine": min_cosine, "require_same_argmax": require_same_argmax},
              "failures": []}
    for field, threshold in (("max_absolute_error", max_absolute_error), ("rmse", max_rmse)):
        if threshold is not None and report[field] > threshold:
            report["failures"].append(f"{field} {report[field]} exceeds {threshold}")
    if min_cosine is not None and (report["cosine_similarity"] is None or report["cosine_similarity"] < min_cosine):
        report["failures"].append(f"Cosine similarity {report['cosine_similarity']} is below {min_cosine} or undefined")
    if require_same_argmax and not report["same_argmax"]:
        report["failures"].append("Argmax token differs")
    report["thresholds"]["max_relative_l2"] = max_relative_l2
    if max_relative_l2 is not None and (report["relative_l2_error"] is None or report["relative_l2_error"] > max_relative_l2):
        report["failures"].append(f"Relative L2 error {report['relative_l2_error']} exceeds {max_relative_l2} or is undefined")
    gated = any(value is not None for value in (max_absolute_error, max_rmse, min_cosine, max_relative_l2)) or require_same_argmax
    report["status"] = "failed" if report["failures"] else "passed" if gated else "measured_not_gated"
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--max-absolute-error", type=float)
    p.add_argument("--max-rmse", type=float)
    p.add_argument("--min-cosine", type=float)
    p.add_argument("--max-relative-l2", type=float)
    p.add_argument("--require-same-argmax", action="store_true")
    a = p.parse_args()
    for value in (a.max_absolute_error, a.max_rmse, a.max_relative_l2):
        if value is not None and (not math.isfinite(value) or value < 0):
            p.error("Error thresholds must be finite and nonnegative")
    if a.min_cosine is not None and (not math.isfinite(a.min_cosine) or not -1 <= a.min_cosine <= 1):
        p.error("min-cosine must be finite and between -1 and 1")
    reference, candidate = read(a.reference), read(a.candidate)
    report = compare(reference, candidate, a.max_absolute_error, a.max_rmse, a.min_cosine, a.require_same_argmax, a.max_relative_l2)
    report.update(reference=str(a.reference), candidate=str(a.candidate),
                  reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
                  limitations="One prefill vector on fixed input, not full-model quality or autoregressive token parity; callers must bind model and input identity.")
    encoded = json.dumps(report, indent=2) + "\n"
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(encoded)
    print(encoded)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
