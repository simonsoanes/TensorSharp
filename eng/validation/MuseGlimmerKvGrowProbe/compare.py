#!/usr/bin/env python3
"""Compare two MuseGlimmerKvGrowProbe reports position by position.

    compare.py REFERENCE.json CANDIDATE.json [--logits REF.f16 CAND.f16] [--mark 2048 4096]

Reports the first step whose argmax differs, argmax agreement before and after each
--mark position, and (with --logits) the cosine similarity of the full logit vectors
in the same windows. Exit status 1 when any compared step's argmax differs from the
reference, or (with --logits) any cosine drops below --min-cos.
"""
import argparse
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--logits", nargs=2, metavar=("REF", "CAND"))
    ap.add_argument("--mark", nargs="*", type=int, default=[2048])
    ap.add_argument("--min-cos", type=float, default=0.99)
    ap.add_argument("--vocab", type=int, default=0)
    args = ap.parse_args()

    ref, cand = load(args.reference), load(args.candidate)
    if ref["PromptTokens"] != cand["PromptTokens"]:
        print("prompts differ", file=sys.stderr)
        return 2
    rs, cs = ref["Steps"], cand["Steps"]
    n = min(len(rs), len(cs))
    forced = any(c["Chosen"] != c["Argmax"] for c in cs)
    first_diff = next((i for i in range(n) if rs[i]["Argmax"] != cs[i]["Argmax"]), None)
    print(f"reference {args.reference}: scenario={ref['Scenario']} capacity->{ref['FinalCapacity']} env={ref['Environment']}")
    print(f"candidate {args.candidate}: scenario={cand['Scenario']} capacity->{cand['FinalCapacity']} env={cand['Environment']} teacher-forced={forced}")
    print(f"prompt tokens {len(ref['PromptTokens'])}, compared steps {n}")
    if first_diff is None:
        print("argmax identical at every step")
    else:
        print(f"first argmax difference at step {first_diff} (position {cs[first_diff]['Position']}): "
              f"ref {rs[first_diff]['Argmax']} cand {cs[first_diff]['Argmax']}")

    marks = sorted(args.mark)
    edges = [-1] + marks + [1 << 30]
    windows = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    def window_of(pos):
        for lo, hi in windows:
            if lo <= pos < hi:
                return (lo, hi)
        return windows[-1]

    agree = {w: [0, 0] for w in windows}
    for i in range(n):
        w = window_of(cs[i]["Position"])
        agree[w][1] += 1
        agree[w][0] += rs[i]["Argmax"] == cs[i]["Argmax"]
    for w in windows:
        a, t = agree[w]
        if t:
            print(f"  positions [{max(w[0], 0)}, {w[1] if w[1] < (1 << 30) else 'end'}): argmax agreement {a}/{t} = {a / t:.3f}")

    failed = False
    if first_diff is not None and cs[first_diff]["Position"] < marks[0]:
        # Before the first mark nothing about the grow has happened yet; any
        # difference there is a different bug (or nondeterminism) worth knowing.
        print("  NOTE: divergence precedes the first mark")

    if args.logits:
        import numpy as np
        vocab = args.vocab or None
        a = np.fromfile(args.logits[0], dtype=np.float16)
        b = np.fromfile(args.logits[1], dtype=np.float16)
        if vocab is None:
            vocab = a.size // len(rs)
        a = a.reshape(-1, vocab)[:n].astype(np.float32)
        b = b.reshape(-1, vocab)[:n].astype(np.float32)
        num = (a * b).sum(axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cos = num / np.maximum(den, 1e-30)
        for w in windows:
            idx = [i for i in range(n) if window_of(cs[i]["Position"]) == w]
            if idx:
                c = cos[idx]
                print(f"  positions [{max(w[0], 0)}, {w[1] if w[1] < (1 << 30) else 'end'}): cosine min {c.min():.5f} "
                      f"mean {c.mean():.5f} (steps {len(idx)})")
        worst = int(cos.argmin())
        print(f"  worst cosine {cos[worst]:.5f} at step {worst} position {cs[worst]['Position']}")
        if cos.min() < args.min_cos:
            failed = True
    if first_diff is not None:
        failed = True
        if forced:
            # Teacher-forced: both runs saw identical inputs, so argmax differences
            # are attributable to the forward pass alone.
            mism = sum(1 for i in range(n) if rs[i]["Argmax"] != cs[i]["Argmax"])
            print(f"  teacher-forced argmax mismatches: {mism}/{n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
