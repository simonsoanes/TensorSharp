#!/usr/bin/env python3
"""Compare two ParityHarness --ppl logs chunk by chunk.

Each log prints the running perplexity after every chunk; the NLL of chunk k alone is
k*ln(P_k) - (k-1)*ln(P_{k-1}). Both runs score the same tokens in the same chunks, so the
per-chunk differences are paired samples: their mean says which backend predicts the text better,
and its standard error says whether the difference is larger than the two backends' numerics
disagree by from chunk to chunk.

    paired-ppl-chunks.py REFERENCE.log CANDIDATE.log
"""
import math
import re
import sys


def chunk_nll(path):
    running = [float(m.group(1)) for m in re.finditer(r"running PPL = ([0-9.]+)", open(path).read())]
    out, prev = [], 0.0
    for k, ppl in enumerate(running, start=1):
        total = k * math.log(ppl)
        out.append(total - prev)
        prev = total
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ref, cand = chunk_nll(sys.argv[1]), chunk_nll(sys.argv[2])
    n = min(len(ref), len(cand))
    if n < 2:
        print("need at least two chunks in both logs")
        return 2
    diffs = [cand[i] - ref[i] for i in range(n)]
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    for i in range(n):
        print(f"chunk {i + 1:2d}  reference {math.exp(ref[i]):8.4f}  candidate {math.exp(cand[i]):8.4f}  "
              f"nll diff {diffs[i]:+.4f}")
    ppl_ref = math.exp(sum(ref[:n]) / n)
    ppl_cand = math.exp(sum(cand[:n]) / n)
    print(f"{n} chunks: PPL reference {ppl_ref:.4f}, candidate {ppl_cand:.4f} ({(ppl_cand / ppl_ref - 1) * 100:+.2f}%)")
    print(f"mean NLL difference per chunk {mean:+.4f} nats (SE {se:.4f}); "
          f"{'within' if abs(mean) <= 2 * se else 'outside'} two standard errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
