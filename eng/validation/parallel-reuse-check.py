#!/usr/bin/env python3
"""What KV every turn of a parallel-multiturn-webui.py run reused, and which later turns reused
too little: a turn's prompt starts with its previous turn's whole prompt, so a turn that reused
less than that prompt minus SLACK tokens (default 64: room for a template's re-rendered tail)
lost its conversation's cache under concurrency.

    parallel-reuse-check.py OUT_DIR [SLACK]

Exit status 1 when any later turn reused too little.
"""
import json
import os
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out = sys.argv[1]
    slack = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    short = 0
    for i in range(64):
        path = os.path.join(out, f"conversation{i}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        cells = []
        for k, t in enumerate(rec["turns"]):
            prompt, reused = t.get("promptTokens"), t.get("kvReusedTokens")
            cell = f"{reused}/{prompt}"
            if k > 0 and prompt is not None and reused is not None:
                previous = rec["turns"][k - 1].get("promptTokens") or 0
                if reused < previous - slack:
                    short += 1
                    cell += "!"
            cells.append(cell)
        note = "" if rec["ok"] else f" failures={rec['failures']}"
        print(f"conversation{i} ok={rec['ok']} " + " ".join(cells) + note)
    print(f"later turns reusing less than the previous prompt - {slack}: {short}")
    return 1 if short else 0


if __name__ == "__main__":
    sys.exit(main())
