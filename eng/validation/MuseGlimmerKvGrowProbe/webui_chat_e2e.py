#!/usr/bin/env python3
"""Drive a running TensorSharp.Server.Host through the Web UI chat API and flag
degenerate repetition in the reply.

    webui_chat_e2e.py --url http://127.0.0.1:5001 --prompt "请详细介绍最终幻想7" \
        --runs 3 --think on --max-tokens 3000 --out artifacts/muse-glimmer-e2e

Each run creates a fresh session (POST /api/sessions), posts the prompt exactly as
index.html does (POST /api/chat, SSE), and records the streamed thinking and answer
plus the done frame's token counts. A reply FAILS when a unit of 1-60 characters
repeats back to back at least --min-repeats times (the loops this was written for:
"（（（（", "respond respond", "```\\n\\n```"), or when the reply's tail has almost
no distinct 8-grams left. Exit status 1 if any run fails.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request


def post_json(url, body=None, timeout=60):
    data = json.dumps(body).encode("utf-8") if body is not None else b""
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def stream_chat(base, payload, timeout):
    req = urllib.request.Request(base + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                                 method="POST", headers={"Content-Type": "application/json"})
    thinking, text, done, frames = [], [], None, 0
    t0 = time.time()
    first = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line.startswith("data: "):
                continue
            frames += 1
            data = json.loads(line[6:])
            if data.get("thinking"):
                thinking.append(data["thinking"])
                first = first or time.time()
            if data.get("token"):
                text.append(data["token"])
                first = first or time.time()
            if data.get("replace") is not None:
                text = [data["replace"]]
            if data.get("error"):
                raise RuntimeError(data["error"])
            if data.get("done"):
                done = data
    return "".join(thinking), "".join(text), done, frames, time.time() - t0, (first - t0) if first else None


def longest_loop(s, min_repeats):
    best = None
    for m in re.compile(r"(.{1,60}?)\1{%d,}" % (min_repeats - 1), re.S).finditer(s):
        unit = m.group(1)
        if not unit.strip():
            continue  # runs of whitespace are layout, not a loop
        count = len(m.group(0)) // len(unit)
        if best is None or len(m.group(0)) > best[2]:
            best = (unit, count, len(m.group(0)), m.start())
    return best


def tail_diversity(s, n=8, tail=600):
    t = s[-tail:]
    grams = [t[i:i + n] for i in range(max(0, len(t) - n + 1))]
    return len(set(grams)) / max(1, len(grams))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5001")
    ap.add_argument("--prompt", default="请详细介绍最终幻想7")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--think", choices=["on", "off"], default="on")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--min-repeats", type=int, default=12)
    ap.add_argument("--min-tail-diversity", type=float, default=0.25)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="artifacts/muse-glimmer-e2e")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    failures = 0
    summary = []
    for run in range(args.runs):
        session = post_json(args.url + "/api/sessions").get("sessionId")
        payload = {"messages": [{"role": "user", "content": args.prompt}],
                   "maxTokens": args.max_tokens, "think": args.think == "on",
                   "sessionId": session, "newChat": True}
        thinking, text, done, frames, elapsed, ttft = stream_chat(args.url, payload, args.timeout)
        done = done or {}
        prompt_tokens = done.get("promptTokens")
        gen_tokens = done.get("tokenCount")
        whole = thinking + text
        loop = longest_loop(whole, args.min_repeats)
        div = tail_diversity(whole)
        ok = loop is None and div >= args.min_tail_diversity
        failures += 0 if ok else 1
        rec = {
            "run": run, "think": args.think, "ok": ok, "promptTokens": prompt_tokens,
            "generatedTokens": gen_tokens,
            "finalPosition": (prompt_tokens or 0) + (gen_tokens or 0),
            "kvReusedTokens": done.get("kvReusedTokens"), "tokPerSec": done.get("tokPerSec"),
            "truncated": done.get("truncated"), "elapsedSec": round(elapsed, 1),
            "ttftSec": round(ttft, 2) if ttft else None, "frames": frames,
            "thinkingChars": len(thinking), "answerChars": len(text),
            "tailDiversity": round(div, 3),
            "loop": None if loop is None else {"unit": loop[0], "count": loop[1], "chars": loop[2], "at": loop[3]},
        }
        summary.append(rec)
        with open(os.path.join(args.out, f"run{run}-{args.think}.json"), "w", encoding="utf-8") as f:
            json.dump({**rec, "thinking": thinking, "answer": text, "done": done}, f, ensure_ascii=False, indent=1)
        print(json.dumps(rec, ensure_ascii=False))
        print("  answer tail:", repr(text[-240:]))
    with open(os.path.join(args.out, f"summary-{args.think}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"{args.runs - failures}/{args.runs} runs clean")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
