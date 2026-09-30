#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#
# TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
"""End-to-end chat checks against a RUNNING TensorAgent app, through the same loopback
API its page uses: the Mac app launched by run-mac.sh, or the simulator by run-sim.sh.

Usage:
  chat-e2e.py <app stdout log> [--scenarios fact,follow,newchat,long,think,tool,image,audio]
              [--media <dir with image.png and sample.wav>] [--out report.json]
  chat-e2e.py --base http://127.0.0.1:5000/ ...   (a TensorSharp.Server, which needs no token)

The log carries the Debug build's "entry URL" line with the port and the launch token;
every request presents that token as the cookie the page gets. Each scenario asks for
something only a working pipeline can produce: the answer to a question, a follow-up
the KV cache should serve, a new chat the shared-prefix checkpoint should start, a
number only a program the model ran can know, the title printed on an image, and a
word spoken in a recording. Every turn also reports what a user feels: the time to the
first token, the decode rate, and how much of the prompt the cache served.

Exits non-zero when any scenario fails, so it can gate a build.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid


def entry_from_log(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    found = re.findall(r"entry URL (http://127\.0\.0\.1:\d+/)\?token=([0-9a-f]+)", text)
    if not found:
        sys.exit(f"No 'entry URL' line in {path}; is this a Debug build and has the app started?")
    return found[-1]


class App:
    def __init__(self, base, token):
        self.base = base if base.endswith("/") else base + "/"
        self.cookie = f"tensoragent_token={token}" if token else None

    def request(self, method, path, body=None, content_type="application/json", timeout=60):
        data = None
        headers = {"Cookie": self.cookie} if self.cookie else {}
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            headers["Content-Type"] = content_type
        elif method == "POST":
            data = b""
        req = urllib.request.Request(self.base + path.lstrip("/"), data=data, headers=headers, method=method)
        return urllib.request.urlopen(req, timeout=timeout)

    def json(self, method, path, body=None):
        with self.request(method, path, body) as response:
            return json.loads(response.read().decode())

    def new_session(self):
        return self.json("POST", "api/sessions?conversation=new")["sessionId"]

    def upload(self, path):
        boundary = "----tensoragent" + uuid.uuid4().hex
        name = os.path.basename(path)
        payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                   f"Content-Type: application/octet-stream\r\n\r\n").encode()
        payload += open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
        with self.request("POST", "api/upload", payload, f"multipart/form-data; boundary={boundary}") as response:
            return json.loads(response.read().decode())

    def chat(self, session, history, max_tokens, think=False, timeout=900):
        """Send one turn and read the SSE stream the way the page does."""
        body = {"sessionId": session, "messages": history, "maxTokens": max_tokens, "think": think}
        start = time.monotonic()
        first = None
        answer, thinking, tools, error, done = [], [], [], None, {}
        try:
            with self.request("POST", "api/chat", body, timeout=timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line.startswith("data: "):
                        continue
                    try:
                        frame = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if "thinking" in frame and isinstance(frame["thinking"], str):
                        first = first or time.monotonic()
                        thinking.append(frame["thinking"])
                    if "token" in frame and isinstance(frame["token"], str):
                        first = first or time.monotonic()
                        answer.append(frame["token"])
                    if isinstance(frame.get("replace"), str):
                        answer = [frame["replace"]]
                    if "skill_step" in frame or "tool_calls" in frame:
                        tools.append(frame)
                    if frame.get("error"):
                        error = frame["error"]
                    if frame.get("done") is True:
                        done = frame
        except (urllib.error.URLError, TimeoutError, ConnectionError) as ex:
            error = f"transport: {ex}"
        total = time.monotonic() - start
        return {
            "answer": "".join(answer),
            "thinking": "".join(thinking),
            "tools": tools,
            "error": error,
            "ttft": (first - start) if first else None,
            "total": total,
            "tokens": done.get("tokenCount", 0),
            "tokPerSec": done.get("tokPerSec", 0.0),
            "promptTokens": done.get("promptTokens", 0),
            "reused": done.get("kvReusedTokens", 0),
            "reusePct": done.get("kvReusePercent", 0.0),
            "truncated": done.get("truncated", False),
        }


def squash(text):
    return re.sub(r"\s+", "", text or "").lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", help="the app's stdout log with the Debug 'entry URL' line")
    parser.add_argument("--base", help="talk to this base URL instead, without a token (a TensorSharp.Server)")
    parser.add_argument("--scenarios", default="fact,follow,newchat,long,think,tool,image,audio")
    parser.add_argument("--media", default=os.environ.get("TS_TEST_MEDIA_DIR", os.path.expanduser("~/work/models/testmedia")))
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.base:
        base, token = args.base, None
    elif args.log:
        base, token = entry_from_log(args.log)
    else:
        parser.error("give the app's log, or --base")
    app = App(base, token)
    wanted = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    try:
        engine = app.json("GET", "api/agent/engine")
        model = engine.get("model") or {}
        print(f"==> {app.base} · {engine.get('engine', '?')}")
        print(f"    model {model.get('id')} {model.get('state')} (prefix cache warm: {model.get('prefixCacheWarm')})")
    except (urllib.error.HTTPError, ValueError):
        engine = {}
        print(f"==> {app.base} (no TensorAgent engine route: a TensorSharp.Server)")

    rows, failures = [], []

    def turn(name, session, history, prompt, check, max_tokens=256, think=False, extra=None):
        message = {"role": "user", "content": prompt}
        message.update(extra or {})
        history.append(message)
        result = app.chat(session, history, max_tokens, think=think)
        history.append({"role": "assistant", "content": result["answer"]})
        ok, why = check(result)
        if result["error"]:
            ok, why = False, f"error: {result['error']}"
        decode = result["tokPerSec"]
        print(f"--- {name}: {'ok ' if ok else 'FAIL'} ttft {result['ttft'] or 0:.2f}s, "
              f"{result['tokens']} tokens at {decode:.1f} tok/s, total {result['total']:.1f}s, "
              f"prompt {result['promptTokens']} (reused {result['reused']}, {result['reusePct']:.1f}%)"
              + (f", {len(result['tools'])} tool events" if result["tools"] else ""))
        print(f"    answer: {result['answer'][:160]!r}")
        if not ok:
            print(f"    why: {why}")
            failures.append(name)
        rows.append({"scenario": name, "ok": ok, "why": why, **{k: v for k, v in result.items() if k != "tools"},
                     "toolEvents": len(result["tools"])})
        return result

    def contains(word):
        return lambda r: (word.lower() in r["answer"].lower(), f"expected '{word}' in the answer")

    history = []
    session = None
    if "fact" in wanted or "follow" in wanted:
        session = app.new_session()
        turn("fact", session, history, "What is the capital of France? Reply with one word.", contains("Paris"))
    if "follow" in wanted:
        def reuses(r):
            if "rome" not in r["answer"].lower():
                return False, "expected 'Rome' in the answer"
            if r["reusePct"] < 50:
                return False, f"a follow-up should reuse most of its prompt, reused {r['reusePct']:.1f}%"
            return True, ""
        turn("follow", session, history, "And the capital of Italy? One word.", reuses)
    if "newchat" in wanted:
        def warm_start(r):
            if "jupiter" not in r["answer"].lower():
                return False, "expected 'Jupiter' in the answer"
            if r["reused"] <= 0:
                return False, "a new chat should start from the shared-prefix checkpoint (0 tokens reused)"
            return True, ""
        turn("newchat", app.new_session(), [], "What is the largest planet in the solar system? One word.", warm_start)
    if "long" in wanted:
        # The decode rate needs an answer long enough for the first token not to dominate.
        turn("long", app.new_session(), [],
             "Write a 250-word story about a lighthouse keeper who finds a message in a bottle.",
             lambda r: (r["tokens"] >= 150, f"expected a long answer, got {r['tokens']} tokens"), max_tokens=400)
    if "think" in wanted:
        def reasoned(r):
            if not r["thinking"].strip():
                return False, "a thinking turn produced no reasoning"
            return ("no" in r["answer"].lower(), "expected 'no' (91 = 7 x 13)")
        turn("think", app.new_session(), [], "Is 91 a prime number? Answer yes or no.", reasoned, max_tokens=4096, think=True)
    if "tool" in wanted:
        digest = hashlib.sha256(b"tensoragent").hexdigest()[:12]

        def ran_code(r):
            if not r["tools"]:
                return False, "no tool was run"
            return (digest in r["answer"].lower(), f"expected the digest prefix {digest}")
        turn("tool", app.new_session(), [],
             "Run a shell command that prints the SHA-256 hex digest of the exact ASCII string "
             "tensoragent (no trailing newline), then reply with only the first 12 hex characters.",
             ran_code, max_tokens=1024)
    def attach(path):
        """Upload a file and describe it in a message the way the page's messageFor does."""
        uploaded = app.upload(path)
        stored, kind = uploaded["file"], uploaded.get("mediaType")
        extra = {"attachments": [{"file": stored, "fileName": uploaded.get("fileName"), "mediaType": kind}]}
        if kind == "image":
            extra.update({"imagePaths": [stored], "stillImagePaths": [stored]})
        elif kind == "audio":
            extra["audioPaths"] = [stored]
        return extra

    if "image" in wanted:
        turn("image", app.new_session(), [], "What is the title written on this banner? Reply with just the title.",
             lambda r: ("tensorsharp" in squash(r["answer"]), "expected the banner's title, TensorSharp"),
             extra=attach(os.path.join(args.media, "image.png")))
    if "audio" in wanted:
        # The engine suite's own wording (EngineParallelInferenceTests): "this recording"
        # alone gets E2B answering that it has no transcription tool.
        turn("audio", app.new_session(), [], "Transcribe the first sentence of this audio clip.",
             lambda r: ("fox" in r["answer"].lower(), "expected the pangram's 'fox'"),
             extra=attach(os.path.join(args.media, "sample.wav")))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"base": base, "engine": engine, "rows": rows, "failures": failures}, f, indent=2)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"All {len(rows)} chat scenarios passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
