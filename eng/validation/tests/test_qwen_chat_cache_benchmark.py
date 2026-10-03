"""Exercise incremental HTTP timing, completion integrity, and comparison gates."""
import copy
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location("qwen_chat_cache", Path(__file__).parents[1] / "qwen-chat-cache-benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class DelayedSseHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b'data: {"token":"hello"}\n\n')
        self.wfile.flush()
        time.sleep(.25)
        final = {"done": True, "tokenCount": 1, "promptTokens": 5, "kvReusedTokens": 3,
                 "truncated": False, "error": None, "aborted": False}
        self.wfile.write(("data: " + json.dumps(final) + "\n\n").encode())
        self.wfile.flush()

    def log_message(self, *args):
        pass


class QwenChatCacheBenchmarkTests(unittest.TestCase):
    def test_history_replay_uses_baseline_answers_with_current_session_and_image(self):
        class RecordingClient:
            def __init__(self):
                self.bodies = []

            def json(self, method, route):
                return {"sessionId": "new-session"} if method == "POST" else {}

            def chat(self, route, body, protocol, row):
                self.bodies.append(copy.deepcopy(body))
                row.update(answer=f"different candidate {len(self.bodies)}", done=True,
                           ttft_ms=1, cached_tokens=0, prompt_tokens=20, completion_tokens=3,
                           finish_reason="length")

        with TemporaryDirectory() as directory:
            args = SimpleNamespace(model="qwen", protocol="webui", thinking=False, max_tokens=64,
                text_prompt="text", continuation_prompt="continue", turns=3, repeats=1,
                branches=True, invalidation=True, cases=["image"], require_reuse=False,
                require_full_reuse=False, output=Path(directory) / "candidate.json",
                replay_history_from=Path(directory) / "baseline.json")
            first = {"role": "user", "content": bench.IMAGE_PROMPT, "imagePaths": ["old-upload.jpg"]}
            baseline = {"model": "qwen", "protocol": "webui", "label": "baseline", "runs": []}
            history = [first]
            initial = None
            for index in range(3):
                if index:
                    history.append({"role": "user", "content": args.continuation_prompt})
                canonical = bench.canonical_request(bench.make_body(args, history, "old-session"), "image-digest")
                baseline["runs"].append({"scenario": "image", "repetition": 1, "turn": f"turn-{index + 1}",
                    "status": "passed", "done": True, "answer": f"baseline answer {index + 1}",
                    "request_sha256": bench.digest(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode())})
                history.append({"role": "assistant", "content": f"baseline answer {index + 1}"})
                if index == 0:
                    initial = copy.deepcopy(history)
            for turn, extra in (("branch", initial + [{"role": "user", "content": "请补充说明刚才提到的细节。"}]),
                                ("invalidation", [{"role": "user", "content": bench.INVALIDATION_PROMPT}])):
                canonical = bench.canonical_request(bench.make_body(args, extra, "old-session"), "image-digest")
                baseline["runs"].append({"scenario": "image", "repetition": 1, "turn": turn,
                    "status": "passed", "done": True, "answer": "579" if turn == "invalidation" else "baseline branch",
                    "request_sha256": bench.digest(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode())})
            args.replay_history_from.write_text(json.dumps(baseline), encoding="utf-8")
            report = {"model": "qwen", "runs": [], "failures": [], "image": {"sha256": "image-digest"}}
            bench.load_replay_history(args, report)
            client = RecordingClient()
            # Invalidation tests the arithmetic invariant separately; return its exact answer here.
            original_chat = client.chat
            def chat(route, body, protocol, row):
                original_chat(route, body, protocol, row)
                if row["turn"] == "invalidation":
                    row["answer"] = "579"
            client.chat = chat
            current = dict(first, imagePaths=["new-upload.jpg"])
            bench.run_workflow(args, client, "image", 1, current, report)
            self.assertFalse(report["failures"])
            self.assertEqual(len(client.bodies), 5)
            self.assertEqual(client.bodies[2]["messages"][1]["content"], "baseline answer 1")
            self.assertEqual(client.bodies[2]["messages"][3]["content"], "baseline answer 2")
            self.assertEqual(client.bodies[3]["messages"][1]["content"], "baseline answer 1")
            self.assertEqual(client.bodies[2]["messages"][0]["imagePaths"], ["new-upload.jpg"])
            self.assertTrue(all(body["sessionId"] == "new-session" for body in client.bodies))
            self.assertFalse(report["runs"][0]["answer_matches_replay"])

            # The default still appends the candidate's actual generated answers.
            del args._replay_rows
            args.branches = args.invalidation = False
            fresh = RecordingClient()
            bench.run_workflow(args, fresh, "image", 1, current,
                               {"runs": [], "failures": [], "image": {"sha256": "image-digest"}})
            self.assertEqual(fresh.bodies[2]["messages"][1]["content"], "different candidate 1")
            self.assertEqual(fresh.bodies[2]["messages"][3]["content"], "different candidate 2")

    def test_history_replay_rejects_missing_failed_or_duplicate_baseline_requests(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            args = SimpleNamespace(replay_history_from=path, protocol="webui", turns=1,
                branches=False, invalidation=False, cases=["text"], repeats=1)
            valid = {"scenario": "text", "repetition": 1, "turn": "turn-1", "status": "passed",
                     "done": True, "answer": "answer", "request_sha256": "hash"}
            for rows in ([], [dict(valid, done=False)], [valid, valid]):
                path.write_text(json.dumps({"model": "qwen", "protocol": "webui", "runs": rows}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    bench.load_replay_history(args, {"model": "qwen"})

    def test_webui_uses_loaded_model_and_openai_names_the_model(self):
        args = SimpleNamespace(model="qwen", thinking=False, protocol="webui", max_tokens=64)
        web = bench.make_body(args, [{"role": "user", "content": "hello"}], session="session")
        self.assertNotIn("model", web)
        self.assertNotIn("backend", web)
        self.assertEqual(web["sessionId"], "session")
        self.assertEqual(web["maxTokens"], 64)
        args.protocol = "openai"
        api = bench.make_body(args, [], session="ignored")
        self.assertEqual(api["model"], "qwen")
        self.assertNotIn("sessionId", api)
        self.assertEqual(api["max_tokens"], 64)

    def test_http_ttft_observes_incremental_delta_before_completion(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), DelayedSseHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = bench.Client(f"http://127.0.0.1:{server.server_port}", 5)
            evidence = {}
            message = client.chat("/api/chat", {"messages": []}, "webui", evidence)
            self.assertEqual(message["content"], "hello")
            self.assertGreater(evidence["elapsed_ms"] - evidence["ttft_ms"], 180)
            self.assertEqual(evidence["cached_tokens"], 3)
            self.assertEqual(evidence["prefilled_tokens"], 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

    def test_openai_reasoning_is_ttft_and_answer_latency_is_separate(self):
        state = bench.StreamState("openai")
        state.add({"choices": [{"delta": {"role": "assistant"}}]}, 2)
        state.add({"choices": [{"delta": {"reasoning_content": "consider"}}]}, 5)
        state.add({"choices": [{"delta": {"content": "42"}}]}, 15)
        state.add({"choices": [{"delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 8, "completion_tokens": 3,
                             "prompt_tokens_details": {"cached_tokens": 7}}}, 20)
        state.add("[DONE]", 22)
        state.validate()
        self.assertEqual(state.result()["ttft_ms"], 5)
        self.assertEqual(state.result()["first_answer_ms"], 15)
        self.assertEqual(state.message(), {"role": "assistant", "content": "42"})

    def test_unfinished_or_failed_streams_never_pass_validation(self):
        for final in (None, {"done": True, "tokenCount": 2, "promptTokens": 3},
                      {"done": True, "tokenCount": 2, "promptTokens": 3, "kvReusedTokens": 4},
                      {"done": True, "tokenCount": 2, "promptTokens": 3, "kvReusedTokens": 0, "aborted": True},
                      {"done": True, "tokenCount": 2, "promptTokens": 3, "kvReusedTokens": 0, "error": "OOM"}):
            with self.subTest(final=final):
                state = bench.StreamState("webui")
                state.add({"token": "answer"}, 10)
                if final:
                    state.add(final, 15)
                with self.assertRaises(ValueError):
                    state.validate()

    def test_canonical_history_retains_answer_text_and_compares_image_identity(self):
        body = {"sessionId": "one", "messages": [{"role": "user", "content": "describe", "imagePaths": ["a.jpg"]},
            {"role": "assistant", "content": "actual first answer"}, {"role": "user", "content": "continue"}]}
        candidate = copy.deepcopy(body)
        candidate["sessionId"] = "two"
        candidate["messages"][0]["imagePaths"] = ["b.jpg"]
        self.assertEqual(bench.canonical_request(body, "digest"), bench.canonical_request(candidate, "digest"))
        self.assertEqual(body["messages"][0]["imagePaths"], ["a.jpg"])
        candidate["messages"][1]["content"] = "changed first answer"
        self.assertNotEqual(bench.canonical_request(body, "digest"), bench.canonical_request(candidate, "digest"))

    def test_openai_evidence_does_not_duplicate_base64_image_bytes(self):
        body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]}
        canonical = bench.canonical_request(body, "digest")
        self.assertEqual(canonical["messages"][0]["content"][0]["image_url"]["url"], "sha256:digest")
        self.assertIn("base64", body["messages"][0]["content"][0]["image_url"]["url"])

    def test_changed_history_or_incomplete_streams_do_not_qualify_speedup(self):
        old = {"scenario": "image", "repetition": 1, "turn": "turn-2", "request_sha256": "history",
               "answer_sha256": "answer", "done": True, "ttft_ms": 100, "completion_tokens": 20,
               "finish_reason": "length", "cached_tokens": 0}
        new = dict(old, ttft_ms=10, cached_tokens=15)
        report = bench.compare_reports({"runs": [old]}, {"runs": [new]})
        self.assertEqual(report["pairs"][0]["ttft_speedup"], 10)
        for changed in (dict(new, request_sha256="different-history"), dict(new, done=False), dict(new, error="OOM")):
            report = bench.compare_reports({"runs": [old]}, {"runs": [changed]})
            self.assertIsNone(report["pairs"][0]["ttft_speedup"])
        for baseline, candidate in (({"model": "first", "runs": [old]}, {"model": "other", "runs": [new]}),
                                    ({"runs": [old, old]}, {"runs": [new, new]}),
                                    ({"runs": [old]}, {"runs": [dict(new, done=False)]})):
            report = bench.compare_reports(baseline, candidate)
            self.assertFalse(report["performance_qualified"])


if __name__ == "__main__":
    unittest.main()
