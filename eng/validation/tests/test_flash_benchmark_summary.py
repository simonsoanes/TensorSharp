"""Guard benchmark qualification against missing runs and timed-token confusion."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("flash_summary", Path(__file__).parents[1] / "summarize-flash-benchmarks.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def log(decode="2,3,4"):
    lines = ["Kernel warmup completed in 2000.0 ms",
             "inference benchmark starting: prefillTokens=512 decodeTokens=3 runs=5 chunked=False decodeMode=fixed-inference"]
    for index, speed in enumerate((9999, 10, 20, 30, 40), 1):
        lines.append(f"benchmark run {index}/5: prefillMs=1000 prefillTps=512 decodeMs=100 decodeTps={speed} msPerTok=10 decodeMode=fixed-inference")
    lines.append(f"benchmark sampled tokens (untimed correctness): prefillTopToken=1 decode={decode}")
    return "\n".join(lines)


def run_metadata(start=1):
    return {"status": "completed", "exit_code": 0, "shutdown": {"exit_code": 0},
            "ggml_clean": True, "ggml_revision": "upstream", "native_sha256": "native",
            "managed_sha256": {"cli": "managed"}, "source_identity": {"groups": {"native": "source"}},
            "model_shards": [{"sha256": "model"}], "environment": {"MAX_CONTEXT": "4096"},
            "started_unix": start, "model_load_timings": [{"source": "cli_model_load", "seconds": 4}]}


class BenchmarkSummaryTests(unittest.TestCase):
    def test_first_timing_run_is_preserved_but_excluded_from_steady_median(self):
        report = summary.parse_log(log())
        self.assertFalse(report["failures"])
        self.assertEqual(report["first_run"]["decodeTps"], 9999)
        self.assertEqual(report["steady_medians_runs_2_to_5"]["decodeTps"], 25)
        self.assertEqual(len(report["rows"]), 5)
        self.assertEqual(report["kernel_warmup_seconds"], [2.])

    def test_missing_or_duplicate_timing_rows_never_qualify(self):
        for text in (log().replace("benchmark run 5/5", "omitted run 5/5"),
                     log().replace("benchmark run 5/5", "benchmark run 4/5")):
            self.assertTrue(summary.parse_log(text)["failures"])

    def test_timed_chain_does_not_establish_greedy_correctness(self):
        self.assertTrue(summary.parse_log(log().replace("(untimed correctness)", "(run1)"))["failures"])
        self.assertTrue(summary.parse_log(log("2,3"))["failures"])

    def setup_runs(self, root):
        plan = {"expected_starts": {"qwen": {"baseline-layer": 1, "candidate-tp": 1}},
                "runs": [{"family": "qwen", "profile": profile, "path": profile}
                         for profile in ("baseline-layer", "candidate-tp")]}
        for i, entry in enumerate(plan["runs"]):
            directory = root / entry["path"]
            directory.mkdir()
            (directory / "run.json").write_text(json.dumps(run_metadata(i + 1)))
            (directory / "process.log").write_text(log())
        return plan

    def test_missing_planned_start_stays_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.setup_runs(root)
            plan["expected_starts"]["qwen"]["candidate-layer"] = 1
            plan["runs"].append({"family": "qwen", "profile": "candidate-layer", "path": "absent"})
            self.assertEqual(summary.summarize(plan, root)["status"], "incomplete")

    def test_full_greedy_drift_fails_and_unqualifies_timing_ratios(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.setup_runs(root)
            report = summary.summarize(plan, root)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["loading_comparisons_same_placement"], [])
            (root / "candidate-tp/process.log").write_text(log("2,9,4"))
            report = summary.summarize(plan, root)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["throughput_comparisons"][0]["qualified_by_greedy_parity"])
            self.assertEqual(report["greedy_comparisons"][1]["first_decode_divergence"], 1)

    def test_copied_start_or_changed_binary_cannot_establish_repeatability(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.setup_runs(root)
            plan["expected_starts"]["qwen"]["candidate-tp"] = 2
            plan["runs"].append({"family": "qwen", "profile": "candidate-tp", "path": "second"})
            (root / "second").mkdir()
            (root / "second/process.log").write_text(log())
            duplicate = run_metadata(2)
            (root / "second/run.json").write_text(json.dumps(duplicate))
            self.assertEqual(summary.summarize(plan, root)["status"], "incomplete")
            duplicate["started_unix"] = 3
            duplicate["native_sha256"] = "different-native"
            (root / "second/run.json").write_text(json.dumps(duplicate))
            self.assertEqual(summary.summarize(plan, root)["status"], "incomplete")

    def test_candidate_only_plan_and_duplicate_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.setup_runs(root)
            no_control = copy.deepcopy(plan)
            del no_control["expected_starts"]["qwen"]["baseline-layer"]
            with self.assertRaisesRegex(ValueError, "control"):
                summary.summarize(no_control, root)
            plan["runs"][1]["path"] = plan["runs"][0]["path"]
            with self.assertRaisesRegex(ValueError, "independent"):
                summary.summarize(plan, root)

    def setup_same_source_runs(self, root):
        plan = self.setup_runs(root)
        plan["expected_starts"]["qwen"]["candidate-layer"] = 1
        plan["runs"].append({"family": "qwen", "profile": "candidate-layer", "path": "candidate-layer"})
        (root / "candidate-layer").mkdir()
        (root / "candidate-layer/process.log").write_text(log())
        for i, entry in enumerate(plan["runs"], 1):
            data = run_metadata(i)
            data["source_identity"]["groups"] = {name: name + "-hash" for name in
                ("native_sources", "managed_sources", "shared_build_inputs")}
            data["command"] = ["dotnet", "/candidate/Cli.dll", "--model", "/weights/model.gguf",
                               "--tp" if entry["profile"] == "candidate-tp" else "--layer-split", "2"]
            (root / entry["path"] / "run.json").write_text(json.dumps(data))
        return plan

    def test_same_source_pass_cannot_replace_original_baseline_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.setup_same_source_runs(root)
            for profile in ("candidate-layer", "candidate-tp"):
                (root / profile / "process.log").write_text(log("2,9,4"))
            report = summary.summarize(plan, root)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(sum(not pair["passed"] for pair in report["greedy_comparisons"]), 2)
            self.assertTrue(all(not pair["qualified_by_greedy_parity"] for pair in report["throughput_comparisons"]))
            self.assertTrue(report["same_source_greedy_comparisons"][0]["passed"])
            measured = report["same_source_throughput_comparisons"][0]
            self.assertTrue(measured["qualified_by_same_source_greedy_parity"])
            self.assertEqual(measured["decode_ratio"], 1.)
            text = summary.markdown(report)
            self.assertIn("**failed**", text)
            self.assertIn("candidate-tp: FAIL", text)
            self.assertIn("candidate-layer → candidate-tp: pass", text)

    def test_different_or_incomplete_build_identity_never_qualifies_same_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.setup_same_source_runs(root)
            path = root / "candidate-tp/run.json"
            good = json.loads(path.read_text())
            self.assertEqual(summary.summarize(plan, root)["status"], "passed")
            mutations = ({"native_sha256": "other"}, {"managed_sha256": {"cli": "other"}},
                         {"source_identity": {"groups": {"native_sources": "only-one-group"}}},
                         {"source_identity": {}}, {"native_sha256": ""})
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    path.write_text(json.dumps({**good, **mutation}))
                    report = summary.summarize(plan, root)
                    self.assertEqual(report["status"], "failed")
                    self.assertFalse(report["same_source_throughput_comparisons"][0]["qualified_by_same_source_greedy_parity"])

    def test_same_source_requires_matched_tokens_model_environment_and_command(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = self.setup_same_source_runs(root)
            path = root / "candidate-tp/run.json"
            good = json.loads(path.read_text())
            for mutation in ({"model_shards": [{"sha256": "other"}]},
                             {"environment": {"MAX_CONTEXT": "8192"}},
                             {"command": good["command"] + ["--chunked"]},
                             {"command": good["command"][:-1] + ["4"]},
                             {"command": []}, {"companions": {"draft": {"sha256": "other"}}}):
                with self.subTest(mutation=mutation):
                    path.write_text(json.dumps({**good, **mutation}))
                    report = summary.summarize(plan, root)
                    self.assertEqual(report["status"], "failed")
                    self.assertFalse(report["same_source_greedy_comparisons"][0]["passed"])
            path.write_text(json.dumps(good))
            (root / "candidate-tp/process.log").write_text(log().replace("prefillTokens=512", "prefillTokens=256"))
            self.assertFalse(summary.summarize(plan, root)["same_source_greedy_comparisons"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
