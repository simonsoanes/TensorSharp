"""Numerical acceptance must fail on incomplete evidence and token divergence."""
import importlib.util
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[1] / "qwen4exp-expert-cache.py"
spec = importlib.util.spec_from_file_location("qwen4exp_expert_cache", path)
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class ExpertCacheComparisonTests(unittest.TestCase):
    def capture(self, rows):
        return {"model_sha256": "same-model", "native_sha256": "same-native", "captures": {"a1": rows}}

    def test_complete_identical_logits_pass(self):
        self.assertTrue(cache.compare(self.capture([[1.0, 2.0], [4.0, -1.0]]),
                                      self.capture([[1.0, 2.0], [4.0, -1.0]]), strict=True)["passed"])

    def test_small_numeric_difference_cannot_hide_an_argmax_change(self):
        result = cache.compare(self.capture([[1.0, 1.0000001]]), self.capture([[1.0000001, 1.0]]), strict=True)
        self.assertLess(result["scenarios"]["a1"]["worst_relative_l2"], 1e-6)
        self.assertFalse(result["passed"])

    def test_same_token_does_not_hide_numerical_regression(self):
        self.assertFalse(cache.compare(self.capture([[1.0, 3.0]]), self.capture([[1.0, 3.1]]), strict=True)["passed"])

    def test_incomplete_or_nonfinite_logits_fail(self):
        for rows in ([], [[]], [[float("nan"), 2.0]], [[1.0]], [[1.0, 2.0], [1.0, 2.0]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                cache.difference(rows, [[1.0, 2.0]])

    def test_checkpoint_and_binary_identity_must_match(self):
        for key in ("model_sha256", "native_sha256", "managed_assemblies_sha256"):
            left, right = self.capture([[1.0, 2.0]]), self.capture([[1.0, 2.0]])
            right[key] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                cache.compare(left, right, strict=True)

    def test_host_rounding_report_does_not_claim_strict_acceptance(self):
        result = cache.compare(self.capture([[1.0, 3.0]]), self.capture([[1.0, 3.1]]), strict=False)
        self.assertIsNone(result["passed"])
        self.assertIsNone(result["bound_relative_l2"])

    def real_capture(self, mode="greedy"):
        generated = [7, 8, 9] if mode == "greedy" else []
        forced = None if mode == "greedy" else [10, 11]
        return {"synthetic": False, "model_sha256": None, "model_path": str(path.parent / "checkpoint.gguf"),
                "model_bytes": 1234, "native_sha256": "same-native", "managed_assemblies_sha256": {"models.dll": "same-managed"},
                "captures": None, "final_logits": [1.0, 2.0, 3.0], "vocabulary_size": 3,
                "decode_mode": mode, "prompt_tokens": [1, 2], "forced_tokens": forced, "generated_tokens": generated,
                "runs": [{"warmup": warmup, "prefill_tokens": 2, "decode_tokens": 2,
                          "generated_tokens": list(generated), "finish_reason": "eos" if mode == "greedy" else "teacher-forced",
                          "final_logit_sha256": "same-final"} for warmup in (True, False, False)]}

    def test_real_greedy_acceptance_is_final_vocabulary_only(self):
        result = cache.compare(self.real_capture(), self.real_capture(), strict=True)
        self.assertTrue(result["passed"])
        self.assertEqual("final-vocabulary-only", result["capture_scope"])
        self.assertEqual(1, result["scenarios"]["final"]["rows"])
        self.assertTrue(result["conditioning"]["all_greedy_repetitions_eos"])
        self.assertFalse(result["language_quality_validated"])

    def test_real_teacher_forced_acceptance_requires_identical_forced_ids(self):
        left, right = self.real_capture("teacher-forced"), self.real_capture("teacher-forced")
        self.assertTrue(cache.compare(left, right, strict=True)["passed"])
        right["forced_tokens"][0] += 1
        with self.assertRaisesRegex(ValueError, "conditioning differs"):
            cache.compare(left, right, strict=True)

    def test_real_missing_incomplete_or_nonfinite_final_capture_fails(self):
        for logits in (None, [], [1.0, 2.0], [float("nan"), 2.0, 3.0], [1.0, float("inf"), 3.0], [1.0, "2", 3.0]):
            left = self.real_capture()
            left["final_logits"] = logits
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                cache.compare(left, self.real_capture(), strict=True)

    def test_real_shared_null_hash_does_not_hide_checkpoint_or_binary_changes(self):
        for key, value in (("model_path", str(path.parent / "other.gguf")), ("model_bytes", 2345),
                           ("native_sha256", "different"), ("managed_assemblies_sha256", {"models.dll": "different"}),
                           ("model_path", None), ("managed_assemblies_sha256", None)):
            left, right = self.real_capture(), self.real_capture()
            right[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                cache.compare(left, right, strict=True)

    def test_real_prompt_or_generated_id_change_fails_even_with_identical_final_logits(self):
        for field in ("prompt_tokens", "generated_tokens"):
            left, right = self.real_capture(), self.real_capture()
            right[field][0] += 1
            if field == "generated_tokens":
                for run in right["runs"]:
                    run["generated_tokens"] = list(right[field])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "conditioning differs"):
                cache.compare(left, right, strict=True)
        with self.assertRaisesRegex(ValueError, "conditioning differs"):
            cache.compare(self.real_capture(), self.real_capture("teacher-forced"), strict=True)

    def test_real_every_greedy_repetition_must_finish_at_eos(self):
        for index in range(3):
            left = self.real_capture()
            left["runs"][index]["finish_reason"] = "length"
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "EOS"):
                cache.compare(left, self.real_capture(), strict=True)

    def test_real_repeated_ids_logits_and_conditioning_must_be_complete(self):
        for edit in (lambda data: data.pop("prompt_tokens"), lambda data: data.pop("generated_tokens"),
                     lambda data: data.update(runs=[]), lambda data: data["runs"][0]["generated_tokens"].append(10),
                     lambda data: data["runs"][0].update(final_logit_sha256="different"),
                     lambda data: data["runs"][0].update(prefill_tokens=3),
                     lambda data: data["runs"][0].update(decode_tokens=1)):
            left = copy.deepcopy(self.real_capture())
            edit(left)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                cache.compare(left, self.real_capture(), strict=True)

    def test_real_same_greedy_tokens_cannot_hide_numerical_regression(self):
        left, right = self.real_capture(), self.real_capture()
        right["final_logits"][0] = 1.1
        self.assertFalse(cache.compare(left, right, strict=True)["passed"])

    def test_real_command_forwards_exact_prompt_file_and_generation_mode(self):
        args = SimpleNamespace(prefill_tokens=40, decode_tokens=256, warmup=1, iterations=3,
                               generation="greedy", model=path.parent / "checkpoint.gguf", tokens_file=path.parent / "prompt.ids")
        command = cache.probe_command(Path("probe.dll"), Path("output"), args, "host", 8192, "mixed")
        self.assertEqual("greedy", command[command.index("--generation") + 1])
        self.assertEqual(str(args.tokens_file.resolve()), command[command.index("--tokens-file") + 1])
        self.assertEqual(str(args.model.resolve()), command[command.index("--model") + 1])
        self.assertEqual("1", command[command.index("--warmup") + 1])
        self.assertEqual("3", command[command.index("--iterations") + 1])


if __name__ == "__main__":
    unittest.main()
