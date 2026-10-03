#!/usr/bin/env python3
"""Compare isolated Qwen4Exp device/host/cache processes and retain complete evidence.

Build the probe first, then run this tool. Each arm owns a fresh process because
the native cache configuration is read once. Small/large cache arms must agree
at relative L2 <= 1e-6 with equal argmaxes. Cache-versus-device arithmetic is
recorded independently and checked at that same strict bound. Host CPU arithmetic
is reported separately. Synthetic success does not qualify trained-model quality
or Strata performance parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else digest_stream(source)


def digest_stream(source):
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8").strip()


def gpu_sample():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return {"unix": time.time(), "exit_code": result.returncode, "csv": result.stdout.strip(),
                "error": result.stderr.strip() or None}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unix": time.time(), "error": str(error)}


def difference(actual, expected):
    if not actual or len(actual) != len(expected):
        raise ValueError("Missing or different row counts")
    metrics = []
    for row, (a, b) in enumerate(zip(actual, expected)):
        if not a or len(a) != len(b) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in a + b):
            raise ValueError("Incomplete/nonfinite vocabulary logits")
        relative = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(sum(y * y for y in b), 1e-30))
        metrics.append({"row": row, "relative_l2": relative,
                        "max_absolute": max(abs(x - y) for x, y in zip(a, b)),
                        "argmax_equal": max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)})
    return {"rows": len(metrics), "worst_relative_l2": max(row["relative_l2"] for row in metrics),
            "max_absolute": max(row["max_absolute"] for row in metrics),
            "argmax_mismatches": sum(not row["argmax_equal"] for row in metrics),
            "first_argmax_divergence": next((row["row"] for row in metrics if not row["argmax_equal"]), None)}


def real_final_capture(data):
    """Validate final-row evidence and the conditioning of every repetition.

    The probe retains one complete final vocabulary row, not every decode row.
    Greedy acceptance also requires EOS; answer semantics remain an external gate.
    """
    logits = data.get("final_logits")
    if not isinstance(logits, list) or not logits:
        raise ValueError("Complete real final vocabulary logits are missing")
    difference([logits], [logits])
    if data.get("vocabulary_size", len(logits)) != len(logits):
        raise ValueError("Real final vocabulary logits are incomplete")
    mode = data.get("decode_mode")
    if mode not in ("teacher-forced", "greedy"):
        raise ValueError("Real decode conditioning is missing")

    def tokens(name, nonempty):
        values = data.get(name)
        if not isinstance(values, list) or (nonempty and not values) or not all(
                isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in values):
            raise ValueError("Real " + name + " are missing or invalid")
        return values

    prompt = tokens("prompt_tokens", True)
    generated = tokens("generated_tokens", mode == "greedy")
    if mode == "teacher-forced":
        forced = tokens("forced_tokens", True)
        if generated:
            raise ValueError("Teacher-forced evidence unexpectedly contains generated IDs")
    else:
        forced = None
        if "forced_tokens" not in data or data["forced_tokens"] is not None:
            raise ValueError("Greedy evidence unexpectedly contains forced IDs")
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Real repeated-request captures are missing")
    hashes = set()
    for run in runs:
        if not isinstance(run, dict) or run.get("generated_tokens") != generated:
            raise ValueError("Real repetitions changed generated token IDs")
        if run.get("prefill_tokens") != len(prompt):
            raise ValueError("Real repetition prompt conditioning differs")
        if mode == "greedy" and run.get("finish_reason") != "eos":
            raise ValueError("Every real greedy repetition must complete with EOS")
        if mode == "greedy" and run.get("decode_tokens") != len(generated) - 1:
            raise ValueError("Real repetition generated conditioning differs")
        if mode == "teacher-forced" and (run.get("finish_reason") != "teacher-forced"
                or run.get("decode_tokens") != len(forced)):
            raise ValueError("Real repetition forced conditioning differs")
        digest = run.get("final_logit_sha256")
        if not isinstance(digest, str) or not digest:
            raise ValueError("Real repetition final logit hash is missing")
        hashes.add(digest)
    if len(hashes) != 1:
        raise ValueError("Real repetitions changed final vocabulary logits")
    return {"final": [logits]}, {"decode_mode": mode, "prompt_tokens": prompt,
                                 "forced_tokens": forced, "generated_tokens": generated}


def compare(left, right, strict):
    if left.get("model_sha256") != right.get("model_sha256") or not left.get("native_sha256") or left.get("native_sha256") != right.get("native_sha256"):
        raise ValueError("Checkpoint or mapped native binary differs between arms")
    if left.get("managed_assemblies_sha256") != right.get("managed_assemblies_sha256"):
        raise ValueError("Loaded managed assemblies differ between arms")
    conditioning = None
    if left.get("synthetic") is False and right.get("synthetic") is False:
        paths = [data.get("model_path") for data in (left, right)]
        if not all(isinstance(path, str) and path for path in paths) or os.path.normcase(os.path.abspath(paths[0])) != os.path.normcase(os.path.abspath(paths[1])):
            raise ValueError("Real checkpoint path differs or is missing between arms")
        if left.get("model_bytes") != right.get("model_bytes") or not isinstance(left.get("model_bytes"), int) or left["model_bytes"] <= 0:
            raise ValueError("Real checkpoint file size differs or is missing between arms")
        assemblies = left.get("managed_assemblies_sha256")
        if not isinstance(assemblies, dict) or not assemblies or not all(
                isinstance(path, str) and path and isinstance(digest, str) and digest for path, digest in assemblies.items()):
            raise ValueError("Real mapped managed binary identities are missing")
        left_captures, left_conditioning = real_final_capture(left)
        right_captures, right_conditioning = real_final_capture(right)
        if left_conditioning != right_conditioning:
            raise ValueError("Real prompt, forced or generated token conditioning differs between arms")
        conditioning = {"decode_mode": left_conditioning["decode_mode"],
                        "prompt_tokens": len(left_conditioning["prompt_tokens"]),
                        "forced_tokens": len(left_conditioning["forced_tokens"] or []),
                        "generated_tokens": len(left_conditioning["generated_tokens"]),
                        "token_ids_equal": True, "all_greedy_repetitions_eos": left_conditioning["decode_mode"] == "greedy"}
        scope = "final-vocabulary-only"
    else:
        left_captures, right_captures = left.get("captures"), right.get("captures")
        if not left_captures or not right_captures or left_captures.keys() != right_captures.keys():
            raise ValueError("Complete synthetic diagnostic captures are missing")
        scope = "synthetic-per-step-vocabulary"
    scenarios = {name: difference(left_captures[name], right_captures[name]) for name in left_captures}
    passed = all(row["worst_relative_l2"] <= 1e-6 and row["argmax_mismatches"] == 0 for row in scenarios.values())
    return {"strict": strict, "bound_relative_l2": 1e-6 if strict else None,
            "passed": passed if strict else None, "capture_scope": scope,
            "conditioning": conditioning, "language_quality_validated": False, "scenarios": scenarios}


def probe_command(probe, directory, args, placement, budget, quantization):
    command = ["dotnet", str(probe), "--output", str(directory / "probe.json"), "--placement", placement,
               "--backend", "ggml_cuda", "--quantization", quantization, "--prefill-tokens", str(args.prefill_tokens),
               "--decode-tokens", str(args.decode_tokens), "--warmup", str(args.warmup), "--iterations", str(args.iterations),
               "--generation", args.generation, "--require-cache", "1" if budget > 1 and placement == "host" else "0"]
    if args.model:
        command += ["--model", str(args.model.resolve())]
    if args.tokens_file:
        command += ["--tokens-file", str(args.tokens_file.resolve())]
    return command


def run_arm(root, output, probe, args, placement, budget, quantization, bridge=1, output_bridge=1, prefetch=0):
    name = f"{quantization}-{placement}-cache{budget}"
    if bridge == 0:
        name += "-bridge0"
    elif output_bridge == 0:
        name += "-output0"
    if prefetch:
        name += "-prefetch1"
    directory = output / name
    directory.mkdir()
    env = os.environ.copy()
    env.update(TS_HOST_MOE_EXPERT_CACHE_MB=str(budget), TS_HOST_MOE_EXPERT_CACHE_LAYERS="8" if not args.model else "48",
               TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS="1", TS_HOST_MOE_DECODE="1", TS_HOST_MOE_PIN_MAX_MB="0",
               TS_HOST_MOE_PIN="0")
    env["TS_HOST_MOE_EXPERT_CACHE_BRIDGE"] = str(bridge)
    env["TS_HOST_MOE_EXPERT_CACHE_OUTPUT_BRIDGE"] = str(output_bridge)
    env["TS_HOST_MOE_EXPERT_CACHE_PREFETCH"] = str(prefetch)
    for key in ("TS_CPU_MOE", "TS_N_CPU_MOE", "TS_Q4E_EXPERT_CACHE_CAPTURE_DIR", "TS_Q4E_EXPERT_CACHE_REFERENCE_DIR",
                "TS_SPEC", "TS_SPEC_TYPE", "TS_SPEC_DRAFT", "TS_SPEC_PMIN", "TS_SPEC_DRAFT_MODEL",
                "TS_MTP_SPEC", "TS_MTP_DRAFT", "TS_MTP_PMIN", "TS_MTP_DRAFT_MODEL", "TS_DSV4_DSPARK",
                "TS_QWEN35_DFLASH", "TS_MUSE_GLIMMER_DFLASH", "TS_NEMOTRON_DFLASH"):
        env.pop(key, None)
    command = probe_command(probe, directory, args, placement, budget, quantization)
    samples, stopped = [], threading.Event()

    def sample():
        while not stopped.is_set():
            samples.append(gpu_sample())
            stopped.wait(1)

    started = time.monotonic()
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        with (directory / "process.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        evidence = {"exit_code": process.returncode, "seconds": time.monotonic() - started, "command": command,
                    "cache_budget_mb": budget, "input_bridge": bridge, "output_bridge": output_bridge,
                    "raw_miss_prefetch": prefetch, "placement": placement, "environment": {key: value for key, value in env.items()
                        if key.startswith(("TS_HOST_MOE_", "MAX_CONTEXT", "KV_CACHE_DTYPE"))}}
    finally:
        stopped.set()
        thread.join(timeout=15)
        (directory / "gpu.json").write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    (directory / "run.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    if process.returncode:
        raise ValueError(f"{name} exited {process.returncode}; see {directory / 'process.log'}")
    data = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
    if data.get("passed") is not True or len([row for row in data["runs"] if not row["warmup"]]) != args.iterations:
        raise ValueError(name + ": missing completed timed repetitions")
    if data["native_sha256"] != sha(data["native_path"]):
        raise ValueError(name + ": mapped native binary changed during validation")
    for path, expected in data.get("managed_assemblies_sha256", {}).items():
        if sha(path) != expected:
            raise ValueError(name + ": loaded managed assembly changed during validation")
    if args.model:
        real_final_capture(data)
    stats = data.get("cache_stats") or {}
    if budget == 1 and placement == "host" and (stats.get("Calls", 0) != 0 or stats.get("ReservedBytes", 0) != 0):
        raise ValueError(name + ": an insufficient budget unexpectedly engaged the cache")
    if budget > 1 and placement == "host":
        if not all(stats.get(key, 0) > 0 for key in ("Hits", "Misses", "Calls")):
            raise ValueError(name + ": cache never engaged and reused weights")
        if stats["BudgetBytes"] != budget * 1024 * 1024 or not 0 <= stats["ReservedBytes"] <= stats["BudgetBytes"]:
            raise ValueError(name + ": cache device reservation does not respect requested budget")
    rows = [row for row in data["runs"] if not row["warmup"]]
    evidence.update(probe_sha256=sha(directory / "probe.json"), native_path=data["native_path"],
                    native_sha256=data["native_sha256"], model_sha256=data["model_sha256"], cache_stats=stats,
                    model_path=data.get("model_path"), model_bytes=data.get("model_bytes"), decode_mode=data.get("decode_mode"),
                    managed_assemblies_sha256=data.get("managed_assemblies_sha256"),
                    model_load_ms=data["model_load_ms"], process_peak_working_set_bytes=data["process_peak_working_set_bytes"],
                    median_prefill_tps=statistics.median(row["prefill_tps"] for row in rows),
                    median_decode_tps=statistics.median(row["decode_tps"] for row in rows))
    return name, data, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configuration", default="Release", choices=("Release", "Debug"))
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--generation", choices=("teacher-forced", "greedy"), default="teacher-forced")
    parser.add_argument("--tokens-file", type=Path,
                        help="real-model prompt IDs used verbatim for every fresh process and repeated request")
    parser.add_argument("--cache-mb", type=int, nargs="+", default=[12, 128])
    parser.add_argument("--prefill-tokens", type=int, default=40)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--skip-bridge-comparison", action="store_true",
                        help="omit staged input/output and prefetch arms; the default checks complete-logit equality for every transfer variant")
    args = parser.parse_args()
    if min(args.cache_mb) <= 1 or min(args.prefill_tokens, args.decode_tokens, args.iterations) < 1 or args.warmup < 0:
        parser.error("positive workloads, budgets > 1 MiB and nonnegative warmup are required")
    if args.tokens_file and (not args.model or not args.tokens_file.is_file()):
        parser.error("--tokens-file requires --model and an existing token file")
    if args.model and args.skip_bridge_comparison and len(set(args.cache_mb)) < 2:
        parser.error("real validation requires transfer variants or at least two cache budgets")
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    probe = (args.probe or root / "eng/validation/Qwen4ExpExpertCacheProbe/bin" / args.configuration / "net10.0/Qwen4ExpExpertCacheProbe.dll").resolve()
    report = {"schema_version": 1, "passed": False, "language_quality_validated": False, "strata_parity_validated": False,
              "platform": platform.platform(), "processor": platform.processor(), "logical_processors": os.cpu_count(),
              "probe_assembly_sha256": sha(probe), "runner_sha256": sha(__file__),
              "generation": args.generation, "tokens_file": str(args.tokens_file.resolve()) if args.tokens_file else None,
              "tokens_file_sha256": sha(args.tokens_file) if args.tokens_file else None,
              "real_comparison_scope": "final-vocabulary-only" if args.model else None,
              "ggml_revision": git(root / "ExternalProjects/ggml", "rev-parse", "HEAD"),
              "ggml_status": git(root / "ExternalProjects/ggml", "status", "--porcelain"),
              "source_head": git(root, "rev-parse", "HEAD"), "source_status": git(root, "status", "--short"),
              "source_diff_sha256": hashlib.sha256(git(root, "diff", "HEAD", "--binary").encode()).hexdigest(),
              "source_files_sha256": {str(path.relative_to(root)): sha(path) for path in [
                  root / "TensorSharp.GGML.Native/ggml_ops_host_moe_cache.cpp",
                  root / "TensorSharp.GGML.Native/ggml_ops_moe.cpp",
                  root / "TensorSharp.GGML.Native/ggml_ops_core.cpp",
                  root / "InferenceWeb.Tests/Qwen4ExpExpertCacheScenario.cs",
                  root / "eng/validation/Qwen4ExpExpertCacheProbe/Program.cs"] if path.is_file()},
              "arms": {}, "comparisons": {}, "limitations": [
                  "Synthetic teacher-forced measurements cannot establish trained model quality or parity with Strata.",
                  "Real cached variants compare exact conditioning/output IDs and the final vocabulary row only; intermediate decode logits and answer semantics are not validated.",
                  "Real CPU-offloaded baseline arithmetic is recorded separately and is not a strict cached-variant comparator.",
                  "GPU samples include desktop and other processes; WDDM does not provide reliable per-process nvidia-smi memory.",
                  "Each arm is isolated but run once in fixed order; repeat rotated arm order before claiming performance improvement.",
                  "Peak working set is a process OS measure, not a complete peak allocation accounting."]}
    try:
        if report["ggml_status"]:
            raise ValueError("Validation requires an unchanged upstream ggml checkout")
        for quantization in (["checkpoint"] if args.model else ["mixed", "q2kxl"]):
            # A real checkpoint can be far larger than VRAM; do not silently try
            # to load it all on the device. Real-model mode measures host/cache only.
            configurations = ([] if args.model else [("device", 0)]) + [("host", 0)] + ([] if args.model else [("host", 1)]) + [("host", budget) for budget in args.cache_mb]
            data = {}
            for placement, budget in configurations:
                print(f"Running {quantization}, {placement}, cache={budget} MiB", flush=True)
                name, captures, evidence = run_arm(root, output, probe, args, placement, budget, "mixed" if args.model else quantization)
                report["arms"][name], data[(placement, budget)] = evidence, captures
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            if not args.skip_bridge_comparison:
                budget = args.cache_mb[0]
                print(f"Running {quantization}, host, cache={budget} MiB, staged inputs", flush=True)
                name, staged, evidence = run_arm(root, output, probe, args, "host", budget,
                                                 "mixed" if args.model else quantization, bridge=0, output_bridge=0)
                report["arms"][name] = evidence
                report["comparisons"][f"{quantization}-bridge0-vs-bridge1"] = compare(
                    staged, data[("host", budget)], strict=True)
                for output_bridge, prefetch, label in ((0, 0, "staged output"), (1, 1, "raw miss prefetch")):
                    print(f"Running {quantization}, host, cache={budget} MiB, {label}", flush=True)
                    name, variant, evidence = run_arm(root, output, probe, args, "host", budget,
                        "mixed" if args.model else quantization, output_bridge=output_bridge, prefetch=prefetch)
                    report["arms"][name] = evidence
                    report["comparisons"][name + "-vs-full-d2d"] = compare(variant, data[("host", budget)], strict=True)
            if not args.model:
                device = data[("device", 0)]
                report["comparisons"][quantization + "-host-vs-device"] = compare(data[("host", 0)], device, strict=False)
                report["comparisons"][quantization + "-insufficient-cache-vs-disabled"] = compare(data[("host", 1)], data[("host", 0)], strict=True)
                for budget in args.cache_mb:
                    comparison = compare(data[("host", budget)], device, strict=True)
                    report["comparisons"][f"{quantization}-cache{budget}-vs-device"] = comparison
            for budget in args.cache_mb[1:]:
                report["comparisons"][f"{quantization}-cache{budget}-vs-cache{args.cache_mb[0]}"] = compare(
                    data[("host", budget)], data[("host", args.cache_mb[0])], strict=True)
        report["passed"] = all(row["passed"] is not False for row in report["comparisons"].values())
        if git(root / "ExternalProjects/ggml", "status", "--porcelain"):
            raise ValueError("Upstream ggml was modified during validation")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report["passed"] = False
        report["error"] = str(error)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(output / "report.json"), "error": report.get("error")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
