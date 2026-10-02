#!/usr/bin/env python3
"""Summarize an explicit plan of isolated five-run CLI benchmarks, failing closed.

The plan has `expected_starts` (family -> profile -> count) and `runs` entries
with `family`, `profile`, and `path`. Paths are relative to --root. Missing
directories remain incomplete. Fixed-token timings and the separate untimed
greedy correctness chain are reported independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

METRICS = ("prefillMs", "prefillTps", "decodeMs", "decodeTps", "msPerTok",
           "decodeModelMs", "decodeModelTps", "greedySampleMs")
PROFILES = {"baseline-layer", "candidate-layer", "candidate-tp"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_log(log):
    rows, failures = [], []
    for match in re.finditer(r"benchmark run (\d+)/(\d+): ([^\r\n]+)", log):
        fields = dict(re.findall(r"(\w+)=([^\s]+)", match[3]))
        row = {"run": int(match[1]), "runs": int(match[2]), "decodeMode": fields.get("decodeMode")}
        for name in METRICS:
            if name in fields:
                value = float(fields[name])
                row[name] = value if math.isfinite(value) else None
        rows.append(row)
    if [row["run"] for row in rows] != [1, 2, 3, 4, 5] or any(row["runs"] != 5 for row in rows):
        failures.append("Expected exactly five ordered timing rows numbered 1..5")
    if any(row.get("decodeMode") != "fixed-inference" for row in rows):
        failures.append("Timed rows must use fixed-inference mode")
    for row in rows:
        for metric in ("prefillMs", "prefillTps", "decodeMs", "decodeTps", "msPerTok"):
            if row.get(metric) is None or row[metric] <= 0:
                failures.append(f"Run {row['run']} has no finite positive {metric}")
    starts = list(re.finditer(r"inference benchmark starting: prefillTokens=(\d+) decodeTokens=(\d+) runs=(\d+) chunked=(\w+) decodeMode=(\S+)", log))
    configuration = None
    if len(starts) != 1:
        failures.append("Expected one benchmark configuration record")
    else:
        match = starts[0]
        configuration = {"prefill_tokens": int(match[1]), "decode_tokens": int(match[2]),
                         "runs": int(match[3]), "chunked": match[4].lower() == "true", "decode_mode": match[5]}
        if configuration["runs"] != 5 or configuration["decode_mode"] != "fixed-inference":
            failures.append("Configuration must declare five fixed-inference runs")
    chains = list(re.finditer(r"benchmark sampled tokens \(untimed correctness\): prefillTopToken=(\d+) decode=([0-9,]+)", log))
    greedy = None
    if len(chains) != 1:
        failures.append("Expected exactly one separate untimed greedy correctness chain")
    else:
        greedy = {"prefill_token": int(chains[0][1]), "decode_tokens": [int(t) for t in chains[0][2].split(",")]}
        if configuration and len(greedy["decode_tokens"]) != configuration["decode_tokens"]:
            failures.append("Untimed greedy chain does not contain every requested decode token")
    warmups = [float(value) / 1000 for value in re.findall(r"Kernel warmup completed in ([0-9.]+) ms", log)]
    first = rows[0] if rows and rows[0]["run"] == 1 else None
    medians = {}
    if len(rows) == 5 and [r["run"] for r in rows] == list(range(1, 6)):
        for metric in METRICS:
            values = [r.get(metric) for r in rows[1:]]
            if all(value is not None for value in values):
                medians[metric] = statistics.median(values)
    return {"rows": rows, "first_run": first, "steady_medians_runs_2_to_5": medians,
            "configuration": configuration, "untimed_greedy": greedy,
            "kernel_warmup_seconds": warmups, "failures": failures}


def read_run(entry, root):
    directory = root / entry["path"]
    result = {**entry, "path": str(directory), "status": "incomplete", "failures": []}
    if not (directory / "run.json").is_file() or not (directory / "process.log").is_file():
        result["failures"].append("Planned run has no complete run.json/process.log evidence")
        return result
    run = json.loads((directory / "run.json").read_text())
    parsed = parse_log((directory / "process.log").read_text(errors="replace"))
    result.update(parsed)
    if run.get("status") != "completed" or run.get("exit_code") != 0:
        result["failures"].append("Runner did not report successful completion")
    if run.get("shutdown", {}).get("exit_code") != 0 or run.get("shutdown_failures"):
        result["failures"].append("Model did not shut down cleanly")
    if run.get("ggml_clean") is not True or not run.get("ggml_revision"):
        result["failures"].append("Clean upstream dependency identity is missing")
    for name in ("native_sha256", "managed_sha256", "source_identity", "model_shards"):
        if not run.get(name):
            result["failures"].append(f"Missing provenance: {name}")
    result["provenance"] = {name: run.get(name) for name in (
        "started_unix", "command", "environment", "native_sha256", "managed_sha256", "source_identity",
        "ggml_revision", "ggml_clean", "model_shards", "model_verification_report_sha256", "companions")}
    result["provenance"].update(run_json_sha256=digest(directory / "run.json"), process_log_sha256=digest(directory / "process.log"))
    result["model_load_seconds"] = [sample["seconds"] for sample in run.get("model_load_timings", [])
                                      if sample.get("source") == "cli_model_load"]
    result["process_to_liveness_seconds"] = run.get("process_to_liveness_seconds")
    if len(result["model_load_seconds"]) != 1 or len(result["kernel_warmup_seconds"]) != 1:
        result["failures"].append("Expected separate CLI model-load and kernel-warmup durations")
    elif any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
             for value in result["model_load_seconds"] + result["kernel_warmup_seconds"]):
        result["failures"].append("Load and kernel-warmup durations must be finite and positive")
    result["status"] = "passed" if not result["failures"] else "failed"
    return result


def compare_greedy(reference, candidate):
    left, right = reference.get("untimed_greedy"), candidate.get("untimed_greedy")
    failures = []
    for name in ("configuration",):
        if reference.get(name) != candidate.get(name):
            failures.append(f"Different {name}")
    for name in ("model_shards", "environment", "ggml_revision"):
        if reference.get("provenance", {}).get(name) != candidate.get("provenance", {}).get(name):
            failures.append(f"Different {name}")
    if not left or not right:
        failures.append("Missing untimed greedy chain")
    elif left != right:
        failures.append("Untimed greedy prefill token or full decode token chain differs")
    divergence = None
    if left and right and left["decode_tokens"] != right["decode_tokens"]:
        divergence = next((i for i, (a, b) in enumerate(zip(left["decode_tokens"], right["decode_tokens"])) if a != b),
                          min(len(left["decode_tokens"]), len(right["decode_tokens"])))
    return {"reference": reference["path"], "candidate": candidate["path"], "passed": not failures,
            "failures": failures, "prefill_token_equal": bool(left and right and left["prefill_token"] == right["prefill_token"]),
            "first_decode_divergence": divergence}


def placement_command(run):
    """Keep every CLI setting except the intended layer/TP placement switch."""
    command = run.get("provenance", {}).get("command")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command):
        raise ValueError("Missing complete command provenance")
    expected = "--layer-split" if run["profile"] == "candidate-layer" else "--tp"
    normalized, placements = [], []
    index = 0
    while index < len(command):
        value = command[index]
        flag, equals, degree = value.partition("=")
        if flag in ("--tp", "--layer-split"):
            if not equals:
                index += 1
                if index >= len(command):
                    raise ValueError("Missing placement degree")
                degree = command[index]
            if not degree.isdecimal() or int(degree) < 2:
                raise ValueError("Same-source comparison requires an explicit multi-GPU placement")
            placements.append((flag, int(degree)))
        else:
            normalized.append(value)
        index += 1
    if len(placements) != 1 or placements[0][0] != expected:
        raise ValueError("Command placement does not match its declared profile")
    return normalized, placements[0][1]


def compare_same_source(reference, candidate):
    result = compare_greedy(reference, candidate)
    failures = result["failures"]
    left, right = reference.get("provenance", {}), candidate.get("provenance", {})
    for name in ("native_sha256", "managed_sha256"):
        if not left.get(name) or not right.get(name) or left[name] != right[name]:
            failures.append(f"Different or missing same-source {name}")
    left_source = left.get("source_identity", {}).get("groups")
    right_source = right.get("source_identity", {}).get("groups")
    required_groups = ("native_sources", "managed_sources", "shared_build_inputs")
    if (not isinstance(left_source, dict) or not isinstance(right_source, dict)
            or any(not left_source.get(name) or not right_source.get(name) for name in required_groups)
            or left_source != right_source):
        failures.append("Different or missing same-source source groups")
    for name in ("companions", "model_verification_report_sha256"):
        if left.get(name) != right.get(name):
            failures.append(f"Different {name}")
    try:
        if placement_command(reference) != placement_command(candidate):
            failures.append("Commands differ beyond the layer/TP placement switch, or GPU degrees differ")
    except ValueError as error:
        failures.append(str(error))
    result.update(passed=not failures, same_source=not failures,
                  scope="Current candidate layer split versus TP; does not replace original-baseline comparisons.")
    return result


def summarize(plan, root):
    expected = {(family, profile): count for family, profiles in plan["expected_starts"].items() for profile, count in profiles.items()}
    if any(profile not in PROFILES or type(count) is not int or count < 1 for (_, profile), count in expected.items()):
        raise ValueError("Expected starts require known profiles and positive integer counts")
    if any("baseline-layer" not in profiles for profiles in plan["expected_starts"].values()):
        raise ValueError("Every model family requires a baseline-layer control")
    if len({entry["path"] for entry in plan["runs"]}) != len(plan["runs"]):
        raise ValueError("One directory cannot establish multiple independent starts")
    if any((entry["family"], entry["profile"]) not in expected for entry in plan["runs"]):
        raise ValueError("Every run must belong to a declared expected profile")
    rows = [read_run(entry, root) for entry in plan["runs"]]
    comparisons, groups = [], []
    for (family, profile), count in expected.items():
        selected = [row for row in rows if (row["family"], row["profile"]) == (family, profile)]
        passed = [row for row in selected if row["status"] == "passed"]
        issues = []
        if len(passed) != count or len(selected) != count:
            issues.append(f"Expected {count} complete independent starts, received {len(passed)} passing out of {len(selected)} planned directories")
        starts = [row.get("provenance", {}).get("started_unix") for row in passed]
        if any(start is None for start in starts) or len(set(starts)) != len(starts):
            issues.append("Repeated startup identity cannot establish independent starts")
        # The source sidecar has a different artifact path for each start.
        source_groups = [json.dumps({"native": row["provenance"]["native_sha256"], "managed": row["provenance"]["managed_sha256"],
                                     "source": row["provenance"]["source_identity"].get("groups")}, sort_keys=True) for row in passed]
        if len(set(source_groups)) > 1:
            issues.append("Source or binary identity changed between starts of one profile")
        group = {"family": family, "profile": profile, "expected_starts": count, "completed_starts": len(passed),
                 "status": "passed" if not issues else "incomplete", "failures": issues}
        if passed:
            group["median_model_load_seconds"] = statistics.median(row["model_load_seconds"][0] for row in passed)
            group["median_kernel_warmup_seconds"] = statistics.median(row["kernel_warmup_seconds"][0] for row in passed)
            group["median_of_start_steady_medians"] = {metric: statistics.median(row["steady_medians_runs_2_to_5"][metric] for row in passed)
                                                        for metric in METRICS if all(metric in row["steady_medians_runs_2_to_5"] for row in passed)}
        groups.append(group)
    for family in {row["family"] for row in rows}:
        controls = [row for row in rows if row["family"] == family and row["profile"] == "baseline-layer" and row["status"] == "passed"]
        if controls:
            reference = controls[0]
            for candidate in (row for row in rows if row["family"] == family and row["status"] == "passed"):
                comparisons.append(compare_greedy(reference, candidate))
    throughput, loading = [], []
    for family in {g["family"] for g in groups}:
        indexed = {g["profile"]: g for g in groups if g["family"] == family}
        baseline = indexed.get("baseline-layer")
        if not baseline or baseline["status"] != "passed":
            continue
        for profile in ("candidate-layer", "candidate-tp"):
            candidate = indexed.get(profile)
            if not candidate or candidate["status"] != "passed":
                continue
            paths = {row["path"] for row in rows if row["family"] == family and row["profile"] in ("baseline-layer", profile)}
            pairs = [pair for pair in comparisons if pair["candidate"] in paths]
            qualified = len(pairs) == len(paths) and all(pair["passed"] for pair in pairs)
            a, b = baseline["median_of_start_steady_medians"], candidate["median_of_start_steady_medians"]
            throughput.append({"family": family, "profile": profile, "qualified_by_greedy_parity": qualified,
                               "prefill_ratio": b["prefillTps"] / a["prefillTps"], "decode_ratio": b["decodeTps"] / a["decodeTps"]})
            if profile == "candidate-layer":
                loading.append({"family": family, "placement": "layer", "qualified_by_greedy_parity": qualified,
                                "baseline_over_candidate_load_ratio": baseline["median_model_load_seconds"] / candidate["median_model_load_seconds"],
                                "scope": "Repeated mixed/warm-cache process loads at the same placement; no cold-storage speedup claim."})
    same_source, same_source_throughput = [], []
    for family, profiles in plan["expected_starts"].items():
        if not {"candidate-layer", "candidate-tp"}.issubset(profiles):
            continue
        indexed = {g["profile"]: g for g in groups if g["family"] == family}
        layer_group, tp_group = indexed["candidate-layer"], indexed["candidate-tp"]
        controls = [row for row in rows if row["family"] == family and row["profile"] == "candidate-layer" and row["status"] == "passed"]
        candidates = [row for row in rows if row["family"] == family and row["profile"] == "candidate-tp" and row["status"] == "passed"]
        pairs = [compare_same_source(control, candidate) for control in controls for candidate in candidates]
        same_source.extend(pairs)
        complete_pair = layer_group["status"] == "passed" and tp_group["status"] == "passed"
        qualified = complete_pair and bool(pairs) and all(pair["passed"] for pair in pairs)
        record = {"family": family, "reference_profile": "candidate-layer", "profile": "candidate-tp",
                  "qualified_by_same_source_greedy_parity": qualified,
                  "status": "passed" if qualified else "failed" if any(not pair["passed"] for pair in pairs) else "incomplete",
                  "scope": "Same candidate binaries and source groups, identical configuration/environment/model and untimed tokens; original-baseline drift remains independently failed."}
        if complete_pair:
            a, b = layer_group["median_of_start_steady_medians"], tp_group["median_of_start_steady_medians"]
            record.update(prefill_ratio=b["prefillTps"] / a["prefillTps"], decode_ratio=b["decodeTps"] / a["decodeTps"])
        same_source_throughput.append(record)
    failed = (any(row["status"] == "failed" for row in rows)
              or any(not pair["passed"] for pair in comparisons + same_source))
    complete = bool(groups) and all(group["status"] == "passed" for group in groups)
    return {"status": "failed" if failed else "passed" if complete else "incomplete", "runs": rows, "groups": groups,
            "greedy_comparisons": comparisons, "throughput_comparisons": throughput, "loading_comparisons_same_placement": loading,
            "same_source_greedy_comparisons": same_source,
            "same_source_throughput_comparisons": same_source_throughput,
            "limitations": ["Fixed-token timing measures synthetic inference throughput; greedy correctness is a separate untimed chain.",
                            "Run 1 is shown separately; steady medians include only timing runs 2 through 5 in each process.",
                            "Model load excludes kernel warmup; CLI runs have no HTTP readiness measurement.",
                            "No page-cache eviction or cold-storage qualification is implied; cache pressure and concurrent I/O affect load times.",
                            "GPU clocks are observed, not locked; gpu.csv records SM/memory clocks and P-state for interpreting clock drift.",
                            "Same-source candidate layer/TP agreement cannot qualify a failed comparison with the original baseline.",
                            "Synthetic token parity and small HTTP probes do not establish broad model quality."]}


def markdown(report):
    lines = [f"CLI benchmark evidence: **{report['status']}**", "",
             "| Model | Profile | Completed starts | Load s | Warmup s | Steady prefill tok/s | Steady decode tok/s | Status |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"
    for group in report["groups"]:
        metrics = group.get("median_of_start_steady_medians", {})
        lines.append(f"| {group['family']} | {group['profile']} | {group['completed_starts']}/{group['expected_starts']} | "
                     f"{fmt(group.get('median_model_load_seconds'))} | {fmt(group.get('median_kernel_warmup_seconds'))} | "
                     f"{fmt(metrics.get('prefillTps'))} | {fmt(metrics.get('decodeTps'))} | {group['status']} |")
    lines += ["", "Individual timing rows (run 1 is excluded from the steady median):", "",
              "| Model/profile/start | Timing run | Prefill tok/s | Decode tok/s |",
              "|---|---:|---:|---:|"]
    for run in report["runs"]:
        label = f"{run['family']}/{run['profile']}/{Path(run['path']).name}"
        for row in run.get("rows", []):
            lines.append(f"| {label} | {row['run']} | {fmt(row.get('prefillTps'))} | {fmt(row.get('decodeTps'))} |")
    lines += ["", "Untimed greedy comparisons:", ""]
    for pair in report["greedy_comparisons"]:
        lines.append(f"- {Path(pair['candidate']).name}: {'pass' if pair['passed'] else 'FAIL'}; " + ("; ".join(pair["failures"]) or "all prefill/decode token IDs identical"))
    if report["same_source_greedy_comparisons"]:
        lines += ["", "Same-source candidate layer→TP comparisons (original-baseline results above remain binding):", ""]
        for pair in report["same_source_greedy_comparisons"]:
            lines.append(f"- {Path(pair['reference']).name} → {Path(pair['candidate']).name}: {'pass' if pair['passed'] else 'FAIL'}; "
                         + ("; ".join(pair["failures"]) or "identical native/managed/source, model/configuration/environment, and all untimed token IDs"))
        for pair in report["same_source_throughput_comparisons"]:
            lines.append(f"- {pair['family']} same-source TP/layer throughput ratios: prefill {fmt(pair.get('prefill_ratio'))}, "
                         f"decode {fmt(pair.get('decode_ratio'))}; {pair['status']}.")
    issues = [f"{Path(run['path']).name}: {failure}" for run in report["runs"] for failure in run["failures"]]
    issues += [f"{group['family']}/{group['profile']}: {failure}" for group in report["groups"] for failure in group["failures"]]
    if issues:
        lines += ["", "Incomplete or failed evidence:", "", *("- " + issue for issue in issues)]
    lines += ["", "Measurement limits:", "", *("- " + item for item in report["limitations"])]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    report = summarize(plan, args.root.resolve())
    report["plan_sha256"] = digest(args.plan)
    report["harness_sha256"] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report))
    print(report["status"])
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
