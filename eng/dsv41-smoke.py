#!/usr/bin/env python3
"""Run a reproducible native GGUF smoke test from exact input token IDs.

This bypasses chat rendering to isolate inference. Use the HTTP validation
suite for tools, structured output, and concurrent request behavior.
Requires numpy; decoded-text validation additionally requires tokenizers.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def trace_fresh_slot(lib, handle, prompt, primary_logits, directory, output):
    """Replay only the prompt in a new slot; never overwrite primary results."""
    result = {"scope": "Traced fresh-slot diagnostic after the untraced primary generation; not a performance measurement.",
              "requested": True, "complete": False, "forward_complete": False,
              "trace_coverage_scope": "Nonempty position-zero files only; architecture-specific completeness is checked by the guarded runner.",
              "directory": str(directory.resolve()),
              "prompt_tokens": prompt, "primary_slot_id": 0}
    slot = -1
    cleanup_errors = []
    previous_trace = os.environ.get("TS_DSV41_TRACE_DIR")
    primary_n_past = None
    try:
        primary_n_past = lib.TSGgml_Dsv4NPast(handle)
        result["primary_n_past_before"] = primary_n_past
        if previous_trace is not None:
            raise RuntimeError("Fresh-slot tracing requires an untraced primary environment")
        if directory.exists():
            raise RuntimeError("Refusing to overwrite an existing diagnostic trace directory")
        allocated = lib.TSGgml_Dsv4SlotAlloc(handle)
        if allocated <= 0:
            raise RuntimeError(f"Fresh slot allocation failed or returned the primary slot ({allocated})")
        slot = allocated
        result["slot_id"] = slot
        if lib.TSGgml_Dsv4SetActiveSlot(handle, slot):
            raise RuntimeError("Cannot select the fresh diagnostic slot")
        if lib.TSGgml_Dsv4NPast(handle) != 0:
            raise RuntimeError("Diagnostic slot did not start at position zero")
        os.environ["TS_DSV41_TRACE_DIR"] = str(directory.resolve())
        tokens = np.asarray(prompt, dtype=np.int32)
        traced = np.empty_like(primary_logits)
        started = time.monotonic()
        status = lib.TSGgml_Dsv4Forward(handle, tokens.ctypes.data, len(tokens), traced.ctypes.data)
        result["diagnostic_seconds"] = time.monotonic() - started
        if status or not np.isfinite(traced).all():
            raise RuntimeError(f"Traced forward failed or produced nonfinite logits (status={status})")
        result["n_past_after"] = lib.TSGgml_Dsv4NPast(handle)
        if result["n_past_after"] != len(prompt):
            raise RuntimeError("Traced forward did not consume the exact prompt")
        result["forward_complete"] = True
        trace_files = sorted(directory.glob("p000000_v41.*"))
        if not trace_files or any(path.stat().st_size == 0 for path in trace_files):
            raise RuntimeError("Traced forward did not retain nonempty position-zero V4.1 tensors")
        logits_path = output.with_suffix(".traced-first-logits.npy")
        np.save(logits_path, traced)
        primary64, traced64 = primary_logits.astype(np.float64), traced.astype(np.float64)
        primary_norm, traced_norm = np.linalg.norm(primary64), np.linalg.norm(traced64)
        if primary_norm <= 0 or traced_norm <= 0:
            raise RuntimeError("Cannot compare zero-norm primary or traced logits")
        difference = traced64 - primary64
        result.update(complete=True, logits_path=str(logits_path.resolve()), trace_file_count=len(trace_files),
                      bitwise_equal=bool(np.array_equal(traced.view(np.uint32), primary_logits.view(np.uint32))),
                      max_absolute_error=float(np.max(np.abs(difference))),
                      relative_l2=float(np.linalg.norm(difference) / primary_norm),
                      cosine=float(traced64 @ primary64 / (traced_norm * primary_norm)),
                      strict_allclose_atol_rtol_2e_5=bool(np.allclose(traced64, primary64, atol=2e-5, rtol=2e-5)),
                      argmax=int(traced.argmax()), primary_argmax=int(primary_logits.argmax()))
    except Exception as error:
        result.update(complete=False, error=str(error))
    finally:
        if previous_trace is None:
            os.environ.pop("TS_DSV41_TRACE_DIR", None)
        else:
            os.environ["TS_DSV41_TRACE_DIR"] = previous_trace
        if slot > 0:
            # SlotFree rejects the active slot. Restore the known initial slot
            # first; if that fails, outer model destruction still owns cleanup.
            try:
                if lib.TSGgml_Dsv4SetActiveSlot(handle, 0):
                    raise RuntimeError("Cannot restore primary slot")
                if lib.TSGgml_Dsv4SlotFree(handle, slot):
                    raise RuntimeError("Cannot free diagnostic slot")
                if lib.TSGgml_Dsv4NPast(handle) != primary_n_past:
                    raise RuntimeError("Primary slot position changed during diagnostic replay")
                result["primary_slot_restored"] = True
            except Exception as error:
                cleanup_errors.append(str(error))
        if cleanup_errors:
            result.update(complete=False, cleanup_errors=cleanup_errors)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True, help="JSON array of exact prompt token IDs")
    parser.add_argument("--output", type=Path, required=True, help="Report JSON; first logits are saved beside it")
    parser.add_argument("--tokenizer", type=Path, help="Official tokenizer.json for decoded-text validation")
    parser.add_argument("--expected-text")
    parser.add_argument("--backend", default="CUDA", choices=("CUDA", "CPU"))
    parser.add_argument("--gpus", type=int, default=8, help="Number of selected GPUs")
    parser.add_argument("--tp", type=int, default=0, help="Routed-MoE tensor-parallel ranks: 0 for layer placement, otherwise equal to --gpus")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--cpu-moe", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--eos", type=int, default=1)
    tracing = parser.add_mutually_exclusive_group()
    tracing.add_argument("--trace-dir", type=Path, help="Save first-forward intermediate tensors; excludes it from timing comparisons")
    tracing.add_argument("--trace-after-dir", type=Path,
                         help="After untraced generation, replay the prompt in a fresh slot and save diagnostic tensors")
    args = parser.parse_args()
    if args.trace_after_dir and "TS_DSV41_TRACE_DIR" in os.environ:
        parser.error("--trace-after-dir requires TS_DSV41_TRACE_DIR to be unset for the primary pass")
    if args.trace_after_dir and args.trace_after_dir.exists():
        parser.error("Refusing to overwrite an existing diagnostic trace directory")
    if args.expected_text is not None and args.tokenizer is None:
        parser.error("--expected-text requires --tokenizer")
    if min(args.steps, args.context, args.ubatch, args.threads, args.gpus) < 1:
        parser.error("steps/context/ubatch/threads/gpus must be positive")
    prompt = json.loads(args.tokens.read_text())
    if not isinstance(prompt, list) or not prompt or any(type(t) is not int or t < 0 for t in prompt):
        parser.error("tokens must be a nonempty array of nonnegative integers")
    tokenizer = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    native_hash = hashlib.sha256()
    with args.library.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            native_hash.update(block)
    environment = {name: os.environ.get(name) for name in (
        "CUDA_VISIBLE_DEVICES", "TS_DSV4_FA", "TS_DSV4_GATHER",
        "TS_DSV4_FUSED", "TS_CPU_MOE_THREADS", "NVIDIA_TF32_OVERRIDE",
        "TS_DSV41_SPARSE_FA", "TS_DSV41_COMPACT_RAW_GATHER",
        "TS_DSV41_ENGRAM_THREADS", "TS_DSV41_ENGRAM_WARM")}
    tp_ranks = args.tp
    report = {"model": str(args.model.resolve()), "library": str(args.library.resolve()),
              "library_sha256": native_hash.hexdigest(),
              "backend": args.backend, "placement": "routed_moe_tensor_parallel" if tp_ranks else "layer",
              "gpus": args.gpus, "tensor_parallel_ranks": tp_ranks, "environment": environment,
              "context": args.context, "ubatch": args.ubatch, "threads": args.threads,
              "cpu_moe_layers": args.cpu_moe, "prompt_tokens": prompt, "steps": [],
              "generated_tokens": [], "first_forward_traced": args.trace_dir is not None}
    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lib = C.CDLL(str(args.library.resolve()))
    lib.TSGgml_Dsv4LoadModel.argtypes = [C.c_char_p] + [C.c_int] * 4 + [C.c_char_p, C.c_int, C.c_char_p, C.c_int]
    lib.TSGgml_Dsv4LoadModel.restype = C.c_void_p
    lib.TSGgml_Dsv4Forward.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]
    lib.TSGgml_Dsv4Forward.restype = C.c_int
    lib.TSGgml_Dsv4VocabSize.argtypes = [C.c_void_p]
    lib.TSGgml_Dsv4VocabSize.restype = C.c_int
    lib.TSGgml_Dsv4Free.argtypes = [C.c_void_p]
    lib.TSGgml_Dsv4Free.restype = None
    if args.trace_after_dir:
        for name in ("TSGgml_Dsv4SlotAlloc", "TSGgml_Dsv4NPast"):
            function = getattr(lib, name)
            function.argtypes, function.restype = [C.c_void_p], C.c_int
        for name in ("TSGgml_Dsv4SetActiveSlot", "TSGgml_Dsv4SlotFree"):
            function = getattr(lib, name)
            function.argtypes, function.restype = [C.c_void_p, C.c_int], C.c_int
    started = time.monotonic()
    handle = lib.TSGgml_Dsv4LoadModel(str(args.model.resolve()).encode(), args.gpus,
                                     args.context, args.ubatch, args.threads, None, args.cpu_moe,
                                     args.backend.encode(), tp_ranks)
    report["load_seconds"] = time.monotonic() - started
    if not handle:
        report["error"] = "Native load failed; see stderr"
        save()
        return 1
    print(f"Loaded in {report['load_seconds']:.3f}s", flush=True)
    trace_previous = os.environ.get("TS_DSV41_TRACE_DIR")
    if args.trace_dir:
        os.environ["TS_DSV41_TRACE_DIR"] = str(args.trace_dir.resolve())
    primary_logits = None
    try:
        logits = np.empty(lib.TSGgml_Dsv4VocabSize(handle), dtype=np.float32)
        if max(prompt) >= len(logits):
            raise ValueError("Prompt token exceeds model vocabulary")
        tokens = np.asarray(prompt, dtype=np.int32)
        for step in range(args.steps):
            started = time.monotonic()
            status = lib.TSGgml_Dsv4Forward(handle, tokens.ctypes.data, len(tokens), logits.ctypes.data)
            elapsed = time.monotonic() - started
            if status or not np.isfinite(logits).all():
                raise RuntimeError(f"Forward failed or produced nonfinite logits (status={status})")
            if step == 0:
                np.save(args.output.with_suffix(".first-logits.npy"), logits)
                primary_logits = logits.copy()
                if args.trace_dir:
                    os.environ.pop("TS_DSV41_TRACE_DIR", None)
            token = int(logits.argmax())
            row = {"step": step, "input_tokens": len(tokens), "token": token,
                   "seconds": elapsed, "max_logit": float(logits[token])}
            report["steps"].append(row)
            report["generated_tokens"].append(token)
            print(json.dumps(row), flush=True)
            save()
            if token == args.eos:
                break
            tokens = np.asarray([token], dtype=np.int32)
        report["ended_with_eos"] = report["generated_tokens"][-1] == args.eos
        if tokenizer:
            report["text"] = tokenizer.decode(report["generated_tokens"], skip_special_tokens=True)
        report["passed"] = args.expected_text is None or report["text"].strip() == args.expected_text
        save()
        if args.trace_after_dir:
            report["trace_diagnostic"] = trace_fresh_slot(lib, handle, prompt, primary_logits,
                                                         args.trace_after_dir, args.output)
    except Exception as error:
        report.update(passed=False, error=str(error))
    finally:
        try:
            lib.TSGgml_Dsv4Free(handle)
        finally:
            if trace_previous is None:
                os.environ.pop("TS_DSV41_TRACE_DIR", None)
            else:
                os.environ["TS_DSV41_TRACE_DIR"] = trace_previous
            save()
    return int(not report["passed"] or
               (args.trace_after_dir is not None and not report.get("trace_diagnostic", {}).get("complete", False)))


if __name__ == "__main__":
    raise SystemExit(main())
