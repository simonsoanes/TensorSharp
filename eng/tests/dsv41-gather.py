#!/usr/bin/env python3
"""Compare V4.1 sparse decode gather with masked-dense attention on a fixture.

Requires only NumPy and a built GgmlOps library. This exercises actual native
forward passes, compression boundaries, candidate pruning, and graph-cache
reuse. It never loads a downloaded full-size model or runs a throughput test.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", default="CPU")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--prefixes", nargs="+", type=int, help="Explicit prefill lengths, e.g.1020 1023 1024 1025 for topk512")
    parser.add_argument("--decode-tokens", type=int, help="Limit per-prefix decode length when testing a large boundary fixture")
    parser.add_argument("--ubatch", type=int, default=32, help="Native prefill microbatch; also determines physical raw-ring size")
    parser.add_argument("--compare-compact-raw", action="store_true", help="Also compare opt-in raw-window compaction with the unchanged gathered and masked-dense paths")
    args = parser.parse_args()
    meta = json.loads((args.fixture_dir / "deepseek41.config.json").read_text())
    if not meta.get("fixture"):
        parser.error("only the small generated numerical fixture is accepted")
    config = meta["config"]["text_config"]
    tokens = np.asarray(json.loads((args.fixture_dir / "tokens.json").read_text()), dtype=np.int32)
    threshold = 2 * config["index_topk"] + 1
    if len(tokens) <= threshold + 2:
        parser.error("fixture needs enough tokens to cross the gather visibility threshold")
    prefixes = sorted(set(args.prefixes or [1, max(1, threshold-1), threshold, threshold+1, min(8, len(tokens)-1)]))
    if any(prefix < 1 or prefix >= len(tokens) for prefix in prefixes) or (args.decode_tokens is not None and args.decode_tokens < 1):
        parser.error("prefixes must fit the fixture and decode-tokens must be positive")
    if args.ubatch < 1:
        parser.error("ubatch must be positive")
    lib = ctypes.CDLL(str(args.library.resolve()))
    load = lib.TSGgml_Dsv4LoadModel
    load.argtypes = [ctypes.c_char_p] + [ctypes.c_int]*4 + [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    load.restype = ctypes.c_void_p
    forward = lib.TSGgml_Dsv4Forward
    forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    forward.restype = ctypes.c_int
    reset_checked = lib.TSGgml_Dsv4ResetChecked
    reset_checked.argtypes, reset_checked.restype = [ctypes.c_void_p], ctypes.c_int
    def reset(handle):
        if reset_checked(handle) != 1:
            raise RuntimeError("native reset was refused")
    free = lib.TSGgml_Dsv4Free
    free.argtypes, free.restype = [ctypes.c_void_p], None
    previous = {name: os.environ.get(name) for name in ("TS_DSV4_GATHER", "TS_DSV41_COMPACT_RAW_GATHER")}
    outputs = {}
    try:
        modes = [("dense", False, False), ("gather", True, False)]
        if args.compare_compact_raw:
            modes.append(("compact", True, True))
        for name, enabled, compact in modes:
            # The native model reads this at load time. Reusing a single model
            # would accidentally compare the same mode with itself.
            os.environ["TS_DSV4_GATHER"] = "1" if enabled else "0"
            os.environ["TS_DSV41_COMPACT_RAW_GATHER"] = "1" if compact else "0"
            handle = load(str(args.fixture_dir / "deepseek41-fixture.gguf").encode(),
                          args.gpus, max(256, len(tokens)+1), args.ubatch, 2, None, 0, args.backend.encode(), 0)
            if not handle:
                raise RuntimeError("native fixture load failed")
            results = {}
            try:
                for prefix in prefixes:
                    reset(handle)
                    stop = min(len(tokens), prefix + args.decode_tokens) if args.decode_tokens else len(tokens)
                    for start in [0] + list(range(prefix, stop)):
                        ids = np.ascontiguousarray(tokens[:prefix] if start == 0 else tokens[start:start+1])
                        output = np.empty(config["vocab_size"], dtype=np.float32)
                        code = forward(handle, ids.ctypes.data, len(ids), output.ctypes.data)
                        if code:
                            raise RuntimeError(f"native forward failed: {code}")
                        results[(prefix, start+len(ids))] = output
            finally:
                free(handle)
            outputs[name] = results
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    checks = []
    for (prefix, position), dense in outputs["dense"].items():
        gathered = outputs["gather"][(prefix, position)]
        checks.append(dict(prefill_tokens=prefix, position=position,
                           max_absolute_error=float(np.max(np.abs(dense-gathered))),
                           relative_l2=float(np.linalg.norm(dense-gathered) / np.linalg.norm(dense)),
                           dense_argmax=int(dense.argmax()), gather_argmax=int(gathered.argmax()),
                           passed=bool(np.allclose(dense, gathered, atol=args.atol, rtol=args.rtol))))
    compact_checks = []
    if args.compare_compact_raw:
        for reference in ("dense", "gather"):
            for (prefix, position), expected in outputs[reference].items():
                actual = outputs["compact"][(prefix, position)]
                compact_checks.append(dict(reference=reference, prefill_tokens=prefix, position=position,
                    max_absolute_error=float(np.max(np.abs(expected-actual))),
                    relative_l2=float(np.linalg.norm(expected-actual) / np.linalg.norm(expected)),
                    reference_argmax=int(expected.argmax()), compact_argmax=int(actual.argmax()),
                    passed=bool(np.allclose(expected, actual, atol=args.atol, rtol=args.rtol))))
    result = dict(backend=args.backend, gpus=args.gpus, atol=args.atol, rtol=args.rtol,
                  flash_attention=os.environ.get("TS_DSV4_FA"), gather_threshold=threshold,
                  prefixes=prefixes, decode_tokens=args.decode_tokens, ubatch=args.ubatch,
                  raw_window=config["sliding_window"],
                  ring_rows=((config["sliding_window"]+args.ubatch+255)//256)*256,
                  checks=checks)
    if args.compare_compact_raw:
        result["compact_checks"] = compact_checks
        result["compact_raw_rows"] = ((config["sliding_window"]+config["index_topk"]+255)//256)*256-config["index_topk"]
    path = args.report or args.fixture_dir / f"validation-gather-{args.backend.lower()}-{args.gpus}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    all_checks = checks + compact_checks
    passed = sum(check["passed"] for check in all_checks)
    print(f"Passed {passed}/{len(all_checks)} native gather comparisons; {path}")
    return int(passed != len(all_checks))


if __name__ == "__main__":
    raise SystemExit(main())
