#!/usr/bin/env python3
"""Check actual V4.1 shared-expert placement and independent fixture logits.

Requires native test hooks. The legacy control disables only the new shared
projection pins, preserving every other operation and allocation policy.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backend", choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    if args.backend == "CUDA" and args.gpus < 2:
        parser.error("CUDA placement checks require at least two GPUs for TP")
    manifest = json.loads((args.fixture_dir / "deepseek41.config.json").read_text())
    if not manifest.get("fixture"):
        raise ValueError("Only the tiny synthetic fixture may be used")
    config = manifest["config"]
    text = config["text_config"]
    spec = importlib.util.spec_from_file_location("reference", Path(__file__).parents[1] / "dsv41-reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(2)
    tokens = np.array(json.loads((args.fixture_dir / "tokens.json").read_text())[:8], dtype=np.int32)
    weights = reference.GgufWeights(args.fixture_dir / "deepseek41-fixture.gguf")
    layout = reference.load_engram(weights)
    expected = reference.Reference(weights, config, layout, "model").forward(tokens.tolist()).cpu().numpy()
    lib = ctypes.CDLL(str(args.library.resolve()))
    def bind(name, parameters, result=ctypes.c_int):
        function = getattr(lib, "TSGgml_" + name)
        function.argtypes, function.restype = parameters, result
        return function
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    load = bind("Dsv4LoadModel", [ctypes.c_char_p] + [integer] * 4 + [ctypes.c_char_p, integer, ctypes.c_char_p, integer], ptr)
    free = bind("Dsv4Free", [ptr], None)
    forward = bind("Dsv4Forward", [ptr, ptr, integer, ptr])
    past = bind("Dsv4NPast", [ptr])
    placement = bind("Dsv4TestSharedPlacement", [ptr, integer, integer, ptr, integer])
    settings = {"TS_DSV4_FA": "0", "TS_DSV4_GATHER": "0", "TS_CPU_MOE_THREADS": "2",
                "TS_DSV41_ENGRAM_THREADS": "1", "TS_DSV41_ENGRAM_WARM": "0",
                "TS_DSV41_TEST_LEGACY_SHARED_PLACEMENT": "0"}
    original = {key: os.environ.get(key) for key in settings}
    runs, checks = [], []
    def check(name, valid, **details):
        checks.append(dict(name=name, passed=bool(valid), **details))
    configurations = [(0, 0), (0, 1)]
    if args.backend == "CUDA":
        configurations += [(args.gpus, 0), (args.gpus, 1)]
    try:
        os.environ.update(settings)
        for tp, offload in configurations:
            for legacy in (True, False):
                os.environ["TS_DSV41_TEST_LEGACY_SHARED_PLACEMENT"] = str(int(legacy))
                label = f"tp{tp}_cpumoe{offload}_{'legacy' if legacy else 'pinned'}"
                handle = load(str(args.fixture_dir / "deepseek41-fixture.gguf").encode(),
                              args.gpus, 64, 4, 2, None, offload, args.backend.encode(), tp)
                if not handle:
                    raise RuntimeError(label + " load failed")
                run = dict(name=label, tp=tp, cpu_moe=offload, legacy=legacy, placement=[], logits=[])
                runs.append(run)
                try:
                    start = 0
                    for stop in (3, 4, 8):
                        output = np.empty(text["vocab_size"], dtype=np.float32)
                        status = forward(handle, tokens[start:].ctypes.data, stop - start, output.ctypes.data)
                        check(label + f"_forward_{stop}", status == 0 and past(handle) == stop)
                        if status != 0:
                            raise RuntimeError(label + " forward failed")
                        target = expected[stop - 1]
                        valid = np.allclose(output, target, atol=2e-5, rtol=2e-5)
                        metrics = dict(position=stop, max_absolute_error=float(np.max(np.abs(output - target))),
                                       relative_l2=float(np.linalg.norm(output - target) / np.linalg.norm(target)),
                                       argmax=int(output.argmax()), reference_argmax=int(target.argmax()))
                        run["logits"].append(metrics)
                        check(label + f"_oracle_{stop}", valid, **metrics)
                        misplaced = 0
                        for layer in range(text["num_hidden_layers"]):
                            for projection, name in enumerate(("gate", "up", "down")):
                                description = ctypes.create_string_buffer(128)
                                match = placement(handle, layer, projection, description, len(description))
                                row = dict(position=stop, layer=layer, projection=name,
                                           matches_layer_backend=match == 1,
                                           backend=description.value.decode())
                                run["placement"].append(row)
                                check(label + f"_found_{stop}_{layer}_{name}", match >= 0)
                                if not legacy or args.backend == "CPU":
                                    check(label + f"_placement_{stop}_{layer}_{name}", match == 1, **row)
                                misplaced += match == 0
                        if legacy and args.backend == "CUDA" and (tp or offload):
                            check(label + f"_reproduced_old_misplacement_{stop}", misplaced > 0,
                                  misplaced_projections=misplaced)
                        start = stop
                finally:
                    free(handle)
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        hasher = hashlib.sha256()
        with args.library.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                hasher.update(block)
        args.report.write_text(json.dumps(dict(backend=args.backend, gpus=args.gpus,
            library_sha256=hasher.hexdigest(), test_hooks_required=True,
            runs=runs, checks=checks), indent=2) + "\n")
    passed = sum(row["passed"] for row in checks)
    print(f"Passed {passed}/{len(checks)} placement/oracle checks; {args.report}")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
