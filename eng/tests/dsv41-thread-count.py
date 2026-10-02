#!/usr/bin/env python3
"""Observe actual tiny-model worker pools for native CLI/environment precedence.

Requires a test-enabled native library. No inference or full weights are used.
The observation API reports the width used to allocate the real worker pool.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not json.loads((args.fixture_dir / "deepseek41.config.json").read_text()).get("fixture"):
        raise ValueError("This diagnostic accepts only synthetic fixtures")
    lib = ctypes.CDLL(str(args.library.resolve()))
    load = lib.TSGgml_Dsv4LoadModel
    load.argtypes = [ctypes.c_char_p] + [ctypes.c_int] * 4 + [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    load.restype = ctypes.c_void_p
    free = lib.TSGgml_Dsv4Free
    free.argtypes, free.restype = [ctypes.c_void_p], None
    count = lib.TSGgml_Dsv4TestCpuThreads
    count.argtypes, count.restype = [ctypes.c_void_p], ctypes.c_int
    setter = lib.TSGgml_SetHostMoeThreads
    setter.argtypes, setter.restype = [ctypes.c_int], None
    settings = {"TS_CPU_MOE_THREADS": None,
                "TS_DSV41_ENGRAM_THREADS": "1", "TS_DSV41_ENGRAM_WARM": "0"}
    originals = {key: os.environ.get(key) for key in settings}
    checks = []
    try:
        for key, value in settings.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for offload in (0, 1):
            baseline = None
            for name, requested, environment, expected in (
                ("automatic", 0, None, None),
                ("cli", 7, None, 7),
                ("environment_over_cli", 7, "5", 5),
                ("invalid_environment_ignored", 7, "invalid", 7),
                ("zero_environment_ignored", 7, "0", 7),
                ("negative_environment_ignored", 7, "-2", 7),
                ("cli_zero_resets", 0, None, None),
                ("cli_negative_resets", -1, None, None),
                ("environment_after_cli_reset", 0, "5", 5),
            ):
                setter(requested)
                if environment is None:
                    os.environ.pop("TS_CPU_MOE_THREADS", None)
                else:
                    os.environ["TS_CPU_MOE_THREADS"] = environment
                handle = load(str(args.fixture_dir / "deepseek41-fixture.gguf").encode(),
                              1, 64, 3, 2, offload, b"CPU")
                if not handle:
                    raise RuntimeError("Tiny model failed to load")
                try:
                    actual = count(handle)
                finally:
                    free(handle)
                if baseline is None:
                    baseline = actual
                    if offload == 0:
                        expected = 2  # Explicit ordinary n_threads argument.
                    elif baseline < 1:
                        raise AssertionError("Offload automatic pool is empty")
                if expected is None:
                    expected = baseline
                passed = actual == expected
                checks.append(dict(name=name, cpu_moe=offload, cli=requested, environment=environment,
                                   actual_pool_threads=actual, expected=expected, passed=passed))
                if not passed:
                    raise AssertionError(checks[-1])
    finally:
        setter(0)
        for key, value in originals.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        with args.library.open("rb") as library:
            hasher = hashlib.sha256()
            for block in iter(lambda: library.read(1048576), b""):
                hasher.update(block)
            digest = hasher.hexdigest()
        args.report.write_text(json.dumps(dict(library_sha256=digest, backend="CPU",
            test_hooks_required=True, checks=checks), indent=2) + "\n")
    print(f"Passed {len(checks)}/{len(checks)} actual loader thread-count checks")


if __name__ == "__main__":
    main()
