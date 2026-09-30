#!/usr/bin/env python3
"""Copy a Release host without custom native assets and verify CPU embedding APIs.

macOS vmmap and lsof inspect the live process after inference. Standard .NET and
Apple native libraries are allowed. No benchmark timings are collected.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import struct
import sys
import time
import urllib.request


def native_asset(path):
    return "native" in path.parts or path.suffix in (".dylib", ".metallib") or re.search(r"\.so(?:\.|$)", path.name)


def prepare_copy(source, target):
    marker = target / ".tensorsharp-nativefree-proof.json"
    if target.exists():
        if not marker.exists() or json.loads(marker.read_text()).get("source") != str(source):
            raise RuntimeError(f"Refusing to replace an unrelated directory: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    marker.write_text(json.dumps({"source": str(source)}))
    excluded = {"prefix-cache", "logs", "uploads", "code-scratch", "code-artifacts"}
    removed = []
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if relative.parts[0] in excluded or not path.is_file():
            continue
        if native_asset(relative):
            removed.append(str(relative))
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    assert not any(native_asset(path.relative_to(target)) for path in target.rglob("*") if path.is_file())
    return {"source": str(source), "copy": str(target), "excluded_runtime_data_directories": sorted(excluded),
            "removed_native_assets": sorted(removed), "remaining_custom_native_assets": [],
            "managed_assembly_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in sorted(target.glob("TensorSharp*.dll"))}}


def request(base, path, body=None):
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(base + path, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        assert response.status == 200
        return json.load(response)


def validate_vectors(rows, dimensions):
    assert len(rows) == 3, len(rows)
    norms = []
    for row in rows:
        assert len(row) == dimensions
        assert all(math.isfinite(value) for value in row)
        norm = math.sqrt(sum(value * value for value in row))
        assert abs(norm - 1) < 1e-5, norm
        norms.append(norm)
    return norms


def inspect_libraries(pid, dotnet, log_base):
    commands = [["/usr/bin/vmmap", "-w", str(pid)], ["/usr/sbin/lsof", "-p", str(pid), "-Fn"]]
    paths = set()
    inspections = []
    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        output_path = Path(str(log_base) + "." + Path(command[0]).name + ".txt")
        output_path.write_text(completed.stdout)
        inspections.append({"command": command, "exit_code": completed.returncode, "raw_output": str(output_path)})
        for line in completed.stdout.splitlines():
            # lsof's -Fn prefixes file names with n; vmmap has permission
            # fields such as r-x/r-x before the absolute image path.
            candidate = line[1:] if line.startswith("n/") else line
            match = re.search(r"(?:^|\s)(/[^\n]+(?:\.dylib|\.so(?:\.[0-9]+)*|\.bundle))\s*$", candidate)
            if match:
                paths.add(match.group(1))
    assert paths, "Library inspection did not identify any loaded native images."
    dotnet_root = str(Path(dotnet).resolve().parent) + "/"
    allowed = ("/usr/lib/", "/System/Library/", dotnet_root)
    custom = sorted(path for path in paths if not path.startswith(allowed))
    assert not custom, f"Unexpected custom native images loaded: {custom}"
    return {"tools": inspections, "loaded_native_images": sorted(paths), "allowed_native_roots": list(allowed),
            "custom_native_images": custom, "custom_native_library_loaded": False}


def verify_model(model, target, port, threads, dotnet, context_boundaries=False):
    name = model.stem
    log_base = target.parent / ("nativefree-" + name)
    command = [dotnet, str(target / "TensorSharp.Server.Host.dll"), "--model", str(model), "--embeddings",
               "--backend", "cpu", "--embedding-threads", str(threads), "--host", "127.0.0.1",
               "--port", str(port), "--no-webui", "--no-skills"]
    environment = dict(os.environ, TENSORSHARP_LOG_FILE="0", TENSORSHARP_LOG_LEVEL="Information")
    for key in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH",
                "TENSORSHARP_MLX_LIBRARY", "TENSORSHARP_MLX_LIBRARY_DIR"):
        environment.pop(key, None)
    base = f"http://127.0.0.1:{port}"
    result = {"model": str(model), "command": command, "working_directory": str(target),
              "environment_overrides": {"TENSORSHARP_LOG_FILE": "0", "TENSORSHARP_LOG_LEVEL": "Information"}}
    print(f"Starting native-free CPU host: {name}", flush=True)
    with Path(str(log_base) + ".host.log").open("w") as log:
        process = subprocess.Popen(command, cwd=target, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                assert process.poll() is None, f"Host exited: {log_base}.host.log"
                try:
                    request(base, "/health")
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Host did not become ready: {log_base}.host.log")
                    time.sleep(0.1)
            models = request(base, "/v1/models")["data"]
            assert len(models) == 1 and models[0]["id"] == name
            dimensions = models[0]["embedding_dimensions"]
            metadata = request(base, "/api/models")
            assert metadata["loadedBackend"] == "cpu"
            assert [backend["value"] for backend in metadata["supportedBackends"]] == ["cpu"]
            texts = ["Hello world", "Searching documents using semantic similarity.", "检索多语言文档。"]
            openai = request(base, "/v1/embeddings", {"model": name, "input": texts, "encoding_format": "float"})
            assert [row["index"] for row in openai["data"]] == [0, 1, 2]
            vectors = [row["embedding"] for row in openai["data"]]
            norms = validate_vectors(vectors, dimensions)
            ollama = request(base, "/api/embed", {"model": name, "input": texts})
            encoded = request(base, "/v1/embeddings", {"model": name, "input": texts, "encoding_format": "base64"})
            decoded = [list(struct.unpack("<" + "f" * dimensions, base64.b64decode(row["embedding"]))) for row in encoded["data"]]
            differences = {}
            for protocol, rows in (("ollama", ollama["embeddings"]), ("openai_base64", decoded)):
                validate_vectors(rows, dimensions)
                difference = max(abs(a - b) for expected, actual in zip(vectors, rows) for a, b in zip(expected, actual))
                assert difference < 1e-6, (protocol, difference)
                differences[protocol] = difference
            tokens = openai["usage"]["prompt_tokens"]
            assert tokens > 0 and tokens == openai["usage"]["total_tokens"] == ollama["prompt_eval_count"] == encoded["usage"]["prompt_tokens"]
            result.update(dimensions=dimensions, input_count=len(texts), prompt_tokens=tokens, vector_norms=norms,
                          protocol_max_absolute_differences=differences, loaded_backend=metadata["loadedBackend"],
                          passed=True)
            if context_boundaries:
                from context_boundaries import verify_boundaries
                result["context_boundaries"] = verify_boundaries(base, name, dimensions, models[0]["context_length"])
            result["library_inspection"] = inspect_libraries(process.pid, dotnet, log_base)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("Owned proof host did not stop gracefully.")
            result["host_exit_code"] = process.returncode
    assert result["host_exit_code"] == 0, result["host_exit_code"]
    assert "execution=pure-csharp" in Path(str(log_base) + ".host.log").read_text()
    print(f"Passed {name}: {dimensions} dimensions; all APIs agree; no custom native images; host stopped.", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("TensorSharp.Server.Host/bin"))
    parser.add_argument("--copy", type=Path, default=Path("/tmp/tensorsharp-embedding-managed-host"))
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--port", type=int, default=18383)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-output", type=Path, help="Also run full-context boundaries and write their compact evidence here.")
    args = parser.parse_args()
    assert shutil.which("dotnet"), "dotnet must be installed"
    result = {"purpose": "Native-free managed CPU deployment correctness; no benchmark timings collected.",
              "verification_command": [sys.executable, *sys.argv],
              "configuration": "Release", "copy": prepare_copy(args.source.resolve(), args.copy.resolve()), "models": []}
    for model in args.model:
        result["models"].append(verify_model(model.resolve(), args.copy.resolve(), args.port, args.threads,
                                           shutil.which("dotnet"), args.context_output is not None))
    result["passed"] = True
    result["owned_hosts_stopped"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.context_output:
        args.context_output.parent.mkdir(parents=True, exist_ok=True)
        args.context_output.write_text(json.dumps({"verification_command": result["verification_command"],
            "backend": "cpu", "native_free_host": True, "passed": True,
            "managed_assembly_sha256": result["copy"]["managed_assembly_sha256"],
            "models": [model["context_boundaries"] for model in result["models"]]}, indent=2) + "\n")
    print(f"Evidence: {args.output}", flush=True)


if __name__ == "__main__":
    main()
