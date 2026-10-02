#!/usr/bin/env python3
"""Finish a verified local mirror after its current writer exits.

Run detached to keep destination delivery independent of an interactive agent
turn. Existing publisher-verified report rows are retained; unfinished shards
use the downloader's resumable copy and destination SHA256 gate.
"""
import argparse
import contextlib
import ctypes
from ctypes import wintypes
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import unquote, urlsplit


SPEC = importlib.util.spec_from_file_location("model_downloader", Path(__file__).with_name("download-validated-model.py"))
DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADER)


def wait_for_process(pid):
    """Wait on a process handle, so later PID reuse cannot start a second writer."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x100000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error in (87, 1168):  # The original writer has already exited.
            return
        raise ctypes.WinError(error)
    try:
        if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) != 0:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def retain_verified_rows(manifest, report, directory):
    """Accept only publisher-matching rows published at the intended destination."""
    rows = []
    for item in manifest["shards"]:
        name = unquote(urlsplit(item["url"]).path.rsplit("/", 1)[-1])
        if not name or Path(name).name != name:
            raise ValueError("Invalid model basename")
        path = directory / name
        matches = [row for row in report.get("shards", [])
                   if all(row.get(key) == item[key] for key in ("url", "bytes", "sha256"))
                   and row.get("verified") is True and row.get("actual_sha256") == item["sha256"]
                   and Path(row.get("path", "")).resolve() == path.resolve()]
        if len(matches) == 1 and path.is_file() and path.stat().st_size == item["bytes"]:
            rows.append(matches[0])
    return rows


def finish(manifest_path, directory, source_directory, report_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    previous = json.loads(report_path.read_text(encoding="utf-8-sig")) if report_path.exists() else {}
    verified = retain_verified_rows(manifest, previous, directory)
    completed_urls = {row["url"] for row in verified}
    pending = [item for item in manifest["shards"] if item["url"] not in completed_urls]
    if pending:
        pending_manifest = report_path.with_name(report_path.stem + "-pending-manifest.json")
        pending_report = report_path.with_name(report_path.stem + "-pending-report.json")
        pending_manifest.write_text(json.dumps(dict(manifest, shards=pending), indent=2) + "\n", encoding="utf-8")
        old_argv = sys.argv
        try:
            sys.argv = [str(SPEC.origin), str(pending_manifest), "--directory", str(directory),
                        "--source-directory", str(source_directory), "--report", str(pending_report),
                        "--workers", "1", "--transport", "copy"]
            DOWNLOADER.main()
        finally:
            sys.argv = old_argv
        resumed = json.loads(pending_report.read_text(encoding="utf-8-sig"))
        verified.extend(retain_verified_rows(manifest, resumed, directory))
    by_url = {row["url"]: row for row in verified}
    if len(by_url) != len(manifest["shards"]):
        raise ValueError("Destination publisher verification is incomplete")
    final = dict(previous, status="verified", manifest=str(manifest_path), transport="copy",
                 source_directory=str(source_directory),
                 shards=[by_url[item["url"]] for item in manifest["shards"]],
                 completed_unix=time.time(), completion_guardian_pid=os.getpid(),
                 resumed_pending_shards=len(pending))
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--wait-for-pid", type=int, required=True)
    args = parser.parse_args()
    if args.wait_for_pid <= 0 or args.wait_for_pid == os.getpid():
        parser.error("wait-for-pid must name another positive process ID")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(json.dumps({"guardian_pid": os.getpid(), "waiting_for_pid": args.wait_for_pid,
                          "started_unix": time.time()}), flush=True)
        wait_for_process(args.wait_for_pid)
        final = finish(args.manifest, args.directory, args.source_directory, args.report)
        if args.environment:
            environment = json.loads(args.environment.read_text(encoding="utf-8-sig"))
            environment["mirror_process"].update(status="complete; all destination publisher hashes verified",
                destination_verified=True, completion_guardian_pid=os.getpid(), completed_unix=final["completed_unix"])
            args.environment.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "verified", "shards": len(final["shards"]),
                          "report": str(args.report), "completed_unix": final["completed_unix"]}), flush=True)


if __name__ == "__main__":
    main()
