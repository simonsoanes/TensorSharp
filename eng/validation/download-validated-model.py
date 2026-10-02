#!/usr/bin/env python3
"""Download public model shards from a pinned manifest and verify size and SHA256."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit, unquote


def download_xet(url, directory, expected_bytes, log):
    """Fetch an immutable Hub source; the caller verifies the whole-file SHA256."""
    parsed = urlsplit(url)
    match = re.fullmatch(r"/([^/]+/[^/]+)/resolve/([0-9a-f]{40})/(.+)", parsed.path)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or not match:
        raise ValueError("xet requires an immutable Hugging Face resolve URL")
    repository, revision, filename = match.groups()
    filename = unquote(filename)
    if any(part in ("", ".", "..") for part in filename.split("/")) or "\\" in filename:
        raise ValueError("Invalid Hugging Face filename")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import tqdm

    class LoggedProgress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            self.completed = 0
            self.started = time.monotonic()
            self.last_log = self.started

        def update(self, delta=1):
            self.completed += delta
            now = time.monotonic()
            if now - self.last_log >= 15 or self.completed >= expected_bytes:
                seconds = max(now - self.started, 0.001)
                row = {"file": filename, "reconstructed_bytes": self.completed,
                       "expected_bytes": expected_bytes, "seconds": seconds,
                       "MiB_per_second": self.completed / seconds / 1024**2,
                       "verified": False}
                print(json.dumps(row), file=log, flush=True)
                print(json.dumps(row), flush=True)
                self.last_log = now

    hub_directory = directory / ".hub"
    candidate = Path(hf_hub_download(repository, filename, revision=revision,
                                    local_dir=hub_directory, tqdm_class=LoggedProgress))
    if not candidate.resolve().is_relative_to(hub_directory.resolve()):
        raise ValueError("Hub returned a file outside its staging directory")
    return candidate


def download_buffered(url, part, expected_bytes, log, retries=5, source_offset=0, source_total_bytes=None):
    """Resume through large positioned writes, avoiding tiny O_APPEND writes on FUSE."""
    for attempt in range(retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset > expected_bytes:
            raise ValueError("Partial file exceeds expected size: " + part.name)
        if offset == expected_bytes:
            return
        if source_offset < 0:
            raise ValueError("Source offset must not be negative")
        remote_offset = source_offset + offset
        source_bytes = source_total_bytes if source_total_bytes is not None else source_offset + expected_bytes
        source_end = source_offset + expected_bytes - 1
        if source_bytes <= source_end:
            raise ValueError("Source range exceeds its total byte count")
        headers = {"Accept-Encoding": "identity"}
        if source_total_bytes is not None:
            headers["Range"] = f"bytes={remote_offset}-{source_end}"
        elif remote_offset:
            headers["Range"] = f"bytes={remote_offset}-"
        print(f"attempt={attempt + 1} offset={offset} expected_bytes={expected_bytes}", file=log, flush=True)
        try:
            request_url = url
            if source_total_bytes is not None:
                # A resumed subrange needs a fresh signed redirect, including when
                # the prior process stopped before its chunk was complete.
                separator = "&" if "?" in url else "?"
                request_url += separator + f"range_resume={remote_offset}&nonce={time.time_ns()}"
            request = urllib.request.Request(request_url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                ranged = remote_offset or source_total_bytes is not None
                expected_status = 206 if ranged else 200
                expected_range = f"bytes {remote_offset}-{source_end}/{source_bytes}" if ranged else None
                if response.status != expected_status or response.headers.get("Content-Range") != expected_range:
                    raise ValueError("Server did not honor the exact resumable byte range")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != expected_bytes - offset:
                    raise ValueError("Response Content-Length differs from the expected remaining bytes")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unexpected encoded model response")
                # r+b + seek preserves the existing prefix without O_APPEND's
                # per-write serialization on some network filesystems.
                with part.open("r+b" if part.exists() else "wb") as output:
                    output.seek(offset)
                    while offset < expected_bytes:
                        block = response.read(min(8 * 1024**2, expected_bytes - offset))
                        if not block:
                            raise OSError("Response ended before the expected file size")
                        output.write(block)
                        offset += len(block)
                    if response.read(1):
                        raise ValueError("Response exceeds the expected file size")
            return
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as error:
            print(f"retryable_error={error!r}", file=log, flush=True)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))


def download_ranged(item, part, log, workers=8, chunk_bytes=256 * 1024**2):
    """Resume bounded parallel ranges, then assemble for whole-file verification."""
    if workers < 1 or chunk_bytes < 1:
        raise ValueError("Range workers and chunk size must be positive")
    chunks = part.with_suffix(part.suffix + ".ranges")
    chunks.mkdir(exist_ok=True)
    identity_path = chunks / "source.json"
    source = {"url": item["url"], "bytes": item["bytes"], "sha256": item["sha256"], "chunk_bytes": chunk_bytes}
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if {key: identity.get(key) for key in source} != source:
            raise ValueError("Existing ranges belong to another source or chunk size")
        prefix_bytes = identity["prefix_bytes"]
    else:
        if any(chunks.iterdir()):
            raise ValueError("Existing ranges have no source identity")
        prefix_bytes = part.stat().st_size if part.exists() else 0
        identity_path.write_text(json.dumps(dict(source, prefix_bytes=prefix_bytes), indent=2) + "\n", encoding="utf-8")
    current = part.stat().st_size if part.exists() else 0
    if not 0 <= prefix_bytes <= current <= item["bytes"]:
        raise ValueError("Existing range prefix has an invalid byte count")
    jobs = [(offset, min(chunk_bytes, item["bytes"] - offset))
            for offset in range(prefix_bytes, item["bytes"], chunk_bytes)]
    started = time.monotonic()
    completed = prefix_bytes
    for offset, count in jobs:
        final = chunks / f"{offset:020d}.range"
        partial = final.with_suffix(".part")
        existing = final if final.exists() else partial
        if existing.exists():
            if existing.stat().st_size > count:
                raise ValueError("Existing range exceeds its expected byte count")
            completed += existing.stat().st_size
    received_this_run = 0

    def fetch(job):
        offset, count = job
        final = chunks / f"{offset:020d}.range"
        partial = final.with_suffix(".part")
        previous_bytes = final.stat().st_size if final.exists() else partial.stat().st_size if partial.exists() else 0
        if not final.exists():
            with final.with_suffix(".log").open("a") as chunk_log:
                # Bypass stale cached redirects whose signed policy names another range.
                separator = "&" if "?" in item["url"] else "?"
                url = item["url"] + separator + f"validation_range={offset}-{offset + count - 1}"
                download_buffered(url, partial, count, chunk_log, source_offset=offset,
                                  source_total_bytes=item["bytes"])
            partial.rename(final)
        if final.stat().st_size != count:
            raise ValueError("Completed range has an invalid byte count")
        return count - previous_bytes

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(fetch, job) for job in jobs]):
            received = future.result()
            completed += received
            received_this_run += received
            elapsed = max(time.monotonic() - started, 0.001)
            row = {"file": part.name, "received_bytes": completed, "expected_bytes": item["bytes"],
                   "seconds": elapsed, "MiB_per_second": received_this_run / elapsed / 1024**2,
                   "verified": False}
            print(json.dumps(row), file=log, flush=True)
            print(json.dumps(row), flush=True)
    # A partial assembly can be rerun from the recorded prefix without duplicating bytes.
    with part.open("r+b" if part.exists() else "wb") as output:
        output.seek(prefix_bytes)
        for offset, count in jobs:
            chunk_path = chunks / f"{offset:020d}.range"
            with chunk_path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b""):
                    output.write(block)
            output.flush()
            # Advance the reusable contiguous prefix before removing this redundant
            # range. Peak disk use stays near one checkpoint, even during assembly.
            identity = dict(source, prefix_bytes=offset + count)
            identity_temp = identity_path.with_suffix(".json.tmp")
            identity_temp.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
            identity_temp.replace(identity_path)
            chunk_path.unlink()
        output.truncate(item["bytes"])


def verify_copy_prefix(source, prefix, expected_bytes):
    """Compare a stopped local prefix with a fully verified source before reuse.

    This only qualifies the prefix. The completed destination must still pass
    the manifest's whole-file SHA256 check before it is published.
    """
    count = prefix.stat().st_size
    if source.stat().st_size != expected_bytes:
        raise ValueError("Local source size differs: " + source.name)
    if not 0 < count <= expected_bytes:
        raise ValueError("Prefix byte count must be inside its source")
    digests = []
    for path in (prefix, source):
        digest = hashlib.sha256()
        remaining = count
        with path.open("rb") as stream:
            while remaining:
                block = stream.read(min(8 * 1024**2, remaining))
                if not block:
                    raise ValueError("Prefix source ended before its byte count")
                digest.update(block)
                remaining -= len(block)
        digests.append(digest.hexdigest())
    if prefix.stat().st_size != count or source.stat().st_size != expected_bytes:
        raise ValueError("Prefix or source changed during verification")
    if digests[0] != digests[1]:
        raise ValueError("Prefix SHA256 differs from the verified source")
    return {"bytes": count, "sha256": digests[0], "whole_file_verified": False}


def copy_buffered(source, part, expected_bytes, log):
    """Resume a local mirror; destination bytes still require the manifest hash."""
    if source.stat().st_size != expected_bytes:
        raise ValueError("Local source size differs: " + source.name)
    offset = part.stat().st_size if part.exists() else 0
    if offset > expected_bytes:
        raise ValueError("Partial file exceeds expected size: " + part.name)
    started = last_log = time.monotonic()
    initial = offset
    with source.open("rb") as stream, part.open("r+b" if part.exists() else "wb") as output:
        stream.seek(offset)
        output.seek(offset)
        while offset < expected_bytes:
            block = stream.read(min(8 * 1024**2, expected_bytes - offset))
            if not block:
                raise OSError("Local source ended before the expected file size")
            output.write(block)
            offset += len(block)
            now = time.monotonic()
            if now - last_log >= 15 or offset == expected_bytes:
                elapsed = max(now - started, 0.001)
                row = {"file": part.name, "copied_bytes": offset, "expected_bytes": expected_bytes,
                       "seconds": elapsed, "MiB_per_second": (offset - initial) / elapsed / 1024**2,
                       "verified": False}
                print(json.dumps(row), file=log, flush=True)
                print(json.dumps(row), flush=True)
                last_log = now


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--range-workers", type=int, default=8, help="Parallel bounded ranges per shard in ranged mode")
    p.add_argument("--source-directory", type=Path, help="Read pinned source files locally for copy transport")
    p.add_argument("--transport", choices=("curl", "buffered", "ranged", "xet", "copy"), default="curl",
                   help="buffered/ranged use resumable HTTP/1.1; xet uses Hub chunks; copy mirrors local files")
    a = p.parse_args()
    if a.transport == "copy" and a.source_directory is None:
        p.error("copy transport requires --source-directory")
    manifest = json.loads(a.manifest.read_text(encoding="utf-8-sig"))
    a.directory.mkdir(parents=True, exist_ok=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    def download(item):
        name = unquote(urlsplit(item["url"]).path.rsplit("/", 1)[-1])
        if not name or Path(name).name != name:
            raise ValueError("Invalid model basename")
        path = a.directory / name
        part = path.with_suffix(path.suffix + (".mirror.part" if a.transport == "copy" else ".part"))
        if not path.exists():
            candidate = part
            if part.exists() and part.stat().st_size > item["bytes"]:
                raise ValueError("Partial file exceeds expected size: " + name)
            # Hash a complete partial after an interrupted verification. Asking
            # the server for an EOF byte range can otherwise refuse a valid file.
            if not part.exists() or part.stat().st_size < item["bytes"]:
                with (a.report.parent / (name + ".download.log")).open("a") as log:
                    if a.transport == "copy":
                        copy_buffered(a.source_directory / name, part, item["bytes"], log)
                    elif a.transport == "xet":
                        # Hub's chunk state stays separate from an HTTP prefix.
                        candidate = download_xet(item["url"], a.directory, item["bytes"], log)
                    elif a.transport == "ranged":
                        download_ranged(item, part, log, workers=a.range_workers)
                    elif a.transport == "buffered":
                        download_buffered(item["url"], part, item["bytes"], log)
                    else:
                        subprocess.run(["curl", "-fL", "--retry", "5", "--retry-all-errors", "--retry-delay", "2", "--continue-at", "-",
                                        "--output", str(part), item["url"]], stdout=log, stderr=subprocess.STDOUT, check=True)
        else:
            candidate = path
        if candidate.stat().st_size != item["bytes"]:
            raise ValueError("Size differs: " + name)
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != item["sha256"]:
            raise ValueError("SHA256 differs: " + name)
        if candidate != path:
            candidate.rename(path)
        row = dict(item, path=str(path), verified=True, actual_sha256=actual)
        print(name, "verified", flush=True)
        return row
    report = {"status": "running", "manifest": str(a.manifest), "transport": a.transport,
              "started_unix": started, "shards": []}
    if a.source_directory is not None:
        report["source_directory"] = str(a.source_directory)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            for row in pool.map(download, manifest["shards"]):
                report["shards"].append(row)
                a.report.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "verified"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["wall_seconds"] = time.time() - started
        a.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
