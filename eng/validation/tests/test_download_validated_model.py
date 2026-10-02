#!/usr/bin/env python3
"""Local recovery tests for completed and invalid partial model downloads."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "download-validated-model.py"
SPEC = importlib.util.spec_from_file_location("download_validated_model", SCRIPT)
DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADER)


class PartialDownloadRecoveryTests(unittest.TestCase):
    def run_partial(self, payload, expected, error=None, transport="curl"):
        with tempfile.TemporaryDirectory(prefix="tensorsharp-download-test-") as temporary:
            directory = Path(temporary).resolve()
            self.assertEqual(directory.parent, Path(tempfile.gettempdir()).resolve())
            partial = directory / "weights.gguf.part"
            final = directory / "weights.gguf"
            partial.write_bytes(payload)
            manifest = directory / "manifest.json"
            report = directory / "report.json"
            expected_hash = hashlib.sha256(expected).hexdigest()
            manifest.write_text(json.dumps({"shards": [{
                "url": "https://example.invalid/weights.gguf",
                "bytes": len(expected), "sha256": expected_hash,
            }]}), encoding="utf-8")
            argv = [str(SCRIPT), str(manifest), "--directory", str(directory),
                    "--report", str(report), "--transport", transport]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(DOWNLOADER.subprocess, "run") as curl, \
                    mock.patch.object(DOWNLOADER.urllib.request, "urlopen") as urlopen, \
                    contextlib.redirect_stdout(io.StringIO()):
                if error:
                    with self.assertRaisesRegex(ValueError, error):
                        DOWNLOADER.main()
                else:
                    DOWNLOADER.main()
                curl.assert_not_called()
                urlopen.assert_not_called()
            result = json.loads(report.read_text(encoding="utf-8"))
            if error:
                self.assertEqual(result["status"], "failed")
                self.assertFalse(final.exists(), "An unverified file must not be published")
                self.assertEqual(partial.read_bytes(), payload)
                self.assertEqual(result["shards"], [])
            else:
                self.assertEqual(result["status"], "verified")
                self.assertEqual(final.read_bytes(), expected)
                self.assertFalse(partial.exists())
                self.assertTrue(result["shards"][0]["verified"])
                self.assertEqual(result["shards"][0]["actual_sha256"], expected_hash)

    def test_completed_verified_partial_bypasses_curl_and_publishes(self):
        for transport in ("curl", "buffered", "ranged", "xet"):
            with self.subTest(transport=transport):
                self.run_partial(b"complete validated payload", b"complete validated payload", transport=transport)

    def test_completed_bad_hash_stays_unpublished(self):
        for transport in ("curl", "buffered", "ranged", "xet"):
            with self.subTest(transport=transport):
                self.run_partial(b"bad data", b"gooddata", "SHA256 differs", transport=transport)

    def test_oversized_partial_refuses_without_network(self):
        for transport in ("curl", "buffered", "ranged", "xet"):
            with self.subTest(transport=transport):
                self.run_partial(b"oversized payload", b"small", "Partial file exceeds expected size", transport=transport)


class BufferedTransportTests(unittest.TestCase):
    def response(self, status, headers, reads):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = status
        response.headers = headers
        response.read.side_effect = reads
        return response

    def test_rejected_full_response_preserves_existing_prefix(self):
        response = self.response(200, {"Content-Length": "8"}, [])
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.part"
            part.write_bytes(b"abc")
            with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", return_value=response), \
                    self.assertRaisesRegex(ValueError, "exact resumable byte range"):
                DOWNLOADER.download_buffered("https://example.invalid/weights", part, 8, io.StringIO())
            self.assertEqual(part.read_bytes(), b"abc")
            response.read.assert_not_called()

    def test_mismatched_range_preserves_existing_prefix(self):
        for content_range in ("bytes 2-7/8", "bytes 3-7/9", "bytes 3-6/8"):
            with self.subTest(content_range=content_range), tempfile.TemporaryDirectory() as temporary:
                response = self.response(206, {"Content-Range": content_range}, [])
                part = Path(temporary) / "weights.part"
                part.write_bytes(b"abc")
                with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", return_value=response), \
                        self.assertRaisesRegex(ValueError, "exact resumable byte range"):
                    DOWNLOADER.download_buffered("https://example.invalid/weights", part, 8, io.StringIO())
                self.assertEqual(part.read_bytes(), b"abc")
                response.read.assert_not_called()

    def test_interrupted_response_resumes_without_duplicate_bytes(self):
        first = self.response(200, {"Content-Length": "8"}, [b"abc", OSError("interrupted")])
        second = self.response(206, {"Content-Range": "bytes 3-7/8", "Content-Length": "5"}, [b"defgh", b""])
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.part"
            with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", side_effect=[first, second]) as urlopen, \
                    mock.patch.object(DOWNLOADER.time, "sleep"):
                DOWNLOADER.download_buffered("https://example.invalid/weights", part, 8, io.StringIO())
            self.assertEqual(part.read_bytes(), b"abcdefgh")
            requests = [call.args[0] for call in urlopen.call_args_list]
            self.assertIsNone(requests[0].get_header("Range"))
            self.assertEqual(requests[1].get_header("Range"), "bytes=3-")

    def test_retry_exhaustion_preserves_received_prefix(self):
        response = self.response(200, {"Content-Length": "8"}, [b"abc", OSError("interrupted")])
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.part"
            with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", return_value=response) as urlopen, \
                    self.assertRaisesRegex(OSError, "interrupted"):
                DOWNLOADER.download_buffered("https://example.invalid/weights", part, 8, io.StringIO(), retries=0)
            self.assertEqual(part.read_bytes(), b"abc")
            self.assertEqual(urlopen.call_count, 1)

    def test_staged_tail_resume_uses_absolute_source_offsets(self):
        response = self.response(206, {"Content-Range": "bytes 6-7/8", "Content-Length": "2"}, [b"gh", b""])
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.tail.part"
            part.write_bytes(b"def")
            with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", return_value=response) as urlopen:
                DOWNLOADER.download_buffered("https://example.invalid/weights", part, 5, io.StringIO(), source_offset=3)
            self.assertEqual(part.read_bytes(), b"defgh")
            self.assertEqual(urlopen.call_args.args[0].get_header("Range"), "bytes=6-")

    def test_bounded_range_checks_source_total_and_absolute_end(self):
        response = self.response(206, {"Content-Range": "bytes 4-5/8", "Content-Length": "2"}, [b"ef", b""])
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "chunk.part"
            part.write_bytes(b"d")
            with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", return_value=response) as urlopen:
                DOWNLOADER.download_buffered("https://example.invalid/weights", part, 3, io.StringIO(),
                                             source_offset=3, source_total_bytes=8)
            self.assertEqual(part.read_bytes(), b"def")
            self.assertEqual(urlopen.call_args.args[0].get_header("Range"), "bytes=4-5")

    def test_bounded_range_refuses_wrong_total_before_overwriting_prefix(self):
        response = self.response(206, {"Content-Range": "bytes 4-5/9"}, [])
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "chunk.part"
            part.write_bytes(b"d")
            with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", return_value=response), \
                    self.assertRaisesRegex(ValueError, "exact resumable byte range"):
                DOWNLOADER.download_buffered("https://example.invalid/weights", part, 3, io.StringIO(),
                                             source_offset=3, source_total_bytes=8)
            self.assertEqual(part.read_bytes(), b"d")

    def test_bounded_range_retry_refreshes_redirect_for_resumed_offset(self):
        first = self.response(206, {"Content-Range": "bytes 8-15/32", "Content-Length": "8"},
                              [b"abc", OSError("interrupted")])
        second = self.response(206, {"Content-Range": "bytes 11-15/32", "Content-Length": "5"},
                               [b"defgh", b""])
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "chunk.part"
            with mock.patch.object(DOWNLOADER.urllib.request, "urlopen", side_effect=[first, second]) as urlopen, \
                    mock.patch.object(DOWNLOADER.time, "time_ns", side_effect=[123, 456]), \
                    mock.patch.object(DOWNLOADER.time, "sleep"):
                DOWNLOADER.download_buffered("https://example.invalid/weights?chunk=1", part, 8, io.StringIO(),
                                             source_offset=8, source_total_bytes=32)
            self.assertEqual(part.read_bytes(), b"abcdefgh")
            requests = [call.args[0] for call in urlopen.call_args_list]
            self.assertEqual(requests[0].get_header("Range"), "bytes=8-15")
            self.assertEqual(requests[1].get_header("Range"), "bytes=11-15")
            self.assertTrue(requests[0].full_url.endswith("&range_resume=8&nonce=123"))
            self.assertTrue(requests[1].full_url.endswith("&range_resume=11&nonce=456"))

    def test_buffered_download_is_hashed_before_publish(self):
        for payload, expected_status in ((b"abcdefgh", "verified"), (b"bad-data", "failed")):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                manifest, report = directory / "manifest.json", directory / "report.json"
                manifest.write_text(json.dumps({"shards": [{"url": "https://example.invalid/weights.gguf",
                    "bytes": 8, "sha256": hashlib.sha256(b"abcdefgh").hexdigest()}]}))
                response = self.response(200, {"Content-Length": "8"}, [payload, b""])
                argv = [str(SCRIPT), str(manifest), "--directory", str(directory), "--report", str(report),
                        "--transport", "buffered"]
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(DOWNLOADER.urllib.request, "urlopen", return_value=response), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if expected_status == "failed":
                        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                            DOWNLOADER.main()
                    else:
                        DOWNLOADER.main()
                self.assertEqual(json.loads(report.read_text())["status"], expected_status)
                self.assertEqual((directory / "weights.gguf").exists(), expected_status == "verified")


class XetTransportTests(unittest.TestCase):
    revision = "0123456789abcdef0123456789abcdef01234567"
    url = f"https://huggingface.co/example/model/resolve/{revision}/UD-IQ1_M/weights.gguf"

    def hub_modules(self, download):
        class ProgressBase:
            def __init__(self, *args, **kwargs):
                pass
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = download
        utils = types.ModuleType("huggingface_hub.utils")
        utils.tqdm = ProgressBase
        return {"huggingface_hub": hub, "huggingface_hub.utils": utils}

    def test_rejects_unpinned_or_untrusted_urls_before_network(self):
        urls = (self.url.replace(self.revision, "main"), self.url.replace("https:", "http:"),
                self.url.replace("huggingface.co", "huggingface.co.example.invalid"),
                self.url.replace("UD-IQ1_M/", "%2e%2e/"))
        download = mock.Mock()
        with mock.patch.dict(sys.modules, self.hub_modules(download)):
            for url in urls:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    DOWNLOADER.download_xet(url, Path("unused"), 8, io.StringIO())
        download.assert_not_called()

    def test_pinned_repository_revision_and_filename_reach_hub(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            candidate = directory / ".hub" / "UD-IQ1_M" / "weights.gguf"
            download = mock.Mock(return_value=str(candidate))
            with mock.patch.dict(sys.modules, self.hub_modules(download)):
                result = DOWNLOADER.download_xet(self.url, directory, 8, io.StringIO())
            self.assertEqual(result, candidate)
            self.assertEqual(download.call_args.args, ("example/model", "UD-IQ1_M/weights.gguf"))
            self.assertEqual(download.call_args.kwargs["revision"], self.revision)
            self.assertEqual(download.call_args.kwargs["local_dir"], directory / ".hub")
            self.assertIn("tqdm_class", download.call_args.kwargs)

    def test_refuses_hub_output_outside_staging_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            download = mock.Mock(return_value=str(directory / "outside.gguf"))
            with mock.patch.dict(sys.modules, self.hub_modules(download)), \
                    self.assertRaisesRegex(ValueError, "outside its staging directory"):
                DOWNLOADER.download_xet(self.url, directory, 8, io.StringIO())

    def test_whole_file_size_and_hash_gate_xet_publish_and_preserve_http_prefix(self):
        for payload, error in ((b"abcdefgh", None), (b"bad-data", "SHA256 differs"),
                               (b"too-short", "Size differs")):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                manifest, report = directory / "manifest.json", directory / "report.json"
                expected_hash = hashlib.sha256(b"abcdefgh").hexdigest()
                manifest.write_text(json.dumps({"shards": [{"url": self.url, "bytes": 8,
                    "sha256": expected_hash}]}), encoding="utf-8-sig")
                prefix = directory / "weights.gguf.part"
                prefix.write_bytes(b"old")
                candidate = directory / ".hub" / "UD-IQ1_M" / "weights.gguf"
                candidate.parent.mkdir(parents=True)
                candidate.write_bytes(payload)
                download = mock.Mock(return_value=str(candidate))
                argv = [str(SCRIPT), str(manifest), "--directory", str(directory), "--report", str(report),
                        "--transport", "xet"]
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.dict(sys.modules, self.hub_modules(download)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            DOWNLOADER.main()
                    else:
                        DOWNLOADER.main()
                final = directory / "weights.gguf"
                result = json.loads(report.read_text())
                self.assertEqual(prefix.read_bytes(), b"old")
                self.assertEqual(final.exists(), error is None)
                self.assertEqual(result["status"], "failed" if error else "verified")
                if not error:
                    self.assertEqual(final.read_bytes(), payload)
                    self.assertEqual(result["shards"][0]["actual_sha256"], expected_hash)


class RangedTransportTests(unittest.TestCase):
    payload = b"abcdefghijklmnopqrstuvwxyz"

    def item(self):
        return {"url": "https://example.invalid/weights.gguf", "bytes": len(self.payload),
                "sha256": hashlib.sha256(self.payload).hexdigest()}

    def fetch(self, url, partial, count, log, source_offset=0, source_total_bytes=None):
        self.assertEqual(source_total_bytes, len(self.payload))
        existing = partial.read_bytes() if partial.exists() else b""
        self.assertEqual(existing, self.payload[source_offset:source_offset + len(existing)])
        partial.write_bytes(self.payload[source_offset:source_offset + count])

    def test_parallel_ranges_preserve_prefix_and_assemble_in_offset_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.gguf.part"
            part.write_bytes(self.payload[:3])
            with mock.patch.object(DOWNLOADER, "download_buffered", side_effect=self.fetch) as fetch, \
                    contextlib.redirect_stdout(io.StringIO()):
                DOWNLOADER.download_ranged(self.item(), part, io.StringIO(), workers=3, chunk_bytes=5)
            self.assertEqual(part.read_bytes(), self.payload)
            self.assertEqual(fetch.call_count, 5)
            self.assertEqual(sorted(call.kwargs["source_offset"] for call in fetch.call_args_list), [3, 8, 13, 18, 23])
            chunks = part.with_suffix(part.suffix + ".ranges")
            self.assertEqual(list(chunks.glob("*.range")), [])
            self.assertEqual(json.loads((chunks / "source.json").read_text())["prefix_bytes"], len(self.payload))

    def test_interrupted_chunk_resume_keeps_completed_chunks_and_source_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.gguf.part"
            part.write_bytes(self.payload[:3])
            chunks = part.with_suffix(part.suffix + ".ranges")
            chunks.mkdir()
            identity = dict(self.item(), chunk_bytes=5, prefix_bytes=3)
            (chunks / "source.json").write_text(json.dumps(identity))
            (chunks / f"{3:020d}.range").write_bytes(self.payload[3:8])
            (chunks / f"{8:020d}.part").write_bytes(self.payload[8:10])
            # A previous assembly was also interrupted after writing four tail bytes.
            part.write_bytes(self.payload[:7])
            with mock.patch.object(DOWNLOADER, "download_buffered", side_effect=self.fetch) as fetch, \
                    contextlib.redirect_stdout(io.StringIO()):
                DOWNLOADER.download_ranged(self.item(), part, io.StringIO(), workers=2, chunk_bytes=5)
            self.assertEqual(part.read_bytes(), self.payload)
            self.assertEqual(fetch.call_count, 4)

    def test_compacted_assembly_resumes_after_removed_range_without_redownload(self):
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.gguf.part"
            part.write_bytes(self.payload[:10])
            chunks = part.with_suffix(part.suffix + ".ranges")
            chunks.mkdir()
            # The first assembled tail range was removed after the source prefix advanced.
            (chunks / "source.json").write_text(json.dumps(dict(self.item(), chunk_bytes=5, prefix_bytes=8)))
            for offset in (8, 13, 18, 23):
                (chunks / f"{offset:020d}.range").write_bytes(self.payload[offset:offset + 5])
            with mock.patch.object(DOWNLOADER, "download_buffered") as fetch, \
                    contextlib.redirect_stdout(io.StringIO()):
                DOWNLOADER.download_ranged(self.item(), part, io.StringIO(), workers=2, chunk_bytes=5)
            fetch.assert_not_called()
            self.assertEqual(part.read_bytes(), self.payload)

    def test_source_change_refuses_existing_chunks_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            part = Path(temporary) / "weights.gguf.part"
            chunks = part.with_suffix(part.suffix + ".ranges")
            chunks.mkdir()
            (chunks / "source.json").write_text(json.dumps(dict(self.item(), chunk_bytes=5, prefix_bytes=0)))
            for changed in (dict(self.item(), sha256="0" * 64), dict(self.item(), bytes=27)):
                with self.subTest(changed=changed), mock.patch.object(DOWNLOADER, "download_buffered") as fetch, \
                        self.assertRaisesRegex(ValueError, "another source"):
                    DOWNLOADER.download_ranged(changed, part, io.StringIO(), workers=2, chunk_bytes=5)
                fetch.assert_not_called()


class CopyTransportTests(unittest.TestCase):
    def test_existing_prefix_matches_only_the_corresponding_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, prefix = Path(temporary) / "source.gguf", Path(temporary) / "prefix.part"
            source.write_bytes(b"abcdefgh")
            prefix.write_bytes(b"abc")
            result = DOWNLOADER.verify_copy_prefix(source, prefix, 8)
            self.assertEqual(result["bytes"], 3)
            self.assertEqual(result["sha256"], hashlib.sha256(b"abc").hexdigest())
            self.assertFalse(result["whole_file_verified"])
            self.assertEqual(prefix.read_bytes(), b"abc")

    def test_invalid_prefix_is_refused_without_mutating_it(self):
        for data, message in ((b"bad", "SHA256 differs"), (b"", "byte count"),
                              (b"abcdefghi", "byte count")):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as temporary:
                source, prefix = Path(temporary) / "source.gguf", Path(temporary) / "prefix.part"
                source.write_bytes(b"abcdefgh")
                prefix.write_bytes(data)
                with self.assertRaisesRegex(ValueError, message):
                    DOWNLOADER.verify_copy_prefix(source, prefix, 8)
                self.assertEqual(prefix.read_bytes(), data)

    def test_copy_resumes_destination_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, part = Path(temporary) / "source.gguf", Path(temporary) / "target.part"
            source.write_bytes(b"abcdefgh")
            part.write_bytes(b"abc")
            with contextlib.redirect_stdout(io.StringIO()):
                DOWNLOADER.copy_buffered(source, part, 8, io.StringIO())
            self.assertEqual(part.read_bytes(), b"abcdefgh")

    def test_destination_hash_gate_and_separate_mirror_partial_preserve_download_prefix(self):
        for payload, error in ((b"abcdefgh", None), (b"bad-data", "SHA256 differs")):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                source_dir, destination = directory / "source", directory / "destination"
                source_dir.mkdir()
                destination.mkdir()
                (source_dir / "weights.gguf").write_bytes(payload)
                ordinary_partial = destination / "weights.gguf.part"
                ordinary_partial.write_bytes(b"old-unverified-download-prefix")
                manifest, report = directory / "manifest.json", directory / "report.json"
                expected_hash = hashlib.sha256(b"abcdefgh").hexdigest()
                manifest.write_text(json.dumps({"shards": [{"url": "https://example.invalid/weights.gguf",
                    "bytes": 8, "sha256": expected_hash}]}))
                argv = [str(SCRIPT), str(manifest), "--directory", str(destination), "--report", str(report),
                        "--transport", "copy", "--source-directory", str(source_dir)]
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            DOWNLOADER.main()
                    else:
                        DOWNLOADER.main()
                result = json.loads(report.read_text())
                self.assertEqual(result["status"], "failed" if error else "verified")
                self.assertEqual((destination / "weights.gguf").exists(), error is None)
                self.assertEqual(ordinary_partial.read_bytes(), b"old-unverified-download-prefix")
                if not error:
                    self.assertEqual(result["shards"][0]["actual_sha256"], expected_hash)


if __name__ == "__main__":
    unittest.main()
