import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "finish-model-mirror.py"
SPEC = importlib.util.spec_from_file_location("finish_model_mirror", SCRIPT)
MIRROR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIRROR)


class MirrorContinuationTests(unittest.TestCase):
    def fixture(self, root):
        source, destination = root / "source", root / "destination"
        source.mkdir()
        destination.mkdir()
        items = []
        for index, payload in enumerate((b"first-model-shard", b"second-model-shard")):
            name = f"weights-{index}.gguf"
            (source / name).write_bytes(payload)
            items.append({"url": "https://example.invalid/" + name, "bytes": len(payload),
                          "sha256": hashlib.sha256(payload).hexdigest()})
        path = root / "manifest.json"
        path.write_text(json.dumps({"shards": items}))
        return source, destination, path, items

    def test_retains_verified_row_and_resumes_only_the_unfinished_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination, manifest, items = self.fixture(root)
            first = destination / "weights-0.gguf"
            first.write_bytes((source / first.name).read_bytes())
            (destination / "weights-1.gguf.mirror.part").write_bytes(b"second")
            row = dict(items[0], path=str(first), verified=True, actual_sha256=items[0]["sha256"])
            report = root / "report.json"
            report.write_text(json.dumps({"status": "running", "shards": [row]}))
            result = MIRROR.finish(manifest, destination, source, report)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["resumed_pending_shards"], 1)
            self.assertEqual(len(result["shards"]), 2)
            self.assertEqual((destination / "weights-1.gguf").read_bytes(), b"second-model-shard")
            self.assertEqual(json.loads(report.read_text())["status"], "verified")

    def test_mismatched_hash_path_or_missing_file_cannot_skip_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, destination, manifest_path, items = self.fixture(Path(temporary))
            manifest = json.loads(manifest_path.read_text())
            path = destination / "weights-0.gguf"
            path.write_bytes((source / path.name).read_bytes())
            good = dict(items[0], path=str(path), verified=True, actual_sha256=items[0]["sha256"])
            for row in (dict(good, actual_sha256="0" * 64), dict(good, path=str(source / path.name)),
                        dict(good, verified=False), dict(good, bytes=1)):
                with self.subTest(row=row):
                    self.assertEqual(MIRROR.retain_verified_rows(manifest, {"shards": [row]}, destination), [])
            path.unlink()
            self.assertEqual(MIRROR.retain_verified_rows(manifest, {"shards": [good]}, destination), [])


if __name__ == "__main__":
    unittest.main()
