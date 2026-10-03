"""Source/reference ordering in the CLI and HTTP masked-edit benchmark (no GPU)."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw


MODULE = Path(__file__).resolve().parents[1] / "qwen-image21-mask-bench.py"
SPEC = importlib.util.spec_from_file_location("qwen_image21_mask_bench", MODULE)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class QwenImage21MaskBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.png"
        self.mask = self.root / "mask.png"
        self.references = [self.root / "reference-wide.png", self.root / "reference-tall.png"]
        source = Image.new("RGBA", (32, 32), (20, 30, 40, 80))
        source.save(self.source)
        mask = Image.new("L", source.size)
        ImageDraw.Draw(mask).rectangle((8, 8, 23, 23), fill=255)
        mask.save(self.mask)
        for path, size in zip(self.references, [(64, 32), (32, 64)]):
            Image.new("RGB", size, "white").save(path)
        source.putpixel((12, 12), (10, 80, 200, 200))
        encoded = io.BytesIO()
        source.save(encoded, format="PNG")
        self.result = encoded.getvalue()

    def args(self, references=(), server=False):
        args = [str(MODULE), "--image", str(self.source), "--mask", str(self.mask),
                "--width", "32", "--height", "32", "--crop", "off", "--steps", "1",
                "--out", str(self.root / "evidence")]
        for reference in references:
            args.extend(["--reference", str(reference)])
        if server:
            args.extend(["--server-url", "http://local.test"])
        return args

    def run_main(self, args):
        with patch("sys.argv", args), patch.object(BENCH, "revision", return_value={}), redirect_stdout(io.StringIO()):
            self.assertEqual(0, BENCH.main())
        report = json.loads((self.root / "evidence/report.json").read_text())
        pixels = report["runs"][0]["pixels"]
        self.assertEqual((32, 32), (pixels["width"], pixels["height"]))
        self.assertEqual(0, pixels["changed_protected_pixels"])
        self.assertEqual(1, pixels["changed_editable_pixels"])
        return report

    def cli(self, references):
        def run(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(self.result)
            return SimpleNamespace(returncode=0)
        with patch.object(BENCH.subprocess, "run", side_effect=run) as launch:
            report = self.run_main(self.args(references))
        command = launch.call_args.args[0]
        images = [command[index + 1] for index, word in enumerate(command) if word == "--image"]
        self.assertEqual([str(self.source), *map(str, references)], images)
        self.assertEqual(str(self.mask), command[command.index("--mask") + 1])
        self.assertEqual([str(path) for path in references], report["parameters"]["reference"])
        self.assertEqual([str(path) for path in references], [r["path"] for r in report["references"]])

    def test_cli_keeps_source_before_repeatable_references(self):
        self.cli(self.references)

    def test_default_cli_retains_single_source(self):
        self.cli([])

    def test_api_keeps_source_reference_order_and_mask_out_of_references(self):
        uploads, payloads = [], []

        class Stream:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_lines(self):
                yield b'data: {"done":true,"url":"/uploads/result.png"}'

        def post(url, **kwargs):
            if url.endswith("/api/upload"):
                name = kwargs["files"]["file"][0]
                uploads.append(name)
                return SimpleNamespace(raise_for_status=lambda: None,
                                       json=lambda: {"file": "uploaded-" + name})
            self.assertTrue(url.endswith("/api/image-edit/stream"))
            payloads.append(kwargs["json"])
            return Stream()

        with patch.object(BENCH.requests, "post", side_effect=post), patch.object(
                BENCH.requests, "get", return_value=SimpleNamespace(
                    raise_for_status=lambda: None, content=self.result)):
            self.run_main(self.args(self.references, server=True))
        self.assertEqual([self.source.name, *[r.name for r in self.references], self.mask.name], uploads)
        self.assertEqual(["uploaded-" + name for name in uploads[:-1]], payloads[0]["imagePaths"])
        self.assertEqual("uploaded-mask.png", payloads[0]["maskPath"])
        self.assertEqual("grayscale", payloads[0]["maskMode"])


if __name__ == "__main__":
    unittest.main()
