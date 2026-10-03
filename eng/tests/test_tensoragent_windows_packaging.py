"""Contract checks for desktop MSI authoring; no Windows installation is claimed."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ENG = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tensoragent_wix", ENG / "generate-tensoragent-wix.py")
wix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wix)
NS = {"w": wix.WIX_NAMESPACE}


class WindowsDesktopPackagingTests(unittest.TestCase):
    def setUp(self):
        artifact_root = ENG.parent / "artifacts"
        artifact_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="windows-packaging-fixture-", dir=artifact_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.publish = self.root / "publish"
        for name in wix.REQUIRED_FILES + ("skills/example/SKILL.md", "webui/assets/a&b.js", ".hidden-payload"):
            target = self.publish / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture:" + name.encode("utf-8"))
        self.write_runtime({"includedFrameworks": [{"name": "Microsoft.NETCore.App", "version": "10.0.0"}]})

    def write_runtime(self, options):
        (self.publish / "TensorAgent.Maui.runtimeconfig.json").write_text(json.dumps({"runtimeOptions": options}), encoding="utf-8")

    def test_version_triplets_and_calendar_mapping(self):
        for release, numeric in (("2.8.6", "2.8.6"), ("2026.10.03", "26.10.3"),
                                 ("2026.10.03-preview.1+build.2", "26.10.3"), ("2255.255.65535", "255.255.65535")):
            with self.subTest(release=release):
                self.assertEqual(wix.msi_version(release), numeric)

    def test_reject_invalid_or_unrepresentable_versions(self):
        for version in ("v2.8.6", "2.8", "2.8.6.1", "2.8.6-", "2.8.6..preview", "2/8/6",
                        "2.256.1", "2.0.65536", "256.0.0", "2256.1.1", "2.8.6\n", "2.8.6+$oops"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                wix.msi_version(version)

    def test_payload_manifest_preserves_hidden_nested_and_escaped_files(self):
        files = wix.payload_manifest(self.publish, "win-x64-cpu")
        by_path = {item["path"]: item for item in files}
        for name in (".hidden-payload", "webui/assets/a&b.js", "skills/example/SKILL.md"):
            self.assertIn(name, by_path)
            self.assertEqual(by_path[name]["sha256"], hashlib.sha256((self.publish / name).read_bytes()).hexdigest())

    def test_missing_runtime_native_webui_or_crt_fails(self):
        for name in ("coreclr.dll", "GgmlOps.dll", "Microsoft.UI.Xaml.dll", "webui/index.html", "vcruntime140.dll"):
            target = self.publish / name
            content = target.read_bytes()
            target.unlink()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "missing/empty"):
                wix.payload_manifest(self.publish, "win-x64-cpu")
            target.write_bytes(content)

    def test_missing_skills_fails(self):
        (self.publish / "skills/example/SKILL.md").unlink()
        with self.assertRaisesRegex(ValueError, "no bundled skills"):
            wix.payload_manifest(self.publish, "win-x64-cpu")

    def test_framework_dependent_runtime_fails(self):
        self.write_runtime({"framework": {"name": "Microsoft.NETCore.App", "version": "10.0.0"}})
        with self.assertRaisesRegex(ValueError, "framework-dependent"):
            wix.payload_manifest(self.publish, "win-x64-cpu")

    def test_cuda_must_bundle_runtime_and_ptx(self):
        with self.assertRaisesRegex(ValueError, "cudart64_12.dll"):
            wix.payload_manifest(self.publish, "win-x64-cuda")
        for name in wix.CUDA_FILES:
            target = self.publish / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"cuda fixture")
        self.assertTrue(wix.payload_manifest(self.publish, "win-x64-cuda"))
        (self.publish / "cuda_kernels/tensorsharp_kernels.ptx").unlink()
        with self.assertRaisesRegex(ValueError, "tensorsharp_kernels.ptx"):
            wix.payload_manifest(self.publish, "win-x64-cuda")

    def test_symbolic_links_are_rejected(self):
        (self.publish / "linked.dll").symlink_to(self.publish / "GgmlOps.dll")
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            wix.payload_manifest(self.publish, "win-x64-cpu")

    def test_windows_invalid_names_are_rejected(self):
        for name in ("bad:name.txt", "CON.txt", "trailing. "):
            target = self.publish / name
            target.write_bytes(b"fixture")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Invalid Windows"):
                wix.payload_manifest(self.publish, "win-x64-cpu")
            target.unlink()

    def test_both_variants_share_upgrade_family_and_replace_same_version(self):
        files = wix.payload_manifest(self.publish, "win-x64-cpu")
        for variant in wix.VARIANTS:
            tree = wix.generate_wix(self.publish, "2026.10.03-preview.1", variant, files)
            # Serialize and parse to check XML escaping, including ampersands.
            root = ET.fromstring(ET.tostring(tree.getroot()))
            package = root.find("w:Package", NS)
            self.assertEqual(package.get("UpgradeCode"), wix.UPGRADE_CODE)
            self.assertEqual(package.get("Scope"), "perUser")
            self.assertEqual(package.get("Version"), "26.10.3")
            upgrade = package.find("w:MajorUpgrade", NS)
            self.assertEqual(upgrade.get("AllowSameVersionUpgrades"), "yes")
            self.assertEqual(upgrade.get("Schedule"), "afterInstallInitialize")
            installed = root.findall(".//w:File", NS)
            self.assertEqual(len(installed), len(files))
            self.assertEqual({item.get("Source") for item in installed}, {str(self.publish / item["path"]) for item in files})
            self.assertEqual(package.find("w:Property[@Id='ARPHELPLINK']", NS).get("Value"), "https://tensorsharp.ai/tensoragent.html")
            shortcut = root.find(".//w:Shortcut", NS)
            self.assertEqual(shortcut.get("Target"), "[INSTALLFOLDER]TensorAgent.Maui.exe")
            self.assertEqual(shortcut.get("WorkingDirectory"), "INSTALLFOLDER")
            self.assertEqual(len(root.findall(".//w:Component", NS)), len(root.findall(".//w:ComponentRef", NS)))
            self.assertTrue(all(item.find("w:RegistryValue", NS).get("Root") == "HKCU" for item in root.findall(".//w:Component", NS)))

    def test_component_ids_and_guids_are_stable_across_release_and_variant(self):
        files = wix.payload_manifest(self.publish, "win-x64-cpu")
        first = wix.generate_wix(self.publish, "2.8.6", "win-x64-cpu", files)
        second = wix.generate_wix(self.publish, "2.8.7", "win-x64-cuda", files)
        components = lambda tree: {(item.get("Id"), item.get("Guid")) for item in tree.findall(".//w:Component", NS)}
        self.assertEqual(components(first), components(second))

    def test_cli_creates_manifest_and_refuses_generated_files_in_payload(self):
        command = [sys.executable, str(ENG / "generate-tensoragent-wix.py"), "--publish-directory", str(self.publish),
                   "--version", "2026.10.03", "--variant", "win-x64-cpu", "--manifest", str(self.root / "payload.json")]
        result = subprocess.run(command + ["--output", str(self.root / "TensorAgent.wxs")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root / "payload.json").read_text())["msi_version"], "26.10.3")
        result = subprocess.run(command + ["--output", str(self.publish / "TensorAgent.wxs")], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the publish directory", result.stderr)


if __name__ == "__main__":
    unittest.main()
