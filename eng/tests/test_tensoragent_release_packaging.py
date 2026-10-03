"""Release tags must work in archive names, app metadata and Windows Installer."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ENG = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolver = load("release_version", ENG / "resolve-release-version.py")
wix = load("wix_version", ENG / "generate-tensoragent-wix.py")


class ReleaseVersionTests(unittest.TestCase):
    def test_calendar_tag_preserves_asset_name_and_normalizes_app_metadata(self):
        self.assertEqual(resolver.resolve("2026.10.03"), ("2026.10.03", "2026.10.3"))
        self.assertEqual(wix.msi_version("2026.10.03"), "26.10.3")

    def test_prerelease_and_build_label_stay_in_asset_name(self):
        version = "2.8.6-rc.1+build.2"
        self.assertEqual(resolver.resolve(version), (version, "2.8.6"))
        self.assertEqual(wix.msi_version(version), "2.8.6")

    def test_unrepresentable_installer_versions_fail_before_building(self):
        for version in ("256.1.0", "2026.256.0", "2.8.65536", "2256.1.1"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                resolver.resolve(version)

    def test_manual_input_cannot_inject_actions_outputs_or_paths(self):
        for version in ("2.8.6\ntag=v0.0.0", "../../2.8.6", "2.8.6$(id)", "v2.8.6", "", "2.8"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                resolver.resolve(version)

    def test_tag_dispatch_emits_consistent_actions_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            environment = dict(os.environ, RELEASE_VERSION="", GITHUB_REF_TYPE="tag",
                               GITHUB_REF_NAME="v2026.10.03", GITHUB_OUTPUT=str(output))
            subprocess.run([sys.executable, str(ENG / "resolve-release-version.py")],
                           env=environment, check=True, capture_output=True, text=True)
            self.assertEqual(output.read_text(), "version=2026.10.03\ntag=v2026.10.03\napp_version=2026.10.3\n")


if __name__ == "__main__":
    unittest.main()
