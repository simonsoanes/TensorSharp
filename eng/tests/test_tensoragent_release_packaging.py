"""Release versions and platform packaging commands must satisfy their contracts."""
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import shlex
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


@unittest.skipUnless(sys.platform == "darwin", "Mac packager uses macOS bundle tools")
class MacDesktopPackagingTests(unittest.TestCase):
    def test_checks_both_binaries_with_xcode26_lipo_argument_order(self):
        artifact_root = ENG.parent / "artifacts"
        artifact_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mac packaging fixture ", dir=artifact_root) as directory:
            root = Path(directory)
            app = root / "TensorAgent.app"
            executable = app / "Contents/MacOS/TensorAgent.Maui"
            library = app / "Contents/MonoBundle/libGgmlOps.dylib"
            for file in (executable, library, app / "Contents/Resources/webui/index.html"):
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"fixture")
            executable.chmod(0o755)
            (app / "Contents/Resources/skills").mkdir()
            (app / "Contents/Info.plist").write_bytes(plistlib.dumps({"CFBundleExecutable": executable.name}))

            commands = root / "commands"
            commands.mkdir()
            log = root / "lipo.jsonl"
            # Reproduce Xcode 26.6's grammar: all arguments after -verify_arch
            # are architecture names. Stop after both calls, before packaging;
            # real signatures and archives are validated separately on macOS.
            shim = root / "strict_lipo.py"
            shim.write_text('''import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if "-verify_arch" not in args:
    sys.exit(64)
command = args.index("-verify_arch")
if command == 0 or args[command + 1:] != ["arm64"]:
    sys.exit(64)
if not all(Path(source).is_file() for source in args[:command]):
    sys.exit(65)
with open(os.environ["LIPO_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(args[:command]) + "\\n")
if any(source.endswith("libGgmlOps.dylib") for source in args[:command]):
    sys.exit(73)
''', encoding="utf-8")
            (commands / "codesign").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            (commands / "lipo").write_text(
                f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(shim))} \"$@\"\n",
                encoding="utf-8")
            for command in commands.iterdir():
                command.chmod(0o755)
            environment = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"], LIPO_LOG=str(log))
            result = subprocess.run(["bash", str(ENG / "package-tensoragent-macos.sh"), str(app), "2026.10.03",
                                     str(root / "packages")], env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 73, result.stderr)
            self.assertEqual([json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()],
                             [[str(executable)], [str(library)]])


if __name__ == "__main__":
    unittest.main()
