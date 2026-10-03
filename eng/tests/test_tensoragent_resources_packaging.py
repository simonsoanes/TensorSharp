"""Exercise the SDK's content publishing without compiling MAUI or native code."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET


REPOSITORY = Path(__file__).resolve().parents[2]
APP_PROJECT = REPOSITORY / "TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj"


class DesktopResourcePublishingTests(unittest.TestCase):
    def test_windows_resources_publish_with_inherited_x64_platform(self):
        # The release runner's MSVC environment supplies Platform=x64. Its build
        # output is therefore bin/x64/Release/..., while -o chooses a separate
        # publish directory. Ship the SDK's published content, not a guessed bin
        # directory. Keep the real app's item conditions, globs, links and copy
        # metadata, but omit its MAUI/native compilation and project references.
        dotnet = shutil.which("dotnet")
        self.assertIsNotNone(dotnet, "The .NET SDK is required for resource publishing validation")
        artifact_root = REPOSITORY / "artifacts"
        artifact_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="desktop-resource-fixture-", dir=artifact_root) as temporary:
            fixture = Path(temporary)
            project_directory = fixture / "TensorAgent/src/TensorAgent.Maui"
            project_directory.mkdir(parents=True)
            project = ET.Element("Project", Sdk="Microsoft.NET.Sdk")
            properties = ET.SubElement(project, "PropertyGroup")
            for name, value in {
                "TargetFramework": "net10.0",
                "TensorAgentPlatform": "windows",
                "EnableDefaultItems": "false",
                "UseAppHost": "false",
                "SelfContained": "false",
            }.items():
                ET.SubElement(properties, name).text = value

            for source_group in ET.parse(APP_PROJECT).getroot().findall("ItemGroup"):
                contents = source_group.findall("Content")
                if contents:
                    group = ET.SubElement(project, "ItemGroup", source_group.attrib)
                    for content in contents:
                        group.append(copy.deepcopy(content))
            ET.SubElement(project, "Target", {
                "Name": "PublishFixtureResources",
                "DependsOnTargets": "_ComputeCopyToPublishDirectoryItems;"
                    "_CopyResolvedFilesToPublishPreserveNewest;"
                    "_CopyResolvedFilesToPublishAlways;"
                    "_CopyResolvedFilesToPublishIfDifferent",
            })
            project_path = project_directory / "ResourceFixture.csproj"
            ET.ElementTree(project).write(project_path, encoding="utf-8", xml_declaration=True)

            expected = {
                "webui/index.html": b"<html>current page</html>\n",
                "webui/assets/nested/a&b.js": b"export const revision = 3;\n",
                "skills/example/SKILL.md": b"# Example\n",
                "skills/example/.metadata": b"hidden skill metadata\n",
                "skills/example/assets/nested/payload.bin": bytes(range(256)),
                "skills/playwright/SKILL.md": b"# Desktop browser skill\n",
                "skills/web-artifacts-builder/SKILL.md": b"# Desktop artifact skill\n",
                "skills/web-artifacts-builder/scripts/bundle-artifact.sh": b"#!/bin/sh\nexit 0\n",
            }
            sources = {}
            for relative, payload in expected.items():
                if relative.startswith("webui/"):
                    source = project_directory / "wwwroot" / relative.removeprefix("webui/")
                else:
                    source = fixture / "TensorAgent" / relative
                sources[source] = payload
            for relative in (
                "verdicts.json",
                "example/__pycache__/cached.pyc",
                "example/node_modules/package/index.js",
                "example/assets/nested/__pycache__/cached.pyc",
                "example/assets/nested/node_modules/package/index.js",
            ):
                sources[fixture / "TensorAgent/skills" / relative] = b"must not ship\n"
            for source, payload in sources.items():
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(payload)

            publish_directory = fixture / "requested-publish-directory"
            metadata_path = fixture / "publish-metadata.json"
            command = [
                dotnet, "msbuild", str(project_path), "-nologo", "-verbosity:quiet",
                "-p:ImportDirectoryBuildProps=false", "-p:ImportDirectoryBuildTargets=false",
                "-p:Configuration=Release", "-p:Platform=x64", "-p:RuntimeIdentifier=win-x64",
                f"-p:PublishDir={publish_directory.as_posix()}/", "-t:PublishFixtureResources",
                "-getProperty:Platform,OutputPath", "-getItem:ResolvedFileToPublish",
                f"-getResultOutputFile:{metadata_path}",
            ]
            result = subprocess.run(command, text=True, encoding="utf-8", errors="replace",
                                    capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # Keep structured results separate from first-use .NET banners and
            # other console diagnostics, including on fresh Windows runners.
            self.assertTrue(metadata_path.is_file(), result.stdout + result.stderr)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(metadata["Properties"]["Platform"], "x64")
            output_path = metadata["Properties"]["OutputPath"].replace("\\", "/")
            self.assertEqual(output_path, "bin/x64/Release/net10.0/win-x64/")
            self.assertEqual(
                {item["RelativePath"].replace("\\", "/")
                 for item in metadata["Items"]["ResolvedFileToPublish"]},
                set(expected),
            )
            published = {
                path.relative_to(publish_directory).as_posix(): path.read_bytes()
                for path in publish_directory.rglob("*") if path.is_file()
            }
            self.assertEqual(published, expected)
            self.assertFalse((project_directory / "bin").exists(), "The fixture must publish without a build")


if __name__ == "__main__":
    unittest.main()
