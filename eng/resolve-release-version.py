#!/usr/bin/env python3
"""Validate a tag/manual release version and emit safe GitHub Actions outputs."""
import os
import re


def resolve(version: str) -> tuple[str, str]:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?(?:\+[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?", version):
        raise ValueError("Supply a three-part version such as 2.8.6 or 2026.10.03 (without v).")
    parts = [int(p) for p in re.split(r"[-+]", version)[0].split(".")]
    if any(p > 65535 for p in parts):
        raise ValueError("Release version components must fit .NET assembly version fields (0–65535).")
    # Windows Installer limits major/minor to 255 and patch to 65535.
    # Keep calendar tags usable: 2026.10.03 maps to MSI 26.10.3.
    msi_major = parts[0] - 2000 if 2000 <= parts[0] <= 2255 else parts[0]
    if msi_major > 255 or parts[1] > 255:
        raise ValueError("Version cannot be represented by Windows Installer; use semver or a 2000–2255 calendar year.")
    return version, ".".join(map(str, parts))


if __name__ == "__main__":
    version = os.environ.get("RELEASE_VERSION", "")
    if not version and os.environ.get("GITHUB_REF_TYPE") == "tag":
        version = os.environ["GITHUB_REF_NAME"].removeprefix("v")
    version, app_version = resolve(version)
    outputs = f"version={version}\ntag=v{version}\napp_version={app_version}\n"
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(outputs)
    print(outputs, end="")
