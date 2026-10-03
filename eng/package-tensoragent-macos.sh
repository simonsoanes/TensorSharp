#!/usr/bin/env bash
# Package a published, ad-hoc signed Mac Catalyst app without changing its signature.
# Usage: bash eng/package-tensoragent-macos.sh APP VERSION [OUTPUT_DIRECTORY]
set -euo pipefail

APP="${1:?Supply the path to TensorAgent.app}"
VERSION="${2:?Supply the release version without a leading v}"
OUTPUT="${3:-artifacts}"
[[ "$(uname -s)" == Darwin ]] || { echo "macOS packaging requires macOS." >&2; exit 1; }
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z]+([.-][0-9A-Za-z]+)*)?(\+[0-9A-Za-z]+([.-][0-9A-Za-z]+)*)?$ ]] || {
    echo "Expected a three-part release version (for example 2026.10.03 or 2.8.6)." >&2; exit 1;
}
[[ -d "$APP" && "$(basename "$APP")" == TensorAgent.app ]] || { echo "TensorAgent.app not found: $APP" >&2; exit 1; }
for file in Contents/Info.plist Contents/MonoBundle/libGgmlOps.dylib Contents/Resources/webui/index.html; do
    [[ -s "$APP/$file" ]] || { echo "Missing app payload: $file" >&2; exit 1; }
done
[[ -d "$APP/Contents/Resources/skills" ]] || { echo "Missing bundled skills." >&2; exit 1; }
EXECUTABLE="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$APP/Contents/Info.plist")"
[[ -x "$APP/Contents/MacOS/$EXECUTABLE" ]] || { echo "Missing app executable." >&2; exit 1; }
codesign --verify --deep --strict "$APP"
lipo -verify_arch arm64 "$APP/Contents/MacOS/$EXECUTABLE"
lipo -verify_arch arm64 "$APP/Contents/MonoBundle/libGgmlOps.dylib"
python3 - "$APP" "$VERSION" <<'PY'
import plistlib
from pathlib import Path
import re
import subprocess
import sys

app = Path(sys.argv[1])
info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
expected = ".".join(str(int(part)) for part in re.split(r"[-+]", sys.argv[2])[0].split("."))
if info.get("CFBundleShortVersionString") != expected:
    raise SystemExit(f"App version {info.get('CFBundleShortVersionString')} does not match release {expected}")
def os_version(value):
    parts = list(map(int, value.split(".")))
    return tuple((parts + [0, 0, 0])[:3])

if os_version(info["LSMinimumSystemVersion"]) > (14, 0, 0):
    raise SystemExit("The published app requires newer than the documented macOS 14 floor")
library = app / "Contents/MonoBundle/libGgmlOps.dylib"
build = subprocess.check_output(["xcrun", "vtool", "-show-build", str(library)], text=True)
minimum = re.search(r"\bminos\s+([0-9.]+)", build)
if not minimum or os_version(minimum[1]) > (14, 0, 0):
    raise SystemExit("GGML native library must be built with MACOSX_DEPLOYMENT_TARGET=14.0")
dependencies = subprocess.check_output(["otool", "-L", str(library)], text=True).splitlines()[2:]
for line in dependencies:
    name = line.strip().split(" ")[0]
    if not name.startswith(("/System/Library/", "/usr/lib/")):
        raise SystemExit(f"Unbundled GGML native dependency: {name}")
PY

mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
STEM="tensoragent-desktop-$VERSION-osx-arm64"
WORK="$(mktemp -d "$OUTPUT/.tensoragent-macos.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/disk" "$WORK/root/Applications"
ditto "$APP" "$WORK/disk/TensorAgent.app"
ln -s /Applications "$WORK/disk/Applications"
cat > "$WORK/disk/INSTALL.txt" <<'EOF'
TensorAgent Desktop for Apple Silicon (macOS 14 or later)

DMG: drag TensorAgent.app to Applications, eject the disk, then open TensorAgent.
ZIP: extract, move TensorAgent.app to Applications, then open TensorAgent.
PKG: open the installer and follow the prompts to install into /Applications.
Quit TensorAgent before updating. Models and chats are stored outside the app.

These community packages are ad-hoc signed, not Developer ID signed/notarized.
After checking the official release and SHA-256, attempt to open the app, then
use System Settings > Privacy & Security > Open Anyway if macOS blocks it.

Open menu > Models, choose a model that fits your RAM, Download, then Use.
Model files are separate multi-GB downloads; .NET and Metal/CPU are bundled.
Python/Node.js are optional system tools for code and browser skills.

Download, installation, first chat and troubleshooting guide:
https://tensorsharp.ai/tensoragent.html
https://github.com/zhongkaifu/TensorSharp/releases
EOF

# ditto preserves executable permissions, symlinks and the bundle signature.
ditto -c -k --sequesterRsrc --keepParent "$WORK/disk/TensorAgent.app" "$OUTPUT/$STEM.zip"
hdiutil create -volname TensorAgent -srcfolder "$WORK/disk" -format UDZO -ov "$OUTPUT/$STEM.dmg"
ditto "$APP" "$WORK/root/Applications/TensorAgent.app"
# Explicitly disable bundle relocation: Installer must update /Applications,
# rather than a developer's bin/ tree or a copy mounted on the DMG.
pkgbuild --analyze --root "$WORK/root" "$WORK/components.plist"
python3 - "$WORK/components.plist" <<'PY'
import plistlib
from pathlib import Path
import sys

path = Path(sys.argv[1])
components = plistlib.loads(path.read_bytes())
if not any(component.get("RootRelativeBundlePath") == "Applications/TensorAgent.app" for component in components):
    raise SystemExit("pkgbuild did not find the TensorAgent application bundle")
for component in components:
    # Newer pkgbuild versions omit this default-valued key from --analyze.
    component["BundleIsRelocatable"] = False
path.write_bytes(plistlib.dumps(components))
PY
PKG_VERSION="${VERSION%%[-+]*}"
pkgbuild --root "$WORK/root" --component-plist "$WORK/components.plist" \
    --identifier ai.tensorsharp.tensoragent.desktop --version "$PKG_VERSION" \
    --install-location / "$WORK/component.pkg"
cat > "$WORK/requirements.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>os</key><array><string>14.0</string></array>
<key>arch</key><array><string>arm64</string></array>
</dict></plist>
EOF
productbuild --product "$WORK/requirements.plist" --package "$WORK/component.pkg" "$OUTPUT/$STEM.pkg"

# Validate the actual archives, including the signature after ZIP extraction.
mkdir -p "$WORK/extracted"
ditto -x -k "$OUTPUT/$STEM.zip" "$WORK/extracted"
codesign --verify --deep --strict "$WORK/extracted/TensorAgent.app"
hdiutil verify "$OUTPUT/$STEM.dmg"
pkgutil --expand "$OUTPUT/$STEM.pkg" "$WORK/pkg-expanded"
test -s "$WORK/pkg-expanded/Distribution"
(
    cd "$OUTPUT"
    shasum -a 256 "$STEM.dmg" "$STEM.pkg" "$STEM.zip" > "SHA256SUMS-$STEM.txt"
    shasum -a 256 -c "SHA256SUMS-$STEM.txt"
)
echo "Packaged TensorAgent Desktop $VERSION for osx-arm64 in $OUTPUT"
