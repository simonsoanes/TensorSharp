#!/usr/bin/env bash
# Builds the TensorAgent Mac app: TensorAgent.Maui for Mac Catalyst.
#
# The Mac app is the phone app's code under Mac Catalyst with the desktop engine: the
# same libGgmlOps.dylib the desktop hosts load, which the build brings up to date before
# the app (TensorAgentBuildDesktopEngine in the csproj), and real python3, node and npm
# processes confined by Seatbelt instead of the phone's embedded interpreters.
#
# Needs the user-local ~/.dotnet SDK with the maui-maccatalyst workload (see README.md),
# and CMake with the Xcode command-line tools for the engine library.
#
# Env:
#   CONFIGURATION   Debug (default) | Release
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIGURATION="${CONFIGURATION:-Debug}"

export DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}"
export PATH="${DOTNET_ROOT}:${PATH}"
# The Mac app never uses MLX. The GGML native build is NOT skipped: the app ships the
# desktop engine library, and a library older than the native sources beside it fails
# at its first missing export.
export TENSORSHARP_MLX_NATIVE_SKIP=true

echo "==> dotnet $(dotnet --version): building TensorAgent.Maui (${CONFIGURATION}, Mac Catalyst)"
# TensorSharpAppleTargets=true must be on the command line for the same reason as in
# build-sim.sh: restore resolves TensorSharp.Models' target frameworks before a
# ProjectReference's AdditionalProperties apply. Single-node for the same reason too.
dotnet build "${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj" \
    -f net10.0-maccatalyst \
    -c "${CONFIGURATION}" \
    -p:TensorSharpAppleTargets=true \
    -m:1 \
    -nologo

APP="$(ls -d "${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/bin/${CONFIGURATION}/net10.0-maccatalyst/"*/TensorAgent.app 2>/dev/null | head -1)"
if [[ -z "${APP}" || ! -d "${APP}" ]]; then
    echo "Build finished but no TensorAgent.app is under bin/${CONFIGURATION}/net10.0-maccatalyst/" >&2
    exit 1
fi
echo "App: ${APP}"
