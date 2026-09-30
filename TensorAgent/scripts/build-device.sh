#!/usr/bin/env bash
# Builds TensorAgent.Maui for a physical iPhone (ios-arm64).
#
# Same two traps as build-sim.sh (user-local SDK carries the maui-ios workload;
# the xcframework is gitignored and built on demand), plus the one a device adds:
# a device build must be SIGNED, so a codesigning identity and a provisioning
# profile for the app id have to exist. Simulator builds sign with "-" and need
# neither.
#
# Env:
#   CONFIGURATION   Debug (default) | Release
#   CODESIGN_KEY    signing identity, e.g. "Apple Development: you@example.com (XXXXXXXXXX)"
#   CODESIGN_PROVISION  profile NAME or UUID for ai.tensorsharp.tensoragent
#   CODESIGN_SHARE_PROVISION  independent profile NAME or UUID for
#                     ai.tensorsharp.tensoragent.share (used only when the extension
#                     is enabled; never reuse CODESIGN_PROVISION)
#   TENSORAGENT_SHARE_EXTENSION  true (default) | false. A false build is the
#                     intentionally app-only bundle and needs no extension profile or
#                     App Group entitlement.
#   SKIP_SIGNING=1  build without signing (compile/link/AOT check only; NOT installable)
#   CLEAN=1         run a targeted clean for this configuration/RID first; the deploy
#                     script uses it so stale nested extension code cannot survive
#   NO_INCREMENTAL=1  force a non-incremental build (used by deploy-device.sh so
#                     a stale unsigned bundle can never be reused)
#   TENSORAGENT_REBUILD_XCFRAMEWORK=1  rebuild the device + simulator native
#                     slices even when the xcframework already exists
#   TENSORAGENT_DOTNET_ARGS  additional arguments for `dotnet build`
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIGURATION="${CONFIGURATION:-Debug}"
TENSORAGENT_SHARE_EXTENSION="${TENSORAGENT_SHARE_EXTENSION:-true}"

case "${TENSORAGENT_SHARE_EXTENSION}" in
    1|true|TRUE|yes|YES) TENSORAGENT_SHARE_EXTENSION=true ;;
    0|false|FALSE|no|NO) TENSORAGENT_SHARE_EXTENSION=false ;;
    *)
        echo "TENSORAGENT_SHARE_EXTENSION must be true or false." >&2
        exit 1
        ;;
esac

export DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}"
export PATH="${DOTNET_ROOT}:${PATH}"
export TENSORSHARP_GGML_NO_UPDATE=1
export TENSORSHARP_GGML_NATIVE_SKIP=true
export TENSORSHARP_MLX_NATIVE_SKIP=true

XCFRAMEWORK="${REPO_ROOT}/TensorSharp.GGML.Native/build-ios/GgmlOps.xcframework"
if [[ ! -d "${XCFRAMEWORK}/ios-arm64" || "${TENSORAGENT_REBUILD_XCFRAMEWORK:-0}" == "1" ]]; then
    if [[ ! -d "${REPO_ROOT}/ExternalProjects/ggml" ]]; then
        echo "ExternalProjects/ggml is missing; run eng/fetch-ggml.sh once before building the xcframework." >&2
        exit 1
    fi
    echo "==> Building ${XCFRAMEWORK}"
    TENSORSHARP_IOS_SLICES="device sim" \
        bash "${REPO_ROOT}/TensorSharp.GGML.Native/build-ios.sh"
fi

ARGS=(
  "${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj"
  -f net10.0-ios
  -p:RuntimeIdentifier=ios-arm64
  -c "${CONFIGURATION}"
  -p:TensorSharpAppleTargets=true
  -p:TensorAgentShareExtension="${TENSORAGENT_SHARE_EXTENSION}"
  # AdvUtils, TensorSharp.Runtime and a few siblings write one shared bin/ even
  # though this graph reaches them under more than one property set. Parallel nodes
  # can race on the same deps.json; a deterministic device artifact matters more than
  # shaving a few seconds from deployment.
  -m:1
  -nologo
)
if [[ "${SKIP_SIGNING:-0}" == "1" ]]; then
    # Enough to prove the slice compiles, links against the device xcframework and
    # passes AOT. The bundle it leaves cannot be installed on a phone.
    ARGS+=( -p:_RequireCodeSigning=false -p:CodesignKey= -p:EnableCodeSigning=false )
else
    [[ -n "${CODESIGN_KEY:-}" ]] || { echo "CODESIGN_KEY is required (or SKIP_SIGNING=1)" >&2; exit 1; }
    if [[ "${TENSORAGENT_SHARE_EXTENSION}" == "true"
          && -n "${CODESIGN_PROVISION:-}"
          && "${CODESIGN_SHARE_PROVISION:-}" == "${CODESIGN_PROVISION}" ]]; then
        echo "CODESIGN_SHARE_PROVISION must be the extension's own profile, not CODESIGN_PROVISION." >&2
        exit 1
    fi
    ARGS+=( -p:CodesignKey="${CODESIGN_KEY}" )
    [[ -n "${CODESIGN_PROVISION:-}" ]] && ARGS+=( -p:CodesignProvision="${CODESIGN_PROVISION}" )
    if [[ "${TENSORAGENT_SHARE_EXTENSION}" == "true" && -n "${CODESIGN_SHARE_PROVISION:-}" ]]; then
        ARGS+=( -p:TensorAgentShareProvision="${CODESIGN_SHARE_PROVISION}" )
    fi
fi
if [[ "${CLEAN:-0}" == "1" ]]; then
    echo "==> Restoring TensorAgent.Maui (${CONFIGURATION}, ios-arm64) before clean"
    # The iOS SDK keeps one project.assets.json for this project. A simulator
    # build can therefore leave that file with only iossimulator-arm64 in it,
    # while several MAUI clean targets still resolve packages for the requested
    # RID before deleting outputs. Restore the device graph explicitly so a
    # clean device deployment never depends on whichever RID was built last.
    dotnet restore "${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj" \
        -r ios-arm64 \
        -p:Configuration="${CONFIGURATION}" \
        -p:TensorSharpAppleTargets=true \
        -p:TensorAgentShareExtension="${TENSORAGENT_SHARE_EXTENSION}" \
        -m:1 \
        -nologo
    echo "==> Cleaning TensorAgent.Maui (${CONFIGURATION}, ios-arm64)"
    # Several referenced projects share an output directory, so clean them on one
    # MSBuild node for the same reason the documented build uses -m:1.
    dotnet clean "${ARGS[@]}"
fi
[[ "${NO_INCREMENTAL:-0}" == "1" ]] && ARGS+=( --no-incremental )
if [[ -n "${TENSORAGENT_DOTNET_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    ARGS+=( ${TENSORAGENT_DOTNET_ARGS} )
fi

echo "==> dotnet $(dotnet --version): TensorAgent.Maui (${CONFIGURATION}, ios-arm64; share extension ${TENSORAGENT_SHARE_EXTENSION})"
dotnet build "${ARGS[@]}"

APP="${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/bin/${CONFIGURATION}/net10.0-ios/ios-arm64/TensorAgent.Maui.app"
[[ -d "${APP}" ]] || { echo "Build finished but the bundle is not at ${APP}" >&2; exit 1; }

# The extension's two startup properties are a physical-device reliability contract,
# not merely build preferences. Validate the linker inputs emitted by the iOS SDK so a
# condition/default change cannot silently put the crashing full-AOT managed-static
# callback path back into an otherwise installable bundle.
if [[ "${TENSORAGENT_SHARE_EXTENSION}" == "true" ]]; then
    SHARE_LINKER_OPTIONS="${REPO_ROOT}/TensorAgent/src/TensorAgent.ShareExtension/obj/macos/${CONFIGURATION}/net10.0-ios/ios-arm64/custom-linker-options.txt"
    [[ -f "${SHARE_LINKER_OPTIONS}" ]] \
        || { echo "Share extension linker options are missing at ${SHARE_LINKER_OPTIONS}" >&2; exit 1; }
    grep -Eq '^[[:space:]]*Interpreter=all[[:space:]]*$' "${SHARE_LINKER_OPTIONS}" \
        || { echo "Share extension was not linked with Interpreter=all" >&2; exit 1; }
    grep -Eq '^[[:space:]]*Registrar=static[[:space:]]*$' "${SHARE_LINKER_OPTIONS}" \
        || { echo "Share extension was not linked with Registrar=static" >&2; exit 1; }
fi
echo "Bundle: ${APP}"
