#!/usr/bin/env bash
# Installs and launches the simulator build of TensorAgent on an iOS simulator
# and streams the app's stdout (the loopback URL, the engine probe, request log)
# to this terminal. Build first with build-sim.sh.
#
# Env:
#   CONFIGURATION   Debug (default) | Release - must match build-sim.sh
#   SIM_UDID        simulator to use (default: iPhone 17 Pro, iOS 26.5 on this Mac)
#   SIM_NAME        used to look the UDID up when SIM_UDID is unset
#   TENSORAGENT_USE_MODEL     Debug builds only: catalog id to load at launch, as
#                   tapping "Use" on the model list would.
#   TENSORAGENT_UI_CHECK=1    Debug builds only: run the composer's gesture checks in
#                   the WebView and log one 'uicheck' line per assertion.
#   TENSORAGENT_SHARE_CHECK=1 Debug builds only: write a page + image into the App
#                   Group inbox, import it, and verify the real WebView applied and
#                   retained it without sending, then explicitly discarded both the
#                   durable envelope and staged attachment.
#   TENSORAGENT_DEMO_PROMPT   Debug builds only: once the Web UI has loaded the app
#                   types this prompt into the composer and sends it, so the
#                   canned /api/chat stream renders on screen without anyone
#                   tapping the simulator (simctl cannot type into a WebView).
#   TENSORAGENT_BACKGROUND_CHECK=1  Debug builds only: ask the loaded model for a long
#                   answer and report, one 'bgcheck' line at a time, what becomes of it
#                   when the app is sent to the background and brought back. Pair it
#                   with verify-background.sh, which does the sending and the reading.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIGURATION="${CONFIGURATION:-Debug}"
BUNDLE_ID="ai.tensorsharp.tensoragent"
# MAUI names the bundle after the assembly, not the ApplicationTitle.
APP="${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/bin/${CONFIGURATION}/net10.0-ios/iossimulator-arm64/TensorAgent.Maui.app"

if [[ ! -d "${APP}" ]]; then
    echo "No bundle at ${APP}; run TensorAgent/scripts/build-sim.sh first." >&2
    exit 1
fi

SIM_NAME="${SIM_NAME:-iPhone 17 Pro}"
if [[ -z "${SIM_UDID:-}" ]]; then
    SIM_UDID="$(xcrun simctl list devices available | grep -F "${SIM_NAME} (" | head -1 | sed -E 's/.*\(([0-9A-F-]{36})\).*/\1/')"
fi
if [[ -z "${SIM_UDID}" ]]; then
    echo "No simulator named '${SIM_NAME}' found; set SIM_UDID or SIM_NAME." >&2
    exit 1
fi

# boot is a no-op error when already booted; ignore that one case.
xcrun simctl boot "${SIM_UDID}" 2>/dev/null || true
# The window is for a person watching; the device boots and runs without it. Xcode 27 as
# installed here has no Simulator.app for `open` to find, and that must not end the run.
open -a Simulator 2>/dev/null || echo "note: no Simulator app to open; the simulator runs headless"
xcrun simctl bootstatus "${SIM_UDID}" -b >/dev/null

echo "==> Installing ${APP} on ${SIM_UDID}"
xcrun simctl install "${SIM_UDID}" "${APP}"

echo "==> Launching ${BUNDLE_ID} (stdout follows; the 'entry URL' line carries the token to curl /api from this Mac)"
# simctl forwards SIMCTL_CHILD_* variables to the app with the prefix stripped.
if [[ -n "${TENSORAGENT_DEMO_PROMPT:-}" ]]; then
    export SIMCTL_CHILD_TENSORAGENT_DEMO_PROMPT="${TENSORAGENT_DEMO_PROMPT}"
fi
# Debug builds only: open a page other than the chat, so a screenshot can be taken of
# the model list or the settings. simctl cannot tap, so there is no other way in.
if [[ -n "${TENSORAGENT_START_PAGE:-}" ]]; then
    export SIMCTL_CHILD_TENSORAGENT_START_PAGE="${TENSORAGENT_START_PAGE}"
fi
# Debug builds only: load a catalog entry at launch, as tapping "Use" would. Choosing
# a model is a native tap and simctl cannot tap, so without this every scripted run is
# a run with no model -- which is every path except the one the app exists for.
if [[ -n "${TENSORAGENT_USE_MODEL:-}" ]]; then
    export SIMCTL_CHILD_TENSORAGENT_USE_MODEL="${TENSORAGENT_USE_MODEL}"
fi
# Debug builds only: drive the composer's own gestures in the real WebView and print
# one 'uicheck' line per assertion. verify-sim.sh asserts on them.
if [[ -n "${TENSORAGENT_UI_CHECK:-}" ]]; then
    export SIMCTL_CHILD_TENSORAGENT_UI_CHECK="${TENSORAGENT_UI_CHECK}"
fi
if [[ -n "${TENSORAGENT_SHARE_CHECK:-}" ]]; then
    export SIMCTL_CHILD_TENSORAGENT_SHARE_CHECK="${TENSORAGENT_SHARE_CHECK}"
fi
# Debug builds only: a generation carried across a background switch (see the header
# and verify-background.sh), and the start-a-download probe, which stops after
# TENSORAGENT_DOWNLOAD_SECONDS. Both are mostly for a physical device, where leaving
# the app is the only way to exercise the background-task assertion and the GPU rule.
for VAR in TENSORAGENT_BACKGROUND_CHECK TENSORAGENT_BACKGROUND_PROMPT TENSORAGENT_BACKGROUND_TOKENS \
           TENSORAGENT_DOWNLOAD TENSORAGENT_DOWNLOAD_SECONDS; do
    if [[ -n "${!VAR:-}" ]]; then
        export "SIMCTL_CHILD_${VAR}=${!VAR}"
    fi
done
exec xcrun simctl launch --console --terminate-running-process "${SIM_UDID}" "${BUNDLE_ID}"
