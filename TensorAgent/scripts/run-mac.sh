#!/usr/bin/env bash
# Launches the Mac app built by build-mac.sh from this terminal, so that its stdout -- the
# engine probe, the self-test, the request log and, in a Debug build, the 'entry URL' line
# with the launch token -- goes to a log that verify-sim.sh and chat-e2e.py can read.
# (Opening the app from the Finder works too; it just prints nowhere.)
#
# Usage: run-mac.sh [log]      default log: artifacts/tensoragent-mac/app.log
#
# Env (read by the app itself, so they pass straight through):
#   CONFIGURATION             Debug (default) | Release - must match build-mac.sh
#   TENSORAGENT_USE_MODEL     Debug builds: catalog id to load at launch, as tapping
#                             "Use" on the model list would
#   TENSORAGENT_DEMO_PROMPT   Debug builds: typed into the composer and sent once the
#                             page has loaded
#   TENSORAGENT_SPEC_BENCH=1  any build: run the in-process decode benchmark after the
#                             model loads (with TENSORAGENT_USE_MODEL); results in the log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIGURATION="${CONFIGURATION:-Debug}"
APP="$(ls -d "${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/bin/${CONFIGURATION}/net10.0-maccatalyst/"*/TensorAgent.app 2>/dev/null | head -1)"
if [[ -z "${APP}" || ! -d "${APP}" ]]; then
    echo "No ${CONFIGURATION} Mac app; run TensorAgent/scripts/build-mac.sh first." >&2
    exit 1
fi

LOG="${1:-${REPO_ROOT}/artifacts/tensoragent-mac/app.log}"
mkdir -p "$(dirname "${LOG}")"
echo "==> ${APP}"
echo "    stdout -> ${LOG} (Ctrl-C quits the app)"
"${APP}/Contents/MacOS/TensorAgent.Maui" 2>&1 | tee "${LOG}"
