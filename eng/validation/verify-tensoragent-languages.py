#!/usr/bin/env python3
"""Validate TensorAgent language selection in an isolated iOS simulator.

Build a Debug iossimulator-arm64 TensorAgent.Maui.app first, then run from the
repository root:

    python3 eng/validation/verify-tensoragent-languages.py

The tool creates its own iPhone 17 Pro on the available iOS 26.5 runtime. It does
not boot, install on, or change an existing simulator. AppleLanguages launch
arguments exercise NSLocale.PreferredLanguages without changing global system
preferences. Each first-launch case reinstalls the app to obtain fresh settings.
The Debug Settings hook selects the native picker row through its usual change
handler; subsequent launches have no hook and must read the persisted choice.
The English and Simplified Chinese first-launch cases, and the Simplified
Chinese picker case, return to the real chat and capture it for visual review.
Use --chat-only to repeat just these three screen checks after a full run.

Logs, screenshots, HTTP snapshots and the report are written only beneath the
ignored docs/validation/tensoragent-localization directory. No inference model
is installed or loaded, and this is not a performance benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import time
from typing import Any
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen
import uuid


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPO / "docs/validation/tensoragent-localization"
DEFAULT_APP = REPO / (
    "TensorAgent/src/TensorAgent.Maui/bin/Debug/net10.0-ios/"
    "iossimulator-arm64/TensorAgent.Maui.app"
)
DEFAULT_RUNTIME = "com.apple.CoreSimulator.SimRuntime.iOS-26-5"
DEFAULT_DEVICE = "com.apple.CoreSimulator.SimDeviceType.iPhone-17-Pro"
SYSTEM_CASES = [
    ("en", ["en-US"]),
    ("zh-Hans", ["zh-Hans-CN"]),
    ("zh-Hant", ["zh-Hant-TW"]),
    ("ja", ["ja-JP"]),
    ("ko", ["ko-KR"]),
    ("es", ["es-MX"]),
    ("fr", ["fr-CA"]),
    ("de", ["de-AT"]),
]
LIMITATIONS = [
    "One isolated iPhone 17 Pro simulator on iOS 26.5; no physical device or other platform was tested.",
    "AppleLanguages launch arguments exercise the app's system-language API; the simulator's Settings app is not operated.",
    "The Debug hook selects the native Settings picker through its ordinary change handler; no automated touch opens the picker.",
    "The running app's language, settings, served catalog, and saved settings file are checked. Screenshots require visual review for layout and rendered text.",
    "Share-extension UI and operating-system permission prompts are not exercised by this tool.",
    "No inference model is installed or loaded. Model replies, downloads, image/video generation, dictation, and audio are not tested.",
    "This is a functional language-selection check, not a performance benchmark.",
]


class ValidationFailure(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_catalog(tag: str) -> dict[str, str]:
    root = REPO / "TensorAgent/src/TensorAgent.Core/Localization" / tag
    strings: dict[str, str] = {}
    for path in sorted(root.glob("*.json")):
        strings.update(json.loads(path.read_text(encoding="utf-8")))
    check(bool(strings), f"No source catalog for {tag}")
    return strings


class SimulatorValidation:
    def __init__(self, args: argparse.Namespace, output: Path, report: dict[str, Any]):
        self.args = args
        self.output = output
        self.report = report
        self.device: str | None = None
        self.console: subprocess.Popen[bytes] | None = None
        self.console_file: Any = None
        with (args.app / "Info.plist").open("rb") as source:
            self.bundle = plistlib.load(source)
        self.bundle_id = self.bundle["CFBundleIdentifier"]

    def run(self, *args: str, optional: bool = False, timeout: int = 60) -> str:
        command = ["xcrun", "simctl", *args]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        with (self.output / "simctl.log").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"command": command, "exitCode": result.returncode}) + "\n")
            log.write(result.stdout)
            log.write(result.stderr)
        if result.returncode and not optional:
            raise ValidationFailure(f"simctl {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def create(self) -> None:
        runtimes = json.loads(self.run("list", "runtimes", "--json"))["runtimes"]
        runtime = next((item for item in runtimes if item["identifier"] == self.args.runtime), None)
        check(runtime is not None and runtime.get("isAvailable"), f"Runtime unavailable: {self.args.runtime}")
        devices = json.loads(self.run("list", "devicetypes", "--json"))["devicetypes"]
        check(any(item["identifier"] == self.args.device_type for item in devices),
              f"Device type unavailable: {self.args.device_type}")
        self.report["runtime"] = {key: runtime.get(key) for key in ("identifier", "name", "version", "buildversion")}
        self.report["deviceType"] = self.args.device_type
        name = "TensorAgent localization " + uuid.uuid4().hex[:10]
        self.device = self.run("create", name, self.args.device_type, self.args.runtime)
        check(bool(re.fullmatch(r"[0-9A-Fa-f-]{36}", self.device)), "simctl create did not return a device UUID")
        self.report["temporaryDevice"] = self.device
        self.report["temporaryDeviceName"] = name
        self.run("boot", self.device)
        self.run("bootstatus", self.device, "-b", timeout=180)
        self.run("install", self.device, str(self.args.app), timeout=180)
        self.run("status_bar", self.device, "override", "--time", "9:41", "--batteryState", "charged", "--batteryLevel", "100")

    def stop_app(self) -> None:
        if self.device:
            self.run("terminate", self.device, self.bundle_id, optional=True)
        if self.console:
            try:
                self.console.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.console.terminate()
                try:
                    self.console.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.console.kill()
                    self.console.wait(timeout=5)
            self.console = None
        if self.console_file:
            self.console_file.close()
            self.console_file = None

    def fresh_install(self) -> None:
        check(self.device is not None, "No temporary simulator")
        self.stop_app()
        self.run("uninstall", self.device, self.bundle_id)
        self.run("install", self.device, str(self.args.app), timeout=180)
        data = Path(self.run("get_app_container", self.device, self.bundle_id, "data"))
        check(not (data / "Library/Application Support/TensorAgent/settings.json").exists(),
              "Fresh install retained the app's settings file")

    def http(self, base: str, token: str, route: str, body: dict[str, Any] | None = None) -> str:
        headers = {"Cookie": f"tensoragent_token={token}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(base + route, headers=headers, data=data)
        with urlopen(request, timeout=3) as response:
            return response.read().decode("utf-8")

    def case(self, name: str, system: list[str], expected: str, saved: str,
             *, pick: str | None = None, initial: str | None = None,
             fresh: bool = False) -> None:
        result: dict[str, Any] = {"name": name, "status": "failed", "systemLanguages": system,
                                  "expectedLanguage": expected, "expectedSavedChoice": saved,
                                  "pickerChoice": pick, "freshInstall": fresh}
        self.report["cases"].append(result)
        print(f"Running {name}: system={','.join(system)}, expected={expected}, pick={pick or 'none'}", flush=True)
        log_path = self.output / f"{name}.log"
        try:
            if fresh:
                self.fresh_install()
            else:
                self.stop_app()
            # Do not inherit another harness's model-loading or validation hooks.
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("SIMCTL_CHILD_")}
            environment["SIMCTL_CHILD_TENSORAGENT_START_PAGE"] = "settings"
            if pick:
                environment["SIMCTL_CHILD_TENSORAGENT_PICK_LANGUAGE"] = pick
            languages = "(" + ",".join(json.dumps(tag) for tag in system) + ")"
            command = ["xcrun", "simctl", "launch", "--console", "--terminate-running-process",
                       self.device, self.bundle_id, "-AppleLanguages", languages]
            self.console_file = log_path.open("wb")
            self.console = subprocess.Popen(command, env=environment, stdout=self.console_file,
                                            stderr=subprocess.STDOUT)
            deadline = time.monotonic() + self.args.launch_timeout
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                entry = re.search(r"entry URL (http://127\.0\.0\.1:\d+/\?token=[0-9a-f]+)", text)
                appeared = re.search(r"open settings -> stack \d+, top SettingsPage", text)
                picked = re.search(r"languagecheck pick " + re.escape(pick or "") + r" -> row (\d+)", text) if pick else True
                if entry and appeared and picked:
                    url = urlsplit(entry.group(1))
                    base = f"{url.scheme}://{url.netloc}"
                    token = parse_qs(url.query)["token"][0]
                    try:
                        settings_text = self.http(base, token, "/api/agent/settings")
                        settings = json.loads(settings_text)
                        script = self.http(base, token, "/i18n.js?lang=" + expected)
                        prefix = "window.TensorAgentI18n = "
                        check(script.startswith(prefix), "App did not serve the i18n payload")
                        payload = json.loads(script[len(prefix):script.index(";\n")])
                        if payload.get("lang") == expected and settings.get("uiLanguage", "") == saved:
                            break
                        last_error = ValidationFailure(
                            f"Language {payload.get('lang')!r}, saved choice {settings.get('uiLanguage')!r}; "
                            f"expected {expected!r}, {saved!r}")
                    except Exception as error:
                        last_error = error
                if self.console.poll() is not None:
                    raise ValidationFailure(f"App exited during {name}; see {log_path.name}")
                time.sleep(0.2)
            else:
                raise ValidationFailure(f"Timed out waiting for {name}: {last_error or 'entry URL / settings page / picker hook missing'}")

            startup = re.search(r"interface language (\S+) \(saved choice '([^']*)', system ([^)]*)\)", text)
            check(startup is not None, "Startup language diagnostic was missing")
            check(startup.group(1) == (initial or expected),
                  f"Initial language was {startup.group(1)}, expected {initial or expected}")
            actual_system = startup.group(3).split(",") if startup.group(3) else []
            check(actual_system == system, f"NSLocale.PreferredLanguages was {actual_system}, expected {system}")
            result["startupLanguage"] = startup.group(1)
            result["startupSavedChoice"] = startup.group(2)
            result["observedSystemLanguages"] = actual_system
            if pick:
                check(int(picked.group(1)) >= 0, "Settings picker could not find the requested row")
                result["selectedPickerRow"] = int(picked.group(1))
            source = expected_catalog(expected)
            check(payload["strings"] == source,
                  f"Served {expected} catalog differs from this checkout's complete catalog")
            check(not re.search(r"settings screen failed|open settings failed", text),
                  "Native settings page logged a build/repaint/navigation failure")
            models_text = self.http(base, token, "/api/models")
            models = json.loads(models_text)
            check(not models.get("loaded"), "Language validation must run with no loaded inference model")
            # Check the actual file as well as its route, then relaunch to prove it survives.
            container = Path(self.run("get_app_container", self.device, self.bundle_id, "data"))
            settings_path = container / "Library/Application Support/TensorAgent/settings.json"
            disk = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
            check(disk.get("uiLanguage", "") == saved, "Persisted settings file does not contain the chosen language")
            result["settingsFileExisted"] = settings_path.exists()
            result["persistedChoice"] = disk.get("uiLanguage", "")
            result["servedCatalogKeys"] = len(payload["strings"])
            result["settingsLanguageLabel"] = payload["strings"]["settings.language.title"]
            result["modelLoaded"] = False
            (self.output / f"{name}-settings.json").write_text(settings_text + "\n", encoding="utf-8")
            (self.output / f"{name}-i18n.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (self.output / f"{name}-persisted-settings.json").write_text(json.dumps(disk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            page = self.http(base, token, "/")
            check(f'/i18n.js?lang={expected}"' in page,
                  "Served page does not select the current language's cache key")
            (self.output / f"{name}-page.html").write_text(page, encoding="utf-8")
            time.sleep(self.args.screenshot_delay)
            self.run("io", self.device, "screenshot", str(self.output / f"{name}.png"))
            if name in ("first-launch-en", "first-launch-zh-Hans", "picker-zh-Hans"):
                self.capture_chat(name, base, token, log_path, expected, result,
                                  expect_reload=bool(pick and expected != initial))
            result["status"] = "passed"
            print(f"PASS {name}", flush=True)
        except Exception as error:
            result["error"] = str(error)
            if self.device:
                self.run("io", self.device, "screenshot", str(self.output / f"{name}-failed.png"), optional=True)
            raise
        finally:
            self.stop_app()
            (self.output / "report.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def capture_chat(self, name: str, base: str, token: str, log_path: Path,
                     expected: str, result: dict[str, Any], *, expect_reload: bool) -> None:
        response = json.loads(self.http(base, token, "/api/agent/events",
                                        {"type": "open-route", "route": "main"}))
        check(response.get("ok") is True, "Native chat navigation event was rejected")
        deadline = time.monotonic() + self.args.launch_timeout
        while time.monotonic() < deadline:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            opened = re.search(r"open main -> stack (\d+), top chat", text)
            reloaded = f"reloading the page in {expected}" in text
            if opened and (not expect_reload or reloaded):
                break
            if self.console.poll() is not None:
                raise ValidationFailure("App exited while returning to the chat")
            time.sleep(0.2)
        else:
            raise ValidationFailure("Native chat navigation or deferred language reload did not complete")
        check(not re.search(r"open main failed|refresh on appearing failed|page did not reload in the new language", text),
              "Returning to the native chat logged a failure")
        result["chatNativeNavigation"] = opened.group(0)
        result["chatLanguageReloadObserved"] = reloaded
        result["chatScreenshot"] = f"{name}-chat.png"
        result["chatVisualReviewRequired"] = True
        # Navigation is asserted above; this small delay only lets WebKit paint before
        # recording the actual screen. It is not used as a success criterion.
        time.sleep(self.args.screenshot_delay)
        self.run("io", self.device, "screenshot", str(self.output / result["chatScreenshot"]))

    def cleanup(self) -> None:
        errors = []
        try:
            self.stop_app()
        except Exception as error:
            errors.append(str(error))
        if self.device:
            try:
                self.run("shutdown", self.device, optional=True)
            except Exception as error:
                errors.append(str(error))
            try:
                self.run("delete", self.device)
                self.report["temporaryDeviceDeleted"] = True
            except Exception as error:
                errors.append(str(error))
                self.report["temporaryDeviceDeleted"] = False
        self.report["cleanupErrors"] = errors
        if errors:
            self.report["status"] = "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", type=Path, default=DEFAULT_APP, help="Built Debug iossimulator-arm64 .app")
    parser.add_argument("--out", type=Path, help="New evidence directory under docs/validation/tensoragent-localization")
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME)
    parser.add_argument("--device-type", default=DEFAULT_DEVICE)
    parser.add_argument("--launch-timeout", type=float, default=90)
    parser.add_argument("--screenshot-delay", type=float, default=1)
    parser.add_argument("--chat-only", action="store_true", help="Repeat only English/Chinese first-launch and Chinese picker chat screenshots")
    args = parser.parse_args()
    args.app = args.app.resolve()
    check(args.app.is_dir() and (args.app / "Info.plist").exists(),
          f"No built bundle at {args.app}; build TensorAgent/scripts/build-sim.sh first")
    check(args.launch_timeout > 0 and args.screenshot_delay >= 0, "Invalid timeout or screenshot delay")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.out or EVIDENCE_ROOT / stamp).resolve()
    check(output.is_relative_to(EVIDENCE_ROOT.resolve()) and output != EVIDENCE_ROOT.resolve(),
          f"Evidence must be in a subdirectory of {EVIDENCE_ROOT}")
    check(not output.exists() or not any(output.iterdir()), f"Evidence directory is not empty: {output}")
    ignored = subprocess.run(["git", "check-ignore", "--quiet", str(output / "report.json")], cwd=REPO)
    check(ignored.returncode == 0, "Evidence directory is not ignored by Git")
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"status": "failed", "startedUtc": datetime.now(timezone.utc).isoformat(),
                              "app": str(args.app), "cases": [], "limitations": LIMITATIONS,
                              "scope": "chat-only" if args.chat_only else "full",
                              "plannedCases": 3 if args.chat_only else 29, "temporaryDeviceDeleted": False}
    validation = SimulatorValidation(args, output, report)
    executable = args.app / validation.bundle["CFBundleExecutable"]
    report["bundleIdentifier"] = validation.bundle_id
    report["bundleFingerprints"] = {"Info.plist": sha256(args.app / "Info.plist"),
                                     executable.name: sha256(executable)}
    for assembly in sorted(args.app.rglob("TensorAgent*.dll")):
        report["bundleFingerprints"][str(assembly.relative_to(args.app))] = sha256(assembly)
    report["sourceRevision"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                                capture_output=True, text=True, check=True).stdout.strip()
    print(f"Evidence: {output}", flush=True)
    try:
        validation.create()
        if args.chat_only:
            validation.case("first-launch-en", ["en-US"], "en", "", fresh=True)
            validation.case("first-launch-zh-Hans", ["zh-Hans-CN"], "zh-Hans", "", fresh=True)
            validation.case("picker-zh-Hans", ["en-US"], "zh-Hans", "zh-Hans", pick="zh-Hans", initial="en", fresh=True)
        else:
            for tag, system in SYSTEM_CASES:
                validation.case(f"first-launch-{tag}", system, tag, "", fresh=True)
            validation.case("first-supported-system-language", ["pt-BR", "zh-Hans-CN", "en-US"], "zh-Hans", "", fresh=True)
            validation.case("unsupported-system-fallback", ["pt-BR", "ru-RU"], "en", "", fresh=True)
            # A fixed English startup makes every picker choice independently measurable.
            for tag, _ in SYSTEM_CASES:
                validation.case(f"picker-{tag}", ["en-US"], tag, tag, pick=tag, initial="en", fresh=True)
                other = ["ja-JP"] if tag != "ja" else ["de-DE"]
                validation.case(f"persisted-{tag}", other, tag, tag)
            validation.case("reset-to-system", ["zh-Hant-TW"], "zh-Hant", "", pick="system", initial="de")
            validation.case("system-follows-next-launch", ["ko-KR"], "ko", "")
            validation.case("system-reset-unsupported-fallback", ["pt-BR", "ru-RU"], "en", "")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
        print(f"FAIL: {error}", file=sys.stderr, flush=True)
    finally:
        validation.cleanup()
        report["finishedUtc"] = datetime.now(timezone.utc).isoformat()
        report["passedCases"] = sum(case["status"] == "passed" for case in report["cases"])
        report["notRunCases"] = report["plannedCases"] - len(report["cases"])
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status'].upper()}: {report['passedCases']}/{report['plannedCases']} cases; report: {output / 'report.json'}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationFailure, subprocess.SubprocessError, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
