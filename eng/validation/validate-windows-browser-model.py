#!/usr/bin/env python3
"""Add Windows desktop evidence and a prepared npm cache to the browser model oracle.

Pass the normal validate-browser-skill.py arguments after this adapter's options.
Run with python -X utf8 so the original oracle can write Chinese text on Windows.
The server must already be running with native Node.js/npm. This adapter changes
only a fresh session's cache; it does not change server environment or npm config.
Exit 2 retains the original oracle's manual-review requirement; exit 3 is skipped.
"""
import argparse
import ctypes
from ctypes import wintypes
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
import threading
import time


def load_harness():
    source = Path(__file__).with_name("validate-browser-skill.py")
    spec = importlib.util.spec_from_file_location("windows_browser_model_oracle", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def desktop_windows(title):
    """Inspect existing windows without activating or changing any of them."""
    user = ctypes.WinDLL("user32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    dwm = ctypes.WinDLL("dwmapi", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user.IsWindowVisible.argtypes = user.IsIconic.argtypes = [wintypes.HWND]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.ProcessIdToSessionId.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    dwm.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    own_session = wintypes.DWORD()
    if not kernel.ProcessIdToSessionId(os.getpid(), ctypes.byref(own_session)):
        raise ctypes.WinError(ctypes.get_last_error())
    rows = []

    @callback_type
    def visit(handle, _):
        text = ctypes.create_unicode_buffer(2048)
        user.GetWindowTextW(handle, text, len(text))
        if title not in text.value:
            return True
        pid, session, rect, cloaked = wintypes.DWORD(), wintypes.DWORD(), wintypes.RECT(), wintypes.DWORD()
        user.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        session_ok = kernel.ProcessIdToSessionId(pid.value, ctypes.byref(session))
        rect_ok = user.GetWindowRect(handle, ctypes.byref(rect))
        cloak_ok = dwm.DwmGetWindowAttribute(handle, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked)) == 0
        process = kernel.OpenProcess(0x1000, False, pid.value)
        name = ""
        if process:
            try:
                image, length = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
                if kernel.QueryFullProcessImageNameW(process, 0, image, ctypes.byref(length)):
                    name = Path(image.value).stem.lower()
            finally:
                kernel.CloseHandle(process)
        left, top = user.GetSystemMetrics(76), user.GetSystemMetrics(77)
        on_screen = bool(rect_ok and rect.right > left and rect.bottom > top
                         and rect.left < left + user.GetSystemMetrics(78)
                         and rect.top < top + user.GetSystemMetrics(79))
        row = dict(handle=int(handle), process_id=pid.value, process_name=name,
                   session_id=session.value, title=text.value,
                   visible=bool(user.IsWindowVisible(handle)), minimized=bool(user.IsIconic(handle)),
                   cloaked=bool(cloaked.value), cloak_query_ok=cloak_ok, on_screen=on_screen,
                   width=rect.right - rect.left, height=rect.bottom - rect.top)
        row["usable"] = bool(row["visible"] and not row["minimized"] and cloak_ok and not row["cloaked"]
                             and on_screen and row["width"] > 100 and row["height"] > 100
                             and name in ("chrome", "chromium", "msedge")
                             and session_ok and session.value == own_session.value)
        rows.append(row)
        return True

    if not user.EnumWindows(visit, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return rows


def copy_path(path):
    """Give Python's Windows file APIs long-path support after normal path checks."""
    value = str(path)
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def seed_cache(seed, scratch, session, timeout, stop, complete, evidence):
    started = time.monotonic()
    deadline = started + timeout
    safe_id = re.sub(r"[^a-zA-Z0-9]", "", session)[:64]
    try:
        while not stop.is_set() and time.monotonic() < deadline:
            matches = [path / "work" for path in scratch.glob(f"ts-session-{safe_id}-*")
                       if (path / "work").is_dir()]
            if len(matches) > 1:
                raise RuntimeError("Multiple workspaces match the newly created session")
            if matches:
                work = matches[0].resolve()
                if not work.is_relative_to(scratch):
                    raise RuntimeError("Session workspace resolves outside server scratch")
                destination = work / ".home" / "AppData" / "Local" / "npm-cache"
                if not destination.resolve().is_relative_to(work):
                    raise RuntimeError("Session cache resolves outside session workspace")
                evidence["workspace_wait_seconds"] = round(time.monotonic() - started, 3)
                copying = time.monotonic()

                def copy_file(source, target):
                    if stop.is_set() or time.monotonic() >= deadline:
                        raise TimeoutError("Cache seeding exceeded its deadline")
                    if Path(source).is_symlink():
                        raise RuntimeError("Cache seed must contain ordinary files")
                    return shutil.copy2(source, target)

                # npm's SHA512 filenames plus a session's private-home prefix can
                # exceed MAX_PATH even when both roots are short. Keep all scope
                # checks above on ordinary resolved paths; extend only file I/O.
                shutil.copytree(copy_path(seed), copy_path(destination), copy_function=copy_file)
                evidence.update(status="seeded", destination=str(destination),
                                copy_seconds=round(time.monotonic() - copying, 3))
                complete.set()
                return
            stop.wait(0.025)
        raise TimeoutError("Fresh session workspace did not become available before cache seed deadline")
    except Exception as error:
        evidence.update(status="failed", error=str(error))
    finally:
        evidence["elapsed_seconds"] = round(time.monotonic() - started, 3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npm-cache-seed", type=Path, required=True)
    parser.add_argument("--server-scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-timeout", type=float, default=60)
    parser.add_argument("--window-timeout", type=float, default=5)
    parser.add_argument("--scenario", choices=("account-handoff", "account-handoff-continuation", "missing-form-info"),
                        default="account-handoff-continuation")
    options, forwarded = parser.parse_known_args(argv)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if options.output.exists():
        parser.error("Use a fresh output directory")
    if not 0 < options.seed_timeout <= 60 or not 0 < options.window_timeout <= 60:
        parser.error("Timeouts must be between 0 and 60 seconds")
    report = {"status": "failed", "scenario": options.scenario, "cache_seed": {}, "windows": [],
              "cleanup": {"requested": False},
              "limitations": ["Prepared npm cache excludes package-download latency; provenance must accompany its seed.",
                              "Window evidence establishes desktop availability, not successful real-account sign-in or model response quality.",
                              "Session deletion and subsequent window disappearance are reported separately; disappearance does not prove every daemon process exited."]}
    reason = ("Requires Windows" if os.name != "nt" else
              "Prepared npm cache is unavailable" if not (options.npm_cache_seed / "_cacache").is_dir() else
              "Server scratch directory is unavailable" if not options.server_scratch.is_dir() else None)
    if reason:
        options.output.mkdir(parents=True)
        report.update(status="skipped", reason=reason)
        (options.output / "windows-model-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "skipped", "reason": reason}))
        return 3

    harness = load_harness()
    token = "TensorSharp Windows " + secrets.token_hex(8)
    report["fixture_window_marker"] = token
    original_fixture = harness.AuthenticatedForumFixture

    class MarkedFixture(original_fixture):
        def __init__(self):
            super().__init__()
            handler = self.server.RequestHandlerClass
            original_page = handler.page
            handler.page = lambda instance, title, body: original_page(instance, title + " [" + token + "]", body)

    harness.AuthenticatedForumFixture = MarkedFixture
    original_turn, original_events = harness.run_chat_turn, harness.iter_sse
    stop, complete = threading.Event(), threading.Event()
    worker = None

    def events(*args, **kwargs):
        for event in original_events(*args, **kwargs):
            if event.get("tool_progress") == "running" and event.get("tool") == "skills_run":
                report.setdefault("cache_ready_before_first_skills_run", complete.is_set())
                if not complete.is_set():
                    raise RuntimeError("skills_run began before prepared cache seeding completed")
            yield event

    def run_turn(args, client, session, messages, turn, artifacts):
        nonlocal worker
        if worker is None:
            report["session"] = session
            worker = threading.Thread(target=seed_cache, daemon=True,
                                      args=(options.npm_cache_seed.resolve(), options.server_scratch.resolve(), session,
                                            options.seed_timeout, stop, complete, report["cache_seed"]))
            worker.start()
        original_turn(args, client, session, messages, turn, artifacts)
        if not complete.is_set():
            raise RuntimeError("Prepared cache seeding did not complete: " + str(report["cache_seed"]))
        if report.get("cache_ready_before_first_skills_run") is not True:
            raise RuntimeError("No skills_run progress event established that cache seeding completed before execution")
        if turn["name"] in ("handoff", "handoff-continuation"):
            started = time.monotonic()
            while True:
                windows = desktop_windows(token)
                if any(row["usable"] for row in windows) or time.monotonic() - started >= options.window_timeout:
                    break
                time.sleep(0.1)
            report["windows"].append({"turn": turn["name"], "elapsed_seconds": round(time.monotonic() - started, 3),
                                      "before_next_turn_or_cleanup": True, "matches": windows})
            if not any(row["usable"] for row in windows):
                raise RuntimeError("No visible, usable native browser window for the unique fixture at " + turn["name"])

    original_cleanup = harness.delete_session

    def delete_session(*args, **kwargs):
        cleanup = report["cleanup"]
        cleanup["requested"] = True
        try:
            error = original_cleanup(*args, **kwargs)
        except Exception as cleanup_error:
            # Preserve the original oracle's fixture/report cleanup even if its
            # HTTP cleanup helper unexpectedly throws instead of returning text.
            error = "Session cleanup raised: " + str(cleanup_error)
        cleanup["error"] = error
        if report["windows"]:
            started = time.monotonic()
            try:
                while True:
                    windows = desktop_windows(token)
                    if not windows or time.monotonic() - started >= options.window_timeout:
                        break
                    time.sleep(0.1)
                cleanup.update(window_status="disappeared" if not windows else "windows_remain",
                               window_check_seconds=round(time.monotonic() - started, 3), matches=windows)
            except Exception as observation_error:
                cleanup["window_observation_error"] = str(observation_error)
        return error

    harness.run_chat_turn, harness.iter_sse, harness.delete_session = run_turn, events, delete_session
    original_argv = sys.argv
    started = time.monotonic()
    result = 1
    try:
        sys.argv = [str(Path(__file__).with_name("validate-browser-skill.py")),
                    "--output", str(options.output), "--scenario", options.scenario, *forwarded]
        result = harness.main()
        report["oracle_exit_code"] = result
        report["status"] = "requires_review" if result == 2 else "failed"
        if report["cleanup"].get("error"):
            report.update(status="failed", error="Session cleanup failed: " + str(report["cleanup"]["error"]))
            result = 1
    except Exception as error:
        report["error"] = str(error)
        result = 1
    finally:
        stop.set()
        if worker:
            worker.join(timeout=1)
        sys.argv = original_argv
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        options.output.mkdir(parents=True, exist_ok=True)
        (options.output / "windows-model-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"windows_validation": report["status"], "elapsed_seconds": report["elapsed_seconds"]}))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
