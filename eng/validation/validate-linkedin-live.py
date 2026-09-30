#!/usr/bin/env python3
"""Run the requested LinkedIn prompt and retain its chat for manual sign-in.

The first message is exact. Sampling and generation limits inherit the running
server configuration unless explicitly overridden. This harness never signs in,
reads a browser credential store, or deletes the session. The user completes
authentication in the browser, then a later invocation can continue the same
transcript with --continue-session --message. Reports may contain account page
text returned by the agent; keep them in ignored artifacts/ or docs/validation/.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time


PROMPT = "请使用我的账户登录LinkedIn，然后搜索从事local LLM的高级人员"


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_turn(args, client, harness, session, messages, turn, seed_complete, report):
    payload = {"messages": messages, "sessionId": session, "newChat": False,
               "tools": [], "skills_discovery": True}
    if args.max_tokens is not None:
        payload["maxTokens"] = args.max_tokens
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    turn.update(request=payload, steps=[], commands=[], answer="", final_answer="")
    connection = None
    active_command = None
    started = time.monotonic()
    try:
        connection, response = client.open_sse("/api/chat", payload, args.timeout)
        turn["http_status"] = response.status
        if response.status != 200:
            # Preserve diagnostic text on disk without echoing a response body.
            turn["http_error"] = response.read().decode("utf-8", "replace")
            raise RuntimeError(f"Chat returned HTTP {response.status}; see report")
        events_path = args.report_dir / f"events-{turn['name']}.jsonl"
        with events_path.open("w", encoding="utf-8") as events:
            for event in harness.iter_sse(response, started + args.timeout):
                events.write(json.dumps(event, ensure_ascii=False) + "\n")
                events.flush()
                if event.get("tool_progress") == "running" and isinstance(event.get("detail"), str):
                    if event.get("tool") == "skills_run" and seed_complete is not None:
                        report.setdefault("cache_ready_before_first_skills_run", seed_complete.is_set())
                        if not seed_complete.is_set():
                            raise RuntimeError("skills_run began before prepared cache seeding completed")
                    command = {"tool": event.get("tool"), "detail": event["detail"]}
                    if active_command is None or any(active_command.get(k) != v for k, v in command.items()):
                        active_command = command
                        turn["commands"].append(command)
                        print(json.dumps({"turn": turn["name"], "tool_running": command["tool"]}), flush=True)
                if event.get("skill_step"):
                    turn["steps"].append(event)
                    turn["final_answer"] = ""
                    if active_command is not None and active_command["tool"] == event["skill_step"]:
                        active_command.update(ok=event.get("ok"), round=event.get("round"))
                    active_command = None
                    print(json.dumps({"turn": turn["name"], "tool": event["skill_step"],
                                      "ok": event.get("ok"), "round": event.get("round")}), flush=True)
                if isinstance(event.get("replace"), str):
                    turn["answer"] = turn["final_answer"] = event["replace"]
                elif isinstance(event.get("token"), str):
                    turn["answer"] += event["token"]
                    turn["final_answer"] += event["token"]
                if event.get("done"):
                    turn["terminal"] = event
                    break
        terminal = turn.get("terminal")
        if not terminal or any(terminal.get(k) for k in ("error", "aborted", "truncated")):
            raise RuntimeError("Missing or failed terminal event; see report")
        turn["status"] = "completed"
    finally:
        turn["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if connection:
            connection.close()
        (args.report_dir / f"answer-{turn['name']}.txt").write_text(turn["final_answer"], encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Running TensorSharp server; required for a new session")
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--continue-session", action="store_true")
    parser.add_argument("--message", help="Continuation only; enter credentials directly in the browser")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-tokens", type=int, help="Optional override; otherwise inherit server configuration")
    parser.add_argument("--temperature", type=float, help="Optional override; otherwise inherit server configuration")
    parser.add_argument("--npm-cache-seed", type=Path, help="Optional prepared cache for a new Windows session")
    parser.add_argument("--server-scratch", type=Path)
    parser.add_argument("--seed-timeout", type=float, default=90)
    args = parser.parse_args(argv)
    args.report_dir = args.report_dir.resolve()
    report_path = args.report_dir / "report.json"
    if bool(args.npm_cache_seed) != bool(args.server_scratch):
        parser.error("--npm-cache-seed and --server-scratch must be supplied together")
    if args.continue_session:
        if not args.message or not report_path.is_file():
            parser.error("Continuation requires --message and an existing --report-dir/report.json")
        if args.npm_cache_seed:
            parser.error("Prepared cache seeding is only supported for a fresh session")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("format_version") != 1 or not report.get("turns"):
            parser.error("Unsupported or incomplete report")
        if report["turns"][-1].get("status") != "completed":
            parser.error("Previous turn did not complete; inspect its report before continuing")
        if args.base_url and args.base_url.rstrip("/") != report["base_url"]:
            parser.error("Continuation must use the original server")
        args.base_url = report["base_url"]
        messages = [*report["history"], {"role": "user", "content": args.message}]
    else:
        if args.message:
            parser.error("The first prompt is fixed; --message requires --continue-session")
        if not args.base_url:
            parser.error("A new session requires --base-url")
        if args.report_dir.exists():
            parser.error("Use a fresh --report-dir for a new session")
        if args.npm_cache_seed and not args.npm_cache_seed.is_dir():
            parser.error("--npm-cache-seed must be an existing directory")
        args.report_dir.mkdir(parents=True)
        report = {"format_version": 1, "base_url": args.base_url.rstrip("/"),
                  "initial_prompt": PROMPT, "session_retained": True,
                  "authentication_result": "requires visible-page verification and user interaction",
                  "turns": [], "history": []}
        messages = [{"role": "user", "content": PROMPT}]
    harness = load_module("validate-browser-skill.py", "linkedin_live_browser_harness")
    client = harness.HttpClient(args.base_url, 15)
    if not args.continue_session:
        report["session"] = harness.create_session(client, 30)
    turn = {"name": f"turn-{len(report['turns']) + 1:02d}", "status": "running"}
    report["turns"].append(turn)
    save_report(report_path, report)
    print(json.dumps({"session": report["session"], "turn": turn["name"],
                      "report": str(report_path), "session_retained": True}), flush=True)
    stop, complete, worker = threading.Event(), None, None
    try:
        if args.npm_cache_seed:
            adapter = load_module("validate-windows-browser-model.py", "linkedin_live_windows_adapter")
            complete = threading.Event()
            report["cache_seed"] = {"status": "waiting", "source": str(args.npm_cache_seed.resolve())}
            worker = threading.Thread(target=adapter.seed_cache, daemon=True,
                                      args=(args.npm_cache_seed.resolve(), args.server_scratch.resolve(),
                                            report["session"], args.seed_timeout, stop, complete,
                                            report["cache_seed"]))
            worker.start()
        run_turn(args, client, harness, report["session"], messages, turn, complete, report)
        report["history"] = [*messages, {"role": "assistant", "content": turn["answer"]}]
    except Exception as error:
        turn.update(status="failed", error=str(error))
    finally:
        stop.set()
        if worker:
            worker.join(timeout=5)
        save_report(report_path, report)
    browser_commands = harness.successful_browser_commands(turn.get("commands", []))
    print(json.dumps({"status": turn["status"], "session_retained": True,
                      "elapsed_seconds": turn.get("elapsed_seconds"),
                      "successful_browser_commands": len(browser_commands),
                      "answer_file": str(args.report_dir / f"answer-{turn['name']}.txt"),
                      "report": str(report_path)}), flush=True)
    return 0 if turn["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
