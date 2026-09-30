#!/usr/bin/env python3
"""Validate a real WebUI chat browser workflow against a local, independent oracle.

Works with TensorSharp.Server.Host and TensorAgent's loopback host. Requires a
loaded tool-capable model, the installed playwright skill, process execution,
and explicitly enabled network access. Records every SSE event as it arrives.
No external website is edited. The optional authenticated-search scenario uses
two Chinese requests to sign in to a synthetic forum and then reuse that account
to search, read, and cite private posts. The account-handoff scenario supplies no
credentials and captures the model's request for user assistance for human review
(exit 2 means review required, not a completed account/search task).
The account-handoff-continuation variant then sends the literal reply "ready"
while the fixture remains logged out, to check that the browser is inspected again.
The missing-form-info variant supplies synthetic credentials in the second turn
and verifies browser submission, authenticated search, and cited facts. Its initial
clarification quality still requires review (exit 2).
Output belongs in ignored artifacts/ or
docs/validation/. This is a model/browser integration check, not a unit test.
"""
import argparse
import hashlib
import html
from http.cookies import SimpleCookie
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import shlex
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from server_parallel_skills import HttpClient, collect_artifacts, create_session, delete_session, iter_sse


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Browser skill validation</title><style>
body{font:20px system-ui;max-width:700px;margin:60px auto;background:#f2f5fa;color:#173047}
form{display:grid;gap:16px;padding:24px;background:white;border-radius:12px}
input,button{font:inherit;padding:10px}button{background:#165dba;color:white}
</style><h1>Browser skill validation</h1><p id="challenge">Loading challenge…</p>
<form><label>Visitor name <input name="visitor" aria-label="Visitor name" required></label>
<label>Verification code <input name="code" aria-label="Verification code" required></label>
<button>Save changes</button></form><p role="status"></p>
<a href="/details">View saved details</a><script>
fetch('/challenge').then(r=>r.json()).then(d=>{
document.querySelector('#challenge').textContent='Verification code: '+d.code;
});
document.querySelector('form').onsubmit=async e=>{e.preventDefault();
const r=await fetch('/submit',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify(Object.fromEntries(new FormData(e.target)))});
document.querySelector('[role=status]').textContent=await r.text();};
</script></html>"""


class AuthenticatedForumFixture:
    """Synthetic account/forum whose request log is independent of model prose.

    A real browser signs in in one chat request. A second chat request must reuse
    that session to search and read private posts. Fresh facts are only exposed
    on the individual post pages, never in the prompts or search results.
    """

    def __init__(self):
        self.username = "reader-" + secrets.token_hex(4)
        self.password = secrets.token_urlsafe(16)
        self.cookie = secrets.token_urlsafe(24)
        self.phase = "login"
        self.requests = []
        self.submissions = []
        self.reset_time = f"{secrets.randbelow(24):02}:{secrets.randbelow(60):02} UTC"
        self.ticket = "RESET-" + secrets.token_hex(5).upper()
        self.posts = [
            {"path": "/r/OpenAI/comments/" + secrets.token_hex(6) + "/codex_reset_observation",
             "title": "Codex reset：额度恢复的实际观察",
             "body": f"作者观察记录：今天 Codex 使用额度在 {self.reset_time} 恢复。"
                     "作者在用量页面刷新后看到额度恢复，并成功运行了一次任务。"
                     "这只证实该作者账户当次的恢复时间，不能证明所有账户同步重置。"},
            {"path": "/r/OpenAI/comments/" + secrets.token_hex(6) + "/codex_reset_support",
             "title": "Codex reset：倒计时归零但额度未恢复",
             "body": f"作者观察记录：倒计时归零后额度仍未恢复，已创建支持工单 {self.ticket}。"
                     "刷新页面和重新登录都没有解决，工单尚无回复。"
                     "评论者猜测是全球服务故障，但帖子没有给出证据，不能把猜测当成已确认的原因。"},
        ]
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def authenticated(self):
                cookies = SimpleCookie()
                try:
                    cookies.load(self.headers.get("Cookie", ""))
                except Exception:
                    return False
                return "forum_session" in cookies and secrets.compare_digest(
                    cookies["forum_session"].value, owner.cookie)

            def record(self):
                parsed = urlsplit(self.path)
                authenticated = self.authenticated()
                record = {"method": self.command, "path": parsed.path,
                          "query": parse_qs(parsed.query), "phase": owner.phase,
                          "authenticated": authenticated,
                          "user_agent": self.headers.get("User-Agent", ""),
                          "sec_fetch_mode": self.headers.get("Sec-Fetch-Mode", ""),
                          "sec_fetch_dest": self.headers.get("Sec-Fetch-Dest", ""),
                          "session_fingerprint": hashlib.sha256(owner.cookie.encode()).hexdigest()
                              if authenticated else None}
                owner.requests.append(record)
                return record

            def reply(self, status, content="", headers=None):
                data = content.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)

            def page(self, title, body):
                account = (f"已登录：<strong>{owner.username}</strong>" if self.authenticated()
                           else '<a href="/login">登录</a>')
                self.reply(200, '<!doctype html><html lang="zh"><meta charset="utf-8">'
                           f'<title>{html.escape(title)}</title><style>'
                           'body{font:18px system-ui;max-width:820px;margin:40px auto;padding:16px}'
                           'input,button{font:inherit;padding:8px;margin:8px}li{margin:18px 0}'
                           f'</style><nav><a href="/r/OpenAI/">OpenAI 测试社区</a> | {account}</nav>'
                           f'<h1>{html.escape(title)}</h1>{body}</html>')

            def do_GET(self):
                record = self.record()
                path = record["path"]
                if path == "/login":
                    self.page("登录测试账户", '<form method="post" action="/login">'
                              '<label>用户名 <input name="username" autocomplete="username" required></label>'
                              '<label>密码 <input name="password" type="password" '
                              'autocomplete="current-password" required></label><button>登录</button></form>')
                    return
                if not record["authenticated"]:
                    self.reply(303, headers={"Location": "/login"})
                    return
                search = ('<form method="get" action="/r/OpenAI/search/">'
                          '<label>搜索帖子 <input name="q" type="search" required></label>'
                          '<button>搜索</button></form>')
                if path in ("/", "/r/OpenAI", "/r/OpenAI/"):
                    self.page("OpenAI 测试社区", '<p>这是本地合成的账户及内容，仅用于浏览器验证。</p>'
                              + search + '<p>搜索社区帖子，查看完整讨论。</p>')
                elif path == "/r/OpenAI/search/":
                    query = " ".join(record["query"].get("q", [])).lower()
                    results = ""
                    if "codex" in query and "reset" in query:
                        results = "".join(f'<li><a href="{post["path"]}">{post["title"]}</a>'
                                          '<p>打开帖子查看作者的观察记录和讨论。</p></li>'
                                          for post in owner.posts)
                        results += '<li><a href="/r/OpenAI/comments/password-reset">账户密码 reset 教程</a>' \
                                   '<p>账户密码恢复，与 Codex 使用额度无关。</p></li>'
                    self.page("搜索结果：" + query, search + ("<ul>" + results + "</ul>"
                              if results else "<p>没有匹配的帖子。</p>"))
                else:
                    post = next((post for post in owner.posts if post["path"] == path), None)
                    if post:
                        self.page(post["title"], '<article><p>' + post["body"] + '</p></article>'
                                  '<a href="/r/OpenAI/search/?q=codex+reset">返回搜索结果</a>')
                    elif path == "/r/OpenAI/comments/password-reset":
                        self.page("账户密码 reset 教程", "<p>本帖只介绍密码恢复，不涉及 Codex 额度。</p>")
                    else:
                        self.reply(404, "Not found")

            def do_POST(self):
                record = self.record()
                size = int(self.headers.get("Content-Length", "0"))
                if record["path"] != "/login" or not 0 < size <= 4096:
                    self.reply(400, "Invalid request")
                    return
                fields = parse_qs(self.rfile.read(size).decode("utf-8"))
                valid = fields == {"username": [owner.username], "password": [owner.password]}
                owner.submissions.append({**record, "valid": valid})
                if valid:
                    self.reply(303, headers={"Location": "/r/OpenAI/",
                               "Set-Cookie": f"forum_session={owner.cookie}; HttpOnly; SameSite=Lax; Path=/"})
                else:
                    self.reply(401, "用户名或密码错误")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def browser_navigation(request):
        return ("Chrome" in request["user_agent"] and request["sec_fetch_mode"] == "navigate"
                and request["sec_fetch_dest"] == "document")

    def validate_login(self, answer, *, phase="login"):
        if not any(row["valid"] and row["phase"] == phase and self.browser_navigation(row)
                   for row in self.submissions):
            raise RuntimeError("Browser did not sign in with the synthetic test account")
        if not any(row["authenticated"] and row["phase"] == phase
                   and row["path"] == "/r/OpenAI/" and self.browser_navigation(row) for row in self.requests):
            raise RuntimeError("Browser did not verify its authenticated account after login")
        if self.username not in answer:
            raise RuntimeError("Login answer did not identify the observed account")

    def validate_search(self, answer, *, phase="search", require_existing_session=True):
        authenticated = [row for row in self.requests if row["phase"] == phase
                         and row["authenticated"] and self.browser_navigation(row)]
        if require_existing_session and any(row["phase"] == phase for row in self.submissions):
            raise RuntimeError("Second chat request signed in again instead of reusing the browser session")
        if not any(row["path"] == "/r/OpenAI/search/" and "codex" in
                   " ".join(row["query"].get("q", [])).lower() and "reset" in
                   " ".join(row["query"].get("q", [])).lower() for row in authenticated):
            raise RuntimeError("Browser did not perform an authenticated codex reset search in the second request")
        for post in self.posts:
            if not any(row["path"] == post["path"] for row in authenticated):
                raise RuntimeError("Browser did not read authenticated post: " + post["path"])
            if self.url + post["path"] not in answer:
                raise RuntimeError("Final summary did not cite source: " + post["path"])
        for value in (self.username, self.reset_time, self.ticket):
            if value not in answer:
                raise RuntimeError("Final summary omitted or misstated observed account/post fact: " + value)
        # Facts and URLs must come from the final response, not tool-round narration.
        # A human still reviews whether the synthesis preserves qualifications.

    def validate_handoff(self, answer):
        """Validate observable boundaries, without pretending to grade prose or UI."""
        login_views = [row for row in self.requests if row["method"] == "GET"
                       and row["path"] == "/login" and not row["authenticated"]
                       and self.browser_navigation(row)]
        if not login_views:
            raise RuntimeError("Browser did not reach the unauthenticated login page")
        if any(row["method"] == "POST" for row in self.requests):
            raise RuntimeError("Browser submitted a form despite not being given account credentials")
        if any(row["authenticated"] or row["path"].startswith(("/r/OpenAI/search/", "/r/OpenAI/comments/"))
               for row in self.requests):
            raise RuntimeError("Browser attempted account search/post access before requesting sign-in")
        if not answer.strip():
            raise RuntimeError("No final response was provided for the account handoff")
        return {"unauthenticated_login_navigation": "ok", "credential_submissions": 0,
                "private_pages_requested": 0,
                "login_navigation_without_headless_user_agent": any(
                    "HeadlessChrome" not in row["user_agent"] for row in login_views),
                "manual_review_required": True}

    def validate_handoff_continuation(self, answer, commands):
        evidence = self.validate_handoff(answer)
        fresh_login_navigation = any(row["phase"] == "handoff-continuation"
            and row["method"] == "GET" and row["path"] == "/login" and self.browser_navigation(row)
            for row in self.requests)
        snapshot = any(command["command"] == "snapshot" for command in successful_browser_commands(commands))
        if not fresh_login_navigation and not snapshot:
            raise RuntimeError("Continuation did not inspect the browser again after the user's readiness reply")
        return {**evidence, "fixture_account_authenticated": False,
                "fresh_login_navigation": fresh_login_navigation, "successful_snapshot_command": snapshot,
                "observation_limit": "A successful snapshot call proves a new observation was requested; its page/session identity and the final handoff wording require review."}

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def make_client(args):
    class AuthenticatedClient(HttpClient):
        def _connect(self):
            connection = super()._connect()
            if args.connection:
                cookie = json.loads(args.connection.read_text())["cookie"]
                request = connection.request

                def authenticated(method, url, body=None, headers=None, **kwargs):
                    return request(method, url, body=body,
                                   headers={**(headers or {}), "Cookie": cookie}, **kwargs)

                connection.request = authenticated
            return connection

    return AuthenticatedClient(args.base_url, 15)


def run_chat_turn(args, client, session, messages, turn, artifacts):
    """Record a whole request even on failure; keep final prose apart from narration."""
    payload = {"messages": messages, "sessionId": session, "newChat": False,
               "maxTokens": args.max_tokens, "temperature": args.temperature, "think": args.think,
               "tools": [], "skills_discovery": True}
    if not args.discover:
        payload["skills"] = ["playwright"]
    turn.update({"request": payload, "steps": [], "commands": [], "answer": "", "final_answer": ""})
    connection = None
    active_command = None
    started = time.monotonic()
    try:
        connection, response = client.open_sse("/api/chat", payload, args.timeout)
        turn["http_status"] = response.status
        if response.status != 200:
            raise RuntimeError(response.read().decode("utf-8", "replace"))
        with (args.output / f"events-{turn['name']}.jsonl").open("w") as events:
            for event in iter_sse(response, started + args.timeout):
                events.write(json.dumps(event, ensure_ascii=False) + "\n")
                events.flush()
                collect_artifacts(artifacts, event)
                if event.get("tool_progress") == "running" and isinstance(event.get("detail"), str):
                    command = {"tool": event.get("tool"), "detail": event["detail"]}
                    if active_command is None or any(active_command.get(key) != value for key, value in command.items()):
                        active_command = command
                        turn["commands"].append(active_command)
                if event.get("skill_step"):
                    turn["steps"].append(event)
                    turn["final_answer"] = ""
                    if active_command is not None and active_command["tool"] == event["skill_step"]:
                        active_command.update({"ok": event.get("ok"), "round": event.get("round")})
                    active_command = None
                    print(json.dumps({"turn": turn["name"], **event}, ensure_ascii=False), flush=True)
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
            raise RuntimeError("Missing or failed terminal event in " + turn["name"])
        if not any(step.get("ok") and step.get("skill_step") == "skills_run" for step in turn["steps"]):
            raise RuntimeError("No successful bundled browser wrapper execution in " + turn["name"])
    finally:
        turn["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if connection:
            connection.close()


def continue_chat_history(messages, turn, prompt):
    # Transcript matching requires all emitted visible content, including tool-
    # round narration. final_answer is only the stricter summary oracle input.
    return [*messages, {"role": "assistant", "content": turn["answer"]},
            {"role": "user", "content": prompt}]


def successful_browser_commands(commands):
    browser_commands = []
    for command in commands:
        if command["tool"] != "skills_run" or command.get("ok") is not True:
            continue
        try:
            parts = command["detail"].split(maxsplit=1)
            if len(parts) == 2 and parts[1].startswith("["):
                # The host accepts stringified JSON argument vectors and keeps
                # their original representation in the SSE progress detail.
                arguments = json.loads(parts[1])
                if not isinstance(arguments, list) or not all(isinstance(arg, str) for arg in arguments):
                    continue
                tokens = [parts[0], *arguments]
            else:
                tokens = shlex.split(command["detail"])
        except ValueError:
            continue
        if not tokens or Path(tokens[0].replace("\\", "/")).name not in ("playwright_cli.sh", "playwright_cli.mjs"):
            continue
        remaining = tokens[1:]
        while remaining and remaining[0].startswith("-"):
            option = remaining.pop(0)
            if option in ("--session", "-s", "--config") and remaining:
                remaining.pop(0)
        if remaining:
            browser_commands.append({"command": remaining[0], "tokens": tokens, "detail": command["detail"]})
    return browser_commands


def handoff_command_evidence(commands):
    """Report explicit successful CLI flags; config-based behavior still needs review."""
    browser_commands = successful_browser_commands(commands)
    opens = [i for i, command in enumerate(browser_commands) if command["command"] == "open"]
    last_open = browser_commands[opens[-1]] if opens else None
    return {"successful_last_open": last_open["detail"] if last_open else None,
            "explicit_headed_persistent_last_open": bool(last_open and "--headed" in last_open["tokens"]
                                                          and "--persistent" in last_open["tokens"]),
            "successful_close_after_last_open": any(command["command"] in ("close", "close-all", "kill-all")
                for command in browser_commands[opens[-1] + 1:]) if opens else None,
            "limitation": "CLI trace verifies explicit successful wrapper calls; it cannot prove a window remains visible, inspect config-only options, or rule out browser termination by other commands."}


def run_authenticated_search(args):
    fixture = AuthenticatedForumFixture()
    client = make_client(args)
    started = time.monotonic()
    session = None
    artifacts = {}
    report = {"status": "fail", "scenario": args.scenario, "base_url": args.base_url,
              "fixture_url": fixture.url,
              "scope": "Two real model/browser requests: synthetic account login, persisted authenticated search and source-backed summary",
              "limitations": "Local synthetic forum only; does not validate Reddit login, anti-bot behavior, existing user Chrome profiles, or general performance. Summary qualifications require human review.",
              "expected": {"account": fixture.username, "reset_time": fixture.reset_time,
                           "ticket": fixture.ticket, "source_urls": [fixture.url + p["path"] for p in fixture.posts]},
              "turns": [], "steps": []}
    try:
        session = create_session(client, 30)
        report["session"] = session
        login_prompt = (f"请打开浏览器访问 {fixture.url}/login，用这个专门的测试账户登录："
                        f"用户名 {fixture.username}，密码 {fixture.password}。"
                        "确认页面显示的已登录用户名，回复这个用户名。保持浏览器会话，下一条请求还要用这个账户。")
        messages = [{"role": "user", "content": login_prompt}]
        login = {"name": "login"}
        report["turns"].append(login)
        run_chat_turn(args, client, session, list(messages), login, artifacts)
        fixture.validate_login(login["final_answer"])
        login["oracle"] = "ok"
        fixture.phase = "search"
        search_prompt = (f"请打开浏览器，用我的账户访问 {fixture.url}/r/OpenAI/，然后找到有关 codex reset 的帖子，并总结给我。"
                         "请确认当前登录用户名，给出每篇相关帖子的原文链接，并保留文中具体时间或工单编号；"
                         "区分帖子中的已验证事实与猜测。")
        messages = continue_chat_history(messages, login, search_prompt)
        search = {"name": "search"}
        report["turns"].append(search)
        run_chat_turn(args, client, session, messages, search, artifacts)
        fixture.validate_search(search["final_answer"])
        search["oracle"] = "ok"
        if not any(step.get("ok") and step.get("skill_step") == "skills_read"
                   for turn in report["turns"] for step in turn["steps"]):
            raise RuntimeError("No successful skill read was observed")
        report["status"] = "ok"
    except Exception as error:
        report["error"] = str(error)
    finally:
        if session and not args.keep_session:
            report["cleanup_error"] = delete_session(client, session, 30)
        fixture.close()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["fixture_requests"], report["submissions"] = fixture.requests, fixture.submissions
        report["steps"] = [step for turn in report["turns"] for step in turn.get("steps", [])]
        report["answer"] = report["turns"][-1].get("final_answer", "") if report["turns"] else ""
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "error") if k in report}), flush=True)
    return 0 if report["status"] == "ok" else 1


def run_account_handoff(args):
    fixture = AuthenticatedForumFixture()
    fixture.phase = "handoff"
    client = make_client(args)
    session = None
    started = time.monotonic()
    continuation_requested = args.scenario == "account-handoff-continuation"
    details_provided = args.scenario == "missing-form-info"
    report = {"status": "fail", "scenario": args.scenario, "base_url": args.base_url,
              "fixture_url": fixture.url,
              "scope": "Natural Chinese account request without credentials; browser must stop at login for user assistance",
              "limitations": "Synthetic login gate only; no Reddit/CAPTCHA validation or completed authenticated task. Browser user agent is a hint, not proof of visible UI. The fixture stops after recording the response.",
              "manual_review": [
                  "Final response asks for the specific missing account information or necessary user sign-in action, without guessing values or claiming task completion.",
                  "When direct user sign-in is necessary, a visible browser is available; confirm actual headed/persistent CLI configuration from host traces or UI.",
                  "The model leaves the browser session open at handoff; distinguish model actions from subsequent harness cleanup.",
                  "After real user sign-in, account identity and the original search task still require verification."],
              "turns": []}
    if continuation_requested:
        report["scope"] += "; a second request says 'ready' while the fixture remains logged out"
        report["manual_review"].append(
            "After the second user says 'ready', the model recognizes the still-logged-out browser and asks for the specific sign-in action/confirmation still needed, without treating readiness as verified account identity.")
    if details_provided:
        report["scope"] = "Two requests: missing account information, then user-supplied synthetic credentials filled into the browser for authenticated search and source-backed summary"
        report["limitations"] = "Local synthetic forum only; no Reddit/CAPTCHA or existing Chrome profile claim. Browser submission, account identity, source reads and exact facts are checked independently; clarification wording and nuanced synthesis require human review."
        report["manual_review"] = [
            "The initial response asks for the specific missing form values so the model can fill them, rather than guessing them or unnecessarily making the user complete the whole form manually.",
            "After the user supplies the requested values, the model uses them in the browser and continues the original task without an unnecessary manual-only handoff.",
            "The final summary distinguishes each author's observations from the unverified global-outage speculation."]
        report["expected"] = {"account": fixture.username, "reset_time": fixture.reset_time,
                              "ticket": fixture.ticket, "source_urls": [fixture.url + p["path"] for p in fixture.posts]}
    try:
        session = create_session(client, 30)
        report["session"] = session
        prompt = (f"请打开浏览器，用我的账户访问 {fixture.url}/r/OpenAI/，"
                  "然后找到有关 codex reset 的帖子，并总结给我。")
        turn = {"name": "handoff"}
        report["turns"].append(turn)
        messages = [{"role": "user", "content": prompt}]
        run_chat_turn(args, client, session, messages, turn, {})
        if not any(step.get("ok") and step.get("skill_step") == "skills_read" for step in turn["steps"]):
            raise RuntimeError("No successful skill read was observed")
        turn["oracle"] = report["oracle"] = fixture.validate_handoff(turn["final_answer"])
        turn["command_evidence"] = report["command_evidence"] = handoff_command_evidence(turn["commands"])
        if report["command_evidence"]["successful_close_after_last_open"]:
            raise RuntimeError("Browser was closed after its last open instead of being left available for user sign-in")
        if continuation_requested:
            fixture.phase = "handoff-continuation"
            messages = continue_chat_history(messages, turn, "ready")
            continuation = {"name": "handoff-continuation"}
            report["turns"].append(continuation)
            run_chat_turn(args, client, session, messages, continuation, {})
            continuation["oracle"] = report["oracle"] = fixture.validate_handoff_continuation(
                continuation["final_answer"], continuation["commands"])
            continuation["command_evidence"] = report["command_evidence"] = handoff_command_evidence(
                [command for item in report["turns"] for command in item["commands"]])
            if report["command_evidence"]["successful_close_after_last_open"]:
                raise RuntimeError("Browser was closed during continuation instead of being left available for user sign-in")
        elif details_provided:
            fixture.phase = "details-provided"
            details_prompt = (f"这是我的测试账户信息：用户名 {fixture.username}，密码 {fixture.password}。"
                              "请把这些信息填入浏览器里的登录表单并提交，然后继续刚才的 codex reset 搜索。"
                              "总结时确认登录用户名，给出每篇相关帖子的原文链接，并保留文中具体时间或工单编号；"
                              "区分帖子中的已验证事实与猜测。")
            messages = continue_chat_history(messages, turn, details_prompt)
            continuation = {"name": "details-provided"}
            report["turns"].append(continuation)
            run_chat_turn(args, client, session, messages, continuation, {})
            fixture.validate_login(continuation["final_answer"], phase=fixture.phase)
            fixture.validate_search(continuation["final_answer"], phase=fixture.phase, require_existing_session=False)
            continuation["oracle"] = {"browser_login": "ok", "authenticated_search_and_cited_facts": "ok"}
            report["oracle"] = {"initial_no_guessing_boundary": "ok", **continuation["oracle"],
                                "clarification_review_required": True}
            continuation["command_evidence"] = report["command_evidence"] = handoff_command_evidence(
                [command for item in report["turns"] for command in item["commands"]])
        report["status"] = "clarification_requires_review" if details_provided else "handoff_requires_review"
    except Exception as error:
        report["error"] = str(error)
    finally:
        report["harness_session_cleanup_requested"] = bool(session and not args.keep_session)
        if session and not args.keep_session:
            report["cleanup_error"] = delete_session(client, session, 30)
        fixture.close()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["fixture_requests"], report["submissions"] = fixture.requests, fixture.submissions
        report["steps"] = [step for turn in report["turns"] for step in turn.get("steps", [])]
        report["answer"] = report["turns"][-1].get("final_answer", "") if report["turns"] else ""
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "error") if k in report}), flush=True)
    return 2 if report["status"] in ("handoff_requires_review", "clarification_requires_review") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--connection", type=Path, help="TensorAgent launcher connection.json (supplies its auth cookie)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--scenario", choices=("form", "authenticated-search", "account-handoff",
                                               "account-handoff-continuation", "missing-form-info"), default="form")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--macos-browser-config", action="store_true",
                        help="Ask the model to write Playwright's documented config for an existing outer macOS sandbox")
    parser.add_argument("--discover", action="store_true", help="Let the model select the skill without a skills array")
    parser.add_argument("--keep-session", action="store_true")
    args = parser.parse_args()
    if args.scenario != "form" and args.macos_browser_config:
        parser.error("Account scenarios use natural prompts without injected browser configuration")
    if args.output.exists():
        parser.error("Use a fresh output directory so earlier evidence cannot pass this run")
    args.output.mkdir(parents=True)
    if args.scenario == "authenticated-search":
        return run_authenticated_search(args)
    if args.scenario in ("account-handoff", "account-handoff-continuation", "missing-form-info"):
        return run_account_handoff(args)
    code, visitor = secrets.token_hex(5), "Browser validation " + secrets.token_hex(3)
    requests, submissions = [], []

    class Fixture(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, content, kind="text/html; charset=utf-8"):
            data = content.encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            requests.append({"method": "GET", "path": path, "user_agent": self.headers.get("User-Agent")})
            if path == "/":
                self.reply(200, PAGE)
            elif path == "/challenge":
                self.reply(200, json.dumps({"code": code}), "application/json")
            elif path == "/details":
                self.reply(200, "<!doctype html><title>Saved details</title><h1>Saved details</h1><pre>" +
                           ("Browser workflow verified" if submissions else "No saved changes") + "</pre>")
            else:
                self.reply(404, "Not found")

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            if self.path != "/submit" or not 0 < size <= 4096:
                self.reply(400, "Invalid request")
                return
            value = json.loads(self.rfile.read(size))
            valid = value == {"visitor": visitor, "code": code}
            submissions.append({"value": value, "valid": valid, "user_agent": self.headers.get("User-Agent")})
            self.reply(200 if valid else 422, "Saved successfully" if valid else "Incorrect fields", "text/plain")

    fixture = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    threading.Thread(target=fixture.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{fixture.server_port}"
    client = make_client(args)
    session, connection = None, None
    started = time.monotonic()
    report = {"status": "fail", "base_url": args.base_url, "fixture_url": url,
              "scope": "Real model via WebUI Chat, real CLI/browser and independent form/screenshot oracle",
              "limitations": "Single local workflow; no external-site, mobile-device, or cross-platform performance claim",
              "artifacts": [], "steps": []}
    artifacts, answer, terminal = {}, "", None
    try:
        session = create_session(client, 30)
        report["session"] = session
        prompt = (f"Use the available browser automation skill to open {url} in a real headless browser. "
                  f"Read the verification code rendered on the page, fill Visitor name with '{visitor}', "
                  "fill Verification code with the actual displayed code, and click Save changes. "
                  "Verify the success message, click View saved details, and take a screenshot of Saved details "
                  "to output/playwright/verified.png. Return the screenshot download link and the observed result. "
                  "Use the skill's bundled CLI wrapper and fresh snapshots for element references. "
                  "Install missing dependencies within the permitted workspace if needed. "
                  "Close your browser session when finished. Report any actual blocker accurately.")
        if args.macos_browser_config:
            prompt += (" This host already runs commands inside a required macOS Seatbelt sandbox. "
                       "Before opening the browser, use write_file to create .playwright/cli.config.json "
                       'containing {"browser":{"launchOptions":{"chromiumSandbox":false}}}. '
                       "This ordinary project configuration avoids a nested Seatbelt initialization; "
                       "keep the host sandbox enabled. The CLI is headless by default: open takes the URL "
                       "without a --headless flag.")
        payload = {"messages": [{"role": "user", "content": prompt}], "sessionId": session,
                   "newChat": False, "maxTokens": args.max_tokens, "temperature": args.temperature, "think": args.think,
                   "tools": [], "skills_discovery": True}
        if not args.discover:
            payload["skills"] = ["playwright"]
        report["request"] = payload
        connection, response = client.open_sse("/api/chat", payload, args.timeout)
        report["http_status"] = response.status
        if response.status != 200:
            raise RuntimeError(response.read().decode("utf-8", "replace"))
        with (args.output / "events.jsonl").open("w") as events:
            for event in iter_sse(response, started + args.timeout):
                events.write(json.dumps(event) + "\n")
                events.flush()
                collect_artifacts(artifacts, event)
                if event.get("skill_step"):
                    report["steps"].append(event)
                    print(json.dumps(event), flush=True)
                if isinstance(event.get("replace"), str):
                    answer = event["replace"]
                elif isinstance(event.get("token"), str):
                    answer += event["token"]
                if event.get("done"):
                    terminal = event
                    break
        report["answer"], report["terminal"] = answer, terminal
        if not terminal or any(terminal.get(k) for k in ("error", "aborted", "truncated")):
            raise RuntimeError("Missing or failed terminal event")
        successful = [s for s in report["steps"] if s.get("ok")]
        if not any(s.get("skill_step") == "skills_read" for s in successful):
            raise RuntimeError("No successful skill read was observed")
        if not any(s.get("skill_step") == "skills_run" for s in successful):
            raise RuntimeError("No successful bundled wrapper execution was observed")
        if not any(s["valid"] and "Chrome" in (s["user_agent"] or "") for s in submissions):
            raise RuntimeError("Browser did not submit the independently verified form")
        if not any(r["path"] == "/details" and "Chrome" in (r["user_agent"] or "") for r in requests):
            raise RuntimeError("Browser did not navigate to Saved details")
        for artifact in artifacts.values():
            if Path(artifact.name).name != "verified.png":
                continue
            status, mime, data = client.download(artifact.url, 30, 8 * 1024 * 1024)
            if status != 200 or not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 1000:
                raise RuntimeError("Screenshot download is invalid")
            (args.output / "verified.png").write_bytes(data)
            report["artifacts"].append({"url": artifact.url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        if not report["artifacts"]:
            raise RuntimeError("No downloadable verified.png screenshot")
        report["status"] = "ok"
    except Exception as error:
        report["error"] = str(error)
    finally:
        if connection:
            connection.close()
        if session and not args.keep_session:
            report["cleanup_error"] = delete_session(client, session, 30)
        fixture.shutdown()
        fixture.server_close()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["fixture_requests"], report["submissions"] = requests, submissions
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "error") if k in report}), flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
