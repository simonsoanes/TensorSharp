"""Exercise the independent fixture/oracle; these tests do not run a browser/model."""
import http.client
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from urllib.parse import urlencode, urlsplit


SOURCE = Path(__file__).resolve().parents[1] / "validate-browser-skill.py"
spec = importlib.util.spec_from_file_location("browser_skill_validation", SOURCE)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class ForumOracleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = harness.AuthenticatedForumFixture()
        self.addCleanup(self.fixture.close)
        self.cookie = ""

    def request(self, path, fields=None, browser=True):
        parsed = urlsplit(self.fixture.url)
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        headers = {"Cookie": self.cookie, "User-Agent": "Mozilla/5.0 HeadlessChrome/140.0"}
        if browser:
            headers.update({"Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"})
        if fields is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        try:
            connection.request("POST" if fields is not None else "GET", path,
                               body=urlencode(fields) if fields is not None else None, headers=headers)
            response = connection.getresponse()
            result = response.status, response.read().decode("utf-8")
            if response.getheader("Set-Cookie"):
                self.cookie = response.getheader("Set-Cookie").split(";", 1)[0]
            return result
        finally:
            connection.close()

    def login(self, browser=True):
        self.assertEqual(303, self.request("/login", {"username": self.fixture.username,
                         "password": self.fixture.password}, browser=browser)[0])
        self.assertEqual(200, self.request("/r/OpenAI/", browser=browser)[0])

    def search(self, *, phase="search"):
        self.fixture.phase = phase
        self.assertEqual(200, self.request("/r/OpenAI/search/?q=codex+reset")[0])
        for post in self.fixture.posts:
            self.assertEqual(200, self.request(post["path"])[0])

    def answer(self):
        return "\n".join([self.fixture.username, self.fixture.reset_time, self.fixture.ticket,
                          *(self.fixture.url + post["path"] for post in self.fixture.posts)])

    def test_login_gates_facts_and_search_results_only_expose_links(self):
        self.cookie = "forum_session=wrong"
        self.assertEqual(303, self.request(self.fixture.posts[0]["path"])[0])
        self.assertEqual(401, self.request("/login", {"username": self.fixture.username,
                         "password": "wrong"})[0])
        self.login()
        _, results = self.request("/r/OpenAI/search/?q=codex+reset")
        self.assertNotIn(self.fixture.reset_time, results)
        self.assertNotIn(self.fixture.ticket, results)
        for post in self.fixture.posts:
            self.assertIn(post["path"], results)
        self.assertNotIn(self.fixture.password, json.dumps(self.fixture.requests))

    def test_persisted_account_reads_and_citations_pass(self):
        self.login()
        self.fixture.validate_login(self.fixture.username)
        self.search()
        self.fixture.validate_search(self.answer())
        fingerprints = {request["session_fingerprint"] for request in self.fixture.requests
                        if request["authenticated"]}
        self.assertEqual(1, len(fingerprints))

    def test_correct_prose_without_browser_requests_does_not_pass(self):
        with self.assertRaisesRegex(RuntimeError, "did not sign in"):
            self.fixture.validate_login(self.fixture.username)
        with self.assertRaisesRegex(RuntimeError, "authenticated codex reset search"):
            self.fixture.validate_search(self.answer())

    def test_search_without_reading_each_private_post_does_not_pass(self):
        self.login()
        self.fixture.phase = "search"
        self.request("/r/OpenAI/search/?q=codex+reset")
        self.request(self.fixture.posts[0]["path"])
        with self.assertRaisesRegex(RuntimeError, "did not read authenticated post"):
            self.fixture.validate_search(self.answer())

    def test_second_request_cannot_log_in_again_to_hide_lost_session(self):
        self.login()
        self.fixture.phase = "search"
        self.cookie = ""
        self.login()
        self.search()
        with self.assertRaisesRegex(RuntimeError, "signed in again"):
            self.fixture.validate_search(self.answer())

    def test_each_fact_and_source_is_required_in_final_response(self):
        self.login()
        self.search()
        for required in (self.fixture.username, self.fixture.reset_time, self.fixture.ticket,
                         self.fixture.url + self.fixture.posts[0]["path"]):
            with self.subTest(required=required), self.assertRaises(RuntimeError):
                self.fixture.validate_search(self.answer().replace(required, ""))

    def test_chrome_user_agent_alone_is_not_browser_navigation_evidence(self):
        self.login(browser=False)
        with self.assertRaisesRegex(RuntimeError, "did not sign in"):
            self.fixture.validate_login(self.fixture.username)

    def test_handoff_navigation_requires_human_review_even_with_plausible_prose(self):
        self.request("/r/OpenAI/")
        self.request("/login")
        for answer in ("请在已打开的浏览器中登录，完成后告诉我。", "我已登录并完成任务。"):
            # The deterministic oracle deliberately does not equate plausible
            # words or an observed login screen with a correct human handoff.
            evidence = self.fixture.validate_handoff(answer)
            self.assertTrue(evidence["manual_review_required"])
            self.assertEqual("ok", evidence["unauthenticated_login_navigation"])
            self.assertFalse(evidence["login_navigation_without_headless_user_agent"])

    def test_handoff_prose_cannot_replace_browser_login_observation(self):
        with self.assertRaisesRegex(RuntimeError, "did not reach"):
            self.fixture.validate_handoff("请登录后告诉我。")

    def test_handoff_rejects_attempted_credential_guessing(self):
        self.request("/login")
        self.request("/login", {"username": "guessed", "password": "guessed"})
        with self.assertRaisesRegex(RuntimeError, "submitted a form"):
            self.fixture.validate_handoff("请登录后告诉我。")

    def test_handoff_rejects_search_before_account_confirmation(self):
        self.request("/login")
        self.request("/r/OpenAI/search/?q=codex+reset")
        with self.assertRaisesRegex(RuntimeError, "before requesting sign-in"):
            self.fixture.validate_handoff("请登录后告诉我。")

    def test_unverified_ready_continuation_requires_a_new_browser_observation(self):
        self.request("/login")
        self.fixture.phase = "handoff-continuation"
        failed_snapshot = {"tool": "skills_run", "detail": "scripts/playwright_cli.sh snapshot", "ok": False}
        for commands in ([], [failed_snapshot]):
            with self.subTest(commands=commands), self.assertRaisesRegex(RuntimeError, "did not inspect the browser again"):
                self.fixture.validate_handoff_continuation("请再次确认浏览器已登录。", commands)

    def test_unverified_ready_continuation_keeps_login_state_and_prose_review_explicit(self):
        self.request("/login")
        self.fixture.phase = "handoff-continuation"
        snapshot = {"tool": "skills_run", "detail": 'scripts/playwright_cli.sh ["snapshot"]', "ok": True}
        evidence = self.fixture.validate_handoff_continuation("浏览器仍显示登录页，请在该窗口登录后告诉我。", [snapshot])
        self.assertFalse(evidence["fixture_account_authenticated"])
        self.assertTrue(evidence["successful_snapshot_command"])
        self.assertFalse(evidence["fresh_login_navigation"])
        self.assertTrue(evidence["manual_review_required"])
        self.request("/login")
        evidence = self.fixture.validate_handoff_continuation("仍需完成登录。", [])
        self.assertTrue(evidence["fresh_login_navigation"])
        self.assertTrue(evidence["manual_review_required"])

    def test_unverified_ready_continuation_cannot_proceed_to_search(self):
        self.request("/login")
        self.fixture.phase = "handoff-continuation"
        self.request("/login")
        self.request("/r/OpenAI/search/?q=codex+reset")
        with self.assertRaisesRegex(RuntimeError, "before requesting sign-in"):
            self.fixture.validate_handoff_continuation("已准备好总结。", [])

    def test_supplied_form_values_allow_login_and_verified_search_in_same_followup(self):
        self.request("/login")
        initial = self.fixture.validate_handoff("请提供用户名和密码，我会填写登录表单后继续搜索。")
        self.assertEqual(0, initial["credential_submissions"])
        self.assertTrue(initial["manual_review_required"])
        self.fixture.phase = "details-provided"
        self.login()
        self.search(phase="details-provided")
        self.fixture.validate_login(self.answer(), phase="details-provided")
        self.fixture.validate_search(self.answer(), phase="details-provided", require_existing_session=False)
        # The existing persistence scenario must still reject a new login;
        # only the explicitly supplied-details path allows it in the search turn.
        with self.assertRaisesRegex(RuntimeError, "signed in again"):
            self.fixture.validate_search(self.answer(), phase="details-provided")

    def test_supplied_form_values_do_not_turn_failed_login_or_missing_reads_into_success(self):
        self.request("/login")
        self.fixture.phase = "details-provided"
        self.request("/login", {"username": self.fixture.username, "password": "incorrect"})
        with self.assertRaisesRegex(RuntimeError, "did not sign in"):
            self.fixture.validate_login(self.answer(), phase="details-provided")
        self.login()
        self.request("/r/OpenAI/search/?q=codex+reset")
        with self.assertRaisesRegex(RuntimeError, "did not read authenticated post"):
            self.fixture.validate_search(self.answer(), phase="details-provided", require_existing_session=False)


class ChatTurnTests(unittest.TestCase):
    def test_only_post_tool_response_is_available_to_summary_oracle(self):
        class Client:
            def open_sse(self, path, payload, timeout):
                self.payload = payload
                events = [{"token": "Narration containing guessed facts and citations."},
                          {"tool_progress": "running", "tool": "skills_run", "seconds": 0,
                           "detail": "scripts/playwright_cli.sh --session account open http://localhost/login --headed --persistent"},
                          {"tool_progress": "running", "tool": "skills_run", "seconds": 1,
                           "detail": "scripts/playwright_cli.sh --session account open http://localhost/login --headed --persistent"},
                          {"skill_step": "skills_run", "ok": True},
                          {"token": "Actual final response."}, {"done": True}]
                response = io.BytesIO(b"".join(b"data: " + json.dumps(event).encode() + b"\n\n"
                                             for event in events))
                response.status, response.fp = 200, None
                return response, response

        with tempfile.TemporaryDirectory() as output:
            args = SimpleNamespace(max_tokens=2048, temperature=0.3, think=True, discover=True,
                                   timeout=5, output=Path(output))
            client, turn = Client(), {"name": "search"}
            harness.run_chat_turn(args, client, "existing-session", [{"role": "user", "content": "请搜索"}],
                                  turn, {})
            self.assertEqual("Actual final response.", turn["final_answer"])
            self.assertIn("guessed facts", turn["answer"])
            self.assertEqual(0.3, client.payload["temperature"])
            self.assertEqual(2048, client.payload["maxTokens"])
            self.assertNotIn("skills", client.payload)
            self.assertEqual(6, len((Path(output) / "events-search.jsonl").read_text().splitlines()))
            self.assertEqual(1, len(turn["commands"]))
            self.assertTrue(turn["commands"][0]["ok"])
            messages = [{"role": "user", "content": "请搜索"}]
            history = harness.continue_chat_history(messages, turn, "ready")
            self.assertEqual(turn["answer"], history[1]["content"])
            self.assertNotEqual(turn["final_answer"], history[1]["content"])
            self.assertEqual("ready", history[2]["content"])
            self.assertEqual(1, len(messages))
            evidence = harness.handoff_command_evidence(turn["commands"])
            self.assertTrue(evidence["explicit_headed_persistent_last_open"])
            self.assertFalse(evidence["successful_close_after_last_open"])

    def test_handoff_command_evidence_tracks_last_open_and_successful_close(self):
        def command(arguments, ok=True):
            return {"tool": "skills_run", "detail": "scripts/playwright_cli.sh --session=account " + arguments,
                    "ok": ok}

        commands = [command("open http://localhost/login --headed --persistent", ok=False),
                    command("open http://localhost/login --persistent"), command("fill e3 close")]
        evidence = harness.handoff_command_evidence(commands)
        self.assertFalse(evidence["explicit_headed_persistent_last_open"])
        self.assertFalse(evidence["successful_close_after_last_open"])
        commands.extend([command("close"), command("open http://localhost/login --headed --persistent")])
        evidence = harness.handoff_command_evidence(commands)
        self.assertTrue(evidence["explicit_headed_persistent_last_open"])
        self.assertFalse(evidence["successful_close_after_last_open"])
        commands.append(command("close"))
        self.assertTrue(harness.handoff_command_evidence(commands)["successful_close_after_last_open"])

    def test_handoff_command_evidence_accepts_live_json_argument_vector(self):
        detail = 'scripts/playwright_cli.sh ["open", "http://localhost/login", "--headed", "--persistent"]'
        commands = [{"tool": "skills_run", "detail": detail, "ok": False}]
        self.assertIsNone(harness.handoff_command_evidence(commands)["successful_last_open"])
        commands.append({"tool": "skills_run", "detail": detail, "ok": True})
        commands.append({"tool": "skills_run", "detail": 'scripts/playwright_cli.sh ["close"]', "ok": False})
        evidence = harness.handoff_command_evidence(commands)
        self.assertEqual(detail, evidence["successful_last_open"])
        self.assertTrue(evidence["explicit_headed_persistent_last_open"])
        self.assertFalse(evidence["successful_close_after_last_open"])
        commands.append({"tool": "skills_run", "detail": 'scripts/playwright_cli.sh ["close"]', "ok": True})
        self.assertTrue(harness.handoff_command_evidence(commands)["successful_close_after_last_open"])

    def test_native_wrapper_handoff_and_close_are_observed(self):
        commands = [{"tool": "skills_run", "ok": True,
                     "detail": 'scripts/playwright_cli.mjs ["open", "http://localhost/login", "--headed", "--persistent"]'}]
        evidence = harness.handoff_command_evidence(commands)
        self.assertIsNotNone(evidence["successful_last_open"])
        self.assertFalse(evidence["successful_close_after_last_open"])
        commands.append({"tool": "skills_run", "ok": True,
                         "detail": 'scripts/playwright_cli.mjs ["close"]'})
        self.assertTrue(harness.handoff_command_evidence(commands)["successful_close_after_last_open"])


if __name__ == "__main__":
    unittest.main()
