---
name: "playwright"
description: "Use when the task requires automating a real browser from the terminal (navigation, form filling, snapshots, screenshots, data extraction, UI-flow debugging) via `playwright-cli` or the bundled wrapper script."
---


# Playwright CLI Skill

Drive a real browser from the terminal using `playwright-cli`. Prefer the bundled wrapper script so the CLI works even when it is not globally installed.
Treat this skill as CLI-first automation. Do not pivot to `@playwright/test` unless the user explicitly asks for test files.

## Prerequisite check (required)

Use the operating system reported by the host's shell tool. Check native
Node.js 20 or later and npm once per workspace. In **Windows PowerShell**:

```powershell
node --version
npm.cmd --version
```

On **macOS/Linux**:

```bash
node --version
npm --version
```

If Node.js/npm is missing, report the missing dependency and ask the user to
install it for the host operating system, then restart the host so it inherits
the new PATH. The wrapper needs npm but does not require a global CLI install.
On Windows, use native `node.exe` and npm; WSL Node/npm run in a different
environment and cannot provide this desktop login handoff. Do not use `bash`,
WSL, `uname`, or `command -v` for the Windows workflow.

## Browser setup (before the first open)

On **macOS inside an existing `sandbox-exec` / Seatbelt sandbox** (reported by
the shell tool), Chromium cannot initialize a second sandbox. Before `open`,
create `.playwright/cli.config.json` in the workspace using `write_file`:

```json
{"browser":{"launchOptions":{"chromiumSandbox":false}}}
```

If the file already exists, read it and patch only that setting. Keep the host's
outer sandbox enabled. Do not apply this setting on other hosts without a
documented need. The CLI reads this project config automatically; its default
is headless (there is no `--headless` flag). Reuse this setup for later calls.

When `skills_run` is available, invoke the bundled wrapper with
`skill="playwright"`, `path="scripts/playwright_cli.mjs"`, and separate `args`,
for example `["open", "https://example.com"]`. The host resolves its path and
uses native Node.js. Use this entry point on every operating system. The `.sh`
wrapper remains available for existing macOS/Linux integrations only.

## Missing information and user interaction

For any browser step that needs information or a choice you do not have, inspect
the page and ask a focused question naming what is missing. This applies to form
fields, dates, selections, verification information, and account details. Wait
before the dependent action, while continuing useful independent work.

When the user answers, use their values to fill or select the corresponding
controls and perform the requested action within the task's authorized scope.
Use fresh page observations, verify the result, and continue the original task.
If an answer is partial, complete the supported steps and ask only for what is
still missing. Do not invent values, repeat answered questions, or merely tell
the user to perform actions you can now carry out. Ask for direct participation
only when information alone cannot enable the step. Use the language of the
original request for questions and the final response.

## Accounts and existing sessions

An automated browser does **not** inherit the user's desktop browser login.
A named CLI session preserves cookies while it is running; `--persistent`
saves its own profile across browser restarts, without importing another profile.
If a new session needs an explicit profile path, use the workspace's reserved
private directory, for example `--profile=.home/p/linkedin`. Browser profiles
contain authentication state: keep them outside `output/` and other artifact
directories, and do not export their contents. Reuse any established profile.
Reuse the current session with `goto` and verify the visible account identity
before claiming to browse as the user. Read `references/workflows.md` only if
login, persistent state, or an existing browser connection is needed.

After the user confirms sign-in, take a fresh snapshot of the current page
before navigating. Their confirmation starts verification; it is not evidence
of the browser's account state. Verify identity in the site's account controls
or profile menu. Judge `Log In` / `Sign Up` controls in that account context;
the same words in ads, posts, or comments do not establish account state or
override a verified profile identity. If the account controls show sign-in is
required or the identity is unclear, inspect the account menu or login page
and ask for the remaining sign-in step. Do not continue an account-required search
until the visible account is confirmed, even when public results are accessible.
Once a fresh page establishes that sign-in is still required, ask the user
immediately. Repeated navigation or screenshots cannot complete their sign-in.

Apply the interaction pattern above to sign-in: ask for the missing method or
account details, fill supplied values, submit the authorized login, and verify
the observed account. Do not default to handing the whole login form back.

Use credentials deliberately supplied by the user only for the requested site
and action. Prefer a host-supported protected input path or direct browser entry
for secrets; do not claim ordinary chat or tool traces are secret storage. Do
not echo secret values in responses or copy them into reports. Do not read
desktop credential stores or import cookies from another profile.

Use manual handoff for steps that need the user's direct participation, such as
CAPTCHA or unavailable sign-in information. When a desktop is available, keep
that page open with `--headed --persistent`; ask a specific question and wait
before the dependent action. Identify the page title and site for the user.
On Windows, confirm that this session was launched through the native `.mjs`
wrapper, not WSL. If they cannot find the window, use `tab-list` and `tab-select`
to bring the existing tab forward; headed launch alone does not prove the window
is visible.
If Chrome reports `Failed To Create Data Directory`, inspect the profile path.
A deeply nested Windows workspace can exceed Chrome's path limits. For a new
session, use a shorter explicit workspace profile, such as
`--profile=.home/p/linkedin`, and keep that profile with the same
session. Preserve any existing profile and sign-in state; do not silently switch
an established account session to an empty profile. See `references/workflows.md`.
Explain any remaining display limitation instead of insisting they use an
unseen window. An account-required task must wait for actual sign-in; do not
substitute anonymous research unless the user agrees.

## Skill path (when using the shell directly)

Prefer `skills_run` above. Otherwise set `PWCLI` to the wrapper in the configured
`--skills-dir`. For a standard user-scoped install on macOS/Linux:

```bash
export PWCLI="${CODEX_HOME:-$HOME/.codex}/skills/playwright/scripts/playwright_cli.mjs"
```

On Windows PowerShell, use the actual native skill path, for example:

```powershell
$pwcli = 'C:\path\to\skills\playwright\scripts\playwright_cli.mjs'
node $pwcli open https://example.com/login --headed --persistent
```

The examples below use Bash quoting; in PowerShell use `node $pwcli` with separate
arguments. Do not paste Bash exports or aliases into PowerShell.

## Quick start

Use the wrapper script:

```bash
node "$PWCLI" open https://playwright.dev --headed
# Read the linked snapshot to obtain current element refs.
node "$PWCLI" click e15
node "$PWCLI" type "Playwright"
node "$PWCLI" press Enter
node "$PWCLI" screenshot
```

If the user prefers a global install, this is also valid:

```bash
npm install -g @playwright/cli@0.1.21
playwright-cli --help
```

## Core workflow

1. Open the page.
2. Read the snapshot file linked in the CLI output with `read_file` to get stable element refs. `open`, `goto`, and interactions often already save a snapshot; avoid requesting the same snapshot twice.
3. Interact using refs from the latest snapshot.
4. Read the new snapshot after navigation or significant DOM changes; call `snapshot` if none was produced or it is stale.
5. Capture artifacts (screenshot, pdf, traces) when useful.

Read snapshots in bounded sections: start with `read_file` and `limit=80`
for navigation or account controls, then read the relevant section as needed.
For a large feed or search page, use a targeted shell search on the saved snapshot
to locate headings, result links, or post text before reading more. Avoid loading
an entire feed with repeated advertising/tracking URLs just to find a control.
The wrapper saves `snapshot` to a unique file under `output/playwright/` and
returns its link; an explicit `--filename` is preserved. When using the CLI
directly, pass `snapshot --filename=output/playwright/current.yml` to avoid
printing the entire page into the tool response.

Keep routine tool steps brief; do not restate the task or repeat a full plan
between commands. When several actions use known refs and do not depend on one
another's results (for example, filling two fields), issue them in the same tool
round. Wait for navigation results before using refs on the new page.

Minimal loop:

```bash
node "$PWCLI" open https://example.com
# Read the linked snapshot and choose the actual element ref.
node "$PWCLI" click e3
# Read the new snapshot linked by the interaction.
```

## When to snapshot again

Use a fresh snapshot after:

- navigation
- clicking elements that change the UI substantially
- opening/closing modals or menus
- tab switches

Read the new snapshot linked by the command. Call `snapshot` when none was
produced or when a user action changed the page since the last snapshot.
Refs can go stale. When a command fails due to a missing ref, snapshot again.

```bash
node "$PWCLI" snapshot
# Read the linked file in bounded sections; the wrapper supplies a unique filename.
```

## Recommended patterns

### Form fill and submit

```bash
node "$PWCLI" open https://example.com/form
# Read the linked snapshot to obtain the form's actual refs.
node "$PWCLI" fill e1 "user@example.com"
node "$PWCLI" fill e2 "password123"
node "$PWCLI" click e3
# Read the new snapshot linked by the submission.
```

### Debug a UI flow with traces

```bash
node "$PWCLI" open https://example.com --headed
node "$PWCLI" tracing-start
# ...interactions...
node "$PWCLI" tracing-stop
```

### Multi-tab work

```bash
node "$PWCLI" tab-new https://example.com
node "$PWCLI" tab-list
node "$PWCLI" tab-select 0
node "$PWCLI" snapshot
```

## Wrapper script

The wrapper pins the validated CLI version and prefers npm's local cache, so the
CLI can run without a global install or repeated registry checks. On Windows it
launches native Node.js with npm's JavaScript entry point and separate arguments;
the wrapper does not invoke `npx.cmd` through a shell or use WSL:

```bash
node "$PWCLI" --help
```

Prefer the wrapper unless the repository already standardizes on a global install.

## References

Open only what you need:

- CLI command reference: `references/cli.md`
- Practical workflows and troubleshooting: `references/workflows.md`

## Guardrails

- Use element ids like `e12` only from a fresh snapshot you have read.
- Re-snapshot when refs seem stale.
- Prefer explicit commands over `eval` and `run-code` unless needed.
- When you do not have a fresh snapshot, use placeholder refs like `eX` and say why; do not bypass refs with `run-code`.
- Use `--headed` when a visual check will help.
- When capturing artifacts in this repo, use `output/playwright/` and avoid introducing new top-level artifact folders. Browser profiles belong in the reserved private `.home/p/` directory, never in artifact output.
- To choose a screenshot path, use `screenshot --filename=output/playwright/verified.png`; the positional argument to `screenshot` is an element ref, not a file path. Create the output directory first if necessary.
- Default to CLI commands and workflows, not Playwright test specs.
- For search and summaries, filter results by their titles/snippets, then open relevant posts and read their content before summarizing. Include source links and distinguish posts from comments or your own inference. Attribute claims to their authors; reading a post does not independently verify its claims.
- Keep attribution explicit for unverified explanations: say an author reports or a commenter suggests something, rather than declaring its cause or an official policy. A comment citing documentation remains a comment until that documentation is checked. Do not infer community consensus from a few selected comments; omit unrelated disputes and identify uncertainty where it affects the answer.
- Preserve the basis of numerical comparisons (for example, this week versus last week is not one model versus another). Describe limits of the content you inspected; missing comments in a snapshot do not establish that the discussion contains no answer.
- On `Session closed`, check the setup above and the CLI's documented configuration. Retry only after changing the cause. Do not repeatedly run the same failing command or investigate installed dependency source for an ordinary browsing task.
- For login, use the account and missing-information workflow above: ask for missing information, fill supplied values, and verify the resulting account. Keep the session open while waiting for the user.
- For CAPTCHA or another step requiring direct participation, make the handoff actionable: on a desktop, open that page with `--headed --persistent` if the current browser is headless, ask the user to complete that step and reply, and leave the browser open. Leave CAPTCHA controls (including "I'm not a robot") to the user; do not click or solve them. For other access blocks, report the observed blocker. Do not invent results or repeatedly try alternate clients to evade it.
- Close your browser when finished, except while waiting for the user to sign in or when they ask to keep it open. Use `detach` for an externally attached browser.
