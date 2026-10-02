// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using TensorSharp.AgentHost.Skills;

if (args.Contains("--help") || args.Length == 0)
{
    Console.WriteLine("""
        WindowsBrowserSkillProbe --output <fresh evidence directory>
          [--sandbox preferred|off] [--script scripts/playwright_cli.mjs]
          [--iterations 3] [--idle-seconds 3] [--help-only] [--seed-npm-cache <directory>]
          [--debug-browser] [--timeout-seconds 120] [--short-profile]

        Run from the repository with native Node + npm on PATH and Chrome installed.
        Launches headed browsers through the real SkillScriptRunner, verifies visible
        Windows desktop windows after each call and after an idle user-handoff interval,
        signs into a synthetic local fixture, reuses its browser session, then closes it.
        --help-only runs the selected wrapper's help without starting a browser.
        --seed-npm-cache copies an existing package cache into this fresh workspace.
        --debug-browser retains synthetic browser protocol diagnostics in daemon-logs/.
        --short-profile uses a short explicit profile inside this probe's workspace.
        Evidence must be under ignored artifacts/ or docs/validation/.
        No real account, external website, LLM, CUDA, or other OS is tested here.
        """);
    return 0;
}

if (!OperatingSystem.IsWindows())
{
    Console.Error.WriteLine("This probe verifies native Windows desktop windows.");
    return 2;
}

string? outputArgument = null, npmCache = null;
string mode = "preferred", script = "scripts/playwright_cli.mjs";
int iterations = 3, idleSeconds = 3;
bool helpOnly = false, debugBrowser = false, shortProfile = false;
int timeoutSeconds = 120;
try
{
    for (int i = 0; i < args.Length; i++)
        switch (args[i])
        {
            case "--output" when i + 1 < args.Length: outputArgument = args[++i]; break;
            case "--sandbox" when i + 1 < args.Length: mode = args[++i]; break;
            case "--script" when i + 1 < args.Length: script = args[++i]; break;
            case "--seed-npm-cache" when i + 1 < args.Length: npmCache = Path.GetFullPath(args[++i]); break;
            case "--iterations" when i + 1 < args.Length: iterations = int.Parse(args[++i]); break;
            case "--idle-seconds" when i + 1 < args.Length: idleSeconds = int.Parse(args[++i]); break;
            case "--help-only": helpOnly = true; break;
            case "--debug-browser": debugBrowser = true; break;
            case "--short-profile": shortProfile = true; break;
            case "--timeout-seconds" when i + 1 < args.Length: timeoutSeconds = int.Parse(args[++i]); break;
            default: throw new ArgumentException("Unknown or incomplete option: " + args[i]);
        }
    if (mode is not ("preferred" or "off") || string.IsNullOrWhiteSpace(outputArgument)
        || iterations is < 1 or > 20 || idleSeconds is < 1 or > 60 || timeoutSeconds is < 1 or > 600)
        throw new ArgumentException("Specify a fresh output directory, preferred/off sandbox, 1-20 iterations and 1-60 idle seconds.");
}
catch (Exception ex) when (ex is ArgumentException or FormatException or OverflowException)
{
    Console.Error.WriteLine(ex.Message);
    return 2;
}

string repo = Directory.GetCurrentDirectory();
while (!Directory.Exists(Path.Combine(repo, "TensorSharp.AgentHost")))
    repo = Path.GetDirectoryName(repo) ?? throw new InvalidOperationException("Run from the TensorSharp repository.");
string output = Path.GetFullPath(outputArgument);
if (!new[] { "artifacts", Path.Combine("docs", "validation") }.Any(root =>
        output.StartsWith(Path.Combine(repo, root) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
    || Directory.Exists(output) || File.Exists(output))
{
    Console.Error.WriteLine("Output must be a new directory under repository artifacts/ or docs/validation/.");
    return 2;
}
Directory.CreateDirectory(output);
var json = new JsonSerializerOptions { WriteIndented = true };
// Chrome's profile is nested under the workspace. Keep the synthetic session name
// short so this probe does not manufacture a Windows MAX_PATH profile failure.
string sessionRoot = "wv-" + Guid.NewGuid().ToString("N")[..8];
var workspaces = new SessionWorkspaceManager(Path.Combine(output, "scratch"));
SessionWorkspace workspace = workspaces.GetOrCreate(sessionRoot);
if (npmCache != null)
{
    string target = Path.Combine(workspace.WorkDirectory, ".home", "AppData", "Local", "npm-cache");
    foreach (string source in Directory.EnumerateFiles(npmCache, "*", SearchOption.AllDirectories))
    {
        string destination = Path.Combine(target, Path.GetRelativePath(npmCache, source));
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        File.Copy(source, destination);
    }
}
var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { Path.Combine(repo, "TensorAgent", "skills", "playwright") } });
if (!registry.TryGet("playwright", out Skill skill))
    throw new InvalidOperationException("The repository Playwright skill did not load.");
var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
{
    Sandbox = mode == "off" ? SkillSandboxMode.Off : SkillSandboxMode.Preferred,
    AllowNetwork = true,
    Timeout = TimeSpan.FromSeconds(timeoutSeconds),
    MaxOutputBytes = 24 * 1024,
    Workspace = workspace,
    MaxAutoInstallAttempts = 0,
    EnvironmentVariables = debugBrowser ? new() { ["DEBUG"] = "pw:browser,pw:protocol" } : new(),
});

var commands = new List<CommandEvidence>();
var checks = new List<object>();
var requests = new List<object>();
var failures = new List<string>();
var launchedSessions = new HashSet<string>();
var ownedBrowsers = new Dictionary<int, DateTime>();
string title = "TensorSharp Windows browser probe " + Guid.NewGuid().ToString("N");
string credential = "synthetic-" + Guid.NewGuid().ToString("N");
string cookie = Guid.NewGuid().ToString("N");
int submissions = 0, authenticatedVisits = 0;
using var stopping = new CancellationTokenSource();
using HttpListener fixture = StartFixture(out int port);
string fixtureUrl = $"http://127.0.0.1:{port}/";
string page = $$"""
    <!doctype html><html lang="en"><meta charset="utf-8"><title>{{title}}</title>
    <style>body{font:24px system-ui;margin:60px;background:#ecf4ff;color:#102443}input,button{font:inherit;margin:8px;padding:8px}</style>
    <h1>TensorSharp Windows browser visibility test</h1><p>Temporary local test. No real account is involved.</p>
    <form method="post" action="/login"><label>Synthetic credential <input name="credential"></label><button>Sign in</button></form>
    """;
Task fixtureTask = Task.Run(async () =>
{
    try
    {
        while (!stopping.IsCancellationRequested)
        {
            HttpListenerContext context = await fixture.GetContextAsync().WaitAsync(stopping.Token);
            string body = "";
            if (context.Request.HttpMethod == "POST")
                body = await new StreamReader(context.Request.InputStream).ReadToEndAsync(stopping.Token);
            bool loginAccepted = context.Request.RawUrl == "/login" && body == "credential=" + credential;
            bool authenticated = loginAccepted || context.Request.Cookies["synthetic_session"]?.Value == cookie;
            if (loginAccepted)
            {
                Interlocked.Increment(ref submissions);
                context.Response.SetCookie(new Cookie("synthetic_session", cookie, "/") { HttpOnly = true });
            }
            if (authenticated && context.Request.RawUrl == "/account") Interlocked.Increment(ref authenticatedVisits);
            lock (requests) requests.Add(new { utc = DateTime.UtcNow, method = context.Request.HttpMethod,
                path = context.Request.RawUrl, context.Request.UserAgent, authenticated });
            string content = authenticated
                ? $"<!doctype html><title>{title}</title><h1 id=authenticated>Signed in to synthetic account</h1>"
                : page;
            byte[] bytes = Encoding.UTF8.GetBytes(content);
            context.Response.ContentType = "text/html; charset=utf-8";
            context.Response.Headers["Cache-Control"] = "no-store";
            context.Response.ContentLength64 = bytes.Length;
            await context.Response.OutputStream.WriteAsync(bytes);
            context.Response.Close();
        }
    }
    catch (Exception ex) when (ex is HttpListenerException or ObjectDisposedException or OperationCanceledException) { }
});

File.WriteAllText(Path.Combine(output, "identity.json"), JsonSerializer.Serialize(new
{
    sessionRoot, title, script, mode, iterations, idleSeconds, helpOnly, fixtureUrl, npmCache,
    debugBrowser, timeoutSeconds, shortProfile,
    workspace.WorkDirectory, sandbox = runner.Sandbox?.Name ?? "none",
    hostPid = Environment.ProcessId, hostSessionId = Process.GetCurrentProcess().SessionId,
    interactive = Environment.UserInteractive,
    os = RuntimeInformation.OSDescription, framework = RuntimeInformation.FrameworkDescription,
    architecture = RuntimeInformation.OSArchitecture.ToString(), logicalProcessors = Environment.ProcessorCount,
    limitations = "Local synthetic fixture and native Windows desktop only; no real accounts, LLM, CUDA or cross-platform benchmark.",
}, json));

SkillToolResult Run(string session, string phase, params string[] commandArgs)
{
    string[] argv = new[] { "--session", session }.Concat(commandArgs).ToArray();
    var clock = Stopwatch.StartNew();
    SkillToolResult result = runner.Run(skill, script, argv);
    clock.Stop();
    var evidence = new CommandEvidence(session, phase, clock.Elapsed.TotalMilliseconds, result.Ok, argv, result.Content);
    commands.Add(evidence);
    File.WriteAllText(Path.Combine(output, $"{commands.Count:D2}-{phase}.json"), JsonSerializer.Serialize(evidence, json));
    Console.WriteLine($"{phase}: ok={result.Ok}, {clock.Elapsed.TotalMilliseconds:F1} ms");
    if (!result.Ok || result.Content.Contains("### Error", StringComparison.Ordinal))
        throw new InvalidOperationException($"{phase} failed: {result.Content}");
    return result;
}

async Task<List<WindowEvidence>> RequireVisible(string phase, int? expectedPid = null)
{
    var wait = Stopwatch.StartNew();
    List<WindowEvidence> windows;
    do
    {
        windows = DesktopWindows.Find(title);
        if (windows.Any(w => w.IsUsable && (expectedPid == null || w.ProcessId == expectedPid)))
            break;
        await Task.Delay(100);
    } while (wait.Elapsed < TimeSpan.FromSeconds(10));
    checks.Add(new { phase, elapsedMs = wait.Elapsed.TotalMilliseconds, expectedPid, windows });
    if (!windows.Any(w => w.IsUsable && (expectedPid == null || w.ProcessId == expectedPid)))
        throw new InvalidOperationException($"{phase}: no visible desktop browser window for the unique fixture title.");
    foreach (WindowEvidence window in windows.Where(w => w.IsUsable))
    {
        using Process process = Process.GetProcessById(window.ProcessId);
        ownedBrowsers[window.ProcessId] = process.StartTime.ToUniversalTime();
    }
    return windows;
}

try
{
    if (!runner.CanRun) throw new InvalidOperationException(runner.UnavailableReason);
    if (helpOnly)
        Run(sessionRoot, "help", "--help");
    else
    {
        for (int i = 0; i < iterations; i++)
        {
            string session = sessionRoot + "-" + i;
            string[] openArgs = shortProfile
                ? new[] { "open", fixtureUrl, "--headed", "--persistent", "--profile", Path.Combine(workspace.WorkDirectory, ".pw", i.ToString()) }
                : new[] { "open", fixtureUrl, "--headed", "--persistent" };
            // A failed open can still leave a daemon or blank/error-title browser.
            // Always attempt session-scoped close, including partial startup failures.
            launchedSessions.Add(session);
            SkillToolResult open = Run(session, "open", openArgs);
            if (!open.Content.Contains("opened with pid", StringComparison.Ordinal))
                throw new InvalidOperationException("CLI did not confirm a browser launch.");
            int pid = (await RequireVisible("after-open")).First(w => w.IsUsable).ProcessId;
            await Task.Delay(TimeSpan.FromSeconds(idleSeconds));
            await RequireVisible("after-idle-handoff", pid);
            Run(session, "login", "run-code", "async page => { await page.locator('input[name=credential]').fill('" + credential
                + "'); await page.getByRole('button', { name: 'Sign in' }).click(); return await page.locator('#authenticated').innerText(); }");
            if (Volatile.Read(ref submissions) != i + 1)
                throw new InvalidOperationException("The independent fixture did not observe exactly one authenticated form submission.");
            await RequireVisible("after-login", pid);
            Run(session, "account-navigation", "goto", fixtureUrl + "account");
            if (Volatile.Read(ref authenticatedVisits) != i + 1)
                throw new InvalidOperationException("The independent fixture did not receive the session cookie on a later navigation call.");
            for (int action = 0; action < 3; action++)
            {
                SkillToolResult result = Run(session, "reuse", "run-code", "async page => ({ title: await page.title(), signedIn: await page.locator('#authenticated').count() })");
                const string resultHeader = "### Result\n";
                int resultStart = result.Content.IndexOf(resultHeader, StringComparison.Ordinal);
                if (resultStart < 0) throw new InvalidOperationException("CLI did not return the requested browser state.");
                string resultJson = result.Content[(resultStart + resultHeader.Length)..].Split("\n###", 2)[0].Trim();
                using JsonDocument state = JsonDocument.Parse(resultJson);
                if (state.RootElement.GetProperty("title").GetString() != title
                    || state.RootElement.GetProperty("signedIn").GetInt32() != 1)
                    throw new InvalidOperationException("The reused browser did not retain the authenticated page.");
            }
            await RequireVisible("after-reuse", pid);
            Run(session, "close", "close");
            launchedSessions.Remove(session);
            var closeWait = Stopwatch.StartNew();
            while (DesktopWindows.Find(title).Count > 0 && closeWait.Elapsed < TimeSpan.FromSeconds(10))
                await Task.Delay(100);
            if (DesktopWindows.Find(title).Count != 0)
                throw new InvalidOperationException("Closing this test session left its browser window open.");
            while (IsOwnedBrowserAlive(pid) && closeWait.Elapsed < TimeSpan.FromSeconds(10))
                await Task.Delay(100);
            if (IsOwnedBrowserAlive(pid))
                throw new InvalidOperationException("Closing this test session left its browser process alive.");
        }
    }
}
catch (Exception error)
{
    failures.Add(error.ToString());
    Console.Error.WriteLine(error.Message);
}
finally
{
    foreach (string session in launchedSessions)
        try { Run(session, "cleanup-close", "close"); }
        catch (Exception error) { failures.Add("Cleanup: " + error.Message); }
    // Only terminate processes positively matched to this probe's unique page title.
    // This is a fallback for a broken CLI close, never a global browser cleanup.
    foreach (WindowEvidence window in DesktopWindows.Find(title))
        try { Process.GetProcessById(window.ProcessId).Kill(entireProcessTree: true); }
        catch (ArgumentException) { }
        catch (Exception error) { failures.Add("Window cleanup: " + error.Message); }
    foreach (int pid in ownedBrowsers.Keys.Where(IsOwnedBrowserAlive))
    {
        failures.Add($"Browser process {pid} required fallback cleanup after CLI close.");
        try { Process.GetProcessById(pid).Kill(entireProcessTree: true); }
        catch (ArgumentException) { }
        catch (Exception error) { failures.Add("Process cleanup: " + error.Message); }
    }
    try
    {
        stopping.Cancel();
        fixture.Stop();
        await fixtureTask;
    }
    catch (Exception error) { failures.Add("Fixture cleanup: " + error); }
    // Releasing exercises the host's session cleanup once all CLI sessions are closed.
    try
    {
        foreach (string source in Directory.EnumerateFiles(workspace.WorkDirectory, "*.err", SearchOption.AllDirectories))
        {
            string destination = Path.Combine(output, "daemon-logs", Path.GetFileName(source));
            Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
            File.Copy(source, destination, true);
        }
    }
    catch (Exception error) { failures.Add("Log capture: " + error); }
    try { workspaces.Release(sessionRoot); }
    catch (Exception error) { failures.Add("Workspace cleanup: " + error); }
}

var report = new
{
    status = failures.Count == 0 ? "ok" : "fail", script, mode, iterations, idleSeconds, helpOnly,
    submissions, authenticatedVisits, commands, desktopChecks = checks, fixtureRequests = requests, failures,
    timings = commands.GroupBy(c => c.Phase).ToDictionary(g => g.Key, g => new
    {
        count = g.Count(), medianMs = Percentile(g.Select(c => c.ElapsedMs), 0.5),
        p95Ms = Percentile(g.Select(c => c.ElapsedMs), 0.95),
    }),
    firstOpenMs = commands.FirstOrDefault(c => c.Phase == "open")?.ElapsedMs,
    warmOpenMedianMs = commands.Count(c => c.Phase == "open") > 1
        ? (double?)Percentile(commands.Where(c => c.Phase == "open").Skip(1).Select(c => c.ElapsedMs), 0.5) : null,
    limitations = (npmCache == null ? "First open may include npm package acquisition. "
        : "Npm package cache was pre-seeded; dependency download is excluded. ")
        + "Later opens reuse this workspace's package cache. Small local sample, no throughput claim. "
        + "Window visibility is verified independently of CLI text. No live LLM/account/CUDA validation.",
};
File.WriteAllText(Path.Combine(output, "report.json"), JsonSerializer.Serialize(report, json));
Console.WriteLine(JsonSerializer.Serialize(new { report.status, output, submissions, report.timings }, json));
return failures.Count == 0 ? 0 : 1;

bool IsOwnedBrowserAlive(int pid)
{
    try
    {
        using Process process = Process.GetProcessById(pid);
        return !process.HasExited && ownedBrowsers.TryGetValue(pid, out DateTime started)
            && process.StartTime.ToUniversalTime() == started;
    }
    catch (ArgumentException) { return false; }
    catch (InvalidOperationException) { return false; }
}

static double Percentile(IEnumerable<double> values, double fraction)
{
    double[] ordered = values.Order().ToArray();
    if (fraction == 0.5)
        return (ordered[(ordered.Length - 1) / 2] + ordered[ordered.Length / 2]) / 2;
    return ordered[Math.Clamp((int)Math.Ceiling(fraction * ordered.Length) - 1, 0, ordered.Length - 1)];
}

static HttpListener StartFixture(out int port)
{
    for (int attempt = 0; attempt < 5; attempt++)
    {
        using var reservation = new TcpListener(IPAddress.Loopback, 0);
        reservation.Start();
        port = ((IPEndPoint)reservation.LocalEndpoint).Port;
        reservation.Stop();
        var listener = new HttpListener();
        listener.Prefixes.Add($"http://127.0.0.1:{port}/");
        try { listener.Start(); return listener; }
        catch (HttpListenerException) { listener.Close(); }
    }
    throw new IOException("Could not bind the local synthetic login fixture.");
}

sealed record CommandEvidence(string Session, string Phase, double ElapsedMs, bool Ok, string[] Arguments, string Content);
sealed record WindowEvidence(long Handle, int ProcessId, string ProcessName, int SessionId, string Title,
    bool Visible, bool Minimized, bool Cloaked, bool OnScreen, int Width, int Height)
{
    public bool IsUsable => Visible && !Minimized && !Cloaked && OnScreen && Width > 100 && Height > 100
        && ProcessName is "chrome" or "chromium" or "msedge"
        && SessionId == Process.GetCurrentProcess().SessionId;
}

static class DesktopWindows
{
    public static List<WindowEvidence> Find(string title)
    {
        var windows = new List<WindowEvidence>();
        EnumWindows((handle, _) =>
        {
            var text = new StringBuilder(1024);
            GetWindowTextW(handle, text, text.Capacity);
            if (!text.ToString().Contains(title, StringComparison.Ordinal)) return true;
            GetWindowThreadProcessId(handle, out uint pid);
            string processName;
            int sessionId;
            try
            {
                using Process process = Process.GetProcessById((int)pid);
                processName = process.ProcessName;
                sessionId = process.SessionId;
            }
            catch (Exception error) when (error is ArgumentException or InvalidOperationException or System.ComponentModel.Win32Exception)
            {
                // The window's process can exit between EnumWindows and this query.
                return true;
            }
            GetWindowRect(handle, out Rect rect);
            int cloaked = 0;
            int result = DwmGetWindowAttribute(handle, 14, out cloaked, sizeof(int));
            int left = GetSystemMetrics(76), top = GetSystemMetrics(77);
            bool onScreen = rect.Right > left && rect.Bottom > top
                && rect.Left < left + GetSystemMetrics(78) && rect.Top < top + GetSystemMetrics(79);
            windows.Add(new WindowEvidence(handle.ToInt64(), (int)pid, processName, sessionId, text.ToString(),
                IsWindowVisible(handle), IsIconic(handle), result == 0 && cloaked != 0, onScreen,
                rect.Right - rect.Left, rect.Bottom - rect.Top));
            return true;
        }, IntPtr.Zero);
        return windows;
    }

    private delegate bool EnumWindowsCallback(IntPtr handle, IntPtr parameter);
    [StructLayout(LayoutKind.Sequential)] private struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool EnumWindows(EnumWindowsCallback callback, IntPtr parameter);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] private static extern int GetWindowTextW(IntPtr handle, StringBuilder text, int count);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr handle, out uint pid);
    [DllImport("user32.dll")] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool IsWindowVisible(IntPtr handle);
    [DllImport("user32.dll")] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool IsIconic(IntPtr handle);
    [DllImport("user32.dll")] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool GetWindowRect(IntPtr handle, out Rect rect);
    [DllImport("user32.dll")] private static extern int GetSystemMetrics(int index);
    [DllImport("dwmapi.dll")] private static extern int DwmGetWindowAttribute(IntPtr handle, uint attribute, out int value, int size);
}
