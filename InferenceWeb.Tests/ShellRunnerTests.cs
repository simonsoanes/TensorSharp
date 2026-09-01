// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.IO;
using System.Threading;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// The shell tool end to end: real commands, a real workspace, real files on disk.
///
/// <para>
/// The five program-shaped tools this replaced could be tested almost entirely without
/// running anything, because the host built the command line and every interesting
/// decision was made before a process started. A shell inverts that. The model types the
/// command line now, so the properties worth defending are ones that only exist once
/// something has actually run: that the working directory a <c>cd</c> chose is still
/// there on the next call, that an <c>export</c> survives it, that a per-call
/// <c>workdir</c> does NOT survive it, and that a patch aimed at <c>main.c</c> after a
/// <c>cd build</c> lands on <c>build/main.c</c>. Each of those is a file the wrapper
/// script writes and re-reads between two separate confined processes, and none of them
/// can be checked by inspecting a request.
/// </para>
/// <para>
/// Two of these are recorded regressions rather than hypotheses. A per-call
/// <c>workdir</c> used to move the session permanently, so one call asking to run in
/// <c>build/</c> silently changed what every later relative path meant. And a timeout
/// used to come back empty, which forces a model to re-run an expensive command blind
/// when the output it already had would have told it what it was stuck on.
/// </para>
/// <para>
/// Sandboxing is OFF here: confinement has its own tests, and these are about what the
/// runner and the session wrapper do. The commands are POSIX, so a host without a POSIX
/// shell gates out of the behavioural half — but every gated test also asserts something
/// that needs no shell at all, so a green run on such a host still means something.
/// </para>
/// </summary>
public class ShellRunnerTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;

    public ShellRunnerTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-shellrun-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    /// <summary>Enabled, unsandboxed, and rooted in this test's own temp directory.</summary>
    private CodeExecOptions Options(Action<CodeExecOptions>? tweak = null)
    {
        var options = new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
        };
        tweak?.Invoke(options);
        return options;
    }

    private SessionWorkspace Workspace(string id = "shellrun") => _workspaces.GetOrCreate(id);

    /// <summary>
    /// Every behavioural test below types POSIX. A host whose shell is PowerShell can
    /// still run the unconditional half of each one.
    /// </summary>
    private static bool HavePosixShell =>
        ShellProgram.TryResolve(null, out ShellProgram? shell, out _) && shell is { Kind: ShellKind.Posix };

    private static bool HavePython =>
        CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out _, out _);

    /// <summary>
    /// The wrapper scripts written so far. Empty means literally nothing was executed,
    /// which is the only way to prove a refusal happened BEFORE the shell was reached.
    /// </summary>
    private static string[] ScriptsWritten(SessionWorkspace workspace) =>
        Directory.GetFiles(workspace.StateDirectory, "cmd-*");

    /// <summary>Read a file another process may still be writing into.</summary>
    private static string ReadWhileOpen(string path)
    {
        if (!File.Exists(path))
            return string.Empty;
        try
        {
            using var stream = new FileStream(
                path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }
        catch (IOException)
        {
            return string.Empty;
        }
    }

    private static int LineCount(string text) =>
        text.Split('\n', StringSplitOptions.RemoveEmptyEntries).Length;

    // ---- the switch, and what happens without a shell ----------------------

    [Fact]
    public void EitherThisHostHasAShell_OrEveryCommandIsRefusedWithTheReason()
    {
        // Unconditional on purpose: it is the assertion that keeps the gates below
        // honest. On a host with a shell the tool is live; on one without, the refusal
        // has to name the problem rather than looking like a command that printed nothing.
        using var runner = new ShellRunner(Options());
        CodeExecResult result = runner.Run(new ShellRequest("echo probe"), Workspace());

        if (runner.Shell != null)
        {
            Assert.True(runner.CanRun, runner.UnavailableReason);
            Assert.Null(runner.UnavailableReason);
        }
        else
        {
            Assert.False(result.Ok);
            Assert.NotNull(runner.UnavailableReason);
            Assert.Contains("The command was not run:", result.Content, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void WithoutTheOperatorsFlag_NothingRuns_AndTheFlagIsNamed()
    {
        using var runner = new ShellRunner(new CodeExecOptions { Sandbox = SkillSandboxMode.Off });

        Assert.False(runner.CanRun);
        CodeExecResult result = runner.Run(new ShellRequest("echo hi"), Workspace());

        Assert.False(result.Ok);
        Assert.Contains(CodeExecOptions.EnabledFlag, result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEmptyCommand_IsRefusedWithAnExampleRatherThanRunningAnEmptyScript()
    {
        if (!HavePosixShell) return;

        using var runner = new ShellRunner(Options());
        SessionWorkspace workspace = Workspace();
        CodeExecResult result = runner.Run(new ShellRequest("   "), workspace);

        Assert.False(result.Ok);
        Assert.Contains("'command' argument was empty", result.Content, StringComparison.Ordinal);
        Assert.Empty(ScriptsWritten(workspace));
    }

    // ---- output, exit codes and the two streams ----------------------------

    [Fact]
    public void ACommandRuns_AndItsOutputAndItsExitCodeBothComeBack()
    {
        if (!HavePosixShell) return;

        using var runner = new ShellRunner(Options());
        CodeExecResult result = runner.Run(new ShellRequest("echo hello-from-the-shell"), Workspace());

        Assert.True(result.Ok, result.Content);
        Assert.Contains("hello-from-the-shell", result.Content, StringComparison.Ordinal);
        Assert.Contains("exit 0", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ANonZeroExit_IsNotOk_ButStillCarriesEverythingTheCommandPrinted()
    {
        if (!HavePosixShell) return;

        // A failure whose output is replaced by an error summary is a failure the model
        // cannot fix: the printed part is the evidence it needs to write the next command.
        using var runner = new ShellRunner(Options());
        CodeExecResult result = runner.Run(
            new ShellRequest("echo printed-before-failing; exit 3"), Workspace());

        Assert.False(result.Ok);
        Assert.Contains("exit 3", result.Content, StringComparison.Ordinal);
        Assert.Contains("printed-before-failing", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void StdoutAndStderrAreMergedIntoOneStream_StdoutFirst()
    {
        if (!HavePosixShell) return;

        // Unlabelled and in terminal order, because that is what every example the model
        // has ever seen looks like — a run whose useful output is a traceback must not
        // read as though something separate went wrong.
        using var runner = new ShellRunner(Options());
        CodeExecResult result = runner.Run(
            new ShellRequest("echo alpha-on-stdout; echo beta-on-stderr >&2"), Workspace());

        int stdout = result.Content.IndexOf("alpha-on-stdout", StringComparison.Ordinal);
        int stderr = result.Content.IndexOf("beta-on-stderr", StringComparison.Ordinal);
        Assert.InRange(stdout, 0, int.MaxValue);
        Assert.InRange(stderr, stdout + 1, int.MaxValue);
    }

    // ---- what persists between calls ---------------------------------------

    [Fact]
    public void AFileWrittenWithAHeredoc_IsStillThereOnALaterCall()
    {
        if (!HavePosixShell) return;

        // The workspace is the session's, not the call's. A heredoc is also how a patch
        // envelope reaches an intercepted apply_patch, so this shape has to work.
        using var runner = new ShellRunner(Options());
        SessionWorkspace workspace = Workspace();

        CodeExecResult wrote = runner.Run(
            new ShellRequest("cat > note.txt <<'EOF'\npersisted-body\nEOF"), workspace);
        Assert.True(wrote.Ok, wrote.Content);

        CodeExecResult read = runner.Run(new ShellRequest("cat note.txt"), workspace);
        Assert.True(read.Ok, read.Content);
        Assert.Contains("persisted-body", read.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ACdMovesTheSessionForEveryLaterCall()
    {
        if (!HavePosixShell) return;

        // There is no long-lived shell process; the directory survives only because the
        // wrapper's EXIT trap wrote it to a file the next wrapper reads back.
        using var runner = new ShellRunner(Options());
        SessionWorkspace workspace = Workspace();

        CodeExecResult moved = runner.Run(
            new ShellRequest("mkdir -p deep-probe && cd deep-probe"), workspace);
        Assert.True(moved.Ok, moved.Content);
        Assert.Contains("Working directory is now deep-probe", moved.Content, StringComparison.Ordinal);

        CodeExecResult where = runner.Run(new ShellRequest("pwd"), workspace);
        Assert.Contains("/deep-probe", where.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AnExportedVariableSurvivesIntoTheNextCall()
    {
        if (!HavePosixShell) return;

        using var runner = new ShellRunner(Options());
        SessionWorkspace workspace = Workspace();

        CodeExecResult exported = runner.Run(
            new ShellRequest("export PROBE_COLOUR=tangerine"), workspace);
        Assert.True(exported.Ok, exported.Content);

        CodeExecResult echoed = runner.Run(new ShellRequest("echo \"[$PROBE_COLOUR]\""), workspace);
        Assert.Contains("[tangerine]", echoed.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void APerCallWorkdir_RunsThere_ButDoesNotMoveTheSession()
    {
        if (!HavePosixShell) return;

        // The regression this exists for: one call asking to run in a subdirectory used
        // to move the conversation permanently, and every later relative path — including
        // every path in a patch — quietly meant somewhere else.
        using var runner = new ShellRunner(Options());
        SessionWorkspace workspace = Workspace();

        CodeExecResult made = runner.Run(new ShellRequest("mkdir -p sub-probe"), workspace);
        Assert.True(made.Ok, made.Content);

        CodeExecResult inside = runner.Run(
            new ShellRequest("pwd") { WorkDirectory = "sub-probe" }, workspace);
        Assert.True(inside.Ok, inside.Content);
        Assert.Contains("/sub-probe", inside.Content, StringComparison.Ordinal);

        CodeExecResult after = runner.Run(new ShellRequest("pwd"), workspace);
        Assert.True(after.Ok, after.Content);
        Assert.DoesNotContain("/sub-probe", after.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AWorkdirThatDoesNotExist_IsRefusedBeforeAnythingRuns_AndNamesWhatIsThere()
    {
        if (!HavePosixShell) return;

        // A refusal that only says the path is wrong gets the same wrong path back next
        // round. Listing the directory is what turns it into a correction.
        using var runner = new ShellRunner(Options());
        SessionWorkspace workspace = Workspace();
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "marker.txt"), "x");

        CodeExecResult result = runner.Run(
            new ShellRequest("echo should-not-run") { WorkDirectory = "no-such-dir" }, workspace);

        Assert.False(result.Ok);
        Assert.Contains("is not a directory", result.Content, StringComparison.Ordinal);
        Assert.Contains("marker.txt", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("should-not-run", result.Content, StringComparison.Ordinal);
        Assert.Empty(ScriptsWritten(workspace));
    }

    // ---- deadlines and output size -----------------------------------------

    [Fact]
    public void ACommandThatOverrunsItsDeadline_IsStopped_AndKeepsWhatItAlreadyPrinted()
    {
        if (!HavePosixShell) return;

        // A timeout that returns nothing forces the model to re-run an expensive command
        // blind, when the part it already printed usually says what it was stuck on.
        using var runner = new ShellRunner(Options());
        CodeExecResult result = runner.Run(
            new ShellRequest("echo printed-before-the-wait; sleep 20") { Timeout = TimeSpan.FromSeconds(2) },
            Workspace());

        Assert.False(result.Ok);
        Assert.Contains("did not finish within", result.Content, StringComparison.Ordinal);
        Assert.Contains("printed-before-the-wait", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void LongOutputIsTruncatedFromTheMiddle_KeepingBothEnds_AndSaysHowMuchWentMissing()
    {
        if (!HavePosixShell) return;

        // Head-only truncation, which this used to do, reliably discards exactly the part
        // that was wanted: the head of a build is the command echoing, and the failure is
        // at the end. Both ends have to survive for a truncated result to answer
        // "did it work".
        using var runner = new ShellRunner(Options(o => o.MaxOutputBytes = 2048));
        CodeExecResult result = runner.Run(
            new ShellRequest("i=1; while [ $i -le 800 ]; do echo line-$i; i=$((i+1)); done"),
            Workspace());

        Assert.True(result.Ok, result.Content);
        Assert.Contains("line-1\n", result.Content, StringComparison.Ordinal);
        Assert.Contains("line-800\n", result.Content, StringComparison.Ordinal);
        Assert.Contains("of output were dropped from the middle", result.Content, StringComparison.Ordinal);
    }

    // ---- apply_patch typed into the shell ----------------------------------

    [Fact]
    public void ApplyPatchTypedAsAHeredoc_IsAnsweredByTheHost_AndNothingIsExecuted()
    {
        if (!HavePosixShell) return;

        // Codex answers apply_patch in the harness rather than executing it, and here
        // that is not only fidelity: there is no apply_patch binary in this workspace and
        // no interpreter guaranteed to be present to implement one. The proof that
        // nothing ran is that no wrapper script was ever written.
        using var runner = new ShellRunner(Options());
        SessionWorkspace workspace = Workspace();
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "greet.txt"), "hello\n");

        CodeExecResult result = runner.Run(new ShellRequest(
            SkillToolNames.ApplyPatch + " <<'PATCH'\n"
            + "*** Begin Patch\n"
            + "*** Update File: greet.txt\n"
            + "-hello\n"
            + "+hola\n"
            + "*** End Patch\n"
            + "PATCH"), workspace);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("Applied the patch to 1 file", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("command not found", result.Content, StringComparison.Ordinal);
        Assert.Empty(ScriptsWritten(workspace));

        string patched = File.ReadAllText(Path.Combine(workspace.WorkDirectory, "greet.txt"));
        Assert.Contains("hola", patched, StringComparison.Ordinal);
        Assert.DoesNotContain("hello", patched, StringComparison.Ordinal);
    }

    [Fact]
    public void APatchWhoseAnchorIsNotInTheFile_LeavesItByteIdentical()
    {
        if (!HavePosixShell) return;

        // All-or-nothing is the property that makes a multi-file patch safe to offer, and
        // a single file is where it is cheapest to check: a hunk that did not resolve must
        // not have written a partial file first.
        using var runner = new ShellRunner(Options());
        SessionWorkspace workspace = Workspace();
        string path = Path.Combine(workspace.WorkDirectory, "data.txt");
        byte[] original = System.Text.Encoding.UTF8.GetBytes("alpha\nbeta\n");
        File.WriteAllBytes(path, original);

        CodeExecResult result = runner.Run(new ShellRequest(
            SkillToolNames.ApplyPatch + " <<'PATCH'\n"
            + "*** Begin Patch\n"
            + "*** Update File: data.txt\n"
            + "-gamma\n"
            + "+delta\n"
            + "*** End Patch\n"
            + "PATCH"), workspace);

        Assert.False(result.Ok);
        Assert.Contains("did not match", result.Content, StringComparison.Ordinal);
        Assert.Equal(original, File.ReadAllBytes(path));
    }

    [Fact]
    public void PatchPathsResolveFromWhereTheShellIs_NotFromTheWorkDirectory()
    {
        if (!HavePosixShell) return;

        // The two halves of the tool surface have to agree about what "note.txt" means.
        // A model that cd'd into sub/ and then patches note.txt means sub/note.txt;
        // resolving from the work directory instead silently patched the wrong file, and
        // the decoy here is that wrong file.
        using var runner = new ShellRunner(Options());
        SessionWorkspace workspace = Workspace();

        CodeExecResult setup = runner.Run(new ShellRequest(
            "mkdir -p sub && printf 'decoy\\n' > note.txt && printf 'hello\\n' > sub/note.txt && cd sub"),
            workspace);
        Assert.True(setup.Ok, setup.Content);

        CodeExecResult result = runner.Run(new ShellRequest(
            SkillToolNames.ApplyPatch + " <<'PATCH'\n"
            + "*** Begin Patch\n"
            + "*** Update File: note.txt\n"
            + "-hello\n"
            + "+hola\n"
            + "*** End Patch\n"
            + "PATCH"), workspace);

        Assert.True(result.Ok, result.Content);
        string inner = File.ReadAllText(Path.Combine(workspace.WorkDirectory, "sub", "note.txt"));
        string outer = File.ReadAllText(Path.Combine(workspace.WorkDirectory, "note.txt"));
        Assert.Contains("hola", inner, StringComparison.Ordinal);
        Assert.Contains("decoy", outer, StringComparison.Ordinal);
        Assert.DoesNotContain("hola", outer, StringComparison.Ordinal);
    }

    // ---- what the host classifies as an install ----------------------------

    [Fact]
    public void AnInstallIsRefusedWhenInstallingIsOff_AndAPlainCommandStillRuns()
    {
        // The classification is the enforceable half and needs no shell to check: a line
        // that names a package manager is an install whatever host this is.
        Assert.True(ShellCommand.ContainsInstall("pip install rich"));
        Assert.False(ShellCommand.ContainsInstall("echo plain-command-ran"));

        if (!HavePosixShell) return;

        using var runner = new ShellRunner(Options(o => o.AllowInstall = false));
        SessionWorkspace workspace = Workspace();

        CodeExecResult refused = runner.Run(new ShellRequest("pip install rich"), workspace);
        Assert.False(refused.Ok);
        Assert.Contains(CodeExecOptions.AllowInstallFlag, refused.Content, StringComparison.Ordinal);
        // Refused before the shell was reached, not after pip failed to find a network.
        Assert.Empty(ScriptsWritten(workspace));

        CodeExecResult plain = runner.Run(new ShellRequest("echo plain-command-ran"), workspace);
        Assert.True(plain.Ok, plain.Content);
        Assert.Contains("plain-command-ran", plain.Content, StringComparison.Ordinal);
    }

    // ---- coaching ----------------------------------------------------------

    [Fact]
    public void AMissingPythonModule_IsAnsweredWithTheCommandThatInstallsIt()
    {
        // Measured, not hypothesised: handed a bare traceback, a small model runs the
        // identical code again and then tells the user the file exists. The result has to
        // end in an action.
        Assert.True(CodeDiagnostics.TryFindMissingModule(
            "ModuleNotFoundError: No module named 'ts_absent_probe_pkg'",
            out CodeLanguage language, out string module));
        Assert.Equal(CodeLanguage.Python, language);
        Assert.Equal("ts_absent_probe_pkg", module);

        if (!HavePosixShell || !HavePython) return;

        // With installing DISABLED the result names the constraint and stops there: there
        // is no command to give, and offering one would send the model at a wall.
        using (var closed = new ShellRunner(Options(o => o.AllowInstall = false)))
        {
            CodeExecResult refused = closed.Run(
                new ShellRequest("python3 -c 'import ts_absent_probe_pkg'"), Workspace("closed"));

            Assert.False(refused.Ok);
            Assert.Contains("is not installed", refused.Content, StringComparison.Ordinal);
            Assert.Contains("not enabled on this host", refused.Content, StringComparison.Ordinal);
        }

        // With installing ENABLED the host no longer ASKS the model to install and re-run
        // — it does both itself, inside the same call (see ShellRunner.RunWithAutoInstall;
        // "the first run dies on a missing dependency" was 68 rounds and 116 minutes in
        // the measured logs, most of it spent re-typing the program verbatim). Here the
        // package does not exist and there is no network, so what has to be true is that
        // the result says the host tried and could not — never that it installed
        // something, and never "install it and run the command again", which is the thing
        // that just failed.
        using var runner = new ShellRunner(Options(o => o.AllowInstall = true));
        CodeExecResult result = runner.Run(
            new ShellRequest("python3 -c 'import ts_absent_probe_pkg'"), Workspace());

        Assert.False(result.Ok);
        Assert.Contains("is not installed", result.Content, StringComparison.Ordinal);
        Assert.Contains("ts_absent_probe_pkg' was missing", result.Content, StringComparison.Ordinal);
        Assert.Contains("tried to install it for you and could not", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("was installed and the command was run again", result.Content);
    }

    [Fact]
    public void AnUnknownProgram_IsToldThatNoPackageManagerCanSupplyOne()
    {
        // A missing PROGRAM and a missing LIBRARY both arrive as exit 127, and the advice
        // for them is opposite. Told to install a program that pip cannot supply, a model
        // retries the same line forever.
        Assert.Equal(
            "ts-no-such-program-9f3a",
            CodeDiagnostics.MissingCommand("command line 1: ts-no-such-program-9f3a: command not found"));
        Assert.False(CodeDiagnostics.IsPackageManager("ts-no-such-program-9f3a"));

        if (!HavePosixShell) return;

        using var runner = new ShellRunner(Options(o => o.AllowInstall = true));
        CodeExecResult result = runner.Run(new ShellRequest("ts-no-such-program-9f3a"), Workspace());

        Assert.False(result.Ok);
        Assert.Contains("ts-no-such-program-9f3a", result.Content, StringComparison.Ordinal);
        Assert.Contains("no package manager here can supply it", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingPip_IsToldTheSpellingThatDoesExistOnThisHost()
    {
        // Not "this host has no installer" — it has one, spelled differently. Plenty of
        // machines have python3 and no `pip`, and a model that followed an instruction to
        // run `pip install` there has nowhere left to go.
        Assert.True(CodeDiagnostics.IsPackageManager("pip"));
        Assert.NotEmpty(CodeDiagnostics.PythonInstallPrefix());

        if (!HavePosixShell) return;

        // The failure is STAGED rather than provoked: this host may well have a real pip,
        // in which case `pip --version` would simply succeed and the advice would never
        // be reached. What is under test is the coaching, not pip's absence.
        using var runner = new ShellRunner(Options(o => o.AllowInstall = true));
        CodeExecResult result = runner.Run(
            new ShellRequest("echo 'command line 1: pip: command not found' >&2; exit 127"),
            Workspace());

        Assert.False(result.Ok);
        Assert.Contains("is not on this host's PATH under that name", result.Content, StringComparison.Ordinal);
        Assert.Contains(CodeDiagnostics.PythonInstallPrefix(), result.Content, StringComparison.Ordinal);
    }

    // ---- background jobs ---------------------------------------------------

    [Fact]
    public void ABackgroundJob_ReturnsAtOnceWithALog_AndIsKilledWhenTheWorkspaceIsReleased()
    {
        if (!HavePosixShell) return;

        // The ticks file lives OUTSIDE the workspace on purpose: after the workspace is
        // released its own directory is gone, so a job that survived would still be
        // appending here and nowhere else. That is the only observation that separates
        // "killed" from "its output had nowhere left to go".
        using var runner = new ShellRunner(Options());
        SessionWorkspace workspace = Workspace("background");
        string ticks = Path.Combine(_base, "ticks.txt");

        CodeExecResult started = runner.Run(
            new ShellRequest(
                "i=1; while [ $i -le 60 ]; do echo tick-$i; echo tick-$i >> '" + ticks + "'; "
                + "sleep 1; i=$((i+1)); done")
            { Background = true },
            workspace);

        Assert.True(started.Ok, started.Content);
        Assert.Contains("Started in the background as job-1", started.Content, StringComparison.Ordinal);
        Assert.Contains(".jobs/job-1.log", started.Content, StringComparison.Ordinal);

        string log = Path.Combine(workspace.WorkDirectory, ".jobs", "job-1.log");
        for (int waited = 0; waited < 100 && LineCount(ReadWhileOpen(ticks)) < 2; waited++)
            Thread.Sleep(100);

        // The host writes the log from its own output tap, so it fills as the job runs
        // rather than only at exit.
        Assert.True(File.Exists(log), "the background job's log file was never created");
        Assert.Contains("tick-1", ReadWhileOpen(log), StringComparison.Ordinal);
        Assert.InRange(LineCount(ReadWhileOpen(ticks)), 2, int.MaxValue);

        _workspaces.Release("background");
        Thread.Sleep(500);

        int atRelease = LineCount(ReadWhileOpen(ticks));
        Thread.Sleep(2500);
        Assert.Equal(atRelease, LineCount(ReadWhileOpen(ticks)));
    }

    // ---- artifacts ---------------------------------------------------------

    [Fact]
    public void AFileTheCommandWrote_IsKeptAndPointedAt_AndOneItNeverTouchedIsNot()
    {
        if (!HavePosixShell) return;

        // Capture is CHANGE-based. A session's working directory accumulates for the whole
        // conversation, so handing back everything in it would re-offer the user the same
        // files after every command and bury the one thing this call actually produced.
        string artifacts = Path.Combine(_base, "artifacts");
        Directory.CreateDirectory(artifacts);

        var store = new CodeArtifactStore(artifacts);
        using var runner = new ShellRunner(Options(), null, store);
        SessionWorkspace workspace = Workspace();

        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "already-here.txt"), "untouched\n");

        CodeExecResult result = runner.Run(
            new ShellRequest("printf 'produced\\n' > report.csv && echo done"), workspace);

        Assert.True(result.Ok, result.Content);
        CodeArtifact artifact = Assert.Single(result.Artifacts);
        Assert.Equal("report.csv", artifact.Path);

        // With no URI prefix the pointer is the path on disk, and it has to still be there
        // after the call: the artifact is the answer, not a side effect.
        Assert.True(File.Exists(artifact.Pointer), "the kept artifact is not on disk");
        Assert.Contains("Files produced", result.Content, StringComparison.Ordinal);
        Assert.Contains(artifact.Pointer, result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("already-here.txt", result.Content, StringComparison.Ordinal);
    }
}

/// <summary>
/// The same shell with the OS sandbox actually engaged.
///
/// <para>
/// Separate from the rest because what these assert is different in kind: not "does the
/// runner decide correctly" but "does the confinement hold once a general shell is
/// running inside it". Nothing about the sandbox changed when the tool surface did —
/// which is exactly the claim worth re-checking, because a shell reaches far more of the
/// host than a single interpreter invocation ever did.
/// </para>
/// <para>
/// They skip themselves where the platform provides no real confinement: a green run on
/// a host with none proves nothing and must not pretend to. The one unconditional test
/// is the one that matters most on such a host — that asking for a sandbox it cannot
/// provide is a refusal rather than a quietly unconfined run.
/// </para>
/// </summary>
public class ShellRunnerSandboxTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;

    public ShellRunnerSandboxTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-shellbox-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private CodeExecOptions Sandboxed() => new()
    {
        Enabled = true,
        Sandbox = SkillSandboxMode.Required,
        Timeout = TimeSpan.FromSeconds(30),
        ScratchDirectory = _base,
    };

    private SessionWorkspace Workspace(string id = "sandboxed") => _workspaces.GetOrCreate(id);

    private static bool HavePosixShell =>
        ShellProgram.TryResolve(null, out ShellProgram? shell, out _) && shell is { Kind: ShellKind.Posix };

    private static bool HavePython =>
        CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out _, out _);

    [Fact]
    public void ASandboxIsEitherRealOrTheCommandIsRefused()
    {
        // The default must never behave like "preferred". A Windows job object bounds CPU
        // and memory but cannot restrict one file or one socket, and an existence test
        // once made that host run model-written commands with the filesystem open.
        using var runner = new ShellRunner(Sandboxed());

        if (runner.CanRun)
        {
            Assert.NotNull(runner.Sandbox);
            Assert.Null(runner.UnavailableReason);
        }
        else if (runner.Shell == null)
        {
            Assert.NotNull(runner.UnavailableReason);
        }
        else
        {
            Assert.Contains(
                CodeExecOptions.UnconfinedFlag, runner.UnavailableReason!, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void OrdinaryFileWorkStillWorksUnderTheSandbox()
    {
        if (!HavePosixShell) return;

        // A confinement that leaves the feature unusable is just a refusal wearing a
        // different hat: a heredoc, a read-back and a persisted directory all have to
        // survive it.
        using var runner = new ShellRunner(Sandboxed());
        if (!runner.CanRun) return;

        SessionWorkspace workspace = Workspace();
        CodeExecResult wrote = runner.Run(
            new ShellRequest("mkdir -p out && cat > out/scratch.txt <<'EOF'\nsandboxed-body\nEOF"),
            workspace);
        Assert.True(wrote.Ok, wrote.Content);

        CodeExecResult read = runner.Run(new ShellRequest("cat out/scratch.txt"), workspace);
        Assert.True(read.Ok, read.Content);
        Assert.Contains("sandboxed-body", read.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void APlainCommandUnderTheSandboxHasNoSocket()
    {
        if (!HavePosixShell || !HavePython) return;

        using var runner = new ShellRunner(Sandboxed());
        if (!runner.CanRun) return;

        // Written to a file with a heredoc and then run, rather than squeezed through
        // `python3 -c`, so the quoting stays readable and the heredoc path is exercised
        // under confinement too.
        CodeExecResult result = runner.Run(new ShellRequest(
            "cat > probe.py <<'EOF'\n"
            + "import socket\n"
            + "try:\n"
            + "    socket.create_connection(('1.1.1.1', 80), timeout=4)\n"
            + "    print('CONNECTED')\n"
            + "except Exception as e:\n"
            + "    print('DENIED', type(e).__name__)\n"
            + "EOF\n"
            + "python3 probe.py"), Workspace());

        Assert.DoesNotContain("CONNECTED", result.Content, StringComparison.Ordinal);
        Assert.Contains("DENIED", result.Content, StringComparison.Ordinal);
    }

    [Fact]
    public void TheCommandCannotReadTheUsersRealHomeDirectory()
    {
        if (!HavePosixShell) return;

        using var runner = new ShellRunner(Sandboxed());
        if (!runner.CanRun) return;

        // The REAL home, not `~`: HOME is deliberately pointed at the workspace, so `~`
        // would test the sandbox against a path that is supposed to be writable. The
        // witness is a file we put there ourselves, because asserting on ~/.ssh/id_rsa
        // proves nothing on a machine that has no such file.
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (string.IsNullOrEmpty(home)) return;

        string witness = Path.Combine(home, "ts-shell-witness-" + Guid.NewGuid().ToString("N")[..12] + ".txt");
        File.WriteAllText(witness, "WITNESS-SECRET\n");
        try
        {
            CodeExecResult result = runner.Run(
                new ShellRequest("cat '" + witness + "' 2>&1 | head -2"), Workspace());

            Assert.DoesNotContain("WITNESS-SECRET", result.Content, StringComparison.Ordinal);
        }
        finally
        {
            try { File.Delete(witness); } catch (IOException) { /* best effort */ }
        }
    }
}
