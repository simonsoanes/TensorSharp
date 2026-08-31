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
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Telling an environment failure from a code failure — and never claiming the code is
/// correct, which is not knowable from a traceback.
///
/// <para>
/// The reference implementation instructs its own model to "fix the problem at the root
/// cause rather than applying surface-level patches". In this server's logs the model
/// could not tell what the root cause WAS: handed a host failure, it re-emitted 15,000
/// characters of program and then switched language. Re-typing a program costs about 24x
/// what re-reading it from the prompt costs, so editing code over a failure the code did
/// not cause is the most expensive wrong turn the loop can take.
/// </para>
/// </summary>
public class FailureSourceTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;

    public FailureSourceTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-fault-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private static CodeDiagnostics.FailureCause Classify(string output) =>
        CodeDiagnostics.ClassifyFailure(output);

    // ---- the environment -----------------------------------------------------

    [Fact]
    public void AMissingModuleIsTheEnvironment_AndTheHostCanFixIt()
    {
        CodeDiagnostics.FailureCause cause = Classify(
            "ModuleNotFoundError: No module named 'openpyxl'");

        Assert.Equal(CodeDiagnostics.FailureSource.Environment, cause.Source);
        // HostCanFix is what keeps the host from LECTURING about a problem it is about to
        // solve itself — a sentence about something already dealt with sends the model
        // looking for a problem that is no longer there.
        Assert.True(cause.HostCanFix);
    }

    [Fact]
    public void AnAttemptToReachTheNetworkIsTheEnvironment_AndItCannotBeFixed()
    {
        CodeDiagnostics.FailureCause cause = Classify(
            "urllib.error.URLError: <urlopen error [Errno 8] nodename nor servname provided>");

        Assert.Equal(CodeDiagnostics.FailureSource.Environment, cause.Source);
        Assert.False(cause.HostCanFix);
        Assert.Contains("network", cause.Reason, StringComparison.Ordinal);
    }

    /// <summary>
    /// What the CHILD printed is evidence. Seatbelt's own refusal to the process is.
    /// </summary>
    [Fact]
    public void ADenialTheChildItselfReportedIsTheEnvironment()
    {
        CodeDiagnostics.FailureCause cause = Classify(
            "PermissionError: [Errno 1] Operation not permitted: '/etc/hosts'");

        Assert.Equal(CodeDiagnostics.FailureSource.Environment, cause.Source);
        Assert.False(cause.HostCanFix);
    }

    /// <summary>
    /// The violation monitor's block is NOT evidence. It is a system-wide
    /// <c>log stream</c> filtered by substring against a list including "sh" — a substring
    /// of almost any path — so another process's log lines could produce this classifier's
    /// strongest environmental verdict. In the corpus that block is attached to a
    /// <c>pandoc: command not found</c> it had nothing to do with. The lines are still
    /// shown to the model; they are just no longer reasoned from.
    /// </summary>
    [Fact]
    public void TheViolationMonitorsSystemWideNoiseIsNotACause()
    {
        CodeDiagnostics.FailureCause cause = Classify(
            "bash: pandoc: command not found\n"
            + "[sandbox denials observed during this run (may include unrelated processes):]\n"
            + "  Dropbox(4211) deny(1) file-read-data /Users/x/.config/other");

        // The real cause — the missing program — must win, and the noise must not be able
        // to manufacture a verdict of its own.
        Assert.Equal(CodeDiagnostics.FailureSource.Environment, cause.Source);
        Assert.Contains("pandoc", cause.Reason, StringComparison.Ordinal);

        // On its own, the block says nothing at all.
        Assert.Equal(
            CodeDiagnostics.FailureSource.Unknown,
            Classify("[sandbox denials observed during this run (may include unrelated processes):]\n"
                     + "  Dropbox(4211) deny(1) file-read-data /Users/x/.config/other").Source);
    }

    /// <summary>
    /// A DNS failure only proves confinement where confinement exists. On Windows the
    /// job-object sandbox reports <c>ConfinesNetwork: false</c> and the network really
    /// works, so "nothing here can reach the network — not this one, not any of them" is a
    /// false statement of a constraint.
    /// </summary>
    [Fact]
    public void ANetworkFailureIsNotBlamedOnConfinementWhereThereIsNone()
    {
        const string output = "socket.gaierror: [Errno 8] nodename nor servname provided, or not known";

        Assert.Equal(
            CodeDiagnostics.FailureSource.Environment,
            CodeDiagnostics.ClassifyFailure(output, CodeLanguage.Unknown, networkConfined: true).Source);

        Assert.Equal(
            CodeDiagnostics.FailureSource.Unknown,
            CodeDiagnostics.ClassifyFailure(output, CodeLanguage.Unknown, networkConfined: false).Source);
    }

    /// <summary>The musl / non-Darwin glibc spelling was the one genuinely missing.</summary>
    [Fact]
    public void NameOrServiceNotKnownIsANetworkFailure()
    {
        Assert.Equal(
            CodeDiagnostics.FailureSource.Environment,
            Classify("socket.gaierror: [Errno -2] Name or service not known").Source);
    }

    [Fact]
    public void AMissingProgramIsTheEnvironment()
    {
        CodeDiagnostics.FailureCause cause = Classify("bash: line 1: pdftoppm: command not found");

        Assert.Equal(CodeDiagnostics.FailureSource.Environment, cause.Source);
        Assert.Contains("pdftoppm", cause.Reason, StringComparison.Ordinal);
    }

    /// <summary>
    /// The ABI mismatch, verbatim from this host's logs. The host installs wheels with the
    /// newest Python it can find; when the command ran under an older one, Pillow's own
    /// C extension failed to load. The import-name pattern matches this message, so it was
    /// being routed to the API probe — which then ran under the NEWER interpreter, found
    /// the module perfectly importable, and told the model it had guessed the API about
    /// code that was correct.
    /// </summary>
    [Fact]
    public void AnAbiMismatchInsideAPackageIsTheEnvironment_NotAGuessedApi()
    {
        const string output = """
            Traceback (most recent call last):
              File "command", line 1, in <module>
                from PIL import Image
              File "../env/PIL/Image.py", line 95, in <module>
                from . import _imaging as core
            ImportError: cannot import name '_imaging' from 'PIL' (../env/PIL/__init__.py)
            """;

        // It must NOT be read as the model guessing a name...
        Assert.False(CodeDiagnostics.TryFindApiMiss(output, out _));
        // ...and it must be read as the environment.
        Assert.Equal(CodeDiagnostics.FailureSource.Environment, Classify(output).Source);
    }

    /// <summary>
    /// But a genuine API guess — the failure raised in the MODEL's own file — must still
    /// reach the probe, or the fix above has traded one wrong answer for another.
    /// </summary>
    [Fact]
    public void AGuessedApiInTheModelsOwnFileStillReachesTheProbe()
    {
        const string output = """
            Traceback (most recent call last):
              File "command", line 86, in <module>
                slide.notes_page.shapes.add_textbox()
            AttributeError: 'Slide' object has no attribute 'notes_page'
            """;

        Assert.True(CodeDiagnostics.TryFindApiMiss(output, out CodeDiagnostics.ApiMiss miss));
        Assert.Equal("notes_page", miss.Member);
        Assert.Equal(CodeDiagnostics.FailureSource.Code, Classify(output).Source);
    }

    /// <summary>
    /// A wheel that installed and cannot load because it needs a system library. No
    /// installer available here can supply one, so a model reading only the ctypes noise
    /// reinstalls the package forever.
    /// </summary>
    [Fact]
    public void AMissingNativeLibraryIsTheEnvironmentAndCannotBeFixedHere()
    {
        CodeDiagnostics.FailureCause cause = Classify(
            "OSError: cannot load library 'libgobject-2.0-0': dlopen(libgobject-2.0-0, 0x0002): "
            + "tried: 'libgobject-2.0-0' (no such file)\n"
            + "WeasyPrint could not import some external libraries.");

        Assert.Equal(CodeDiagnostics.FailureSource.Environment, cause.Source);
        Assert.False(cause.HostCanFix);
        Assert.Contains("system library", cause.Reason, StringComparison.Ordinal);
    }

    /// <summary>
    /// The ESM resolution shape, for the same reason: "Cannot find package" from an
    /// `import` is a resolution gap, and telling the model the package is "called something
    /// else" would send it renaming a dependency that is installed and correctly spelled.
    /// </summary>
    [Fact]
    public void AnEsmResolutionFailureIsNotAGuessedName()
    {
        Assert.Equal(
            CodeDiagnostics.FailureSource.Environment,
            Classify("Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'jszip' imported from b.mjs").Source);
    }

    // ---- the code ------------------------------------------------------------

    [Fact]
    public void AnExceptionTheProgramRaisedIsTheCode()
    {
        foreach (string output in new[]
        {
            "NameError: name 'style' is not defined",
            "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
            "KeyError: 'title'",
            "IndexError: list index out of range",
            "AssertionError",
        })
        {
            Assert.Equal(CodeDiagnostics.FailureSource.Code, Classify(output).Source);
        }
    }

    /// <summary>
    /// The one that has to be ordered correctly. Apple's frozen <c>/usr/bin/python3</c> is
    /// 3.9, and a <c>match</c> statement dies there as a bare "SyntaxError: invalid
    /// syntax" — an ENVIRONMENT failure wearing a code failure's clothes. Classified as
    /// code, it would send the model to rewrite a program that is correct.
    /// </summary>
    [Fact]
    public void ASyntaxErrorFromAnInterpreterTooOldIsTheEnvironment_NotTheCode()
    {
        bool hostIsOld =
            CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
            && python != null
            && CodeEnvironment.PythonVersionOf(python) is { } v
            && v < new Version(3, 10);

        CodeDiagnostics.FailureCause cause = Classify(
            "  File \"command\", line 124\n    match family:\n          ^\nSyntaxError: invalid syntax");

        // On a host with a modern Python this really IS a code defect, and the classifier
        // must say so; only where the interpreter is genuinely too old does it become the
        // environment. Both directions are asserted, because a classifier that answers
        // "environment" regardless would be worse than none.
        Assert.Equal(
            hostIsOld ? CodeDiagnostics.FailureSource.Environment : CodeDiagnostics.FailureSource.Code,
            cause.Source);
    }

    /// <summary>
    /// The narrowing that keeps the classifier honest. "The deepest frame is in a library"
    /// is a fine reason to stop routing something to the API probe — the model did not
    /// guess a NAME — but it is NOT a reason to say the code is not at fault: a
    /// <c>TypeError</c> raised deep inside a library is usually caused by the argument the
    /// model passed in. Only an IMPORT failure inside a package is safe to call
    /// environmental, because the caller has not run yet.
    /// </summary>
    [Fact]
    public void ARuntimeErrorRaisedInsideALibraryIsNotCalledEnvironmental()
    {
        const string output = """
            Traceback (most recent call last):
              File "command", line 12, in <module>
                slide.shapes.add_picture("x.png", "left")
              File "../env/pptx/shapes/shapetree.py", line 320, in add_picture
                return Emu(int(value))
            TypeError: int() argument must be a string, not 'NoneType'
            """;

        // The model passed a bad argument. Telling it "the environment, not your code"
        // would send it to re-run a genuinely broken program.
        Assert.NotEqual(CodeDiagnostics.FailureSource.Environment, Classify(output).Source);
    }

    /// <summary>Nothing recognised produces no claim. The classifier's value is in being right.</summary>
    [Fact]
    public void AnUnrecognisedFailureMakesNoClaim()
    {
        Assert.Equal(CodeDiagnostics.FailureSource.Unknown, Classify("exit 3").Source);
        Assert.Equal(CodeDiagnostics.FailureSource.Unknown, Classify(string.Empty).Source);
        Assert.Equal(CodeDiagnostics.FailureSource.Unknown, Classify("something went wrong").Source);
    }

    // ---- what the model actually reads ---------------------------------------

    /// <summary>
    /// The sentence must never assert the code is CORRECT — a program can carry a bug and
    /// hit a missing library on the same run, and a host that says "your code is fine"
    /// when it is not has told the model to re-run something broken. What it may say is
    /// the thing the output establishes: this failure did not come from the code.
    /// </summary>
    [Fact]
    public void TheMessageSaysEditingWontHelp_NotThatTheCodeIsCorrect()
    {
        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
        });
        if (!runner.CanRun || runner.Shell is not { Kind: ShellKind.Posix })
            return;

        CodeExecResult result = runner.Run(
            new ShellRequest("ts-no-such-program-xyz"), _workspaces.GetOrCreate("fault"));

        Assert.False(result.Ok);
        Assert.Contains("This is the ENVIRONMENT, not the code you wrote", result.Content, StringComparison.Ordinal);
        Assert.Contains("will not change it", result.Content, StringComparison.Ordinal);
        // The forbidden claim.
        Assert.DoesNotContain("your code is correct", result.Content);
        Assert.DoesNotContain("your code is fine", result.Content);
    }

    /// <summary>
    /// And nothing is said for a failure the host is about to fix itself, or the model
    /// goes looking for a problem that no longer exists.
    /// </summary>
    [Fact]
    public void AFailureTheHostFixesItselfGetsNoLecture()
    {
        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
            AllowInstall = false,
        });
        if (!runner.CanRun || !CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out _, out _))
            return;

        CodeExecResult result = runner.Run(
            new ShellRequest("python3 -c 'import ts_absent_probe_pkg'"),
            _workspaces.GetOrCreate("nolecture"));

        Assert.False(result.Ok);
        Assert.DoesNotContain("This is the ENVIRONMENT", result.Content);
    }

    // ---- item 4's safe half: look at disk before acting on a guess -----------

    /// <summary>
    /// <c>import helpers</c> failing when <c>helpers.py</c> is the model's own file is an
    /// import-PATH problem. Reaching out to a package registry for that name is both the
    /// wrong answer and a way to install an arbitrary package chosen by whatever the model
    /// happened to type.
    /// </summary>
    [Fact]
    public void AFailedImportOfTheModelsOwnFileIsNotInstalledFromARegistry()
    {
        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
            AllowInstall = true,
        });
        if (!runner.CanRun || runner.Shell is not { Kind: ShellKind.Posix }
            || !CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out _, out _))
        {
            return;
        }

        SessionWorkspace workspace = _workspaces.GetOrCreate("ownfile");
        Directory.CreateDirectory(Path.Combine(workspace.WorkDirectory, "src"));
        File.WriteAllText(
            Path.Combine(workspace.WorkDirectory, "src", "ts_local_helper.py"), "VALUE = 1\n");

        CodeExecResult result = runner.Run(
            new ShellRequest("python3 -c 'import ts_local_helper'"), workspace);

        Assert.False(result.Ok);
        Assert.Contains("is your own file at", result.Content, StringComparison.Ordinal);
        Assert.Contains("src/ts_local_helper.py", result.Content, StringComparison.Ordinal);
        Assert.Contains("not on the import path", result.Content, StringComparison.Ordinal);
        // And the host did NOT go shopping for it.
        Assert.DoesNotContain("was installed and the command was run again", result.Content);
    }
}
