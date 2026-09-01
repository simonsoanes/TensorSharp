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
/// A patch that applied and a write that succeeded must not be reported as fine when the
/// file they produced cannot be parsed.
/// </summary>
public class SyntaxCheckTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _manager;
    private readonly SessionWorkspace _workspace;
    private readonly ShellRunner _runner;

    public SyntaxCheckTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-syntax-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _manager = new SessionWorkspaceManager(Path.Combine(_base, "sessions"));
        _workspace = _manager.GetOrCreate("s");
        _runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            ScratchDirectory = Path.Combine(_base, "scratch"),
            Timeout = TimeSpan.FromSeconds(60),
        });
    }

    public void Dispose()
    {
        _runner.Dispose();
        try { _manager.Release("s"); } catch { /* best effort */ }
        try { Directory.Delete(_base, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private string Work(string name) => Path.Combine(_workspace.WorkDirectory, name);

    /// <summary>
    /// A command that writes <paramref name="body"/> into <paramref name="path"/>, in the
    /// dialect THIS host's shell actually speaks.
    ///
    /// <para>
    /// These tests were written as bash heredocs - <c>cat &gt; f &lt;&lt;'EOF' ... EOF</c>
    /// - which PowerShell cannot parse at all. On Windows every one of them therefore
    /// failed in the COMMAND rather than in the thing it was checking, and the three that
    /// assert on a syntax diagnostic reported a defect in a feature that was working. The
    /// feature is dialect-aware already: SyntaxCheck.RedirectTargets matches
    /// <c>Set-Content</c> and <c>Out-File</c> alongside <c>&gt;</c>, <c>&gt;&gt;</c> and
    /// <c>tee</c>. Only the tests were not, so on Windows they exercised nothing.
    /// </para>
    /// <para>
    /// The PowerShell form is a here-string, whose terminator <c>'@</c> has to sit at
    /// column zero - which is why it is built here once rather than inline per test.
    /// </para>
    /// </summary>
    private string Heredoc(string path, string body) =>
        _runner.Shell is { Kind: ShellKind.PowerShell }
            ? "@'\n" + body + "\n'@ | Set-Content -LiteralPath " + path + "\n"
            : "cat > " + path + " <<'EOF'\n" + body + "\nEOF";

    /// <summary>The spelling of "run this python file" this host understands.</summary>
    private string RunPython(string path) =>
        _runner.Shell is { Kind: ShellKind.PowerShell } ? "python " + path : "python3 " + path;


    // ---- which files a command wrote -----------------------------------------

    private static string[] Paths(string command) =>
        SyntaxCheck.RedirectTargets(command).Select(t => t.Path).ToArray();

    [Fact]
    public void AHeredocRedirectTargetIsFound()
    {
        Assert.Contains("deck.py", Paths("cat > deck.py <<'EOF'\nprint(1)\nEOF"));
        // The PowerShell spelling of the same write, which the host must see as well.
        Assert.Contains("deck.py", Paths("@'\nprint(1)\n'@ | Set-Content -LiteralPath deck.py"));
    }

    [Fact]
    public void QuotedAndAppendedTargetsAreFound()
    {
        Assert.Contains("a b.py", Paths("cat > 'a b.py' <<EOF\nEOF"));
        Assert.Contains("log.json", Paths("echo x >> log.json"));
        Assert.Contains("out.js", Paths("printf x | tee out.js"));
        Assert.Contains("deck.py", Paths("@'\nx\n'@ | Set-Content -LiteralPath deck.py"));
    }

    /// <summary>
    /// A heredoc BODY is data, not shell. Writing documentation is the commonest heredoc
    /// there is, and prose that happens to contain a redirection must not be read as one:
    /// scanning the raw command string for `cat > README.md &lt;&lt;EOF / Run it with:
    /// python gen.py &gt; out.py / EOF` found `out.py` in the prose and then reported a
    /// parse failure for a file the command never opened.
    /// </summary>
    [Fact]
    public void ARedirectionInsideAHeredocBodyIsNotATarget()
    {
        string[] targets = Paths(
            Heredoc("README.md", "Run it with: python gen.py > out.py"));

        Assert.Contains("README.md", targets);
        Assert.DoesNotContain("out.py", targets);
        Assert.Single(targets);
    }

    /// <summary>
    /// And the consequence, end to end: a command that wrote only documentation must not
    /// be told that some other file on disk does not parse.
    /// </summary>
    [Fact]
    public void AHeredocWritingProseIsNotBlamedForAnUnrelatedBrokenFile()
    {
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
            || python == null)
        {
            return;
        }

        // A broken file that exists from some earlier step and is NOT what this command wrote.
        File.WriteAllText(Work("out.py"), "def broken(:\n    pass\n");

        CodeExecResult result = _runner.Run(
            new ShellRequest(Heredoc("README.md", "Run it with: python gen.py > out.py")),
            _workspace);

        Assert.True(result.Ok, result.Content);
        Assert.DoesNotContain("out.py", result.Content);
        Assert.DoesNotContain("does not parse", result.Content);
    }

    /// <summary>
    /// An APPEND has to be distinguishable from a REPLACE. A syntax check does not care
    /// how the bytes arrived, but anything reasoning about what the command DID does —
    /// and calling an append a rewrite made the host tell the model its correct action
    /// had been wasteful.
    /// </summary>
    [Fact]
    public void AnAppendIsMarkedAsOneAndAReplaceIsNot()
    {
        Assert.True(SyntaxCheck.RedirectTargets("echo x >> log.py").Single().Appends);
        Assert.True(SyntaxCheck.RedirectTargets("printf x | tee -a log.py").Single().Appends);
        Assert.True(SyntaxCheck.RedirectTargets("'x' | Out-File -Append log.py").Single().Appends);

        Assert.False(SyntaxCheck.RedirectTargets("echo x > log.py").Single().Appends);
        Assert.False(SyntaxCheck.RedirectTargets("printf x | tee log.py").Single().Appends);
        // The path must still come out right when the operator group is the one that is
        // optional: a positional scan across the alternatives read "-a " as the path.
        Assert.Equal("log.py", SyntaxCheck.RedirectTargets("printf x | tee -a log.py").Single().Path);
    }

    /// <summary>
    /// Descriptor plumbing is not a file, and treating it as one would have the host
    /// looking for a file called "1" after every <c>2&gt;&amp;1</c> — which is most
    /// commands a model writes.
    /// </summary>
    [Fact]
    public void DescriptorPlumbingAndSinksAreNotFiles()
    {
        Assert.Empty(Paths("python3 x.py 2>&1"));
        Assert.Empty(Paths("python3 x.py > /dev/null"));
        Assert.Empty(Paths("echo hi >&2"));
        // A target built from a variable cannot be known without running the command.
        Assert.Empty(Paths("cat > $OUT.py <<EOF\nEOF"));
    }

    // ---- the check itself ----------------------------------------------------

    [Fact]
    public void BrokenJsonIsReportedWithoutASubprocess()
    {
        File.WriteAllText(Work("data.json"), "{\"a\": 1,}");

        var check = new SyntaxCheck(
            new CodeExecOptions { Enabled = true, Sandbox = SkillSandboxMode.Off }, sandbox: null);
        string? report = check.Verify(new[] { "data.json" }, _workspace);

        Assert.NotNull(report);
        Assert.Contains("data.json", report!, StringComparison.Ordinal);
        Assert.Contains("does not parse", report!, StringComparison.Ordinal);
    }

    [Fact]
    public void ValidFilesAreSilent()
    {
        File.WriteAllText(Work("data.json"), "{\"a\": 1}");
        File.WriteAllText(Work("ok.py"), "def f():\n    return 1\n");

        var check = new SyntaxCheck(
            new CodeExecOptions { Enabled = true, Sandbox = SkillSandboxMode.Off }, sandbox: null);
        Assert.Null(check.Verify(new[] { "data.json", "ok.py", "absent.py" }, _workspace));
    }

    [Fact]
    public void BrokenPythonIsReportedWithItsLine()
    {
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
            || python == null)
        {
            return;
        }

        File.WriteAllText(Work("deck.py"), "def f():\n    return 1\n  bad indent here\n");

        var check = new SyntaxCheck(
            new CodeExecOptions { Enabled = true, Sandbox = SkillSandboxMode.Off }, sandbox: null);
        string? report = check.Verify(new[] { "deck.py" }, _workspace);

        Assert.NotNull(report);
        Assert.Contains("deck.py line 3", report!, StringComparison.Ordinal);
    }

    /// <summary>
    /// The check must never leave files behind. <c>py_compile</c> writes a
    /// <c>__pycache__</c>, and anything appearing in the work directory is captured and
    /// handed to the user as a download — which is why this compiles in memory instead.
    /// </summary>
    [Fact]
    public void CheckingPythonLeavesNoBytecodeBehind()
    {
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
            || python == null)
        {
            return;
        }

        File.WriteAllText(Work("ok.py"), "print(1)\n");
        new SyntaxCheck(new CodeExecOptions { Enabled = true, Sandbox = SkillSandboxMode.Off }, null)
            .Verify(new[] { "ok.py" }, _workspace);

        Assert.Empty(Directory.GetDirectories(_workspace.WorkDirectory, "__pycache__"));
        Assert.Empty(Directory.GetFiles(_workspace.WorkDirectory, "*.pyc"));
    }

    // ---- through the real tools ----------------------------------------------

    /// <summary>
    /// The whole point on the patch side: the patch APPLIED — the result says so — and
    /// the same result also says the file it produced no longer parses. Reporting only
    /// the first half is the defect.
    /// </summary>
    [Fact]
    public void APatchThatAppliesButBreaksTheFileSaysBoth()
    {
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
            || python == null)
        {
            return;
        }

        File.WriteAllText(Work("deck.py"), "def build():\n    total = 0\n    return total\n");

        CodeExecResult result = _runner.ApplyPatch(
            """
            *** Begin Patch
            *** Update File: deck.py
             def build():
            -    total = 0
            +    total = (0
                 return total
            *** End Patch
            """,
            _workspace);

        Assert.True(result.Ok);
        Assert.Contains("updated deck.py", result.Content, StringComparison.Ordinal);
        Assert.Contains("does not parse", result.Content, StringComparison.Ordinal);
        Assert.Contains("deck.py line", result.Content, StringComparison.Ordinal);
    }

    /// <summary>
    /// A patch that changed nothing must not be reported as an edit. This is case B of the
    /// reference implementation's own behaviour, executed against it: a hunk with only
    /// context lines writes the file back byte-identical and reports
    /// <c>Updated &lt;path&gt;</c> — and the reference's prompt tells its model not to
    /// re-read after a patch because "the tool call will fail if it didn't work", so there
    /// is no way for it to find out.
    /// </summary>
    [Fact]
    public void APatchThatChangesNothingSaysSo()
    {
        File.WriteAllText(Work("deck.py"), "def build():\n    total = 0\n    return total\n");

        CodeExecResult result = _runner.ApplyPatch(
            """
            *** Begin Patch
            *** Update File: deck.py
            @@ def build():
                 total = 0
                 return total
            *** End Patch
            """,
            _workspace);

        Assert.Equal(
            "def build():\n    total = 0\n    return total\n", File.ReadAllText(Work("deck.py")));
        Assert.Contains("nothing", result.Content, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("(+0 -0)", result.Content);
    }

    /// <summary>
    /// And on the shell side: a heredoc that wrote a broken file and exited 0. Without
    /// this the result is "exit 0" and nothing else, which the model reads as done.
    /// </summary>
    [Fact]
    public void AHeredocThatWritesBrokenCodeAndExitsZeroIsNotSilent()
    {
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
            || python == null)
        {
            return;
        }

        CodeExecResult result = _runner.Run(
            new ShellRequest(Heredoc("deck.py", "def build(:\n    pass")), _workspace);

        Assert.True(result.Ok);
        Assert.Contains("does not parse", result.Content, StringComparison.Ordinal);
    }

    /// <summary>
    /// A command that failed already printed its own error. Saying it twice is noise, and
    /// noise in a failure result is what pushes the real cause out of the model's view.
    /// </summary>
    [Fact]
    public void AFailedCommandIsNotToldTwice()
    {
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? python, out _)
            || python == null)
        {
            return;
        }

        CodeExecResult result = _runner.Run(
            new ShellRequest(Heredoc("deck.py", "def build(:\n    pass") + "\n" + RunPython("deck.py")),
            _workspace);

        Assert.False(result.Ok);
        Assert.Contains("SyntaxError", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("does not parse", result.Content);
    }
}
