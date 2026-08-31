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
/// The session environment's console scripts are reachable from a command.
///
/// <para>
/// This is the LIVE test for a property that three tests in
/// <c>BuiltInToolRegistryTests</c> still assert against dead code. Those three call
/// <c>CodeEnvironment.PythonLaunch</c> and <c>CodeEnvironment.ShellLaunch</c>, which
/// built the PATH for the five-tool program runner. Nothing in the product calls either
/// any more: the shell tool builds its own environment in
/// <c>ShellRunner.BuildEnvironment</c>, which prefixes PATH with the session's
/// <c>env/bin</c>, <c>env/node_modules/.bin</c> and the venv bin directory itself. So
/// the old tests pass whatever the product does, and would keep passing if the shell
/// stopped putting the environment on PATH at all.
/// </para>
/// <para>
/// The property still matters for the same reason it always did: pip <c>--target</c>
/// drops console scripts in the environment's bin directory, a skill that says "run
/// <c>markitdown deck.pptx</c>" means the one just installed, and without the prefix the
/// model gets exit 127 with nothing to explain it. So assert it where it is actually
/// implemented — by running a command and reading the PATH the child really got.
/// </para>
/// </summary>
public class ShellEnvironmentPathTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;

    public ShellEnvironmentPathTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-shellenv-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private static bool HavePosixShell =>
        ShellProgram.TryResolve(null, out ShellProgram? shell, out _) && shell is { Kind: ShellKind.Posix };

    [Fact]
    public void ACommandsPath_StartsWithTheSessionEnvironmentsBinDirectories()
    {
        if (!HavePosixShell) return;

        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
        });
        SessionWorkspace workspace = _workspaces.GetOrCreate("envpath");

        // Asserted in the form the MODEL sees, which since OutputPaths is relative to the
        // directory the command ran in: the host's absolute layout is rewritten out of
        // every result. The invariant is unchanged — these two directories lead PATH —
        // and only its rendering moved. (A relative PATH entry is not a usable PATH
        // value, but nothing here needs one: PATH is deliberately not persisted between
        // calls, so a model cannot set it, and the session id in the absolute form is
        // exactly the noise that rewriting exists to remove.)
        string venvBin = Path.GetRelativePath(
            workspace.WorkDirectory, CodeEnvironment.VenvBin(workspace.EnvDirectory));
        string nodeBin = Path.GetRelativePath(
            workspace.WorkDirectory, Path.Combine(workspace.EnvDirectory, "node_modules", ".bin"));

        CodeExecResult result = runner.Run(new ShellRequest("printf '%s' \"$PATH\""), workspace);

        Assert.True(result.Ok, result.Content);
        Assert.Contains(venvBin.Replace('\\', '/'), result.Content.Replace('\\', '/'), StringComparison.Ordinal);
        Assert.Contains(nodeBin.Replace('\\', '/'), result.Content.Replace('\\', '/'), StringComparison.Ordinal);
        // And the thing the rewrite is for: no host path, and no session id.
        Assert.DoesNotContain(workspace.EnvDirectory, result.Content);
        Assert.DoesNotContain("ts-session-", result.Content);
    }

    /// <summary>
    /// 9 rounds and 5 incidents in this server's logs were one message:
    /// <c>python: command not found</c>. The host's only Python is <c>python3</c>, and the
    /// shell description names it — with its version — and the models type <c>python</c>
    /// anyway. So the host answers what the model says.
    /// </summary>
    [Fact]
    public void PythonMeansPython3_BecauseThatIsWhatModelsType()
    {
        if (!HavePosixShell) return;

        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
        });
        SessionWorkspace workspace = _workspaces.GetOrCreate("pyalias");

        bool havePython = CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out _, out _);

        CodeExecResult result = runner.Run(
            new ShellRequest("python -c \"print('shimmed')\""), workspace);

        string alias = Path.Combine(workspace.EnvDirectory, "shim", "python");
        if (!havePython)
        {
            // No interpreter, no shim. A shim over nothing turns an honest "command not
            // found" into a broken script, which is strictly worse than the truth.
            Assert.False(File.Exists(alias));
            return;
        }

        Assert.True(File.Exists(alias));
        Assert.True(result.Ok, result.Content);
        Assert.Contains("shimmed", result.Content, StringComparison.Ordinal);

        // Its own directory, not env/bin: that is where PIP_TARGET drops console scripts,
        // so a package shipping a `python` entry point would collide with the shim there
        // and which one won would depend on install order.
        Assert.False(File.Exists(Path.Combine(workspace.EnvDirectory, "bin", "python")));
    }

    /// <summary>
    /// The concrete host case this exists for, asserted as an OUTCOME rather than a
    /// mechanism: a `match` statement — 3.10 syntax — must run. The bundled pptx skill's
    /// own scripts/office/validate.py fails to parse under Apple's 3.9, and that failure
    /// reads as a broken script.
    /// </summary>
    [Fact]
    public void ThreeTenSyntaxRuns_WhenThisHostHasAThreeTenPython()
    {
        if (!HavePosixShell) return;
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? resolved, out _)
            || resolved == null
            || CodeEnvironment.PythonVersionOf(resolved) is not { } version
            || version < new Version(3, 10))
        {
            return;   // this host genuinely cannot run it; nothing to prove
        }

        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
        });
        SessionWorkspace workspace = _workspaces.GetOrCreate("match");

        // Typed as `python3`, which is what a skill script's shebang and a model both use.
        CodeExecResult result = runner.Run(
            new ShellRequest(
                "cat > m.py <<'EOF'\n"
                + "def kind(x):\n"
                + "    match x:\n"
                + "        case 1: return \"one\"\n"
                + "        case _: return \"many\"\n"
                + "print(kind(1))\n"
                + "EOF\n"
                + "python3 m.py"),
            workspace);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("one", result.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("SyntaxError", result.Content);
    }

    /// <summary>
    /// The second shadowing case, and the one that broke a bundled skill: `python3` can
    /// EXIST and be the wrong one. Interpreter resolution prefers 3.14 down to 3.10 before
    /// a bare `python3`, but a command typing `python3` gets whatever PATH finds first —
    /// Apple's frozen 3.9 on a Mac — where the pptx skill's own validate.py fails to parse
    /// on a `match` statement, and reads as a broken script rather than an old interpreter.
    /// </summary>
    [Fact]
    public void Python3IsShadowedOnlyWhenTheHostHasANewerOne()
    {
        if (!HavePosixShell) return;
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out string? resolved, out _)
            || resolved == null)
        {
            return;
        }

        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
        });
        SessionWorkspace workspace = _workspaces.GetOrCreate("py3shim");
        runner.Run(new ShellRequest("true"), workspace);

        string alias = Path.Combine(workspace.EnvDirectory, "shim", "python3");
        string? onPath = CodeEnvironment.Which("python3");
        Version? resolvedVersion = CodeEnvironment.PythonVersionOf(resolved);
        Version? pathVersion = onPath == null ? null : CodeEnvironment.PythonVersionOf(onPath);

        bool shouldShadow = onPath == null
            || (resolvedVersion != null && pathVersion != null && resolvedVersion > pathVersion);

        // Both directions asserted: a host with one Python must NOT get a pointless
        // wrapper around the interpreter the model already asked for by name.
        Assert.Equal(shouldShadow, File.Exists(alias));

        if (!shouldShadow)
            return;

        CodeExecResult result = runner.Run(
            new ShellRequest("python3 -c \"import sys; print(sys.version_info[:2])\""), workspace);
        Assert.True(result.Ok, result.Content);
        Assert.Contains(
            $"({resolvedVersion!.Major}, {resolvedVersion.Minor})", result.Content, StringComparison.Ordinal);
    }

    /// <summary>
    /// Belt to the shim's braces: on a host where the shim could not be written, the
    /// coaching must still name the spelling that exists. It used to say "no package
    /// manager here can supply it — do the step another way", because `python` is not a
    /// package manager — on a host whose own shell description had already reported
    /// `python3 (3.9.6)`.
    /// </summary>
    [Fact]
    public void ACommandSpelledDifferentlyHere_IsAnsweredWithTheSpellingThatExists()
    {
        if (!CodeEnvironment.TryResolveInterpreter(CodeLanguage.Python, out _, out _))
            return;

        Assert.Equal("python3", CodeDiagnostics.SpelledDifferentlyHere("python"));

        // And a program that genuinely is not here keeps the honest answer.
        Assert.Null(CodeDiagnostics.SpelledDifferentlyHere("ts-no-such-program-xyz"));
    }

    [Fact]
    public void AConsoleScriptDroppedInTheEnvironment_IsRunnableByItsBareName()
    {
        if (!HavePosixShell || OperatingSystem.IsWindows()) return;

        using var runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            Timeout = TimeSpan.FromSeconds(30),
            ScratchDirectory = _base,
        });
        SessionWorkspace workspace = _workspaces.GetOrCreate("envscript");

        // Exactly what `pip install --target` leaves behind: an executable in the
        // environment's bin directory and nothing anywhere else that names it.
        string bin = CodeEnvironment.VenvBin(workspace.EnvDirectory);
        Directory.CreateDirectory(bin);
        string script = Path.Combine(bin, "ts-probe-tool");
        File.WriteAllText(script, "#!/bin/sh\necho probe-ran\n");
        File.SetUnixFileMode(
            script,
            UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);

        CodeExecResult result = runner.Run(new ShellRequest("ts-probe-tool"), workspace);

        Assert.True(result.Ok, result.Content);
        Assert.Contains("probe-ran", result.Content, StringComparison.Ordinal);
    }
}
