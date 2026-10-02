using TensorAgent.Core.Python;
using TensorAgent.Core.Sandbox;
using TensorAgent.Core.Shell;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace TensorAgent.Tests;

/// <summary>
/// The seam test: the agent host's own <see cref="ShellLaunch"/> going through the
/// in-process backend and coming back as a <see cref="ConfinedResult"/> the host can
/// read. What is being checked is the translation — permissions in, confinement out,
/// session state persisted the way a wrapper script's exit trap would have persisted it.
/// </summary>
public sealed class InProcessShellBackendTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-be-" + Guid.NewGuid().ToString("N"));
    private readonly string _work;
    private readonly string _readonly;
    private readonly InProcessShellBackend _concrete = new();
    private IShellBackend _backend => _concrete;

    public InProcessShellBackendTests()
    {
        _work = Path.Combine(_root, "work");
        _readonly = Path.Combine(_root, "skill");
        Directory.CreateDirectory(_work);
        Directory.CreateDirectory(_readonly);
    }

    public void Dispose() { try { Directory.Delete(_root, true); } catch { } }

    private ShellLaunch Argv(params string[] argv) => new()
    {
        Argv = argv,
        WorkingDirectory = _work,
        WriteDirectory = _work,
        ReadOnlyDirectory = _readonly,
        Timeout = TimeSpan.FromSeconds(20),
    };

    private ShellLaunch Command(string command, SessionWorkspace workspace, ShellSession session) => new()
    {
        Command = command,
        Session = session,
        WorkingDirectory = session.CurrentDirectory,
        WriteDirectory = workspace.WorkDirectory,
        ReadOnlyDirectory = _readonly,
        Timeout = TimeSpan.FromSeconds(20),
    };

    private (SessionWorkspace Workspace, ShellSession Session) NewSession()
    {
        var manager = new SessionWorkspaceManager(Path.Combine(_root, "sessions"));
        SessionWorkspace workspace = manager.GetOrCreate("s1");
        return (workspace, new ShellSession(workspace, ShellProgram.InProcess()));
    }

    [Fact]
    public void ItPresentsItselfAsAPosixShellThatConfinesWhatItClaims()
    {
        Assert.Equal("in-process", _backend.Name);
        Assert.True(_backend.CanRun);
        Assert.Null(_backend.UnavailableReason);
        Assert.Equal(ShellKind.Posix, _backend.Shell!.Kind);
        Assert.Equal("sh", _backend.Shell.Name);

        SkillSandboxCapabilities capabilities = _backend.Sandbox!.Capabilities;
        Assert.True(capabilities.ConfinesWrites);
        Assert.True(capabilities.ConfinesNetwork);
        Assert.True(capabilities.ConfinesHomeReads);
        Assert.Empty(capabilities.Gaps());
    }

    [Fact]
    public void TheDescriptionNamesTheInterpretersItDoesNotHave()
    {
        string description = _concrete.Describe();
        Assert.Contains("no python", description, StringComparison.Ordinal);
        Assert.Contains("no node", description, StringComparison.Ordinal);
        Assert.Contains("no installs", description, StringComparison.Ordinal);
    }

    [Fact]
    public void AnArgvLaunchRunsWithoutAShellParsingAnything()
    {
        ConfinedResult result = _backend.Run(Argv("echo", "a;b|c", "$HOME"));
        Assert.True(result.Ok, result.Stderr);
        Assert.Equal("a;b|c $HOME\n", result.Stdout);
        Assert.Equal("in-process", result.SandboxName);
    }

    [Fact]
    public void ACommandLaunchRunsTheShellLanguage()
    {
        (SessionWorkspace workspace, ShellSession session) = NewSession();
        ConfinedResult result = _backend.Run(Command("for i in 1 2; do echo line $i; done", workspace, session));
        Assert.True(result.Ok, result.Stderr);
        Assert.Equal("line 1\nline 2\n", result.Stdout);
    }

    [Fact]
    public void ALaunchThatCarriesNeitherACommandNorAnArgvIsRefusedBeforeAnythingRuns()
    {
        var launch = new ShellLaunch { WorkingDirectory = _work, WriteDirectory = _work, ReadOnlyDirectory = _readonly };
        Assert.False(_backend.TryStart(launch, out IShellJob? job, out ConfinedResult failure));
        Assert.Null(job);
        Assert.False(failure.Started);
        Assert.NotNull(failure.Error);
    }

    [Fact]
    public void TheSessionKeepsItsDirectoryAndExportsBetweenCalls()
    {
        (SessionWorkspace workspace, ShellSession session) = NewSession();
        Directory.CreateDirectory(Path.Combine(workspace.WorkDirectory, "inner"));

        ConfinedResult first = _backend.Run(new ShellLaunch
        {
            Command = "cd inner && export STAGE=two",
            Session = session,
            WorkingDirectory = session.CurrentDirectory,
            WriteDirectory = workspace.WorkDirectory,
            ReadOnlyDirectory = _readonly,
            Timeout = TimeSpan.FromSeconds(20),
        });
        Assert.True(first.Ok, first.Stderr);
        Assert.EndsWith("inner", session.CurrentDirectory, StringComparison.Ordinal);

        ConfinedResult second = _backend.Run(new ShellLaunch
        {
            Command = "echo $STAGE",
            Session = session,
            WorkingDirectory = session.CurrentDirectory,
            WriteDirectory = workspace.WorkDirectory,
            ReadOnlyDirectory = _readonly,
            Timeout = TimeSpan.FromSeconds(20),
        });
        Assert.Equal("two\n", second.Stdout);
    }

    [Fact]
    public void APerCallDirectoryMovesOneCommandAndNotTheConversation()
    {
        (SessionWorkspace workspace, ShellSession session) = NewSession();
        string inner = Path.Combine(workspace.WorkDirectory, "inner");
        Directory.CreateDirectory(inner);
        string before = session.CurrentDirectory;

        ConfinedResult result = _backend.Run(new ShellLaunch
        {
            Command = "pwd",
            Session = session,
            WorkingDirectory = workspace.WorkDirectory,
            CallWorkDirectory = inner,
            WriteDirectory = workspace.WorkDirectory,
            ReadOnlyDirectory = _readonly,
            Timeout = TimeSpan.FromSeconds(20),
        });

        Assert.Contains("inner", result.Stdout, StringComparison.Ordinal);
        Assert.Equal(before, session.CurrentDirectory);
    }

    [Fact]
    public void TheWriteDirectoryIsTheOnlyPlaceACommandCanWrite()
    {
        ConfinedResult inside = _backend.Run(Argv("sh", "-c", "echo ok > allowed.txt"));
        Assert.True(inside.Ok, inside.Stderr);
        Assert.Equal("ok\n", File.ReadAllText(Path.Combine(_work, "allowed.txt")));

        ConfinedResult outside = _backend.Run(Argv("sh", "-c", $"echo no > {Path.Combine(_readonly, "denied.txt")}"));
        Assert.False(outside.Ok);
        Assert.False(File.Exists(Path.Combine(_readonly, "denied.txt")));
    }

    [Fact]
    public void AnExactWritablePathIsGrantedWithoutGrantingItsSiblings()
    {
        string allowed = Path.Combine(_root, "handoff.json");
        string sibling = Path.Combine(_root, "secret.json");
        File.WriteAllText(sibling, "private");

        ConfinedResult write = _backend.Run(new ShellLaunch
        {
            Argv = new[] { "sh", "-c", $"echo written > '{allowed}'" },
            WorkingDirectory = _work,
            WriteDirectory = _work,
            WritablePaths = new[] { allowed },
            ReadOnlyDirectory = _readonly,
            Timeout = TimeSpan.FromSeconds(20),
        });
        Assert.True(write.Ok, write.Stderr);
        Assert.Equal("written\n", File.ReadAllText(allowed));

        ConfinedResult neighbour = _backend.Run(new ShellLaunch
        {
            Argv = new[] { "sh", "-c", $"echo clobbered > '{sibling}'" },
            WorkingDirectory = _work,
            WriteDirectory = _work,
            WritablePaths = new[] { allowed },
            ReadOnlyDirectory = _readonly,
            Timeout = TimeSpan.FromSeconds(20),
        });
        Assert.False(neighbour.Ok);
        Assert.Equal("private", File.ReadAllText(sibling));
    }

    [Fact]
    public void TheReadOnlyDirectoryIsReadableAndStillNotWritable()
    {
        File.WriteAllText(Path.Combine(_readonly, "SKILL.md"), "# how to");
        ConfinedResult read = _backend.Run(Argv("cat", Path.Combine(_readonly, "SKILL.md")));
        Assert.True(read.Ok, read.Stderr);
        Assert.Contains("# how to", read.Stdout, StringComparison.Ordinal);
    }

    [Fact]
    public void TheNetworkIsOffUnlessTheLaunchAsksForIt()
    {
        ConfinedResult refused = _backend.Run(Argv("sh", "-c", "curl https://example.com"));
        Assert.False(refused.Ok);
        Assert.Contains("network", refused.Stderr, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ADeadlineStopsTheRunAndSaysSo()
    {
        ConfinedResult result = _backend.Run(new ShellLaunch
        {
            Argv = new[] { "sh", "-c", "while true; do :; done" },
            WorkingDirectory = _work,
            WriteDirectory = _work,
            ReadOnlyDirectory = _readonly,
            Timeout = TimeSpan.FromMilliseconds(800),
        });
        Assert.True(result.TimedOut);
        Assert.False(result.Ok);
    }

    [Fact]
    public void OutputArrivesLineByLineWhileTheCommandIsStillRunning()
    {
        var lines = new List<string>();
        ConfinedResult result = _backend.Run(new ShellLaunch
        {
            Argv = new[] { "sh", "-c", "echo one; echo two; echo three" },
            WorkingDirectory = _work,
            WriteDirectory = _work,
            ReadOnlyDirectory = _readonly,
            Timeout = TimeSpan.FromSeconds(20),
            OnOutputLine = line => { lock (lines) lines.Add(line); },
        });
        Assert.True(result.Ok, result.Stderr);
        Assert.Equal(new[] { "one", "two", "three" }, lines);
    }

    [Fact]
    public void AJobCanBeStartedAndStoppedWithoutWaitingForIt()
    {
        Assert.True(_backend.TryStart(new ShellLaunch
        {
            Argv = new[] { "sh", "-c", "while true; do :; done" },
            WorkingDirectory = _work,
            WriteDirectory = _work,
            ReadOnlyDirectory = _readonly,
            Timeout = Timeout.InfiniteTimeSpan,
        }, out IShellJob? job, out _));

        Assert.NotNull(job);
        Assert.Equal(-1, job!.ProcessId);
        Assert.Equal("in-process", job.SandboxName);
        job.Kill();
        job.Dispose();
    }

    [Fact]
    public void AMissingWorkingDirectoryIsReportedRatherThanThrown()
    {
        Assert.False(_backend.TryStart(new ShellLaunch
        {
            Argv = new[] { "echo", "hi" },
            WorkingDirectory = Path.Combine(_root, "gone"),
            WriteDirectory = _work,
            ReadOnlyDirectory = _readonly,
        }, out _, out ConfinedResult failure));
        Assert.False(failure.Started);
        Assert.Contains("does not exist", failure.Error!, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("pip list")]
    [InlineData("pip freeze")]
    [InlineData("python3 -m pip list")]
    [InlineData("python3 -m pip freeze")]
    public void PackageInspectionReadsTheSessionTargetForEverySupportedSpelling(string command)
    {
        (SessionWorkspace workspace, ShellSession session) = NewSession();
        Directory.CreateDirectory(Path.Combine(workspace.EnvDirectory, "img2pdf-0.6.1.dist-info"));
        Directory.CreateDirectory(Path.Combine(workspace.EnvDirectory, "yfinance-0.2.65.dist-info"));

        // PIP_TARGET is the authoritative package root. A different PYTHONPATH also
        // proves the backend did not confuse an appended import path with the
        // session environment whose installed distributions pip must report.
        string decoy = Path.Combine(workspace.Root, "decoy");
        Directory.CreateDirectory(decoy);
        Directory.CreateDirectory(Path.Combine(decoy, "not-installed-9.9.dist-info"));
        var backend = new InProcessShellBackend(new PipMustNotReachPython());
        ConfinedResult result = ((IShellBackend)backend).Run(new ShellLaunch
        {
            Command = command,
            Session = session,
            WorkingDirectory = session.CurrentDirectory,
            WriteDirectory = workspace.WorkDirectory,
            ReadOnlyDirectory = workspace.Root,
            Environment = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["PIP_TARGET"] = workspace.EnvDirectory,
                ["PYTHONPATH"] = decoy,
            },
            Timeout = TimeSpan.FromSeconds(20),
        });

        Assert.True(result.Ok, result.Stderr);
        Assert.Contains("img2pdf==0.6.1", result.Stdout, StringComparison.Ordinal);
        Assert.Contains("yfinance==0.2.65", result.Stdout, StringComparison.Ordinal);
        Assert.DoesNotContain("not-installed", result.Stdout, StringComparison.Ordinal);
    }

    [Fact]
    public void NestedPythonPipInstallStillHasNoRawInstallerHook()
    {
        (SessionWorkspace workspace, ShellSession session) = NewSession();
        var backend = new InProcessShellBackend(new PipMustNotReachPython());
        ConfinedResult result = ((IShellBackend)backend).Run(new ShellLaunch
        {
            Command = "sh -c 'python3 -m pip install img2pdf'",
            Session = session,
            WorkingDirectory = session.CurrentDirectory,
            WriteDirectory = workspace.WorkDirectory,
            ReadOnlyDirectory = workspace.Root,
            Environment = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["PIP_TARGET"] = workspace.EnvDirectory,
                ["PYTHONPATH"] = workspace.EnvDirectory,
            },
            Timeout = TimeSpan.FromSeconds(20),
        });

        Assert.False(result.Ok);
        Assert.Contains(ExecutionPolicy.InstallsByHostMessage, result.Stderr, StringComparison.Ordinal);
    }

    private sealed class PipMustNotReachPython : IPythonRuntime
    {
        public bool IsAvailable => true;
        public string? UnavailableReason => null;
        public string Version => "3.13-test";

        public Task<ExecutionResult> RunScriptAsync(
            string scriptPath, IReadOnlyList<string> arguments,
            InterpreterContext context, CancellationToken cancellationToken)
            => throw new InvalidOperationException("pip inspection reached embedded Python");

        public Task<ExecutionResult> RunCodeAsync(
            string source, IReadOnlyList<string> arguments,
            InterpreterContext context, CancellationToken cancellationToken)
            => throw new InvalidOperationException("pip inspection reached embedded Python");

        public Task<ExecutionResult> RunModuleAsync(
            string module, IReadOnlyList<string> arguments,
            InterpreterContext context, CancellationToken cancellationToken)
            => throw new InvalidOperationException("pip inspection reached embedded Python");

        public Task<SyntaxCheckResult> CheckSyntaxAsync(
            string scriptPath, CancellationToken cancellationToken)
            => throw new InvalidOperationException("pip inspection reached embedded Python");
    }
}
