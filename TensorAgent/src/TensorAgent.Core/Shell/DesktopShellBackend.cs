// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace TensorAgent.Core.Shell;

/// <summary>Which execution environment an AgentAppHost should use.</summary>
public enum AgentExecutionMode
{
    /// <summary>Native processes on desktop; embedded interpreters on mobile.</summary>
    Auto,
    /// <summary>Embedded interpreters, including when emulating a mobile host.</summary>
    InProcess,
    /// <summary>Native processes; refused on platforms that prohibit them.</summary>
    Process,
}

/// <summary>
/// Uses the shared desktop sandbox while preserving TensorAgent's host allow-list
/// promise. The process sandbox cannot currently enforce DNS host allow-lists for
/// arbitrary executables, so it refuses network-enabled launches under that policy.
/// </summary>
internal sealed class DesktopShellBackend : IShellBackend
{
    private readonly IShellBackend _inner;

    // Mac Catalyst is the desktop TensorAgent on a Mac: .NET reports it as iOS, but it
    // is a macOS process that can start children and confine them with Seatbelt.
    internal static bool IsSupported =>
        OperatingSystem.IsMacOS() || OperatingSystem.IsMacCatalyst()
        || OperatingSystem.IsLinux() || OperatingSystem.IsWindows();

    internal DesktopShellBackend(IShellBackend inner, IReadOnlyList<string> networkHosts)
    {
        _inner = inner;
        NetworkHosts = networkHosts;
    }

    internal IReadOnlyList<string> NetworkHosts { get; set; }

    public string Name => _inner.Name;
    public bool UsesHostProcesses => _inner.UsesHostProcesses;
    public ShellProgram? Shell => _inner.Shell;
    public ISkillSandbox? Sandbox => _inner.Sandbox;
    public bool CanRun => _inner.CanRun;
    public string? UnavailableReason => _inner.UnavailableReason;

    public bool TryStart(ShellLaunch launch, out IShellJob? job, out ConfinedResult failure)
    {
        ArgumentNullException.ThrowIfNull(launch);
        if ((launch.AllowNetwork || launch.AllowLoopbackPort is not null) && NetworkHosts.Count > 0)
        {
            job = null;
            failure = new ConfinedResult(false, false, -1, string.Empty, string.Empty, TimeSpan.Zero, "none",
                "Native process execution cannot enforce the configured networkHosts allow-list. "
                + "Disable network access, or remove the host restriction in settings to allow general network access.");
            return false;
        }
        return _inner.TryStart(launch, out job, out failure);
    }
}
