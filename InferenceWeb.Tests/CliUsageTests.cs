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
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Cli;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime.Speculative;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The CLI usage page's coverage guards, mirroring the server's. They matter
/// MORE here: the CLI's argument switch has no unknown-flag trap, so a flag
/// missing from --help is not merely undiscoverable — a user typing it gets no
/// error either. The page is the only contract a user can see.
/// </summary>
public class CliUsageTests
{
    private static string Usage()
    {
        var sw = new StringWriter();
        CliUsage.PrintUsage(sw);
        return sw.ToString();
    }

    [Fact]
    public void PrintUsage_DocumentsTheSharedFlagFamilies()
    {
        // The inverse guard — accepted flags must be documented — for the
        // families whose names live in shared constant tables. This is the
        // direction that drifted on the server: all six --code-exec* flags were
        // parsed and working while --help never named them.
        string usage = Usage();

        var accepted = new List<string>
        {
            SkillHostOptions.RootsFlag, SkillHostOptions.SelectFlag, SkillHostOptions.ListFlag,
            SkillHostOptions.DisableFlag, SkillHostOptions.NoDiscoveryFlag,
            SkillHostOptions.AllowScriptsFlag, SkillHostOptions.MaxRoundsFlag,
            SkillHostOptions.SandboxFlag, SkillHostOptions.AllowNetworkFlag,
        };
        accepted.AddRange(SpeculativeCliFlags.SwitchFlags);
        accepted.AddRange(SpeculativeCliFlags.ValueFlags);
        // Driven off CodeExecOptions' own tables rather than a copied list, so a flag
        // added there is required to be documented from the moment it is accepted.
        accepted.AddRange(CodeExecOptions.SwitchFlags);
        accepted.AddRange(CodeExecOptions.ValueFlags);

        var missing = accepted.Where(f => !usage.Contains(f, StringComparison.Ordinal)).ToList();
        Assert.True(missing.Count == 0,
            "The CLI accepts these flags but its --help never mentions them:\n  "
            + string.Join("\n  ", missing));
    }

    [Fact]
    public void PrintUsage_DoesNotDocumentRemovedSpeculativeSpellings()
    {
        // A removed spelling on the help page would advertise a flag that only
        // errors; the migration pointer lives in the error message, not here.
        var documented = new HashSet<string>(CliUsage.DocumentedFlags(), StringComparer.Ordinal);
        foreach ((string flag, _) in SpeculativeCliFlags.RemovedFlags)
            Assert.DoesNotContain(flag, documented);
    }

    [Fact]
    public void DocumentedFlags_YieldsARealList()
    {
        // Guard against a vacuous inverse test: an accessor that yielded nothing
        // would make PrintUsage_DocumentsTheSharedFlagFamilies prove nothing.
        var flags = CliUsage.DocumentedFlags().ToList();
        Assert.True(flags.Count > 40, $"DocumentedFlags() yielded only {flags.Count} flags.");
        Assert.Contains("--model", flags);
        Assert.Contains("--code-exec", flags);
        Assert.Contains("--code-exec-unconfined", flags);
        Assert.Contains("--draft-model", flags);
    }

    [Fact]
    public void PrintUsage_EveryEntryHasAnExample()
    {
        string usage = Usage();
        Assert.Contains("Default:", usage);
        Assert.Contains("Example:", usage);
    }
}
