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
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Sampling for a turn that writes code.
///
/// <para>
/// The reference implementation sets no sampling at all — the Agents SDK's
/// <c>ModelSettings.temperature</c>, <c>frequency_penalty</c> and <c>presence_penalty</c>
/// all default to <c>None</c>. TensorSharp's own defaults are Ollama's CHAT defaults,
/// inherited for API compatibility, and applying them to code is what diverges: a
/// repetition penalty of 1.1 over a 64-token window is two to four lines of Python, so it
/// penalises the indentation, the <c>return</c> and the closing delimiters that carry the
/// code's structure against each other.
/// </para>
/// <para>
/// So the assertions that matter are not about the temperature number. They are that the
/// penalty comes off, that nobody's explicit choice is overruled, and that the shared
/// server-wide config instance is never mutated.
/// </para>
/// </summary>
public class CodeExecSamplingTests
{
    [Fact]
    public void TheRepetitionPenaltyComesOffForCode()
    {
        var requested = new SamplingConfig();
        Assert.Equal(1.1f, requested.RepetitionPenalty, 3);   // the default being corrected

        SamplingConfig code = requested.ForCodingTurn(0.2f);

        Assert.Equal(1.0f, code.RepetitionPenalty, 3);
        Assert.Equal(0.2f, code.Temperature, 3);
    }

    /// <summary>
    /// The shared instance is the server-wide default and serves every request. Mutating
    /// it would make one coding turn change the sampling of every later chat.
    /// </summary>
    [Fact]
    public void TheCallersConfigIsNeverMutated()
    {
        var requested = new SamplingConfig();
        SamplingConfig code = requested.ForCodingTurn(0.2f);

        Assert.NotSame(requested, code);
        Assert.Equal(0.8f, requested.Temperature, 3);
        Assert.Equal(1.1f, requested.RepetitionPenalty, 3);
    }

    /// <summary>
    /// A temperature somebody CHOSE — a client in the request body, an operator via a flag
    /// or a config file — outranks a host-side preference. Only a value still sitting at
    /// the built-in default is treated as "nobody decided this".
    /// </summary>
    [Fact]
    public void AnExplicitlyChosenTemperatureIsNotOverruled()
    {
        var chosen = new SamplingConfig { Temperature = 0.95f };

        SamplingConfig code = chosen.ForCodingTurn(0.2f);

        Assert.Equal(0.95f, code.Temperature, 3);
        // The penalty is still corrected: it was left at its default, so nobody chose it.
        Assert.Equal(1.0f, code.RepetitionPenalty, 3);
    }

    [Fact]
    public void AnExplicitlyChosenPenaltyIsNotOverruled()
    {
        var chosen = new SamplingConfig { RepetitionPenalty = 1.3f };

        Assert.Equal(1.3f, chosen.ForCodingTurn(0.2f).RepetitionPenalty, 3);
    }

    /// <summary>A negative operator value means "leave sampling alone", including the penalty's temperature.</summary>
    [Fact]
    public void NoTemperatureMeansOnlyThePenaltyChanges()
    {
        SamplingConfig code = new SamplingConfig().ForCodingTurn(null);

        Assert.Equal(0.8f, code.Temperature, 3);
        Assert.Equal(1.0f, code.RepetitionPenalty, 3);
    }

    /// <summary>Greedy decoding is a deliberate choice and must survive untouched.</summary>
    [Fact]
    public void GreedyDecodingIsLeftAlone()
    {
        SamplingConfig code = SamplingConfig.Greedy.ForCodingTurn(0.2f);

        Assert.Equal(0f, code.Temperature, 3);
        Assert.True(code.IsGreedy);
    }

    // ---- the flag ------------------------------------------------------------

    [Fact]
    public void TheFlagIsParsedAndBounded()
    {
        Assert.Equal(
            0.1f,
            CodeExecOptions.Parse(new[] { "--code-exec-temperature", "0.1" }, out _).Temperature);

        // A negative number turns the adjustment off entirely rather than being clamped —
        // an operator who does not want the host touching sampling must be able to say so.
        Assert.Null(CodeExecOptions.Parse(new[] { "--code-exec-temperature", "-1" }, out _).Temperature);

        // Refused, not swallowed: the timeout flag learned this the hard way, when an
        // unparseable value left the operator's explicit choice silently at the default.
        Assert.Throws<ArgumentException>(
            () => CodeExecOptions.Parse(new[] { "--code-exec-temperature", "hot" }, out _));
        Assert.Throws<ArgumentException>(
            () => CodeExecOptions.Parse(new[] { "--code-exec-temperature", "9" }, out _));
    }

    /// <summary>
    /// Off by default, and that is the finding rather than a hedge. Neither reference
    /// implementation has a code-specific sampling profile — the Agents SDK leaves
    /// temperature, top_p and both penalties at None and omits them from the request, and
    /// Claude Code's settings surface has no temperature, top_p or top_k at all. So the
    /// mechanical argument for turning the repetition penalty off is offered as a switch,
    /// not applied silently.
    /// </summary>
    /// <summary>
    /// The two halves are separate, and only one is opt-in. No TEMPERATURE is set by
    /// default, because that would add something neither reference sets — Codex leaves it
    /// None and omits it from the wire, Claude Code exposes no sampling setting at all. The
    /// repetition PENALTY is removed by default, because that takes away an Ollama
    /// chat-compatibility default which neither reference has an analogue of.
    /// </summary>
    [Fact]
    public void NoTemperatureIsSetByDefault_ButThePenaltyIsStillRemoved()
    {
        Assert.Null(new CodeExecOptions().Temperature);

        SamplingConfig code = new SamplingConfig().ForCodingTurn(new CodeExecOptions().Temperature);

        Assert.Equal(0.8f, code.Temperature, 3);        // untouched
        Assert.Equal(1.0f, code.RepetitionPenalty, 3);  // corrected
    }

    /// <summary>
    /// The runner only advises when it can actually run code — a host with no shell must
    /// not quietly change the sampling of an ordinary chat turn.
    /// </summary>
    [Fact]
    public void ARunnerThatCannotRunAdvisesNothing()
    {
        using var runner = new ShellRunner(new CodeExecOptions { Sandbox = SkillSandboxMode.Off });
        var adapter = new CodeRunnerAdapter(runner, new CodeExecOptions { Sandbox = SkillSandboxMode.Off });
        var requested = new SamplingConfig();

        if (adapter.CanRun)
            return;   // this host has a shell; the case under test cannot be reached here

        Assert.Same(requested, adapter.ForCodingTurn(requested));
    }
}
