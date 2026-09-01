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
using TensorSharp.Runtime;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Keeps <see cref="SkillCapabilities"/> honest about what a family can actually do.
///
/// <para>
/// Offering a tool is two halves decided in two places: <see cref="ChatProtocolRegistry"/>
/// says whether the renderer writes the declaration, and <see cref="OutputParserFactory"/>
/// decides what reads the reply. Nothing structural forces them to agree, and when they
/// disagree the failure is silent and total — TensorSharp declares <c>skills_read</c>,
/// the model calls it, no parser extracts the call, nobody answers it, and the raw tool
/// markup reaches the user as if it were the answer. Progressive disclosure simply does
/// not happen, and nothing logs a complaint.
/// </para>
/// <para>
/// That is exactly what shipped: <c>qwen4exp</c> is registered with no
/// <c>CreateOutputParser</c>, so it got <see cref="PassthroughOutputParser"/> and claimed
/// full tool support anyway; every architecture with no table entry did the same. These
/// tests are the standing check that a new family cannot reintroduce it.
/// </para>
/// </summary>
public class SkillCapabilityConsistencyTests
{
    /// <summary>Every architecture the protocol table knows, plus families it does not.</summary>
    public static TheoryData<string> Architectures
    {
        get
        {
            var data = new TheoryData<string>();
            foreach (string arch in ChatProtocolRegistry.All.SelectMany(p => p.Architectures).Distinct())
                data.Add(arch);
            // Unregistered, so they take the generic path: two common families and a
            // name that cannot exist. (qwen2vl used to head this list; it is a
            // registered protocol now and arrives via the registry sweep above.)
            foreach (string arch in new[] { "llama", "phi3", "not-a-real-architecture" })
                data.Add(arch);
            return data;
        }
    }

    [Theory]
    [MemberData(nameof(Architectures))]
    public void ToolsAreOnlyOfferedWhenSomethingCanParseTheCallBack(string architecture)
    {
        SkillModelCapabilities caps = SkillCapabilities.For(architecture);
        IOutputParser parser = OutputParserFactory.Create(architecture);

        if (!parser.HasToolSupport)
        {
            Assert.False(caps.ToolsRendered,
                $"'{architecture}' is parsed by {parser.GetType().Name}, which never extracts a tool " +
                "call, so declaring skills_read to it can only produce markup the user sees as the answer");
        }
    }

    [Fact]
    public void AFamilyWithNoProtocolEntry_GetsNoTools_RatherThanOptimisticFullSupport()
    {
        // The generic path is PassthroughOutputParser, which returns every byte as
        // content. Claiming tool support here was the original bug. Tool RESULTS are a
        // separate question — a renderer we know nothing about is assumed to carry them,
        // and it does not matter either way while no tools are offered.
        Assert.False(SkillCapabilities.For("not-a-real-architecture").ToolsRendered);
    }

    [Fact]
    public void Qwen25Family_IsRegistered_WithToolsButNoThinking()
    {
        // qwen2vl used to be this suite's example of an unregistered family, and its
        // chats got no tools at all. It speaks the same ChatML tool syntax as Qwen3,
        // so it is registered now — with thinking pinned off, because it has no
        // <think> channel for the parser to wait on.
        Assert.True(SkillCapabilities.For("qwen2vl").ToolsRendered);
        Assert.True(SkillCapabilities.For("qwen2").ToolsRendered);
        Assert.False(OutputParserFactory.Create("qwen2vl").HasThinkingSupport);
    }

    [Fact]
    public void Qwen4Exp_IsRegisteredButUnparseable_SoItsSkillBodiesAreInlinedInstead()
    {
        // Registered, and RendersToolDeclarations defaults to true — but the entry has no
        // CreateOutputParser, so nothing would read the call back.
        Assert.NotNull(ChatProtocolRegistry.For("qwen4exp"));
        Assert.False(OutputParserFactory.Create("qwen4exp").HasToolSupport);
        Assert.False(SkillCapabilities.For("qwen4exp").ToolsRendered);
    }

    [Theory]
    [InlineData("gemma4")]
    [InlineData("qwen35")]
    [InlineData("gpt-oss")]
    [InlineData("muse-glimmer")]
    [InlineData("nemotron_h_moe")]
    [InlineData("deepseek4")]
    [InlineData("glm-dsa")]
    [InlineData("qwen3")]
    public void TheFamiliesThatCanDoTheRoundTrip_StillOfferTools(string architecture)
    {
        // The guard above must not have quietly disabled skills everywhere.
        Assert.True(SkillCapabilities.For(architecture).ToolsRendered,
            $"'{architecture}' can parse tool calls and must still be offered them");
    }

    /// <summary>
    /// The consequence, asserted where it actually bites: the request plan.
    ///
    /// <para>
    /// This is the test that would have caught the bug. The capability lookup is only
    /// interesting because <see cref="SkillRequestPlan"/> reads it to decide whether to
    /// splice <c>skills_read</c> into the request — so assert on the plan, not on the
    /// lookup. Before the fix this plan offered tools to a model whose replies nothing
    /// would parse.
    /// </para>
    /// </summary>
    [Fact]
    public void AnUnparseableFamily_GetsAPlanWithNoToolsAndAnInlinedBody()
    {
        string dir = Path.Combine(Path.GetTempPath(), "ts-cap-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string skill = Path.Combine(dir, "acme");
            Directory.CreateDirectory(skill);
            File.WriteAllText(Path.Combine(skill, "SKILL.md"),
                "---\nname: acme\ndescription: Formats ACME invoice numbers.\n---\n\nThe format is ACME-<region>.\n");
            File.WriteAllText(Path.Combine(skill, "reference.md"), "EMA, APC, NAM\n");

            var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { dir } });
            ServerHostingOptions options = ServerOptionsBuilder.Build(
                new[] { "--model", "x.gguf", "--skills-dir", dir }, dir);

            SkillRequestPlan plan = SkillRequestPlan.Create(
                registry, new[] { "acme" }, discovery: false, clientTools: null,
                architecture: "not-a-real-architecture", contextTokens: 32768, options,
                out IReadOnlyList<string> unknown);

            Assert.Empty(unknown);
            Assert.NotNull(plan);
            Assert.False(plan.ToolsOffered,
                "a family whose replies are parsed by PassthroughOutputParser must not be offered skills_read");
            Assert.DoesNotContain("skills_read", plan.Prompt.Instructions, StringComparison.Ordinal);
            // The body is still delivered — losing the tools must not lose the skill.
            Assert.Contains("ACME-<region>", plan.Prompt.Instructions, StringComparison.Ordinal);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { /* best effort */ }
        }
    }

    [Fact]
    public void GemmaThreeAndMistral_KeepTheirExistingOptOut()
    {
        Assert.False(SkillCapabilities.For("gemma3").ToolsRendered);
        Assert.False(SkillCapabilities.For("mistral3").ToolsRendered);
        Assert.False(SkillCapabilities.For("mistral3").ToolResultsRendered);
    }
}
