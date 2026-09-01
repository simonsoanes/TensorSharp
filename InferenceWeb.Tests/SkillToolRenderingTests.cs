// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;
using System.Linq;

namespace InferenceWeb.Tests;

/// <summary>
/// Runs the synthesized skill tools through the real renderers and the real output
/// parsers, one family at a time.
///
/// <para>
/// <c>skills_list</c> and <c>skills_read</c> are not written by the caller — TensorSharp
/// invents them and splices them into whatever tool list the request already carried, so
/// nobody outside this repository ever looks at the declaration and notices it came out
/// wrong. Each family formats a tool completely differently: Harmony writes a TypeScript
/// namespace, Qwen 3.5 writes a JSON schema plus an XML-ish call format, generic
/// ChatML writes the raw serialized list, and Gemma 4 writes
/// <c>&lt;|tool&gt;declaration:NAME{...}</c>
/// with its own quoting. A declaration that one of them cannot express does not fail —
/// the request succeeds, the model simply never calls a tool it was never shown, and
/// skills quietly stop working on that family alone.
/// </para>
/// <para>
/// The round trip matters as much as the rendering, which is why the parse direction is
/// asserted too. <c>skills_list</c> takes no parameters at all and <c>skills_read</c>
/// takes three flat scalars precisely so that both survive the trip out and back; a call
/// that comes back with the wrong name, or with its arguments collapsed, reaches
/// <see cref="SkillTools.Execute"/> as an unanswerable request and the agent loop burns
/// its whole round budget on it.
/// </para>
/// </summary>
public class SkillToolRenderingTests
{
    private static List<ToolFunction> Tools() => SkillTools.BuiltIn();

    private static List<ChatMessage> Ask() => new()
    {
        new ChatMessage { Role = "user", Content = "Fill in this PDF form." },
    };

    // ---- rendering ---------------------------------------------------------

    [Fact]
    public void RenderHarmony_DeclaresBothToolsInTheFunctionsNamespace()
    {
        string prompt = ChatTemplate.RenderHarmony(Ask(), addGenerationPrompt: true, tools: Tools());

        Assert.Contains("namespace functions {", prompt, StringComparison.Ordinal);
        Assert.Contains(SkillTools.ListToolName, prompt, StringComparison.Ordinal);
        Assert.Contains(SkillTools.ReadToolName, prompt, StringComparison.Ordinal);

        // A parameterless tool has to come out as the arrow form, not as an empty
        // object literal the model would try to fill in.
        Assert.Contains("type skills_list = () => any;", prompt, StringComparison.Ordinal);

        // The required/optional split has to survive: skill and path carry no '?',
        // offset does, or the model is told it must page every read.
        Assert.Contains("skill: string,", prompt, StringComparison.Ordinal);
        Assert.Contains("offset?: number,", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void RenderQwen35_DeclaresBothToolsInItsToolsBlock()
    {
        string prompt = ChatTemplate.RenderQwen35(Ask(), addGenerationPrompt: true, enableThinking: false, tools: Tools());

        Assert.Contains("<tools>", prompt, StringComparison.Ordinal);
        Assert.Contains("\"name\": \"skills_list\"", prompt, StringComparison.Ordinal);
        Assert.Contains("\"name\": \"skills_read\"", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void RenderGemma4_DeclaresBothToolsInTheSystemTurn()
    {
        string prompt = ChatTemplate.RenderGemma4(Ask(), addGenerationPrompt: true, tools: Tools());

        // The name is spliced into the markup unescaped, which is why the declarations
        // keep to [a-z_]; this pins that the splice actually happens.
        Assert.Contains("<|tool>declaration:skills_list{", prompt, StringComparison.Ordinal);
        Assert.Contains("<|tool>declaration:skills_read{", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void RenderChatMl_DeclaresBothTools()
    {
        string prompt = ChatTemplate.RenderChatMl(Ask(), addGenerationPrompt: true, tools: Tools());

        Assert.Contains(SkillTools.ListToolName, prompt, StringComparison.Ordinal);
        Assert.Contains(SkillTools.ReadToolName, prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void RenderWithoutTools_MentionsNeitherSkillTool()
    {
        // The control. Without it, a renderer that hardcoded the word "skills_read"
        // somewhere in its preamble would satisfy every assertion above.
        string prompt = ChatTemplate.RenderHarmony(Ask(), addGenerationPrompt: true, tools: null);

        Assert.DoesNotContain(SkillTools.ListToolName, prompt, StringComparison.Ordinal);
        Assert.DoesNotContain(SkillTools.ReadToolName, prompt, StringComparison.Ordinal);
    }

    // ---- parsing back --------------------------------------------------------

    [Fact]
    public void HarmonyParser_RecoversASkillsReadCall()
    {
        var parser = new HarmonyOutputParser();
        parser.Init(enableThinking: true, tools: Tools());

        ParsedOutput parsed = parser.Add(
            "<|channel|>analysis<|message|>I should read the skill first.<|end|>"
            + "<|start|>assistant<|channel|>commentary to=functions.skills_read <|constrain|>json"
            + "<|message|>{\"skill\":\"pdf\",\"path\":\"references/api.md\",\"offset\":512}",
            done: true);

        ToolCall call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal(SkillTools.ReadToolName, call.Name);
        Assert.Equal("pdf", call.Arguments["skill"]);
        Assert.Equal("references/api.md", call.Arguments["path"]);

        // Straight into the executor, which is the only thing the loop does with it.
        Assert.Equal(512L, SkillTools.ReadInt64(call, "offset"));
    }

    [Fact]
    public void ChatMlParser_RecoversASkillsReadCall()
    {
        var parser = new ChatMlOutputParser();
        parser.Init(enableThinking: false, tools: Tools());

        ParsedOutput parsed = parser.Add(
            "<tool_call>{\"name\":\"skills_read\",\"arguments\":{\"skill\":\"pdf\",\"path\":\"SKILL.md\"}}</tool_call>",
            done: true);

        ToolCall call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal(SkillTools.ReadToolName, call.Name);
        Assert.Equal("pdf", call.Arguments["skill"]);
        Assert.Equal("SKILL.md", call.Arguments["path"]);
        Assert.True(SkillTools.IsSkillTool(call.Name));
    }

    [Fact]
    public void Qwen35Parser_RecoversASkillsReadCallInTheXmlCallFormat()
    {
        // Qwen 3.5's own template tells the model to answer in this shape rather than
        // with a JSON object, so this is the form the parser actually sees in production.
        var parser = new Qwen35OutputParser();
        parser.Init(enableThinking: false, tools: Tools());

        ParsedOutput parsed = parser.Add(
            "<tool_call>\n<function=skills_read>\n"
            + "<parameter=skill>\npdf\n</parameter>\n"
            + "<parameter=path>\nreferences/api.md\n</parameter>\n"
            + "</function>\n</tool_call>",
            done: true);

        ToolCall call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal(SkillTools.ReadToolName, call.Name);
        Assert.Equal("pdf", call.Arguments["skill"]);
        Assert.Equal("references/api.md", call.Arguments["path"]);
    }

    [Fact]
    public void Gemma4Parser_RecoversASkillsReadCall()
    {
        var parser = new Gemma4OutputParser();
        parser.Init(enableThinking: false, tools: Tools());

        ParsedOutput parsed = parser.Add(
            "<|tool_call>call:skills_read{path:<|\"|>SKILL.md<|\"|>,skill:<|\"|>pdf<|\"|>}<tool_call|>",
            done: true);

        ToolCall call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal(SkillTools.ReadToolName, call.Name);
        Assert.Equal("pdf", call.Arguments["skill"]);
        Assert.Equal("SKILL.md", call.Arguments["path"]);
    }

    [Fact]
    public void Gemma4Parser_RecoversAParameterlessSkillsListCall()
    {
        // skills_list takes no arguments, and a call with an empty argument object is
        // the shape most likely to be dropped as unparseable.
        var parser = new Gemma4OutputParser();
        parser.Init(enableThinking: false, tools: Tools());

        ParsedOutput parsed = parser.Add("<|tool_call>call:skills_list{}<tool_call|>", done: true);

        ToolCall call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal(SkillTools.ListToolName, call.Name);
        Assert.Empty(call.Arguments);
    }

    [Fact]
    public void EveryFamilysParsedCall_IsRecognisedAsATensorSharpAnsweredTool()
    {
        // The routing decision the agent loop makes on every turn: a call the loop does
        // not recognise as a skill tool is handed back to the client, which has no
        // implementation for it and stalls the conversation.
        foreach (ToolFunction tool in SkillTools.BuiltIn(allowScripts: true))
            Assert.True(SkillTools.IsSkillTool(tool.Name), tool.Name);

        Assert.False(SkillTools.IsSkillTool("skills_readx"));
        Assert.False(SkillTools.IsSkillTool("SKILLS_READ"));   // matched ordinally, on purpose
    }
}
