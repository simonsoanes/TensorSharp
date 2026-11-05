// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;

namespace InferenceWeb.Tests;

/// <summary>
/// LFM3's chat protocol: the GGUF template opens the reasoning block in the generation
/// prompt (<c>&lt;think&gt;</c>, or the closed <c>&lt;think&gt;&lt;/think&gt;</c> with
/// thinking off), so without a parser a thinking-on reply streamed its reasoning and the
/// <c>&lt;/think&gt;</c> as the answer; and the conv state cannot be rewound, so a
/// follow-up whose re-rendered history left the cache re-prefilled the conversation.
///
/// <para><see cref="Template"/> keeps the published template's framing - the ChatML
/// turns, the empty block it writes into past turns, the XML call body and the
/// generation prompt - and drops what these tests do not reach (media, deferred tools,
/// the <c>&lt;additional_tools&gt;</c> forms). The full template rendered to the same
/// prompt token counts as llama.cpp against the real checkpoint (docs/models/lfm3.md).</para>
/// </summary>
public class Lfm3ChatProtocolTests
{
    private const string Arch = "lfm3";

    private const string Template =
        "{{- bos_token -}}" +
        "{%- set enable_thinking = enable_thinking | default(true) -%}" +
        "{%- set ns = namespace(system_prompt=\"\", last_user_index=-1) -%}" +
        "{%- if messages and messages[0][\"role\"] == \"system\" -%}" +
        "{%- set ns.system_prompt = messages[0][\"content\"] -%}{%- set messages = messages[1:] -%}" +
        "{%- endif -%}" +
        "{%- if tools -%}" +
        "{%- set tools_ns = namespace(lines=[]) -%}" +
        "{%- for tool in tools -%}{%- set tools_ns.lines = tools_ns.lines + [tool | tojson] -%}{%- endfor -%}" +
        "{%- set ns.system_prompt = ns.system_prompt + (\"\\n\\n\" if ns.system_prompt else \"\") + \"<tools>\\n\" + (tools_ns.lines | join(\"\\n\")) + \"\\n</tools>\" -%}" +
        "{%- endif -%}" +
        "{%- if ns.system_prompt -%}{{- \"<|im_start|>system\\n\" + ns.system_prompt + \"<|im_end|>\\n\" -}}{%- endif -%}" +
        "{%- for message in messages -%}{%- if message[\"role\"] == \"user\" -%}{%- set ns.last_user_index = loop.index0 -%}{%- endif -%}{%- endfor -%}" +
        "{%- for message in messages -%}" +
        "{{- \"<|im_start|>\" + message.role + \"\\n\" -}}" +
        "{%- if message.role == \"assistant\" -%}" +
        "{%- set keep_thinking = loop.index0 > ns.last_user_index -%}" +
        "{%- set content = message.get(\"content\") or \"\" -%}" +
        "{%- if keep_thinking and \"</think>\" not in content -%}{{- \"<think></think>\" -}}{%- endif -%}" +
        "{%- if not keep_thinking -%}" +
        "{%- if \"</think>\" in content -%}{%- set content = content.split(\"</think>\")[-1] | trim -%}{%- endif -%}" +
        "{%- set content = \"<think></think>\" + content -%}" +
        "{%- endif -%}" +
        "{{- content -}}" +
        "{%- if message.tool_calls -%}{%- for tool_call in message.tool_calls -%}" +
        "{{- \"<tool_call>\\n<function=\" + tool_call.function.name + \">\\n\" -}}" +
        "{%- for arg_name, arg_value in tool_call.function.arguments.items() -%}" +
        "{{- \"<parameter=\" + arg_name + \">\\n\" + arg_value + \"\\n</parameter>\\n\" -}}" +
        "{%- endfor -%}" +
        "{{- \"</function>\\n</tool_call>\" -}}" +
        "{%- endfor -%}{%- endif -%}" +
        "{{- \"<|im_end|>\\n\" -}}" +
        "{%- elif message.role == \"tool\" -%}" +
        "{{- \"<tool_response>\\n\" + message.content + \"\\n</tool_response><|im_end|>\\n\" -}}" +
        "{%- else -%}" +
        "{{- message.content + \"<|im_end|>\\n\" -}}" +
        "{%- endif -%}" +
        "{%- endfor -%}" +
        "{%- if add_generation_prompt -%}" +
        "{{- \"<|im_start|>assistant\\n\" -}}" +
        "{%- if enable_thinking -%}{{- \"<think>\" -}}{%- else -%}{{- \"<think></think>\" -}}{%- endif -%}" +
        "{%- endif -%}";

    /// <summary>One token per character, so a token index is a character index.</summary>
    private sealed class CharacterTokenizer : ITokenizer
    {
        public string[] Vocab => [];
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public int VocabSize => char.MaxValue + 1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> ids) => new(ids.Select(i => (char)i).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer)
            => buffer.AddRange(Encoding.UTF8.GetBytes(((char)tokenId).ToString()));
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 ? tokenStr[0] : -1;
    }

    private static readonly List<ToolFunction> WeatherTool =
    [
        new ToolFunction
        {
            Name = "get_weather",
            Description = "Current weather for a city",
            Parameters = new Dictionary<string, ToolParameter>
            {
                ["city"] = new ToolParameter { Type = "string", Description = "City name" },
            },
            Required = ["city"],
        },
    ];

    [Fact]
    public void GenerationSuffixIsTheTemplatesOpenOrClosedBlock()
    {
        Assert.Equal("<think>", KVCachePromptRenderer.GetAssistantGenerationSuffix(Arch, true));
        Assert.Equal("<think></think>", KVCachePromptRenderer.GetAssistantGenerationSuffix(Arch, false));

        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var tokenizer = new CharacterTokenizer();
        List<ChatMessage> history = [new() { Role = "user", Content = "Hi" }];
        foreach (bool thinking in new[] { true, false })
        {
            string prompt = tokenizer.Decode(renderer.RenderToTokens(
                tokenizer, Template, history, Arch, true, enableThinking: thinking));
            Assert.EndsWith("<|im_start|>assistant\n" + KVCachePromptRenderer.GetAssistantGenerationSuffix(Arch, thinking), prompt);
        }
    }

    [Fact]
    public void ThinkingOnReplySeparatesReasoningFromTheAnswer()
    {
        IOutputParser parser = ChatProtocolRegistry.For(Arch)!.CreateOutputParser!();
        parser.Init(enableThinking: true, tools: null);
        ParsedOutput parsed = parser.Add("The user asks for three primary colours.</think>Red, blue, and yellow.", done: true);

        Assert.Equal("The user asks for three primary colours.", parsed.Thinking.Trim());
        Assert.Equal("Red, blue, and yellow.", parsed.Content.Trim());
        Assert.DoesNotContain("</think>", parsed.Content);
    }

    [Fact]
    public void ThinkingOffReplyIsAllAnswer()
    {
        IOutputParser parser = ChatProtocolRegistry.For(Arch)!.CreateOutputParser!();
        parser.Init(enableThinking: false, tools: null);
        ParsedOutput parsed = parser.Add("7 times 8 is 56.", done: true);

        Assert.Equal("7 times 8 is 56.", parsed.Content.Trim());
        Assert.Equal("", parsed.Thinking);
    }

    [Fact]
    public void XmlToolCallAfterReasoningParses()
    {
        IOutputParser parser = ChatProtocolRegistry.For(Arch)!.CreateOutputParser!();
        parser.Init(enableThinking: true, tools: WeatherTool);
        ParsedOutput parsed = parser.Add(
            "The user wants the weather in Paris.</think><tool_call>\n<function=get_weather>\n" +
            "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", done: true);

        ToolCall call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]?.ToString());
        Assert.Equal("The user wants the weather in Paris.", parsed.Thinking.Trim());
        Assert.Equal("", parsed.Content.Trim());
    }

    /// <summary>
    /// The follow-up prompt starts with everything the cache holds - the first prompt and
    /// the reply's raw tokens - in both thinking modes. A divergence anywhere before the
    /// reply's end would force a re-prefill: the conv state cannot be truncated back to it.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void FollowUpExtendsEverythingTheCacheHolds(bool thinking)
    {
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var tokenizer = new CharacterTokenizer();
        List<ChatMessage> history =
        [
            new() { Role = "system", Content = "You are terse." },
            new() { Role = "user", Content = "Name three primary colours." },
        ];
        List<int> cache = renderer.RenderToTokens(tokenizer, Template, history, Arch, true, enableThinking: thinking);

        string reasoning = thinking ? "Red, blue, yellow.</think>" : "";
        List<int> generated = tokenizer.Encode(reasoning + "Red, blue and yellow.", addSpecial: false);
        cache.AddRange(generated);
        history.Add(new ChatMessage
        {
            Role = "assistant",
            Content = "Red, blue and yellow.",
            Thinking = thinking ? "Red, blue, yellow." : null,
            RawOutputTokens = generated,
        });
        history.Add(new ChatMessage { Role = "user", Content = "And secondary?" });

        List<int> followUp = renderer.RenderToTokens(tokenizer, Template, history, Arch, true, enableThinking: thinking);

        Assert.True(followUp.Count > cache.Count);
        Assert.Equal(tokenizer.Decode(cache), tokenizer.Decode(followUp.Take(cache.Count).ToList()));
    }

    /// <summary>
    /// A tool round re-renders as its raw tokens (reasoning and call included), and the
    /// result after it renders from the tool message alone.
    /// </summary>
    [Fact]
    public void ToolRoundExtendsEverythingTheCacheHolds()
    {
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var tokenizer = new CharacterTokenizer();
        List<ChatMessage> history = [new() { Role = "user", Content = "Weather in Paris?" }];
        List<int> cache = renderer.RenderToTokens(tokenizer, Template, history, Arch, true, WeatherTool, enableThinking: true);

        const string callText = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        List<int> generated = tokenizer.Encode("Look it up.</think>" + callText, addSpecial: false);
        cache.AddRange(generated);
        history.Add(new ChatMessage
        {
            Role = "assistant",
            Content = "",
            Thinking = "Look it up.",
            ToolCalls = [new ToolCall { Id = "call_0", Name = "get_weather", Arguments = new() { ["city"] = "Paris" } }],
            RawOutputTokens = generated,
        });
        history.Add(new ChatMessage { Role = "tool", ToolCallId = "call_0", Content = "{\"temperature\": 18}" });

        List<int> next = renderer.RenderToTokens(tokenizer, Template, history, Arch, true, WeatherTool, enableThinking: true);
        string nextText = tokenizer.Decode(next);

        Assert.Equal(tokenizer.Decode(cache), tokenizer.Decode(next.Take(cache.Count).ToList()));
        Assert.Contains("<tool_response>\n{\"temperature\": 18}\n</tool_response>", nextText);
        Assert.EndsWith("<|im_start|>assistant\n<think>", nextText);
    }
}
