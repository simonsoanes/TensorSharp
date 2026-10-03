using System.Text.Json;
using System.Text.RegularExpressions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

// Metadata-only check: replay the actual generated token IDs from TS_CB_DEBUG
// and report where the next rendered turn stops matching the cache. No GPU load.
if (args.Length != 4)
    throw new ArgumentException("Usage: QwenCacheReplayProbe MODEL GGML_DEBUG_LOG REQUEST_ID USER_TEXT");
using var gguf = new GgufFile(args[0]);
var tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
string arch = gguf.GetString("general.architecture") ?? throw new InvalidDataException("Missing architecture.");
string template = gguf.GetString("tokenizer.chat_template") ?? throw new InvalidDataException("Missing chat template.");
var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
var history = new List<ChatMessage> { new() { Role = "user", Content = args[3] } };
List<int> original = renderer.RenderToTokens(tokenizer, template, history, arch, true,
    out _, out string whitespace, enableThinking: false);
using var logFile = new FileStream(args[1], FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
using var reader = new StreamReader(logFile);
string[] lines = reader.ReadToEnd().Split('\n');
var raw = new List<int>();
foreach (string line in lines)
{
    if (!line.Contains(args[2] + ":")) continue;
    Match match = Regex.Match(line, @"top1=(\d+):");
    if (match.Success) raw.Add(int.Parse(match.Groups[1].Value));
}
// The final synchronization forward does not sample; its debug top1 is not output.
int releaseCount = 0;
foreach (string line in lines)
{
    if (!line.Contains("release " + args[2])) continue;
    Match m = Regex.Match(line, @"computed=(\d+)/(\d+)");
    if (m.Success) releaseCount = int.Parse(m.Groups[2].Value) - original.Count;
}
if (releaseCount > 0 && releaseCount < raw.Count) raw.RemoveRange(releaseCount, raw.Count - releaseCount);
string suffix = KVCachePromptRenderer.GetAssistantGenerationSuffix(arch, false);
history.Add(new ChatMessage { Role = "assistant", Content = tokenizer.Decode(raw),
    RawOutputTokens = raw, RawPromptTrailingWhitespace = whitespace, RawGenerationSuffix = suffix });
history.Add(new ChatMessage { Role = "user", Content = "Please continue." });
List<int> next = renderer.RenderToTokens(tokenizer, template, history, arch, true, enableThinking: false);
var cache = original.Concat(raw).ToList();
int common = 0;
while (common < cache.Count && common < next.Count && cache[common] == next[common]) common++;
Console.WriteLine(JsonSerializer.Serialize(new { arch, originalTokens = original.Count, rawTokens = raw.Count,
    nextTokens = next.Count, common, expected = cache.Count, whitespace,
    originalText = tokenizer.Decode(original), replayText = tokenizer.Decode(next),
    originalTail = cache.Skip(Math.Max(0, common - 8)).Take(20),
    replayTail = next.Skip(Math.Max(0, common - 8)).Take(20) }, new JsonSerializerOptions { WriteIndented = true }));
if (common != cache.Count) Environment.ExitCode = 1;
