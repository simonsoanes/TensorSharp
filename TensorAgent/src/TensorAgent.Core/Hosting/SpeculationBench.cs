// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace TensorAgent.Core.Hosting;

/// <summary>
/// On-device benchmark: plain decoding against speculative decoding, on this phone,
/// through the app's own chat service.
///
/// <para>
/// Launched with <c>TENSORAGENT_SPEC_BENCH=1</c> (and a model, via
/// <c>TENSORAGENT_USE_MODEL</c> or the remembered choice). It waits for the load and
/// the prefix warm-up, then runs the same four turns under each mode - plain, then
/// speculative, then plain and speculative again so a thermal drift shows up as a
/// disagreement between the two halves rather than as a result - switching the
/// engine's policy in place, without a reload. The turns are the shapes that decide
/// whether speculation pays: a one-word answer (the thinking preamble, and the
/// first-token time), free prose (where a lookup drafter finds little), a file
/// quoted back from the prompt, and the same text quoted from the model's own
/// previous answer.
/// </para>
/// <para>
/// One <c>specbench</c> line per turn - prompt tokens, tokens served from the cache,
/// first-token time, generated tokens, and the prefill and decode rates derived from
/// them - on stdout and in <c>Library/Caches/TensorAgent/logs/specbench.log</c>, which
/// <c>scripts/bench-spec-device.sh</c> pulls back, plus a summary comparing the
/// modes per turn. The same code runs on the Mac host benchmark's process, so a
/// number from the phone and a number from the Mac mean the same thing.
/// </para>
/// </summary>
public sealed class SpeculationBench
{
    public const string EnableVariable = "TENSORAGENT_SPEC_BENCH";
    public const string RunVariable = "TENSORAGENT_SPEC_BENCH_RUN";
    public const string TokensVariable = "TENSORAGENT_SPEC_BENCH_TOKENS";
    public const string ModesVariable = "TENSORAGENT_SPEC_BENCH_MODES";
    /// <summary>
    /// Seconds to wait after the cache is warm before the first turn. A desktop app is
    /// still bringing its window and page up for several seconds after a warm-cache load,
    /// and a benchmark that starts inside that window measures the launch.
    /// </summary>
    public const string DelayVariable = "TENSORAGENT_SPEC_BENCH_DELAY";

    public static bool Requested =>
        string.Equals(Environment.GetEnvironmentVariable(EnableVariable), "1", StringComparison.Ordinal);

    private readonly AgentAppHost _app;
    private readonly string _logPath;
    private readonly List<Row> _rows = new();

    public SpeculationBench(AgentAppHost app)
    {
        _app = app ?? throw new ArgumentNullException(nameof(app));
        _logPath = Path.Combine(app.Paths.LogsDirectory, "specbench.log");
    }

    private sealed record Row(string Mode, int Pass, string Label, int Prompt, int Reused, double TtftMs,
        int Tokens, double ElapsedMs, double PrefillTps, double DecodeTps, string? Error);

    private void Say(string line)
    {
        Console.WriteLine("TensorAgent: " + line);
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_logPath)!);
            File.AppendAllText(_logPath, $"{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss.fff} {line}{Environment.NewLine}");
        }
        catch (Exception) { /* diagnostic only */ }
    }

    public async Task RunAsync(CancellationToken token)
    {
        try
        {
            string run = Environment.GetEnvironmentVariable(RunVariable) is { Length: > 0 } r ? r : "unnamed";
            int tokens = int.TryParse(Environment.GetEnvironmentVariable(TokensVariable), NumberStyles.Integer,
                CultureInfo.InvariantCulture, out int t) && t > 0 ? t : 160;
            string[] modes = (Environment.GetEnvironmentVariable(ModesVariable) is { Length: > 0 } m ? m : "plain,spec,plain,spec")
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            Say($"specbench run {run}: tokens {tokens}, modes {string.Join(",", modes)}");

            for (int i = 0; i < 300 && _app.ModelLoad != AgentAppHost.ModelLoadState.Loaded; i++)
                await Task.Delay(1000, token).ConfigureAwait(false);
            if (_app.ModelLoad != AgentAppHost.ModelLoadState.Loaded)
            {
                Say($"specbench FAIL no model (state {_app.ModelLoad})");
                return;
            }
            // As a user does: wait for the warm-up rather than cancel it with a turn.
            for (int i = 0; i < 180 && !_app.PrefixCacheIsWarm; i++)
                await Task.Delay(1000, token).ConfigureAwait(false);
            Say(_app.PrefixCacheIsWarm ? "specbench cache warm" : "specbench cache NOT warm (warm-up did not finish)");
            if (double.TryParse(Environment.GetEnvironmentVariable(DelayVariable), NumberStyles.Float,
                    CultureInfo.InvariantCulture, out double delay) && delay > 0)
            {
                Say($"specbench waiting {delay:0.#}s before the first turn");
                await Task.Delay(TimeSpan.FromSeconds(delay), token).ConfigureAwait(false);
            }
            Say($"specbench model {_app.ModelService.LoadedModelName} on {_app.ModelService.LoadedBackend}; "
                + $"draft head {(_app.DraftHeadAttached ? "attached" : "none")}; {ProcessMemory.Describe()}");

            string quoted = QuotedText();
            var passes = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (string mode in modes)
            {
                bool speculative = string.Equals(mode, "spec", StringComparison.OrdinalIgnoreCase);
                int pass = passes.TryGetValue(mode, out int seen) ? seen + 1 : 1;
                passes[mode] = pass;
                Say($"specbench mode {mode} pass {pass}: {_app.SetSpeculationEnabled(speculative)}");

                string session = NewSession();
                var history = new List<object>();
                await TurnAsync(mode, pass, "1 one word", history, session, "Say the single word: apple.", 32, token);
                await TurnAsync(mode, pass, "2 prose", history, session, "Write three short paragraphs about the sea.", tokens, token);
                await TurnAsync(mode, pass, "3 quote prompt", history, session,
                    $"Repeat the following text exactly, character for character:\n```\n{quoted}```", tokens + 60, token);
                await TurnAsync(mode, pass, "4 quote own answer", history, session,
                    "Now repeat that same text once more, exactly.", tokens + 60, token);
            }
            Summarise();
            Say("specbench done");
        }
        catch (OperationCanceledException)
        {
            Say("specbench cancelled");
        }
        catch (Exception ex)
        {
            Say("specbench FAIL " + ex.Message);
        }
    }

    private string NewSession()
    {
        object created = _app.Chat.CreateSession();
        return created.GetType().GetProperty("sessionId")?.GetValue(created) as string ?? string.Empty;
    }

    private async Task TurnAsync(string mode, int pass, string label, List<object> history, string session,
        string question, int maxTokens, CancellationToken token)
    {
        history.Add(new { role = "user", content = question });
        bool think = _app.Settings.Load().ThinkByDefault;
        string json = JsonSerializer.Serialize(new { sessionId = session, messages = history.ToArray(), maxTokens, think });
        JsonElement body = JsonDocument.Parse(json).RootElement;

        // The phone takes the GPU away from a backgrounded app; wait like a turn does.
        await _app.Compute.WaitAsync(token).ConfigureAwait(false);

        var clock = Stopwatch.StartNew();
        double ttftMs = 0;
        int prompt = 0, reused = 0, count = 0;
        double elapsedSeconds = 0;
        string? error = null;
        var answer = new StringBuilder();
        await foreach (object frame in _app.Chat.ChatStreamAsync(body, token).ConfigureAwait(false))
        {
            string? tok = AgentAppHost.StringIn(frame, "token");
            bool thinking = AgentAppHost.StringIn(frame, "thinking") is not null;
            if (tok is not null || thinking)
            {
                if (ttftMs == 0) ttftMs = clock.Elapsed.TotalMilliseconds;
                if (tok is not null) answer.Append(tok);
            }
            if (AgentAppHost.StringIn(frame, "replace") is { } replaced)
                answer.Clear().Append(replaced);
            error ??= AgentAppHost.StringIn(frame, "error");
            if (AgentAppHost.ValueIn(frame, "done") is true)
            {
                prompt = AgentAppHost.IntIn(frame, "promptTokens");
                reused = AgentAppHost.IntIn(frame, "kvReusedTokens");
                count = AgentAppHost.IntIn(frame, "tokenCount");
                elapsedSeconds = AgentAppHost.DoubleIn(frame, "elapsed");
            }
        }
        clock.Stop();
        double elapsedMs = elapsedSeconds > 0 ? elapsedSeconds * 1000 : clock.Elapsed.TotalMilliseconds;
        history.Add(new { role = "assistant", content = answer.ToString() });

        int fresh = Math.Max(0, prompt - reused);
        double prefillTps = ttftMs > 0 ? fresh / (ttftMs / 1000.0) : 0;
        double decodeMs = elapsedMs - ttftMs;
        double decodeTps = count > 1 && decodeMs > 0 ? (count - 1) / (decodeMs / 1000.0) : 0;
        var row = new Row(mode, pass, label, prompt, reused, ttftMs, count, elapsedMs, prefillTps, decodeTps, error);
        _rows.Add(row);
        Say(error is { Length: > 0 }
            ? $"specbench {mode}#{pass} {label} FAILED: {error}"
            : $"specbench {mode}#{pass} {label}: prompt {prompt} reused {reused} fresh {fresh} ttftMs {ttftMs:0} "
              + $"tokens {count} elapsedMs {elapsedMs:0} prefillTps {prefillTps:0} decodeTps {decodeTps:0.0} "
              + $"text \"{Shorten(answer.ToString(), 60)}\"");
    }

    private void Summarise()
    {
        Say("specbench summary (per turn: mode -> mean decode tok/s, mean prefill tok/s, mean first token ms over the passes)");
        foreach (var byLabel in _rows.Where(r => r.Error is null).GroupBy(r => r.Label))
        {
            var parts = new List<string>();
            double plain = 0, spec = 0;
            foreach (var byMode in byLabel.GroupBy(r => r.Mode))
            {
                double decode = byMode.Average(r => r.DecodeTps);
                double prefill = byMode.Average(r => r.PrefillTps);
                double ttft = byMode.Average(r => r.TtftMs);
                if (byMode.Key == "plain") plain = decode; else spec = decode;
                parts.Add($"{byMode.Key} decode {decode:0.0} prefill {prefill:0} ttft {ttft:0}");
            }
            string ratio = plain > 0 && spec > 0 ? $" | spec/plain decode {spec / plain:0.00}x" : string.Empty;
            Say($"specbench summary {byLabel.Key}: {string.Join(" ; ", parts)}{ratio}");
        }
    }

    /// <summary>~600 characters of source, deterministic, so the quote turns have the
    /// same target on every device and every run.</summary>
    public static string QuotedText()
    {
        var sb = new StringBuilder();
        string[] lines =
        {
            "using System;",
            "using System.Collections.Generic;",
            "namespace Demo.Tools",
            "{",
            "    public sealed class Widget{N} : IDisposable",
            "    {",
            "        private readonly List<int> _items = new();",
            "        public int Count => _items.Count;",
            "        public void Add(int value) { if (value < 0) throw new ArgumentOutOfRangeException(nameof(value)); _items.Add(value * {N}); }",
            "        public int Sum() { int s = 0; foreach (int v in _items) s += v; return s; }",
            "        public void Dispose() { _items.Clear(); }",
            "    }",
            "}",
        };
        for (int i = 1; i <= 18; i++)
            sb.Append(i.ToString(CultureInfo.InvariantCulture).PadLeft(3)).Append("  ")
              .Append(lines[i % lines.Length].Replace("{N}", i.ToString(CultureInfo.InvariantCulture))).Append('\n');
        return sb.ToString();
    }

    private static string Shorten(string s, int max)
    {
        s = s.Replace("\n", "\\n");
        return s.Length <= max ? s : s.Substring(0, max) + "...";
    }
}
