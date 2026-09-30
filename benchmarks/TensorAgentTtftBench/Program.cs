// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

// TensorAgentTtftBench: the phone's chat, driven from a terminal, with the numbers
// that decide whether it feels fast.
//
// It starts a REAL TensorAgent host -- AgentAppHost, the loopback server, the Web UI
// chat service, the bundled skills, the engine memory budget the catalog entry asks
// for, the phone's prefill chunk -- loads a catalog model the way tapping "Use" does,
// and then talks to /api/chat exactly as the page does: one session per
// conversation, the whole history resent on every turn. For every turn it reports
// how long the first token took, how many prompt tokens there were, and how many of
// them the KV cache served. The scenarios are the shapes a conversation actually
// takes: a cold first turn, follow-ups, a brand-new chat, a thinking turn, a turn the
// user stopped, and a turn that ran a tool.
//
//   dotnet run -c Release --project benchmarks/TensorAgentTtftBench -- \
//       --model gemma-4-e2b-q8 --source ~/work/models/gemma-4-E2B
//
// Options:
//   --model <catalog id>      which catalog entry to run (required)
//   --source <dir>            a directory holding the entry's own file names
//   --weights <file>          the GGUF, when it is not named as the catalog names it
//   --projector <file>        its projector, likewise
//   --backends a,b            backends to try, best first (default ggml_metal,ggml_cpu)
//   --no-skills               do not bundle the repository's skills (a much shorter prompt)
//   --python <root>           a CPython prefix for the in-process shell
//   --scenarios a,b,c         cold, follow, newchat, think, stop, tool (default: all); specbench runs the
//                             app's own in-process SpeculationBench (TENSORAGENT_SPEC_BENCH_TOKENS/_MODES),
//                             the same code the Mac app runs under TENSORAGENT_SPEC_BENCH=1
//   --follow N                follow-up turns in the first conversation (default 2)
//   --max-tokens N            answer budget for the short turns (default 32)
//   --kv f16|q8_0|q4_0        the K/V cache precision setting (default: the app's default)
//   --no-spec                 turn the speculative-decoding setting off (on by default), for an A/B
//   --context N               the context-length setting (default: the catalog entry's)
//   --chunk N                 TS_SCHED_SOLO_PREFILL_CHUNK (default 1024, as the phone sets it; unset with --desktop)
//   --desktop                 run the desktop app's configuration instead of the phone's: the desktop
//                             engine budget and first-launch settings (DeviceClass.Desktop), real
//                             python3/node processes rather than the embedded runtimes, and no phone
//                             prefill chunk. The CoreCLR reference for the Mac and Windows apps.
//   --device-gb N             the device memory tier to pretend to be (default 16)
//   --warm                    wait for the host's prefix-cache warm-up before the first turn
//   --delay <seconds>         wait this long after the load before the first turn (a
//                             message typed while the warm-up is still forwarding)
//   --root <dir>              where to put the host's data and cache (default: a temp dir)
//   --out <file>              write the rows as JSON as well
//   --verbose                 print every host log line, not only the cache-related ones
//   --prompt <text>           the message the "agentic" scenario sends (a research + document task by default)
//   --network                 allow the skills' scripts to reach the network (the app's Network switch)
//   --agentic-max-tokens N    answer budget for the agentic turn (default 2048, the app's default)
//   --hold S                  keep the host alive S seconds after the last turn, for an external memory snapshot

using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;

namespace TensorAgentTtftBench;

internal sealed record TurnRow(
    string Scenario,
    string Label,
    int PromptTokens,
    int ReusedTokens,
    double ReusePercent,
    double FirstTokenSeconds,
    double TotalSeconds,
    int Tokens,
    bool Aborted,
    string Answer,
    string? Error);

internal sealed class Chat
{
    public required string SessionId { get; init; }
    public List<Dictionary<string, object?>> History { get; } = new();
}

internal static class Program
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    private static async Task<int> Main(string[] args)
    {
        var opts = Options.Parse(args);
        if (opts is null)
            return 2;

        CatalogModel? model = ModelCatalog.BuiltIn.FirstOrDefault(m => string.Equals(m.Id, opts.ModelId, StringComparison.Ordinal));
        if (model is null)
        {
            Console.Error.WriteLine($"no catalog entry '{opts.ModelId}'. Known: {string.Join(", ", ModelCatalog.BuiltIn.Select(m => m.Id))}");
            return 2;
        }

        string root = opts.Root ?? Path.Combine(Path.GetTempPath(), "tensoragent-ttft-" + Guid.NewGuid().ToString("N"));
        var paths = new AgentPaths(Path.Combine(root, "data"), Path.Combine(root, "cache"))
        {
            // By default this benchmark emulates the mobile host, including its embedded
            // runtimes; --desktop runs what the desktop app runs.
            ExecutionMode = opts.Desktop
                ? TensorAgent.Core.Shell.AgentExecutionMode.Auto
                : TensorAgent.Core.Shell.AgentExecutionMode.InProcess,
            DeviceClass = opts.Desktop ? DeviceClass.Desktop : DeviceClass.Phone,
            DeviceMemoryGB = opts.DeviceGb,
            BundledSkillsDirectory = opts.Skills ? RepoSkillsDirectory() : string.Empty,
            PythonRuntimeDirectory = opts.PythonRoot ?? string.Empty,
        };
        paths.EnsureCreated();

        // The catalog's files, under the catalog's names, exactly where the store expects
        // them. Linked rather than copied: they are gigabytes.
        string modelDir = Path.Combine(paths.ModelsDirectory, model.Id);
        Directory.CreateDirectory(modelDir);
        string? weightsPath = null;
        foreach (CatalogFile file in model.Files)
        {
            string? source = file.Role switch
            {
                CatalogFileRole.Weights => opts.Weights,
                CatalogFileRole.Projector => opts.Projector,
                _ => null,
            };
            if (source is null && opts.Source is { } dir)
                source = Path.Combine(dir, file.FileName);
            if (source is null || !File.Exists(source))
            {
                if (file.Role == CatalogFileRole.Weights)
                {
                    Console.Error.WriteLine($"weights for {model.Id} not found: give --source <dir> holding {file.FileName}, or --weights <file>");
                    return 2;
                }
                continue;
            }
            string target = Path.Combine(modelDir, file.FileName);
            if (!File.Exists(target))
                LinkOrCopy(Path.GetFullPath(source), target);
            if (file.Role == CatalogFileRole.Weights)
                weightsPath = target;
        }
        if (weightsPath is null)
        {
            Console.Error.WriteLine($"{model.Id} has no weights file in its catalog entry");
            return 2;
        }

        var settingsStore = new SettingsStore(paths.SettingsFile, opts.Desktop ? AppSettings.DesktopDefaults : null);
        AppSettings settings = settingsStore.Load();
        settings.SelectedModelId = model.Id;
        if (opts.KvCacheDtype is { Length: > 0 } kv)
            settings.KvCacheDtype = kv;
        settings.SpeculativeDecoding = !opts.NoSpec;
        if (opts.ContextLength is int ctx && ctx > 0)
            settings.ContextLength = ctx;
        if (opts.Network)
            settings.AllowNetwork = true;
        settingsStore.Save(settings);

        // What MauiProgram sets for the phone. A desktop default of 8192 tokens per solo
        // prefill pass is not what the phone app runs; it is what the desktop app runs.
        if ((!opts.Desktop || opts.ChunkGiven)
            && Environment.GetEnvironmentVariable("TS_SCHED_SOLO_PREFILL_CHUNK") is not { Length: > 0 })
            Environment.SetEnvironmentVariable("TS_SCHED_SOLO_PREFILL_CHUNK", opts.Chunk.ToString(CultureInfo.InvariantCulture));

        var backends = opts.Backends.Select(b => new TensorSharp.Server.BackendOption(b, b)).ToList();
        Console.WriteLine($"ttft-bench: {model.Id} ({model.DisplayName}) {(opts.Desktop ? "desktop" : "phone")} skills={(opts.Skills ? "on" : "off")} " +
                          $"kv={settings.KvCacheDtype} context={(settings.ContextLength > 0 ? settings.ContextLength : model.ContextLength)} " +
                          $"chunk={Environment.GetEnvironmentVariable("TS_SCHED_SOLO_PREFILL_CHUNK")} device={opts.DeviceGb}GB root={root}");

        using var host = new AgentAppHost(paths, loggerFactory: new StdoutLoggerFactory(opts.Verbose), backends: backends);
        host.Server.Start();
        using var client = new HttpClient
        {
            BaseAddress = new Uri(host.Server.BaseUrl),
            Timeout = TimeSpan.FromMinutes(30),
        };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={host.Server.Token}");

        var loadClock = Stopwatch.StartNew();
        string backend = host.UseModel(model);
        loadClock.Stop();
        Console.WriteLine($"ttft-bench: loaded on {backend} in {loadClock.Elapsed.TotalSeconds:0.0}s");

        if (opts.DelaySeconds > 0)
        {
            Console.WriteLine($"ttft-bench: waiting {opts.DelaySeconds:0.#}s after the load before the first turn");
            await Task.Delay(TimeSpan.FromSeconds(opts.DelaySeconds));
        }
        if (opts.Warm)
            await WaitForWarmCacheAsync(host);

        var rows = new List<TurnRow>();
        var runner = new Runner(client, opts, rows);
        try
        {
            foreach (string scenario in opts.Scenarios)
            {
                switch (scenario)
                {
                    case "cold":
                        await runner.ColdAndFollowUpsAsync();
                        break;
                    case "follow":
                        // Part of "cold": follow-ups need a conversation to follow.
                        break;
                    case "newchat":
                        await runner.NewChatAsync();
                        break;
                    case "think":
                        await runner.ThinkingAsync(model.SupportsThinking);
                        break;
                    case "stop":
                        await runner.StoppedTurnAsync();
                        break;
                    case "tool":
                        await runner.ToolTurnAsync();
                        break;
                    case "spec":
                        await runner.SpeculationAsync();
                        break;
                    case "agentic":
                        await runner.AgenticAsync();
                        break;
                    case "concurrent":
                        await runner.ConcurrentAsync();
                        break;
                    case "restore":
                        await runner.RestoreAsync();
                        break;
                    case "specbench":
                        await new SpeculationBench(host).RunAsync(CancellationToken.None);
                        break;
                    default:
                        Console.Error.WriteLine($"unknown scenario '{scenario}'");
                        break;
                }
            }
        }
        finally
        {
            Console.WriteLine($"ttft-bench: memory after the last turn -- {ProcessMemoryProbe.Describe()}");
            if (opts.HoldSeconds > 0)
            {
                Console.WriteLine($"ttft-bench: holding the host for {opts.HoldSeconds:0}s (pid {Environment.ProcessId})");
                await Task.Delay(TimeSpan.FromSeconds(opts.HoldSeconds));
            }
            PrintTable(rows, model, backend);
            if (opts.Out is { Length: > 0 } outPath)
            {
                File.WriteAllText(outPath, JsonSerializer.Serialize(new
                {
                    model = model.Id,
                    backend,
                    skills = opts.Skills,
                    kv = settings.KvCacheDtype,
                    rows,
                }, JsonOptions));
                Console.WriteLine($"ttft-bench: wrote {outPath}");
            }
        }

        // The host's own shutdown waits for the engine; the model must not be released
        // under a step that is still computing.
        return rows.Any(r => r.Error is { Length: > 0 }) ? 1 : 0;
    }

    private static async Task WaitForWarmCacheAsync(AgentAppHost host)
    {
        // Reflection so this tool builds against a host with or without the warm-up.
        PropertyInfo? warm = host.GetType().GetProperty("PrefixCacheIsWarm");
        if (warm is null)
        {
            Console.WriteLine("ttft-bench: this host has no prefix-cache warm-up; asking cold");
            return;
        }
        var clock = Stopwatch.StartNew();
        while (clock.Elapsed < TimeSpan.FromMinutes(3))
        {
            if (warm.GetValue(host) is true)
            {
                Console.WriteLine($"ttft-bench: the prefix cache is warm after {clock.Elapsed.TotalSeconds:0.0}s");
                return;
            }
            await Task.Delay(250);
        }
        Console.WriteLine("ttft-bench: the warm-up did not finish within 3 minutes; asking anyway");
    }

    /// <summary>
    /// A hard link where the volume allows one, a copy otherwise. Not a symbolic link:
    /// the store's completeness check compares <c>FileInfo.Length</c> with the catalog's
    /// byte count, and on macOS that is the length of the link itself, so a symlinked
    /// projector reads as "not downloaded yet" and the load refuses.
    /// </summary>
    private static void LinkOrCopy(string source, string target)
    {
        try
        {
            var ln = new ProcessStartInfo("ln", new[] { source, target }) { RedirectStandardError = true, UseShellExecute = false };
            using Process? p = Process.Start(ln);
            p?.WaitForExit();
            if (p is { ExitCode: 0 } && File.Exists(target))
                return;
        }
        catch (Exception)
        {
            // fall through to the copy
        }
        Console.WriteLine($"ttft-bench: copying {Path.GetFileName(source)} (no hard link possible)");
        File.Copy(source, target);
    }

    private static string RepoSkillsDirectory()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
            directory = directory.Parent;
        if (directory is null)
            throw new InvalidOperationException($"no TensorSharp.slnx above {AppContext.BaseDirectory}");
        return Path.Combine(directory.FullName, "TensorAgent", "skills");
    }

    private static void PrintTable(List<TurnRow> rows, CatalogModel model, string backend)
    {
        Console.WriteLine();
        Console.WriteLine($"## {model.Id} on {backend}");
        Console.WriteLine();
        Console.WriteLine("| scenario | turn | prompt | reused | reuse % | first token | total | tokens | prefill tok/s | decode tok/s | note |");
        Console.WriteLine("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
        foreach (TurnRow r in rows)
        {
            string note = r.Error is { Length: > 0 } ? "ERROR " + Shorten(r.Error, 60)
                : r.Aborted ? "stopped: " + Shorten(r.Answer, 40)
                : Shorten(r.Answer, 48);
            double prefillTps = r.FirstTokenSeconds > 0 ? (r.PromptTokens - r.ReusedTokens) / r.FirstTokenSeconds : 0;
            double decodeSeconds = r.TotalSeconds - r.FirstTokenSeconds;
            double decodeTps = r.Tokens > 1 && decodeSeconds > 0 ? (r.Tokens - 1) / decodeSeconds : 0;
            Console.WriteLine($"| {r.Scenario} | {r.Label} | {r.PromptTokens} | {r.ReusedTokens} | {r.ReusePercent:0.0} | " +
                              $"{r.FirstTokenSeconds:0.00}s | {r.TotalSeconds:0.0}s | {r.Tokens} | {prefillTps:0} | {decodeTps:0.0} | {note} |");
        }
        Console.WriteLine();
    }

    internal static string Shorten(string text, int max)
    {
        string flat = text.Replace("\r", " ").Replace("\n", " ").Replace("|", "/").Trim();
        return flat.Length <= max ? flat : flat[..max] + "…";
    }
}

internal sealed class Runner
{
    private readonly HttpClient _client;
    private readonly Options _opts;
    private readonly List<TurnRow> _rows;

    public Runner(HttpClient client, Options opts, List<TurnRow> rows)
    {
        _client = client;
        _opts = opts;
        _rows = rows;
    }

    public async Task ColdAndFollowUpsAsync()
    {
        Chat chat = await NewChatSessionAsync();
        await AskAsync(chat, "cold", "turn 1", "Say the single word: apple.", _opts.MaxTokens, think: false);
        string[] follow = { "Now say: banana.", "Now say: cherry.", "Now say: date.", "Now say: elderberry.", "Now say: fig." };
        for (int i = 0; i < _opts.FollowUps; i++)
            await AskAsync(chat, "follow", $"turn {i + 2}", follow[i % follow.Length], _opts.MaxTokens, think: false);
    }

    public async Task NewChatAsync()
    {
        Chat chat = await NewChatSessionAsync();
        await AskAsync(chat, "newchat", "turn 1", "Say the single word: kiwi.", _opts.MaxTokens, think: false);
        await AskAsync(chat, "newchat", "turn 2", "Now say: mango.", _opts.MaxTokens, think: false);
    }

    public async Task ThinkingAsync(bool supportsThinking)
    {
        Chat chat = await NewChatSessionAsync();
        string label = supportsThinking ? "think" : "think(unsupported)";
        await AskAsync(chat, label, "turn 1 (no think)", "Say the single word: plum.", _opts.MaxTokens, think: false);
        await AskAsync(chat, label, "turn 2 (think)", "What is 17 times 23? Reply with just the number.", 512, think: true);
        await AskAsync(chat, label, "turn 3 (think)", "And 17 times 24? Just the number.", 512, think: true);
        await AskAsync(chat, label, "turn 4 (no think)", "Now say: done.", _opts.MaxTokens, think: false);
    }

    public async Task StoppedTurnAsync()
    {
        Chat chat = await NewChatSessionAsync();
        await AskAsync(chat, "stop", "turn 1 (stopped)", "Write three paragraphs about the sea.", 400, think: false, stopAfterTokens: 8);
        await AskAsync(chat, "stop", "turn 2", "Now say: banana.", _opts.MaxTokens, think: false);
    }

    /// <summary>
    /// The turns that decide whether speculative decoding pays, in one conversation:
    /// a one-word answer (the thinking preamble), free prose, a file quoted back from
    /// the prompt, and the same text quoted from the model's own previous answer. Run
    /// once with the setting on and once with --no-spec, and compare the decode
    /// tok/s column. The same four turns are what the phone's SpeculationBench runs.
    /// </summary>
    public async Task SpeculationAsync()
    {
        Chat chat = await NewChatSessionAsync();
        string quoted = TensorAgent.Core.Hosting.SpeculationBench.QuotedText();
        int tokens = Math.Max(96, _opts.MaxTokens * 5);
        await AskAsync(chat, "spec", "1 one word", "Say the single word: apple.", 32, think: false);
        await AskAsync(chat, "spec", "2 prose", "Write three short paragraphs about the sea.", tokens, think: false);
        await AskAsync(chat, "spec", "3 quote prompt",
            $"Repeat the following text exactly, character for character:\n```\n{quoted}```", tokens + 60, think: false);
        await AskAsync(chat, "spec", "4 quote own answer", "Now repeat that same text once more, exactly.", tokens + 60, think: false);
    }

    public async Task ToolTurnAsync()
    {
        Chat chat = await NewChatSessionAsync();
        await AskAsync(chat, "tool", "turn 1 (tool)",
            "Use the shell tool to run exactly this command: echo hello-from-shell\nThen tell me what it printed, nothing else.",
            400, think: false);
        await AskAsync(chat, "tool", "turn 2", "Now say: banana.", _opts.MaxTokens, think: false);
    }

    /// <summary>
    /// The conversation shape that was killing the app: one user request that the model
    /// answers through several tool rounds (research on the open web, then a document
    /// built by a script), with thinking on. What matters here is not the first-token time
    /// but how much memory the host holds by the end of it.
    /// </summary>
    public async Task AgenticAsync()
    {
        Chat chat = await NewChatSessionAsync();
        string prompt = _opts.Prompt
            ?? "请找出apple这周发布会的安排与内容，然后做个pptx发给我";
        await AskAsync(chat, "agentic", "turn 1 (agentic)", prompt, _opts.AgenticMaxTokens, think: true);
        Console.WriteLine($"    memory: {ProcessMemoryProbe.Describe()}");
        await AskAsync(chat, "agentic", "turn 2", "Now say: banana.", _opts.MaxTokens, think: false);
        Console.WriteLine($"    memory: {ProcessMemoryProbe.Describe()}");
    }

    /// <summary>
    /// Two chats at once, several times over: the per-sequence fused path (N >= 2)
    /// with holders created, adopted, retained, evicted and disposed under the host's
    /// own knobs. On a phone the prefix warm-up can land beside the user's first
    /// message, which is exactly this shape.
    /// </summary>
    public async Task ConcurrentAsync()
    {
        for (int round = 0; round < 3; round++)
        {
            Chat a = await NewChatSessionAsync();
            Chat b = await NewChatSessionAsync();
            // A long first message, so A is still in its prefill when B arrives and both
            // sequences really share the engine (N == 2), as the phone saw.
            string longTask = string.Concat(Enumerable.Repeat(
                "The quick brown fox jumps over the lazy dog while the orchestra tunes its strings before the evening concert begins. ", 120))
                + " After all of that, say the single word: apple.";
            Task<TurnRow> ta = AskAsync(a, "concurrent", $"round {round + 1} A", longTask, 8, think: false);
            await Task.Delay(250);
            Task<TurnRow> tb = AskAsync(b, "concurrent", $"round {round + 1} B", "Say the single word: pear.", 1, think: false);
            await Task.WhenAll(ta, tb);
            await AskAsync(a, "concurrent", $"round {round + 1} A follow", "Now say: fig.", 8, think: false);
            Console.WriteLine($"    memory: {ProcessMemoryProbe.Describe()}");
        }
    }

    /// <summary>
    /// The first message of a launch, sent the moment the model is loaded and before
    /// any warm-up has had time to run -- which is what a person does. Run it twice
    /// against the same --root: the first run prefills the shared prompt and saves the
    /// checkpoint, the second restores it, and the first-token time of "turn 1" is the
    /// difference the saved checkpoint makes.
    /// </summary>
    public async Task RestoreAsync()
    {
        Chat chat = await NewChatSessionAsync();
        await AskAsync(chat, "restore", "turn 1 (first message of the launch)", "Say the single word: apple.", _opts.MaxTokens, think: false);
        Chat second = await NewChatSessionAsync();
        await AskAsync(second, "restore", "turn 2 (a new chat)", "Say the single word: pear.", _opts.MaxTokens, think: false);
    }

    private async Task<Chat> NewChatSessionAsync()
    {
        using HttpResponseMessage response = await _client.PostAsync("/api/sessions?conversation=new", null);
        string payload = await response.Content.ReadAsStringAsync();
        if (!response.IsSuccessStatusCode)
            throw new InvalidOperationException($"opening a session failed: {(int)response.StatusCode} {payload}");
        JsonElement created = JsonSerializer.Deserialize<JsonElement>(payload);
        return new Chat { SessionId = created.GetProperty("sessionId").GetString()! };
    }

    private async Task<TurnRow> AskAsync(Chat chat, string scenario, string label, string text, int maxTokens, bool think, int stopAfterTokens = 0)
    {
        chat.History.Add(new Dictionary<string, object?> { ["role"] = "user", ["content"] = text });
        var body = new Dictionary<string, object?>
        {
            ["sessionId"] = chat.SessionId,
            ["messages"] = chat.History.ToArray(),
            ["maxTokens"] = maxTokens,
            ["think"] = think,
        };
        if (!_opts.Skills)
            body["skills_discovery"] = false;

        using var request = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
        {
            Content = new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json"),
        };

        Console.WriteLine($"--- {scenario} / {label}: \"{Program.Shorten(text, 60)}\" (maxTokens {maxTokens}, think {think})");
        var clock = Stopwatch.StartNew();
        var answer = new StringBuilder();
        var thinking = new StringBuilder();
        double firstToken = 0;
        int prompt = 0, reused = 0, tokens = 0, tokenFrames = 0;
        double reusePct = 0, elapsed = 0;
        bool aborted = false, stopped = false;
        string? error = null;
        string? turnId = null;

        try
        {
            using HttpResponseMessage response = await _client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
            if (response.Headers.TryGetValues(WebUiRoutes.TurnHeader, out IEnumerable<string>? ids))
                turnId = ids.FirstOrDefault();
            if (!response.IsSuccessStatusCode)
            {
                error = $"{(int)response.StatusCode} {await response.Content.ReadAsStringAsync()}";
            }
            else
            {
                await using Stream stream = await response.Content.ReadAsStreamAsync();
                using var reader = new StreamReader(stream);
                while (await reader.ReadLineAsync() is { } line)
                {
                    if (!line.StartsWith("data: ", StringComparison.Ordinal))
                        continue;
                    JsonElement frame;
                    try { frame = JsonSerializer.Deserialize<JsonElement>(line[6..]); }
                    catch (JsonException) { continue; }

                    if (frame.TryGetProperty("thinking", out JsonElement th) && th.ValueKind == JsonValueKind.String)
                    {
                        if (firstToken == 0) firstToken = clock.Elapsed.TotalSeconds;
                        thinking.Append(th.GetString());
                    }
                    if (frame.TryGetProperty("token", out JsonElement tok) && tok.ValueKind == JsonValueKind.String)
                    {
                        if (firstToken == 0) firstToken = clock.Elapsed.TotalSeconds;
                        answer.Append(tok.GetString());
                        tokenFrames++;
                        if (stopAfterTokens > 0 && tokenFrames >= stopAfterTokens && !stopped && turnId is { Length: > 0 })
                        {
                            stopped = true;
                            using HttpResponseMessage stop = await _client.PostAsync($"/api/agent/turns/{Uri.EscapeDataString(turnId)}/stop", null);
                            Console.WriteLine($"    stop requested after {tokenFrames} tokens: {(int)stop.StatusCode}");
                        }
                    }
                    if (frame.TryGetProperty("replace", out JsonElement rep) && rep.ValueKind == JsonValueKind.String)
                        answer.Clear().Append(rep.GetString());
                    if (frame.TryGetProperty("error", out JsonElement err) && err.ValueKind == JsonValueKind.String)
                        error = err.GetString();
                    if (frame.TryGetProperty("done", out JsonElement done) && done.ValueKind == JsonValueKind.True)
                    {
                        prompt = ReadInt(frame, "promptTokens");
                        reused = ReadInt(frame, "kvReusedTokens");
                        reusePct = ReadDouble(frame, "kvReusePercent");
                        elapsed = ReadDouble(frame, "elapsed");
                        tokens = ReadInt(frame, "tokenCount");
                        aborted = frame.TryGetProperty("aborted", out JsonElement ab) && ab.ValueKind == JsonValueKind.True;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            error = ex.Message;
        }
        clock.Stop();

        string answerText = answer.ToString();
        chat.History.Add(new Dictionary<string, object?> { ["role"] = "assistant", ["content"] = answerText });

        var row = new TurnRow(scenario, label, prompt, reused, reusePct, firstToken, clock.Elapsed.TotalSeconds, tokens, aborted, answerText, error);
        _rows.Add(row);
        Console.WriteLine($"    => first token {row.FirstTokenSeconds:0.00}s, prompt {prompt}, reused {reused} ({reusePct:0.0}%), " +
                          $"{tokens} tokens in {clock.Elapsed.TotalSeconds:0.0}s{(aborted ? " (stopped)" : "")}" +
                          $"{(error is { Length: > 0 } ? " ERROR " + Program.Shorten(error, 120) : "")}");
        if (thinking.Length > 0)
            Console.WriteLine($"    thinking: {Program.Shorten(thinking.ToString(), 100)}");
        Console.WriteLine($"    memory: {ProcessMemoryProbe.Describe()}");
        Console.WriteLine($"    answer: {Program.Shorten(answerText, 140)}");
        return row;
    }

    private static int ReadInt(JsonElement frame, string name)
        => frame.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : 0;

    private static double ReadDouble(JsonElement frame, string name)
        => frame.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : 0;
}

internal sealed class Options
{
    public string ModelId { get; private set; } = string.Empty;
    public string? Source { get; private set; }
    public string? Weights { get; private set; }
    public string? Projector { get; private set; }
    public List<string> Backends { get; } = new() { "ggml_metal", "ggml_cpu" };
    public bool Skills { get; private set; } = true;
    public string? PythonRoot { get; private set; }
    public List<string> Scenarios { get; } = new() { "cold", "newchat", "think", "stop", "tool" };
    public int FollowUps { get; private set; } = 2;
    public int MaxTokens { get; private set; } = 32;
    public string? KvCacheDtype { get; private set; }
    public bool NoSpec { get; private set; }
    public int? ContextLength { get; private set; }
    public int Chunk { get; private set; } = 1024;
    public bool ChunkGiven { get; private set; }
    public bool Desktop { get; private set; }
    public int DeviceGb { get; private set; } = 16;
    public bool Warm { get; private set; }

    public double DelaySeconds;
    public string? Prompt { get; private set; }
    public bool Network { get; private set; }
    public int AgenticMaxTokens { get; private set; } = 2048;
    public double HoldSeconds { get; private set; }
    public string? Root { get; private set; }
    public string? Out { get; private set; }
    public bool Verbose { get; private set; }

    public static Options? Parse(string[] args)
    {
        var o = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            string Next()
            {
                if (i + 1 >= args.Length) throw new ArgumentException($"{a} needs a value");
                return args[++i];
            }
            try
            {
                switch (a)
                {
                    case "--model": o.ModelId = Next(); break;
                    case "--source": o.Source = Next(); break;
                    case "--weights": o.Weights = Next(); break;
                    case "--projector": o.Projector = Next(); break;
                    case "--backends": o.Backends.Clear(); o.Backends.AddRange(Next().Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)); break;
                    case "--no-skills": o.Skills = false; break;
                    case "--python": o.PythonRoot = Next(); break;
                    case "--scenarios": o.Scenarios.Clear(); o.Scenarios.AddRange(Next().Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)); break;
                    case "--follow": o.FollowUps = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-tokens": o.MaxTokens = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--kv": o.KvCacheDtype = Next(); break;
                    case "--no-spec": o.NoSpec = true; break;
                    case "--context": o.ContextLength = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--chunk": o.Chunk = int.Parse(Next(), CultureInfo.InvariantCulture); o.ChunkGiven = true; break;
                    case "--desktop": o.Desktop = true; break;
                    case "--device-gb": o.DeviceGb = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--warm": o.Warm = true; break;
                    case "--delay": o.DelaySeconds = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--prompt": o.Prompt = Next(); break;
                    case "--network": o.Network = true; break;
                    case "--agentic-max-tokens": o.AgenticMaxTokens = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--hold": o.HoldSeconds = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--root": o.Root = Next(); break;
                    case "--out": o.Out = Next(); break;
                    case "--verbose": o.Verbose = true; break;
                    case "--help": case "-h": Usage(); return null;
                    default:
                        Console.Error.WriteLine($"unknown option {a}");
                        Usage();
                        return null;
                }
            }
            catch (Exception ex) when (ex is ArgumentException or FormatException)
            {
                Console.Error.WriteLine(ex.Message);
                return null;
            }
        }
        if (o.Scenarios.Contains("all"))
        {
            o.Scenarios.Clear();
            o.Scenarios.AddRange(new[] { "cold", "newchat", "think", "stop", "tool", "agentic", "concurrent", "restore" });
        }
        if (string.IsNullOrWhiteSpace(o.ModelId))
        {
            Console.Error.WriteLine("--model <catalog id> is required");
            Usage();
            return null;
        }
        return o;
    }

    private static void Usage()
    {
        Console.Error.WriteLine("usage: TensorAgentTtftBench --model <catalog id> (--source <dir> | --weights <file> [--projector <file>])");
        Console.Error.WriteLine("       [--backends ggml_metal,ggml_cpu] [--no-skills] [--python <root>] [--scenarios cold,newchat,think,stop,tool,agentic,concurrent,restore,specbench]");
        Console.Error.WriteLine("       [--follow N] [--max-tokens N] [--kv f16|q8_0|q4_0] [--context N] [--chunk N] [--desktop] [--device-gb N] [--warm] [--delay S]");
        Console.Error.WriteLine("       [--prompt <text>] [--network] [--agentic-max-tokens N] [--hold S]");
        Console.Error.WriteLine("       [--root <dir>] [--out <file>] [--verbose]");
    }
}

/// <summary>
/// Prints the host's log to stdout. By default only what explains a KV-cache number:
/// the engine's own line about why a continuation was declined, prefix-cache
/// adoption, history compaction, and every warning or error.
/// </summary>
internal sealed class StdoutLoggerFactory : ILoggerFactory
{
    private readonly bool _verbose;

    public StdoutLoggerFactory(bool verbose) => _verbose = verbose;

    public ILogger CreateLogger(string categoryName) => new StdoutLogger(categoryName, _verbose);

    public void AddProvider(ILoggerProvider provider) { }

    public void Dispose() { }

    private sealed class StdoutLogger : ILogger
    {
        private static readonly string[] Interesting =
        {
            "Live-cache", "live-cache", "Prefix cache", "prefix cache", "prefix-cache",
            "kvReused", "compacted", "declined", "revoked", "Retained", "retained",
            "re-prefill", "Truncat", "ownership", "chat.complete", "chat.cancelled",
            "checkpoint", "cloned",
        };

        private readonly string _category;
        private readonly bool _verbose;

        public StdoutLogger(string category, bool verbose)
        {
            _category = category;
            _verbose = verbose;
        }

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => logLevel >= (_verbose ? LogLevel.Debug : LogLevel.Information);

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            if (!IsEnabled(logLevel))
                return;
            string message = formatter(state, exception);
            if (!_verbose && logLevel < LogLevel.Warning && !Interesting.Any(k => message.Contains(k, StringComparison.Ordinal)))
                return;
            Console.WriteLine($"    log[{logLevel}] {_category}: {Program.Shorten(message, 600)}");
            if (exception is not null)
                Console.WriteLine($"    log[{logLevel}] {exception.GetType().Name}: {exception.Message}");
        }
    }
}


/// <summary>
/// What the kernel charges this process and the whole machine, read the way jetsam
/// reads it. <c>phys_footprint</c> is the number a per-process limit is judged
/// against; the wired total is where Metal's claim on the mapped weights and on
/// every device buffer shows up, which the footprint does NOT include -- so a host
/// that only watches its own footprint is watching the wrong number on a phone.
/// </summary>
internal static class ProcessMemoryProbe
{
    [System.Runtime.InteropServices.DllImport("libSystem.dylib")]
    private static extern int task_info(uint target, int flavor, byte[] info, ref int count);

    [System.Runtime.InteropServices.DllImport("libSystem.dylib")]
    private static extern uint mach_task_self();

    [System.Runtime.InteropServices.DllImport("libSystem.dylib")]
    private static extern uint mach_host_self();

    [System.Runtime.InteropServices.DllImport("libSystem.dylib")]
    private static extern int host_statistics64(uint host, int flavor, byte[] info, ref int count);

    public static string Describe()
    {
        if (!OperatingSystem.IsMacOS() && !OperatingSystem.IsIOS())
            return "n/a";
        try
        {
            // task_vm_info: phys_footprint at byte 144 (TASK_VM_INFO_REV0 layout).
            var vm = new byte[152];
            int count = vm.Length / 4;
            long footprint = task_info(mach_task_self(), 22, vm, ref count) == 0 ? BitConverter.ToInt64(vm, 144) : -1;
            // vm_statistics64: free 0, active 4, inactive 8, wire 12 (natural_t), compressor_page_count 128.
            var st = new byte[152];
            int hc = st.Length / 4;
            long page = Environment.SystemPageSize;
            string system = "system n/a";
            if (host_statistics64(mach_host_self(), 4, st, ref hc) == 0)
            {
                long free = BitConverter.ToUInt32(st, 0) * page, wired = BitConverter.ToUInt32(st, 12) * page;
                long compressor = BitConverter.ToUInt32(st, 128) * page;
                system = $"system wired {wired / 1048576} MB, free {free / 1048576} MB, compressor {compressor / 1048576} MB";
            }
            return $"footprint {footprint / 1048576} MB; {system}";
        }
        catch (Exception ex)
        {
            return "probe failed: " + ex.Message;
        }
    }
}
