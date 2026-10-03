// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using InferenceWeb.Tests;
using TensorSharp.Models;
using TensorSharp.Runtime;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i++)
{
    if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Length)
        throw new ArgumentException("Expected --name value pairs.");
    options.Add(args[i][2..], args[++i]);
}
string Value(string name, string fallback) => options.GetValueOrDefault(name, fallback);
int Number(string name, int fallback, int minimum = 1)
{
    int value = int.Parse(Value(name, fallback.ToString()), System.Globalization.CultureInfo.InvariantCulture);
    return value >= minimum ? value : throw new ArgumentOutOfRangeException(name);
}
string output = Path.GetFullPath(Value("output", "artifacts/qwen4exp-expert-cache-probe.json"));
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
bool synthetic = !options.ContainsKey("model");
bool host = Value("placement", "host") switch
{
    "host" => true,
    "device" => false,
    _ => throw new ArgumentException("placement must be host or device"),
};
BackendType backend = Value("backend", "ggml_cuda") switch
{
    "ggml_cuda" => BackendType.GgmlCuda,
    "ggml_cpu" => BackendType.GgmlCpu,
    _ => throw new ArgumentException("backend must be ggml_cuda or ggml_cpu"),
};
bool q2kxl = Value("quantization", "mixed") switch
{
    "mixed" => false,
    "q2kxl" => true,
    _ => throw new ArgumentException("quantization must be mixed or q2kxl"),
};
int prefill = Number("prefill-tokens", 40), decode = Number("decode-tokens", 64);
int warmups = Number("warmup", 2, 0), iterations = Number("iterations", 5);
string generation = Value("generation", "teacher-forced");
if (generation is not ("teacher-forced" or "greedy"))
    throw new ArgumentException("generation must be greedy or teacher-forced");
foreach (string name in TensorSharp.Runtime.Speculative.SpeculationEnvVars.RemovedNames.Select(pair => pair.Name)
    .Concat(new[] { "TS_SPEC", "TS_SPEC_TYPE", "TS_SPEC_DRAFT", "TS_SPEC_PMIN", "TS_SPEC_DRAFT_MODEL" }))
    Environment.SetEnvironmentVariable(name, null);
Environment.SetEnvironmentVariable("MAX_CONTEXT", Value("max-context", Math.Max(256, prefill + decode + 16).ToString()));
KvCacheDtypeConfig.Set(KvCacheDtype.F16);
string modelPath = synthetic ? Path.Combine(Path.GetDirectoryName(output)!, "fixture.gguf") : Path.GetFullPath(options["model"]);
Dictionary<string, float[][]>? captures = null;
if (synthetic)
{
    Qwen4ExpSyntheticModelBuilder.Write(modelPath, q2kxlExperts: q2kxl);
    string other = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(Path.GetDirectoryName(output)!, "other-fixture.gguf"), q2kxlExperts: !q2kxl);
    captures = Qwen4ExpExpertCacheScenario.Exercise(modelPath, other, backend, host);
    foreach (string name in new[] { "a2", "reloaded-a" })
    {
        var difference = Qwen4ExpExpertCacheScenario.Difference(captures[name], captures["a1"]);
        if (difference.RelativeL2 != 0 || difference.ArgmaxMismatches != 0)
            throw new InvalidOperationException("Reset/disposal changed deterministic logits: " + name);
    }
}

MoeCpuOffloadConfig.Reset();
if (host) MoeCpuOffloadConfig.SetAllLayers();
var load = Stopwatch.StartNew();
using var model = ModelBase.Create(modelPath, backend);
load.Stop();
string nativePath = Qwen4ExpExpertCacheScenario.MappedNativePath();
string nativeHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(nativePath))).ToLowerInvariant();
var managedAssemblies = AppDomain.CurrentDomain.GetAssemblies()
    .Where(assembly => assembly.GetName().Name?.StartsWith("TensorSharp", StringComparison.Ordinal) == true && !assembly.IsDynamic)
    .Select(assembly => assembly.Location).Distinct().OrderBy(path => path)
    .ToDictionary(path => path, path => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant());
int[] tokenPool = synthetic ? Enumerable.Range(0, 251).ToArray() : model.Tokenizer.Encode(
    "The history of computing spans many centuries, beginning with counting tools and algorithms. ", addSpecial: false).ToArray();
if (tokenPool.Length == 0) throw new InvalidOperationException("Empty benchmark token pool.");
int[] prompt = Enumerable.Range(0, prefill).Select(i => tokenPool[(i * 37 + 11) % tokenPool.Length]).ToArray();
int[] forced = Enumerable.Range(0, decode).Select(i => tokenPool[(i * 53 + 17) % tokenPool.Length]).ToArray();
string? renderedPrompt = null;
if (options.TryGetValue("tokens-file", out string? tokenFile))
    prompt = File.ReadAllText(tokenFile).Split(new[] { ',', ' ', '\r', '\n', '\t' }, StringSplitOptions.RemoveEmptyEntries)
        .Select(value => int.Parse(value, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
else if (options.TryGetValue("prompt-raw-file", out string? rawFile))
{
    renderedPrompt = File.ReadAllText(rawFile);
    prompt = model.Tokenizer.Encode(renderedPrompt, addSpecial: true).ToArray();
}
else if (options.ContainsKey("prompt") || options.ContainsKey("prompt-file"))
{
    string text = options.TryGetValue("prompt-file", out string? promptFile) ? File.ReadAllText(promptFile) : options["prompt"];
    renderedPrompt = new GgufPromptRenderer().Render(model.Config.ChatTemplate, new List<ChatMessage>
    {
        new() { Role = "system", Content = Value("system", "You are a helpful assistant.") },
        new() { Role = "user", Content = text },
    }, architecture: model.Config.Architecture, enableThinking: Value("thinking", "false") == "true");
    prompt = model.Tokenizer.Encode(renderedPrompt, addSpecial: true).ToArray();
}
if (prompt.Length == 0 || prompt.Any(token => token < 0 || token >= model.Tokenizer.VocabSize))
    throw new ArgumentException("Prompt must contain valid vocabulary token IDs.");
if (prompt.Length + decode > int.Parse(Environment.GetEnvironmentVariable("MAX_CONTEXT")!))
    throw new ArgumentException("Prompt plus generation exceeds --max-context.");
if (options.TryGetValue("prompt-tokens-output", out string? tokenOutput))
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(tokenOutput))!);
    File.WriteAllText(tokenOutput, string.Join(",", prompt) + Environment.NewLine);
}
var runs = new List<object>();
string DecodeGeneration(IEnumerable<int> generated) => model.Tokenizer.Decode(
    generated.TakeWhile(token => !model.Tokenizer.IsEos(token)).ToList());
var statsBeforeTimedWork = backend == BackendType.GgmlCuda ? Qwen4ExpExpertCacheScenario.CacheStats() : null;
string? referenceHash = null;
float[]? finalLogits = null;
int[]? finalGenerated = null;
for (int i = -warmups; i < iterations; i++)
{
    model.ResetKVCache();
    var timer = Stopwatch.StartNew();
    float[] logits = model.ForwardRefill(prompt);
    timer.Stop();
    double prefillMs = timer.Elapsed.TotalMilliseconds;
    var generated = new List<int>();
    bool eos = false;
    int forwardSteps = 0;
    timer.Restart();
    if (generation == "teacher-forced")
    {
        foreach (int token in forced) logits = model.Forward(new[] { token });
        forwardSteps = forced.Length;
    }
    else
    {
        for (int position = 0; position < decode; position++)
        {
            int token = 0;
            for (int j = 1; j < logits.Length; j++) if (logits[j] > logits[token]) token = j;
            generated.Add(token);
            if (model.Tokenizer.IsEos(token)) { eos = true; break; }
            if (position + 1 < decode) { logits = model.Forward(new[] { token }); forwardSteps++; }
        }
    }
    timer.Stop();
    if (logits.Length == 0 || logits.Any(v => !float.IsFinite(v)))
        throw new InvalidOperationException("The benchmark returned empty or nonfinite final logits.");
    byte[] bytes = new byte[logits.Length * sizeof(float)];
    Buffer.BlockCopy(logits, 0, bytes, 0, bytes.Length);
    string hash = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    referenceHash ??= hash;
    if (referenceHash != hash) throw new InvalidOperationException("Identical repeated inputs changed final logits.");
    finalLogits = (float[])logits.Clone();
    finalGenerated = generated.ToArray();
    var row = new { warmup = i < 0, iteration = i < 0 ? i + warmups : i, prefill_tokens = prompt.Length,
        decode_tokens = forwardSteps, generated_tokens = finalGenerated,
        selected_text = generation == "greedy" ? DecodeGeneration(generated) : null,
        finish_reason = generation == "greedy" ? (eos ? "eos" : "length") : "teacher-forced",
        prefill_ms = prefillMs, decode_ms = timer.Elapsed.TotalMilliseconds,
        prefill_tps = prompt.Length / (prefillMs / 1000), decode_tps = forwardSteps / timer.Elapsed.TotalSeconds,
        final_logit_sha256 = hash };
    runs.Add(row);
    Console.WriteLine(JsonSerializer.Serialize(row));
}
Qwen4ExpExpertCacheScenario.Stats? stats = null;
if (backend == BackendType.GgmlCuda)
{
    stats = Qwen4ExpExpertCacheScenario.CacheStats();
    if (stats.ReservedBytes < 0 || stats.ReservedBytes > stats.BudgetBytes)
        throw new InvalidOperationException("Expert-cache device reservation exceeds its budget.");
    bool hasReuse = stats.Hits > statsBeforeTimedWork!.Hits;
    if (host && Number("require-cache", 0, 0) != 0 && !(stats.Calls > statsBeforeTimedWork.Calls
        && hasReuse && stats.Misses > statsBeforeTimedWork.Misses))
        throw new InvalidOperationException("Expert cache did not engage and reuse selected weights.");
}
using var process = Process.GetCurrentProcess();
process.Refresh();
var report = new
{
    schema_version = 1,
    passed = true,
    synthetic,
    language_quality_validated = false,
    decode_mode = generation,
    backend = backend.ToString(),
    placement = host ? "host" : "device",
    quantization = synthetic ? (q2kxl ? "q2kxl" : "mixed") : "checkpoint",
    model_path = modelPath,
    model_bytes = new FileInfo(modelPath).Length,
    model_sha256 = synthetic ? Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(modelPath))).ToLowerInvariant() : null,
    native_path = nativePath,
    native_sha256 = nativeHash,
    managed_assemblies_sha256 = managedAssemblies,
    model_load_ms = load.Elapsed.TotalMilliseconds,
    process_peak_working_set_bytes = process.PeakWorkingSet64,
    process_working_set_bytes = process.WorkingSet64,
    managed_bytes = GC.GetTotalMemory(false),
    cache_stats = stats,
    cache_stats_before_timed_work = statsBeforeTimedWork,
    prompt_tokens = prompt,
    rendered_prompt = renderedPrompt,
    generated_tokens = finalGenerated,
    selected_text = generation == "greedy" ? DecodeGeneration(finalGenerated!) : null,
    raw_decoded_output = generation == "greedy" ? model.Tokenizer.Decode(finalGenerated!.ToList()) : null,
    final_logits = finalLogits,
    forced_tokens = generation == "teacher-forced" ? forced : null,
    captures,
    runs,
    limitations = new[]
    {
        "Synthetic checkpoints exercise engines and cache ownership; they cannot establish trained language quality or real-model throughput.",
        "Teacher-forced decode excludes sampling and serves identical token inputs; it is not an HTTP end-to-end throughput result.",
        "Working-set and external GPU telemetry are process/driver measurements, not a full allocator peak.",
        "The cache is opt-in; eligible bias-free SiLU expert paths with one through eight rows engage it."
    }
};
File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + Environment.NewLine);
Console.WriteLine("report=" + output);
