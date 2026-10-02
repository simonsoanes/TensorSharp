using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

string Option(string name, string fallback)
{
    int index = Array.IndexOf(args, name);
    return index < 0 ? fallback : index + 1 < args.Length ? args[index + 1]
        : throw new ArgumentException($"Missing value for {name}.");
}
string modelPath = Option("--model", Environment.GetEnvironmentVariable("TS_TEST_QWEN35_MODEL")
    ?? @"C:\Works\models\Qwen\Qwen3.8-27B-UD-IQ3_XXS.gguf");
string output = Option("--out", "artifacts/qwen35-parent-prefix-reuse/direct-probe.json");
bool describeOnly = args.Contains("--describe-only", StringComparer.Ordinal);
bool diagnoseState = args.Contains("--diagnose-state", StringComparer.Ordinal);
if (describeOnly && diagnoseState) throw new ArgumentException("Choose describe-only or diagnose-state, not both.");
string baselinePolicy = Option("--baseline", "full-public");
if (baselinePolicy is not ("full-public" or "child-boundaries"))
    throw new ArgumentException("Use --baseline full-public or --baseline child-boundaries.");
int steps = int.Parse(Option("--steps", "32"));
int pairs = int.Parse(Option("--pairs", "3"));
if (steps is < 1 or > 256 || pairs is < 1 or > 20)
    throw new ArgumentException("Use 1..256 steps and 1..20 pairs.");
Environment.SetEnvironmentVariable("MAX_CONTEXT", "8192");
Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS_MAX", "2");
KvCacheDtypeConfig.ConfigureFromEnvironment();
var runs = new List<WorkflowRun>();
var failures = new List<string>();
FixtureSet? fixtures = null;
StateDiagnosticResult? stateDiagnostics = null;
double loadMilliseconds = 0;
string? kvDtype = null;
int? initialCacheCapacity = null;
string? error = null;
bool passed = false;
try
{
    if (describeOnly)
    {
        using var gguf = new GgufFile(modelPath); // Metadata and vocabulary only; no tensor payload load.
        var metadataModel = (Qwen35Model)RuntimeHelpers.GetUninitializedObject(typeof(Qwen35Model));
        typeof(ModelBase).GetProperty(nameof(ModelBase.Tokenizer))!.SetValue(metadataModel, ModelBase.CreateTokenizerFromGguf(gguf));
        typeof(ModelBase).GetField("<Config>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(metadataModel, new ModelConfig
            {
                Architecture = gguf.GetString("general.architecture") ?? throw new InvalidDataException("Missing GGUF architecture."),
                ChatTemplate = gguf.GetString("tokenizer.chat_template") ?? throw new InvalidDataException("Missing GGUF chat template."),
            });
        fixtures = await CapturePrompts(metadataModel);
        Console.WriteLine($"[parent-prefix] describe-only shared={fixtures.Parent.SharedPrefixTokens}/{fixtures.Children[0].SharedPrefixTokens} ancestor={string.Join(',', fixtures.Parent.PublicCheckpointBoundaries)} tools={fixtures.Parent.Tools.Count}/{fixtures.Children[0].Tools.Count}");
    }
    else
    {
        var load = Stopwatch.StartNew();
        using var model = ModelBase.Create(modelPath, BackendType.GgmlCuda);
        loadMilliseconds = load.Elapsed.TotalMilliseconds;
        if (model is not Qwen35Model) throw new InvalidOperationException("Requires a Qwen35 family model.");
        initialCacheCapacity = typeof(Qwen35Model).GetField("_initialKvCacheCapacity", BindingFlags.Instance | BindingFlags.NonPublic)
            ?.GetValue(model) as int?;
        kvDtype = model.KvCacheDtype.ToString();
        fixtures = await CapturePrompts(model);
        Require(fixtures.Parent.PublicCheckpointBoundaries.Length == 1, "Fixture does not have one production-nominated ancestor checkpoint.");
        Require(fixtures.Children.All(child => child.PublicCheckpointBoundaries.SequenceEqual(fixtures.Parent.PublicCheckpointBoundaries)),
            "Root and children disagree on the production-nominated ancestor checkpoint.");
        Require(fixtures.All.All(fixture => fixture.Tokens.Count + steps < 8192), "Fixture exceeds context capacity.");
        Console.WriteLine($"[parent-prefix] shared={fixtures.Parent.SharedPrefixTokens}/{fixtures.Children[0].SharedPrefixTokens} ancestor={fixtures.Parent.PublicCheckpointBoundaries[0]} tools={fixtures.Parent.Tools.Count}/{fixtures.Children[0].Tools.Count}");
        if (diagnoseState)
        {
            stateDiagnostics = StateDiagnostics.Run((Qwen35Model)model, fixtures, steps);
            if (!stateDiagnostics.SamePartitionCloneExact || !stateDiagnostics.IntermediateCaptureExact) Environment.ExitCode = 1;
        }
        else
        {
        for (int pair = 0; pair < pairs; pair++)
        {
            foreach (bool enabled in pair % 2 == 0 ? new[] { true, false } : new[] { false, true })
            {
                var run = new WorkflowRun($"pair-{pair + 1}-{(enabled ? "ancestor" : "baseline")}", pair + 1, enabled, baselinePolicy);
                runs.Add(run);
                try { await RunCase(model, fixtures, run, steps, pair == 0, WriteReport); }
                catch (Exception exception)
                {
                    run.Error = exception.ToString();
                    failures.Add($"{run.Label}: {exception.Message}");
                    Console.Error.WriteLine(run.Error);
                }
                WriteReport(); // Preserve raw rows before checking correctness.
            }
            WorkflowRun on = runs.Single(run => run.Pair == pair + 1 && run.AncestorEnabled);
            WorkflowRun baseline = runs.Single(run => run.Pair == pair + 1 && !run.AncestorEnabled);
            CheckPair(on, baseline, fixtures, steps, pair == 0, failures);
            WriteReport();
        }
        passed = failures.Count == 0;
        if (!passed) Environment.ExitCode = 1;
        }
    }
}
catch (Exception exception)
{
    error = exception.ToString();
    Console.Error.WriteLine(error);
    Environment.ExitCode = 1;
}
finally
{
    WriteReport();
    Console.WriteLine($"[parent-prefix] description={describeOnly} passed={passed} report={Path.GetFullPath(output)}");
}

void WriteReport()
{
    string? nativePath = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
        .FirstOrDefault(module => Path.GetFileName(module.FileName).Equals("GgmlOps.dll", StringComparison.OrdinalIgnoreCase))?.FileName;
    WorkflowRun[] on = runs.Where(run => run.AncestorEnabled && run.TotalMilliseconds.HasValue).ToArray();
    WorkflowRun[] baseline = runs.Where(run => !run.AncestorEnabled && run.TotalMilliseconds.HasValue).ToArray();
    bool benchmarkComplete = on.Length == pairs && baseline.Length == pairs;
    double? onMedian = benchmarkComplete ? Median(on.Select(run => run.TotalMilliseconds!.Value)) : null;
    double? baselineMedian = benchmarkComplete ? Median(baseline.Select(run => run.TotalMilliseconds!.Value)) : null;
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
    File.WriteAllText(output, JsonSerializer.Serialize(new
    {
        DescribeOnly = describeOnly, DiagnoseState = diagnoseState, StateDiagnostics = stateDiagnostics,
        ValidationPassed = passed, PerformanceQualified = passed && benchmarkComplete,
        BaselinePolicy = baselinePolicy,
        BenchmarkComparison = baselinePolicy == "child-boundaries"
            ? "Mechanistic counterfactual: both arms nominate the same intermediate child checkpoints; only the ancestor arm nominates the parent checkpoint. This isolates parent-to-child state reuse with identical child prefill partitions, including all parent and child checkpoint capture costs. It does not measure speed relative to an older release."
            : "Previous full-public-checkpoint policy: baseline omits intermediate checkpoint nominations for both parent and children. Prefill partitions may differ; strict paired output equality is still required to qualify performance.",
        Error = error, Failures = failures, Model = Path.GetFullPath(modelPath), ModelBytes = new FileInfo(modelPath).Length,
        Backend = describeOnly ? "metadata-only" : "ggml_cuda", KvDtype = kvDtype, Context = 8192,
        MaxNewTokens = steps, Pairs = pairs, ModelLoadMilliseconds = loadMilliseconds,
        InitialCacheTokensEnvironment = Environment.GetEnvironmentVariable("TS_KV_INITIAL_TOKENS"),
        ActualInitialCacheCapacityTokens = initialCacheCapacity, NativePath = nativePath,
        NativeSha256 = nativePath == null ? null : Hash(nativePath),
        ModelAssemblySha256 = Hash(typeof(Qwen35Model).Assembly.Location),
        RuntimeAssemblySha256 = Hash(typeof(InferenceEngine).Assembly.Location),
        AgentAssemblySha256 = Hash(typeof(MultiAgentSession).Assembly.Location),
        ChatAssemblySha256 = Hash(Assembly.Load("TensorSharp.Chat").Location),
        ProbeAssemblySha256 = Hash(typeof(WorkflowRun).Assembly.Location),
        GgmlRevision = Environment.GetEnvironmentVariable("TS_VALIDATION_GGML_REVISION"),
        Device = Environment.GetEnvironmentVariable("TS_VALIDATION_DEVICE"),
        Settings = Config(), Execution = ExecutionOptions.FromEnvironment(), Fixtures = fixtures, Runs = runs,
        AncestorMedianTotalMilliseconds = onMedian, BaselineMedianTotalMilliseconds = baselineMedian,
        EndToEndSpeedup = onMedian.HasValue ? baselineMedian / onMedian : null,
        AncestorMedianParentMilliseconds = benchmarkComplete ? Median(on.Select(run => run.Stages.Single(stage => stage.Name == "parent").Milliseconds)) : (double?)null,
        BaselineMedianParentMilliseconds = benchmarkComplete ? Median(baseline.Select(run => run.Stages.Single(stage => stage.Name == "parent").Milliseconds)) : (double?)null,
        AncestorMedianChildrenMilliseconds = benchmarkComplete ? Median(on.Select(run => run.Stages.Single(stage => stage.Name == "children").Milliseconds)) : (double?)null,
        BaselineMedianChildrenMilliseconds = benchmarkComplete ? Median(baseline.Select(run => run.Stages.Single(stage => stage.Name == "children").Milliseconds)) : (double?)null,
        Limitations = "Direct engine parent-plus-two-reviewer fixture, not HTTP task completion. Both arms enable radix caching with public checkpoint budget 2. Parent and child token inputs are identical across arms; only automatic intermediate public checkpoint nominations differ. Production MultiAgentSession creates messages and profiles; the production ChatGenerationPipeline computes prefix boundaries through reflection. Nine parent tools versus seven read-only child tools use production skill, shell and agent declarations, but the concise governing prompt omits the HTTP host skill catalog. No declared tool executes. Both child requests are queued behind a closed ComputeGate before awaiting completion. Total measured workflow includes parent prefill, intermediate and full checkpoint creation, parent decoding, both child prefills/adoptions and decoding, plus scheduling between stages. Model loading, prompt rendering and engine creation are excluded. Warm-prefix preservation checks after the first workflow use the same engine and default public budget, but are outside the timed workflow. Strict paired validation checks every generated token ID, actual count, status and finish reason; it does not compare logits or require a complete answer within the output cap. Model-owned checkpoint copies and separate writable state remain; neither physical KV-page sharing nor reduced VRAM is established. Cached payload memory sampled after stages excludes active peak, weights, allocator reserves and private/primary bytes. Only the recorded model/device/cache dtype is covered; dependency and device descriptions are operator supplied."
    }, new JsonSerializerOptions { WriteIndented = true }));
}

static async Task<FixtureSet> CapturePrompts(ModelBase model)
{
    var captured = new ConcurrentDictionary<string, (List<ChatMessage> Messages, List<ToolFunction> Tools)>();
    var history = new List<ChatMessage>
    {
        new() { Role = "system", Content = "You are a careful assistant. Follow the assigned task, preserve supplied facts, and check every calculation. Distinguish verified results from assumptions. Do not speculate about the causes of calculation errors." },
        new() { Role = "user", Content = "Private parent instruction: compare two independent reviews. Proposal A: 120 units at $25 each, variable cost $13 each, fixed cost $300, claimed profit $1,300. Proposal B: 200 customers at $18 each, variable cost $7 each, fixed cost $400, claimed profit $1,900. Return a comparison table with revenue, total cost, actual profit, and exact profit overstatement. This private parent conversation must not appear in the child governing messages." },
    };
    Require(ShellProgram.TryResolve(null, out ShellProgram? shell, out string? shellError), shellError ?? "No shell found for declaration.");
    var available = SkillTools.BuiltIn(allowScripts: true);
    available.Add(ShellTools.DeclareShell(new CodeExecOptions
    {
        Enabled = true, AllowInstall = true, AllowNetwork = true, Unconfined = true,
    }, shell!, persists: false, fileTools: false));
    List<ToolFunction> parentTools = MultiAgentTools.Merge(available);
    var options = new MultiAgentOptions { Enabled = true, AllowWorkerTools = true };
    List<ChatMessage> parentMessages = MultiAgentPrompt.Apply(history, options);
    await using var agents = new MultiAgentSession(history, parentTools, new SkillToolContext([]),
        id => (messages, tools, _) =>
        {
            captured[id] = (new(messages), new(tools!));
            return Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = "Prompt captured." }));
        }, options);
    var profiles = agents.GetPromptProfiles().ToList();
    profiles.Add(new MultiAgentPromptProfile(parentMessages.TakeWhile(message => message.Role is "system" or "developer").ToList(), parentTools));
    string[] names = ["proposal_a", "proposal_b"];
    string[] tasks =
    [
        "Independently review Proposal A. Sell 120 units at $25 each. Variable cost: $13 per unit. Fixed cost: $300. Claimed profit: $1,300. Calculate revenue, total cost, actual profit, and the exact profit overstatement. Show your arithmetic. Do not speculate about the causes of the error.",
        "Independently review Proposal B. Serve 200 customers paying $18 each. Variable cost: $7 per customer. Fixed cost: $400. Claimed profit: $1,900. Calculate revenue, total cost, actual profit, and the exact profit overstatement. Show your arithmetic. Do not speculate about the causes of the error."
    ];
    for (int i = 0; i < names.Length; i++)
    {
        SkillToolResult result = await agents.ExecuteAsync(new ToolCall
        {
            Name = "spawn_agent", Arguments = new() { ["task_name"] = names[i], ["agent_type"] = "reviewer", ["task"] = tasks[i] }
        });
        Require(result.Ok, result.Content);
    }
    SkillToolResult waited = await agents.ExecuteAsync(new ToolCall
        { Name = "wait_agent", Arguments = new() { ["timeout_ms"] = 10000 } });
    Require(waited.Ok && captured.Count == 2, "Could not capture both child prompts.");
    var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
    var boundaries = new ProductionBoundaries(renderer);
    PromptFixture Render(string name, List<ChatMessage> messages, List<ToolFunction> tools)
    {
        List<int> tokens = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, messages,
            model.Config.Architecture, true, tools, enableThinking: false);
        int shared = boundaries.Shared(model, messages, tokens, tools);
        int[] checkpoints = boundaries.Ancestors(model, tokens, shared, profiles);
        return new PromptFixture(name, shared, checkpoints, tokens, messages, tools);
    }
    PromptFixture parent = Render("parent", parentMessages, parentTools);
    PromptFixture[] children = names.Select(name =>
    {
        var child = captured["/root/" + name];
        Require(child.Messages.All(message => message.Content?.Contains("Private parent instruction:", StringComparison.Ordinal) != true),
            "Parent private history appeared in child messages.");
        Require(!child.Messages.TakeWhile(message => message.Role is "system" or "developer")
            .Any(message => message.Content?.Contains("/root/", StringComparison.Ordinal) == true),
            "Agent identity leaked into public governing messages.");
        return Render(name, child.Messages, child.Tools);
    }).ToArray();
    Require(parent.Tools.Count == 9 && children.All(child => child.Tools.Count == 7), "Expected production-shaped nine/seven tool fixture.");
    Require(children[0].SharedPrefixTokens >= 64 && children[0].SharedPrefixTokens == children[1].SharedPrefixTokens
        && children[0].Tokens.Take(children[0].SharedPrefixTokens).SequenceEqual(children[1].Tokens.Take(children[1].SharedPrefixTokens)),
        "Sibling public prefixes differ.");
    int lcp = 0;
    while (lcp < Math.Min(parent.Tokens.Count, children[0].Tokens.Count) && parent.Tokens[lcp] == children[0].Tokens[lcp]) lcp++;
    return new FixtureSet(parent, children, lcp, profiles.Count, shell!.Path);
}

static SchedulerConfig Config() => new()
{
    MaxNumBatchedTokens = 4096, MaxNumRunningSequences = 4,
    MaxPrefillChunkSize = 256, SoloPrefillChunkSize = 8192,
    NumBlocks = 256, BlockSize = 256, EnablePrefixCaching = true,
    StopRepetition = false, DecodeQuantumTokens = 256,
};

static async Task RunCase(ModelBase model, FixtureSet fixtures, WorkflowRun run, int steps, bool warmChecks, Action persist)
{
    var gate = new ComputeGate();
    using var engine = new InferenceEngine(model, Config()) { ComputeGate = gate };
    DateTime? workflowStarted = null;
    void Released(DateTime at) => workflowStarted ??= at;
    run.Stages.Add(await RunStage(model, engine, gate, [fixtures.Parent], run, "parent", steps, Released));
    run.Stages.Add(await RunStage(model, engine, gate, fixtures.Children, run, "children", steps, Released));
    run.TotalMilliseconds = (run.Stages[^1].FinishedAt - workflowStarted!.Value).TotalMilliseconds;
    persist();
    if (warmChecks)
    {
        run.Stages.Add(await RunStage(model, engine, gate, [fixtures.Parent], run, "warm-parent", steps, _ => { }));
        persist();
        run.Stages.Add(await RunStage(model, engine, gate, fixtures.Children, run, "warm-children", steps, _ => { }));
        persist();
    }
}

static async Task<StageRun> RunStage(ModelBase model, InferenceEngine engine, ComputeGate gate,
    PromptFixture[] fixtures, WorkflowRun run, string name, int steps, Action<DateTime> released)
{
    gate.Close();
    long held = engine.StepsHeldByGate;
    var requests = fixtures.Select(fixture => new SequenceState($"{run.Label}-{name}-{fixture.Name}", fixture.Tokens,
        steps, 256, SamplingConfig.Greedy, sharedPrefixTokens: fixture.SharedPrefixTokens,
        cacheScope: Guid.NewGuid().ToString("N"),
        publicCheckpointBoundaries: run.AncestorEnabled
            || run.BaselinePolicy == "child-boundaries" && fixture.Name != "parent"
                ? fixture.PublicCheckpointBoundaries : null)).ToArray();
    var handles = requests.Select(request => engine.SubmitRequest(request)).ToArray();
    var park = Stopwatch.StartNew();
    while (engine.StepsHeldByGate <= held && park.Elapsed < TimeSpan.FromSeconds(10)) await Task.Delay(1);
    Require(engine.StepsHeldByGate > held, "Engine did not park before releasing queued requests.");
    DateTime releasedAt = DateTime.UtcNow;
    TimeSpan before = engine.TotalForwardTime;
    long batchedBefore = ArenaSteps(model);
    var elapsed = Stopwatch.StartNew();
    released(releasedAt);
    gate.Open();
    InferenceCompletion[] completed = await Task.WhenAll(handles.Select(handle => handle.Completion)).WaitAsync(TimeSpan.FromMinutes(10));
    elapsed.Stop();
    DateTime finished = DateTime.UtcNow;
    RequestRun[] rows = completed.Select((completion, index) => new RequestRun(fixtures[index].Name,
        completion.PromptTokenCount, completion.PrefixCacheReusedTokens,
        completion.PromptTokenCount - completion.PrefixCacheReusedTokens, completion.OutputTokenCount,
        requests[index].OutputTokens.Count(token => !model.Tokenizer.IsEos(token)), requests[index].OutputTokens.ToArray(),
        model.Tokenizer.Decode(requests[index].OutputTokens.ToList()), completion.Status.ToString(), completion.FinishReason,
        completion.FirstTokenAt.HasValue ? (completion.FirstTokenAt.Value - releasedAt).TotalMilliseconds : null,
        requests[index].PublicCheckpointBoundaries.ToArray())).ToArray();
    CacheMemorySample? memory = null;
    lock (model.GpuComputeLock)
    {
        if (model is IPrefixCacheModelDiagnostics diagnostics && model is IPrefixCacheModel cache)
        {
            PayloadFootprint[] payloads = diagnostics.RetainedPayloadKeys.Select(cache.MeasureEndState).ToArray();
            memory = new CacheMemorySample(payloads.Length, diagnostics.PrivateHolderCount, diagnostics.PrimaryCacheLength,
                payloads.Sum(payload => payload.Bytes.HostKv), payloads.Sum(payload => payload.Bytes.StateSnapshot),
                payloads.Sum(payload => payload.Bytes.DeviceKv));
        }
    }
    Console.WriteLine($"[parent-prefix] {run.Label}/{name} ms={elapsed.Elapsed.TotalMilliseconds:F2} reused={string.Join(',', rows.Select(row => row.ReusedTokens))} generated={string.Join(',', rows.Select(row => row.ActualGeneratedTokens))}");
    return new StageRun(name, elapsed.Elapsed.TotalMilliseconds, (engine.TotalForwardTime - before).TotalMilliseconds,
        releasedAt, finished, true, ArenaSteps(model) - batchedBefore,
        (model as Qwen35Model)?.BatchedFusedDecodeDeclineReason, rows, memory);
}

static void CheckPair(WorkflowRun on, WorkflowRun baseline, FixtureSet fixtures, int maxTokens, bool warm, List<string> failures)
{
    foreach (WorkflowRun run in new[] { on, baseline })
    {
        foreach (StageRun stage in run.Stages)
        {
            foreach (RequestRun request in stage.Requests)
            {
                bool clean = request.CompletionStatus == nameof(SequenceStatus.FinishedLengthCapped) && request.ActualGeneratedTokens == maxTokens
                    || request.CompletionStatus == nameof(SequenceStatus.FinishedStopped) && request.FinishReason == "eos";
                if (!clean || request.NonEosGeneratedTokens <= 0 || request.ActualGeneratedTokens > maxTokens
                    || request.OutputTokens.Length != request.ActualGeneratedTokens)
                    failures.Add($"{run.Label}/{stage.Name}/{request.Name}: invalid completion {request.CompletionStatus}/{request.FinishReason}, count {request.ActualGeneratedTokens}.");
            }
        }
        Expect(run, "parent", [0]);
        Expect(run, "children", [run.AncestorEnabled ? fixtures.Parent.PublicCheckpointBoundaries[0] : 0, fixtures.Children[1].SharedPrefixTokens]);
        if (warm)
        {
            Expect(run, "warm-parent", [fixtures.Parent.SharedPrefixTokens]);
            Expect(run, "warm-children", fixtures.Children.Select(child => child.SharedPrefixTokens).ToArray());
        }
    }
    foreach (StageRun stage in on.Stages)
    {
        StageRun? other = baseline.Stages.SingleOrDefault(candidate => candidate.Name == stage.Name);
        if (other == null) { failures.Add($"{baseline.Label}: missing stage {stage.Name}."); continue; }
        for (int i = 0; i < Math.Min(stage.Requests.Length, other.Requests.Length); i++)
        {
            RequestRun a = stage.Requests[i], b = other.Requests[i];
            if (!a.OutputTokens.SequenceEqual(b.OutputTokens) || a.ActualGeneratedTokens != b.ActualGeneratedTokens
                || a.CompletionStatus != b.CompletionStatus || a.FinishReason != b.FinishReason)
                failures.Add($"Pair {on.Pair}/{stage.Name}/{a.Name}: generated token IDs, count, completion status or finish reason differ.");
        }
    }
    void Expect(WorkflowRun run, string stageName, int[] expected)
    {
        StageRun? stage = run.Stages.SingleOrDefault(candidate => candidate.Name == stageName);
        if (stage == null) { failures.Add($"{run.Label}: missing {stageName}."); return; }
        if (!stage.Requests.Select(request => request.ReusedTokens).SequenceEqual(expected))
            failures.Add($"{run.Label}/{stageName}: expected reused [{string.Join(',', expected)}], got [{string.Join(',', stage.Requests.Select(request => request.ReusedTokens))}].");
    }
}

static void Require(bool condition, string message) { if (!condition) throw new InvalidOperationException(message); }
static double Median(IEnumerable<double> values)
{
    double[] sorted = values.OrderBy(value => value).ToArray();
    return sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
}
static string Hash(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexString(SHA256.HashData(stream));
}
static long ArenaSteps(ModelBase model) => (long?)typeof(Qwen35Model)
    .GetProperty("ArenaBatchedDecodeSteps", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(model) ?? 0;

// Exercise the exact production boundary algorithm without widening Chat's public API
// or constructing lifecycle services that could load weights in --describe-only mode.
sealed class ProductionBoundaries
{
    private readonly object _pipeline;
    private readonly MethodInfo _shared, _ancestors;
    public ProductionBoundaries(KVCachePromptRenderer renderer)
    {
        Type type = Assembly.Load("TensorSharp.Chat").GetType("TensorSharp.Server.ChatGenerationPipeline", throwOnError: true)!;
        _pipeline = RuntimeHelpers.GetUninitializedObject(type);
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        type.GetField("_kvCacheRenderer", flags)!.SetValue(_pipeline, renderer);
        FieldInfo cache = type.GetField("_sharedPrefixRenders", flags)!;
        cache.SetValue(_pipeline, Activator.CreateInstance(cache.FieldType));
        type.GetField("_logger", flags)!.SetValue(_pipeline, NullLogger.Instance);
        _shared = type.GetMethod("ComputeSharedPrefixTokens", flags)!;
        _ancestors = type.GetMethod("ComputePublicCheckpointBoundaries", flags)!;
    }
    public int Shared(ModelBase model, List<ChatMessage> messages, List<int> tokens, List<ToolFunction> tools) =>
        (int)_shared.Invoke(_pipeline, [model, messages, tokens, model.Config.Architecture, tools, false, null])!;
    public int[] Ancestors(ModelBase model, List<int> tokens, int shared, IReadOnlyList<MultiAgentPromptProfile> profiles) =>
        ((IReadOnlyList<int>)_ancestors.Invoke(_pipeline, [model, tokens, shared, profiles, model.Config.Architecture, false, null])!).ToArray();
}
sealed record PromptFixture(string Name, int SharedPrefixTokens, int[] PublicCheckpointBoundaries, List<int> Tokens,
    List<ChatMessage> Messages, List<ToolFunction> Tools);
sealed record FixtureSet(PromptFixture Parent, PromptFixture[] Children, int ActualParentReviewerCommonTokens,
    int ProductionProfileCount, string DeclaredShellPath)
{
    [System.Text.Json.Serialization.JsonIgnore] public IEnumerable<PromptFixture> All => new[] { Parent }.Concat(Children);
}
sealed record RequestRun(string Name, int PromptTokens, int ReusedTokens, int PrefillTokens, int ActualGeneratedTokens,
    int NonEosGeneratedTokens, int[] OutputTokens, string OutputText, string CompletionStatus, string? FinishReason,
    double? TimeToFirstTokenMilliseconds, int[] DeclaredPublicCheckpointBoundaries);
sealed record StageRun(string Name, double Milliseconds, double EngineForwardMilliseconds, DateTime ReleasedAt,
    DateTime FinishedAt, bool AllSubmittedBeforeGateOpened, long AcceptedBatchedArenaSteps,
    string? LastBatchedDecodeDeclineReason, RequestRun[] Requests, CacheMemorySample? CachedPayloadMemoryAfterCompletion);
sealed class WorkflowRun(string label, int pair, bool ancestorEnabled, string baselinePolicy)
{
    public string Label { get; } = label;
    public int Pair { get; } = pair;
    public bool AncestorEnabled { get; } = ancestorEnabled;
    public string BaselinePolicy { get; } = baselinePolicy;
    public List<StageRun> Stages { get; } = [];
    public double? TotalMilliseconds { get; set; }
    public string? Error { get; set; }
}
sealed record CacheMemorySample(int CachedPayloadCount, int PrivateHolderCount, int PrimaryCacheTokens,
    long CachedHostKvBytes, long CachedRecurrentStateBytes, long CachedDeviceMirrorBytes);
