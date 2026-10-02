using TensorSharp.Runtime.Scheduling;
namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]

public class ModelContextLengthTests
{
    [Fact]
    public void ResolveConfiguredContextLength_PrefersExplicitOverride()
    {
        var metadata = new Dictionary<string, object>
        {
            ["mistral3.context_length"] = 32768u,
            ["mistral3.rope.scaling.original_context_length"] = 4096u
        };

        int resolved = ModelBase.ResolveConfiguredContextLength("mistral3", metadata, 4096, 8192, out string source);

        Assert.Equal(8192, resolved);
        Assert.Equal("MAX_CONTEXT", source);
    }

    [Fact]
    public void ResolveModelContextLength_IgnoresTheHostLimit()
    {
        var metadata = new Dictionary<string, object>
        {
            ["qwen35.context_length"] = 262144u,
            ["qwen35.rope.scaling.original_context_length"] = 32768u
        };

        int modelContext = ModelBase.ResolveModelContextLength(
            "qwen35", metadata, 4096, out string source);
        int activeContext = ModelBase.ResolveConfiguredContextLength(
            "qwen35", metadata, 4096, 32768, out _);

        Assert.Equal(262144, modelContext);
        Assert.Equal("qwen35.context_length", source);
        Assert.Equal(32768, activeContext);
    }

    [Fact]
    public void ModelRetainsItsDeclaredContextWhileServingASmallerHostWindow()
    {
        string path = Path.Combine(Path.GetTempPath(), $"context-probe-{Guid.NewGuid():N}.gguf");
        string previous = Environment.GetEnvironmentVariable("MAX_CONTEXT");
        try
        {
            WriteContextGguf(path, "qwen35", 262144);
            Environment.SetEnvironmentVariable("MAX_CONTEXT", "32768");

            using var model = new ContextProbeModel(path);

            Assert.Equal(262144, model.Config.DeclaredContextLength);
            Assert.Equal(32768, model.EffectiveContextLength);
        }
        finally
        {
            Environment.SetEnvironmentVariable("MAX_CONTEXT", previous);
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    [Trait("Category", "Bench")]
    public void DeclaredContextResolutionBenchmark()
    {
        var metadata = new Dictionary<string, object>
        {
            ["qwen35.context_length"] = 262144u,
            ["qwen35.rope.scaling.original_context_length"] = 32768u
        };
        const int iterations = 1_000_000;
        int checksum = 0;
        Assert.Equal(
            262144,
            ModelBase.ResolveModelContextLength("qwen35", metadata, 4096, out _));

        for (int i = 0; i < 10_000; i++)
            checksum ^= ModelBase.ResolveModelContextLength("qwen35", metadata, 4096, out _);

        long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            checksum ^= ModelBase.ResolveModelContextLength("qwen35", metadata, 4096, out _);
        clock.Stop();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;

        double nanoseconds = clock.Elapsed.TotalMilliseconds * 1_000_000 / iterations;
        Console.WriteLine(
            $"[context-metadata] {iterations:N0} resolutions: {clock.Elapsed.TotalMilliseconds:F1} ms, "
            + $"{nanoseconds:F1} ns/op, {allocated / (double)iterations:F1} B/op");

        Assert.Equal(0, checksum);
    }

    [Fact]
    public void ResolveConfiguredContextLength_UsesStandardContextLengthBeforeOriginalContext()
    {
        var metadata = new Dictionary<string, object>
        {
            ["gptoss.context_length"] = 131072u,
            ["gptoss.rope.scaling.original_context_length"] = 4096u
        };

        int resolved = ModelBase.ResolveConfiguredContextLength("gptoss", metadata, 4096, null, out string source);

        Assert.Equal(131072, resolved);
        Assert.Equal("gptoss.context_length", source);
    }

    [Fact]
    public void ResolveConfiguredContextLength_FallsBackWhenMetadataIsMissing()
    {
        int resolved = ModelBase.ResolveConfiguredContextLength(
            "nemotron_h",
            new Dictionary<string, object>(),
            4096,
            null,
            out string source);

        Assert.Equal(4096, resolved);
        Assert.Equal("fallback", source);
    }

    [Fact]
    public void ResolveInitialCacheAllocationLength_CapsMlxGpuBackendsUnlessContextIsExplicit()
    {
        string previousMaxContext = Environment.GetEnvironmentVariable("MAX_CONTEXT");
        try
        {
            Environment.SetEnvironmentVariable("MAX_CONTEXT", null);

            Assert.Equal(2048, ModelBase.ResolveInitialCacheAllocationLength(BackendType.Mlx, 262144));
            Assert.Equal(2048, ModelBase.ResolveInitialCacheAllocationLength(BackendType.Mlx, 4096));
            Assert.Equal(2048, ModelBase.ResolveInitialCacheAllocationLength(BackendType.Cuda, 262144));
            Assert.Equal(
                8192,
                ModelBase.ResolveInitialCacheAllocationLength(
                    BackendType.Cuda,
                    262144,
                    gpuDefault: 8192,
                    nativeCudaDefault: 8192));
            Assert.Equal(8192, ModelBase.ResolveInitialCacheAllocationLength(BackendType.GgmlCuda, 262144));
            Assert.Equal(262144, ModelBase.ResolveInitialCacheAllocationLength(BackendType.Cpu, 262144));

            // Invalid values are ignored by context resolution and must not
            // accidentally disable the GPU's safe initial-allocation cap.
            Environment.SetEnvironmentVariable("MAX_CONTEXT", "invalid");
            Assert.Equal(2048, ModelBase.ResolveInitialCacheAllocationLength(BackendType.Cuda, 262144));

            Environment.SetEnvironmentVariable("MAX_CONTEXT", "262144");
            Assert.Equal(262144, ModelBase.ResolveInitialCacheAllocationLength(BackendType.Mlx, 262144));
            Assert.Equal(262144, ModelBase.ResolveInitialCacheAllocationLength(BackendType.Cuda, 262144));

            Assert.Equal(
                2049,
                ModelBase.ResolvePrefillWarmupInputLength(
                    targetLength: 2048,
                    maxContextLength: 8192,
                    tokenOverhead: 1,
                    explicitLength: false));
            Assert.Equal(
                2048,
                ModelBase.ResolvePrefillWarmupInputLength(
                    targetLength: 2048,
                    maxContextLength: 8192,
                    tokenOverhead: 1,
                    explicitLength: true));
        }
        finally
        {
            Environment.SetEnvironmentVariable("MAX_CONTEXT", previousMaxContext);
        }
    }

    [Fact]
    public void UsesLightweightPrefillWarmupByDefault_IncludesMetalWithoutChangingDiscreteGgmlBackends()
    {
        Assert.True(ModelBase.UsesLightweightPrefillWarmupByDefault(BackendType.GgmlMetal));
        Assert.True(ModelBase.UsesLightweightPrefillWarmupByDefault(BackendType.Mlx));
        Assert.True(ModelBase.UsesLightweightPrefillWarmupByDefault(BackendType.Cpu));

        Assert.False(ModelBase.UsesLightweightPrefillWarmupByDefault(BackendType.GgmlCuda));
        Assert.False(ModelBase.UsesLightweightPrefillWarmupByDefault(BackendType.GgmlVulkan));
        Assert.False(ModelBase.UsesLightweightPrefillWarmupByDefault(BackendType.Cuda));
        Assert.False(ModelBase.UsesLightweightPrefillWarmupByDefault(BackendType.GgmlCpu));
    }

    [Fact]
    public void ResolvePrefillWarmupTargetLength_UsesSafeMetalDefaultAndPreservesExplicitOverride()
    {
        Assert.Equal(
            32,
            ModelBase.ResolvePrefillWarmupTargetLength(
                BackendType.GgmlMetal, false, false, false, 32, null));
        Assert.Equal(
            2048,
            ModelBase.ResolvePrefillWarmupTargetLength(
                BackendType.GgmlCuda, false, false, false, 32, null));
        Assert.Equal(
            2048,
            ModelBase.ResolvePrefillWarmupTargetLength(
                BackendType.GgmlVulkan, false, false, false, 32, null));
        Assert.Equal(
            96,
            ModelBase.ResolvePrefillWarmupTargetLength(
                BackendType.Cuda, false, false, false, 96, null));

        Assert.Equal(
            2048,
            ModelBase.ResolvePrefillWarmupTargetLength(
                BackendType.GgmlMetal, false, false, false, 32, 2048));
    }

    // --- Prefill KV reservation (BatchExecutor.BuildPrefillChunk -> PrepareForPrefill) ---
    //
    // Regression: Qwen3.8-27B-Q8_0 (29.0 GB of weights) served on a 48 GB M5 Pro
    // with --max-tokens 256000. The first request declared prompt + generation
    // budget = 260,864 tokens, Qwen35Model reserved all of it as dense K/V
    // (16 attention layers x 4 KV heads x 256 head dim x 2 (K+V) x 2 bytes =
    // 64 KiB/token = 16.3 GiB), and weights + KV blew past Metal's 40.2 GB
    // recommendedMaxWorkingSetSize. The command buffer died with
    // kIOGPUCommandBufferCallbackErrorOutOfMemory, ggml-metal latched its sticky
    // error state, and every later graph failed — surfacing as
    // "Native GGML get_rows_quant failed" from the next forward's embedding.

    /// <summary>Per-token K/V cost of the model in the incident above.</summary>
    private const long Qwen3827BKvBytesPerToken = 2L * 16 * 4 * 256 * 2;

    [Fact]
    public void ResolvePrefillReservationLength_TrimsTheDeclaredBudgetToWhatTheDeviceHasSpare()
    {
        // ~10.4 GB spare once 29.0 GB of weights are resident in a 40.2 GB working set.
        const long spare = 10_400L * 1024 * 1024;

        int fitted = ModelBase.ResolvePrefillReservationLength(
            spare, Qwen3827BKvBytesPerToken, requiredContextTokens: 260864, currentCapacityTokens: 2048);

        Assert.True(fitted < 260864, "the declared budget must not be reserved whole");
        Assert.True(fitted >= 2048, "never below what is already reserved");
        Assert.Equal(0, fitted % 256);
        // Half the spare is the KV share; the rest covers graph scratch.
        Assert.True((long)fitted * Qwen3827BKvBytesPerToken <= spare / 2);
    }

    // --- What a request reserves before its first chunk (BatchExecutor.ResolvePrefillReservation) ---
    //
    // Regression: the phone's reply length was set to its largest rung, 262,144 tokens,
    // inside a 32,768-token window. prompt + max_new_tokens capped to the window is the
    // WHOLE window, so every request -- "hi" included -- reserved 32k rows of K/V in its
    // holder, paid in host memory and again in the Metal mirror, and the engine kept
    // several such holders. The cap on the generation share is what a memory-bound host
    // sets; the cache still grows on demand past it.

    [Fact]
    public void ResolvePrefillReservation_CapsTheGenerationShareButNeverThePrompt()
    {
        // Uncapped: today's arithmetic, prompt + budget bounded by the window.
        Assert.Equal(32768, BatchExecutor.ResolvePrefillReservation(5878, 262144, 32768, generationReserveMax: 0));
        Assert.Equal(7926, BatchExecutor.ResolvePrefillReservation(5878, 2048, 32768, generationReserveMax: 0));

        // Capped: the prompt is always reserved whole, the reply share at most the cap,
        // and the sum snapped up to the 2,048-token step so a conversation growing a
        // thousand tokens a round reallocates every other round, not every round.
        Assert.Equal(8192, BatchExecutor.ResolvePrefillReservation(5878, 262144, 32768, generationReserveMax: 1024));
        Assert.Equal(8192, BatchExecutor.ResolvePrefillReservation(5878, 512, 32768, generationReserveMax: 1024));
        Assert.Equal(10240, BatchExecutor.ResolvePrefillReservation(8193, 262144, 32768, generationReserveMax: 1024));
        Assert.Equal(2048, BatchExecutor.ResolvePrefillReservation(100, 100, 32768, generationReserveMax: 1024));

        // A prompt that already fills the window is bounded by the window, not refused.
        Assert.Equal(32768, BatchExecutor.ResolvePrefillReservation(32768, 262144, 32768, generationReserveMax: 1024));

        // No window known: nothing to bound against.
        Assert.Equal(263168, BatchExecutor.ResolvePrefillReservation(1024, 262144, 0, generationReserveMax: 0));
    }

    [Fact]
    public void ResolveInitialCacheAllocationLength_HonoursAnExplicitInitialSizeEvenWithMaxContext()
    {
        // The phone: MAX_CONTEXT is the ceiling it can afford, TS_KV_INITIAL_TOKENS what to
        // commit before a request declares its need. Without the second, an explicit
        // context allocated the whole window for the primary cache and for every holder.
        string prevCtx = Environment.GetEnvironmentVariable("MAX_CONTEXT");
        string prevInitial = Environment.GetEnvironmentVariable("TS_KV_INITIAL_TOKENS");
        try
        {
            Environment.SetEnvironmentVariable("MAX_CONTEXT", "32768");
            Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", null);
            Assert.Equal(32768, ModelBase.ResolveInitialCacheAllocationLength(BackendType.GgmlMetal, 32768));

            Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", "2048");
            Assert.Equal(2048, ModelBase.ResolveInitialCacheAllocationLength(BackendType.GgmlMetal, 32768));
            // Never more than the window itself, and the knob applies to every backend.
            Assert.Equal(1024, ModelBase.ResolveInitialCacheAllocationLength(BackendType.GgmlCpu, 1024));
            Assert.Equal(2048, ModelBase.ResolveInitialCacheAllocationLength(BackendType.GgmlCuda, 65536));
        }
        finally
        {
            Environment.SetEnvironmentVariable("MAX_CONTEXT", prevCtx);
            Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", prevInitial);
        }
    }

    [Fact]
    public void ResolvePrefillReservationLength_OnlyEverTrims()
    {
        // Room to spare: the request keeps exactly what it asked for.
        Assert.Equal(
            260864,
            ModelBase.ResolvePrefillReservationLength(
                long.MaxValue / 4, Qwen3827BKvBytesPerToken, 260864, 2048));

        // Unknown per-token cost: no opinion, today's behaviour stands.
        Assert.Equal(
            260864,
            ModelBase.ResolvePrefillReservationLength(0, 0, 260864, 2048));

        // Already reserved: nothing to do, and never a shrink.
        Assert.Equal(
            4096,
            ModelBase.ResolvePrefillReservationLength(0, Qwen3827BKvBytesPerToken, 4096, 65536));

        // Not one byte spare still keeps what is already reserved rather than
        // reserving nothing and failing the prompt outright.
        Assert.Equal(
            2048,
            ModelBase.ResolvePrefillReservationLength(0, Qwen3827BKvBytesPerToken, 260864, 2048));
    }

    [Fact]
    public void GpuMemoryBudget_BoundsReservationsOnMetalButLeavesItsTunedDefaultsAlone()
    {
        // Metal's recommendedMaxWorkingSetSize is a hard ceiling — past it a command
        // buffer fails outright and ggml-metal never recovers — so a reservation has
        // to be capped there. The steady-state buffers (initial KV allocation,
        // prefill chunk width) keep their own measured Metal constants, so Metal
        // stays out of AppliesTo. Guard both directions against drift.
        Assert.True(GpuMemoryBudget.AppliesToReservations(BackendType.GgmlMetal));
        Assert.False(GpuMemoryBudget.AppliesTo(BackendType.GgmlMetal));

        foreach (BackendType b in new[] { BackendType.GgmlCuda, BackendType.GgmlVulkan })
        {
            Assert.True(GpuMemoryBudget.AppliesTo(b));
            Assert.True(GpuMemoryBudget.AppliesToReservations(b));
        }

        foreach (BackendType b in new[] { BackendType.Cpu, BackendType.GgmlCpu, BackendType.Cuda, BackendType.Mlx })
        {
            Assert.False(GpuMemoryBudget.AppliesTo(b));
            Assert.False(GpuMemoryBudget.AppliesToReservations(b));
        }
    }

    private static void WriteContextGguf(string path, string architecture, uint contextLength)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u);
        writer.Write(0UL); // tensors
        writer.Write(2UL); // metadata entries

        WriteGgufString(writer, "general.architecture");
        writer.Write((uint)GgufValueType.String);
        WriteGgufString(writer, architecture);

        WriteGgufString(writer, architecture + ".context_length");
        writer.Write((uint)GgufValueType.Uint32);
        writer.Write(contextLength);

        int padding = (int)((32 - stream.Position % 32) % 32);
        writer.Write(new byte[padding]);
    }

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private sealed class ContextProbeModel : ModelBase
    {
        public ContextProbeModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig
            {
                Architecture = _gguf.GetString("general.architecture") ?? string.Empty,
            };
            EffectiveContextLength = ResolveConfiguredContextLength();
        }

        public int EffectiveContextLength { get; }

        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();
        protected override void ResetKVCacheCore() { }
    }
}
