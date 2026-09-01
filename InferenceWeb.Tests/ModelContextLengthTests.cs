namespace InferenceWeb.Tests;

public class ModelContextLengthTests
{
    [Fact]
    public void ResolveConfiguredContextLength_PrefersExplicitOverride()
    {
        var metadata = new Dictionary<string, object>
        {
            ["qwen3.context_length"] = 32768u,
            ["qwen3.rope.scaling.original_context_length"] = 4096u
        };

        int resolved = ModelBase.ResolveConfiguredContextLength("qwen3", metadata, 4096, 8192, out string source);

        Assert.Equal(8192, resolved);
        Assert.Equal("MAX_CONTEXT", source);
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
}
