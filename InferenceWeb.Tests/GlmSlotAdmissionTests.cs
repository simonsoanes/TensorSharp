// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

[CollectionDefinition("GLM slot admission environment", DisableParallelization = true)]
public sealed class GlmSlotAdmissionEnvironmentCollection { }

/// <summary>
/// A GLM sequence slot beyond the one the context was sized for must leave its devices room for another
/// compute graph plus the reserve (glm_slot_fits). GLM-5.3-Flash on a 6x A40 layer split admitted a third
/// extra slot that left device 0 116 MiB short of the new request's own first prefill graph, and the
/// request failed. The reserve is raised past any device here so the refusal is observable on a tiny model;
/// the primary the context was sized for must keep working, and a refused slot must leave nothing behind.
/// </summary>
[Collection("GLM slot admission environment")]
public sealed class GlmSlotAdmissionTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-glm-admit-" + Guid.NewGuid().ToString("N"));

    public GlmSlotAdmissionTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    // CUDA only: device memory is what the rule reads (a CPU run has none to check), and the synthetic
    // model's MLA head sizes (k 48, v 32) have no ggml-metal flash-attention kernel.
    [GlmNativeCudaFact]
    public void CudaSlotBeyondThePlannedOne_IsRefusedWithoutRoomForAnotherGraph() => Check("CUDA");

    private void Check(string backend)
    {
        string path = GlmDsaSyntheticModelBuilder.Write(Path.Combine(_dir, "tiny-glm-dsa.gguf"));
        IntPtr handle = GgmlGlmNative.LoadModel(path, 1, 256, 16, 2, backendName: backend, ctxIsHardLimit: true);
        Assert.NotEqual(IntPtr.Zero, handle);
        using var env = new NativeEnvScope();
        try
        {
            int[] prompt = Enumerable.Range(0, 24).Select(i => 65 + (i * 7) % 50).ToArray();
            var expected = new float[GgmlGlmNative.VocabSize(handle)];
            Assert.True(GgmlGlmNative.Forward(handle, prompt, expected));   // a graph is cached now

            env.Set("TS_GLM_SLOT_HEADROOM_MB", "100000000");
            Assert.Equal(-1, GgmlGlmNative.SlotAlloc(handle));
            Assert.Equal(-1, GgmlGlmNative.SlotAlloc(handle));
            // The planned slot is untouched and still the active one.
            Assert.Equal(prompt.Length, GgmlGlmNative.NPast(handle));
            var next = new float[expected.Length];
            Assert.True(GgmlGlmNative.Forward(handle, new[] { 70 }, next));

            env.Set("TS_GLM_SLOT_HEADROOM_MB", null);
            int slot = GgmlGlmNative.SlotAlloc(handle);
            Assert.True(slot > 0, "the default reserve admits a slot of a tiny model");
            Assert.True(GgmlGlmNative.SetActiveSlot(handle, slot));
            Assert.Equal(0, GgmlGlmNative.NPast(handle));
            var actual = new float[expected.Length];
            Assert.True(GgmlGlmNative.Forward(handle, prompt, actual));
            Assert.Equal(expected, actual);   // same prompt, fresh slot: the refusals left no state behind
            Assert.True(GgmlGlmNative.SetActiveSlot(handle, 0));
            Assert.True(GgmlGlmNative.SlotFree(handle, slot));
        }
        finally
        {
            GgmlGlmNative.Free(handle);
        }
    }
}
