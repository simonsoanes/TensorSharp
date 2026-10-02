// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>
/// Every GLM slot the admission rule let in may be retained at once for its conversation's next turn, so a
/// graph that later grows past the reserve (a longer context, a larger batch) has to take a retained slot's
/// memory back rather than fail a running request. A fixture hook makes graph allocations fail as a full
/// device would: the native side frees retained slots oldest first, never a slot the graph reads, reports
/// each one it freed exactly once, and fails the call cleanly once none is left.
/// </summary>
public sealed class GlmGraphReclaimTests
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void FailGraphAllocs(int n);

    [GlmNativeHooksFact]
    public void Cpu_AGraphThatDoesNotFit_FreesRetainedSlotsOldestFirst_NeverOneItReads() => Check("CPU", nGpu: 1, tp: 1);

    [GlmNativeHooksCudaFact]
    public void Cuda_AGraphThatDoesNotFit_FreesRetainedSlotsOldestFirst_NeverOneItReads() => Check("CUDA", nGpu: 1, tp: 1);

    [GlmNativeHooksCudaFact(2)]
    public void CudaTensorParallel_AGraphThatDoesNotFit_FreesRetainedSlotsOldestFirst_NeverOneItReads() => Check("CUDA", nGpu: 2, tp: 2);

    private static void Check(string backend, int nGpu, int tp)
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-glm-reclaim-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        IntPtr handle = IntPtr.Zero, library = IntPtr.Zero;
        FailGraphAllocs fail = null;
        try
        {
            string path = GlmDsaSyntheticModelBuilder.Write(Path.Combine(directory, "tiny-glm-dsa.gguf"));
            handle = GgmlGlmNative.LoadModel(path, nGpu, 256, 16, 2, backendName: backend, tp: tp, ctxIsHardLimit: true);
            Assert.NotEqual(IntPtr.Zero, handle);
            library = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            fail = Marshal.GetDelegateForFunctionPointer<FailGraphAllocs>(
                NativeLibrary.GetExport(library, "TSGgml_GlmTestFailGraphAllocs"));

            int vocab = GgmlGlmNative.VocabSize(handle);
            int[] prompt = { 65, 66, 67, 68, 69 };
            var expected = new float[vocab];
            Assert.True(GgmlGlmNative.Forward(handle, prompt, expected));   // the primary, slot 0

            // Two finished conversations retained, oldest first, and a request running on its own slot. Graphs
            // are cached per slot, so the running slot's first forward has to allocate one.
            int older = GgmlGlmNative.SlotAlloc(handle), newer = GgmlGlmNative.SlotAlloc(handle);
            int running = GgmlGlmNative.SlotAlloc(handle);
            Assert.True(older > 0 && newer > 0 && running > 0);
            Assert.True(GgmlGlmNative.SetReclaimableSlots(handle, new[] { older, newer }));
            Assert.True(GgmlGlmNative.SetActiveSlot(handle, running));

            // The first allocation fails as a full device would: the OLDER retained slot goes, the graph is
            // built again, and the request answers exactly as the primary did.
            fail(1);
            var actual = new float[vocab];
            Assert.True(GgmlGlmNative.Forward(handle, prompt, actual));
            Assert.Equal(expected, actual);
            var ids = new int[4];
            Assert.Equal(1, GgmlGlmNative.TakeReclaimedSlots(handle, ids));
            Assert.Equal(older, ids[0]);
            Assert.Equal(0, GgmlGlmNative.TakeReclaimedSlots(handle, ids));   // reported once
            Assert.False(GgmlGlmNative.SetActiveSlot(handle, older));        // its memory is gone
            Assert.True(GgmlGlmNative.SetActiveSlot(handle, newer));

            // A slot the graph reads is never freed, even when it is listed and nothing else is: the call fails
            // cleanly and leaves the slot as it was.
            fail(8);
            Assert.False(GgmlGlmNative.Forward(handle, prompt, actual));
            fail(0);
            Assert.Equal(0, GgmlGlmNative.TakeReclaimedSlots(handle, ids));
            Assert.Equal(0, GgmlGlmNative.NPast(handle));
            Assert.True(GgmlGlmNative.Forward(handle, prompt, actual));
            Assert.Equal(expected, actual);

            // Nothing listed: a graph that does not fit fails the call and frees nothing.
            Assert.True(GgmlGlmNative.SetReclaimableSlots(handle, Array.Empty<int>()));
            int spare = GgmlGlmNative.SlotAlloc(handle);
            Assert.True(GgmlGlmNative.SetActiveSlot(handle, spare));
            fail(1);
            Assert.False(GgmlGlmNative.Forward(handle, prompt, actual));
            fail(0);
            Assert.Equal(0, GgmlGlmNative.TakeReclaimedSlots(handle, ids));
            Assert.True(GgmlGlmNative.SetActiveSlot(handle, newer));
            Assert.True(GgmlGlmNative.SetActiveSlot(handle, running));
        }
        finally
        {
            fail?.Invoke(0);
            if (handle != IntPtr.Zero) GgmlGlmNative.Free(handle);
            if (library != IntPtr.Zero) NativeLibrary.Free(library);
            try { Directory.Delete(directory, recursive: true); } catch (IOException) { }
        }
    }
}
