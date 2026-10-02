// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Sequence slots of a DeepSeek executor. The native ggml executor (the TSGgml_Dsv4Slot*
    /// exports) and the direct-CUDA engine (Dsv4CudaEngine.Slots.cs) implement the same
    /// contract, so per-request caches, retained conversations, truncation and batched decode
    /// behave the same on either. Forward, reset and truncation act on the ACTIVE slot.
    /// </summary>
    internal interface IDsv4SlotExecutor
    {
        /// <summary>A new, empty slot, not made active; -1 when the devices are full.</summary>
        int SlotAlloc();

        /// <summary>Inspect a slot without selecting it. False for a load that cannot keep a slot
        /// between requests (V4, or a loaded drafter).</summary>
        bool SlotStatus(int slot, out int head, out int checkpoint, out bool healthy);

        bool SlotCanReuse(int slot, int cachedHead, int target);

        bool SlotCanRetain(int slot, int retainedCount, ulong budgetPerDevice);

        /// <summary>Whether one more slot fits beside what the running requests still need.</summary>
        bool SlotCanAlloc();

        /// <summary>Drop what the executor keeps per slot beyond its caches (captured graphs).</summary>
        bool SlotReleaseGraphs(int slot);

        bool SetActiveSlot(int slot);

        /// <summary>Free a slot; the active slot cannot be freed.</summary>
        bool SlotFree(int slot);

        /// <summary>Reset the active slot; true when it is usable at position 0.</summary>
        bool ResetChecked();

        /// <summary>Move the active slot's head back; false when that cannot be exact.</summary>
        bool Truncate(int nPast);

        /// <summary>One token for each of several distinct slots at their heads; false when the
        /// step cannot be batched (the caller runs the slots one at a time).</summary>
        bool ForwardBatchedDecode(int[] slots, int[] tokens, int[] positions, float[] logits);
    }

    /// <summary>The native ggml executor's slots.</summary>
    internal sealed class NativeDsv4Slots : IDsv4SlotExecutor
    {
        private readonly IntPtr _handle;

        public NativeDsv4Slots(IntPtr handle) => _handle = handle;

        public int SlotAlloc() => GgmlDeepSeek4Native.SlotAlloc(_handle);

        public bool SlotStatus(int slot, out int head, out int checkpoint, out bool healthy)
            => GgmlDeepSeek4Native.SlotStatus(_handle, slot, out head, out checkpoint, out healthy);

        public bool SlotCanReuse(int slot, int cachedHead, int target)
            => GgmlDeepSeek4Native.SlotCanReuse(_handle, slot, cachedHead, target);

        public bool SlotCanRetain(int slot, int retainedCount, ulong budgetPerDevice)
            => GgmlDeepSeek4Native.SlotCanRetain(_handle, slot, retainedCount, budgetPerDevice);

        public bool SlotCanAlloc() => GgmlDeepSeek4Native.SlotCanAlloc(_handle);

        public bool SlotReleaseGraphs(int slot) => GgmlDeepSeek4Native.SlotReleaseGraphs(_handle, slot);

        public bool SetActiveSlot(int slot) => GgmlDeepSeek4Native.SetActiveSlot(_handle, slot);

        public bool SlotFree(int slot) => GgmlDeepSeek4Native.SlotFree(_handle, slot);

        public bool ResetChecked() => GgmlDeepSeek4Native.ResetChecked(_handle);

        public bool Truncate(int nPast) => GgmlDeepSeek4Native.Truncate(_handle, nPast);

        public bool ForwardBatchedDecode(int[] slots, int[] tokens, int[] positions, float[] logits)
            => GgmlDeepSeek4Native.ForwardBatchedDecode(_handle, slots, tokens, positions, logits);
    }
}
