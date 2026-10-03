// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Speculative verification on the GLM-5.3-Flash engine, the contract the native executor
// honours (GlmDsaModel.Glm5NextSpeculative.cs): before a verify window the model snapshots every
// KDA recurrent state; the window advances state and head by all its rows; on a partial
// rejection the snapshot comes back, the head returns to where it was taken, and the model
// re-forwards the accepted prefix. MLA rows, indexer keys and pool keys are per position and
// are rewritten by that re-forward before anything reads them, so the KDA state is the only
// thing a rollback restores. The checkpoint's NextN block is not built, so what this serves is
// the weight-free n-gram drafter.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class GlmCudaEngine
    {
        /// <summary>Rows a speculative verify may carry: a drafter's 64 tokens and the pending one.</summary>
        public const int MaxSpecRows = 65;

        /// <summary>Rows the head computes logits for at once: a batched decode step or a verify window.</summary>
        private static int MaxHeadRows => Math.Max(MaxBatchedDecodeRows, MaxSpecRows);

        // The KDA recurrent snapshot: one copy per recurrent layer, on the layer's own device.
        private Tensor[] _snapConv, _snapSsm;
        private int _snapSlot = -1, _snapPos = -1;

        /// <summary>Copy the active slot's KDA states aside and remember its head.</summary>
        public bool KdaStateCapture()
        {
            Slot slot = _active;
            if (slot.Failed)
                return false;
            _snapConv ??= new Tensor[_m.NLayer];
            _snapSsm ??= new Tensor[_m.NLayer];
            for (int il = 0; il < _m.NLayer; il++)
            {
                SlotLayer c = slot.Layers[il];
                if (c.ConvState == null)
                    continue;
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                _snapConv[il] ??= AllocF32(dev, c.ConvState.ElementCount());
                _snapSsm[il] ??= AllocF32(dev, c.Ssm.ElementCount());
                Copy(dev, c.ConvState, _snapConv[il]);
                Copy(dev, c.Ssm, _snapSsm[il]);
            }
            _snapSlot = slot.Id;
            _snapPos = slot.NPast;
            return true;
        }

        /// <summary>Put the snapshot back into the slot it came from and park its head where the
        /// snapshot was taken; that head, or -1 when there is no snapshot of the active slot.</summary>
        public int KdaStateRestore()
        {
            Slot slot = _active;
            if (_snapPos < 0 || _snapSlot != slot.Id || _snapPos > slot.NPast)
                return -1;
            for (int il = 0; il < _m.NLayer; il++)
            {
                SlotLayer c = slot.Layers[il];
                if (c.ConvState == null)
                    continue;
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                Copy(dev, _snapConv[il], c.ConvState);
                Copy(dev, _snapSsm[il], c.Ssm);
            }
            slot.NPast = _snapPos;
            slot.Failed = false;
            return _snapPos;
        }

        /// <summary>Forget the snapshot (its slot was reset, rewound or freed).</summary>
        private void InvalidateSnapshot(Slot slot)
        {
            if (slot != null && slot.Id == _snapSlot)
            {
                _snapSlot = -1;
                _snapPos = -1;
            }
        }

        /// <summary>
        /// A verify window: the tokens at the active slot's head, advancing it by all of them.
        /// <paramref name="hOut"/> gets every row's normed final hidden state [n, NEmbd]; the logits
        /// are every row's when <paramref name="allRows"/>, else the last row's.
        /// </summary>
        public void SpecForward(int[] tokens, float[] hOut, float[] logitsOut, bool allRows)
        {
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("empty token batch", nameof(tokens));
            if (tokens.Length > MaxSpecRows)
                throw new ArgumentException($"a verify window carries at most {MaxSpecRows} rows", nameof(tokens));
            Slot slot = _active;
            if (slot.Failed)
                throw new InvalidOperationException("[glm-cuda] the sequence's last forward failed; reset it before reuse");
            if (slot.NPast + tokens.Length > _m.NCtx)
                throw new InvalidOperationException($"[glm-cuda] context overflow: n_past={slot.NPast} + {tokens.Length} > n_ctx={_m.NCtx}");

            slot.Failed = true;
            ForwardUbatch(slot, tokens, 0, tokens.Length, slot.NPast, logitsOut, allRows: allRows, hOut: hOut);
            slot.NPast += tokens.Length;
            slot.Failed = false;
        }

        private static void Copy(Dev dev, Tensor src, Tensor dst)
            => CudaDriverApi.cuMemcpyDtoDAsync(Ptr(dst), Ptr(src), new UIntPtr((ulong)src.ElementCount() * 4), dev.Stream)
                .ThrowOnError();
    }
}
