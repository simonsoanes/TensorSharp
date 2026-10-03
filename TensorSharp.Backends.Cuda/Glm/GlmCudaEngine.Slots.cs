// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Sequence slots of the GLM-5.3-Flash direct-CUDA engine: one set of caches per sequence (the
// KDA convolution and recurrent states, the MLA latents, the indexer's keys and pooled keys),
// the same contract as the native executor's TSGgml_GlmSlot* exports, and a decode step that
// advances several slots at once.
//
// A KDA recurrence cannot be rewound, so a slot moves back only to position 0 (a reset) or
// stays where it is, as the native executor's glm5next slots do.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class GlmCudaEngine
    {
        /// <summary>Sequences one batched decode step can take.</summary>
        public const int MaxBatchedDecodeRows = 16;

        /// <summary>What every device keeps free beside the slots: the matmul scratch the shared ops
        /// grow on demand.</summary>
        private const long SlotReserveBytes = 1L << 30;

        private readonly Dictionary<int, Slot> _slots = new Dictionary<int, Slot>();
        private int _nextSlotId;

        /// <summary>Bytes of one device's per-row KDA state pointer table, [layer][conv, ssm][row].</summary>
        private long BatchStateTableBytes => (long)_m.NLayer * 2 * MaxBatchedDecodeRows * 8;

        private static IntPtr BatchStatePtr(Dev dev, int il, int kind)
            => (IntPtr)((long)Ptr(dev.BatchStatePtrs) + ((long)il * 2 + kind) * MaxBatchedDecodeRows * 8);

        /// <summary>The active slot's id.</summary>
        public int ActiveSlot => _active.Id;

        /// <summary>A new, empty slot, not made active; -1 when the devices have no room for it beside
        /// what the running sequences still need.</summary>
        public int SlotAlloc()
        {
            if (!SlotFits())
                return -1;
            Slot slot;
            try
            {
                slot = CreateSlot();
            }
            catch (OutOfMemoryException)
            {
                return -1;
            }
            _slots[slot.Id] = slot;
            return slot.Id;
        }

        /// <summary>Whether one more slot fits, counting what the allocators hold pooled.</summary>
        private bool SlotFits()
        {
            long[] need = _active.DeviceBytes;
            for (int d = 0; d < _devs.Length; d++)
            {
                if (need[d] == 0)
                    continue;
                _devs[d].MakeCurrent();
                (long free, _) = _devs[d].Alloc.GetMemoryInfo();
                free += _devs[d].Alloc.GetStats().CachedBytes;
                if (free - need[d] < SlotReserveBytes)
                    return false;
            }
            return true;
        }

        public bool SetActiveSlot(int id)
        {
            if (!_slots.TryGetValue(id, out Slot slot))
                return false;
            _active = slot;
            return true;
        }

        /// <summary>Free a slot; the active slot cannot be freed.</summary>
        public bool SlotFree(int id)
        {
            if (!_slots.TryGetValue(id, out Slot slot) || ReferenceEquals(slot, _active))
                return false;
            FreeSlot(slot);
            _slots.Remove(id);
            return true;
        }

        /// <summary>Head of a slot without selecting it; false when there is no such slot.</summary>
        public bool SlotHead(int id, out int head)
        {
            head = 0;
            if (!_slots.TryGetValue(id, out Slot slot))
                return false;
            head = slot.NPast;
            return true;
        }

        /// <summary><see cref="Reset"/> that reports whether the active slot is usable at position 0.</summary>
        public bool ResetChecked()
        {
            try
            {
                Reset();
                foreach (var dev in _devs)
                {
                    dev.MakeCurrent();
                    CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
                }
                return true;
            }
            catch (Exception ex) when (ex is InvalidOperationException || ex is CudaException)
            {
                return false;
            }
        }

        /// <summary>Move the active slot's head to <paramref name="nPast"/>: its own head, or 0 (the
        /// recurrent states cannot go back to anything in between).</summary>
        public bool Rewind(int nPast)
        {
            if (nPast == _active.NPast)
                return true;
            if (nPast != 0)
                return false;
            return ResetChecked();
        }

        /// <summary>
        /// One token for each of several distinct slots at their heads, every row computed as that
        /// sequence's own step computes it. False (nothing written) when the step cannot be batched:
        /// fewer than two or more than <see cref="MaxBatchedDecodeRows"/> slots, an unknown or
        /// repeated slot, a position that is not the slot's head, or a full context.
        /// </summary>
        public bool ForwardBatchedDecode(int[] slotIds, int[] tokens, int[] positions, float[] logits)
        {
            int n = slotIds?.Length ?? 0;
            if (n < 2 || n > MaxBatchedDecodeRows || tokens.Length != n || positions.Length != n
                || logits.Length < (long)n * _m.NVocab)
                return false;
            var rows = new Slot[n];
            var seen = new HashSet<int>();
            for (int i = 0; i < n; i++)
            {
                if (!seen.Add(slotIds[i]) || !_slots.TryGetValue(slotIds[i], out rows[i])
                    || rows[i].Failed || rows[i].NPast != positions[i] || positions[i] + 1 > _m.NCtx)
                    return false;
            }

            StageBatchStates(rows);
            foreach (var r in rows)
                r.Failed = true;
            ForwardUbatch(null, tokens, 0, n, 0, logits, rows, positions);
            foreach (var r in rows)
            {
                r.NPast++;
                r.Failed = false;
            }
            return true;
        }

        /// <summary>Each device's per-row KDA state pointers for the layers it holds.</summary>
        private void StageBatchStates(Slot[] rows)
        {
            int n = rows.Length;
            foreach (var dev in _devs)
            {
                if (dev.BatchStatePtrs == null)
                    continue;
                long* table = (long*)dev.BatchStatePinned;
                for (int il = 0; il < _m.NLayer; il++)
                {
                    if (_layers[il].Device != dev.Ordinal || !_layers[il].Recurrent)
                        continue;
                    for (int r = 0; r < n; r++)
                    {
                        table[((long)il * 2 + 0) * MaxBatchedDecodeRows + r] = (long)Ptr(rows[r].Layers[il].ConvState);
                        table[((long)il * 2 + 1) * MaxBatchedDecodeRows + r] = (long)Ptr(rows[r].Layers[il].Ssm);
                    }
                }
                dev.MakeCurrent();
                // The previous step's copy out of the pinned table finished with that step's logits.
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.BatchStatePtrs), dev.BatchStatePinned,
                    new UIntPtr((ulong)BatchStateTableBytes), dev.Stream).ThrowOnError();
            }
        }

        /// <summary>The MLA layer for a batched decode step: the projections run for every row at
        /// once, each row's cache writes and attention against its own slot at its own position.</summary>
        private void MlaRows(Dev dev, DevLayer L, int il, Slot[] rows, int[] positions)
        {
            var m = _m;
            int n = rows.Length, h = m.NHead, lat = m.KvLoraRank, d = m.IdxHeadDim, kpool = m.IdxKpool;
            IntPtr cur = QuantizeShared(dev, dev.Cur, m.NEmbd, n);
            MatMul(dev, L.WqA, dev.Cur, cur, dev.Qr, n);
            MatMul(dev, L.WkvA, dev.Cur, cur, dev.KvRaw, n);
            MatMul(dev, L.IdxK, dev.Cur, cur, dev.IdxKey, n);
            MatMul(dev, L.IdxGate, dev.Cur, cur, dev.IdxGateOut, n);
            RmsNorm(dev.Qr, L.QANorm, n);
            IntPtr qr = QuantizeShared(dev, dev.Qr, m.QLoraRank, n);
            MatMul(dev, L.WqB, dev.Qr, qr, dev.Q, n);
            bool anySparse = false;
            for (int r = 0; r < n; r++)
            {
                SlotLayer c = rows[r].Layers[il];
                int p = positions[r];
                dev.GK.MlaKvStore(At(dev.KvRaw, (long)r * lat), Ptr(L.KvANorm), Ptr(c.KvCache), p, 1, lat, m.RmsEps, dev.Stream);
                dev.GK.IndexerStore(At(dev.IdxKey, (long)r * d), At(dev.IdxGateOut, (long)r * d),
                    Ptr(L.IdxKNormW), Ptr(L.IdxKNormB), Ptr(c.IdxCache), p, 1, d, m.NormEps, dev.Stream);
                if ((p + 1) % kpool == 0)
                    dev.GK.PoolKeys(Ptr(c.IdxCache), Ptr(L.IdxApe), Ptr(c.PoolKeys), p / kpool, 1, d, dev.Stream);
                anySparse |= p + 1 > m.IdxTopK;
            }

            HeadProject(dev, L.WkBF16, dev.Q, dev.QF16, dev.QAbs, lat, m.HeadDimK, n);
            if (anySparse)
            {
                MatMul(dev, L.IdxQB, dev.Qr, qr, dev.IdxQ, n);
                MatMulF32(dev, L.IdxProj, dev.Cur, dev.IdxW, m.NEmbd, m.IdxNHead, n);
            }

            float kqScale = 1.0f / MathF.Sqrt(m.HeadDimK);
            int pools = m.NCtx / kpool;
            for (int r = 0; r < n; r++)
            {
                SlotLayer c = rows[r].Layers[il];
                int p = positions[r];
                using Tensor q = Block(dev.QAbs, r, 1, (long)h * lat);
                using Tensor o = Block(dev.AttnO, r, 1, (long)h * lat);
                if (p + 1 > m.IdxTopK)
                {
                    using Tensor scores = Block(dev.PoolScores, r, 1, pools);
                    using Tensor sel = Block(dev.TopkIdx, r, 1, SelectPools);
                    using Tensor selCnt = Block(dev.TopkCnt, r, 1, 1);
                    using Tensor cells = Block(dev.Cells, r, 1, CellStride);
                    using Tensor cellCnt = Block(dev.CellCnt, r, 1, 1);
                    dev.GK.PoolScores(At(dev.IdxQ, (long)r * m.IdxNHead * d), At(dev.IdxW, (long)r * m.IdxNHead),
                        Ptr(c.PoolKeys), Ptr(scores), p, m.IdxNHead, d, pools, (p + 1) / kpool, 1, dev.Stream);
                    dev.DK.TopK(scores, sel, selCnt, p, kpool, SelectPools, pools, 1, dev.Stream);
                    dev.GK.ExpandCells(Ptr(sel), Ptr(selCnt), Ptr(cells), Ptr(cellCnt), p, SelectPools, CellStride, 1, dev.Stream);
                    dev.DK.Attention(q, Ptr(c.KvCache), Ptr(c.KvCache), cells, cellCnt, Ptr(dev.NegInf), o,
                        p, m.NCtx, m.NCtx, h, lat, 4, 1, CellStride, kqScale, 1, dev.Stream);
                }
                else
                {
                    dev.DK.Attention(q, Ptr(c.KvCache), IntPtr.Zero, null, null, Ptr(dev.NegInf), o,
                        p, m.NCtx, m.NCtx, h, lat, 0, 1, 0, kqScale, 1, dev.Stream);
                }
            }

            HeadProject(dev, L.WvBF16, dev.AttnO, dev.AttnOF16, dev.VOut, m.HeadDimV, lat, n);
            MatMulQ(dev, L.Wo, dev.VOut, dev.AttnOut, n);
        }

        // ---------------------------------------------------------------- vision rows

        // Projected image rows queued for the next Forward: (rows, count, first token index).
        private readonly List<(float[] Rows, int Count, int Index)> _vision = new List<(float[], int, int)>();
        private Tensor _visionRows;
        private IntPtr _visionPinned;

        /// <summary>Queue projected vision rows to replace the embeddings of the next Forward's tokens
        /// from <paramref name="index"/> on (the image placeholders); the forward consumes the queue.</summary>
        public bool QueueVisionRows(float[] rows, int nRows, int index)
        {
            if (rows == null || nRows <= 0 || index < 0 || rows.Length < (long)nRows * _m.NEmbd)
                return false;
            var copy = new float[(long)nRows * _m.NEmbd];
            Array.Copy(rows, copy, copy.Length);
            _vision.Add((copy, nRows, index));
            return true;
        }

        public void ClearVisionRows() => _vision.Clear();

        /// <summary>Apply the queued rows that fall in a ubatch covering tokens [first, first + nt)
        /// of the Forward call, after its embedding lookup.</summary>
        private void ApplyVisionRows(int first, int nt)
        {
            if (_vision.Count == 0)
                return;
            var dev = _devs[0];
            int e = _m.NEmbd;
            if (_visionRows == null)
            {
                _visionRows = AllocF32(dev, _m.NUbatch, e);
                CudaDriverApi.cuMemHostAlloc(out _visionPinned, new UIntPtr((ulong)_m.NUbatch * (ulong)e * 4), 0x1).ThrowOnError();
            }
            foreach (var (rows, count, index) in _vision)
            {
                int lo = Math.Max(index, first), hi = Math.Min(index + count, first + nt);
                if (lo >= hi)
                    continue;
                int n = hi - lo;
                // The staging is reused per image: the previous copy out of it must have run.
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
                Marshal.Copy(rows, (lo - index) * e, _visionPinned, n * e);
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(_visionRows), _visionPinned, new UIntPtr((ulong)n * (ulong)e * 4), dev.Stream).ThrowOnError();
                dev.GK.SetStreamRows(Ptr(_visionRows), Ptr(dev.Xs), lo - first, n, e, dev.Stream);
            }
        }

        /// <summary>Address of element <paramref name="offset"/> of an F32 tensor.</summary>
        private static IntPtr At(Tensor t, long offset) => (IntPtr)((long)Ptr(t) + offset * 4);
    }
}
