// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The GLM-5.3-Flash direct-CUDA forward: scratch, one sequence's caches, and the per-layer
// pipeline (see GlmCudaEngine.cs for the model and its references).
using System;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class GlmCudaEngine
    {
        private void AllocateScratch(Dev dev)
        {
            dev.MakeCurrent();
            var m = _m;
            int nt = m.NUbatch, e = m.NEmbd, h = m.NHead;
            long dInner = (long)h * m.KdaHeadDim;
            bool anyKda = false, anyMla = false, anyMoe = false, anyDense = false;
            int kdaLow = 1, ff = 1;
            for (int il = 0; il < m.NLayer; il++)
            {
                if (_layers[il].Device != dev.Ordinal)
                    continue;
                var L = m.Layers[il];
                anyKda |= L.Recurrent;
                anyMla |= !L.Recurrent;
                anyMoe |= L.Moe;
                anyDense |= !L.Moe;
                if (L.Recurrent)
                    kdaLow = Math.Max(kdaLow, Math.Max(L.KdaFA.Ne1, L.KdaGA.Ne1));
                ff = Math.Max(ff, L.Moe ? m.NFfShexp : m.NFf);
            }

            dev.Xs = AllocF32(dev, nt, HC * e);
            dev.XsOut = AllocF32(dev, nt, HC * e);
            dev.Cur = AllocF32(dev, nt, e);
            dev.SharedQ81 = AllocT(dev, DType.UInt8, PerSlotMaxRows, (long)(Math.Max(m.NEmbd, m.QLoraRank) / 32) * 36);
            // MatMulQ's inputs: attention and KDA outputs, FFN hiddens, the low rank, the head's input.
            int innerIn = Math.Max(Math.Max(h * m.HeadDimV, (int)dInner), Math.Max(Math.Max(m.NFf, m.NFfShexp), Math.Max(kdaLow, e)));
            dev.InnerQ81 = AllocT(dev, DType.UInt8, PerSlotMaxRows, (long)((innerIn + 31) / 32) * 36);
            dev.PosDev = AllocT(dev, DType.Int32, 1);
            dev.HcPartials = AllocF32(dev, nt, HC * e / Dsv4Kernels.HcSlice * Dsv4Kernels.HcPartialFloats);
            dev.Inv = AllocF32(dev, nt);
            dev.Mixes = AllocF32(dev, nt, HcMixDim);
            dev.Pre = AllocF32(dev, nt, HC);
            dev.Post = AllocF32(dev, nt, HC);
            dev.Comb = AllocF32(dev, nt, HC * HC);
            dev.AttnOut = AllocF32(dev, nt, e);
            dev.FfnOut = AllocF32(dev, nt, e);
            if (anyKda)
            {
                dev.KQ = AllocF32(dev, nt, dInner);
                dev.KK = AllocF32(dev, nt, dInner);
                dev.KV = AllocF32(dev, nt, dInner);
                dev.KF = AllocF32(dev, nt, dInner);
                dev.KG = AllocF32(dev, nt, dInner);
                dev.KLow = AllocF32(dev, nt, kdaLow);
                dev.KBeta = AllocF32(dev, nt, h);
                dev.KScr = AllocF32(dev, (long)nt * h * GlmKernels.KdaScratch);
                dev.KCore = AllocF32(dev, nt, dInner);
                dev.KOutIn = AllocF32(dev, nt, dInner);
            }
            if (anyMla)
            {
                dev.Qr = AllocF32(dev, nt, m.QLoraRank);
                dev.Q = AllocF32(dev, nt, (long)h * m.HeadDimK);
                dev.QF16 = AllocT(dev, DType.Float16, (long)nt * h * m.HeadDimK);
                dev.KvRaw = AllocF32(dev, nt, m.KvLoraRank);
                dev.IdxKey = AllocF32(dev, nt, m.IdxHeadDim);
                dev.IdxGateOut = AllocF32(dev, nt, m.IdxHeadDim);
                dev.QAbs = AllocF32(dev, nt, (long)h * m.KvLoraRank);
                dev.AttnO = AllocF32(dev, nt, (long)h * m.KvLoraRank);
                dev.AttnOF16 = AllocT(dev, DType.Float16, (long)nt * h * m.KvLoraRank);
                dev.VOut = AllocF32(dev, nt, (long)h * m.HeadDimV);
                int pools = m.NCtx / m.IdxKpool, selK = SelectPools;
                dev.IdxQ = AllocF32(dev, nt, (long)m.IdxNHead * m.IdxHeadDim);
                dev.IdxW = AllocF32(dev, nt, m.IdxNHead);
                dev.PoolScores = AllocF32(dev, nt, pools);
                dev.TopkIdx = AllocT(dev, DType.Int32, nt, selK);
                dev.TopkCnt = AllocT(dev, DType.Int32, nt);
                dev.Cells = AllocT(dev, DType.Int32, nt, CellStride);
                dev.CellCnt = AllocT(dev, DType.Int32, nt);
                var negInf = new float[h];
                Array.Fill(negInf, float.NegativeInfinity);
                dev.NegInf = UploadF32(dev, negInf);
            }
            if (anyMoe)
            {
                dev.RouterLogits = AllocF32(dev, nt, m.NExpert);
                dev.GemvScratch = AllocF32(dev, Dsv4Kernels.GemvScratchFloats);
                dev.Moe = new CudaMoeScratch((type, sizes) => AllocT(dev, type, sizes),
                    nt, m.NExpert, m.NExpertUsed, e, m.NFfExp);
            }
            if (anyMoe || anyDense)
            {
                dev.ShGate = AllocF32(dev, nt, ff);
                dev.ShUp = AllocF32(dev, nt, ff);
                dev.ShDown = AllocF32(dev, nt, e);
            }
            if (dev.Ordinal == 0)
            {
                dev.Tokens = AllocT(dev, DType.Int32, nt);
                // PORTABLE: every device may read them.
                CudaDriverApi.cuMemHostAlloc(out _pinnedTokens, new UIntPtr((ulong)nt * 4), 0x1).ThrowOnError();
                CudaDriverApi.cuMemHostAlloc(out _posPinned, new UIntPtr(4), 0x1).ThrowOnError();
            }
            if (dev.Ordinal == _lastDev)
            {
                dev.Logits = AllocF32(dev, 1, m.NVocab);
                dev.BatchLogits = AllocF32(dev, MaxHeadRows, m.NVocab);
                CudaDriverApi.cuMemHostAlloc(out _pinnedLogits,
                    new UIntPtr((ulong)MaxHeadRows * (ulong)m.NVocab * 4), 0x1).ThrowOnError();
                CudaDriverApi.cuMemHostAlloc(out _pinnedHidden,
                    new UIntPtr((ulong)MaxHeadRows * (ulong)m.NEmbd * 4), 0x1).ThrowOnError();
            }
            if (anyKda)
            {
                long tableBytes = BatchStateTableBytes;
                dev.BatchStatePtrs = AllocT(dev, DType.UInt8, tableBytes);
                CudaDriverApi.cuMemHostAlloc(out dev.BatchStatePinned, new UIntPtr((ulong)tableBytes), 0x1).ThrowOnError();
            }
            CudaDriverApi.cuMemHostAlloc(out dev.BoundaryPinned, new UIntPtr((ulong)((long)nt * HC * e * 4)), 0x1).ThrowOnError();
            CudaDriverApi.cuEventCreate(out dev.XsReadyEv, 0x02 /*DISABLE_TIMING*/).ThrowOnError();
            CudaDriverApi.cuEventCreate(out dev.CopyDoneEv, 0x02).ThrowOnError();
        }

        // -------------------------------------------------------------------
        // Sequence caches
        // -------------------------------------------------------------------

        private Slot CreateSlot()
        {
            var m = _m;
            var slot = new Slot
            {
                Id = _nextSlotId++,
                Layers = new SlotLayer[m.NLayer],
                StatePtrs = new Tensor[_devs.Length],
                DeviceBytes = new long[_devs.Length],
            };
            long dInner = (long)m.NHead * m.KdaHeadDim;
            var ptrs = new long[_devs.Length][];
            for (int d = 0; d < _devs.Length; d++)
                ptrs[d] = new long[2 * m.NLayer];
            for (int il = 0; il < m.NLayer; il++)
            {
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                var c = new SlotLayer();
                if (m.Layers[il].Recurrent)
                {
                    c.ConvState = SlotTensor(slot, dev, DType.Float32, (m.DConv - 1) * 3 * dInner);
                    c.Ssm = SlotTensor(slot, dev, DType.Float32, (long)m.NHead * m.KdaHeadDim * m.KdaHeadDim);
                    ptrs[dev.Ordinal][2 * il] = (long)Ptr(c.ConvState);
                    ptrs[dev.Ordinal][2 * il + 1] = (long)Ptr(c.Ssm);
                }
                else
                {
                    c.KvCache = SlotTensor(slot, dev, DType.Float16, (long)m.NCtx * m.KvLoraRank);
                    c.IdxCache = SlotTensor(slot, dev, DType.Float16, (long)m.NCtx * 2 * m.IdxHeadDim);
                    c.PoolKeys = SlotTensor(slot, dev, DType.Float32, (long)m.NCtx / m.IdxKpool * m.IdxHeadDim);
                }
                slot.Layers[il] = c;
            }
            for (int d = 0; d < _devs.Length; d++)
            {
                var dev = _devs[d];
                dev.MakeCurrent();
                slot.StatePtrs[d] = SlotTensor(slot, dev, DType.UInt8, 16L * m.NLayer);
                fixed (long* p = ptrs[d])
                    CudaDriverApi.cuMemcpyHtoD(Ptr(slot.StatePtrs[d]), (IntPtr)p, new UIntPtr((ulong)(16L * m.NLayer))).ThrowOnError();
            }
            ClearSlot(slot);
            return slot;
        }

        private static Tensor SlotTensor(Slot slot, Dev dev, DType type, long elements)
        {
            dev.MakeCurrent();
            var t = new Tensor(dev.Alloc, type, elements);
            slot.Owned.Add(t);
            slot.DeviceBytes[dev.Ordinal] += elements * type.Size();
            return t;
        }

        /// <summary>Zero the recurrent states (an attention cache needs no clearing: nothing past
        /// the head is ever read).</summary>
        private void ClearSlot(Slot slot)
        {
            for (int il = 0; il < _m.NLayer; il++)
            {
                var c = slot.Layers[il];
                if (c.ConvState == null)
                    continue;
                var dev = _devs[_layers[il].Device];
                dev.MakeCurrent();
                CudaDriverApi.cuMemsetD8Async(Ptr(c.ConvState), 0, new UIntPtr((ulong)c.ConvState.ElementCount() * 4), dev.Stream).ThrowOnError();
                CudaDriverApi.cuMemsetD8Async(Ptr(c.Ssm), 0, new UIntPtr((ulong)c.Ssm.ElementCount() * 4), dev.Stream).ThrowOnError();
            }
            slot.NPast = 0;
            slot.Failed = false;
            InvalidateSnapshot(slot);
        }

        private void FreeSlot(Slot slot)
        {
            if (slot == null)
                return;
            InvalidateSnapshot(slot);
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream);
            }
            DestroySlotGraphs(slot.Id);
            foreach (var t in slot.Owned)
                t.Dispose();
            slot.Owned.Clear();
        }

        /// <summary>Back to position 0 on the active sequence.</summary>
        public void Reset() => ClearSlot(_active);

        /// <summary>Address of this slot's [conv, ssm] state pointer for a KDA layer on
        /// <paramref name="dev"/>: a one-sequence pointer array for the KDA kernels.</summary>
        private static IntPtr StatePtr(Slot slot, Dev dev, int il, int kind)
            => (IntPtr)((long)Ptr(slot.StatePtrs[dev.Ordinal]) + (2L * il + kind) * 8);

        // -------------------------------------------------------------------
        // Forward
        // -------------------------------------------------------------------

        public void Forward(int[] tokens, float[] logitsOut)
        {
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("empty token batch", nameof(tokens));
            Slot slot = _active;
            if (slot.Failed)
                throw new InvalidOperationException("[glm-cuda] the sequence's last forward failed; reset it before reuse");
            if (slot.NPast + tokens.Length > _m.NCtx)
                throw new InvalidOperationException($"[glm-cuda] context overflow: n_past={slot.NPast} + {tokens.Length} > n_ctx={_m.NCtx}");

            var sw = System.Diagnostics.Stopwatch.StartNew();
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            slot.Failed = true;
            try
            {
                if (tokens.Length == 1 && logitsOut != null && _vision.Count == 0 && CanCaptureDecode
                    && ForwardDecodeGraphed(slot, tokens[0], slot.NPast, logitsOut))
                {
                    slot.NPast++;
                }
                else
                {
                    int done = 0;
                    while (done < tokens.Length)
                    {
                        int nt = Math.Min(_m.NUbatch, tokens.Length - done);
                        bool last = done + nt == tokens.Length;
                        ForwardUbatch(slot, tokens, done, nt, slot.NPast, last ? logitsOut : null);
                        slot.NPast += nt;
                        done += nt;
                    }
                }
            }
            finally
            {
                _vision.Clear();   // queued rows belong to this forward alone
            }
            slot.Failed = false;
            if (_perf > 0)
            {
                double secs = sw.Elapsed.TotalSeconds;
                Console.Error.WriteLine($"[glm-cuda] forward {tokens.Length} tokens in {secs:F3}s ({tokens.Length / secs:F1} tok/s), "
                    + $"host issue {IssueMs(start):F1}ms");
            }
        }

        /// <param name="rows">A batched decode step: one row per sequence (<paramref name="slot"/> is
        /// then unused), each at <paramref name="positions"/>[row]; null for one sequence's ubatch at
        /// <paramref name="p0"/>.</param>
        private void ForwardUbatch(Slot slot, int[] tokens, int tokOff, int nt, int p0, float[] logitsOut,
            Slot[] rows = null, int[] positions = null, bool allRows = false, float[] hOut = null)
        {
            var m = _m;
            int e = m.NEmbd;
            var dev0 = _devs[0];
            dev0.MakeCurrent();
            // The pinned tokens are rewritten per ubatch: the previous upload must have run.
            CudaDriverApi.cuStreamSynchronize(dev0.Stream).ThrowOnError();
            Marshal.Copy(tokens, tokOff, _pinnedTokens, nt);
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev0.Tokens), _pinnedTokens, new UIntPtr((ulong)nt * 4), dev0.Stream).ThrowOnError();
            StageBegin(dev0);
            dev0.DK.Embed(_tokEmbd.Ptr, dev0.Tokens, dev0.Xs, _tokEmbd.Type, _tokEmbd.RowBytes, nt, e, dev0.Stream);
            if (rows == null)
                ApplyVisionRows(tokOff, nt);
            StageEnd(dev0, StEmbed);

            int curDev = 0;
            for (int il = 0; il < m.NLayer; il++)
            {
                var L = _layers[il];
                if (L.Device != curDev)
                {
                    Handoff(_devs[curDev], _devs[L.Device], nt);
                    curDev = L.Device;
                    StageEnd(_devs[curDev], StBoundary);
                }
                var dev = _devs[curDev];
                dev.MakeCurrent();
                RunLayer(dev, il, slot, nt, p0, rows, positions, IntPtr.Zero);
            }

            if (logitsOut != null || hOut != null)
                Head(nt, logitsOut, rows != null || allRows, hOut);
        }

        /// <summary>One layer's two mixed halves. <paramref name="pos"/> is the device copy of the row's
        /// position a captured decode step reads (zero: p0 is used). Each half swaps the stream buffers,
        /// so a layer leaves them where it found them.</summary>
        private void RunLayer(Dev dev, int il, Slot slot, int nt, int p0, Slot[] rows, int[] positions, IntPtr pos)
        {
            var L = _layers[il];
            int e = _m.NEmbd;
            HcPre(dev, L.HcAttnFn, L.HcAttnScale, L.HcAttnBase, L.AttnNorm, nt);
            StageEnd(dev, StHc);
            if (L.Recurrent)
                Kda(dev, L, il, slot, nt, rows);
            else if (rows != null)
                MlaRows(dev, L, il, rows, positions);
            else
                Mla(dev, L, il, slot.Layers[il], nt, p0, pos);
            dev.DK.HcPost(dev.Xs, dev.AttnOut, dev.Post, dev.Comb, dev.XsOut, nt, e, dev.Stream);
            (dev.Xs, dev.XsOut) = (dev.XsOut, dev.Xs);

            HcPre(dev, L.HcFfnFn, L.HcFfnScale, L.HcFfnBase, L.FfnNorm, nt);
            StageEnd(dev, StHc);
            if (L.Moe)
                Moe(dev, L, nt);
            else
                DenseFfn(dev, L, nt);
            dev.DK.HcPost(dev.Xs, dev.FfnOut, dev.Post, dev.Comb, dev.XsOut, nt, e, dev.Stream);
            (dev.Xs, dev.XsOut) = (dev.XsOut, dev.Xs);
            StageEnd(dev, StHc);
        }

        /// <summary>Hand the hidden streams to the next device through pinned host memory. Events are
        /// recorded on their own context's stream; only the waits cross contexts.</summary>
        private void Handoff(Dev src, Dev dst, int nt)
        {
            var bytes = new UIntPtr((ulong)((long)nt * HC * _m.NEmbd * 4));
            src.MakeCurrent();
            CudaDriverApi.cuMemcpyDtoHAsync(src.BoundaryPinned, Ptr(src.Xs), bytes, src.Stream).ThrowOnError();
            StageEnd(src, StBoundary);
            CudaDriverApi.cuEventRecord(src.XsReadyEv, src.Stream).ThrowOnError();
            dst.MakeCurrent();
            CudaDriverApi.cuStreamWaitEvent(dst.Stream, src.XsReadyEv, 0).ThrowOnError();
            StageBegin(dst);
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dst.Xs), src.BoundaryPinned, bytes, dst.Stream).ThrowOnError();
            CudaDriverApi.cuEventRecord(dst.CopyDoneEv, dst.Stream).ThrowOnError();
            // Nothing may overwrite the source streams or the pinned slice before the copy ran.
            src.MakeCurrent();
            CudaDriverApi.cuStreamWaitEvent(src.Stream, dst.CopyDoneEv, 0).ThrowOnError();
        }

        /// <summary>Read the streams down to one row per token (into Cur, RMS normed with
        /// <paramref name="normW"/>) and derive the post and comb weights the matching HcPost writes
        /// back with: two kernels, each row computed alike whatever the ubatch.</summary>
        private void HcPre(Dev dev, in DeviceWeight fn, Tensor scale, Tensor baseW, Tensor normW, int nt)
        {
            var m = _m;
            int flat = HC * m.NEmbd;
            if ((fn.Type == TQ8_0 || fn.Type == TF32) && flat % Dsv4Kernels.HcSlice == 0)
            {
                dev.DK.HcMixPartials(dev.Xs, fn.Ptr, fn.Type, dev.HcPartials, flat, nt, dev.Stream);
                dev.DK.HcPreFinish(dev.HcPartials, flat / Dsv4Kernels.HcSlice, dev.Xs, Ptr(scale), Ptr(baseW), Ptr(normW),
                    dev.Pre, dev.Post, dev.Comb, null, false, null, dev.Cur, m.NEmbd, nt, m.HcSinkhornIters, m.HcEps,
                    m.RmsEps, dev.Stream);
                return;
            }
            dev.DK.HcRms(dev.Xs, dev.Inv, nt, flat, m.RmsEps, dev.Stream);
            // The mix projection reads the un-normalized streams and HcGatesComb scales by the
            // inverse RMS: the same function as projecting the normalized streams.
            MatMul(fn, dev.Xs, dev.Mixes, nt);
            dev.DK.HcGatesComb(dev.Mixes, dev.Inv, Ptr(scale), Ptr(baseW), dev.Pre, dev.Post, dev.Comb,
                nt, m.HcSinkhornIters, m.HcEps, dev.Stream);
            dev.DK.HcCollapse(dev.Xs, dev.Pre, dev.Cur, nt, m.NEmbd, dev.Stream);
            RmsNorm(dev.Cur, normW, nt);
        }

        private void Kda(Dev dev, DevLayer L, int il, Slot slot, int nt, Slot[] rows)
        {
            var m = _m;
            int h = m.NHead, dInner = h * m.KdaHeadDim;
            IntPtr cur = QuantizeShared(dev, dev.Cur, m.NEmbd, nt);
            MatMul(dev, L.KdaQ, dev.Cur, cur, dev.KQ, nt);
            MatMul(dev, L.KdaK, dev.Cur, cur, dev.KK, nt);
            MatMul(dev, L.KdaV, dev.Cur, cur, dev.KV, nt);
            // The forget gate, the output gate and beta all read the layer input, not the
            // convolved q/k/v.
            MatMul(dev, L.KdaFA, dev.Cur, cur, dev.KLow, nt);
            MatMulQ(dev, L.KdaFB, dev.KLow, dev.KF, nt);
            MatMul(dev, L.KdaGA, dev.Cur, cur, dev.KLow, nt);
            MatMulQ(dev, L.KdaGB, dev.KLow, dev.KG, nt);
            MatMul(dev, L.KdaBeta, dev.Cur, cur, dev.KBeta, nt);
            StageEnd(dev, StKdaProj);

            // One sequence's nt rows in its own state, or one row of each batched sequence in theirs.
            IntPtr conv, ssm;
            int rowsPerSeq = nt, nSeq = 1;
            if (rows == null)
            {
                conv = StatePtr(slot, dev, il, 0);
                ssm = StatePtr(slot, dev, il, 1);
            }
            else
            {
                conv = BatchStatePtr(dev, il, 0);
                ssm = BatchStatePtr(dev, il, 1);
                rowsPerSeq = 1;
                nSeq = nt;
            }
            dev.GK.KdaPrep(Ptr(dev.KQ), Ptr(dev.KK), Ptr(dev.KV), conv, Ptr(L.ConvQ), Ptr(L.ConvK), Ptr(L.ConvV),
                Ptr(dev.KF), Ptr(L.DtBias), Ptr(L.SsmA), Ptr(dev.KBeta), Ptr(dev.KScr),
                rowsPerSeq, nSeq, h, m.DConv, m.KdaGateLowerBound, dev.Stream);
            // After the prep read the previous inputs.
            dev.GK.KdaConvUpdate(Ptr(dev.KQ), Ptr(dev.KK), Ptr(dev.KV), conv, rowsPerSeq, nSeq, dInner, m.DConv, dev.Stream);
            dev.GK.KdaScan(Ptr(dev.KScr), ssm, Ptr(dev.KCore), rowsPerSeq, nSeq, h, dev.Stream);
            dev.GK.KdaOut(Ptr(dev.KCore), Ptr(dev.KG), Ptr(L.SsmNorm), Ptr(dev.KOutIn), nt, h, m.RmsEps, dev.Stream);
            StageEnd(dev, StKdaCore);
            MatMulQ(dev, L.KdaOut, dev.KOutIn, dev.AttnOut, nt);
            StageEnd(dev, StAttnOut);
        }

        private void Mla(Dev dev, DevLayer L, int il, SlotLayer c, int nt, int p0, IntPtr pos = default)
        {
            var m = _m;
            int h = m.NHead, lat = m.KvLoraRank;
            // Every projection of the layer input, then those of the normed q latent (the shared
            // q8_1 buffer holds one input at a time).
            IntPtr cur = QuantizeShared(dev, dev.Cur, m.NEmbd, nt);
            MatMul(dev, L.WqA, dev.Cur, cur, dev.Qr, nt);
            MatMul(dev, L.WkvA, dev.Cur, cur, dev.KvRaw, nt);
            // The indexer's key and pooling gate are cached for every token: a later token's
            // selection scores the pools they form.
            MatMul(dev, L.IdxK, dev.Cur, cur, dev.IdxKey, nt);
            MatMul(dev, L.IdxGate, dev.Cur, cur, dev.IdxGateOut, nt);
            RmsNorm(dev.Qr, L.QANorm, nt);
            IntPtr qr = QuantizeShared(dev, dev.Qr, m.QLoraRank, nt);
            MatMul(dev, L.WqB, dev.Qr, qr, dev.Q, nt);
            dev.GK.MlaKvStore(Ptr(dev.KvRaw), Ptr(L.KvANorm), Ptr(c.KvCache), p0, nt, lat, m.RmsEps, dev.Stream, pos);
            dev.GK.IndexerStore(Ptr(dev.IdxKey), Ptr(dev.IdxGateOut), Ptr(L.IdxKNormW), Ptr(L.IdxKNormB),
                Ptr(c.IdxCache), p0, nt, m.IdxHeadDim, m.NormEps, dev.Stream, pos);
            // The pools this ubatch completes (their last member now cached).
            int firstPool = p0 / m.IdxKpool, endPool = (p0 + nt) / m.IdxKpool;
            dev.GK.PoolKeys(Ptr(c.IdxCache), Ptr(L.IdxApe), Ptr(c.PoolKeys), firstPool, endPool - firstPool,
                m.IdxHeadDim, dev.Stream, pos);

            // q_abs[h] = wk_b[h] q[h]: the per-head key decompression folded into the query, so
            // every head attends over the one shared latent.
            HeadProject(dev, L.WkBF16, dev.Q, dev.QF16, dev.QAbs, lat, m.HeadDimK, nt);
            StageEnd(dev, StMlaProj);

            float kqScale = 1.0f / MathF.Sqrt(m.HeadDimK);
            if (p0 + nt > m.IdxTopK)
            {
                // Past top_k cached tokens the selection bites: score the visible pools, keep the
                // best, attend over their cells and the query's own incomplete pool.
                MatMul(dev, L.IdxQB, dev.Qr, qr, dev.IdxQ, nt);
                MatMulF32(dev, L.IdxProj, dev.Cur, dev.IdxW, m.NEmbd, m.IdxNHead, nt);
                int pools = m.NCtx / m.IdxKpool;
                // A captured step's grid covers every pool the context holds (those past the row's
                // position are skipped).
                dev.GK.PoolScores(Ptr(dev.IdxQ), Ptr(dev.IdxW), Ptr(c.PoolKeys), Ptr(dev.PoolScores), p0,
                    m.IdxNHead, m.IdxHeadDim, pools, pos != IntPtr.Zero ? pools : (p0 + nt) / m.IdxKpool, nt, dev.Stream, pos);
                dev.DK.TopK(dev.PoolScores, dev.TopkIdx, dev.TopkCnt, p0, m.IdxKpool, SelectPools, pools, nt, dev.Stream, pos);
                dev.GK.ExpandCells(Ptr(dev.TopkIdx), Ptr(dev.TopkCnt), Ptr(dev.Cells), Ptr(dev.CellCnt), p0,
                    SelectPools, CellStride, nt, dev.Stream, pos);
                dev.DK.Attention(dev.QAbs, Ptr(c.KvCache), Ptr(c.KvCache), dev.Cells, dev.CellCnt, Ptr(dev.NegInf), dev.AttnO,
                    p0, m.NCtx, m.NCtx, h, lat, 4, 1, CellStride, kqScale, nt, dev.Stream, pos);
                if (pos == IntPtr.Zero)
                    SparseProbe?.Invoke(CaptureSparse(dev, L, il, c, nt, p0, pools, kqScale));
            }
            else
            {
                dev.DK.Attention(dev.QAbs, Ptr(c.KvCache), IntPtr.Zero, null, null, Ptr(dev.NegInf), dev.AttnO,
                    p0, m.NCtx, m.NCtx, h, lat, 0, 1, 0, kqScale, nt, dev.Stream, pos);
            }

            StageEnd(dev, StMlaAttn);
            // The value is the latent too: wv_b decompresses each head's result.
            HeadProject(dev, L.WvBF16, dev.AttnO, dev.AttnOF16, dev.VOut, m.HeadDimV, lat, nt);
            MatMulQ(dev, L.Wo, dev.VOut, dev.AttnOut, nt);
            StageEnd(dev, StAttnOut);
        }

        /// <summary>Per-head projection of every row. Up to 16 rows (decode, batched decode) take the
        /// row-invariant GEMV, so a batched row computes as its sequence's own step does; wider
        /// batches convert the input to F16 for the tensor-core GEMM.</summary>
        private void HeadProject(Dev dev, Tensor wF16, Tensor x, Tensor xF16, Tensor output, int outDim, int inDim, int nt)
        {
            if (nt <= GlmKernels.HeadGemvMaxRows)
            {
                dev.GK.HeadGemv(Ptr(wF16), Ptr(x), Ptr(output), _m.NHead, outDim, inDim, nt, dev.Stream);
                return;
            }
            dev.Alloc.Kernels.LaunchConvertF32F16(Ptr(x), Ptr(xF16), (long)nt * _m.NHead * inDim, dev.Stream);
            HeadGemm(dev, wF16, xF16, output, outDim, inDim, nt);
        }

        /// <summary>
        /// Per-head projection for every row: out[t, h, :] = W_h x[t, h, :], with W an F16 stack of
        /// NHead row-major [outDim, inDim] matrices, x F16 [nt, NHead, inDim] and out F32
        /// [nt, NHead, outDim]. One strided-batched GEMM over the heads.
        /// </summary>
        private void HeadGemm(Dev dev, Tensor wF16, Tensor xF16, Tensor output, int outDim, int inDim, int nt)
        {
            CudaDecodeCapture.Refuse("cuBLAS");
            int h = _m.NHead;
            dev.Alloc.Blas.SetStream(dev.Stream);
            float alpha = 1.0f, beta = 0.0f;
            CublasApi.cublasGemmStridedBatchedEx(
                dev.Alloc.Blas.Handle,
                CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                outDim, nt, inDim,
                ref alpha,
                Ptr(wF16), CublasApi.CUDA_R_16F, inDim, (long)outDim * inDim,
                Ptr(xF16), CublasApi.CUDA_R_16F, h * inDim, inDim,
                ref beta,
                Ptr(output), CublasApi.CUDA_R_32F, h * outDim, outDim,
                h,
                CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        }

        private void DenseFfn(Dev dev, DevLayer L, int nt)
        {
            int ff = L.FfnGate.Ne1;
            IntPtr cur = QuantizeShared(dev, dev.Cur, _m.NEmbd, nt);
            MatMul(dev, L.FfnGate, dev.Cur, cur, dev.ShGate, nt);
            MatMul(dev, L.FfnUp, dev.Cur, cur, dev.ShUp, nt);
            CudaMoe.SwigluClamp(dev.ShGate, dev.ShUp, (long)nt * ff, _m.SwigluClamp);
            MatMulQ(dev, L.FfnDown, dev.ShGate, dev.FfnOut, nt);
            StageEnd(dev, StShexp);
        }

        private void Moe(Dev dev, DevLayer L, int nt)
        {
            var m = _m;
            int e = m.NEmbd;
            MatMulF32(dev, L.GateInp, dev.Cur, dev.RouterLogits, e, m.NExpert, nt);
            dev.DK.MoeSelect(dev.RouterLogits, Ptr(L.ExpProbsBias), IntPtr.Zero, null, dev.Moe.Sel, dev.Moe.SelW,
                m.NExpert, m.NExpertUsed, m.ExpertWeightsNorm ? 1 : 0, m.ExpertWeightsScale, nt, dev.Stream,
                Dsv4Kernels.RouterSigmoid);
            StageEnd(dev, StRouter);

            int shFf = L.GateShexp.Ne1;
            IntPtr cur = QuantizeShared(dev, dev.Cur, e, nt);
            MatMul(dev, L.GateShexp, dev.Cur, cur, dev.ShGate, nt);
            MatMul(dev, L.UpShexp, dev.Cur, cur, dev.ShUp, nt);
            CudaMoe.SwigluClamp(dev.ShGate, dev.ShUp, (long)nt * shFf, m.SwigluClamp);
            MatMulQ(dev, L.DownShexp, dev.ShGate, dev.ShDown, nt);
            StageEnd(dev, StShexp);

            CudaMoe.Experts(dev.DK, dev.Alloc.Kernels, dev.Moe, L.GateExps, L.UpExps, L.DownExps,
                dev.Cur, dev.ShDown, dev.FfnOut, nt, m.NExpertUsed, m.NExpert, e, m.NFfExp, m.SwigluClamp,
                PerSlotMaxRows, dev.Stream);
            StageEnd(dev, StExperts);
        }

        /// <summary>Test hook: each MLA layer's pooled sparse selection on the single-sequence path,
        /// read back once its attention ran.</summary>
        internal Action<SparseLayerProbe> SparseProbe { get; set; }

        /// <summary>One sparse MLA layer's inputs and results for a ubatch.</summary>
        internal sealed class SparseLayerProbe
        {
            public int Layer, P0, Nt, Kpool, SelectPools, CellStride, PoolStride, IdxHeads, IdxDim, Heads, Latent, NEmbd;
            public float KqScale;
            /// <summary>The layer's normalized input [nt, NEmbd].</summary>
            public float[] Cur;
            /// <summary>Indexer queries [nt, IdxHeads * IdxDim] and head weights [nt, IdxHeads] (scaled).</summary>
            public float[] IdxQ, IdxW;
            /// <summary>Indexer cache rows [P0 + Nt, 2 * IdxDim]: LayerNormed key, then pooling gate.</summary>
            public System.Half[] IdxCache;
            /// <summary>Pool keys [(P0 + Nt) / Kpool, IdxDim] and the scores [Nt, PoolStride].</summary>
            public float[] PoolKeys, Scores;
            public int[] Sel, SelCnt, Cells, CellCnt;
            /// <summary>Absorbed queries and the attention result [Nt, Heads, Latent].</summary>
            public float[] QAbs, AttnO;
            /// <summary>The latent cache rows [P0 + Nt, Latent].</summary>
            public System.Half[] KvCache;
        }

        private SparseLayerProbe CaptureSparse(Dev dev, DevLayer L, int il, SlotLayer c, int nt, int p0, int pools, float kqScale)
        {
            var m = _m;
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            int rowsCached = p0 + nt, d = m.IdxHeadDim;
            return new SparseLayerProbe
            {
                Layer = il, P0 = p0, Nt = nt, Kpool = m.IdxKpool, SelectPools = SelectPools, CellStride = CellStride,
                PoolStride = pools, IdxHeads = m.IdxNHead, IdxDim = d, Heads = m.NHead, Latent = m.KvLoraRank,
                NEmbd = m.NEmbd, KqScale = kqScale,
                Cur = Download<float>(Ptr(dev.Cur), (long)nt * m.NEmbd),
                IdxQ = Download<float>(Ptr(dev.IdxQ), (long)nt * m.IdxNHead * d),
                IdxW = Download<float>(Ptr(dev.IdxW), (long)nt * m.IdxNHead),
                IdxCache = Download<System.Half>(Ptr(c.IdxCache), (long)rowsCached * 2 * d),
                PoolKeys = Download<float>(Ptr(c.PoolKeys), (long)(rowsCached / m.IdxKpool) * d),
                Scores = Download<float>(Ptr(dev.PoolScores), (long)nt * pools),
                Sel = Download<int>(Ptr(dev.TopkIdx), (long)nt * SelectPools),
                SelCnt = Download<int>(Ptr(dev.TopkCnt), nt),
                Cells = Download<int>(Ptr(dev.Cells), (long)nt * CellStride),
                CellCnt = Download<int>(Ptr(dev.CellCnt), nt),
                QAbs = Download<float>(Ptr(dev.QAbs), (long)nt * m.NHead * m.KvLoraRank),
                AttnO = Download<float>(Ptr(dev.AttnO), (long)nt * m.NHead * m.KvLoraRank),
                KvCache = Download<System.Half>(Ptr(c.KvCache), (long)rowsCached * m.KvLoraRank),
            };
        }

        private static T[] Download<T>(IntPtr src, long count) where T : unmanaged
        {
            var a = new T[count];
            fixed (T* p = a)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)p, src, new UIntPtr((ulong)(count * sizeof(T)))).ThrowOnError();
            return a;
        }

        /// <summary>Pools the indexer keeps per query (top_k cells in whole pools).</summary>
        private int SelectPools => _m.IdxTopK / _m.IdxKpool;

        /// <summary>Row stride of the per-query cell lists: every selected pool's cells plus the
        /// query's own incomplete pool.</summary>
        private int CellStride => SelectPools * _m.IdxKpool + _m.IdxKpool;

        /// <summary>Tokens up to which the experts run per (token, expert) slot: a batched decode
        /// step's rows then compute exactly as each sequence's own step does.</summary>
        public const int PerSlotMaxRows = 16;

        // TS_GLM_PERF=2 / 3: per-stage time, synchronized host time or GPU time (see CudaStageTimer).
        private const int StEmbed = 0, StHc = 1, StKdaProj = 2, StKdaCore = 3, StMlaProj = 4, StMlaAttn = 5,
            StAttnOut = 6, StRouter = 7, StExperts = 8, StShexp = 9, StHead = 10, StBoundary = 11;
        private static readonly string[] StageNames =
        {
            "embed", "hc", "kdaproj", "kdacore", "mlaproj", "mlaattn", "attnout", "router", "experts", "shexp/ffn",
            "head", "boundary",
        };
        // TS_GLM_PERF>=1: when the host had queued the whole step (the last launch before the
        // logits sync). Close to the step's wall time = the step is bound by launch overhead.
        private long _issueEnd;

        private double IssueMs(long start)
            => _issueEnd > start ? (_issueEnd - start) * 1000.0 / System.Diagnostics.Stopwatch.Frequency : 0;

        private void StageBegin(Dev dev) => _stages.SpanStart(dev.Alloc.Context, dev.Stream);

        private void StageEnd(Dev dev, int stage) => _stages.End(dev.Alloc.Context, dev.Stream, stage);

        /// <summary>The unweighted mean of the streams and the output norm, then the LM head: the last
        /// row's logits, or every row's (a batched decode step, a verify window). A verify also takes
        /// every row's normed hidden state into <paramref name="hOut"/>.</summary>
        private void Head(int nt, float[] logitsOut, bool allRows, float[] hOut = null)
        {
            IssueHead(nt, logitsOut != null, allRows, hOut != null);
            if (_perf > 0)
                _issueEnd = System.Diagnostics.Stopwatch.GetTimestamp();
            FinishHead(nt, logitsOut, allRows, hOut);
        }

        /// <summary>The head's kernels and the copies of its results into pinned memory.</summary>
        private void IssueHead(int nt, bool logits, bool allRows, bool hidden)
        {
            var m = _m;
            var dev = _devs[_lastDev];
            dev.MakeCurrent();
            int normRows = allRows || hidden ? nt : 1;
            using (Tensor xs = normRows == nt ? dev.Xs.CopyRef() : dev.Xs.Narrow(0, nt - 1, 1))
                dev.DK.HcMean(xs, dev.Cur, normRows, m.NEmbd, m.NEmbd, 0, dev.Stream);
            RmsNorm(dev.Cur, _outputNorm, normRows);
            int headRows = allRows ? nt : 1;
            long count = (long)headRows * m.NVocab;
            if (logits)
            {
                Tensor output = allRows ? dev.BatchLogits : dev.Logits;
                using (Tensor headIn = headRows == normRows ? dev.Cur.CopyRef() : dev.Cur.Narrow(0, nt - 1, 1))
                    MatMulQ(dev, _output, headIn, output, headRows);
                StageEnd(dev, StHead);
                CudaDriverApi.cuMemcpyDtoHAsync(_pinnedLogits, Ptr(output), new UIntPtr((ulong)count * 4), dev.Stream).ThrowOnError();
            }
            long hCount = (long)normRows * m.NEmbd;
            if (hidden)
                CudaDriverApi.cuMemcpyDtoHAsync(_pinnedHidden, Ptr(dev.Cur), new UIntPtr((ulong)hCount * 4), dev.Stream).ThrowOnError();
        }

        /// <summary>Wait for the head and copy its results out of pinned memory.</summary>
        private void FinishHead(int nt, float[] logitsOut, bool allRows, float[] hOut)
        {
            var m = _m;
            var dev = _devs[_lastDev];
            dev.MakeCurrent();
            int normRows = allRows || hOut != null ? nt : 1;
            long count = (long)(allRows ? nt : 1) * m.NVocab;
            long hCount = (long)normRows * m.NEmbd;
            CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            if (logitsOut != null)
                fixed (float* dst = logitsOut)
                    Buffer.MemoryCopy((void*)_pinnedLogits, dst, logitsOut.LongLength * 4L, count * 4L);
            if (hOut != null)
                fixed (float* dst = hOut)
                    Buffer.MemoryCopy((void*)_pinnedHidden, dst, hOut.LongLength * 4L, hCount * 4L);
            _stages.Report();
        }
    }
}
