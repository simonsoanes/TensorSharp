// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Batched paged-attention forward path for Qwen3. Mirrors vLLM's
// flash_attn_varlen_func semantics: a single forward call processes N
// sequences (each with its own length and position range) by packing all
// tokens into one tensor, scattering K/V into a layer-private paged buffer,
// and running per-sequence attention via <see cref="ManagedPagedAttention"/>.
//
// Linear / norm / FFN ops are batched over the full token axis (one big
// matmul covers every sequence in the step), which is the throughput win
// continuous batching is supposed to deliver. Attention itself is computed
// per-sequence inside the kernel; each query gathers K/V from its own
// sequence's block table.
//
// This path coexists with the legacy single-sequence Forward path. Models
// that opt into batched execution by implementing <see cref="IBatchedPagedModel"/>
// see <see cref="BatchExecutor.ExecuteStepBatched"/> dispatch through here;
// callers that still hand the model one-sequence input continue to use the
// existing Forward.
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Qwen3Model : IBatchedPagedModel
    {
        // Per-layer paged K/V floats. Each buffer has shape
        // [pagedNumBlocks, pagedBlockSize, numKvHeads, headDim] as a flat
        // float[]. Lazily allocated on the first ForwardBatch call once we
        // know the engine's block-pool dimensions.
        private float[][] _pagedK;
        private float[][] _pagedV;
        private int _pagedNumBlocks;
        private int _pagedBlockSize;

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            if (IsTensorParallel)
                return ForwardBatchTP(ctx);

            int numSeqs = ctx.Sequences.Count;
            if (numSeqs == 0) return Array.Empty<float[]>();

            int hidden = Config.HiddenSize;
            int numHeads = Config.NumHeads;
            int numKvHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int qDim = numHeads * headDim;
            int kDim = numKvHeads * headDim;
            float scale = 1.0f / MathF.Sqrt(headDim);

            // Determine block-pool dimensions from the FIRST sequence's
            // block table (all sequences in a batch share the same pool).
            int blockSize = ctx.Sequences[0].BlockTable.BlockSize;
            int maxBlockId = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var bt = ctx.BlockTables[s];
                for (int b = 0; b < bt.Length; b++)
                    if (bt[b] > maxBlockId) maxBlockId = bt[b];
            }
            int requiredBlocks = maxBlockId + 1;
            EnsurePagedBuffersAllocated(requiredBlocks, blockSize, numKvHeads, headDim);

            // Flatten metadata into plain arrays for the helpers.
            int numTokens = 0;
            for (int s = 0; s < numSeqs; s++) numTokens += ctx.NumScheduledTokens[s];
            int[] positions = ctx.Positions.ToArray();
            int[] queryStartLoc = ctx.QueryStartLoc.ToArray();
            int[] slotMapping = ctx.SlotMapping.ToArray();
            int[] seqLens = new int[numSeqs];
            for (int s = 0; s < numSeqs; s++)
            {
                var seq = ctx.Sequences[s];
                seqLens[s] = seq.NumComputedTokens + ctx.NumScheduledTokens[s];
            }

            // Build the concatenated input-tokens array.
            int[] flatTokens = new int[numTokens];
            int writeCursor = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var seq = ctx.Sequences[s];
                int startTok = seq.NumComputedTokens;
                int take = ctx.NumScheduledTokens[s];
                for (int i = 0; i < take; i++)
                    flatTokens[writeCursor++] = seq.TokenAt(startTok + i);
            }

            // Build the per-token positions tensor that Ops.RoPEEx wants
            // (one row per (token, head); positions repeat across heads
            // within a token).
            using var positionsTensorQ = BuildRoPEPositionsTensor(positions, numHeads);
            using var positionsTensorK = BuildRoPEPositionsTensor(positions, numKvHeads);

            // ----- embed -----
            Tensor hiddenStates = Embedding(flatTokens);
            // Output: [numTokens, hidden]

            EnsureKvCacheHostSynchronized();

            // ----- per-layer transformer -----
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                string[] wn = _layerWeightNames[layer];

                // Residual snapshot.
                Tensor residual = Ops.NewContiguous(hiddenStates);

                // Pre-attention RMSNorm.
                Tensor normed = RMSNormOp(hiddenStates, wn[0]);
                hiddenStates.Dispose();

                // Q/K/V projection.
                Tensor qkv = LinearForward(normed, wn[1]);
                normed.Dispose();

                Tensor q, k, v;
                using (var qView = qkv.Narrow(1, 0, qDim))
                    q = Ops.NewContiguous(qView);
                using (var kView = qkv.Narrow(1, qDim, kDim))
                    k = Ops.NewContiguous(kView);
                using (var vView = qkv.Narrow(1, qDim + kDim, kDim))
                    v = Ops.NewContiguous(vView);
                qkv.Dispose();

                // Q-norm / K-norm (per-head RMSNorm).
                q = ApplyQKNormBatched(q, wn[2], numTokens, numHeads, headDim);
                k = ApplyQKNormBatched(k, wn[3], numTokens, numKvHeads, headDim);

                // Per-token RoPE on Q and K via Ops.RoPEEx (which supports
                // an arbitrary positions tensor).
                q = ApplyBatchedRoPE(q, positionsTensorQ, numTokens, numHeads, headDim);
                k = ApplyBatchedRoPE(k, positionsTensorK, numTokens, numKvHeads, headDim);

                // Scatter K, V to the layer's paged buffer at the slots
                // dictated by slot_mapping.
                float[] kFlat = k.GetElementsAsFloat(numTokens * kDim);
                float[] vFlat = v.GetElementsAsFloat(numTokens * kDim);
                PagedKvBatchOps.ScatterKv(
                    kFlat, vFlat, _pagedK[layer], _pagedV[layer],
                    slotMapping, numTokens, numKvHeads, headDim, _pagedBlockSize);
                k.Dispose();
                v.Dispose();

                // Per-sequence paged attention.
                float[] qFlat = q.GetElementsAsFloat(numTokens * qDim);
                q.Dispose();
                float[] attnFlat = new float[numTokens * qDim];
                ManagedPagedAttention.Forward(
                    qFlat, _pagedK[layer], _pagedV[layer], attnFlat,
                    numTokens, numHeads, numKvHeads, headDim,
                    _pagedBlockSize,
                    queryStartLoc, seqLens, positions, ctx.BlockTables, numSeqs,
                    scale, causal: true);

                // Output projection.
                Tensor attnOut = CreateFloatTensor(attnFlat, numTokens, qDim);
                Tensor attnProj = LinearForward(attnOut, wn[4]);
                attnOut.Dispose();

                // Residual add.
                Ops.Add(residual, residual, attnProj);
                attnProj.Dispose();

                // FFN: fused dense SwiGLU (norm + gate/up + SiLU·mul + down +
                // residual) in one GGML graph, keeping the large intermediate
                // on-device. Falls back to the unfused chain when ineligible.
                if (!TryFusedDenseSwiGLUFFNInto(residual, wn[5], wn[6], wn[7]))
                {
                    Tensor ffnNormed = RMSNormOp(residual, wn[5]);
                    Tensor ffnOut = FFN(ffnNormed, wn[6], wn[7], numTokens);
                    ffnNormed.Dispose();
                    Ops.Add(residual, residual, ffnOut);
                    ffnOut.Dispose();
                }

                hiddenStates = residual;
            }

            // ----- final norm + LM head per-sequence -----
            Tensor finalNormed = RMSNormOp(hiddenStates, "output_norm.weight");
            hiddenStates.Dispose();

            // Gather the last token of each sequence into a [numSeqs, hidden] tensor.
            float[] finalFlat = finalNormed.GetElementsAsFloat(numTokens * hidden);
            finalNormed.Dispose();
            float[] lastTokensPacked = PagedKvBatchOps.GatherLastTokenPerSeq(
                finalFlat, hidden, queryStartLoc, numSeqs);
            Tensor lastHidden = CreateFloatTensor(lastTokensPacked, numSeqs, hidden);

            Tensor logitsTensor = LinearForward(lastHidden, "output.weight");
            if (logitsTensor == null)
                logitsTensor = LinearForward(lastHidden, "token_embd.weight");
            lastHidden.Dispose();

            float[] allLogits = logitsTensor.GetElementsAsFloat(numSeqs * Config.VocabSize);
            logitsTensor.Dispose();

            // Split per-sequence.
            var perSeq = new float[numSeqs][];
            for (int s = 0; s < numSeqs; s++)
            {
                var slice = new float[Config.VocabSize];
                Buffer.BlockCopy(allLogits, s * Config.VocabSize * sizeof(float),
                                 slice, 0, Config.VocabSize * sizeof(float));
                perSeq[s] = slice;
            }
            return perSeq;
        }

        /// <summary>Per-head RMSNorm over a batched [numTokens, dim] buffer.
        /// dim = numHeads * headDim.</summary>
        private Tensor ApplyQKNormBatched(Tensor data, string weightName, int numTokens, int numHeads, int headDim)
        {
            using var reshaped = data.View(numTokens * numHeads, headDim);
            var alpha = _weights[weightName];
            Tensor normed = Ops.RMSNorm(null, reshaped, alpha, null, Config.Eps);
            data.Dispose();
            Tensor result = normed.View(numTokens, numHeads * headDim);
            normed.Dispose();
            return result;
        }

        /// <summary>Apply Qwen3's RoPE flavour with arbitrary per-token positions.
        /// </summary>
        private Tensor ApplyBatchedRoPE(Tensor data, Tensor positionsTensor, int numTokens, int numHeads, int headDim)
        {
            using var reshaped = data.View(1, numTokens, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, positionsTensor, headDim, 2, 0,
                Config.RopeBase, 1.0f / Config.RopeScale,
                0.0f, 1.0f, 0.0f, 0.0f);
            data.Dispose();
            Tensor flat = result.View(numTokens, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        /// <summary>Build the positions tensor for <see cref="Ops.RoPEEx"/>: one
        /// integer per (token, head) row.</summary>
        private Tensor BuildRoPEPositionsTensor(int[] tokenPositions, int numHeads)
        {
            int total = tokenPositions.Length * numHeads;
            int[] expanded = new int[total];
            for (int t = 0; t < tokenPositions.Length; t++)
                for (int h = 0; h < numHeads; h++)
                    expanded[t * numHeads + h] = tokenPositions[t];
            return CreateIntTensor(expanded, total);
        }

        private void EnsurePagedBuffersAllocated(int numBlocks, int blockSize, int numKvHeads, int headDim)
        {
            if (_pagedK != null && _pagedNumBlocks >= numBlocks && _pagedBlockSize == blockSize)
                return;

            // Pad up so we have at least the requested capacity, but
            // avoid reallocating on every small bump.
            int targetBlocks = Math.Max(numBlocks, _pagedNumBlocks * 2);

            int numLayers = Config.NumLayers;
            // CRITICAL: preserve existing K/V data when growing. The engine
            // submits sequences over multiple scheduler steps; rebuilding
            // without copy zeroes any already-written K/V and breaks the
            // first sequence's continuation. See the matching comment in
            // Mistral3 / Gemma4's EnsurePagedBuffersAllocated.
            float[][] oldK = _pagedK;
            float[][] oldV = _pagedV;
            int oldNumBlocks = _pagedNumBlocks;
            int oldBlockSize = _pagedBlockSize;

            _pagedK = new float[numLayers][];
            _pagedV = new float[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                _pagedK[l] = PagedKvBatchOps.AllocateLayerBuffer(targetBlocks, blockSize, numKvHeads, headDim);
                _pagedV[l] = PagedKvBatchOps.AllocateLayerBuffer(targetBlocks, blockSize, numKvHeads, headDim);
                if (oldK != null && oldK[l] != null && oldBlockSize == blockSize)
                {
                    long copyLen = Math.Min(
                        (long)oldNumBlocks * blockSize * numKvHeads * headDim,
                        oldK[l].LongLength);
                    Array.Copy(oldK[l], _pagedK[l], copyLen);
                    Array.Copy(oldV[l], _pagedV[l], copyLen);
                }
            }
            _pagedNumBlocks = targetBlocks;
            _pagedBlockSize = blockSize;
        }

        /// <summary>The model can copy a solo sequence's linear K/V history
        /// into paged storage (where <c>ForwardBatch</c> reads from). This is the
        /// gate on the N=1 SingleSequenceFused fast path: without it every solo
        /// request is routed through the op-by-op batched paged forward, which on
        /// Qwen3-8B/ggml_cuda measured 8.5 tok/s against the fused path's
        /// 100+ tok/s. Ported from GptOssModel.BatchedForward.cs; block-quantised
        /// caches are excluded (only F32/F16 reads are implemented).</summary>
        public bool SupportsLinearKVMigration =>
            _kvCacheK != null && _kvCacheV != null
            && !_kvCacheDtype.IsBlockQuantized();

        /// <summary>Copy <paramref name="owner"/>'s K/V history out of the linear
        /// per-layer cache (layout <c>[numKVHeads, capacity, headDim]</c>) into
        /// the paged buffers at slot positions derived from
        /// <c>owner.BlockTable</c>. Qwen3 has no SWA circular cache, so every
        /// position from 0..<c>_cacheSeqLen</c> is recoverable.</summary>
        public bool TryMigrateLinearKVToPaged(SequenceState owner, int blockSize)
        {
            if (owner == null) return false;
            if (!SupportsLinearKVMigration) return false;
            int ownerTokens = _cacheSeqLen;
            if (ownerTokens <= 0) return true;
            if (owner.BlockTable.NumBlocks <= 0) return false;

            // Flush rows the whole-model decode graph only ever wrote on-device,
            // then any per-tensor device shadow, so the float reads below see the
            // freshest K/V.
            if (IsGgmlBackend)
            {
                EnsureKvCacheHostSynchronized();
                var seen = new HashSet<Storage>();
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (_kvCacheK[l] != null && seen.Add(_kvCacheK[l].Storage))
                        SyncTensorHostCache(_kvCacheK[l]);
                    if (_kvCacheV[l] != null && seen.Add(_kvCacheV[l].Storage))
                        SyncTensorHostCache(_kvCacheV[l]);
                }
            }

            int maxBlockId = 0;
            int numBlocks = owner.BlockTable.NumBlocks;
            for (int b = 0; b < numBlocks; b++)
            {
                int id = owner.BlockTable.Blocks[b].Id;
                if (id > maxBlockId) maxBlockId = id;
            }
            int kvHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            EnsurePagedBuffersAllocated(maxBlockId + 1, blockSize, kvHeads, headDim);

            int stridePaged = kvHeads * headDim;
            int numLayers = Config.NumLayers;

            for (int layer = 0; layer < numLayers; layer++)
            {
                int cacheLen = (int)_kvCacheK[layer].Sizes[1];
                int totalElems = kvHeads * cacheLen * headDim;
                if (!TryReadCacheAsF32(_kvCacheK[layer], totalElems, out float[] kFlat) ||
                    !TryReadCacheAsF32(_kvCacheV[layer], totalElems, out float[] vFlat))
                {
                    return false;
                }

                float[] kPaged = _pagedK[layer];
                float[] vPaged = _pagedV[layer];

                for (int p = 0; p < ownerTokens; p++)
                {
                    int blockIdx = p / blockSize;
                    int offsetInBlock = p % blockSize;
                    int physBlockId = owner.BlockTable.Blocks[blockIdx].Id;
                    int slot = physBlockId * blockSize + offsetInBlock;
                    int slotOffset = slot * stridePaged;

                    for (int h = 0; h < kvHeads; h++)
                    {
                        int srcOffset = (h * cacheLen + p) * headDim;
                        int dstOffset = slotOffset + h * headDim;
                        Buffer.BlockCopy(
                            kFlat, srcOffset * sizeof(float),
                            kPaged, dstOffset * sizeof(float),
                            headDim * sizeof(float));
                        Buffer.BlockCopy(
                            vFlat, srcOffset * sizeof(float),
                            vPaged, dstOffset * sizeof(float),
                            headDim * sizeof(float));
                    }
                }
            }

            return true;
        }

        private static unsafe bool TryReadCacheAsF32(Tensor cache, int totalElems, out float[] flat)
        {
            if (cache.ElementType == DType.Float32)
            {
                flat = cache.GetElementsAsFloat(totalElems);
                return true;
            }
            if (cache.ElementType == DType.Float16)
            {
                flat = new float[totalElems];
                ushort* src = TensorComputePrimitives.GetHalfPointer(cache);
                fixed (float* dst = flat)
                {
                    TensorComputePrimitives.F16ToF32(dst, src, totalElems);
                }
                return true;
            }
            flat = null;
            return false;
        }
    }
}
