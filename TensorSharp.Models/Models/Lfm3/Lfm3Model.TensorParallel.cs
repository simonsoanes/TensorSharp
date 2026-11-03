// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace TensorSharp.Models
{
    /// <summary>
    /// Tensor-parallel LFM3, Megatron style: each layer's mixer and FFN are a
    /// column-parallel projection, per-rank work, and a row-parallel projection whose
    /// AllReduce returns the replicated hidden state.
    /// <list type="bullet">
    /// <item>Attention: Q, the per-head gate, K and V split by heads (rank g owns
    /// query heads g·H/G.. and KV heads g·KV/G..), each rank attends over its own K/V
    /// cache shard, attn_output is row-parallel.</item>
    /// <item>Short conv: in_proj is [B | C | x], so it is split segment-aware - rank g
    /// gets [B_g | C_g | x_g] for its hidden/G channels - and the depthwise conv runs
    /// per channel, so each rank convolves and keeps the state of its own channels.
    /// out_proj is row-parallel.</item>
    /// <item>FFN: gate and up column-parallel, down row-parallel.</item>
    /// </list>
    /// The sandwich norms, the embedding and the LM head run on the reduced
    /// (replicated) tensors on rank 0.
    /// </summary>
    public sealed partial class Lfm3Model
    {
        // Per attention layer, per local rank; null on conv layers.
        private Tensor[][] _tpKvCacheK;
        private Tensor[][] _tpKvCacheV;
        private int _tpKvCacheCapacity;

        private void ValidateTpConstraints()
        {
            int tp = GlobalTpDegree;
            var errors = new List<string>();
            if (Config.NumHeads % tp != 0)
                errors.Add($"attention heads ({Config.NumHeads}) not divisible by TP degree ({tp})");
            if (Config.NumKVHeads % tp != 0)
                errors.Add($"KV heads ({Config.NumKVHeads}) not divisible by TP degree ({tp})");
            if (Config.HiddenSize % tp != 0)
                errors.Add($"hidden size ({Config.HiddenSize}) not divisible by TP degree ({tp}); the short-conv channels are split by it");
            if (Config.IntermediateSize > 0 && Config.IntermediateSize % tp != 0)
                errors.Add($"FFN length ({Config.IntermediateSize}) not divisible by TP degree ({tp})");
            if (errors.Count > 0)
                throw new NotSupportedException($"lfm3 cannot run with --tp {tp}: {string.Join("; ", errors)}.");
        }

        private void ShardLfm3WeightsForTP()
        {
            // Separate single-segment projections split contiguously by output rows (heads,
            // gate scalars, FFN columns); the outputs and down projections by input columns.
            ShardWeightsForTensorParallelism(
                columnParallelPatterns: new[] { "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_gate.weight",
                    "ffn_gate.weight", "ffn_up.weight" },
                rowParallelPatterns: new[] { "attn_output.weight", "shortconv.out_proj.weight", "ffn_down.weight" });

            // in_proj concatenates three hidden-wide segments; a contiguous split would hand
            // rank 0 all of B and rank 1 all of C (see ShardConcatenatedColumnParallel).
            int hidden = Config.HiddenSize;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerKinds[l] == LayerKind.Conv)
                    ShardConcatenatedColumnParallel($"blk.{l}.shortconv.in_proj.weight", hidden, hidden, hidden);
            }
        }

        private void InitTpCaches(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _tpKvCacheCapacity = initialSeqLen;
            int kvHeadsPerRank = Config.NumKVHeads / GlobalTpDegree;
            DType kvDtype = _kvCacheDtype.ToDType();
            _tpKvCacheK = new Tensor[Config.NumLayers][];
            _tpKvCacheV = new Tensor[Config.NumLayers][];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerKinds[l] == LayerKind.Conv) continue;
                _tpKvCacheK[l] = new Tensor[TpDegree];
                _tpKvCacheV[l] = new Tensor[TpDegree];
                for (int r = 0; r < TpDegree; r++)
                {
                    IAllocator alloc = _tpGroup.GetAllocator(r);
                    _tpKvCacheK[l][r] = new Tensor(alloc, kvDtype, kvHeadsPerRank, initialSeqLen, _headDim);
                    _tpKvCacheV[l][r] = new Tensor(alloc, kvDtype, kvHeadsPerRank, initialSeqLen, _headDim);
                    InitializeCacheTensor(_tpKvCacheK[l][r]);
                    InitializeCacheTensor(_tpKvCacheV[l][r]);
                }
            }
            AllocateConvState();
            _cacheSeqLen = 0;
        }

        private void EnsureTpCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _tpKvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
            {
                throw new InvalidOperationException(
                    $"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");
            }

            int newCapacity = Math.Max(_tpKvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_tpKvCacheK[l] == null) continue;
                for (int r = 0; r < TpDegree; r++)
                {
                    _tpKvCacheK[l][r] = GrowCache(_tpKvCacheK[l][r], newCapacity);
                    _tpKvCacheV[l][r] = GrowCache(_tpKvCacheV[l][r], newCapacity);
                }
            }

            _tpKvCacheCapacity = newCapacity;
            Console.WriteLine($"Expanded lfm3 TP attention cache to {newCapacity} tokens ({TpDegree} ranks).");
        }

        private void ResetTpCaches()
        {
            if (_tpKvCacheK == null)
                return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_tpKvCacheK[l] == null) continue;
                for (int r = 0; r < TpDegree; r++)
                {
                    ResetCacheTensor(_tpKvCacheK[l][r]);
                    ResetCacheTensor(_tpKvCacheV[l][r]);
                }
            }
        }

        private void DisposeTpCaches()
        {
            if (_tpKvCacheK != null)
            {
                for (int l = 0; l < _tpKvCacheK.Length; l++)
                {
                    if (_tpKvCacheK[l] == null) continue;
                    foreach (Tensor t in _tpKvCacheK[l]) t?.Dispose();
                    foreach (Tensor t in _tpKvCacheV[l]) t?.Dispose();
                }
            }
            DisposeTpWeightReplicaCache();
        }

        private float[] ForwardTP(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureTpCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor embedded = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;
            Tensor[] hidden = BroadcastTensorToAllRanks(embedded);
            embedded.Dispose();

            for (int layer = 0; layer < Config.NumLayers; layer++)
                DecoderLayerTP(hidden, layer, seqLen, startPos);

            // hidden is replicated after every AllReduce: the head runs on rank 0.
            ComputeLogits(hidden[0], seqLen);
            DisposeAll(hidden);

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        private void DecoderLayerTP(Tensor[] hidden, int layer, int seqLen, int startPos)
        {
            string p = $"blk.{layer}.";
            bool conv = _layerKinds[layer] == LayerKind.Conv;

            Tensor[] normed = TpRMSNorm(hidden, p + "attn_norm.weight");
            Tensor[] partial = conv
                ? ShortConvTP(normed[0], layer, seqLen)
                : AttentionTP(normed[0], layer, seqLen, startPos);
            DisposeAll(normed);
            Tensor mixed = TpRowParallelLinear(partial, p + (conv ? "shortconv.out_proj.weight" : "attn_output.weight"));
            DisposeAll(partial);
            AddBranchTP(hidden, mixed, _sandwichNorms ? p + "post_attention_norm.weight" : null);

            Tensor[] ffnNormed = TpRMSNorm(hidden, p + "ffn_norm.weight");
            Tensor[] gate = TpColumnParallelLinear(ffnNormed[0], p + "ffn_gate.weight");
            Tensor[] up = TpColumnParallelLinear(ffnNormed[0], p + "ffn_up.weight");
            DisposeAll(ffnNormed);
            _tpGroup.RunPerRank(r => Ops.SiLUMul(gate[r], gate[r], up[r]));
            DisposeAll(up);
            Tensor down = TpRowParallelLinear(gate, p + "ffn_down.weight");
            DisposeAll(gate);
            AddBranchTP(hidden, down, _sandwichNorms ? p + "post_ffw_norm.weight" : null);
        }

        /// <summary>hidden[r] += norm(reduced) on every rank; disposes reduced.</summary>
        private void AddBranchTP(Tensor[] hidden, Tensor reduced, string postNormName)
        {
            if (postNormName != null)
            {
                Tensor normed = RMSNormOp(reduced, postNormName);
                reduced.Dispose();
                reduced = normed;
            }
            Tensor[] replicas = BroadcastTensorToAllRanks(reduced);
            TpResidualAdd(hidden, replicas);
            // On GGML a rank's "replica" is the source tensor itself (host memory every
            // rank reads); only the copies are ours to free.
            for (int r = 0; r < replicas.Length; r++)
            {
                if (!ReferenceEquals(replicas[r], reduced))
                    replicas[r].Dispose();
            }
            reduced.Dispose();
        }

        private Tensor[] ShortConvTP(Tensor normed, int layer, int seqLen)
        {
            long t0 = Stopwatch.GetTimestamp();
            Tensor[] bcx = TpColumnParallelLinear(normed, $"blk.{layer}.shortconv.in_proj.weight");
            int channels = Config.HiddenSize / GlobalTpDegree;
            var gated = new Tensor[TpDegree];
            // Ranks own disjoint channel ranges of the layer's conv state.
            _tpGroup.RunPerRank(r =>
            {
                gated[r] = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, seqLen, channels);
                RunShortConv(bcx[r], gated[r], layer, (TpRankOffset + r) * channels, channels, seqLen);
            });
            DisposeAll(bcx);
            _attnTicks += Stopwatch.GetTimestamp() - t0;
            return gated;
        }

        private Tensor[] AttentionTP(Tensor normed, int layer, int seqLen, int startPos)
        {
            string p = $"blk.{layer}.";
            Tensor[] q = TpColumnParallelLinear(normed, p + "attn_q.weight");
            Tensor[] k = TpColumnParallelLinear(normed, p + "attn_k.weight");
            Tensor[] v = TpColumnParallelLinear(normed, p + "attn_v.weight");
            Tensor[] gate = TpColumnParallelLinear(normed, p + "attn_gate.weight");
            int headsPerRank = Config.NumHeads / GlobalTpDegree;
            int kvHeadsPerRank = Config.NumKVHeads / GlobalTpDegree;

            // One rank after another: the replicated-weight cache and the RoPE tables
            // GatedAttention fills are shared.
            var results = new Tensor[TpDegree];
            for (int r = 0; r < TpDegree; r++)
            {
                results[r] = GatedAttention(q[r], k[r], v[r], gate[r], _tpKvCacheK[layer][r], _tpKvCacheV[layer][r],
                    layer, r, headsPerRank, kvHeadsPerRank, seqLen, startPos);
            }
            return results;
        }

        private static void DisposeAll(Tensor[] tensors)
        {
            foreach (Tensor t in tensors) t?.Dispose();
        }
    }
}
