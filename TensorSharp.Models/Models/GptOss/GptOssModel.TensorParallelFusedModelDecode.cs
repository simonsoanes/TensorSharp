// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Whole-model fused decode for GPT-OSS under tensor parallelism.
    ///
    /// Without this, <c>--tp N</c> disabled the fused whole-model graph outright
    /// (WillUseFusedModelDecode's <c>|| IsTensorParallel</c>) and fell back to the
    /// per-op managed chain, which is where the pre-fused-graph throughput lives:
    /// on 2x RTX 5090 that measured 28.9 tok/s against 349 tok/s on ONE GPU, a 12x
    /// loss for adding a second card. The collective was never the problem - the
    /// per-op chain was.
    ///
    /// Each rank builds the same graph it would build alone, over its own shard,
    /// and hands it back unexecuted as a <c>TpRankPlan</c>. The driver then runs
    /// all ranks segment by segment, summing at the two cut points every layer
    /// has: the attention output projection (row-parallel) and the routed MoE sum
    /// (expert-parallel). That is 2 AllReduce per layer instead of ~30 host round
    /// trips.
    ///
    /// Single node only for now. The native expert LUT derives this rank's expert
    /// slice from the LOCAL device index, which only equals the global expert
    /// offset when <see cref="ModelBase.TpRankOffset"/> is 0; a multi-node run
    /// keeps the per-op chain until the descriptor carries an explicit first-expert
    /// field.
    /// </summary>
    public partial class GptOssModel
    {
        // TS_GPTOSS_TP_FUSED_DECODE=0 falls back to the per-op tensor-parallel chain.
        private static readonly bool TpFusedModelDecodeEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_GPTOSS_TP_FUSED_DECODE"), "0", StringComparison.Ordinal);

        private GptOssLayerDecodeArgs[][] _tpFdLayers;   // [rank][layer]
        private IntPtr[] _tpFdPlans;
        private bool _tpFdChecked;
        private bool _tpFdReady;
        private bool _tpFdFailed;
        private int _tpFdBuiltCapacity = -1;
        private bool _tpFdBanner;

        // One line per DISTINCT reason. A single process-wide latch used to print
        // the outermost refusal and hide the one that actually mattered.
        private readonly HashSet<string> _tpFdDeclineLogged = new HashSet<string>(StringComparer.Ordinal);

        private bool TpFdBail(string reason)
        {
            if (_tpFdDeclineLogged.Add(reason))
            {
                Console.Error.WriteLine(
                    $"[gptoss-tp] fused whole-model decode NOT engaged ({reason}); " +
                    "falling back to the per-op tensor-parallel decode.");
            }
            return false;
        }

        /// <summary>
        /// Whether the fused tensor-parallel decode can serve this model at all.
        /// Answered once and latched, because every term is load-time constant.
        /// </summary>
        private bool TpFusedModelDecodeAvailable()
        {
            if (_tpFdFailed) return false;
            if (_tpFdChecked) return _tpFdReady;
            _tpFdChecked = true;
            _tpFdReady = false;

            if (!TpFusedModelDecodeEnabled) return TpFdBail("TS_GPTOSS_TP_FUSED_DECODE=0");
            if (!FusedModelDecodeEnabled) return TpFdBail("TS_GPTOSS_MODEL_DECODE=0");
            if (!IsGgmlBackend) return TpFdBail("not a GGML backend");
            if (!IsTensorParallel) return TpFdBail("tensor parallelism is not active");
            if (GlobalTpDegree != TpDegree)
                return TpFdBail($"multi-node TP (global={GlobalTpDegree}, local={TpDegree}) is not supported yet");
            if (!UsesExpertParallelMoE)
                return TpFdBail("MoE is not expert-parallel (the fused graph needs whole-expert shards)");
            if (_layerStackedReady == 0) return TpFdBail("stacked expert weights are not ready");
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (MoeCpuOffloadConfig.IsLayerOnCpu(l))
                    return TpFdBail("MoE CPU offload is not supported under fused TP");
            }
            int kvType = _kvCacheDtype.GgmlType();
            if (kvType != 0 && kvType != 1) return TpFdBail("KV cache is neither F32 nor F16");
            if (!_quantWeights.ContainsKey("output.weight") && !_quantWeights.ContainsKey("token_embd.weight"))
                return TpFdBail("no quantized LM head to fold");
            if (!_weights.ContainsKey("output_norm.weight")) return TpFdBail("no output_norm.weight");
            if (!GgmlBasicOps.TensorParallelFusedAvailable(TpDegree))
                return TpFdBail("the native fused TP executor is unavailable");

            _tpFdReady = true;
            return true;
        }

        /// <summary>
        /// Builds the per-rank descriptor arrays once. Every weight is either this
        /// rank's shard or a genuinely replicated tensor; anything missing declines
        /// the whole path rather than silently binding rank 0's copy everywhere.
        /// </summary>
        private unsafe bool TryBuildTpFdLayerDescs()
        {
            if (_tpFdLayers != null) return true;

            int tp = TpDegree;
            int gTp = GlobalTpDegree;
            int numLayers = Config.NumLayers;
            int structBytes = System.Runtime.InteropServices.Marshal.SizeOf<GptOssLayerDecodeArgs>();
            int headsPerRank = Config.NumHeads / gTp;
            int kvHeadsPerRank = Config.NumKVHeads / gTp;

            if (Config.NumHeads % gTp != 0 || Config.NumKVHeads % gTp != 0)
                return TpFdBail("head counts are not divisible by the TP degree");

            var layers = new GptOssLayerDecodeArgs[tp][];
            for (int r = 0; r < tp; r++)
                layers[r] = new GptOssLayerDecodeArgs[numLayers];

            for (int l = 0; l < numLayers; l++)
            {
                string[] wn = _layerNames[l];

                // Replicated: norms and the router. The router deliberately stays
                // global - it selects over ALL experts on every rank, and the
                // native LUT maps the winners onto this rank's slice.
                if (!_weights.TryGetValue(wn[0], out var attnNormW)) return TpFdBail($"layer {l}: no {wn[0]}");
                if (!_weights.TryGetValue(wn[5], out var postAttnNormW)) return TpFdBail($"layer {l}: no {wn[5]}");
                if (!_weights.TryGetValue(wn[6], out var gateInpW)) return TpFdBail($"layer {l}: no {wn[6]}");
                _weights.TryGetValue(wn[7], out var gateInpBias);
                // Row-parallel bias: added AFTER the reduction (the native cut sits
                // on the raw matmul), so every rank adds the same full vector once.
                _weights.TryGetValue(wn[4], out var oBias);

                for (int r = 0; r < tp; r++)
                {
                    if (!TryTpFdShard(wn[1], r, out var qkvQw))
                        return TpFdBail($"layer {l} rank {r}: no TP shard for {wn[1]}");
                    if (!TryTpFdShard(wn[3], r, out var oQw))
                        return TpFdBail($"layer {l} rank {r}: no TP shard for {wn[3]}");

                    // Column-parallel bias: this rank's [Q_r|K_r|V_r] slice.
                    IntPtr qkvBiasPtr = IntPtr.Zero;
                    if (_tpWeights.TryGetValue(wn[2], out var qkvBiasShards) && qkvBiasShards[r] != null)
                        qkvBiasPtr = (IntPtr)GetFloatPtr(qkvBiasShards[r]);
                    else if (_weights.TryGetValue(wn[2], out var qkvBiasWhole) && tp == 1)
                        qkvBiasPtr = (IntPtr)GetFloatPtr(qkvBiasWhole);
                    else if (_weights.ContainsKey(wn[2]))
                        return TpFdBail($"layer {l} rank {r}: no TP shard for {wn[2]}");

                    // Sinks are per query head, so they shard with the heads.
                    IntPtr sinksPtr = IntPtr.Zero;
                    float[] sinks = _layerSinks?[l];
                    if (sinks != null)
                    {
                        if (sinks.Length != Config.NumHeads)
                            return TpFdBail($"layer {l}: sinks length {sinks.Length} != {Config.NumHeads} heads");
                        var slice = new float[headsPerRank];
                        Array.Copy(sinks, (TpRankOffset + r) * headsPerRank, slice, 0, headsPerRank);
                        sinksPtr = PinArray(slice);
                    }

                    var gateW = _tpStackedGate[l]?[r];
                    var upW = _tpStackedUp[l]?[r];
                    var downW = _tpStackedDown[l]?[r];
                    if (gateW == null || upW == null || downW == null)
                        return TpFdBail($"layer {l} rank {r}: no expert shard");

                    layers[r][l] = new GptOssLayerDecodeArgs
                    {
                        AttnNormW = (IntPtr)GetFloatPtr(attnNormW),
                        QkvW = qkvQw.CacheKey,
                        QkvB = qkvBiasPtr,
                        KW = IntPtr.Zero,
                        KB = IntPtr.Zero,
                        VW = IntPtr.Zero,
                        VB = IntPtr.Zero,
                        OW = oQw.CacheKey,
                        OB = oBias != null ? (IntPtr)GetFloatPtr(oBias) : IntPtr.Zero,
                        KCache = IntPtr.Zero,   // refreshed per call
                        VCache = IntPtr.Zero,
                        Sinks = sinksPtr,
                        PostAttnNormW = (IntPtr)GetFloatPtr(postAttnNormW),
                        GateInpW = (IntPtr)GetFloatPtr(gateInpW),
                        GateInpB = gateInpBias != null ? (IntPtr)GetFloatPtr(gateInpBias) : IntPtr.Zero,
                        GateExps = gateW.Data,
                        GateExpsB = PinArray(_tpGateBias?[l]?[r]),
                        UpExps = upW.Data,
                        UpExpsB = PinArray(_tpUpBias?[l]?[r]),
                        DownExps = downW.Data,
                        DownExpsB = PinArray(_tpDownBias?[l]?[r]),

                        QkvNe0 = qkvQw.Ne0, QkvNe1 = qkvQw.Ne1, QkvBytes = qkvQw.RawBytes,
                        KNe0 = 0, KNe1 = 0, KBytes = 0,
                        VNe0 = 0, VNe1 = 0, VBytes = 0,
                        ONe0 = oQw.Ne0, ONe1 = oQw.Ne1, OBytes = oQw.RawBytes,
                        GeNe0 = gateW.PerExpertNe0, GeNe1 = gateW.PerExpertNe1, GeBytes = gateW.TotalRawBytes,
                        UeNe0 = upW.PerExpertNe0, UeNe1 = upW.PerExpertNe1, UeBytes = upW.TotalRawBytes,
                        DeNe0 = downW.PerExpertNe0, DeNe1 = downW.PerExpertNe1, DeBytes = downW.TotalRawBytes,

                        StructBytes = structBytes,
                        HiddenSize = Config.HiddenSize,
                        NumHeads = headsPerRank,
                        NumKvHeads = kvHeadsPerRank,
                        HeadDim = Config.HeadDim,
                        CacheSize = 0,          // refreshed per call
                        IsSwa = (l % 2 == 0) ? 1 : 0,
                        SlidingWindow = _slidingWindow,
                        RopeNDims = Config.HeadDim,
                        OrigCtxLen = Config.OriginalContextLength,
                        KvCacheType = _kvCacheDtype.GgmlType(),
                        // GLOBAL expert count: the router runs over all of them and
                        // the kernel divides by tp_degree for the stacked shard.
                        NumExperts = _numExperts,
                        NumExpertsUsed = _numExpertsUsed,
                        SeparateQkv = 0,
                        QkvType = qkvQw.GgmlType,
                        KType = 0,
                        VType = 0,
                        OType = oQw.GgmlType,
                        GeType = gateW.GgmlType,
                        UeType = upW.GgmlType,
                        DeType = downW.GgmlType,
                        CpuMoe = 0,

                        Eps = Config.Eps,
                        RopeBase = Config.RopeBase,
                        RopeFreqScale = 1.0f / Config.RopeScale,
                        OaiAlpha = SiluAlpha,
                        OaiLimit = SiluLimit,
                    };
                }
            }

            if (!_isQkvFused)
                return TpFdBail("separate Q/K/V weights are not supported by the fused TP graph");

            _tpFdLayers = layers;
            _tpFdPlans = new IntPtr[tp];
            return true;
        }

        /// <summary>
        /// This rank's shard of a weight. CacheKey (not Data) identifies the
        /// rank-resident device copy, which is what the native binder caches on.
        /// </summary>
        private bool TryTpFdShard(string key, int rank, out QuantizedWeight weight)
        {
            weight = null;
            if (_tpQuantWeights.TryGetValue(key, out var shards) && shards != null &&
                rank < shards.Length && shards[rank] != null)
            {
                weight = shards[rank];
                return true;
            }
            return false;
        }

        /// <summary>
        /// Runs a prompt chunk as one graph per rank. Same plan protocol as the
        /// decode path; the per-op prefill chain it replaces measured 194 tok/s
        /// against 19483 on a single GPU.
        /// </summary>
        private unsafe bool TryGptOssFusedModelPrefillTP(Tensor hidden, int startPos, int seqLen)
        {
            if (_tpFpFailed) return false;
            if (!TpFusedModelDecodeAvailable()) return false;
            if (!TryBuildTpFdLayerDescs()) { _tpFdFailed = true; return false; }

            if (!_quantWeights.TryGetValue("output.weight", out var lmHead) &&
                !_quantWeights.TryGetValue("token_embd.weight", out lmHead))
            {
                _tpFpFailed = true;
                return TpFdBail("prefill: no quantized LM head");
            }
            if (!_weights.TryGetValue("output_norm.weight", out var finalNorm))
            {
                _tpFpFailed = true;
                return TpFdBail("prefill: no output_norm.weight");
            }

            EnsureFoldLogitsBuffer();

            int tp = TpDegree;
            int numLayers = Config.NumLayers;
            int cacheSize = (int)_tpKvCacheK[0][0].Sizes[1];
            if (cacheSize <= 0 || startPos + seqLen > cacheSize) return TpFdBail("prefill KV geometry");

            for (int r = 0; r < tp; r++)
            {
                for (int l = 0; l < numLayers; l++)
                {
                    _tpFdLayers[r][l].KCache = TensorComputePrimitives.GetStoragePointer(_tpKvCacheK[l][r]);
                    _tpFdLayers[r][l].VCache = TensorComputePrimitives.GetStoragePointer(_tpKvCacheV[l][r]);
                    _tpFdLayers[r][l].CacheSize = cacheSize;
                }
            }

            int previousRank = GgmlBasicOps.GetActiveRank();
            var planSlot = new IntPtr[1];
            var plans = new IntPtr[tp];
            try
            {
                for (int r = 0; r < tp; r++)
                {
                    GgmlBasicOps.SetActiveRank(r);
                    planSlot[0] = IntPtr.Zero;
                    bool ok = GgmlBasicOps.TryGptOssModelPrefillTP(
                        _tpFdLayers[r], numLayers,
                        (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, seqLen, startPos,
                        _foldLogitsPtr, Config.VocabSize,
                        lmHead.CacheKey, lmHead.GgmlType, lmHead.Ne0, lmHead.Ne1, lmHead.RawBytes,
                        (IntPtr)GetFloatPtr(finalNorm),
                        tp, planSlot);
                    if (!ok || planSlot[0] == IntPtr.Zero)
                    {
                        _tpFpFailed = true;
                        TpFdBail("the native prefill kernel declined in plan mode: "
                                 + GgmlBasicOps.LastNativeError());
                        return false;
                    }
                    plans[r] = planSlot[0];
                }

                GgmlBasicOps.TensorParallelExecutePlans(plans);
            }
            catch (InvalidOperationException ex)
            {
                _tpFpFailed = true;
                TpFdBail("prefill executor threw: " + ex.Message);
                return false;
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }

            if (!_tpFpBanner)
            {
                _tpFpBanner = true;
                Console.WriteLine(
                    $"  GPT-OSS fused tensor-parallel prefill active ({tp} GPUs, " +
                    $"{2 * numLayers} AllReduce points/chunk).");
            }

            _kvCacheHostDirty = true;
            return true;
        }

        private bool _tpFpFailed;
        private bool _tpFpBanner;

        /// <summary>
        /// Runs one decode token as one graph per rank. Returns false (leaving the
        /// caller on the per-op chain) whenever the kernel or the sharding declines.
        /// </summary>
        private unsafe bool TryGptOssFusedModelDecodeTP(Tensor hidden, int startPos)
        {
            if (!TpFusedModelDecodeAvailable()) return false;
            if (!TryBuildTpFdLayerDescs()) { _tpFdFailed = true; return false; }

            if (!_quantWeights.TryGetValue("output.weight", out var lmHead) &&
                !_quantWeights.TryGetValue("token_embd.weight", out lmHead))
            {
                _tpFdFailed = true;
                return false;
            }
            if (!_weights.TryGetValue("output_norm.weight", out var finalNorm))
            {
                _tpFdFailed = true;
                return false;
            }

            EnsureFoldLogitsBuffer();

            int tp = TpDegree;
            int numLayers = Config.NumLayers;
            int cacheSize = (int)_tpKvCacheK[0][0].Sizes[1];
            if (cacheSize <= 0 || startPos >= cacheSize) return TpFdBail("KV cache geometry");

            // The persistent graphs bake the KV window addresses, so a regrown
            // cache has to drop them before the next build.
            if (_tpFdBuiltCapacity >= 0 && _tpFdBuiltCapacity != cacheSize)
            {
                ResetFusedModelDecodeCache();
                _tpFdBuiltCapacity = cacheSize;
            }
            else if (_tpFdBuiltCapacity < 0)
            {
                _tpFdBuiltCapacity = cacheSize;
            }

            for (int r = 0; r < tp; r++)
            {
                for (int l = 0; l < numLayers; l++)
                {
                    _tpFdLayers[r][l].KCache = TensorComputePrimitives.GetStoragePointer(_tpKvCacheK[l][r]);
                    _tpFdLayers[r][l].VCache = TensorComputePrimitives.GetStoragePointer(_tpKvCacheV[l][r]);
                    _tpFdLayers[r][l].CacheSize = cacheSize;
                }
            }

            int previousRank = GgmlBasicOps.GetActiveRank();
            var planSlot = new IntPtr[1];
            try
            {
                for (int r = 0; r < tp; r++)
                {
                    GgmlBasicOps.SetActiveRank(r);
                    planSlot[0] = IntPtr.Zero;
                    bool ok = GgmlBasicOps.TryGptOssModelDecodeTP(
                        _tpFdLayers[r], numLayers,
                        (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, startPos,
                        _foldLogitsPtr, Config.VocabSize,
                        lmHead.CacheKey, lmHead.GgmlType, lmHead.Ne0, lmHead.Ne1, lmHead.RawBytes,
                        (IntPtr)GetFloatPtr(finalNorm),
                        tp, planSlot);
                    if (!ok || planSlot[0] == IntPtr.Zero)
                    {
                        _tpFdFailed = true;
                        TpFdBail("the native kernel declined in plan mode");
                        return false;
                    }
                    _tpFdPlans[r] = planSlot[0];
                }

                GgmlBasicOps.TensorParallelExecutePlans(_tpFdPlans);
            }
            catch (InvalidOperationException)
            {
                _tpFdFailed = true;
                return false;
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }

            if (!_tpFdBanner)
            {
                _tpFdBanner = true;
                Console.WriteLine(
                    $"  GPT-OSS fused tensor-parallel decode active ({tp} GPUs, " +
                    $"{2 * numLayers} AllReduce points/token).");
            }

            // The graph advanced the KV device-side only.
            _kvCacheHostDirty = true;
            return true;
        }
    }
}
