// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// LFM3's whole-model forward on ggml_cuda (TSGgml_Lfm3Forward, ggml_ops_lfm3.cpp):
// one native graph per forward, with TensorSharp's CUDA kernels for the short conv,
// the Q/K preparation and the decode attention.
//
// The K/V caches and conv state are then advanced on the device, and the host copies
// go stale. So while this path is on, whoever reads host state first pulls exactly
// the rows it wants (SyncCacheRows, SyncConvState) and whoever writes it pushes them
// back (UploadCacheRows, UploadConvState). The device copies stay where they are:
// dropping one would mean rebuilding the decode graph and re-uploading the cache.
//
// Under tensor parallelism every local rank gets the same graph over its own shard
// (TSGgml_Lfm3ForwardPlan), and the fused TP executor runs them, reducing after each
// row-parallel projection, across this node's ranks and then across nodes. The conv
// state's host copy keeps its one channel-major layout, for snapshots and the per-op
// path; a rank's device copy is bound to a small mirror of its own channels.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public sealed partial class Lfm3Model
    {
        // A cached device copy needs at least this much (the native bind's threshold); a
        // state any smaller would get a private copy per graph and lose its contents.
        private const int NativeMinStateFloats = 1024;
        private const int NativeMaxConvWidth = 32;
        // Bounds the prefill graph's activations; attention masks grow with it too.
        private const int NativeMaxTokensPerForward = 2048;

        private bool _nativeForward;
        // One entry per local rank; one in all on a single device.
        private Lfm3Layer[][] _nativeLayers;
        private Lfm3Args[] _nativeArgs;
        // [layer][rank], conv layers only: what each rank's device state is bound to.
        private float[][][] _tpConvMirror;
        private IntPtr[] _tpPlans;
        private GgmlBasicOps.CrossNodeAllReduce _tpCrossNodeReduce;
        private float[] _tpCrossNodeBuffer = Array.Empty<float>();
        private float[] _nativeFreqsFull;
        private float[] _nativeFreqsSliding;
        private int _nativeCapacity = -1;
        // Rows [0, this) of every K/V cache are current in host memory.
        private int _nativeHostRows;

        /// <summary>Whether forwards run on the native CUDA graph.</summary>
        internal bool UsesNativeForward => _nativeForward;

        private void InitNativeForward()
        {
            string refusal = _backend != BackendType.GgmlCuda ? $"{_backend} backend"
                : Environment.GetEnvironmentVariable("TS_LFM3_NATIVE") == "0" ? "TS_LFM3_NATIVE=0"
                : BuildNativeDescriptors();
            refusal = AgreeOnNativeForward(refusal);
            if (refusal != null)
            {
                if (_backend == BackendType.GgmlCuda)
                    Console.WriteLine($"lfm3: per-op CUDA path ({refusal}).");
                return;
            }
            _nativeForward = true;
            Console.WriteLine(IsTensorParallel
                ? $"lfm3: native CUDA forward enabled ({TpDegree} of {GlobalTpDegree} ranks here, {2 * Config.NumLayers} reductions per forward)."
                : "lfm3: native CUDA forward enabled.");
        }

        /// <summary>
        /// Nodes must take the same path: a native node's reductions never pair with a
        /// per-op node's, and the two would wait on each other for good. One node's
        /// refusal, for whatever reason, is therefore everyone's.
        /// </summary>
        private string AgreeOnNativeForward(string refusal)
        {
            if (TpNodeCount <= 1)
                return refusal;
            if (_tpGroup is not INestedTensorParallelGroup nested)
                return refusal ?? "no cross-node reduction";
            var votes = new[] { refusal == null ? 1f : 0f };
            nested.CrossNodeAllReduce(votes, 1);
            return refusal ?? (votes[0] == TpNodeCount ? null : "another node cannot run it");
        }

        /// <summary>The descriptors of every weight, or why this model has to stay per-op.</summary>
        private string BuildNativeDescriptors()
        {
            if (_kvCacheDtype != KvCacheDtype.F16 && _kvCacheDtype != KvCacheDtype.F32)
                return $"{_kvCacheDtype.ToShortString()} KV cache";
            if (_convWidth > NativeMaxConvWidth)
                return $"conv width {_convWidth}";
            int group = Config.NumHeads / Config.NumKVHeads;
            if (_headDim > 256 || (group != 1 && group != 2 && group != 4 && group != 8))
                return $"head dim {_headDim} with GQA ratio {group}";
            if (IsTensorParallel && !(TpNodeCount > 1
                    ? GgmlBasicOps.TensorParallelFusedAvailableDistributed(TpDegree)
                    : GgmlBasicOps.TensorParallelFusedAvailable(TpDegree)))
                return "the fused tensor-parallel executor is unavailable";

            int ranks = IsTensorParallel ? TpDegree : 1;
            int shards = IsTensorParallel ? GlobalTpDegree : 1;
            int channels = Config.HiddenSize / shards;
            int history = Math.Max(0, _convWidth - 1);
            int mirrorFloats = Math.Max(channels * history, NativeMinStateFloats);

            var mirrors = IsTensorParallel ? new float[Config.NumLayers][][] : null;
            var layers = new Lfm3Layer[ranks][];
            for (int r = 0; r < ranks; r++)
                layers[r] = new Lfm3Layer[Config.NumLayers];

            for (int l = 0; l < Config.NumLayers; l++)
            {
                string p = $"blk.{l}.";
                bool conv = _layerKinds[l] == LayerKind.Conv;
                if (conv && mirrors != null)
                {
                    mirrors[l] = new float[ranks][];
                    for (int r = 0; r < ranks; r++)
                        mirrors[l][r] = GC.AllocateArray<float>(mirrorFloats, pinned: true);
                }
                for (int r = 0; r < ranks; r++)
                {
                    ref Lfm3Layer d = ref layers[r][l];
                    d.StructBytes = Marshal.SizeOf<Lfm3Layer>();
                    d.Kind = (int)_layerKinds[l];
                    d.AttnNorm = WeightPointer(p + "attn_norm.weight");
                    d.FfnNorm = WeightPointer(p + "ffn_norm.weight");
                    if (_sandwichNorms)
                    {
                        d.PostAttnNorm = WeightPointer(p + "post_attention_norm.weight");
                        d.PostFfnNorm = WeightPointer(p + "post_ffw_norm.weight");
                    }
                    bool ok = TryRankWeight(p + "ffn_gate.weight", r, out d.FfnGate)
                        && TryRankWeight(p + "ffn_up.weight", r, out d.FfnUp)
                        && TryRankWeight(p + "ffn_down.weight", r, out d.FfnDown);
                    if (conv)
                    {
                        ok = ok && TryRankWeight(p + "shortconv.in_proj.weight", r, out d.InProj)
                            && TryRankWeight(p + "shortconv.out_proj.weight", r, out d.OutProj);
                        int firstChannel = (TpRankOffset + r) * channels;
                        d.ConvTaps = Marshal.UnsafeAddrOfPinnedArrayElement(_convWeights[l], firstChannel * _convWidth);
                        d.ConvState = Marshal.UnsafeAddrOfPinnedArrayElement(mirrors != null ? mirrors[l][r] : _convState[l], 0);
                    }
                    else
                    {
                        ok = ok && TryRankWeight(p + "attn_q.weight", r, out d.Q)
                            && TryRankWeight(p + "attn_k.weight", r, out d.K)
                            && TryRankWeight(p + "attn_v.weight", r, out d.V)
                            && TryRankWeight(p + "attn_gate.weight", r, out d.Gate)
                            && TryRankWeight(p + "attn_output.weight", r, out d.O);
                        d.QNorm = WeightPointer(p + "attn_q_norm.weight");
                        d.KNorm = WeightPointer(p + "attn_k_norm.weight");
                    }
                    if (!ok)
                        return $"layer {l} has a weight the native graph cannot bind";
                }
            }

            if (!TryWeight("output.weight", out Lfm3Weight head) && !TryWeight("token_embd.weight", out head))
                return "no usable LM head";

            _nativeFreqsFull = GC.AllocateArray<float>(_ropeHalf, pinned: true);
            _nativeFreqsSliding = GC.AllocateArray<float>(_ropeHalf, pinned: true);
            _ropeFreqsFull.CopyTo(_nativeFreqsFull, 0);
            _ropeFreqsSwa.CopyTo(_nativeFreqsSliding, 0);

            var args = new Lfm3Args[ranks];
            for (int r = 0; r < ranks; r++)
            {
                args[r] = new Lfm3Args
                {
                    StructBytes = Marshal.SizeOf<Lfm3Args>(),
                    Hidden = Config.HiddenSize,
                    NumHeads = Config.NumHeads / shards,
                    NumKVHeads = Config.NumKVHeads / shards,
                    HeadDim = _headDim,
                    KvType = _kvCacheDtype == KvCacheDtype.F16 ? (int)GgmlTensorType.F16 : (int)GgmlTensorType.F32,
                    ConvWidth = _convWidth,
                    Window = _slidingWindow,
                    RopeHalf = _ropeHalf,
                    Vocab = Config.VocabSize,
                    ConvStateFloats = IsTensorParallel ? mirrorFloats : ConvStateAllocationFloats,
                    ConvChannels = channels,
                    Eps = Config.Eps,
                    PositionScale = Config.RopeScale > 0f ? 1.0f / Config.RopeScale : 1.0f,
                    FreqsFull = Marshal.UnsafeAddrOfPinnedArrayElement(_nativeFreqsFull, 0),
                    FreqsSliding = Marshal.UnsafeAddrOfPinnedArrayElement(_nativeFreqsSliding, 0),
                    FinalNorm = WeightPointer("output_norm.weight"),
                    // The head runs once, on the driver's first rank: it is the one sampling.
                    LmHead = HasLmHead(r) ? head : default,
                };
            }

            _nativeLayers = layers;
            _nativeArgs = args;
            _tpConvMirror = mirrors;
            _tpPlans = new IntPtr[ranks];
            return RefreshNativeCaches();
        }

        private bool HasLmHead(int rank) => TpRankOffset + rank == 0;

        private int NativeCacheCapacity => IsTensorParallel ? _tpKvCacheCapacity : _kvCacheCapacity;

        private Tensor NativeKCache(int layer, int rank) => IsTensorParallel ? _tpKvCacheK[layer][rank] : _kvCacheK[layer];

        private Tensor NativeVCache(int layer, int rank) => IsTensorParallel ? _tpKvCacheV[layer][rank] : _kvCacheV[layer];

        /// <summary>Point the descriptors at the caches as they are now.</summary>
        private string RefreshNativeCaches()
        {
            for (int r = 0; r < _nativeLayers.Length; r++)
            {
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (_layerKinds[l] == LayerKind.Conv) continue;
                    Tensor k = NativeKCache(l, r);
                    // The bind needs a byte size worth caching; a test's tiny cache is not.
                    if (k.Storage.ByteLength < NativeMinStateFloats * sizeof(float))
                        return $"a {NativeCacheCapacity}-token cache is too small to keep on the device";
                    _nativeLayers[r][l].KCache = TensorComputePrimitives.GetStoragePointer(k);
                    _nativeLayers[r][l].VCache = TensorComputePrimitives.GetStoragePointer(NativeVCache(l, r));
                    _nativeArgs[r].CacheBytes = k.Storage.ByteLength;
                }
                _nativeArgs[r].CacheCapacity = NativeCacheCapacity;
            }
            _nativeCapacity = NativeCacheCapacity;
            return null;
        }

        private IntPtr WeightPointer(string name) => TensorComputePrimitives.GetStoragePointer(_weights[name]);

        private bool TryWeight(string name, out Lfm3Weight weight)
        {
            weight = default;
            if (_quantWeights.TryGetValue(name, out QuantizedWeight quantized))
                return TryDescribe(quantized, out weight);
            return _weights.TryGetValue(name, out Tensor tensor) && TryDescribe(tensor, out weight);
        }

        /// <summary>A projection: under TP this rank's shard, and never a whole weight that
        /// was left unsharded, whose output would be the wrong width for the rank.</summary>
        private bool TryRankWeight(string name, int rank, out Lfm3Weight weight)
        {
            if (!IsTensorParallel)
                return TryWeight(name, out weight);
            weight = default;
            if (_tpQuantWeights.TryGetValue(name, out QuantizedWeight[] quantized))
                return quantized[rank] != null && TryDescribe(quantized[rank], out weight);
            return _tpWeights.TryGetValue(name, out Tensor[] tensors) && tensors[rank] != null
                && TryDescribe(tensors[rank], out weight);
        }

        private static bool TryDescribe(QuantizedWeight quantized, out Lfm3Weight weight)
        {
            weight = default;
            if (quantized.Scale != 1.0f)
                return false;
            weight = new Lfm3Weight
            {
                Data = quantized.CacheKey,
                Type = quantized.GgmlType,
                Ne0 = quantized.Ne0,
                Ne1 = quantized.Ne1,
                Bytes = quantized.RawBytes,
            };
            return true;
        }

        private static bool TryDescribe(Tensor tensor, out Lfm3Weight weight)
        {
            weight = default;
            if (tensor.ElementType != DType.Float32 || tensor.Sizes.Length != 2)
                return false;
            weight = new Lfm3Weight
            {
                Data = TensorComputePrimitives.GetStoragePointer(tensor),
                Type = (int)GgmlTensorType.F32,
                Ne0 = tensor.Sizes[1],
                Ne1 = tensor.Sizes[0],
                Bytes = tensor.Storage.ByteLength,
            };
            return true;
        }

        private unsafe bool TryForwardNative(int[] tokens, out float[] logits)
        {
            logits = null;
            if (!_nativeForward)
                return false;

            if (IsTensorParallel)
                EnsureTpCacheCapacity(_cacheSeqLen + tokens.Length);
            else
                EnsureCacheCapacity(_cacheSeqLen + tokens.Length);
            if (_nativeCapacity != NativeCacheCapacity && RefreshNativeCaches() is string refusal)
            {
                DisableNativeForward(refusal);
                return false;
            }

            // A rank without the head reports no logits; a worker's are never read.
            bool hasHead = HasLmHead(0);
            float[] result = hasHead
                ? GC.AllocateUninitializedArray<float>(Config.VocabSize)
                : _logitsBuffer ?? new float[Config.VocabSize];
            for (int first = 0; first < tokens.Length; first += NativeMaxTokensPerForward)
            {
                int count = Math.Min(NativeMaxTokensPerForward, tokens.Length - first);
                int[] chunk = count == tokens.Length ? tokens : tokens.AsSpan(first, count).ToArray();
                using Tensor hidden = Embedding(chunk);
                bool ok;
                string error;
                fixed (float* output = result)
                {
                    ok = IsTensorParallel
                        ? RunNativePlans(GetFloatPtr(hidden), count, hasHead ? output : null, out error)
                        : GgmlBasicOps.Lfm3Forward(ref _nativeArgs[0], _nativeLayers[0], GetFloatPtr(hidden), count,
                            _cacheSeqLen, output, out error);
                }
                if (!ok)
                {
                    // Across nodes the others are already waiting in this forward's reductions.
                    if (TpNodeCount > 1)
                    {
                        var failure = new InvalidOperationException($"lfm3 native forward failed: {error}");
                        AbortDistributedModelControl(failure);
                        throw failure;
                    }
                    // A chunk that already ran stays in the cache; the rest takes the per-op path.
                    DisableNativeForward(error);
                    if (first == 0)
                        return false;
                    logits = ForwardCore(tokens.AsSpan(first).ToArray());
                    return true;
                }
                _cacheSeqLen += count;
            }
            _forwardCount++;
            logits = result;
            return true;
        }

        /// <summary>
        /// Build each local rank's plan, then run them together. Returns false only while
        /// nothing has executed. An execution failure throws: by then the ranks have
        /// advanced their caches and conv states by different amounts.
        /// </summary>
        private unsafe bool RunNativePlans(float* hidden, int count, float* output, out string error)
        {
            int previousRank = GgmlBasicOps.GetActiveRank();
            try
            {
                for (int r = 0; r < _tpPlans.Length; r++)
                {
                    GgmlBasicOps.SetActiveRank(r);
                    if (!GgmlBasicOps.Lfm3ForwardPlan(ref _nativeArgs[r], _nativeLayers[r], hidden, count, _cacheSeqLen,
                            HasLmHead(r) ? output : null, out _tpPlans[r], out error))
                        return false;
                }
                if (TpNodeCount > 1)
                    GgmlBasicOps.TensorParallelExecutePlansDistributed(_tpPlans, _tpCrossNodeReduce ??= ReduceAcrossNodes);
                else
                    GgmlBasicOps.TensorParallelExecutePlans(_tpPlans);
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }
            error = null;
            return true;
        }

        /// <summary>The executor's cross-node half of each reduction. An exception cannot
        /// cross into native code: returning false aborts the forward instead.</summary>
        private bool ReduceAcrossNodes(IntPtr user, IntPtr data, int count)
        {
            try
            {
                if (count <= 0)
                    return true;
                if (_tpCrossNodeBuffer.Length < count)
                    _tpCrossNodeBuffer = new float[count];
                Marshal.Copy(data, _tpCrossNodeBuffer, 0, count);
                ((INestedTensorParallelGroup)_tpGroup).CrossNodeAllReduce(_tpCrossNodeBuffer, count);
                Marshal.Copy(_tpCrossNodeBuffer, 0, data, count);
                return true;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"lfm3: cross-node reduction failed: {ex.Message}");
                return false;
            }
        }

        /// <summary>Hand the state back to the per-op path, which reads host memory.</summary>
        private void DisableNativeForward(string reason)
        {
            Console.WriteLine($"lfm3: leaving the native CUDA forward: {reason}");
            EnsureHostRows(_cacheSeqLen);
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerKinds[l] == LayerKind.Conv)
                {
                    SyncConvState(l);
                    InvalidateConvState(l);
                    continue;
                }
                foreach (Tensor cache in NativeCaches(l))
                    InvalidateTensorDeviceCache(cache);
            }
            _nativeForward = false;
        }

        /// <summary>Drop the device copies of every conv state: their host arrays go with the
        /// model, and a later allocation at one of those addresses must not find them.</summary>
        private void ReleaseNativeConvState()
        {
            if (_nativeLayers == null)
                return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerKinds[l] == LayerKind.Conv)
                    InvalidateConvState(l);
            }
        }

        private void InvalidateConvState(int layer)
        {
            for (int r = 0; r < _nativeLayers.Length; r++)
                GgmlBasicOps.InvalidateHostBuffer(_nativeLayers[r][layer].ConvState);
        }

        // ------------------------------------------------------------------ host <-> device

        private static void CacheRowRanges(Tensor cache, int start, int count, out long[] offsets, out long[] lengths)
        {
            int heads = (int)cache.Sizes[0];
            long capacity = cache.Sizes[1];
            long rowBytes = CacheRowBytes(cache);
            offsets = new long[heads];
            lengths = new long[heads];
            for (int h = 0; h < heads; h++)
            {
                offsets[h] = (h * capacity + start) * rowBytes;
                lengths[h] = count * rowBytes;
            }
        }

        /// <summary>
        /// Bring the host's K/V rows up to <paramref name="end"/>. It takes everything the
        /// device has beyond what the host holds, not just what was asked for: the engine
        /// extracts a prefill block by block, and row-by-row syncs of each block are ten
        /// thousand small copies per 2000 tokens where this is one per head and cache.
        /// </summary>
        private void EnsureHostRows(int end)
        {
            if (!_nativeForward || end <= _nativeHostRows) return;
            int count = _cacheSeqLen - _nativeHostRows;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerKinds[l] == LayerKind.Conv) continue;
                foreach (Tensor cache in NativeCaches(l))
                {
                    CacheRowRanges(cache, _nativeHostRows, count, out long[] offsets, out long[] lengths);
                    GgmlBasicOps.SyncHostBufferRanges(TensorComputePrimitives.GetStoragePointer(cache), offsets, lengths);
                }
            }
            _nativeHostRows = _cacheSeqLen;
        }

        private void UploadCacheRows(Tensor cache, int start, int count)
        {
            if (!_nativeForward || count <= 0) return;
            CacheRowRanges(cache, start, count, out long[] offsets, out long[] lengths);
            GgmlBasicOps.UploadHostBufferRanges(TensorComputePrimitives.GetStoragePointer(cache), offsets, lengths);
        }

        /// <summary>Every local rank's K and V cache of an attention layer.</summary>
        private IEnumerable<Tensor> NativeCaches(int layer)
        {
            for (int r = 0; r < _nativeLayers.Length; r++)
            {
                yield return NativeKCache(layer, r);
                yield return NativeVCache(layer, r);
            }
        }

        /// <summary>A rank's channels of the channel-major host state.</summary>
        private Span<float> RankConvState(int layer, int rank)
        {
            int floats = Config.HiddenSize / GlobalTpDegree * Math.Max(0, _convWidth - 1);
            return ConvStateSpan(layer).Slice((TpRankOffset + rank) * floats, floats);
        }

        private void SyncConvState(int layer)
        {
            if (!_nativeForward) return;
            for (int r = 0; r < _nativeLayers.Length; r++)
            {
                long bytes = IsTensorParallel ? (long)RankConvState(layer, r).Length * sizeof(float) : ConvStateBytes;
                GgmlBasicOps.SyncHostBufferRanges(_nativeLayers[r][layer].ConvState, new long[] { 0 }, new long[] { bytes });
                if (IsTensorParallel)
                    _tpConvMirror[layer][r].AsSpan(0, RankConvState(layer, r).Length).CopyTo(RankConvState(layer, r));
            }
        }

        private void UploadConvState(int layer)
        {
            if (!_nativeForward) return;
            for (int r = 0; r < _nativeLayers.Length; r++)
            {
                if (IsTensorParallel)
                    RankConvState(layer, r).CopyTo(_tpConvMirror[layer][r]);
                long bytes = IsTensorParallel ? (long)RankConvState(layer, r).Length * sizeof(float) : ConvStateBytes;
                GgmlBasicOps.UploadHostBufferRanges(_nativeLayers[r][layer].ConvState, new long[] { 0 }, new long[] { bytes });
            }
        }
    }
}
