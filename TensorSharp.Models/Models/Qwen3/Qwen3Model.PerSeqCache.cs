// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Per-request KV-cache holders for the per-sequence fused-decode path
// (the Gemma 4 / Qwen 3.5 pattern, ported to Qwen 3 / Qwen 2.x).
//
// Same defect this fixes as GptOssModel.PerSeqCache.cs: without a
// PerSequenceFused candidate, ExecutionPlanner sent every N>=2 step to the
// op-by-op batched paged forward, whose per-layer GetElementsAsFloat round trips
// and managed float[][] K/V pool cost O(history x layers) of host<->device
// traffic per token. Concurrency then made throughput WORSE than single-stream.
//
// Here each in-flight request gets its own K/V tensors and the model switches
// between them with a pointer swap, so every scheduled sequence runs through the
// normal whole-model fused decode (TSGgml_TransformerModelDecode).
using System;
using System.Collections.Generic;

using TensorSharp;

namespace TensorSharp.Models
{
    public partial class Qwen3Model
    {
        private sealed class Qwen3KvCacheHolder
        {
            public Tensor[] K;
            public Tensor[] V;
            public int Capacity;
            public int SeqLen;
            public bool HostDirty;
        }

        private Dictionary<string, Qwen3KvCacheHolder> _fusedHolders;
        private string _activeFusedKey;
        private Qwen3KvCacheHolder _primaryHolder;
        private int _initialKvCacheLength = 512;

        /// <summary>Per-sequence fused decode: each concurrent request decodes
        /// through its own KV holder instead of the op-by-op batched paged path.
        /// Off under tensor parallelism (KV lives in the TP arrays) and on
        /// non-GGML backends.</summary>
        public bool SupportsPerSequenceFusedForward =>
            IsGgmlBackend && !IsTensorParallel && _kvCacheK != null;

        public bool HasFusedSequenceCache(string requestId)
            => requestId != null && _fusedHolders != null && _fusedHolders.ContainsKey(requestId);

        /// <summary>Re-point the whole-model decode kernel's cached per-layer K/V
        /// pointers at the currently active cache.
        ///
        /// <see cref="_modelDecodeArrays"/> caches raw storage pointers, and they
        /// are captured once at construction. Anything that REPLACES the cache
        /// tensors — a per-request holder swap, or EnsureCacheCapacity's growth —
        /// leaves the kernel reading the old (freed) allocation, so both call
        /// this.</summary>
        private void RefreshDecodeArraysKvCache()
        {
            var a = _modelDecodeArrays;
            if (a == null || _kvCacheK == null || _kvCacheV == null)
                return;
            int numLayers = Config.NumLayers;
            for (int l = 0; l < numLayers; l++)
            {
                if (_kvCacheK[l] == null || _kvCacheV[l] == null) continue;
                a.KCache[l] = TensorComputePrimitives.GetStoragePointer(_kvCacheK[l]);
                a.VCache[l] = TensorComputePrimitives.GetStoragePointer(_kvCacheV[l]);
            }
        }

        private Qwen3KvCacheHolder SnapshotActiveCache() => new Qwen3KvCacheHolder
        {
            K = _kvCacheK,
            V = _kvCacheV,
            Capacity = _kvCacheCapacity,
            SeqLen = _cacheSeqLen,
            HostDirty = _kvCacheHostDirty,
        };

        private void LoadCacheHolder(Qwen3KvCacheHolder h)
        {
            _kvCacheK = h.K;
            _kvCacheV = h.V;
            _kvCacheCapacity = h.Capacity;
            _cacheSeqLen = h.SeqLen;
            _kvCacheHostDirty = h.HostDirty;
            RefreshDecodeArraysKvCache();
        }

        /// <summary>Allocate one request's worth of per-layer K/V cache tensors,
        /// zero-filled so masked-but-read padding rows stay finite.</summary>
        private void AllocateKvCacheArrays(int capacity, out Tensor[] k, out Tensor[] v)
        {
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            DType kvDtype = _kvCacheDtype.ToDType();
            k = new Tensor[Config.NumLayers];
            v = new Tensor[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                k[l] = new Tensor(_allocator, kvDtype, numKVHeads, capacity, headDim);
                v[l] = new Tensor(_allocator, kvDtype, numKVHeads, capacity, headDim);
                InitializeCacheTensor(k[l]);
                InitializeCacheTensor(v[l]);
            }
        }

        private Qwen3KvCacheHolder CreateFreshHolder()
        {
            int capacity = Math.Max(_initialKvCacheLength, 1);
            AllocateKvCacheArrays(capacity, out var k, out var v);
            return new Qwen3KvCacheHolder
            {
                K = k,
                V = v,
                Capacity = capacity,
                SeqLen = 0,
                HostDirty = false,
            };
        }

        private void DisposeHolder(Qwen3KvCacheHolder h)
        {
            if (h?.K == null) return;
            for (int l = 0; l < h.K.Length; l++)
            {
                if (h.K[l] != null) { InvalidateTensorDeviceCache(h.K[l]); h.K[l].Dispose(); }
                if (h.V != null && h.V[l] != null) { InvalidateTensorDeviceCache(h.V[l]); h.V[l].Dispose(); }
            }
            h.K = null;
            h.V = null;
        }

        /// <summary>Make <paramref name="requestId"/>'s KV cache active, creating
        /// an empty one the first time. Returns true when freshly created.</summary>
        public bool BindSequenceCache(string requestId)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            _fusedHolders ??= new Dictionary<string, Qwen3KvCacheHolder>(StringComparer.Ordinal);

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
                return false;

            if (_activeFusedKey == null)
                _primaryHolder = SnapshotActiveCache();
            else
                _fusedHolders[_activeFusedKey] = SnapshotActiveCache();

            bool fresh;
            if (!_fusedHolders.TryGetValue(requestId, out var holder))
            {
                holder = CreateFreshHolder();
                _fusedHolders[requestId] = holder;
                fresh = true;
            }
            else
            {
                fresh = false;
            }

            LoadCacheHolder(holder);
            _activeFusedKey = requestId;
            return fresh;
        }

        /// <summary>Hand the live primary cache to the prior N==1 owner's holder
        /// (zero copy) and give the primary a fresh empty allocation.</summary>
        public void AdoptPrimaryCacheToFused(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            if (!SupportsPerSequenceFusedForward) return;
            _fusedHolders ??= new Dictionary<string, Qwen3KvCacheHolder>(StringComparer.Ordinal);

            if (_activeFusedKey != null) return;
            if (_fusedHolders.ContainsKey(requestId)) return;

            _fusedHolders[requestId] = SnapshotActiveCache();
            _activeFusedKey = requestId;

            int capacity = Math.Max(_initialKvCacheLength, 1);
            AllocateKvCacheArrays(capacity, out var k, out var v);
            _primaryHolder = new Qwen3KvCacheHolder
            {
                K = k,
                V = v,
                Capacity = capacity,
                SeqLen = 0,
                HostDirty = false,
            };
            // The active cache is still the adopted holder's, so the decode
            // pointers stay correct; nothing to refresh here.
        }

        /// <summary>Reinstate the primary cache before an N==1 step that follows
        /// a fused episode.</summary>
        public void RestorePrimaryCache()
        {
            if (_activeFusedKey == null) return;
            _fusedHolders[_activeFusedKey] = SnapshotActiveCache();
            _activeFusedKey = null;
            if (_primaryHolder != null)
            {
                LoadCacheHolder(_primaryHolder);
                _primaryHolder = null;
            }
        }

        /// <summary>Free a finished/aborted request's per-request cache.</summary>
        public void OnSequenceReleased(string requestId)
        {
            if (_fusedHolders == null || string.IsNullOrEmpty(requestId))
                return;
            if (!_fusedHolders.TryGetValue(requestId, out var holder))
                return;

            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
            {
                _activeFusedKey = null;
                if (_primaryHolder != null)
                {
                    LoadCacheHolder(_primaryHolder);
                    _primaryHolder = null;
                }
            }

            _fusedHolders.Remove(requestId);
            DisposeHolder(holder);
        }

        /// <summary>Free every per-request holder (model teardown).</summary>
        private void DisposeFusedSequenceCaches()
        {
            if (_fusedHolders != null)
            {
                foreach (var kv in _fusedHolders)
                    if (!string.Equals(_activeFusedKey, kv.Key, StringComparison.Ordinal))
                        DisposeHolder(kv.Value);
                _fusedHolders.Clear();
                _fusedHolders = null;
            }
            if (_primaryHolder != null)
            {
                DisposeHolder(_primaryHolder);
                _primaryHolder = null;
            }
            _activeFusedKey = null;
        }
    }
}
