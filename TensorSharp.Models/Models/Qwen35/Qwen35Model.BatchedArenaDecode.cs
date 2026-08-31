// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

// TRUE token-batched decode for the qwen35 family over the SLOT-STABLE ARENA
// kernel (ggml_ops_qwen35_batched_arena.cpp) — the GPT-OSS arena design ported
// to the hybrid GDN + attention architecture. One fused graph decodes one
// token for every concurrent request per step: weights are read once, the
// attention KV lives in persistent per-layer arenas written by one set_rows
// and read by one batched flash-attention, and the GDN conv/delta recurrent
// state lives in per-slot device arenas updated in-graph — no per-step host
// round trips, so the graph is CUDA-graph capturable and request churn
// replays one captured graph.
//
// The engine drives this through the same three IBatchedPagedModel hooks
// GPT-OSS implements: TryForwardBatchedFusedDecode (host logits),
// TryForwardBatchedFusedDecodeSampled (in-graph argmax) and CanBatchDecode
// (per-sequence growth/holder gate). Any decline falls back to the serial
// round-robin path, whose solo kernel entry flushes and retires arena slots
// through the native coherence hooks before touching a holder's caches.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        private string _lastArenaDeclineLogged;
        private float[] _arenaLogitsStaging;
        private static readonly bool ArenaPrefillVerifyEnabled =
            Environment.GetEnvironmentVariable("TS_QWEN35_PREFILL_VERIFY") != "0";

        /// <summary>A sequence can join the arena batch when its holder exists
        /// and needs no growth — one growing/unbound sequence falls to the
        /// round-robin loop alone instead of declining the whole batch.</summary>
        public bool CanBatchDecode(string requestId, int position)
        {
            if (_fusedHolders == null || !_fusedHolders.TryGetValue(requestId, out var h) || h.K == null)
                return false;
            return position + 1 <= h.KvCapacity;
        }

        public unsafe bool TryForwardBatchedFusedDecode(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits)
            => ForwardArenaBatchedDecodeCore(requestIds, tokens, positions, outLogits, null);

        public unsafe bool TryForwardBatchedFusedDecodeSampled(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, int[] outNextTokens)
            => ForwardArenaBatchedDecodeCore(requestIds, tokens, positions, null, outNextTokens);

        // Convert a holder's host conv RING (rotating write index) into the
        // ggml [time, channel] layout the fused kernels use, in the holder's
        // own ConvScratch. Identical to the solo decode's reseed transform —
        // after it, the scratch bytes are the state's source of truth and the
        // ring is no longer consulted (FdStateResident = true).
        private unsafe void ConvertHolderRingToScratch(Qwen35KvCacheHolder h)
        {
            int convDim = _convKernel - 1;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int convBlock = convDim * qkvDim;
            float* convBase = (float*)h.ConvScratch;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l]) continue;
                float* conv = convBase + (long)_fdGdnSlot[l] * convBlock;
                float[] ring = h.ConvState[l];
                int w = h.ConvWriteIdx[l];
                for (int t = 0; t < convDim; t++)
                {
                    int slot = (w + t) % convDim;
                    int srcBase = slot * qkvDim;
                    for (int ch = 0; ch < qkvDim; ch++)
                        conv[ch * convDim + t] = ring[srcBase + ch];
                }
                // The resident device copies (if any) predate this rewrite;
                // the solo reseed invalidates both keys, mirror it exactly.
                GgmlBasicOps.InvalidateHostBuffer((IntPtr)conv);
                GgmlBasicOps.InvalidateHostBuffer(GdnDeltaStatePointer(h.DeltaState[l]));
            }
        }

        private bool ArenaDecline(string reason)
        {
            if (reason != _lastArenaDeclineLogged)
            {
                _lastArenaDeclineLogged = reason;
                Console.Error.WriteLine($"[qwen35 arena-decode] declined: {reason}");
            }
            return false;
        }

        private unsafe bool ForwardArenaBatchedDecodeCore(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions,
            float[][] outLogits, int[] outNextTokens)
        {
            if (_backend != BackendType.GgmlCuda || IsTensorParallel || _fusedHolders == null)
                return false;
            if (!_fullDecodeEnabled || _fdUnsupported || _fdSpecSessionActive)
                return false;
            int n = requestIds.Count;
            if (n < 2 || tokens.Length != n || positions.Length != n)
                return false;
            // The slot-invariant weight descriptors are built by the solo fused
            // decode; the first token of the first sequence always warms them
            // through the round-robin path, so declining here costs one step.
            if (_fdLayers == null)
                return false;
            if (MoeCpuOffloadConfig.IsEnabled)
                return ArenaDecline("MoE CPU offload active");
            if (!ArenaPrefillVerifyEnabled)
                return ArenaDecline("TS_QWEN35_PREFILL_VERIFY=0 (unhooked prefill path)");
            DType kvDt = _kvCacheDtype.ToDType();
            if (kvDt != DType.Float32 && kvDt != DType.Float16)
                return ArenaDecline($"KV cache dtype {_kvCacheDtype} (arena supports F32/F16)");
            if (_headKDim != _headVDim || _convKernel <= 1)
                return ArenaDecline("unsupported GDN geometry");

            // Token embedding for the in-graph get_rows.
            (IntPtr ptr, int type, long ne0, long ne1, long bytes) emb;
            if (_quantWeights.TryGetValue("token_embd.weight", out QuantizedWeight tokenQw))
            {
                if (!CanUseGgmlQuantizedGetRows(tokenQw.GgmlType))
                    return ArenaDecline($"token embedding type {tokenQw.GgmlType} lacks CUDA get_rows");
                emb = ResolveW(tokenQw, null);
            }
            else if (_weights.TryGetValue("token_embd.weight", out Tensor tokenF32))
            {
                emb = ResolveW(null, tokenF32);
            }
            else
            {
                return ArenaDecline("token embedding weight missing");
            }
            if (emb.ptr == IntPtr.Zero || emb.ne0 != Config.HiddenSize)
                return ArenaDecline("token embedding shape mismatch");

            var lmh = ResolveW(_lmHeadQW, _lmHeadF32);
            if (lmh.ptr == IntPtr.Zero || _finalNormW == null)
                return ArenaDecline("folded lm_head/final norm missing");

            // Check any checked-out holder back in FIRST: the dict entry for a
            // checked-out request is a stale pre-checkout snapshot (Restore
            // replaces it with a current one), and reading FdStateResident or
            // writing CacheSeqLen through a stale object rolls recurrent state
            // back / desyncs positions.
            RestorePrimaryCache();

            var holders = new Qwen35KvCacheHolder[n];
            for (int i = 0; i < n; i++)
            {
                if (!_fusedHolders.TryGetValue(requestIds[i], out holders[i]) || holders[i].K == null)
                    return false;
                if (positions[i] + 1 > holders[i].KvCapacity)
                    return false;   // growth is the round-robin path's job
                if (tokens[i] < 0 || tokens[i] >= emb.ne1)
                    return false;
            }

            // Canonical order: ascending first-attention-layer K storage pointer.
            // With the native registry keyed on the same pointers, a stable set
            // of in-flight holders keeps replaying its captured graph no matter
            // which requests occupy them.
            int firstAttn = -1;
            for (int l = 0; l < Config.NumLayers; l++)
                if (!_isRecurrent[l]) { firstAttn = l; break; }
            if (firstAttn < 0)
                return ArenaDecline("no attention layers");

            var order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            var keys = new ulong[n];
            for (int i = 0; i < n; i++)
                keys[i] = (ulong)TensorComputePrimitives.GetStoragePointer(holders[i].K[firstAttn]).ToInt64();
            Array.Sort(keys, order);

            int numLayers = Config.NumLayers;
            int attnLayers = 0, gdnLayers = 0;
            for (int l = 0; l < numLayers; l++)
                if (_isRecurrent[l]) gdnLayers++; else attnLayers++;

            int convDim = _convKernel - 1;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            long convBlockBytes = (long)convDim * qkvDim * sizeof(float);

            var tokSorted = new int[n];
            var posSorted = new int[n];
            var cacheSizes = new int[n];
            var gdnHostAuth = new int[n];
            var kPtrs = new IntPtr[attnLayers * n];
            var vPtrs = new IntPtr[attnLayers * n];
            var convPtrs = new IntPtr[gdnLayers * n];
            var deltaPtrs = new IntPtr[gdnLayers * n];
            for (int i = 0; i < n; i++)
            {
                var h = holders[order[i]];
                tokSorted[i] = tokens[order[i]];
                posSorted[i] = positions[order[i]];
                cacheSizes[i] = h.KvCapacity;
                if (!h.FdStateResident)
                {
                    // Host ring is the GDN truth: land it in the scratch layout
                    // once. The residency flip is committed only after the
                    // native call succeeds (conversion is idempotent), so a
                    // decline leaves the solo fallback's reseed contract whole.
                    ConvertHolderRingToScratch(h);
                    gdnHostAuth[i] = 1;
                }
                for (int l = 0, al = 0, gl = 0; l < numLayers; l++)
                {
                    if (!_isRecurrent[l])
                    {
                        kPtrs[al * n + i] = TensorComputePrimitives.GetStoragePointer(h.K[l]);
                        vPtrs[al * n + i] = TensorComputePrimitives.GetStoragePointer(h.V[l]);
                        al++;
                    }
                    else
                    {
                        convPtrs[gl * n + i] = h.ConvScratch + (nint)((long)_fdGdnSlot[l] * convBlockBytes);
                        deltaPtrs[gl * n + i] = GdnDeltaStatePointer(h.DeltaState[l]);
                        gl++;
                    }
                }
            }

            int vocab = Config.VocabSize;
            bool wantLogits = outLogits != null;
            long needed = (long)vocab * n;
            if (_arenaLogitsStaging == null || _arenaLogitsStaging.LongLength < needed)
                _arenaLogitsStaging = new float[needed];
            var logitsBuf = _arenaLogitsStaging;
            var sampledSorted = outNextTokens != null ? new int[n] : null;

            bool ok;
            fixed (float* lp = logitsBuf)
            fixed (int* sp = sampledSorted)
            {
                ok = GgmlBasicOps.TryQwen35ArenaDecodeBatched(
                    _fdLayers, numLayers, n,
                    tokSorted, posSorted,
                    kPtrs, vPtrs, convPtrs, deltaPtrs,
                    gdnHostAuth, cacheSizes,
                    Config.NumHeads, Config.NumKVHeads, Config.HeadDim,
                    _ropeDimCount > 0 ? _ropeDimCount : Config.HeadDim, 2,
                    FusedGraphKvCacheTypeId(kvDt),
                    _convKernel, _headKDim, _headVDim, _numKHeads, _numVHeads,
                    Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                    _numExperts, _numExpertsUsed, _expertFfnLength, _sharedExpertFfnLength,
                    _normTopKProb ? 1 : 0, 1.0f,
                    (IntPtr)lp, vocab,
                    lmh.ptr, lmh.type, lmh.ne0, lmh.ne1, lmh.bytes,
                    TensorComputePrimitives.GetStoragePointer(_finalNormW),
                    emb.ptr, emb.type, emb.ne0, emb.ne1, emb.bytes,
                    (IntPtr)sp, wantLogits);
            }
            if (!ok)
            {
                string err = GgmlBasicOps.LastNativeError();
                if (err != _lastArenaDeclineLogged)
                {
                    _lastArenaDeclineLogged = err;
                    Console.Error.WriteLine($"[qwen35 arena-decode] native declined (n={n}): {err}");
                }
                return false;
            }

            for (int i = 0; i < n; i++)
            {
                var h = holders[order[i]];
                if (gdnHostAuth[i] != 0)
                    h.FdStateResident = true;   // arena/scratch chain is now the truth
                if (wantLogits)
                {
                    h.Logits ??= new float[vocab];
                    Array.Copy(logitsBuf, (long)i * vocab, h.Logits, 0, vocab);
                    outLogits[order[i]] = h.Logits;
                }
                if (outNextTokens != null)
                    outNextTokens[order[i]] = sampledSorted[i];
                h.CacheSeqLen = posSorted[i] + 1;
                h.KvHostDirty = true;
                h.GdnHostDirty = true;
            }
            return true;
        }
    }
}
