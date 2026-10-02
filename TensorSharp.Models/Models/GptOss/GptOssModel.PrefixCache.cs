// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// GPT-OSS's radix prefix cache: pages, the resident primary cache, and each
// finished request's holder donated to the turn that continues it. The sliding
// window masks a linear cache, so a supported rewind is exact.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class GptOssModel : IHolderPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        private IPrefixPayloadSink _prefixCacheSink;

        /// <summary>
        /// Pages as before, and donated holders wherever requests run on holders. Concurrent requests decode
        /// on per-request holders, which read none of the pages, so without end states a conversation that
        /// ran beside others reused nothing on its next turn: 0 of 272-519 tokens for every one of eight
        /// parallel gpt-oss-20b conversations. Holders are donated, never copied: nothing clones them.
        /// The harmony template drops the analysis channel when it re-renders a past answer, so continuing
        /// a conversation rewinds its whole previous answer; the 16-token cap refused every such turn. The
        /// sliding window masks a linear cache, so the rewind is exact at any depth (a 220-token rewind is
        /// bitwise identical to never holding the answer, GptOssHolderRewindTests).
        /// </summary>
        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        {
            PrefixCacheCapabilities pages = PageFamilyCapabilities(FamilyClass.P, TruncationKind.Any, reuseAcrossMediaSpan: true);
            if (!SupportsRetainedFusedCache) return pages;
            return pages with
            {
                EndState = EndStateSupport.DonateOnly,
                AdoptPrimaryOnDisplacement = true,
                MinRetainTokens = 32,
                RewindCapTokens = int.MaxValue,
            };
        }

        public void AttachPrefixCache(IPrefixPayloadSink sink)
            => _prefixCacheSink = sink ?? throw new ArgumentNullException(nameof(sink));

        public void DetachPrefixCache() => _prefixCacheSink = null;

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        // ---------------------------------------------------------------- end states

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            return SupportsRetainedFusedCache
                && HolderPrefixCacheAdapter.TryConvertPrimary(this, payloadKey, length, out footprint);
        }

        /// <summary>Holder present, holding exactly <paramref name="payloadTokens"/> tokens, and a rewind (if
        /// any) the linear cache allows.</summary>
        public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
        {
            if (!TryGetRetained(payloadKey, out var holder)) return false;
            if (holder.SeqLen != payloadTokens || targetTokens < 0 || targetTokens > payloadTokens) return false;
            return targetTokens == payloadTokens || CanTruncateKVCache(payloadTokens, targetTokens);
        }

        /// <summary>Holders are donated, never cloned.</summary>
        public bool SettleForCopy(string payloadKey) => false;

        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            if (!TryGetRetained(payloadKey, out var holder)) return default;
            long bytes = HolderBytes(holder);
            var vector = new ResourceVector { HostKv = bytes, DeviceKv = KeepsDeviceKvMirrors ? bytes : 0 };
            return new PayloadFootprint(holder.SeqLen, holder.Capacity, vector, PositionDelta: 0);
        }

        public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens) => default;

        /// <summary>Release retained holders to the pool, or free them when the prefix cache needs the memory
        /// or the holder is no longer valid; the captured decode graphs are reset at most once.</summary>
        public void DiscardRetainedCaches(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            if (_retainedFusedHolders == null || payloadKeys.IsEmpty) return;
            List<GptOssKvCacheHolder> released = null;
            foreach (string key in payloadKeys)
            {
                if (key != null && _retainedFusedHolders.Remove(key, out var holder))
                    (released ??= new List<GptOssKvCacheHolder>()).Add(holder);
            }
            if (released == null) return;
            bool dispose = reason is ReleaseReason.Invalidated or ReleaseReason.Pressure or ReleaseReason.Reset;
            RecycleOrDisposeHolders(released, dispose);
        }

        /// <summary>Σ bytes of the holder's K and V storages.</summary>
        private static long HolderBytes(GptOssKvCacheHolder holder)
        {
            long bytes = 0;
            foreach (Tensor[] set in new[] { holder.K, holder.V })
                if (set != null)
                    foreach (Tensor t in set)
                        if (t != null)
                            bytes = checked(bytes + t.Storage.ByteLength);
            return bytes;
        }

        private bool TryGetRetained(string payloadKey, out GptOssKvCacheHolder holder)
        {
            holder = null;
            return payloadKey != null && _retainedFusedHolders != null
                && _retainedFusedHolders.TryGetValue(payloadKey, out holder)
                && holder.K != null;
        }

        // ---------------------------------------------------------------- diagnostics

        public IReadOnlyCollection<string> RetainedPayloadKeys =>
            _retainedFusedHolders == null ? Array.Empty<string>() : _retainedFusedHolders.Keys.ToArray();

        public int PrivateHolderCount => _fusedHolders?.Count ?? 0;

        public int PrimaryCacheLength => _activeFusedKey == null ? _cacheSeqLen : (_primaryHolder?.SeqLen ?? 0);
    }
}
