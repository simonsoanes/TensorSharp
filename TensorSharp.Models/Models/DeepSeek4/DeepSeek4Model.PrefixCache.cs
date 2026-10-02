// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// DeepSeek V4 / V4.1's radix prefix cache: donate-only executor slots (native or
// direct-CUDA). The executor's reuse checks and retention budgets remain
// authoritative; reclaimed slots are reported to the tree through the attached
// payload sink.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class DeepSeek4Model : IHolderPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        // Set by AttachPrefixCache: reclaimed slots are reported through it (DEC-23).
        private IPrefixPayloadSink _prefixCacheSink;

        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        {
            bool slots = SupportsRetainedFusedCache && SupportsPerSequenceFusedForward;
            return new PrefixCacheCapabilities
            {
                Class = FamilyClass.N,
                NamespaceFingerprint = KVStateFingerprint,
                EndState = slots ? EndStateSupport.DonateOnly : EndStateSupport.None,
                CanCaptureCopy = false,
                AdoptPrimaryOnDisplacement = slots,
                PrimaryResident = true,
                MinRetainTokens = 32,
                // The executor's slot decides (SlotCanReuse), aligned to its compressor ratio; own scope only.
                Truncation = SupportsKVCacheTruncation ? TruncationKind.ModelDecides : TruncationKind.None,
                TruncationGranularity = Math.Max(1, KVCacheTruncationGranularity),
                RewindCapTokens = int.MaxValue,
                // Only a multi-token forward records the rewind checkpoint (ggml_ops_deepseek4.cpp,
                // Dsv4CudaEngine.Slots.cs), and a rewind truncates away a checkpoint past the new head:
                // forwarding at least two prompt tokens after any reuse leaves the next thinking turn a
                // checkpoint to rewind to.
                MinTailPrefillTokens = SupportsKVCacheTruncation ? 2 : 1,
                Pages = PageSupport.None,
                ReuseAcrossMediaSpan = false,
                Persistable = false,
                SubCapBytes = new ResourceVector
                {
                    NativeSlot = (long)Math.Min(_nativeRetentionBudget, (ulong)long.MaxValue),
                },
            };
        }

        public void AttachPrefixCache(IPrefixPayloadSink sink)
        {
            lock (_sync)
                _prefixCacheSink = sink ?? throw new ArgumentNullException(nameof(sink));
        }

        public void DetachPrefixCache()
        {
            lock (_sync)
                _prefixCacheSink = null;
        }

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            return SupportsRetainedFusedCache
                && HolderPrefixCacheAdapter.TryConvertPrimary(this, payloadKey, length, out footprint);
        }

        /// <summary>The executor's slot decides (<see cref="CanReuseRetainedPrefix"/>: its head is
        /// <paramref name="payloadTokens"/> and the rewind to <paramref name="targetTokens"/> is exact).</summary>
        public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
            => CanReuseRetainedPrefix(payloadKey, payloadTokens, targetTokens);

        /// <summary>The live primary slot decides (<c>SlotCanReuse</c>): its head must be
        /// <paramref name="cachedTokens"/> and the rewind exact - inside the raw ring, or served by the
        /// checkpoint the slot took at its last prompt boundary (dsv41_truncate.h). Every thinking turn
        /// needs the second kind: its render drops the previous answer's reasoning, so keeping the
        /// previous prompt means rewinding past the whole answer. Unlike <see cref="CanReuseLivePrefix"/>
        /// this does not need retention: the primary exists without it (DSpark loaded).</summary>
        public bool CanRewindPrimary(int cachedTokens, int targetTokens)
        {
            lock (_sync)
            {
                return _slotExecutor != null && _truncateAlign > 0
                    && _activeSlotKey == null && _selectedRetainedKey == null && _primarySlot >= 0
                    && _slotExecutor.SlotCanReuse(_primarySlot, cachedTokens, targetTokens);
            }
        }

        /// <summary>A slot is never copied, so there is nothing to settle and no clone to allow.</summary>
        public bool SettleForCopy(string payloadKey) => false;

        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            lock (_sync)
            {
                if (payloadKey == null || _retainedSlotByRequest == null
                    || !_retainedSlotByRequest.TryGetValue(payloadKey, out int slot)
                    || !_slotExecutor.SlotStatus(slot, out int head, out _, out _))
                    return default;
                // The slot's bytes are the executor's (a full-context cache set, plus a graph arena on the
                // native executor); SlotCanRetain's budget bounds retention.
                return new PayloadFootprint(head, _maxContextLength, default, PositionDelta: 0);
            }
        }

        public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens) => default;

        public void DiscardRetainedCaches(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            lock (_sync)
            {
                foreach (string key in payloadKeys)
                    if (key != null) DiscardRetainedCache(key);
            }
        }

        // ---------------------------------------------------------------- diagnostics

        public IReadOnlyCollection<string> RetainedPayloadKeys
        {
            get
            {
                lock (_sync)
                    return _retainedSlotByRequest == null ? Array.Empty<string>() : _retainedSlotByRequest.Keys.ToArray();
            }
        }

        public int PrivateHolderCount
        {
            get
            {
                lock (_sync)
                    return _slotByRequest?.Count ?? 0;
            }
        }

        public int PrimaryCacheLength
        {
            get
            {
                lock (_sync)
                {
                    return _slotExecutor != null && _primarySlot >= 0
                           && _slotExecutor.SlotStatus(_primarySlot, out int head, out _, out _)
                        ? head : 0;
                }
            }
        }
    }
}
