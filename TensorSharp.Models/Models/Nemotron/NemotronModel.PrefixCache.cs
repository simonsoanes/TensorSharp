// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron-H's radix prefix cache: host pages with Mamba state at restorable
// forward boundaries, the resident primary cache, and on the batched route each
// finished sequence's Mamba2 slot, kept beside its pool blocks for the
// conversation's next turn (PrefixCacheCapabilities.PagedEndStates). No rewinds.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class NemotronModel : IPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        // payload key -> the Mamba2 slot of a finished batched sequence and the tokens its state has seen.
        private readonly Dictionary<string, (int Slot, int Tokens)> _nemoRetainedMambaSlots = new(StringComparer.Ordinal);

        /// <summary>Pages, and batched end states where the batched route serves this instance: a loaded
        /// model whose float K/V a sequence can move onto the route (<see cref="SupportsLinearKVMigration"/>).</summary>
        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        {
            PrefixCacheCapabilities pages = PageFamilyCapabilities(FamilyClass.R, TruncationKind.None, pagesNeedStateAtEnd: true);
            return BatchedForwardAvailable && SupportsLinearKVMigration
                ? pages with { EndState = EndStateSupport.DonateOnly, PagedEndStates = true }
                : pages;
        }

        public void AttachPrefixCache(IPrefixPayloadSink sink) { }

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        // ---------------------------------------------------------------- end states (the batched route)

        /// <summary>Keep <paramref name="requestId"/>'s Mamba2 slot under <paramref name="payloadKey"/>: the
        /// prefix cache keeps the sequence's pool blocks, and the release that follows no longer frees the slot.</summary>
        public bool TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            if (string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(payloadKey) || length <= 0
                || _nemoRetainedMambaSlots.ContainsKey(payloadKey)
                || !_nemoMambaSlotByReqId.Remove(requestId, out int slot))
                return false;
            _nemoRetainedMambaSlots.Add(payloadKey, (slot, length));
            footprint = MeasureEndState(payloadKey);
            return true;
        }

        /// <summary>Give a kept slot to the request continuing the conversation, at exact length.</summary>
        public bool TryMaterialize(in MaterializeRequest request)
        {
            if (request.Op != MaterializeOp.Donate || string.IsNullOrEmpty(request.TargetRequestId)
                || !CanMaterialize(request.PayloadKey, request.PayloadTokens, request.TargetTokens)
                || _nemoMambaSlotByReqId.ContainsKey(request.TargetRequestId))
                return false;
            _nemoMambaSlotByReqId[request.TargetRequestId] = _nemoRetainedMambaSlots[request.PayloadKey].Slot;
            _nemoRetainedMambaSlots.Remove(request.PayloadKey);
            return true;
        }

        /// <summary>The recurrent state cannot rewind: only an exact continuation.</summary>
        public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
            => payloadKey != null && _nemoRetainedMambaSlots.TryGetValue(payloadKey, out var kept)
               && kept.Tokens == payloadTokens && targetTokens == payloadTokens;

        public void ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            foreach (string key in payloadKeys)
            {
                if (key != null && _nemoRetainedMambaSlots.Remove(key, out var kept))
                    FreeMambaSlot(kept.Slot);
            }
        }

        /// <summary>The slot's conv ring and SSM state across the Mamba2 layers; the pool blocks are the
        /// prefix cache's to count.</summary>
        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            if (payloadKey == null || !_nemoRetainedMambaSlots.TryGetValue(payloadKey, out var kept)) return default;
            long convFloats = (long)Math.Max(0, _ssmDConv - 1) * (_ssmDInner + 2 * _ssmNGroup * _ssmDState);
            long ssmFloats = (long)_ssmDState * _ssmHeadDim * _ssmNHead;
            int mambaLayers = _layerTypes?.Count(t => t == LayerType.Mamba2) ?? 0;
            var bytes = new ResourceVector { StateSnapshot = mambaLayers * (convFloats + ssmFloats) * sizeof(float) };
            return new PayloadFootprint(kept.Tokens, kept.Tokens, bytes, PositionDelta: 0);
        }

        public bool TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
        {
            footprint = default;
            return false;
        }

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            return false;
        }

        public bool TryReturnDonation(string requestId, string payloadKey) => false;
        public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens) => default;
        public bool TryCopyPagedToHolder(ReadOnlySpan<int> blockIds, int tokens, string requestId) => false;
        public bool TryExport(string payloadKey, Stream destination) => false;

        public bool TryImport(string payloadKey, int tokens, Stream source, out PayloadFootprint footprint)
        {
            footprint = default;
            return false;
        }

        public bool TryBeginImport(int tokens, out object importTicket)
        {
            importTicket = null;
            return false;
        }

        public bool RunImportRead(object importTicket, Stream source) => false;

        public bool TryCommitImport(object importTicket, string payloadKey, out PayloadFootprint footprint)
        {
            footprint = default;
            return false;
        }

        public void AbortImport(object importTicket) { }

        // ---------------------------------------------------------------- diagnostics

        public IReadOnlyCollection<string> RetainedPayloadKeys => _nemoRetainedMambaSlots.Keys.ToArray();

        public int PrivateHolderCount => 0;

        public int PrimaryCacheLength => _cacheSeqLen;
    }
}
