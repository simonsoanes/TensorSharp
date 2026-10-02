// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class DeepSeek4Model : IExactFusedCacheReuse
    {
        // Retained conversations in the order they were retained, so the first is the one
        // whose next turn has waited longest: the one to release when a request needs room.
        private OrderedDictionary<string, int> _retainedSlotByRequest;
        // Non-null means native selection belongs to this retained entry, NOT
        // the primary. _activeSlotKey is null in that state. The primary may be
        // absent after the previous request adopted it.
        private string _selectedRetainedKey;
        private readonly ulong _nativeRetentionBudget = RetentionBudgetFromEnvironment();

        // Ownership has already committed when these diagnostics run. A broken
        // stderr sink or exhausted formatting allocation must not report a
        // successful transfer as a refusal to the scheduler.
        private static void TraceRetainedCommit(string operation, string request, int slot, int count, string source = null)
        {
            try
            {
                if (operation == "retain")
                    Console.Error.WriteLine($"[dsv41 retained] retain request={request} slot={slot} count={count}");
                else if (operation == "rebind")
                    Console.Error.WriteLine($"[dsv41 retained] rebind source={source} request={request} slot={slot}");
                else
                    Console.Error.WriteLine($"[dsv41 retained] discard request={request} primary={slot} count={count}");
            }
            catch (OutOfMemoryException) { }
            catch (System.IO.IOException) { }
            catch (ObjectDisposedException) { }
        }

        internal const ulong DefaultRetentionBudgetBytes = 2048UL * 1024 * 1024;

        /// <summary>
        /// <c>TS_DSV41_RETAINED_CACHE_MB</c>: the per-device budget for retained slots (about 110 MB each for
        /// V4.1 at a 64k context). Retention has no off switch: without it every conversation whose turns
        /// overlapped another's re-prefilled (0 tokens reused on each turn of four concurrent chats), so an
        /// unparsable or zero value keeps the default and says so.
        /// </summary>
        internal static ulong RetentionBudgetFromEnvironment(string text)
        {
            if (text == null) return DefaultRetentionBudgetBytes;
            if (ulong.TryParse(text, out ulong mb) && mb > 0 && mb <= ulong.MaxValue / (1024 * 1024))
                return mb * 1024 * 1024;
            Console.Error.WriteLine(
                $"[dsv41 retained] TS_DSV41_RETAINED_CACHE_MB='{text}' is not a positive size in MB; " +
                $"retention stays on with the {DefaultRetentionBudgetBytes >> 20} MB default.");
            return DefaultRetentionBudgetBytes;
        }

        private static ulong RetentionBudgetFromEnvironment()
            => RetentionBudgetFromEnvironment(Environment.GetEnvironmentVariable("TS_DSV41_RETAINED_CACHE_MB"));

        /// <summary>Retention is always on where the executor can keep a slot: it needs a rewinding
        /// executor with slots (V4.1 on the native or direct-CUDA one) and no DSpark drafter (the
        /// slot status declines with one loaded).</summary>
        public bool SupportsRetainedFusedCache
            => _slotExecutor != null && _truncateAlign > 0 && DraftBlockSize == 0;
        public bool SupportsExactFusedCacheReuse => SupportsRetainedFusedCache;

        internal interface INativeSlotRetention : INativeSlotRelease
        {
            bool Status(int slot, out int head, out bool healthy);
            bool CanRetain(int slot, int retainedCount, ulong budget);
            bool ReleaseGraphs(int slot);
        }

        /// <summary>The whole slot store: retention plus allocation.</summary>
        internal interface INativeSlotStore : INativeSlotRetention
        {
            int Alloc();
            bool CanAlloc();
        }

        private readonly struct NativeSlotRetention : INativeSlotStore
        {
            private readonly IDsv4SlotExecutor _slots;
            public NativeSlotRetention(IDsv4SlotExecutor slots) => _slots = slots;
            public int Alloc() => _slots.SlotAlloc();
            public bool CanAlloc() => _slots.SlotCanAlloc();
            public bool Reset() => _slots.ResetChecked();
            public bool Select(int slot) => _slots.SetActiveSlot(slot);
            public bool Free(int slot) => _slots.SlotFree(slot);
            public bool Status(int slot, out int head, out bool healthy)
                => _slots.SlotStatus(slot, out head, out _, out healthy);
            public bool CanRetain(int slot, int retainedCount, ulong budget)
                => _slots.SlotCanRetain(slot, retainedCount, budget);
            public bool ReleaseGraphs(int slot) => _slots.SlotReleaseGraphs(slot);
        }

        internal static bool RetainNativeSequence<TNative>(Dictionary<string, int> requests,
            IDictionary<string, int> retained, string key, ref string active, ref string selectedRetained,
            ulong budget, TNative native) where TNative : INativeSlotRetention
            => RetainNativeSequence(requests, retained, key, key, ref active, ref selectedRetained, budget, native);

        /// <summary>The key-parameterised retain: <paramref name="requestId"/>'s slot is retained under
        /// <paramref name="retainedKey"/> (the prefix cache's payload key, or the request id itself).</summary>
        internal static bool RetainNativeSequence<TNative>(Dictionary<string, int> requests,
            IDictionary<string, int> retained, string requestId, string retainedKey, ref string active,
            ref string selectedRetained, ulong budget, TNative native) where TNative : INativeSlotRetention
        {
            if (string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(retainedKey) || requests == null
                || !requests.TryGetValue(requestId, out int slot)
                || retained.ContainsKey(retainedKey) || !native.Status(slot, out int head, out bool healthy)
                || !healthy || head <= 0 || !native.CanRetain(slot, retained.Count, budget)) return false;
            // Add first: a managed allocation failure leaves the old owner intact.
            try { retained.Add(retainedKey, slot); }
            catch (OutOfMemoryException) { return false; }
            requests.Remove(requestId);
            if (active == requestId) { active = null; selectedRetained = retainedKey; }
            return true;
        }

        internal static bool RebindNativeSequence<TNative>(Dictionary<string, int> requests,
            IDictionary<string, int> retained, string oldKey, string newKey,
            ref string active, ref string selectedRetained, TNative native)
            where TNative : INativeSlotRetention
        {
            if (string.IsNullOrEmpty(oldKey) || string.IsNullOrEmpty(newKey) || retained == null
                || !retained.TryGetValue(oldKey, out int slot) || requests.ContainsKey(newKey)
                || !native.Status(slot, out _, out bool healthy) || !healthy) return false;
            try { requests.Add(newKey, slot); }
            catch (OutOfMemoryException) { return false; }
            retained.Remove(oldKey);
            if (selectedRetained == oldKey) { selectedRetained = null; active = newKey; }
            return true;
        }

        internal static void DiscardRetainedNativeSequence<TNative>(IDictionary<string, int> retained,
            string key, ref int primary, ref string active, ref string selectedRetained, TNative native)
            where TNative : INativeSlotRetention
        {
            if (string.IsNullOrEmpty(key) || retained == null || !retained.TryGetValue(key, out int slot)) return;
            if (selectedRetained == key)
            {
                if (primary < 0)
                {
                    // An active native slot cannot be freed. Reclaim it without
                    // allocating a replacement; release its captured arenas first.
                    if (!native.ReleaseGraphs(slot)) throw new InvalidOperationException("DSV4 retained graph release failed.");
                    if (!native.Reset())
                        throw new InvalidOperationException("DSV4 retained reset failed; slot remains quarantined.");
                    if (!native.Status(slot, out int head, out bool healthy) || !healthy || head != 0)
                        throw new InvalidOperationException("DSV4 retained slot reset failed; slot remains quarantined.");
                    primary = slot;
                    selectedRetained = null;
                    active = null;
                    retained.Remove(key);
                    return;
                }
                if (!native.Select(primary)) throw new InvalidOperationException("DSV4 retained eviction could not select primary.");
                selectedRetained = null;
                active = null;
            }
            if (!native.Free(slot)) throw new InvalidOperationException("DSV4 retained slot could not be freed.");
            retained.Remove(key);
        }

        public bool RetainSequenceCache(string requestId) => RetainSequenceCacheAs(requestId, requestId);

        /// <summary>The key-parameterised form of <see cref="RetainSequenceCache"/> (DESIGN §4.8).</summary>
        public bool RetainSequenceCacheAs(string requestId, string key)
        {
            lock (_sync)
            {
                if (!SupportsRetainedFusedCache) return false;
                try { _retainedSlotByRequest ??= new OrderedDictionary<string, int>(StringComparer.Ordinal); }
                catch (OutOfMemoryException) { return false; }
                bool kept = RetainNativeSequence(_slotByRequest, _retainedSlotByRequest, requestId, key,
                    ref _activeSlotKey, ref _selectedRetainedKey, _nativeRetentionBudget, new NativeSlotRetention(_slotExecutor));
                if (kept) TraceRetainedCommit("retain", key, _retainedSlotByRequest[key], _retainedSlotByRequest.Count);
                return kept;
            }
        }

        public bool TryRebindRetainedCache(string oldRequestId, string newRequestId)
        {
            lock (_sync)
            {
                if (!SupportsRetainedFusedCache) return false;
                try { _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal); }
                catch (OutOfMemoryException) { return false; }
                bool rebound = RebindNativeSequence(_slotByRequest, _retainedSlotByRequest, oldRequestId, newRequestId,
                    ref _activeSlotKey, ref _selectedRetainedKey, new NativeSlotRetention(_slotExecutor));
                if (rebound) TraceRetainedCommit("rebind", newRequestId, _slotByRequest[newRequestId], 0, oldRequestId);
                return rebound;
            }
        }

        public void DiscardRetainedCache(string requestId)
        {
            lock (_sync)
            {
                if (_slotExecutor == null) return;
                bool owned = _retainedSlotByRequest?.ContainsKey(requestId ?? "") == true;
                DiscardRetainedNativeSequence(_retainedSlotByRequest, requestId, ref _primarySlot,
                    ref _activeSlotKey, ref _selectedRetainedKey, new NativeSlotRetention(_slotExecutor));
                if (owned) TraceRetainedCommit("discard", requestId, _primarySlot, _retainedSlotByRequest.Count);
            }
        }

        public bool CanReuseLivePrefix(int cachedTokenCount, int targetTokenCount)
        {
            lock (_sync)
            {
                // Live-primary metadata must never refer to an idle retained
                // holder or a checked-out request selected by an earlier step.
                return SupportsExactFusedCacheReuse && _activeSlotKey == null && _selectedRetainedKey == null
                    && _primarySlot >= 0 && _slotExecutor.SlotCanReuse(_primarySlot, cachedTokenCount, targetTokenCount);
            }
        }

        public bool CanReuseRetainedPrefix(string retainedKey, int cachedTokenCount, int targetTokenCount)
        {
            lock (_sync)
            {
                return SupportsExactFusedCacheReuse && retainedKey != null && _retainedSlotByRequest != null
                    && _retainedSlotByRequest.TryGetValue(retainedKey, out int slot)
                    && _slotExecutor.SlotCanReuse(slot, cachedTokenCount, targetTokenCount);
            }
        }

        /// <summary>
        /// A slot for a request that has none. The empty primary when there is one; else a new slot
        /// while every device has room for it beside what the running requests still need. A retained
        /// conversation is released to make room only when a device has none, the one retained longest
        /// first: it is what that conversation's next turn reuses, so releasing it while memory is free
        /// throws away a turn's worth of prefill for nothing. Released while there is no primary, a
        /// retained slot is emptied into the primary, which the request then takes
        /// (<paramref name="tookPrimary"/>). Every released key is appended to <paramref name="released"/>.
        /// -1 when no slot could be had.
        /// </summary>
        internal static int SlotForRequest<TNative>(IDictionary<string, int> retained, ref int primary,
            ref string active, ref string selectedRetained, bool retention, List<string> released,
            TNative native, out bool tookPrimary) where TNative : INativeSlotStore
        {
            tookPrimary = false;
            while (true)
            {
                if (retention && primary >= 0 && native.Status(primary, out int head, out bool healthy)
                    && healthy && head == 0)
                {
                    tookPrimary = true;
                    return primary;
                }
                if (native.CanAlloc()
                    || !ReleaseOldestRetained(retained, ref primary, ref active, ref selectedRetained, released, native))
                    break;
            }
            int slot = native.Alloc();
            // The room estimate can be optimistic; a refused allocation releases more.
            while (slot < 0 && ReleaseOldestRetained(retained, ref primary, ref active, ref selectedRetained, released, native))
            {
                if (retention && primary >= 0 && native.Status(primary, out int head, out bool healthy)
                    && healthy && head == 0)
                {
                    tookPrimary = true;
                    return primary;
                }
                slot = native.Alloc();
            }
            return slot;
        }

        /// <summary>Release the conversation retained longest. With no primary it is selected first,
        /// so the release empties it into the primary instead of freeing it.</summary>
        internal static bool ReleaseOldestRetained<TNative>(IDictionary<string, int> retained, ref int primary,
            ref string active, ref string selectedRetained, List<string> released, TNative native)
            where TNative : INativeSlotRetention
        {
            if (retained == null || retained.Count == 0) return false;
            string key = null;
            foreach (string candidate in retained.Keys) { key = candidate; break; }
            if (primary < 0 && selectedRetained != key)
            {
                if (!native.Select(retained[key]))
                    throw new InvalidOperationException("DSV4 idle slot could not be selected for reclamation.");
                active = null;
                selectedRetained = key;
            }
            DiscardRetainedNativeSequence(retained, key, ref primary, ref active, ref selectedRetained, native);
            released.Add(key);
            return true;
        }

        /// <summary>The prefix cache learns of retained slots released behind its back (DEC-23); it applies
        /// the invalidation before its next tree read, so a plan that still names a key re-matches.</summary>
        private void ReportReleased(List<string> released)
        {
            foreach (string key in released)
            {
                TraceRetainedCommit("discard", key, _primarySlot, _retainedSlotByRequest?.Count ?? 0);
                _prefixCacheSink?.OnPayloadInvalidated(key, InvalidationReason.NativeSlotReclaimed);
            }
        }
    }
}
