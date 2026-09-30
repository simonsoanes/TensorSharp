// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// GLM's managed slot retention for the radix prefix cache, on a fake slot store:
// the ownership moves must never leave a retained slot active, never lose a slot
// on a refusal, and never change the native selection they found unless the move
// itself requires it; memory takes retained slots back oldest first.
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class GlmSlotRetentionTests
{
    [Fact]
    public void MemoryTakesRetainedSlotsBackOldestFirst_EvenAfterAConversationWasRetainedAgain()
    {
        // "a" was retained, donated to its next turn and retained again after "b": "b" is now the oldest. A plain
        // Dictionary reuses the freed entry and enumerates "a" first.
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot> { ["a"] = new(1, 40), ["b"] = new(2, 30) };
        retained.Remove("a");
        retained.Add("a", new(1, 52));
        var store = new Slots(active: 0, (0, 0), (1, 52), (2, 30)) { Capacity = 3 };
        var reclaimed = new List<string>();
        Assert.True(SlotRetention.AllocReclaiming(retained, store, reclaimed) >= 0);
        Assert.Equal(new[] { "b" }, reclaimed);
        Assert.True(retained.ContainsKey("a"));
    }

    [Fact]
    public void GraphsMayFreeTheRetainedSlotsOldestFirst_AndTheListIsPushedOnlyWhenItChanges()
    {
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>
        {
            ["a"] = new(5, 40), ["b"] = new(3, 30), ["c"] = new(9, 20),
        };
        var store = new Slots(active: 0, (0, 0));
        int[] synced = SlotRetention.SyncReclaimable(retained, Array.Empty<int>(), store);
        Assert.Equal(new[] { 5, 3, 9 }, synced);
        Assert.Single(store.Pushes);

        // Unchanged: nothing crosses to the native side on the next forward.
        Assert.Same(synced, SlotRetention.SyncReclaimable(retained, synced, store));
        Assert.Single(store.Pushes);

        // A donation to the conversation's next turn takes a slot out of the list before the forward runs.
        retained.Remove("b");
        synced = SlotRetention.SyncReclaimable(retained, synced, store);
        Assert.Equal(new[] { 5, 9 }, store.Pushes[^1]);

        // Nothing retained: an empty list, so no slot of a running request can be freed.
        retained.Clear();
        SlotRetention.SyncReclaimable(retained, synced, store);
        Assert.Empty(store.Pushes[^1]);
        Assert.Empty(SlotRetention.SyncReclaimable(null, Array.Empty<int>(), store));
    }

    [Fact]
    public void SlotsFreedForAGraph_AreForgottenAndReportedOnce()
    {
        var retained = new OrderedDictionary<string, SlotRetention.RetainedSlot>
        {
            ["a"] = new(5, 40), ["b"] = new(3, 30), ["c"] = new(9, 20),
        };
        var store = new Slots(active: 0, (0, 0));
        foreach (int id in new[] { 9, 5, 42 }) store.Reclaimed.Enqueue(id);   // 42: a slot the model no longer lists
        var reclaimed = new List<string>();
        SlotRetention.DrainReclaimed(retained, store, new int[2], reclaimed);   // a buffer smaller than the queue
        Assert.Equal(new[] { "c", "a" }, reclaimed);
        Assert.Equal(new[] { "b" }, retained.Keys);

        SlotRetention.DrainReclaimed(retained, store, new int[2], reclaimed);
        Assert.Equal(2, reclaimed.Count);
    }

    [Fact]
    public void ANewSlotReclaimsAnIdleRetainedSlotWhenMemoryHoldsNoMore()
    {
        // Two slots fit: the primary (active) and a retained conversation. A request that needs a slot now
        // takes the retained one's memory, and the reclaimed key is handed back for the sink.
        var store = new Slots(active: 0, (0, 12), (1, 40)) { Capacity = 2 };
        var retained = new Dictionary<string, SlotRetention.RetainedSlot> { ["pc:1:1"] = new(1, 40) };
        var reclaimed = new List<string>();
        int slot = SlotRetention.AllocReclaiming(retained, store, reclaimed);
        Assert.True(slot >= 0);
        Assert.Empty(retained);
        Assert.Equal(new[] { "pc:1:1" }, reclaimed);
        Assert.False(store.Heads.ContainsKey(1));

        // Nothing left to reclaim: -1, and the active primary is never freed.
        Assert.Equal(-1, SlotRetention.AllocReclaiming(retained, store, reclaimed));
        Assert.True(store.Heads.ContainsKey(0));
        Assert.Single(reclaimed);

        // Room to spare: nothing is reclaimed.
        var roomy = new Slots(active: 0, (0, 12), (1, 40)) { Capacity = 3 };
        var kept = new Dictionary<string, SlotRetention.RetainedSlot> { ["pc:1:1"] = new(1, 40) };
        var none = new List<string>();
        Assert.True(SlotRetention.AllocReclaiming(kept, roomy, none) >= 0);
        Assert.Single(kept);
        Assert.Empty(none);
    }

    [Fact]
    public void ReleasingTheActiveSlotWhileARequestHoldsThePrimary_MakesItThePrimaryWithoutAllocating()
    {
        // Memory holds two slots: A adopted the primary (slot 0) and B has slot 1, B active. Releasing B used
        // to allocate a fresh primary before freeing slot 1, one slot more than memory holds: the allocation
        // threw, slot 1 was never freed, and a request waiting for memory never got it.
        var store = new Slots(active: 1, (0, 12), (1, 40)) { Capacity = 2 };
        var requests = new Dictionary<string, int> { ["A"] = 0, ["B"] = 1 };
        int primary = -1;
        string active = "B";
        Assert.True(SlotRetention.ReleaseActiveSlotAsPrimary(requests, "B", ref primary, ref active, store));
        Assert.Equal(1, primary);
        Assert.Null(active);
        Assert.Equal(0, store.Heads[1]);          // emptied: the next single-stream request starts at 0
        Assert.Equal(12, store.Heads[0]);         // A's slot untouched
        Assert.Equal(0, store.Allocs);
        Assert.Equal(new Dictionary<string, int> { ["A"] = 0 }, requests);

        // Only the active slot of a request, and only while a request holds the primary.
        Assert.False(SlotRetention.ReleaseActiveSlotAsPrimary(requests, "A", ref primary, ref active, store));
        Assert.Equal(1, primary);
        Assert.True(requests.ContainsKey("A"));

        // A slot that cannot be emptied is freed instead; the next single-stream step allocates a primary.
        var broken = new Slots(active: 1, (0, 12), (1, 40)) { RefuseReset = true, FreeActive = true };
        var held = new Dictionary<string, int> { ["A"] = 0, ["B"] = 1 };
        int none = -1;
        string bActive = "B";
        Assert.False(SlotRetention.ReleaseActiveSlotAsPrimary(held, "B", ref none, ref bActive, broken));
        Assert.Equal(-1, none);
        Assert.False(broken.Heads.ContainsKey(1));
        Assert.False(held.ContainsKey("B"));
    }

    [Fact]
    public void TheIdlePrimary_IsHandedOutEmptied_AndDeclinedOnceARequestHoldsIt()
    {
        // Two slots fit and both are held (the primary is idle, request B has slot 1 and is active).
        var store = new Slots(active: 1, (0, 12), (1, 40)) { Capacity = 2 };
        var requests = new Dictionary<string, int> { ["B"] = 1 };
        int primary = 0;
        Assert.Equal(0, SlotRetention.TakeIdlePrimary(requests, ref primary, "B", store));
        Assert.Equal(-1, primary);
        Assert.Equal(0, store.Heads[0]);          // emptied for its new owner
        Assert.Equal(40, store.Heads[1]);
        Assert.Equal(0, store.Allocs);

        // Held by a request now: nothing more to take.
        Assert.Equal(-1, SlotRetention.TakeIdlePrimary(requests, ref primary, "B", store));

        // A primary that cannot be emptied is kept, with the previous selection restored.
        var stuck = new Slots(active: 1, (0, 12), (1, 40)) { RefuseReset = true };
        int stuckPrimary = 0;
        Assert.Equal(-1, SlotRetention.TakeIdlePrimary(requests, ref stuckPrimary, "B", stuck));
        Assert.Equal(0, stuckPrimary);
        Assert.Equal(1, stuck.Active);
        Assert.Equal(12, stuck.Heads[0]);
    }

    [Fact]
    public void RetainingTheActiveSlot_SelectsThePrimaryAndKeysTheSlotByThePayload()
    {
        var store = new Slots(active: 1, (0, 0), (1, 40));
        var requests = new Dictionary<string, int> { ["req"] = 1 };
        var retained = new Dictionary<string, SlotRetention.RetainedSlot>();
        int primary = 0; string active = "req";
        Assert.True(SlotRetention.RetainSlot(requests, retained, "req", "pc:1:1", 40, canRewind: true, ref primary, ref active, store));
        Assert.Equal(new SlotRetention.RetainedSlot(1, 40), retained["pc:1:1"]);
        Assert.Empty(requests);
        Assert.Null(active);
        Assert.Equal(0, store.Active);
        Assert.Equal(0, primary);
    }

    [Fact]
    public void RetainingARequestThatAdoptedThePrimary_AllocatesAFreshPrimary_OrRefusesUntouched()
    {
        var store = new Slots(active: 0, (0, 40)) { AllocFails = true };
        var requests = new Dictionary<string, int> { ["req"] = 0 };
        var retained = new Dictionary<string, SlotRetention.RetainedSlot>();
        int primary = -1; string active = "req";
        Assert.False(SlotRetention.RetainSlot(requests, retained, "req", "pc:1:1", 40, true, ref primary, ref active, store));
        Assert.Equal(0, requests["req"]);
        Assert.Equal("req", active);
        Assert.Equal(-1, primary);
        Assert.Equal(0, store.Active);
        Assert.Empty(retained);

        store.AllocFails = false;
        Assert.True(SlotRetention.RetainSlot(requests, retained, "req", "pc:1:1", 40, true, ref primary, ref active, store));
        Assert.True(primary > 0);
        Assert.Equal(primary, store.Active);
        Assert.Equal(0, store.Heads[primary]);
        Assert.Equal(0, retained["pc:1:1"].Slot);
    }

    [Fact]
    public void ARefusedRewind_FreesTheFreshPrimaryItAllocated_AndLeavesTheSlotAsItWas()
    {
        var store = new Slots(active: 0, (0, 40)) { RefuseRewind = true };
        var requests = new Dictionary<string, int> { ["req"] = 0 };
        var retained = new Dictionary<string, SlotRetention.RetainedSlot>();
        int primary = -1; string active = "req";
        Assert.False(SlotRetention.RetainSlot(requests, retained, "req", "pc:1:1", 36, canRewind: true, ref primary, ref active, store));
        Assert.Equal(new[] { 0 }, store.Heads.Keys);   // the fresh primary was freed again
        Assert.Equal(40, store.Heads[0]);
        Assert.Equal(-1, primary);
        Assert.Equal("req", active);
        Assert.Equal(0, requests["req"]);
    }

    [Fact]
    public void RetainingAnInactiveSlot_ReadsItThroughASelectAndRestoresTheSelection()
    {
        var store = new Slots(active: 0, (0, 5), (1, 40), (2, 9));
        var requests = new Dictionary<string, int> { ["done"] = 1, ["running"] = 2 };
        var retained = new Dictionary<string, SlotRetention.RetainedSlot>();
        int primary = 0; string active = null;

        // Longer than asked: rewound where the architecture can (glm-dsa)...
        Assert.True(SlotRetention.RetainSlot(requests, retained, "done", "pc:1:1", 36, canRewind: true, ref primary, ref active, store));
        Assert.Equal(36, store.Heads[1]);
        Assert.Equal(0, store.Active);
        Assert.Equal(new SlotRetention.RetainedSlot(1, 36), retained["pc:1:1"]);

        // ...refused where it cannot (glm5next), and shorter than asked is always refused.
        Assert.False(SlotRetention.RetainSlot(requests, retained, "running", "pc:1:2", 8, canRewind: false, ref primary, ref active, store));
        Assert.False(SlotRetention.RetainSlot(requests, retained, "running", "pc:1:2", 10, canRewind: true, ref primary, ref active, store));
        Assert.Equal(9, store.Heads[2]);
        Assert.Equal(2, requests["running"]);
        Assert.Equal(0, store.Active);

        // A key already in use consumes nothing.
        Assert.False(SlotRetention.RetainSlot(requests, retained, "running", "pc:1:1", 9, true, ref primary, ref active, store));
        Assert.Equal(2, requests["running"]);
    }

    [Fact]
    public void DonateThenReturn_MovesTheSlotBothWays_AndABoundDonationCannotReturn()
    {
        var store = new Slots(active: 0, (0, 0), (1, 40));
        var requests = new Dictionary<string, int>();
        var retained = new Dictionary<string, SlotRetention.RetainedSlot> { ["pc:1:1"] = new(1, 40) };
        int primary = 0; string active = null;

        Assert.True(SlotRetention.CanDonate(retained, "pc:1:1", 40, 40, canRewind: false));
        Assert.True(SlotRetention.CanDonate(retained, "pc:1:1", 40, 36, canRewind: true));
        Assert.False(SlotRetention.CanDonate(retained, "pc:1:1", 40, 36, canRewind: false));
        Assert.False(SlotRetention.CanDonate(retained, "pc:1:1", 41, 41, canRewind: true));

        Assert.True(SlotRetention.DonateSlot(requests, retained, "pc:1:1", "next", out var donated));
        Assert.Equal(1, donated.Slot);
        Assert.Equal(1, requests["next"]);
        Assert.Empty(retained);
        Assert.False(SlotRetention.DonateSlot(requests, retained, "pc:1:1", "again", out _));

        Assert.True(SlotRetention.ReturnSlot(requests, retained, "next", "pc:1:1", ref primary, ref active, store));
        Assert.Equal(new SlotRetention.RetainedSlot(1, 40), retained["pc:1:1"]);
        Assert.Equal(0, store.Active);

        Assert.True(SlotRetention.DonateSlot(requests, retained, "pc:1:1", "bound", out _));
        active = "bound";
        store.Active = 1;
        Assert.False(SlotRetention.ReturnSlot(requests, retained, "bound", "pc:1:1", ref primary, ref active, store));
        Assert.Equal(1, requests["bound"]);
    }

    [Fact]
    public void ConvertPrimary_RetainsTheLivePrimaryAndSelectsAFreshOne_OrRefusesUntouched()
    {
        var store = new Slots(active: 0, (0, 40), (1, 7));
        var retained = new Dictionary<string, SlotRetention.RetainedSlot>();
        int primary = 0; string active = null;
        Assert.False(SlotRetention.ConvertPrimarySlot(retained, "pc:1:1", 39, ref primary, ref active, store));
        store.AllocFails = true;
        Assert.False(SlotRetention.ConvertPrimarySlot(retained, "pc:1:1", 40, ref primary, ref active, store));
        string busy = "req";
        store.AllocFails = false;
        Assert.False(SlotRetention.ConvertPrimarySlot(retained, "pc:1:1", 40, ref primary, ref busy, store));
        Assert.Equal(0, primary);
        Assert.Empty(retained);

        Assert.True(SlotRetention.ConvertPrimarySlot(retained, "pc:1:1", 40, ref primary, ref active, store));
        Assert.Equal(new SlotRetention.RetainedSlot(0, 40), retained["pc:1:1"]);
        Assert.NotEqual(0, primary);
        Assert.Equal(primary, store.Active);
        Assert.Equal(40, store.Heads[0]);
    }

    [Fact]
    public void Release_FreesKnownSlots_IgnoresUnknownKeys_AndKeepsARefusedSlotForARetry()
    {
        var store = new Slots(active: 0, (0, 0), (1, 40), (2, 30));
        var retained = new Dictionary<string, SlotRetention.RetainedSlot> { ["a"] = new(1, 40), ["b"] = new(2, 30) };
        store.RefuseFree.Add(2);
        SlotRetention.ReleaseSlots(retained, new[] { "a", "missing", "b" }, store);
        Assert.False(store.Heads.ContainsKey(1));
        Assert.True(retained.ContainsKey("b"));
        store.RefuseFree.Clear();
        SlotRetention.ReleaseSlots(retained, new[] { "b" }, store);
        Assert.Empty(retained);
    }

    private sealed class Slots : SlotRetention.ISlotStore
    {
        public readonly Dictionary<int, int> Heads;
        public readonly HashSet<int> RefuseFree = new();
        public int Active;
        public bool AllocFails;
        /// <summary>Slots device memory holds; Alloc fails past it.</summary>
        public int Capacity = int.MaxValue;
        public bool RefuseRewind;
        public bool RefuseReset;
        /// <summary>The native side frees an active slot too (it re-points the selection); off by default so a
        /// move that frees the slot it still has selected is caught.</summary>
        public bool FreeActive;
        public int Allocs;
        /// <summary>Every reclaimable list pushed, in order.</summary>
        public readonly List<int[]> Pushes = new();
        /// <summary>Slots the native side freed for a graph, waiting to be taken.</summary>
        public readonly Queue<int> Reclaimed = new();
        private int _next = 100;

        public Slots(int active, params (int Slot, int Head)[] slots)
        {
            Active = active;
            Heads = slots.ToDictionary(s => s.Slot, s => s.Head);
        }

        public int Alloc()
        {
            Allocs++;
            if (AllocFails || Heads.Count >= Capacity) return -1;
            Heads[_next] = 0;
            return _next++;
        }

        public bool Select(int slot)
        {
            if (!Heads.ContainsKey(slot)) return false;
            Active = slot;
            return true;
        }

        // Like the native side: the active slot cannot be freed (a fresh primary is never selected before it is freed).
        public bool Free(int slot) => (FreeActive || slot != Active) && !RefuseFree.Contains(slot) && Heads.Remove(slot);
        public int ActiveHead() => Heads[Active];

        public bool RewindActive(int tokens)
        {
            if (RefuseRewind || tokens > Heads[Active]) return false;
            Heads[Active] = tokens;
            return true;
        }

        public bool ResetActive()
        {
            if (RefuseReset) return false;
            Heads[Active] = 0;
            return true;
        }

        public bool SetReclaimable(int[] slots)
        {
            Pushes.Add(slots.ToArray());
            return true;
        }

        public int TakeReclaimed(int[] buffer)
        {
            int n = 0;
            while (n < buffer.Length && Reclaimed.Count > 0) buffer[n++] = Reclaimed.Dequeue();
            return n;
        }
    }
}
