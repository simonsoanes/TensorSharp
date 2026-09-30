// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Runtime.Paged
{
    /// <summary>
    /// Metadata for a single physical block in the paged KV-cache pool. Mirrors
    /// vLLM's <c>KVCacheBlock</c>: a fixed-size slab of bytes that holds
    /// <c>blockSize</c> tokens worth of K/V state across all model layers,
    /// referenced by zero or more <see cref="SequenceState"/> instances.
    ///
    /// A block is live (<c>RefCount &gt; 0</c>: it holds K/V for some active
    /// sequence(s), or the prefix cache holds it) or free (<c>RefCount == 0</c>: in the
    /// free queue, holding nothing).
    ///
    /// Mutation is the <see cref="BlockPool"/>'s responsibility; this type is just
    /// the storage. Bytes for the actual K/V payload live in
    /// <see cref="PagedKvStorage"/> keyed by <see cref="Id"/>.
    /// </summary>
    public sealed class KvBlock
    {
        /// <summary>Stable physical id (0..numBlocks-1). Indexes into
        /// <see cref="PagedKvStorage"/> for the actual bytes.</summary>
        public int Id { get; }

        /// <summary>How many <see cref="SequenceState"/> objects currently reference
        /// this block via their <see cref="BlockTable"/>.</summary>
        public int RefCount { get; internal set; }

        /// <summary>Number of valid tokens written into this block (0..blockSize).</summary>
        public int Used { get; internal set; }

        /// <summary>
        /// Whether restoring the block chain through this block reconstructs the
        /// model at this exact prefix end.  Attention-only snapshots are valid at
        /// every block.  Hybrid recurrent models may capture several attention
        /// slices after one large fused prefill; only the final block at that
        /// forward boundary contains the matching recurrent-state checkpoint.
        /// </summary>
        public bool IsRestorablePrefixEnd { get; internal set; }

        /// <summary>
        /// The model's OWN paged K/V arrays hold this block's positions: every token in
        /// it was forwarded through <c>IBatchedPagedModel.ForwardBatch</c> (which writes
        /// through the slot mapping) rather than through the linear-cache
        /// <c>Forward</c>. A block registered by a batched step carries no bytes in
        /// <see cref="PagedKvStorage"/>, so a sequence that adopts it has to keep reading
        /// it from the model's paged arrays; restoring it into the linear cache from the
        /// pool would inject bytes that were never captured.
        /// </summary>
        public bool HoldsModelPagedKv { get; internal set; }

        /// <summary>
        /// <see cref="PagedKvStorage"/> holds a full-block snapshot extracted from the
        /// model's linear cache (<c>TryExtractKVBlock</c>), so the block can be restored
        /// into any sequence's linear cache.
        /// </summary>
        public bool HoldsSnapshotBytes { get; internal set; }

        /// <summary>Doubly-linked-list pointers for the free queue. Maintained by
        /// <see cref="FreeBlockQueue"/>. When the block is allocated both pointers
        /// are null.</summary>
        internal KvBlock? PrevFree;
        internal KvBlock? NextFree;

        public KvBlock(int id)
        {
            Id = id;
            RefCount = 0;
            Used = 0;
            IsRestorablePrefixEnd = true;
        }

        /// <summary>True when the block is in the free queue.</summary>
        public bool IsFree => RefCount == 0;

        public override string ToString()
            => $"KvBlock(id={Id}, refs={RefCount}, used={Used}, restorable={IsRestorablePrefixEnd}, paged={HoldsModelPagedKv}, snapshot={HoldsSnapshotBytes})";
    }

    /// <summary>
    /// Doubly-linked list of free <see cref="KvBlock"/>s in LRU eviction order.
    /// Front = least-recently-used (handed out first); back = most-recently-used.
    /// O(1) push/pop/remove via per-node pointers. Mirrors vLLM's
    /// <c>FreeKVCacheBlockQueue</c>.
    /// </summary>
    public sealed class FreeBlockQueue
    {
        private readonly KvBlock _head;   // sentinel
        private readonly KvBlock _tail;   // sentinel
        private int _count;

        public FreeBlockQueue()
        {
            _head = new KvBlock(-1);
            _tail = new KvBlock(-1);
            _head.NextFree = _tail;
            _tail.PrevFree = _head;
            _count = 0;
        }

        public int Count => _count;

        /// <summary>Append to the tail (most-recently-freed end).</summary>
        public void Enqueue(KvBlock block)
        {
            if (block.PrevFree != null || block.NextFree != null)
                throw new InvalidOperationException($"Block {block.Id} is already on a free queue.");

            KvBlock previous = _tail.PrevFree
                ?? throw new InvalidOperationException("The free queue tail has no predecessor.");
            block.PrevFree = previous;
            block.NextFree = _tail;
            previous.NextFree = block;
            _tail.PrevFree = block;
            _count++;
        }

        /// <summary>Pop from the head (least-recently-used).</summary>
        public KvBlock? Dequeue()
        {
            if (_count == 0)
                return null;
            KvBlock first = _head.NextFree
                ?? throw new InvalidOperationException("The free queue head has no successor.");
            Remove(first);
            return first;
        }

        /// <summary>Remove the given block from anywhere in the queue. The block
        /// must currently be in this queue; otherwise behavior is undefined.</summary>
        public void Remove(KvBlock block)
        {
            if (block.PrevFree == null && block.NextFree == null)
                return; // not in queue
            KvBlock previous = block.PrevFree
                ?? throw new InvalidOperationException($"Block {block.Id} has no predecessor in the free queue.");
            KvBlock next = block.NextFree
                ?? throw new InvalidOperationException($"Block {block.Id} has no successor in the free queue.");
            previous.NextFree = next;
            next.PrevFree = previous;
            block.PrevFree = null;
            block.NextFree = null;
            _count--;
        }

        /// <summary>Peek the first block without removing.</summary>
        public KvBlock? PeekFront()
        {
            if (_count == 0) return null;
            return _head.NextFree;
        }
    }
}
