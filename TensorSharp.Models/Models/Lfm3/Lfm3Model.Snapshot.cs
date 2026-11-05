// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// LFM3's KV-state snapshot: the contract the server's continuous-batching engine
// schedules a model on when it has no batched paged forward. It swaps requests on
// the one live cache by extracting and injecting fixed-size blocks.
//
// A block of N tokens is, layer by layer:
//   * attention layer: K rows then V rows for those N positions, each
//     [kvHeads, N, row] head-major (the cache's own row encoding, any dtype);
//   * conv layer: the conv state AT THE END of the block, [hidden, l_cache - 1]
//     floats, channel-major - only in the full form.
// The conv state is a function of every preceding token, so (like Nemotron-H's
// Mamba2 state) only a block captured exactly at its own end can be resumed from
// (RequiresPerBlockCapture); every other block is written in the K/V-only form
// (ComputeKVBlockByteSizeWithoutRecurrentState), which leaves the state alone on
// inject.
//
// Under tensor parallelism the layout is the same. Every attention layer's K/V
// heads are sharded contiguously by global rank (attn_k / attn_v split by rows), so
// concatenating rank g's [kvHeads/G, N, row] sections in rank order IS the
// single-device section; and the channel-major conv state puts rank g's channels
// in one contiguous run. Across nodes each node holds only its own ranks' heads and
// channels while the engine runs on the driver, so extract and inject are mirrored
// model-control operations, gathered / broadcast between nodes exactly as
// Nemotron-H's are.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Models
{
    public sealed partial class Lfm3Model
    {
        private const int TpControlSnapshotExtract = TpControlModelFirst + 1; // [start, count, withState]
        private const int TpControlSnapshotInject = TpControlModelFirst + 2;  // [dest, count, withState, bytes]

        public override bool RequiresPerBlockCapture => true;

        public override bool SupportsKVStateSnapshot =>
            IsTensorParallel ? _tpKvCacheK != null : _kvCacheK != null && _convState != null;

        public override string KVStateFingerprint
        {
            get
            {
                if (_layerKinds == null)
                    return string.Empty;
                var kinds = new char[_layerKinds.Length];
                for (int l = 0; l < kinds.Length; l++)
                    kinds[l] = _layerKinds[l] switch { LayerKind.Conv => 'c', LayerKind.FullAttention => 'f', _ => 's' };
                return $"lfm3|arch={Config.Architecture}|layers={new string(kinds)}|hidden={Config.HiddenSize}" +
                       $"|H={Config.NumHeads}|KV={Config.NumKVHeads}|D={_headDim}|rope={2 * _ropeHalf}" +
                       $"|window={_slidingWindow}|conv={_convWidth}|dtype={_kvCacheDtype.ToShortString()}";
            }
        }

        public override long ComputeKVBlockByteSize(int tokenCount) => BlockBytes(tokenCount, withState: true);

        /// <summary>The block without its conv-state sections: the attention K/V rows only.</summary>
        public override long ComputeKVBlockByteSizeWithoutRecurrentState(int tokenCount) => BlockBytes(tokenCount, withState: false);

        private long ConvStateBytes => (long)Config.HiddenSize * Math.Max(0, _convWidth - 1) * sizeof(float);

        /// <summary>One attention layer's K (or V) section of a block, across the whole cluster under TP.</summary>
        private long AttentionSectionBytes(int layer, int tokenCount)
            => IsTensorParallel
                ? CacheSectionBytes(_tpKvCacheK[layer][0], tokenCount) * GlobalTpDegree
                : CacheSectionBytes(_kvCacheK[layer], tokenCount);

        private long BlockBytes(int tokenCount, bool withState)
        {
            if (tokenCount <= 0 || !SupportsKVStateSnapshot)
                return 0;
            long total = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerKinds[l] != LayerKind.Conv)
                    total += 2 * AttentionSectionBytes(l, tokenCount);
                else if (withState)
                    total += ConvStateBytes;
            }
            return total;
        }

        /// <summary>Which form a span of <paramref name="length"/> bytes is: true for the full
        /// block, false for the K/V-only one, null for neither.</summary>
        private bool? BlockFormOf(int tokenCount, int length)
        {
            long full = ComputeKVBlockByteSize(tokenCount);
            if (full <= 0) return null;
            if (length == full) return true;
            long kvOnly = ComputeKVBlockByteSizeWithoutRecurrentState(tokenCount);
            return kvOnly > 0 && length == kvOnly ? false : null;
        }

        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!SupportsKVStateSnapshot) return false;
            if (BlockFormOf(tokenCount, destination.Length) is not bool withState) return false;
            if (startToken < 0 || startToken + tokenCount > _cacheSeqLen) return false;
            if (IsTensorParallel) return TryExtractKVBlockTP(startToken, tokenCount, withState, destination);

            EnsureHostRows(startToken + tokenCount);
            int offset = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerKinds[l] == LayerKind.Conv)
                {
                    if (!withState) continue;
                    SyncConvState(l);
                    ReadOnlySpan<byte> state = MemoryMarshal.AsBytes(ConvStateSpan(l));
                    state.CopyTo(destination[offset..]);
                    offset += state.Length;
                    continue;
                }
                if (!CopyCacheRowsOut(_kvCacheK[l], startToken, tokenCount, destination[offset..], out int wK))
                    return false;
                offset += wK;
                if (!CopyCacheRowsOut(_kvCacheV[l], startToken, tokenCount, destination[offset..], out int wV))
                    return false;
                offset += wV;
            }
            return offset == destination.Length;
        }

        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (!SupportsKVStateSnapshot) return false;
            if (destToken != _cacheSeqLen || tokenCount <= 0) return false;
            if (destToken + tokenCount > _maxContextLength) return false;
            if (BlockFormOf(tokenCount, source.Length) is not bool withState) return false;
            if (IsTensorParallel) return TryInjectKVBlockTP(destToken, tokenCount, withState, source);

            EnsureCacheCapacity(destToken + tokenCount);
            int offset = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerKinds[l] == LayerKind.Conv)
                {
                    if (!withState) continue;
                    Span<byte> state = MemoryMarshal.AsBytes(ConvStateSpan(l));
                    source.Slice(offset, state.Length).CopyTo(state);
                    UploadConvState(l);
                    offset += state.Length;
                    continue;
                }
                if (!CopyCacheRowsIn(_kvCacheK[l], destToken, tokenCount, source[offset..], out int rK))
                    return false;
                offset += rK;
                if (!CopyCacheRowsIn(_kvCacheV[l], destToken, tokenCount, source[offset..], out int rV))
                    return false;
                offset += rV;
                // The rows were written through the host copy.
                if (_nativeForward)
                {
                    UploadCacheRows(_kvCacheK[l], destToken, tokenCount);
                    UploadCacheRows(_kvCacheV[l], destToken, tokenCount);
                }
                else
                {
                    InvalidateTensorDeviceCache(_kvCacheK[l]);
                    InvalidateTensorDeviceCache(_kvCacheV[l]);
                }
            }
            // The rows just written are on the host; any before them may not be.
            if (_nativeHostRows == destToken)
                _nativeHostRows = destToken + tokenCount;
            _cacheSeqLen = destToken + tokenCount;
            return offset == source.Length;
        }

        // ------------------------------------------------------------------ tensor parallel

        private bool TryExtractKVBlockTP(int startToken, int tokenCount, bool withState, Span<byte> destination)
        {
            // Every check that can refuse ran before the broadcast: once the workers have
            // been told, they join the collective whatever the driver does next.
            BroadcastModelControl(TpControlSnapshotExtract, new[] { startToken, tokenCount, withState ? 1 : 0 });
            try
            {
                RunTpSnapshotExtract(startToken, tokenCount, withState, destination, isDriver: true);
            }
            catch (Exception ex)
            {
                if (TpNodeCount > 1) AbortDistributedModelControl(ex);
                throw;
            }
            return true;
        }

        private bool TryInjectKVBlockTP(int destToken, int tokenCount, bool withState, ReadOnlySpan<byte> source)
        {
            BroadcastModelControl(TpControlSnapshotInject, new[] { destToken, tokenCount, withState ? 1 : 0, source.Length });
            try
            {
                RunTpSnapshotInject(destToken, tokenCount, withState, source, source.Length);
            }
            catch (Exception ex)
            {
                if (TpNodeCount > 1) AbortDistributedModelControl(ex);
                throw;
            }
            return true;
        }

        protected override bool HandleModelControl(int op, int[] payload)
        {
            switch (op)
            {
                case TpControlSnapshotExtract:
                    RunTpSnapshotExtract(payload[0], payload[1], payload[2] != 0, Span<byte>.Empty, isDriver: false);
                    return true;
                case TpControlSnapshotInject:
                    RunTpSnapshotInject(payload[0], payload[1], payload[2] != 0, ReadOnlySpan<byte>.Empty, payload[3]);
                    return true;
                default:
                    return base.HandleModelControl(op, payload);
            }
        }

        /// <summary>
        /// Lay out the block from this node's shards - its ranks' heads and conv channels,
        /// zeros elsewhere - complete it across nodes, and (on the driver) hand it over.
        /// A worker contributes its shards and keeps nothing.
        /// </summary>
        private void RunTpSnapshotExtract(int startToken, int tokenCount, bool withState, Span<byte> destination, bool isDriver)
        {
            var block = new byte[checked((int)BlockBytes(tokenCount, withState))];
            EnsureHostRows(startToken + tokenCount);
            int channelsPerRank = Config.HiddenSize / GlobalTpDegree;
            int history = Math.Max(0, _convWidth - 1);
            int offset = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerKinds[l] == LayerKind.Conv)
                {
                    if (!withState) continue;
                    SyncConvState(l);
                    ReadOnlySpan<byte> state = MemoryMarshal.AsBytes(ConvStateSpan(l));
                    int perRank = channelsPerRank * history * sizeof(float);
                    for (int r = 0; r < TpDegree; r++)
                    {
                        int at = (TpRankOffset + r) * perRank;
                        state.Slice(at, perRank).CopyTo(block.AsSpan(offset + at, perRank));
                    }
                    offset += state.Length;
                    continue;
                }
                foreach (Tensor[] shards in new[] { _tpKvCacheK[l], _tpKvCacheV[l] })
                {
                    int perRank = checked((int)CacheSectionBytes(shards[0], tokenCount));
                    for (int r = 0; r < TpDegree; r++)
                    {
                        int at = offset + (TpRankOffset + r) * perRank;
                        if (!CopyCacheRowsOut(shards[r], startToken, tokenCount, block.AsSpan(at, perRank), out _))
                            throw new InvalidOperationException(
                                $"lfm3 TP snapshot: layer {l} rank {r} has no rows {startToken}..{startToken + tokenCount}.");
                    }
                    offset += perRank * GlobalTpDegree;
                }
            }
            if (offset != block.Length)
                throw new InvalidOperationException($"lfm3 TP snapshot laid out {offset} of {block.Length} bytes.");

            TpGatherBytesToDriver(block);
            if (isDriver)
                block.CopyTo(destination);
        }

        /// <summary>
        /// Write a block at <paramref name="destToken"/>: across nodes the driver's block
        /// reaches every node first, then each node keeps its own ranks' heads. The conv
        /// state is taken whole; each rank reads only its own channels of it.
        /// </summary>
        private void RunTpSnapshotInject(int destToken, int tokenCount, bool withState, ReadOnlySpan<byte> source, int length)
        {
            byte[] shared = null;
            if (TpNodeCount > 1)
            {
                shared = new byte[length];
                if (!source.IsEmpty) source.CopyTo(shared);
                TpBroadcastBytesFromDriver(shared);
            }
            ReadOnlySpan<byte> block = shared ?? source;

            EnsureTpCacheCapacity(destToken + tokenCount);
            int offset = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerKinds[l] == LayerKind.Conv)
                {
                    if (!withState) continue;
                    Span<byte> state = MemoryMarshal.AsBytes(ConvStateSpan(l));
                    block.Slice(offset, state.Length).CopyTo(state);
                    UploadConvState(l);
                    offset += state.Length;
                    continue;
                }
                foreach (Tensor[] shards in new[] { _tpKvCacheK[l], _tpKvCacheV[l] })
                {
                    int perRank = checked((int)CacheSectionBytes(shards[0], tokenCount));
                    for (int r = 0; r < TpDegree; r++)
                    {
                        int at = offset + (TpRankOffset + r) * perRank;
                        if (!CopyCacheRowsIn(shards[r], destToken, tokenCount, block.Slice(at, perRank), out _))
                            throw new InvalidOperationException(
                                $"lfm3 TP snapshot: layer {l} rank {r} cannot hold rows {destToken}..{destToken + tokenCount}.");
                        // The rows were written through the host copy.
                        if (_nativeForward)
                            UploadCacheRows(shards[r], destToken, tokenCount);
                        else
                            InvalidateTensorDeviceCache(shards[r]);
                    }
                    offset += perRank * GlobalTpDegree;
                }
            }
            if (offset != length)
                throw new InvalidOperationException($"lfm3 TP snapshot read {offset} of {length} bytes.");
            if (_nativeHostRows == destToken)
                _nativeHostRows = destToken + tokenCount;
            _cacheSeqLen = destToken + tokenCount;
        }
    }
}
