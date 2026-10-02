// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// V4.1 state that crosses a layer-split boundary.
//
// V4 layers keep everything they read to themselves, so a device boundary only
// has to hand over the hidden streams. V4.1 layers do not: a ratio group's
// compressed and indexer caches are written by the group's first layer and read
// by every layer in the group; an index-source layer publishes the sparse
// selection the layers after it attend over; the candidate layer publishes the
// mask later indexers prune with; and every block collapses the streams with
// the gates the block before it published. Each of those lives in its device's
// memory. A group that spans a boundary read another device's cache pointer (an
// illegal access on the published Q2_K checkpoint's 6-way split) and the other
// three read whatever the next device's scratch last held.
//
// So every boundary now carries, beside the hidden streams: the cache rows each
// spanning group committed in this forward (into a replica of the group's
// caches on each later device that reads them), the published selection, the
// candidate mask and the delayed gates, whichever the layers after the
// boundary read. All of it moves through the same pinned staging and event
// chain as the streams (peer DMA is not trusted on these topologies).
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Dsv4CudaEngine
    {
        /// <summary>What the boundary into one device carries (V4.1).</summary>
        private sealed class Boundary
        {
            /// <summary>KV-source layers whose caches layers on this device read.</summary>
            public readonly List<int> KvGroups = new List<int>();
            /// <summary>A compressed layer here attends over a selection published before the boundary.</summary>
            public bool Selection;
            /// <summary>An index source here or later prunes with the mask published before the boundary.</summary>
            public bool CandidateMask;
        }

        // Indexed by the device a boundary leads INTO; null for device 0 and for V4.
        private Boundary[] _boundaries;
        // Per KV-source layer: the device of the last layer reading its caches.
        private int[] _groupLastDevice;
        // Cache blocks each spanning group's source committed in the current forward.
        private readonly List<(int Source, Slot Slot, int FirstBlock, int Count)> _committed =
            new List<(int, Slot, int, int)>();

        /// <summary>Work out what each boundary carries, from the layer map. V4.1 only.</summary>
        private void PlanBoundaries(ModelDesc m, int[] assignment)
        {
            if (!m.V41 || _devs.Length < 2)
                return;
            int n = m.NLayer;
            _groupLastDevice = new int[n];
            for (int il = 0; il < n; il++)
                _groupLastDevice[il] = -1;
            for (int il = 0; il < n; il++)
            {
                int source = m.Layers[il].Ratio != 0 ? m.Layers[il].KvSource : -1;
                if (source >= 0)
                    _groupLastDevice[source] = Math.Max(_groupLastDevice[source], assignment[il]);
            }

            _boundaries = new Boundary[_devs.Length];
            for (int d = 1; d < _devs.Length; d++)
            {
                int first = Array.IndexOf(assignment, d);
                if (first < 0)
                    continue;
                var b = new Boundary();
                for (int s = 0; s < n; s++)
                {
                    if (_groupLastDevice[s] >= d && assignment[s] < d)
                        b.KvGroups.Add(s);
                }
                for (int il = first; il < n && assignment[il] == d; il++)
                {
                    var L = m.Layers[il];
                    if (L.Ratio != 0 && L.IndexSource >= 0 && L.IndexSource < first)
                        b.Selection = true;
                }
                if (m.CandidateSource >= 0 && m.CandidateSource < first)
                {
                    for (int il = first; il < n; il++)
                    {
                        if (m.Layers[il].IndexSource == il && il > m.CandidateSource)
                        {
                            b.CandidateMask = true;
                            break;
                        }
                    }
                }
                _boundaries[d] = b;
            }

            // Pinned staging on each sending device: the one that holds the layer before the
            // boundary (a device the split left without layers sends nothing, and the embedding
            // device sends into a first layer placed past it).
            long rowBytes = ((long)m.HeadDim + m.IdxHeadSize) * 2;
            long maskWidth = Math.Max(_compRowsCsa, _compRowsHca);
            for (int d = 1; d < _devs.Length; d++)
            {
                Boundary b = _boundaries[d];
                if (b == null)
                    continue;
                int first = Array.IndexOf(assignment, d);
                long bytes = (long)m.NUbatch * HC * 4;                                      // delayed gates
                bytes += b.KvGroups.Count * ((long)m.NUbatch + 1) * rowBytes;               // committed cache rows
                if (b.Selection)
                    bytes += (long)m.NUbatch * (Math.Max(m.IdxTopK, 1) + 1) * 4;             // selection + counts
                if (b.CandidateMask)
                    bytes += (long)m.NUbatch * maskWidth;                                   // candidate mask
                Dev src = _devs[first > 0 ? assignment[first - 1] : 0];
                src.MakeCurrent();
                CudaDriverApi.cuMemHostAlloc(out IntPtr pinned, new UIntPtr((ulong)bytes), 0x1 /*PORTABLE*/).ThrowOnError();
                src.StatePinned = pinned;
                src.StatePinnedBytes = bytes;
            }
        }

        /// <summary>Whether a group's caches have replicas (its readers span devices).</summary>
        private bool Replicated(int source) => _groupLastDevice != null && _groupLastDevice[source] > _layers[source].Device;

        /// <summary>The replicas of a V4.1 source layer's caches on the later devices that read them.</summary>
        private void AllocReplicas(Slot s, SlotLayer c, int source)
        {
            if (!Replicated(source))
                return;
            int rows = V41Rows(_layers[source].Ratio);
            c.CompKOn = new Tensor[_devs.Length];
            c.LidKOn = new Tensor[_devs.Length];
            for (int d = _layers[source].Device + 1; d <= _groupLastDevice[source]; d++)
            {
                c.CompKOn[d] = SlotTensor(s, _devs[d], DType.Float16, rows, _m.HeadDim);
                c.LidKOn[d] = SlotTensor(s, _devs[d], DType.Float16, rows, _m.IdxHeadSize);
            }
        }

        /// <summary>The compressed cache of <paramref name="source"/>'s group as seen from <paramref name="dev"/>.</summary>
        private Tensor CompKOn(Slot s, int source, Dev dev)
        {
            SlotLayer c = s.Layers[source];
            return dev.Ordinal == _layers[source].Device ? c.CompK : c.CompKOn[dev.Ordinal];
        }

        /// <summary>The indexer cache of <paramref name="source"/>'s group as seen from <paramref name="dev"/>.</summary>
        private Tensor LidKOn(Slot s, int source, Dev dev)
        {
            SlotLayer c = s.Layers[source];
            return dev.Ordinal == _layers[source].Device ? c.LidK : c.LidKOn[dev.Ordinal];
        }

        /// <summary>Note blocks a spanning group's source committed, for the boundaries after it.</summary>
        private void NoteCommitted(int source, Slot s, long firstBoundary, int ratio, int count)
        {
            if (count > 0 && Replicated(source))
                _committed.Add((source, s, (int)(firstBoundary / ratio), count));
        }

        /// <summary>
        /// Queue, on <paramref name="src"/>'s stream, the copies of everything the layers after the
        /// boundary into <paramref name="dst"/> read besides the hidden streams; returns the pinned
        /// bytes used and the transfers to replay on <paramref name="dst"/>'s stream.
        /// </summary>
        private List<(long Offset, IntPtr Dst, long Bytes)> StageBoundaryState(Dev src, Dev dst, int nt)
        {
            var replay = new List<(long, IntPtr, long)>();
            Boundary b = _boundaries?[dst.Ordinal];
            if (b == null)
                return replay;
            long offset = 0;
            void Stage(IntPtr from, IntPtr to, long bytes)
            {
                if (bytes <= 0)
                    return;
                if (offset + bytes > src.StatePinnedBytes)
                    throw new InvalidOperationException($"[dsv4-cuda] boundary state overflows its staging ({offset + bytes} > {src.StatePinnedBytes})");
                CudaDriverApi.cuMemcpyDtoHAsync((IntPtr)((long)src.StatePinned + offset), from, new UIntPtr((ulong)bytes), src.Stream).ThrowOnError();
                replay.Add((offset, to, bytes));
                offset += bytes;
            }

            var m = _m;
            // The delayed gates the first block on dst collapses with.
            Stage(Ptr(src.PreFfn), Ptr(dst.PreFfn), (long)nt * HC * 4);
            foreach (int source in b.KvGroups)
            {
                foreach (var entry in _committed)
                {
                    if (entry.Source != source)
                        continue;
                    long kBytes = (long)m.HeadDim * 2, iBytes = (long)m.IdxHeadSize * 2;
                    Stage(At(CompKOn(entry.Slot, source, src), entry.FirstBlock * kBytes),
                        At(CompKOn(entry.Slot, source, dst), entry.FirstBlock * kBytes), entry.Count * kBytes);
                    Stage(At(LidKOn(entry.Slot, source, src), entry.FirstBlock * iBytes),
                        At(LidKOn(entry.Slot, source, dst), entry.FirstBlock * iBytes), entry.Count * iBytes);
                }
            }
            if (b.Selection)
            {
                int k = Math.Max(m.IdxTopK, 1);
                Stage(Ptr(src.TopkIdx), Ptr(dst.TopkIdx), (long)nt * k * 4);
                Stage(Ptr(src.TopkCnt), Ptr(dst.TopkCnt), (long)nt * 4);
            }
            if (b.CandidateMask && _v41CandActive)
            {
                long width = src.CandMask.Sizes[1];
                Stage(Ptr(src.CandMask), Ptr(dst.CandMask), nt * width);
            }
            return replay;
        }

        private static IntPtr At(Tensor t, long byteOffset) => (IntPtr)((long)Ptr(t) + byteOffset);

        /// <summary>Replay staged boundary state onto <paramref name="dst"/>'s stream (after it waited for the staging).</summary>
        private static void UnstageBoundaryState(Dev src, Dev dst, List<(long Offset, IntPtr Dst, long Bytes)> replay)
        {
            foreach (var (offset, to, bytes) in replay)
                CudaDriverApi.cuMemcpyHtoDAsync(to, (IntPtr)((long)src.StatePinned + offset), new UIntPtr((ulong)bytes), dst.Stream).ThrowOnError();
        }
    }
}
