// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Supplies DeepSeek V4.1's Engram rows to the direct-CUDA engine.
    ///
    /// <para>An Engram table is one row per n-gram hash bucket: hundreds of
    /// millions of rows, tens to hundreds of GiB, of which a token touches
    /// <see cref="HashColumns"/>. It is far too large to live in VRAM beside the
    /// model, so it stays a host mapping and only the selected rows cross the
    /// bus. Which rows those are depends on the embedded GGUF metadata and on
    /// the sequence's own token history, neither of which is the engine's
    /// business — so the executor that owns the GGUF does the hashing and the
    /// dequantization, and the engine only uploads what comes back.</para>
    /// </summary>
    public unsafe interface IDsv41EngramSource
    {
        /// <summary>Rows read per token per table: one per (lookback, head).</summary>
        int HashColumns { get; }

        /// <summary>Values per row.</summary>
        int HeadDim { get; }

        /// <summary>
        /// Extends one sequence's hash history by this ubatch, once, before any layer asks
        /// for rows. The history is indexed by absolute position, so a prompt split across
        /// ubatches hashes exactly as it would in one shot, and a rewind only has to shorten it.
        /// </summary>
        void BeginEngramUbatch(Dsv41EngramHistory history, ReadOnlySpan<int> tokens, int startPos);

        /// <summary>
        /// The batched-decode form: row i is <paramref name="tokens"/>[i] at
        /// <paramref name="positions"/>[i] of the sequence whose history is
        /// <paramref name="histories"/>[i].
        /// </summary>
        void BeginEngramRows(Dsv41EngramHistory[] histories, ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions);

        /// <summary>
        /// Writes this ubatch's rows for one table, laid out
        /// [token][hash column][head dim] — <paramref name="count"/> tokens of
        /// <see cref="HashColumns"/> * <see cref="HeadDim"/> floats each.
        /// </summary>
        void GatherEngramRows(int engramIndex, int count, float* dst);

        /// <summary>
        /// Writes this ubatch's selected row indices into one table, laid out [token][hash column]:
        /// <paramref name="count"/> tokens of <see cref="HashColumns"/> ints each. The engine gathers
        /// the rows of a device-resident table itself from these.
        /// </summary>
        void CopyEngramRowIndices(int engramIndex, int count, int* dst);
    }

    /// <summary>One sequence's Engram hash history: its (mapped) token ids by absolute
    /// position. The engine keeps one per sequence slot.</summary>
    public sealed class Dsv41EngramHistory
    {
        public int[] Tokens;
        public int Length;
    }
}
