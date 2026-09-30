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
    /// Supplies Qwen3.8-Flash-Next's PLE rows to the direct-CUDA engine.
    ///
    /// <para>The PLE table holds one row per (n-gram hash head, bucket): ~320 M rows, tens of
    /// GiB, of which a token reads one per head. It stays a host mapping and only the selected
    /// rows cross the bus. Which rows those are depends on the GGUF's hash constants and on the
    /// sequence's own token history, so the executor that owns the GGUF hashes and dequantizes,
    /// and the engine uploads what comes back.</para>
    /// </summary>
    public unsafe interface IQwen4ExpPleSource
    {
        /// <summary>Values per token: the rows of every head, concatenated (the model's hidden width).</summary>
        int RowWidth { get; }

        /// <summary>
        /// Writes row t of <paramref name="dst"/> ([count, <see cref="RowWidth"/>]): the rows of token
        /// <paramref name="tokens"/>[t] at position <paramref name="positions"/>[t] of the sequence whose
        /// history is <paramref name="histories"/>[t]. Each history already holds every token before
        /// and at the positions asked for.
        /// </summary>
        void GatherPleRows(Qwen4ExpPleHistory[] histories, ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions, float* dst);
    }

    /// <summary>One sequence's token ids by absolute position, for the PLE n-gram hash. The engine
    /// keeps one per sequence slot; a reset only has to empty it.</summary>
    public sealed class Qwen4ExpPleHistory
    {
        public int[] Tokens = Array.Empty<int>();
        public int Length;

        /// <summary>Record <paramref name="tokens"/> at positions <paramref name="start"/>, start + 1, ...</summary>
        public void Put(ReadOnlySpan<int> tokens, int start)
        {
            int end = start + tokens.Length;
            if (Tokens.Length < end)
            {
                var grown = new int[Math.Max(end, Math.Max(64, Tokens.Length * 2))];
                Array.Copy(Tokens, grown, Length);
                Tokens = grown;
            }
            tokens.CopyTo(Tokens.AsSpan(start));
            Length = end;
        }
    }
}
