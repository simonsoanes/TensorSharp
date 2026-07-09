// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// Parity battery for the norm/attention module (GgmlManagedOps.NormAttn.cs
// vs ggml_ops_norm_attn.cpp): Norm, NormGrad, RoPE (forward + grad), RoPEEx,
// RoPEEx with freq factors, MRoPE, ScaledDotProductAttention, Softmax,
// AttentionSoftmaxWithSinks, SoftmaxGrad, and the two fused prefill
// attention entry points.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TensorSharp.GGML;

internal static class CasesNormAttn
{
    public static void Run(Harness.Dump dump)
    {
        // TS_PARITY_NORMATTN_SKIP: comma-separated group names to skip
        // (debug aid for bisecting cross-case interactions).
        string skipEnv = Environment.GetEnvironmentVariable("TS_PARITY_NORMATTN_SKIP") ?? string.Empty;
        var skip = new HashSet<string>(skipEnv.Split(',', StringSplitOptions.RemoveEmptyEntries), StringComparer.OrdinalIgnoreCase);

        if (!skip.Contains("norm")) RunNormCases(dump);
        if (!skip.Contains("normgrad")) RunNormGradCases(dump);
        if (!skip.Contains("rope")) RunRoPECases(dump);
        if (!skip.Contains("ropeex")) RunRoPEExCases(dump);
        if (!skip.Contains("ropeexff")) RunRoPEExFreqFactorsCases(dump);
        if (!skip.Contains("mrope")) RunRoPEMRoPECases(dump);
        if (!skip.Contains("sdpa")) RunScaledDotProductAttentionCases(dump);
        if (!skip.Contains("softmax")) RunSoftmaxCases(dump);
        if (!skip.Contains("sinks")) RunAttentionSoftmaxWithSinksCases(dump);
        if (!skip.Contains("softmaxgrad")) RunSoftmaxGradCases(dump);
        if (!skip.Contains("prefill")) RunFusedPrefillAttentionCases(dump);
        if (!skip.Contains("prefillf16")) RunFusedPrefillAttentionF16KVCases(dump);
    }

    // Single-shot timing for ops that mutate their inputs/outputs in place
    // (accumulating runs are not idempotent, so preserve the deterministic
    // state produced by exactly one call for the byte-level comparison).
    private static double TimeOnce(Action action)
    {
        long start = Stopwatch.GetTimestamp();
        action();
        return (Stopwatch.GetTimestamp() - start) * 1e9 / Stopwatch.Frequency;
    }

    // Deterministic F16 fill: generate the shared xorshift floats, then round
    // to half bits host-side. Runs identically in the native and managed
    // children, so both feed bit-identical F16 K/V data.
    private static void FillDeterministicF16(Span<ushort> dst, float offset = 0f, float scale = 1f)
    {
        float[] tmp = new float[dst.Length];
        Harness.FillDeterministic(tmp, offset, scale);
        for (int i = 0; i < dst.Length; ++i)
            dst[i] = BitConverter.HalfToUInt16Bits((Half)tmp[i]);
    }

    // --------------------------------------------------------------
    // Norm (LayerNorm / RmsNorm, with and without beta)
    // --------------------------------------------------------------

    public static void RunNormCases(Harness.Dump dump)
    {
        const int Hidden = 128, Rows = 9;
        long bytes = Harness.Bytes4D(Hidden, Rows, 1, 1);
        long vecBytes = Harness.Bytes4D(Hidden, 1, 1, 1);
        using var src = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);
        using var gamma = new Harness.AlignedBuffer(vecBytes);
        using var beta = new Harness.AlignedBuffer(vecBytes);

        foreach (GgmlNormOp op in Enum.GetValues<GgmlNormOp>())
        {
            foreach (bool hasBeta in new[] { false, true })
            {
                Harness.FillDeterministic(src.Floats);
                Harness.FillDeterministic(gamma.Floats, offset: 1f, scale: 0.25f);
                Harness.FillDeterministic(beta.Floats, scale: 0.25f);
                dst.Floats.Clear();

                GgmlTensorView4D srcView = Harness.Contiguous4D(src.Ptr, Hidden, Rows, 1, 1);
                GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, Hidden, Rows, 1, 1);
                GgmlTensorView4D gammaView = Harness.Contiguous4D(gamma.Ptr, Hidden, 1, 1, 1);
                GgmlTensorView4D betaView = Harness.Contiguous4D(beta.Ptr, Hidden, 1, 1, 1);
                double ns = Harness.TimeCase(() => GgmlNative.Norm(op, dstView, srcView, gammaView, betaView, hasBeta, 1e-5f));
                dump.Add($"normattn/norm/{op}-beta={hasBeta}", dst.RawBytes, ns);
            }
        }
    }

    // --------------------------------------------------------------
    // NormGrad (accumulates into dx/gradGamma/gradBeta -> single-shot)
    // --------------------------------------------------------------

    public static void RunNormGradCases(Harness.Dump dump)
    {
        const int Hidden = 128, Rows = 9;
        long bytes = Harness.Bytes4D(Hidden, Rows, 1, 1);
        long vecBytes = Harness.Bytes4D(Hidden, 1, 1, 1);
        using var dx = new Harness.AlignedBuffer(bytes);
        using var gradGamma = new Harness.AlignedBuffer(vecBytes);
        using var gradBeta = new Harness.AlignedBuffer(vecBytes);
        using var adj = new Harness.AlignedBuffer(bytes);
        using var x = new Harness.AlignedBuffer(bytes);
        using var gamma = new Harness.AlignedBuffer(vecBytes);

        foreach (GgmlNormOp op in Enum.GetValues<GgmlNormOp>())
        {
            // The native TSGgml_NormGradF32 LayerNorm branch relies on ggml's
            // autodiff, and the vendored ggml aborts the process with
            // "unsupported ggml op for backward pass: NORM" — a latent native
            // bug in a path no inference caller reaches. The managed port
            // mirrors the same graph construction, but the native child dies
            // before producing a dump, so only the RmsNorm branch (explicit
            // backward graph) is comparable here.
            if (op == GgmlNormOp.LayerNorm)
                continue;

            foreach (bool hasGradBeta in new[] { false, true })
            {
                Harness.FillDeterministic(dx.Floats, scale: 0.5f);
                Harness.FillDeterministic(gradGamma.Floats, scale: 0.5f);
                Harness.FillDeterministic(gradBeta.Floats, scale: 0.5f);
                Harness.FillDeterministic(adj.Floats);
                Harness.FillDeterministic(x.Floats);
                Harness.FillDeterministic(gamma.Floats, offset: 1f, scale: 0.25f);

                GgmlTensorView4D dxView = Harness.Contiguous4D(dx.Ptr, Hidden, Rows, 1, 1);
                GgmlTensorView4D gradGammaView = Harness.Contiguous4D(gradGamma.Ptr, Hidden, 1, 1, 1);
                GgmlTensorView4D gradBetaView = Harness.Contiguous4D(gradBeta.Ptr, Hidden, 1, 1, 1);
                GgmlTensorView4D adjView = Harness.Contiguous4D(adj.Ptr, Hidden, Rows, 1, 1);
                GgmlTensorView4D xView = Harness.Contiguous4D(x.Ptr, Hidden, Rows, 1, 1);
                GgmlTensorView4D gammaView = Harness.Contiguous4D(gamma.Ptr, Hidden, 1, 1, 1);

                double ns = TimeOnce(() => GgmlNative.NormGrad(op, dxView, gradGammaView, gradBetaView, adjView, xView, gammaView, hasGradBeta, 1e-5f));
                dump.Add($"normattn/normgrad/{op}-beta={hasGradBeta}/dx", dx.RawBytes, ns);
                dump.Add($"normattn/normgrad/{op}-beta={hasGradBeta}/ggamma", gradGamma.RawBytes, ns);
                if (hasGradBeta)
                    dump.Add($"normattn/normgrad/{op}-beta={hasGradBeta}/gbeta", gradBeta.RawBytes, ns);
            }
        }
    }

    // --------------------------------------------------------------
    // RoPE forward + RoPEGrad (both route through TSGgml_RoPEF32)
    // --------------------------------------------------------------

    public static void RunRoPECases(Harness.Dump dump)
    {
        const int HeadDim = 64, Seq = 16, Heads = 4;
        long bytes = Harness.Bytes4D(HeadDim, Seq, Heads, 1);
        using var src = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);

        foreach (int rowOffset in new[] { 0, 5 })
        {
            Harness.FillDeterministic(src.Floats);
            dst.Floats.Clear();
            GgmlTensorView4D srcView = Harness.Contiguous4D(src.Ptr, HeadDim, Seq, Heads, 1);
            GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, HeadDim, Seq, Heads, 1);
            double ns = Harness.TimeCase(() => GgmlNative.RoPE(dstView, srcView, Seq, rowOffset));
            dump.Add($"normattn/rope/forward-offset{rowOffset}", dst.RawBytes, ns);
        }

        // Backward: RoPEGrad accumulates into the result buffer, so single-shot.
        {
            Harness.FillDeterministic(src.Floats);
            Harness.FillDeterministic(dst.Floats, offset: 2f, scale: 0.5f);
            GgmlTensorView4D adjView = Harness.Contiguous4D(src.Ptr, HeadDim, Seq, Heads, 1);
            GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, HeadDim, Seq, Heads, 1);
            double ns = TimeOnce(() => GgmlNative.RoPEGrad(dstView, adjView, Seq, 0));
            dump.Add("normattn/rope/grad", dst.RawBytes, ns);
        }
    }

    // --------------------------------------------------------------
    // RoPEEx (explicit positions tensor, full parameter set)
    // --------------------------------------------------------------

    public static unsafe void RunRoPEExCases(Harness.Dump dump)
    {
        const int HeadDim = 64, Seq = 16, Heads = 4;
        const int Rows = Seq * Heads;
        long bytes = Harness.Bytes4D(HeadDim, Seq, Heads, 1);
        using var src = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);
        using var positions = new Harness.AlignedBuffer(Rows * sizeof(int));

        Span<int> positionValues = new Span<int>((void*)positions.Ptr, Rows);
        for (int i = 0; i < Rows; ++i)
            positionValues[i] = i % Seq;
        var positionsDesc = new GgmlContiguousTensor(positions.Ptr, Rows, TensorSharp.DType.Int32);

        // (name, ropeDim, mode, addToResult, invertPositions)
        (string Name, int RopeDim, int Mode, bool AddToResult, bool Invert)[] cases =
        {
            ("normal", HeadDim, 0, false, false),
            ("neox", HeadDim, 2, false, false),
            ("neox-partialdim32", 32, 2, false, false),
            ("neox-invert", HeadDim, 2, false, true),
            ("neox-addtoresult", HeadDim, 2, true, false),
        };

        foreach ((string name, int ropeDim, int mode, bool addToResult, bool invert) in cases)
        {
            Harness.FillDeterministic(src.Floats);
            if (addToResult)
                Harness.FillDeterministic(dst.Floats, offset: 2f, scale: 0.5f);
            else
                dst.Floats.Clear();

            GgmlTensorView4D srcView = Harness.Contiguous4D(src.Ptr, HeadDim, Seq, Heads, 1);
            GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, HeadDim, Seq, Heads, 1);

            double ns;
            if (addToResult)
            {
                ns = TimeOnce(() => GgmlNative.RoPEEx(
                    dstView, srcView, positionsDesc, ropeDim, mode,
                    originalContextLength: 4096,
                    freqBase: 10000f, freqScale: 1f,
                    extFactor: 0f, attnFactor: 1f,
                    betaFast: 32f, betaSlow: 1f,
                    addToResult: true, invertPositions: invert));
            }
            else
            {
                ns = Harness.TimeCase(() => GgmlNative.RoPEEx(
                    dstView, srcView, positionsDesc, ropeDim, mode,
                    originalContextLength: 4096,
                    freqBase: 10000f, freqScale: 1f,
                    extFactor: 0f, attnFactor: 1f,
                    betaFast: 32f, betaSlow: 1f,
                    addToResult: false, invertPositions: invert));
            }
            dump.Add($"normattn/ropeex/{name}", dst.RawBytes, ns);
        }
    }

    // --------------------------------------------------------------
    // RoPEEx with a frequency-factors tensor (ropeDim/2 entries)
    // --------------------------------------------------------------

    public static unsafe void RunRoPEExFreqFactorsCases(Harness.Dump dump)
    {
        const int HeadDim = 64, Seq = 16, Heads = 4;
        const int Rows = Seq * Heads;
        const int RopeDim = 64;
        long bytes = Harness.Bytes4D(HeadDim, Seq, Heads, 1);
        using var src = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);
        using var positions = new Harness.AlignedBuffer(Rows * sizeof(int));
        using var freqFactors = new Harness.AlignedBuffer((RopeDim / 2) * sizeof(float));

        Span<int> positionValues = new Span<int>((void*)positions.Ptr, Rows);
        for (int i = 0; i < Rows; ++i)
            positionValues[i] = i % Seq;
        var positionsDesc = new GgmlContiguousTensor(positions.Ptr, Rows, TensorSharp.DType.Int32);

        // Positive scaling factors around 1.0 (LongRoPE-style).
        Harness.FillDeterministic(freqFactors.Floats, offset: 1.5f, scale: 0.25f);

        Harness.FillDeterministic(src.Floats);
        dst.Floats.Clear();

        GgmlTensorView4D srcView = Harness.Contiguous4D(src.Ptr, HeadDim, Seq, Heads, 1);
        GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, HeadDim, Seq, Heads, 1);
        double ns = Harness.TimeCase(() => GgmlNative.RoPEExWithFreqFactors(
            dstView, srcView, positionsDesc, RopeDim, /*mode*/ 2,
            originalContextLength: 4096,
            freqBase: 10000f, freqScale: 1f,
            extFactor: 0f, attnFactor: 1f,
            betaFast: 32f, betaSlow: 1f,
            addToResult: false, invertPositions: false,
            freqFactors.Ptr, RopeDim / 2));
        dump.Add("normattn/ropeexff/neox", dst.RawBytes, ns);
    }

    // --------------------------------------------------------------
    // MRoPE (ggml_rope_multi): input [headDim, numHeads, seqLen, 1],
    // positions length 4*seqLen (per-axis concatenated), sections sum
    // to ropeDim/2.
    // --------------------------------------------------------------

    public static unsafe void RunRoPEMRoPECases(Harness.Dump dump)
    {
        const int HeadDim = 64, Heads = 4, Seq = 16;
        const int RopeDim = 64;
        long bytes = Harness.Bytes4D(HeadDim, Heads, Seq, 1);
        using var src = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);
        using var positions = new Harness.AlignedBuffer(4 * Seq * sizeof(int));

        // Per-axis position table: T / H / W axes get distinct values so a
        // swapped-axis bug shows up; the 4th axis stays 0 (static images).
        Span<int> positionValues = new Span<int>((void*)positions.Ptr, 4 * Seq);
        for (int t = 0; t < Seq; ++t)
        {
            positionValues[0 * Seq + t] = t;
            positionValues[1 * Seq + t] = t / 2;
            positionValues[2 * Seq + t] = t / 3;
            positionValues[3 * Seq + t] = 0;
        }
        var positionsDesc = new GgmlContiguousTensor(positions.Ptr, 4 * Seq, TensorSharp.DType.Int32);

        Harness.FillDeterministic(src.Floats);
        dst.Floats.Clear();

        GgmlTensorView4D srcView = Harness.Contiguous4D(src.Ptr, HeadDim, Heads, Seq, 1);
        GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, HeadDim, Heads, Seq, 1);
        // sections {8, 12, 12, 0} sum to RopeDim/2 = 32; mode 8 = GGML_ROPE_TYPE_MROPE.
        double ns = Harness.TimeCase(() => GgmlNative.RoPEMRoPE(
            dstView, srcView, positionsDesc, RopeDim, /*mode*/ 8,
            8, 12, 12, 0,
            originalContextLength: 4096,
            freqBase: 10000f, freqScale: 1f,
            extFactor: 0f, attnFactor: 1f,
            betaFast: 32f, betaSlow: 1f));
        dump.Add("normattn/mrope/sections8-12-12-0", dst.RawBytes, ns);
    }

    // --------------------------------------------------------------
    // ScaledDotProductAttention: q/k/v/result [head_dim, heads, seq, batch],
    // mask [seq_k, seq_q, heads, batch] (additive F32).
    // --------------------------------------------------------------

    public static void RunScaledDotProductAttentionCases(Harness.Dump dump)
    {
        const int HeadDim = 64, Heads = 4, Seq = 16;
        const float Scale = 0.125f; // 1/sqrt(64)
        long qkvBytes = Harness.Bytes4D(HeadDim, Heads, Seq, 1);
        long maskBytes = Harness.Bytes4D(Seq, Seq, Heads, 1);
        using var query = new Harness.AlignedBuffer(qkvBytes);
        using var key = new Harness.AlignedBuffer(qkvBytes);
        using var value = new Harness.AlignedBuffer(qkvBytes);
        using var mask = new Harness.AlignedBuffer(maskBytes);
        using var dst = new Harness.AlignedBuffer(qkvBytes);

        foreach (bool hasMask in new[] { false, true })
        {
            Harness.FillDeterministic(query.Floats);
            Harness.FillDeterministic(key.Floats);
            Harness.FillDeterministic(value.Floats);
            Harness.FillDeterministic(mask.Floats, scale: 0.5f);
            dst.Floats.Clear();

            GgmlTensorView4D queryView = Harness.Contiguous4D(query.Ptr, HeadDim, Heads, Seq, 1);
            GgmlTensorView4D keyView = Harness.Contiguous4D(key.Ptr, HeadDim, Heads, Seq, 1);
            GgmlTensorView4D valueView = Harness.Contiguous4D(value.Ptr, HeadDim, Heads, Seq, 1);
            GgmlTensorView4D maskView = Harness.Contiguous4D(mask.Ptr, Seq, Seq, Heads, 1);
            GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, HeadDim, Heads, Seq, 1);
            double ns = Harness.TimeCase(() => GgmlNative.ScaledDotProductAttention(dstView, queryView, keyView, valueView, maskView, hasMask, Scale));
            dump.Add($"normattn/sdpa/mask={hasMask}", dst.RawBytes, ns);
        }
    }

    // --------------------------------------------------------------
    // Softmax (plain row-wise ggml_soft_max)
    // --------------------------------------------------------------

    public static void RunSoftmaxCases(Harness.Dump dump)
    {
        const int Ne0 = 65, Ne1 = 9, Ne2 = 4, Ne3 = 1;
        long bytes = Harness.Bytes4D(Ne0, Ne1, Ne2, Ne3);
        using var src = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);

        Harness.FillDeterministic(src.Floats, scale: 2f);
        dst.Floats.Clear();
        GgmlTensorView4D srcView = Harness.Contiguous4D(src.Ptr, Ne0, Ne1, Ne2, Ne3);
        GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, Ne0, Ne1, Ne2, Ne3);
        double ns = Harness.TimeCase(() => GgmlNative.Softmax(dstView, srcView));
        dump.Add("normattn/softmax/basic", dst.RawBytes, ns);
    }

    // --------------------------------------------------------------
    // AttentionSoftmaxWithSinks: in-place on a [numHeads, seqLen, kvLen]
    // scores tensor, so single-shot with a fresh deterministic fill per case.
    // --------------------------------------------------------------

    public static void RunAttentionSoftmaxWithSinksCases(Harness.Dump dump)
    {
        const int Heads = 4, Seq = 16, Kv = 24;
        const int MaskStartPos = Kv - Seq;
        const float Scale = 0.125f;
        long scoresBytes = (long)Heads * Seq * Kv * sizeof(float);
        using var scores = new Harness.AlignedBuffer(scoresBytes);
        using var sinks = new Harness.AlignedBuffer(Heads * sizeof(float));

        Harness.FillDeterministic(sinks.Floats, scale: 0.5f);

        foreach ((bool useSinks, int slidingWindow) in new[] { (true, 0), (true, 8), (false, 0) })
        {
            Harness.FillDeterministic(scores.Floats, scale: 2f);
            var scoresView = new GgmlTensorView3D(scores.Ptr, Heads, Seq, Kv, Seq * Kv, Kv, 1, scoresBytes);
            double ns = TimeOnce(() => GgmlNative.AttentionSoftmaxWithSinks(
                scoresView, useSinks ? sinks.Ptr : IntPtr.Zero,
                Heads, Seq, Kv, MaskStartPos, slidingWindow, Scale));
            dump.Add($"normattn/sinks/sinks={useSinks}-sw={slidingWindow}", scores.RawBytes, ns);
        }
    }

    // --------------------------------------------------------------
    // SoftmaxGrad (addGrad accumulates into the result -> single-shot)
    // --------------------------------------------------------------

    public static void RunSoftmaxGradCases(Harness.Dump dump)
    {
        const int Ne0 = 65, Ne1 = 9, Ne2 = 2, Ne3 = 1;
        long bytes = Harness.Bytes4D(Ne0, Ne1, Ne2, Ne3);
        using var adj = new Harness.AlignedBuffer(bytes);
        using var val = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);

        foreach (bool addGrad in new[] { false, true })
        {
            Harness.FillDeterministic(adj.Floats);
            Harness.FillDeterministic(val.Floats, offset: 0.5f, scale: 0.4f);
            if (addGrad)
                Harness.FillDeterministic(dst.Floats, offset: 1f, scale: 0.5f);
            else
                dst.Floats.Clear();

            GgmlTensorView4D adjView = Harness.Contiguous4D(adj.Ptr, Ne0, Ne1, Ne2, Ne3);
            GgmlTensorView4D valView = Harness.Contiguous4D(val.Ptr, Ne0, Ne1, Ne2, Ne3);
            GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, Ne0, Ne1, Ne2, Ne3);
            double ns = addGrad
                ? TimeOnce(() => GgmlNative.SoftmaxGrad(dstView, adjView, valView, addGrad: true))
                : Harness.TimeCase(() => GgmlNative.SoftmaxGrad(dstView, adjView, valView, addGrad: false));
            dump.Add($"normattn/softmaxgrad/add={addGrad}", dst.RawBytes, ns);
        }
    }

    // --------------------------------------------------------------
    // FusedPrefillAttention (F32): head-first (session-cached path), the
    // sliding-window variant, an MHA (numHeads == numKVHeads) shape, and the
    // flat input format (materialized-graph path).
    // --------------------------------------------------------------

    public static void RunFusedPrefillAttentionCases(Harness.Dump dump)
    {
        const int HeadDim = 64, Seq = 8, Kv = 16;
        const int MaskStartPos = Kv - Seq;
        const float Scale = 0.125f;

        // (name, numHeads, numKvHeads, slidingWindow, inputFormat)
        (string Name, int Heads, int KvHeads, int SlidingWindow, int InputFormat)[] cases =
        {
            ("f32-headfirst-gqa", 4, 2, 0, 0),
            ("f32-headfirst-sw4", 4, 2, 4, 0),
            ("f32-headfirst-mha", 4, 4, 0, 0),
            ("f32-flat", 4, 2, 0, 1),
        };

        foreach ((string name, int heads, int kvHeads, int slidingWindow, int inputFormat) in cases)
        {
            long qBytes = (long)heads * Seq * HeadDim * sizeof(float);
            long kvBytes = (long)kvHeads * Kv * HeadDim * sizeof(float);
            long outBytes = (long)Seq * heads * HeadDim * sizeof(float);
            using var q = new Harness.AlignedBuffer(qBytes);
            using var k = new Harness.AlignedBuffer(kvBytes);
            using var v = new Harness.AlignedBuffer(kvBytes);
            using var outBuf = new Harness.AlignedBuffer(outBytes);

            Harness.FillDeterministic(q.Floats);
            Harness.FillDeterministic(k.Floats);
            Harness.FillDeterministic(v.Floats);
            outBuf.Floats.Clear();

            double ns = Harness.TimeCase(() => GgmlNative.FusedPrefillAttention(
                q.Ptr, k.Ptr, v.Ptr, outBuf.Ptr,
                heads, kvHeads, HeadDim,
                Seq, Kv,
                MaskStartPos, slidingWindow,
                Scale, inputFormat));
            dump.Add($"normattn/prefill/{name}", outBuf.RawBytes, ns);
        }
    }

    // --------------------------------------------------------------
    // FusedPrefillAttentionF16KV: K/V read straight from an F16 cache, with
    // an exactly-sized cache (contiguous upload) and an over-allocated cache
    // (per-head strided upload).
    // --------------------------------------------------------------

    public static unsafe void RunFusedPrefillAttentionF16KVCases(Harness.Dump dump)
    {
        const int Heads = 4, KvHeads = 2, HeadDim = 64, Seq = 8, Kv = 16;
        const int MaskStartPos = Kv - Seq;
        const float Scale = 0.125f;

        foreach (int kvCacheLen in new[] { Kv, Kv + 8 })
        {
            long qBytes = (long)Heads * Seq * HeadDim * sizeof(float);
            long kvBytes = (long)KvHeads * kvCacheLen * HeadDim * sizeof(ushort);
            long outBytes = (long)Seq * Heads * HeadDim * sizeof(float);
            using var q = new Harness.AlignedBuffer(qBytes);
            using var k = new Harness.AlignedBuffer(kvBytes);
            using var v = new Harness.AlignedBuffer(kvBytes);
            using var outBuf = new Harness.AlignedBuffer(outBytes);

            Harness.FillDeterministic(q.Floats);
            FillDeterministicF16(new Span<ushort>((void*)k.Ptr, (int)(kvBytes / sizeof(ushort))));
            FillDeterministicF16(new Span<ushort>((void*)v.Ptr, (int)(kvBytes / sizeof(ushort))));
            outBuf.Floats.Clear();

            double ns = Harness.TimeCase(() => GgmlNative.FusedPrefillAttentionF16KV(
                q.Ptr, k.Ptr, v.Ptr, outBuf.Ptr,
                Heads, KvHeads, HeadDim,
                Seq, Kv, kvCacheLen,
                MaskStartPos, /*slidingWindow*/ 0, Scale));
            dump.Add($"normattn/prefill/f16kv-cache{kvCacheLen}", outBuf.RawBytes, ns);
        }
    }
}
