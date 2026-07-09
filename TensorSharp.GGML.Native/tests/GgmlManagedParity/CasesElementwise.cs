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
// Parity battery for the elementwise module (GgmlManagedOps.Elementwise.cs
// vs ggml_ops_elementwise.cpp).
using System;
using System.Diagnostics;
using TensorSharp.GGML;

internal static class CasesElementwise
{
    public static void Run(Harness.Dump dump)
    {
        RunUnaryCases(dump);
        RunBinaryTensorCases(dump);
        RunBinaryScalarCases(dump);
        RunFusedActMulCases(dump);
        RunFusedActMulSplitCases(dump);
        RunCopyCases(dump);
        RunReductionCases(dump);
        RunIndexReductionCases(dump);
        RunActivationGradCases(dump);
        RunIndexSelectCases(dump);
        RunIndexSelectGradCases(dump);
        RunPerfCases(dump);
    }

    public static void RunUnaryCases(Harness.Dump dump)
    {
        const int Ne0 = 33, Ne1 = 7, Ne2 = 3, Ne3 = 2;
        long bytes = Harness.Bytes4D(Ne0, Ne1, Ne2, Ne3);
        using var src = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);

        foreach (GgmlUnaryOp op in Enum.GetValues<GgmlUnaryOp>())
        {
            // Log/Sqrt need strictly positive inputs.
            bool positive = op == GgmlUnaryOp.Log || op == GgmlUnaryOp.Sqrt;
            Harness.FillDeterministic(src.Floats, offset: positive ? 1.5f : 0f);
            dst.Floats.Clear();

            GgmlTensorView4D srcView = Harness.Contiguous4D(src.Ptr, Ne0, Ne1, Ne2, Ne3);
            GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, Ne0, Ne1, Ne2, Ne3);
            double ns = Harness.TimeCase(() => GgmlNative.Unary(op, dstView, srcView));
            dump.Add($"unary/{op}", dst.RawBytes, ns);
        }
    }

    public static void RunBinaryTensorCases(Harness.Dump dump)
    {
        const int Ne0 = 33, Ne1 = 7, Ne2 = 3, Ne3 = 2;
        long bytes = Harness.Bytes4D(Ne0, Ne1, Ne2, Ne3);
        using var lhs = new Harness.AlignedBuffer(bytes);
        using var rhs = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);

        foreach (GgmlBinaryTensorOp op in Enum.GetValues<GgmlBinaryTensorOp>())
        {
            Harness.FillDeterministic(lhs.Floats);
            Harness.FillDeterministic(rhs.Floats, offset: op == GgmlBinaryTensorOp.Div ? 3f : 0f);
            dst.Floats.Clear();

            GgmlTensorView4D lhsView = Harness.Contiguous4D(lhs.Ptr, Ne0, Ne1, Ne2, Ne3);
            GgmlTensorView4D rhsView = Harness.Contiguous4D(rhs.Ptr, Ne0, Ne1, Ne2, Ne3);
            GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, Ne0, Ne1, Ne2, Ne3);
            double ns = Harness.TimeCase(() => GgmlNative.BinaryTensor(op, dstView, lhsView, rhsView));
            dump.Add($"binary/{op}", dst.RawBytes, ns);
        }

        // Broadcast rhs: one row vector repeated across all rows.
        {
            long rowBytes = Harness.Bytes4D(Ne0, 1, 1, 1);
            using var rhsRow = new Harness.AlignedBuffer(rowBytes);
            Harness.FillDeterministic(lhs.Floats);
            Harness.FillDeterministic(rhsRow.Floats);
            dst.Floats.Clear();

            GgmlTensorView4D lhsView = Harness.Contiguous4D(lhs.Ptr, Ne0, Ne1, Ne2, Ne3);
            GgmlTensorView4D rhsView = Harness.Contiguous4D(rhsRow.Ptr, Ne0, 1, 1, 1);
            GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, Ne0, Ne1, Ne2, Ne3);
            double ns = Harness.TimeCase(() => GgmlNative.BinaryTensor(GgmlBinaryTensorOp.Add, dstView, lhsView, rhsView));
            dump.Add("binary/Add-broadcast", dst.RawBytes, ns);
        }
    }

    public static void RunBinaryScalarCases(Harness.Dump dump)
    {
        const int Ne0 = 33, Ne1 = 7, Ne2 = 3, Ne3 = 2;
        long bytes = Harness.Bytes4D(Ne0, Ne1, Ne2, Ne3);
        using var src = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);

        foreach (GgmlBinaryScalarOp op in Enum.GetValues<GgmlBinaryScalarOp>())
        {
            bool reverseDiv = op == GgmlBinaryScalarOp.ReverseDiv;
            Harness.FillDeterministic(src.Floats, offset: reverseDiv ? 3f : 0f);
            dst.Floats.Clear();

            GgmlTensorView4D srcView = Harness.Contiguous4D(src.Ptr, Ne0, Ne1, Ne2, Ne3);
            GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, Ne0, Ne1, Ne2, Ne3);
            double ns = Harness.TimeCase(() => GgmlNative.BinaryScalar(op, dstView, srcView, 0.37f));
            dump.Add($"binaryscalar/{op}", dst.RawBytes, ns);
        }
    }

    public static void RunFusedActMulCases(Harness.Dump dump)
    {
        const int Ne0 = 65, Ne1 = 9, Ne2 = 2, Ne3 = 1;
        long bytes = Harness.Bytes4D(Ne0, Ne1, Ne2, Ne3);
        using var a = new Harness.AlignedBuffer(bytes);
        using var b = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);

        foreach (GgmlFusedActMulOp op in Enum.GetValues<GgmlFusedActMulOp>())
        {
            Harness.FillDeterministic(a.Floats);
            Harness.FillDeterministic(b.Floats);
            dst.Floats.Clear();

            GgmlTensorView4D aView = Harness.Contiguous4D(a.Ptr, Ne0, Ne1, Ne2, Ne3);
            GgmlTensorView4D bView = Harness.Contiguous4D(b.Ptr, Ne0, Ne1, Ne2, Ne3);
            GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, Ne0, Ne1, Ne2, Ne3);
            double ns = Harness.TimeCase(() => GgmlNative.FusedActMul(op, dstView, aView, bView));
            dump.Add($"actmul/{op}", dst.RawBytes, ns);
        }
    }

    public static void RunFusedActMulSplitCases(Harness.Dump dump)
    {
        const int Rows = 9, HalfDim = 64;
        foreach (int rowStride in new[] { 2 * HalfDim, 2 * HalfDim + 16 })
        {
            long gateUpBytes = (long)Rows * rowStride * sizeof(float);
            long dstBytes = (long)Rows * HalfDim * sizeof(float);
            using var gateUp = new Harness.AlignedBuffer(gateUpBytes);
            using var dst = new Harness.AlignedBuffer(dstBytes);

            foreach (GgmlFusedActMulOp op in Enum.GetValues<GgmlFusedActMulOp>())
            {
                Harness.FillDeterministic(gateUp.Floats);
                dst.Floats.Clear();

                var gateUpView = new GgmlTensorView2D(gateUp.Ptr, Rows, 2 * HalfDim, rowStride, 1, gateUpBytes);
                var dstView = new GgmlTensorView2D(dst.Ptr, Rows, HalfDim, HalfDim, 1, dstBytes);
                double ns = Harness.TimeCase(() => GgmlNative.FusedActMulSplit(op, dstView, gateUpView, HalfDim));
                dump.Add($"actmulsplit/{op}-stride{rowStride}", dst.RawBytes, ns);
            }
        }
    }

    public static void RunCopyCases(Harness.Dump dump)
    {
        const int Ne0 = 33, Ne1 = 7, Ne2 = 3, Ne3 = 2;
        long bytes = Harness.Bytes4D(Ne0, Ne1, Ne2, Ne3);
        using var src = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);

        Harness.FillDeterministic(src.Floats);
        dst.Floats.Clear();
        GgmlTensorView4D srcView = Harness.Contiguous4D(src.Ptr, Ne0, Ne1, Ne2, Ne3);
        GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, Ne0, Ne1, Ne2, Ne3);
        double ns = Harness.TimeCase(() => GgmlNative.Copy(dstView, srcView));
        dump.Add("copy/contiguous", dst.RawBytes, ns);

        // Padded destination rows: nb1 spans more elements than ne0.
        const int Pad = 8;
        long paddedBytes = (long)(Ne0 + Pad) * Ne1 * Ne2 * Ne3 * sizeof(float);
        using var paddedDst = new Harness.AlignedBuffer(paddedBytes);
        paddedDst.Floats.Clear();
        long nb1 = (long)(Ne0 + Pad) * sizeof(float);
        var paddedView = new GgmlTensorView4D(paddedDst.Ptr, Ne0, Ne1, Ne2, Ne3, nb1, nb1 * Ne1, nb1 * Ne1 * Ne2, paddedBytes);
        double nsPadded = Harness.TimeCase(() => GgmlNative.Copy(paddedView, srcView));
        dump.Add("copy/padded-dst", paddedDst.RawBytes, nsPadded);
    }

    public static void RunReductionCases(Harness.Dump dump)
    {
        const int Ne0 = 65, Ne1 = 7, Ne2 = 3, Ne3 = 2;
        long srcBytes = Harness.Bytes4D(Ne0, Ne1, Ne2, Ne3);
        long dstBytes = Harness.Bytes4D(1, Ne1, Ne2, Ne3);
        using var src = new Harness.AlignedBuffer(srcBytes);
        using var dst = new Harness.AlignedBuffer(dstBytes);

        foreach (GgmlReductionOp op in Enum.GetValues<GgmlReductionOp>())
        {
            Harness.FillDeterministic(src.Floats);
            dst.Floats.Clear();
            GgmlTensorView4D srcView = Harness.Contiguous4D(src.Ptr, Ne0, Ne1, Ne2, Ne3);
            GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, 1, Ne1, Ne2, Ne3);
            double ns = Harness.TimeCase(() => GgmlNative.ReduceLastDim(op, dstView, srcView));
            dump.Add($"reduce/{op}", dst.RawBytes, ns);
        }
    }

    public static void RunIndexReductionCases(Harness.Dump dump)
    {
        const int Ne0 = 65, Ne1 = 7, Ne2 = 3, Ne3 = 2;
        long srcBytes = Harness.Bytes4D(Ne0, Ne1, Ne2, Ne3);
        long dstBytes = Harness.Bytes4D(1, Ne1, Ne2, Ne3);
        using var src = new Harness.AlignedBuffer(srcBytes);
        using var dst = new Harness.AlignedBuffer(dstBytes);

        foreach (GgmlIndexReductionOp op in Enum.GetValues<GgmlIndexReductionOp>())
        {
            Harness.FillDeterministic(src.Floats);
            dst.Floats.Clear();
            GgmlTensorView4D srcView = Harness.Contiguous4D(src.Ptr, Ne0, Ne1, Ne2, Ne3);
            GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, 1, Ne1, Ne2, Ne3);
            double ns = Harness.TimeCase(() => GgmlNative.IndexReduction(op, dstView, srcView));
            dump.Add($"indexreduce/{op}", dst.RawBytes, ns);
        }
    }

    public static void RunActivationGradCases(Harness.Dump dump)
    {
        const int Ne0 = 33, Ne1 = 5, Ne2 = 2, Ne3 = 1;
        long bytes = Harness.Bytes4D(Ne0, Ne1, Ne2, Ne3);
        using var src = new Harness.AlignedBuffer(bytes);
        using var grad = new Harness.AlignedBuffer(bytes);
        using var acc = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);

        foreach (GgmlActivationGradOp op in Enum.GetValues<GgmlActivationGradOp>())
        {
            foreach (bool hasAcc in new[] { false, true })
            {
                Harness.FillDeterministic(src.Floats);
                Harness.FillDeterministic(grad.Floats);
                Harness.FillDeterministic(acc.Floats);
                dst.Floats.Clear();

                GgmlTensorView4D srcView = Harness.Contiguous4D(src.Ptr, Ne0, Ne1, Ne2, Ne3);
                GgmlTensorView4D gradView = Harness.Contiguous4D(grad.Ptr, Ne0, Ne1, Ne2, Ne3);
                GgmlTensorView4D accView = Harness.Contiguous4D(acc.Ptr, Ne0, Ne1, Ne2, Ne3);
                GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, Ne0, Ne1, Ne2, Ne3);
                double ns = Harness.TimeCase(() => GgmlNative.ActivationGrad(op, dstView, srcView, gradView, accView, hasAcc));
                dump.Add($"actgrad/{op}-acc={hasAcc}", dst.RawBytes, ns);
            }
        }
    }

    public static unsafe void RunIndexSelectCases(Harness.Dump dump)
    {
        const int SrcRows = 32, Cols = 48, PickCount = 7;
        long srcBytes = (long)SrcRows * Cols * sizeof(float);
        long dstBytes = (long)PickCount * Cols * sizeof(float);
        using var src = new Harness.AlignedBuffer(srcBytes);
        using var dst = new Harness.AlignedBuffer(dstBytes);
        using var indices = new Harness.AlignedBuffer(PickCount * sizeof(int));

        Span<int> indexValues = new Span<int>((void*)indices.Ptr, PickCount);
        int[] picks = { 5, 1, 30, 0, 17, 5, 8 };
        picks.CopyTo(indexValues);

        foreach (bool addToResult in new[] { false, true })
        {
            Harness.FillDeterministic(src.Floats);
            Harness.FillDeterministic(dst.Floats, offset: 10f);

            var srcView = new GgmlTensorView2D(src.Ptr, SrcRows, Cols, Cols, 1, srcBytes);
            var dstView = new GgmlTensorView2D(dst.Ptr, PickCount, Cols, Cols, 1, dstBytes);
            var indicesDesc = new GgmlContiguousTensor(indices.Ptr, PickCount, TensorSharp.DType.Int32);
            // Accumulating runs are not idempotent, so time a single call and
            // preserve the deterministic accumulated state for comparison.
            double ns;
            if (addToResult)
            {
                long start = Stopwatch.GetTimestamp();
                GgmlNative.IndexSelect(dstView, srcView, indicesDesc, addToResult: true);
                ns = (Stopwatch.GetTimestamp() - start) * 1e9 / Stopwatch.Frequency;
            }
            else
            {
                ns = Harness.TimeCase(() => GgmlNative.IndexSelect(dstView, srcView, indicesDesc, addToResult: false));
            }
            dump.Add($"indexselect/add={addToResult}", dst.RawBytes, ns);
        }
    }

    public static unsafe void RunIndexSelectGradCases(Harness.Dump dump)
    {
        const int GradRows = 32, Cols = 48, AdjRows = 7;
        long gradBytes = (long)GradRows * Cols * sizeof(float);
        long adjBytes = (long)AdjRows * Cols * sizeof(float);
        using var grad = new Harness.AlignedBuffer(gradBytes);
        using var adj = new Harness.AlignedBuffer(adjBytes);
        using var indices = new Harness.AlignedBuffer(AdjRows * sizeof(int));

        Span<int> indexValues = new Span<int>((void*)indices.Ptr, AdjRows);
        int[] picks = { 5, 1, -1, 0, 17, 5, 31 };
        picks.CopyTo(indexValues);

        Harness.FillDeterministic(grad.Floats);
        Harness.FillDeterministic(adj.Floats);

        var gradView = new GgmlTensorView2D(grad.Ptr, GradRows, Cols, Cols, 1, gradBytes);
        var adjView = new GgmlTensorView2D(adj.Ptr, AdjRows, Cols, Cols, 1, adjBytes);
        var indicesDesc = new GgmlContiguousTensor(indices.Ptr, AdjRows, TensorSharp.DType.Int32);

        // Scatter-add mutates grad in place; run once for a deterministic result.
        long start = Stopwatch.GetTimestamp();
        GgmlNative.IndexSelectGrad(gradView, adjView, indicesDesc);
        double ns = (Stopwatch.GetTimestamp() - start) * 1e9 / Stopwatch.Frequency;
        dump.Add("indexselectgrad/basic", grad.RawBytes, ns);
    }

    // Perf-shaped cases: realistic activation sizes so the timing ratio is
    // dominated by compute + dispatch, not by tiny-op noise.
    public static void RunPerfCases(Harness.Dump dump)
    {
        const int Hidden = 4096, Rows = 32;
        long bytes = Harness.Bytes4D(Hidden, Rows, 1, 1);
        using var a = new Harness.AlignedBuffer(bytes);
        using var b = new Harness.AlignedBuffer(bytes);
        using var dst = new Harness.AlignedBuffer(bytes);

        Harness.FillDeterministic(a.Floats);
        Harness.FillDeterministic(b.Floats);
        dst.Floats.Clear();

        GgmlTensorView4D aView = Harness.Contiguous4D(a.Ptr, Hidden, Rows, 1, 1);
        GgmlTensorView4D bView = Harness.Contiguous4D(b.Ptr, Hidden, Rows, 1, 1);
        GgmlTensorView4D dstView = Harness.Contiguous4D(dst.Ptr, Hidden, Rows, 1, 1);

        double nsSilu = Harness.TimeCase(() => GgmlNative.Unary(GgmlUnaryOp.SiLU, dstView, aView));
        dump.Add("perf/silu-4096x32", dst.RawBytes, nsSilu);

        double nsAdd = Harness.TimeCase(() => GgmlNative.BinaryTensor(GgmlBinaryTensorOp.Add, dstView, aView, bView));
        dump.Add("perf/add-4096x32", dst.RawBytes, nsAdd);

        double nsActMul = Harness.TimeCase(() => GgmlNative.FusedActMul(GgmlFusedActMulOp.SiLUMul, dstView, aView, bView));
        dump.Add("perf/silumul-4096x32", dst.RawBytes, nsActMul);

        const int HalfDim = 4096, SplitRows = 32;
        long gateUpBytes = (long)SplitRows * 2 * HalfDim * sizeof(float);
        long splitDstBytes = (long)SplitRows * HalfDim * sizeof(float);
        using var gateUp = new Harness.AlignedBuffer(gateUpBytes);
        using var splitDst = new Harness.AlignedBuffer(splitDstBytes);
        Harness.FillDeterministic(gateUp.Floats);
        splitDst.Floats.Clear();
        var gateUpView = new GgmlTensorView2D(gateUp.Ptr, SplitRows, 2 * HalfDim, 2 * HalfDim, 1, gateUpBytes);
        var splitDstView = new GgmlTensorView2D(splitDst.Ptr, SplitRows, HalfDim, HalfDim, 1, splitDstBytes);
        double nsSplit = Harness.TimeCase(() => GgmlNative.FusedActMulSplit(GgmlFusedActMulOp.SiLUMul, splitDstView, gateUpView, HalfDim));
        dump.Add("perf/silumulsplit-4096x32", splitDst.RawBytes, nsSplit);
    }
}
