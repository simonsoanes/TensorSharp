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
// Parity battery for the training module (GgmlManagedOps.Training.cs vs
// ggml_ops_training.cpp): CrossEntropyLoss, CrossEntropyLossBackward, Adam.
using System;
using System.Diagnostics;
using TensorSharp.GGML;

internal static class CasesTraining
{
    public static void Run(Harness.Dump dump)
    {
        RunCrossEntropyLossCases(dump);
        RunCrossEntropyLossBackwardCases(dump);
        RunAdamCases(dump);
    }

    // Row-wise softmax computed host-side so both modes (native/managed) see
    // bit-identical, valid probability inputs.
    private static void SoftmaxRows(Span<float> data, int rows, int cols)
    {
        for (int r = 0; r < rows; ++r)
        {
            Span<float> row = data.Slice(r * cols, cols);
            float max = float.NegativeInfinity;
            for (int i = 0; i < cols; ++i)
            {
                if (row[i] > max)
                    max = row[i];
            }
            float sum = 0f;
            for (int i = 0; i < cols; ++i)
            {
                row[i] = MathF.Exp(row[i] - max);
                sum += row[i];
            }
            for (int i = 0; i < cols; ++i)
                row[i] /= sum;
        }
    }

    private static void FillTargetIndices(Span<int> indices, int cols)
    {
        for (int i = 0; i < indices.Length; ++i)
            indices[i] = (i * 7 + 3) % cols;
    }

    public static unsafe void RunCrossEntropyLossCases(Harness.Dump dump)
    {
        const int Ne0 = 17, Ne1 = 5, Ne2 = 2, Ne3 = 2; // cols=17, rows=20
        const int Rows = Ne1 * Ne2 * Ne3;
        long probsBytes = Harness.Bytes4D(Ne0, Ne1, Ne2, Ne3);
        using var probs = new Harness.AlignedBuffer(probsBytes);
        using var indicesI32 = new Harness.AlignedBuffer(Rows * sizeof(int));
        using var indicesF32 = new Harness.AlignedBuffer(Rows * sizeof(float));

        Span<int> indexValues = new Span<int>((void*)indicesI32.Ptr, Rows);
        FillTargetIndices(indexValues, Ne0);
        Span<float> indexValuesF32 = indicesF32.Floats;
        for (int i = 0; i < Rows; ++i)
            indexValuesF32[i] = indexValues[i];

        Harness.FillDeterministic(probs.Floats, scale: 2f);
        SoftmaxRows(probs.Floats, Rows, Ne0);

        GgmlTensorView4D probsView = Harness.Contiguous4D(probs.Ptr, Ne0, Ne1, Ne2, Ne3);
        var indicesI32Desc = new GgmlContiguousTensor(indicesI32.Ptr, Rows, TensorSharp.DType.Int32);
        var indicesF32Desc = new GgmlContiguousTensor(indicesF32.Ptr, Rows, TensorSharp.DType.Float32);

        // smooth=0, labelSmooth=0, I32 indices.
        {
            float loss = 0f;
            double ns = Harness.TimeCase(() => loss = GgmlNative.CrossEntropyLoss(probsView, indicesI32Desc, smooth: 0f, labelSmooth: 0f));
            dump.Add("training/crossentropyloss/smooth0-i32", BitConverter.GetBytes(loss), ns);
        }

        // Label smoothing, I32 indices.
        {
            float loss = 0f;
            double ns = Harness.TimeCase(() => loss = GgmlNative.CrossEntropyLoss(probsView, indicesI32Desc, smooth: 0f, labelSmooth: 0.1f));
            dump.Add("training/crossentropyloss/labelsmooth0.1-i32", BitConverter.GetBytes(loss), ns);
        }

        // F32 index tensor (values converted to int inside the op).
        {
            float loss = 0f;
            double ns = Harness.TimeCase(() => loss = GgmlNative.CrossEntropyLoss(probsView, indicesF32Desc, smooth: 0f, labelSmooth: 0f));
            dump.Add("training/crossentropyloss/smooth0-f32idx", BitConverter.GetBytes(loss), ns);
        }
    }

    public static unsafe void RunCrossEntropyLossBackwardCases(Harness.Dump dump)
    {
        const int Ne0 = 17, Ne1 = 5, Ne2 = 2, Ne3 = 2; // cols=17, rows=20
        const int Rows = Ne1 * Ne2 * Ne3;
        long bytes = Harness.Bytes4D(Ne0, Ne1, Ne2, Ne3);
        using var grad = new Harness.AlignedBuffer(bytes);
        using var probs = new Harness.AlignedBuffer(bytes);
        using var indices = new Harness.AlignedBuffer(Rows * sizeof(int));

        Span<int> indexValues = new Span<int>((void*)indices.Ptr, Rows);
        FillTargetIndices(indexValues, Ne0);

        GgmlTensorView4D gradView = Harness.Contiguous4D(grad.Ptr, Ne0, Ne1, Ne2, Ne3);
        GgmlTensorView4D probsView = Harness.Contiguous4D(probs.Ptr, Ne0, Ne1, Ne2, Ne3);
        var indicesDesc = new GgmlContiguousTensor(indices.Ptr, Rows, TensorSharp.DType.Int32);

        // addGrad=false: grad is a pure output, so the op is idempotent and
        // can go through the timing loop.
        {
            Harness.FillDeterministic(probs.Floats, scale: 2f);
            SoftmaxRows(probs.Floats, Rows, Ne0);
            grad.Floats.Clear();

            double ns = Harness.TimeCase(() => GgmlNative.CrossEntropyLossBackward(
                gradView, probsView, indicesDesc, lossGradient: 1f, smooth: 0f, labelSmooth: 0f, addGrad: false));
            dump.Add("training/crossentropyloss_backward/add=False-lg1", grad.RawBytes, ns);
        }

        // addGrad=false with lossGradient != 1 and label smoothing.
        {
            Harness.FillDeterministic(probs.Floats, scale: 2f);
            SoftmaxRows(probs.Floats, Rows, Ne0);
            grad.Floats.Clear();

            double ns = Harness.TimeCase(() => GgmlNative.CrossEntropyLossBackward(
                gradView, probsView, indicesDesc, lossGradient: 0.5f, smooth: 0f, labelSmooth: 0.1f, addGrad: false));
            dump.Add("training/crossentropyloss_backward/add=False-lg0.5-ls0.1", grad.RawBytes, ns);
        }

        // addGrad=true accumulates into grad, so run a single timed call to
        // preserve the deterministic accumulated state for comparison.
        {
            Harness.FillDeterministic(probs.Floats, scale: 2f);
            SoftmaxRows(probs.Floats, Rows, Ne0);
            Harness.FillDeterministic(grad.Floats, offset: 5f);

            long start = Stopwatch.GetTimestamp();
            GgmlNative.CrossEntropyLossBackward(
                gradView, probsView, indicesDesc, lossGradient: 1.25f, smooth: 0f, labelSmooth: 0f, addGrad: true);
            double ns = (Stopwatch.GetTimestamp() - start) * 1e9 / Stopwatch.Frequency;
            dump.Add("training/crossentropyloss_backward/add=True-lg1.25", grad.RawBytes, ns);
        }
    }

    public static void RunAdamCases(Harness.Dump dump)
    {
        const int ElementCount = 1000;
        long bytes = (long)ElementCount * sizeof(float);
        using var weight = new Harness.AlignedBuffer(bytes);
        using var gradient = new Harness.AlignedBuffer(bytes);
        using var v = new Harness.AlignedBuffer(bytes);
        using var m = new Harness.AlignedBuffer(bytes);

        var weightDesc = new GgmlContiguousTensor(weight.Ptr, ElementCount, TensorSharp.DType.Float32);
        var gradientDesc = new GgmlContiguousTensor(gradient.Ptr, ElementCount, TensorSharp.DType.Float32);
        var vDesc = new GgmlContiguousTensor(v.Ptr, ElementCount, TensorSharp.DType.Float32);
        var mDesc = new GgmlContiguousTensor(m.Ptr, ElementCount, TensorSharp.DType.Float32);

        // Adam mutates weight/v/m in place and zeroes the gradient, so run a
        // FIXED number of iterations (refilling the gradient host-side before
        // each) and time only the first call, like the other in-place cases.
        // Case 1: gradNormFactor == 1 (no scale node), 3 iterations.
        {
            Harness.FillDeterministic(weight.Floats, scale: 0.5f);
            v.Floats.Clear();
            m.Floats.Clear();

            double ns = 0;
            for (int iter = 1; iter <= 3; ++iter)
            {
                Harness.FillDeterministic(gradient.Floats);
                long start = Stopwatch.GetTimestamp();
                GgmlNative.Adam(
                    weightDesc, gradientDesc, vDesc, mDesc,
                    gradNormFactor: 1.0f, stepSize: 0.001f, clipValue: 0.5f, regc: 0.01f,
                    decayRateV: 0.999f, decayRateM: 0.9f, iter: iter, eps: 1e-8f);
                if (iter == 1)
                    ns = (Stopwatch.GetTimestamp() - start) * 1e9 / Stopwatch.Frequency;
            }
            dump.Add("training/adam/basic-weight", weight.RawBytes, ns);
            dump.Add("training/adam/basic-v", v.RawBytes, ns);
            dump.Add("training/adam/basic-m", m.RawBytes, ns);
            dump.Add("training/adam/basic-gradient", gradient.RawBytes, ns);
        }

        // Case 2: gradNormFactor != 1 (exercises the grad scaling node), 2 iterations.
        {
            Harness.FillDeterministic(weight.Floats, scale: 0.5f);
            v.Floats.Clear();
            m.Floats.Clear();

            double ns = 0;
            for (int iter = 1; iter <= 2; ++iter)
            {
                Harness.FillDeterministic(gradient.Floats, scale: 2f);
                long start = Stopwatch.GetTimestamp();
                GgmlNative.Adam(
                    weightDesc, gradientDesc, vDesc, mDesc,
                    gradNormFactor: 0.25f, stepSize: 0.01f, clipValue: 1.0f, regc: 0f,
                    decayRateV: 0.999f, decayRateM: 0.9f, iter: iter, eps: 1e-8f);
                if (iter == 1)
                    ns = (Stopwatch.GetTimestamp() - start) * 1e9 / Stopwatch.Frequency;
            }
            dump.Add("training/adam/scaled-weight", weight.RawBytes, ns);
            dump.Add("training/adam/scaled-v", v.RawBytes, ns);
            dump.Add("training/adam/scaled-m", m.RawBytes, ns);
            dump.Add("training/adam/scaled-gradient", gradient.RawBytes, ns);
        }
    }
}
