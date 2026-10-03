// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Parity for the Gemma-4 "gemma4v" vision tower against a pure-numpy transcription of the
// HuggingFace reference (eng/diffusiongemma-vision-oracle.py), run on the DiffusionGemma
// 27-layer / 1152-wide tower.
//
// The oracle is transcribed from transformers' modeling_gemma4.py and deliberately follows
// NEITHER this encoder NOR llama.cpp's tools/mtmd/models/gemma4v.cpp, because both were found to
// contain numerical defects that this test exists to prevent from returning:
//
//   1. Gemma4MultimodalEmbedder applies an unweighted RMSNorm BEFORE the projection
//      ("embedding_pre_projection_norm"). This encoder used to project first and norm after.
//   2. The vision MLP activation is gelu_pytorch_tanh, not QuickGELU. Measured divergence of the
//      QuickGELU variant on this fixture: 8.14% relative L2, worst-token cosine 0.771 -- i.e. one
//      soft token in 260 became effectively a different vector. llama.cpp shares this defect by
//      default, because its mmproj files omit clip.use_gelu and clip.cpp falls back to
//      FFN_GELU_QUICK, so llama.cpp cannot be used to arbitrate it.
//
// Fixtures are the exact resized, [-1,1], channel-first pixels the oracle fed itself, plus the
// [n_soft, 2816] projected output it produced. Feeding the same pixels is what makes this a test
// of the TOWER rather than of two independently-written resize routines.
//
// Regenerate with:
//   python3 eng/diffusiongemma-vision-oracle.py \
//       --src ~/work/models/diffusiongemma-vision/model-00011-of-00011.safetensors \
//       --image ~/work/models/testmedia/image.png --act gelu_tanh \
//       --dump-dir <tmp> --out <tmp>/final.npy
//
//   export TS_DIFFUSIONGEMMA_VISION_DIR=~/work/models/diffusiongemma-vision
//   export TS_TEST_GGML_BACKEND=metal        # or cpu
using System;
using System.Globalization;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    public class Gemma4VisionOracleTests
    {
        private const string DirEnv = "TS_DIFFUSIONGEMMA_VISION_DIR";
        private readonly ITestOutputHelper _output;

        public Gemma4VisionOracleTests(ITestOutputHelper output) { _output = output; }

        private static string Root => Environment.GetEnvironmentVariable(DirEnv);
        private static string FixtureDir => Root == null ? null : Path.Combine(Root, "fixtures");
        private static string ShardPath =>
            Path.Combine(Root ?? ".", "model-00011-of-00011.safetensors");
        private static string MmprojPath =>
            Path.Combine(Root ?? ".", "mmproj-diffusiongemma-26B-A4B-it-F16.gguf");

        private static bool HaveFixtures =>
            !string.IsNullOrEmpty(Root)
            && File.Exists(Path.Combine(FixtureDir, "vision_pixels_chw.f32"))
            && File.Exists(Path.Combine(FixtureDir, "vision_oracle_final.f32"));

        private static float[] ReadF32(string name)
        {
            byte[] bytes = File.ReadAllBytes(Path.Combine(FixtureDir, name));
            var data = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            return data;
        }

        private static int[] ReadShape(string name) =>
            File.ReadAllText(Path.Combine(FixtureDir, name + ".shape"))
                .Trim().Split(',').Select(s => int.Parse(s.Trim(), CultureInfo.InvariantCulture)).ToArray();

        private static GgmlBackendType Backend =>
            TestGates.PinnedGgmlBackendType;

        /// <summary>
        /// Relative L2 tolerance. The CPU kernels reproduce the numpy reference to float-rounding
        /// noise; the Metal kernels lose about 2% on this tower because ggml's Metal GEMM path is
        /// lower precision and this tower's residual stream reaches an absmax near 2900, where F16
        /// has a ulp of 2.0. Both bounds are far below the 8.14% that the QuickGELU defect
        /// produced, so either backend still catches a regression of that kind.
        /// </summary>
        private static double RelL2Tolerance =>
            Backend == GgmlBackendType.Cpu ? 0.004 : 0.030;

        private static double MinCosineTolerance =>
            Backend == GgmlBackendType.Cpu ? 0.9995 : 0.95;

        private (double relL2, double minCos, double meanCos) CompareAgainstOracle(string projectorPath)
        {
            int[] pixShape = ReadShape("vision_pixels_chw.f32");   // [3, H, W]
            int[] outShape = ReadShape("vision_oracle_final.f32"); // [n_soft, projDim]
            float[] pixels = ReadF32("vision_pixels_chw.f32");
            float[] expected = ReadF32("vision_oracle_final.f32");

            int height = pixShape[1], width = pixShape[2];
            Assert.Equal(3L * height * width, pixels.LongLength);

            var ctx = new GgmlContext(new[] { 0 }, Backend);
            var allocator = new GgmlAllocator(ctx, 0);

            using var encoder = new Gemma4VisionEncoder(projectorPath, allocator);
            using Tensor actualTensor = encoder.Encode(pixels, width, height);

            Assert.Equal(outShape[0], (int)actualTensor.Sizes[0]);
            Assert.Equal(outShape[1], (int)actualTensor.Sizes[1]);

            float[] actual = actualTensor.GetElementsAsFloat(outShape[0] * outShape[1]);
            Assert.Equal(expected.LongLength, actual.LongLength);

            double diffSq = 0, refSq = 0;
            for (long i = 0; i < expected.LongLength; i++)
            {
                double d = (double)actual[i] - expected[i];
                diffSq += d * d;
                refSq += (double)expected[i] * expected[i];
            }
            double relL2 = Math.Sqrt(diffSq / refSq);

            double minCos = 1.0, sumCos = 0;
            int dim = outShape[1];
            for (int row = 0; row < outShape[0]; row++)
            {
                double dot = 0, na = 0, nb = 0;
                long b = (long)row * dim;
                for (int j = 0; j < dim; j++)
                {
                    double a = actual[b + j], e = expected[b + j];
                    dot += a * e; na += a * a; nb += e * e;
                }
                double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-30);
                minCos = Math.Min(minCos, cos);
                sumCos += cos;
            }

            return (relL2, minCos, sumCos / outShape[0]);
        }

        [ModelFact(DirEnv)]
        public void SafetensorsTower_MatchesNumpyOracle()
        {
            if (!HaveFixtures || !File.Exists(ShardPath))
            {
                _output.WriteLine($"skipped: set {DirEnv} to a directory containing the shard and fixtures/");
                return;
            }

            var (relL2, minCos, meanCos) = CompareAgainstOracle(ShardPath);
            _output.WriteLine(
                $"safetensors on {Backend}: relL2 {relL2 * 100:F5}%  minCos {minCos:F7}  meanCos {meanCos:F7}");

            Assert.True(relL2 <= RelL2Tolerance,
                $"relative L2 {relL2 * 100:F4}% exceeds {RelL2Tolerance * 100:F2}% on {Backend}. " +
                "A jump toward ~8% means the vision MLP activation regressed to QuickGELU, or the " +
                "multimodal embedder's RMSNorm moved back after the projection.");
            Assert.True(minCos >= MinCosineTolerance,
                $"worst-token cosine {minCos:F6} below {MinCosineTolerance:F4} on {Backend}.");
        }

        /// <summary>
        /// Gates PREPROCESSING at the pixel level, which is the only meaningful place to measure
        /// it: the tower amplifies a relative input difference by about 56x, because the
        /// standardization step <c>(x - std_bias) * std_scale</c> subtracts values of order 78,000
        /// to produce values of order 9. Measured on this fixture, a 0.167% pixel difference
        /// becomes a 9.3% embedding difference through the numpy reference tower itself -- so an
        /// embedding-level bound here would be testing the tower's conditioning, not the resize.
        ///
        /// <para>What this catches is real regressions in the resize: the previous letterboxing +
        /// bilinear pipeline produced a 13.9% PIXEL difference, which is two orders of magnitude
        /// outside this bound.</para>
        ///
        /// <para>The residual 0.167% is PIL-vs-this-implementation float rounding in the
        /// antialiased bicubic and is irreducible without emulating PIL bit-for-bit.</para>
        /// </summary>
        [ModelFact(DirEnv)]
        public void Preprocessing_MatchesReferencePixels()
        {
            string image = Path.Combine(FixtureDir ?? ".", "vision_ref_image.png");
            if (!HaveFixtures || !File.Exists(image)
                || !File.Exists(Path.Combine(FixtureDir, "vision_ref_pixels_chw.f32")))
            {
                _output.WriteLine($"skipped: set {DirEnv} to a directory containing fixtures/");
                return;
            }

            int[] shape = ReadShape("vision_ref_pixels_chw.f32");   // [3, H, W]
            float[] expected = ReadF32("vision_ref_pixels_chw.f32");

            var processor = new Gemma4ImageProcessor(
                minTokens: Gemma4ImageProcessor.DefaultSoftTokens,
                maxTokens: Gemma4ImageProcessor.DefaultSoftTokens,
                referenceSizing: true);
            var (actual, width, height) = processor.ProcessImage(image);

            Assert.Equal(shape[2], width);
            Assert.Equal(shape[1], height);
            Assert.Equal(expected.LongLength, actual.LongLength);

            double diffSq = 0, refSq = 0, maxAbs = 0;
            for (long i = 0; i < expected.LongLength; i++)
            {
                double d = (double)actual[i] - expected[i];
                diffSq += d * d;
                refSq += (double)expected[i] * expected[i];
                maxAbs = Math.Max(maxAbs, Math.Abs(d));
            }
            double relL2 = Math.Sqrt(diffSq / refSq);
            _output.WriteLine($"preprocessing: {width}x{height}, pixel relL2 {relL2 * 100:F4}%, max|d| {maxAbs:F5}");

            Assert.True(relL2 <= 0.01,
                $"preprocessed pixels differ from the reference by {relL2 * 100:F3}% (bound 1%). " +
                "A jump to ~14% means the resize reverted to letterboxing and/or bilinear.");
        }

        /// <summary>
        /// The mmproj produced by eng/diffusiongemma-mmproj.py must be interchangeable with the raw
        /// shard. They are independently written mappings of the same weights, so a disagreement
        /// means one of the two renamed or re-laid-out a tensor wrongly -- most likely the
        /// patch-embedding permute, where HF's rows are [ky][kx][c] but the im2col here is
        /// [c][ky][kx] and a mistake runs clean while producing wrong features.
        /// </summary>
        [ModelFact(DirEnv)]
        public void MmprojTower_AgreesWithSafetensorsTower()
        {
            if (!HaveFixtures || !File.Exists(ShardPath) || !File.Exists(MmprojPath))
            {
                _output.WriteLine(
                    $"skipped: needs {DirEnv} with the shard, fixtures/, and an mmproj built by " +
                    "eng/diffusiongemma-mmproj.py");
                return;
            }

            var st = CompareAgainstOracle(ShardPath);
            var gg = CompareAgainstOracle(MmprojPath);
            _output.WriteLine($"safetensors relL2 {st.relL2 * 100:F5}%   mmproj relL2 {gg.relL2 * 100:F5}%");

            // The mmproj stores the 2-D matmul weights as F16, so it is allowed to be slightly
            // worse than the BF16-widened shard, but not categorically different.
            Assert.True(Math.Abs(st.relL2 - gg.relL2) <= 0.01,
                $"mmproj ({gg.relL2 * 100:F4}%) and safetensors ({st.relL2 * 100:F4}%) disagree by " +
                "more than 1 point; the two tensor mappings have diverged.");
        }
    }
}
