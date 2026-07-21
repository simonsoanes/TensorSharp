// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Regression guard for the "edit output is a solid black image" class of bug.
//
// The native Qwen-Image DiT kernels declare the small per-block tensors (attention /
// MLP biases, QK-norm weights) as GGML_TYPE_F32 and upload numElements*4 bytes from
// the pointer the managed side hands them. That pointer used to be the raw GGUF mmap
// address with no type check, which only works because the common Qwen-Image DiT
// quantizations (Q4_K_M and friends) happen to store those tensors as F32.
//
// A low-bit mixed quantization does not: a Q2_K "rapid" DiT stores all 1093 of its
// small tensors as F16. The kernel then read F16 bit patterns as F32 (garbage near
// +/-3.4e38) AND ran a full tensor-length past the end of the buffer. The very first
// bias add turned the forward non-finite, every later denoise step stayed NaN, and
// the VAE decoded a solid black PNG with no error anywhere. See QwenImageDiT.F32Ptr.
//
// These tests are model-gated (like UserSuppliedHeicSmokeTestWhenConfigured): point
// TENSORSHARP_QWEN_IMAGE_DIT at a DiT GGUF to exercise a real forward. They skip when
// no model is configured, so CI without weights stays green.
using System;
using System.IO;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests
{
    public class QwenImageDiTWeightDtypeTests
    {
        private static string DitPath()
        {
            string path = Environment.GetEnvironmentVariable("TENSORSHARP_QWEN_IMAGE_DIT");
            return !string.IsNullOrWhiteSpace(path) && File.Exists(path) ? path : null;
        }

        /// <summary>
        /// End-to-end guard: a DiT forward must be finite. This is the check that actually
        /// fails when the native path is handed a wrong-dtype pointer — the managed reference
        /// path stays correct because it uses the dequantized weight copies, so only a native
        /// forward reproduces it.
        /// </summary>
        [Fact]
        public void DitForward_IsFinite_OnTheConfiguredModel()
        {
            string path = DitPath();
            if (path == null) return;

            var backendName = Environment.GetEnvironmentVariable("TS_QWEN_BACKEND") ?? "ggml_cpu";
            var backend = backendName switch
            {
                "ggml_cuda" => BackendType.GgmlCuda,
                "ggml_metal" => BackendType.GgmlMetal,
                _ => BackendType.GgmlCpu,
            };

            // Small token counts: this is a numeric-health check, not a perf run.
            const int hp = 4, wp = 4, txtSeq = 8;
            int seq = hp * wp, imgSeq = 2 * seq;

            var dit = new TensorSharp.Models.QwenImage.QwenImageDiT(path, backend);
            try
            {
                var rng = new Random(7);
                var img = new float[imgSeq * 64];
                for (int i = 0; i < img.Length; i++) img[i] = (float)(rng.NextDouble() - 0.5);
                var cond = new float[txtSeq * 3584];
                for (int i = 0; i < cond.Length; i++) cond[i] = (float)(rng.NextDouble() - 0.5);

                var modIndex = new int[imgSeq];
                for (int i = seq; i < imgSeq; i++) modIndex[i] = 1;
                var shapes = new (int f, int h, int w)[] { (1, hp, wp), (1, hp, wp) };
                var rope = TensorSharp.Models.QwenImage.DitRope.Build(shapes, txtSeq);

                float[] v = dit.Predict(img, imgSeq, cond, txtSeq, 0.5f, modIndex, rope);

                Assert.NotNull(v);
                Assert.All(v, x => Assert.True(float.IsFinite(x),
                    "DiT velocity is non-finite — the edit would decode to a solid black image"));
            }
            finally { dit.Dispose(); }
        }
    }
}
