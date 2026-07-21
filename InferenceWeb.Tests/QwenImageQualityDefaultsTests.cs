// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Quality-first sampling defaults for Qwen-Image-Edit: the whole-step denoise
// cache (EasyCache) must be strictly OPT-IN — stable-diffusion.cpp ships it
// behind --cache-mode for the same reason (skipped steps are approximated from
// stale diffs and visibly soften fine detail like faces). A 30-step edit was
// observed computing only ~19 of its steps under the old default-on behavior.
using System;
using TensorSharp.Models.QwenImage;
using Xunit;

namespace InferenceWeb.Tests
{
    public class QwenImageQualityDefaultsTests
    {
        private static IDisposable SetEnv(string name, string value)
        {
            string old = Environment.GetEnvironmentVariable(name);
            Environment.SetEnvironmentVariable(name, value);
            return new RestoreEnv(name, old);
        }

        private sealed class RestoreEnv : IDisposable
        {
            private readonly string _name, _old;
            public RestoreEnv(string name, string old) { _name = name; _old = old; }
            public void Dispose() => Environment.SetEnvironmentVariable(_name, _old);
        }

        [Fact]
        public void StepCache_DefaultMode_IsOff()
        {
            using var _ = SetEnv("TS_QWEN_DIT_CACHE_MODE", null);
            using var __ = SetEnv("TS_QWEN_DIT_CACHE", null);
            Assert.Equal("off", QwenImageStepCache.Mode);
            Assert.False(new QwenImageStepCache(30).Enabled);
        }

        [Fact]
        public void StepCache_EasyCacheOptIn_EnablesForLongSchedules()
        {
            using var _ = SetEnv("TS_QWEN_DIT_CACHE_MODE", "easycache");
            using var __ = SetEnv("TS_QWEN_DIT_CACHE", null);
            Assert.True(new QwenImageStepCache(30).Enabled);
            // Few-step (Lightning) schedules stay uncached even when opted in.
            Assert.False(new QwenImageStepCache(4).Enabled);
        }

        [Fact]
        public void StepCache_LegacyDisableKnob_StillWins()
        {
            using var _ = SetEnv("TS_QWEN_DIT_CACHE_MODE", "easycache");
            using var __ = SetEnv("TS_QWEN_DIT_CACHE", "0");
            Assert.Equal("off", QwenImageStepCache.Mode);
            Assert.False(new QwenImageStepCache(30).Enabled);
        }

        // ---- input-detail preservation: vision condition sizing (2511 band) ----

        [Fact]
        public void VisionSmartResize_HighResInput_KeepsThe2511MaxBand_NotTheOld384Sq()
        {
            using var _ = SetEnv("TS_QWEN_IMAGE_VISION_MIN_PIXELS", null);
            using var __ = SetEnv("TS_QWEN_IMAGE_VISION_MAX_PIXELS", null);
            // 12 MP photo (the girl.heic case): must land near the 560^2 max — over 2x the
            // pixels of the old fixed 384^2 squash — with /28 dims and preserved aspect.
            var (hb, wb) = QwenImageVisionProcessor.SmartResize(4032, 3024);
            Assert.Equal(0, hb % 28);
            Assert.Equal(0, wb % 28);
            long px = (long)hb * wb;
            Assert.InRange(px, 384L * 384, 560L * 560);
            Assert.True(px > 250_000, $"expected near-560^2 sizing, got {hb}x{wb} = {px} px");
            Assert.InRange((double)hb / wb, 4032.0 / 3024 * 0.9, 4032.0 / 3024 * 1.1);
        }

        [Fact]
        public void VisionSmartResize_InBandInput_KeepsItsOwnResolution()
        {
            using var _ = SetEnv("TS_QWEN_IMAGE_VISION_MIN_PIXELS", null);
            using var __ = SetEnv("TS_QWEN_IMAGE_VISION_MAX_PIXELS", null);
            // 527x418 (the dress.png case, ~0.22 MP, inside [384^2, 560^2]): kept at its own
            // resolution snapped to /28 — no downscale to 384^2.
            var (hb, wb) = QwenImageVisionProcessor.SmartResize(527, 418);
            Assert.Equal(532, hb);   // round(527/28)*28
            Assert.Equal(420, wb);   // round(418/28)*28
        }

        [Fact]
        public void VisionSmartResize_TinyInput_UpscaledToTheMinBand()
        {
            using var _ = SetEnv("TS_QWEN_IMAGE_VISION_MIN_PIXELS", null);
            using var __ = SetEnv("TS_QWEN_IMAGE_VISION_MAX_PIXELS", null);
            var (hb, wb) = QwenImageVisionProcessor.SmartResize(200, 200);
            Assert.True((long)hb * wb >= 384L * 384);
            Assert.Equal(0, hb % 28);
            Assert.Equal(0, wb % 28);
        }

        // ---- input-detail preservation: reference-latent area ----

        [Fact]
        public void ResolveRefArea_CudaDefault_IsNativeAreaIndependentOfOutput()
        {
            using var _ = SetEnv("TS_QWEN_IMAGE_REF_AREA", null);
            Assert.Equal(1024L * 1024, QwenImagePipeline.ResolveRefArea(TensorSharp.Runtime.BackendType.GgmlCuda));
            // other backends keep the legacy output-coupled rule (0 = decide from output area)
            Assert.Equal(0, QwenImagePipeline.ResolveRefArea(TensorSharp.Runtime.BackendType.GgmlMetal));
        }

        [Fact]
        public void ResolveRefArea_EnvOverride_ClampedToSaneRange()
        {
            using (SetEnv("TS_QWEN_IMAGE_REF_AREA", "2097152"))
                Assert.Equal(2097152, QwenImagePipeline.ResolveRefArea(TensorSharp.Runtime.BackendType.GgmlCuda));
            using (SetEnv("TS_QWEN_IMAGE_REF_AREA", "999999999"))
                Assert.Equal(4L * 1024 * 1024, QwenImagePipeline.ResolveRefArea(TensorSharp.Runtime.BackendType.GgmlCuda));
            using (SetEnv("TS_QWEN_IMAGE_REF_AREA", "1"))
                Assert.Equal(65536, QwenImagePipeline.ResolveRefArea(TensorSharp.Runtime.BackendType.GgmlCuda));
        }
    }
}
