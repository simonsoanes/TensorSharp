// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.IO;

namespace TensorSharp.Models
{
    public class Gemma4ImageProcessor
    {
        public int PatchSize { get; }
        public int NMerge { get; }

        private readonly int _minPixels;
        private readonly int _maxPixels;
        private readonly int _maxSoftTokens;
        private readonly bool _referenceSizing;
        // When set (e.g. the gemma4uv unified embedder declares mean=0, std=1),
        // pixels are normalized as (pixel/255 - mean) / std instead of the SigLIP
        // SigLIP [-1, 1] mapping used by the gemma4v path.
        private readonly float[] _imageMean;
        private readonly float[] _imageStd;

        /// <summary>
        /// Gemma 4's own default soft-token budget per image. The processor documented
        /// for this family resizes an image to a TARGET token count -- 280 by default,
        /// choosing from 70/140/280/560/1120 -- rather than merely capping it, and it
        /// keeps the aspect ratio with both sides divisible by patch*merge.
        ///
        /// <para>
        /// Treating 280 as a ceiling with a low floor is what broke still-image
        /// understanding: a 448x448 picture aligned to 432x432, which is inside both
        /// bounds, so it was left at 9x9 = 81 soft tokens instead of the ~280 the model
        /// was trained to receive. Measured on gemma-4-E4B, that image was not
        /// perceived at all -- the model answered "I cannot directly view images" on 5
        /// runs out of 5 -- while the SAME picture at 896x896 (256 tokens), and the same
        /// picture supplied twice (162 tokens), were both described correctly. The floor
        /// is the fix; the ceiling was never the problem.
        /// </para>
        /// </summary>
        public const int DefaultSoftTokens = 280;
        public const int VideoSoftTokens = 70;

        /// <summary>
        /// The soft-token budgets the reference processor accepts
        /// (<c>_SUPPORTED_SOFT_TOKENS</c> in <c>image_processing_gemma4.py</c>). Raising this
        /// trades encode time for visual detail: the canvas area scales with the budget, so 560
        /// gives the model roughly twice the pixels of the 280 default and costs roughly twice the
        /// vision-encode time. llama.cpp defaults its own cap to 1120, which is why it reports
        /// noticeably more image tokens than this processor for the same picture.
        /// </summary>
        public static readonly int[] SupportedSoftTokens = { 70, 140, 280, 560, 1120 };

        /// <summary>
        /// Rejects a budget the reference processor would not accept, rather than silently
        /// rounding it. An off-list value produces a canvas the position-embedding table and the
        /// 3x3 pooling were never exercised at.
        /// </summary>
        private static int ValidateSoftTokens(int tokens)
        {
            if (Array.IndexOf(SupportedSoftTokens, tokens) >= 0)
                return tokens;
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens,
                "Gemma-4 soft-token budget must be one of " +
                string.Join(", ", SupportedSoftTokens) + ".");
        }

        /// <param name="referenceSizing">
        /// True to size and resize exactly as the reference <c>Gemma4ImageProcessor</c> does:
        /// scale to the soft-token budget (up OR down), floor each side to a multiple of
        /// patch*pooling, and stretch to fill with an antialiased bicubic filter.
        ///
        /// <para>This is correct for the <c>gemma4v</c> SigLIP tower and is verified against a numpy
        /// transcription of the reference (see Gemma4VisionOracleTests). It is deliberately NOT the
        /// default, because the <c>gemma4uv</c> unified embedder was measured to degrade under it:
        /// upscaling a 750x500 photo to the full budget made the model report a "very wide", "blurry"
        /// picture and hallucinate content that was not present. That path keeps the llama.cpp-style
        /// no-upscale + letterbox sizing, which is what it was validated on.</para>
        /// </param>
        public Gemma4ImageProcessor(int patchSize = 16, int nMerge = 3,
            int minTokens = DefaultSoftTokens, int maxTokens = DefaultSoftTokens,
            float[] imageMean = null, float[] imageStd = null, bool referenceSizing = false)
        {
            PatchSize = patchSize;
            NMerge = nMerge;
            int patchArea = patchSize * patchSize * nMerge * nMerge;
            _minPixels = minTokens * patchArea;
            _maxPixels = maxTokens * patchArea;
            _referenceSizing = referenceSizing;
            _maxSoftTokens = referenceSizing ? ValidateSoftTokens(maxTokens) : maxTokens;
            _imageMean = imageMean != null && imageMean.Length >= 3 ? imageMean : null;
            _imageStd = imageStd != null && imageStd.Length >= 3 ? imageStd : null;
        }

        /// <summary>
        /// Process an image file into normalized pixel values in channel-first format [C, H, W],
        /// following the reference <c>Gemma4ImageProcessor</c>:
        ///   1. Pick the canvas with <see cref="CalcAspectRatioPreservingSize"/> -- the largest
        ///      one inside the soft-token budget with both sides a multiple of patch*pooling.
        ///   2. Resize into it with an ANTIALIASED bicubic filter (the processor declares
        ///      resample=3), stretching to fill. No letterboxing: see the note on
        ///      <see cref="CalcAspectRatioPreservingSize"/> for what black bars cost.
        /// Normalization is (pixel/255 - mean) / std when the mmproj declares mean/std
        /// (mean 0 / std 1 for this family, i.e. plain [0,1]), otherwise the SigLIP
        /// [-1,1] map. Returns pixel data and the actual canvas dimensions.
        /// </summary>
        public (float[] pixels, int width, int height) ProcessImage(string imagePath)
        {
            byte[] fileBytes = File.ReadAllBytes(imagePath);
            int origWidth, origHeight;
            byte[] rgba = ImageProcessorUtils.DecodeImageToRGBA(fileBytes, out origWidth, out origHeight);

            int alignSize = PatchSize * NMerge;

            if (_referenceSizing)
            {
                CalcAspectRatioPreservingSize(origWidth, origHeight, PatchSize, NMerge, _maxSoftTokens,
                    out int refW, out int refH);

                // The reference declares resample=3 (BICUBIC), do_rescale=1/255 and
                // do_normalize=false (mean 0 / std 1). The tower itself does the 2*(x-0.5)
                // recentring, so feed [0,1] when the mmproj declares identity statistics and fall
                // back to the SigLIP [-1,1] map only when it does not.
                float[] refMean = _imageMean ?? new[] { 0.5f, 0.5f, 0.5f };
                float[] refStd = _imageStd ?? new[] { 0.5f, 0.5f, 0.5f };
                float[] refPixels = ImageProcessorUtils.ResizeRgbaToChannelFirstBicubic(
                    rgba, origWidth, origHeight, refW, refH, refMean, refStd);
                return (refPixels, refW, refH);
            }

            CalcSizePreservedRatio(origWidth, origHeight, alignSize, _minPixels, _maxPixels,
                out int targetW, out int targetH);
            float[] pixels = BuildLetterboxedNormalized(rgba, origWidth, origHeight, targetW, targetH);
            return (pixels, targetW, targetH);
        }

        /// <summary>
        /// Target canvas for one image, as a direct port of the reference processor's
        /// <c>get_aspect_ratio_preserving_size</c> (transformers
        /// <c>image_processing_gemma4.py</c>).
        ///
        /// <para>The largest canvas that (1) yields at most <c>maxSoftTokens * pooling^2</c>
        /// patches and (2) has both sides divisible by <c>patch * pooling</c>. Each side is
        /// FLOORED independently, so the aspect ratio is approximately, not exactly, preserved --
        /// and the image is then stretched to fill it, never letterboxed.</para>
        ///
        /// <para>This replaced a letterboxing port of llama.cpp's
        /// <c>calc_size_preserved_ratio</c>. Both pick the same canvas for most images, but
        /// letterboxing pads the content with black bars that become real, unmasked, pooled
        /// patches and shift every patch's learned position embedding. Measured end-to-end against
        /// a numpy transcription of the reference on the TensorSharp banner image, the letterboxed
        /// + bilinear pipeline diverged by 63.8% relative L2 with a worst-token cosine of -0.0007
        /// (orthogonal), even though the tower itself was correct to 0.09%.</para>
        /// </summary>
        internal static void CalcAspectRatioPreservingSize(int width, int height,
            int patchSize, int pooling, int maxSoftTokens, out int targetW, out int targetH)
        {
            int sideMult = patchSize * pooling;
            if (width <= 0 || height <= 0 || sideMult <= 0)
            {
                targetW = sideMult;
                targetH = sideMult;
                return;
            }

            long maxPatches = (long)maxSoftTokens * pooling * pooling;
            double targetPx = (double)maxPatches * patchSize * patchSize;
            double factor = Math.Sqrt(targetPx / ((double)width * height));

            targetH = (int)Math.Floor(factor * height / sideMult) * sideMult;
            targetW = (int)Math.Floor(factor * width / sideMult) * sideMult;

            // Either side can floor to zero for an extreme aspect ratio; the reference clamps the
            // degenerate side to one unit and caps the other so the patch budget still holds.
            int maxSide = (int)(maxPatches / (pooling * pooling)) * sideMult;
            if (targetH == 0 && targetW == 0)
            {
                targetH = sideMult;
                targetW = sideMult;
            }
            else if (targetH == 0)
            {
                targetH = sideMult;
                targetW = Math.Min(Math.Max(1, (int)Math.Floor((double)width / height)) * sideMult, maxSide);
            }
            else if (targetW == 0)
            {
                targetW = sideMult;
                targetH = Math.Min(Math.Max(1, (int)Math.Floor((double)height / width)) * sideMult, maxSide);
            }
        }

        /// <summary>
        /// Compute the resized canvas dimensions preserving aspect ratio so that
        /// <c>min_pixels &lt;= W*H &lt;= max_pixels</c>, with each side aligned to
        /// <paramref name="alignSize"/> (patch_size * n_merge). Direct port of
        /// llama.cpp <c>img_tool::calc_size_preserved_ratio(..., min_pixels, max_pixels)</c>
        /// (the "smart_resize" used by the Qwen/Gemma transformers processors).
        ///
        /// <para>Retained for the non-Gemma-4 callers and for A/B comparison against llama.cpp;
        /// the Gemma-4 path now uses <see cref="CalcAspectRatioPreservingSize"/>.</para>
        /// </summary>
        internal static void CalcSizePreservedRatio(int width, int height, int alignSize,
            int minPixels, int maxPixels, out int targetW, out int targetH)
        {
            if (width <= 0 || height <= 0 || alignSize <= 0)
            {
                targetW = alignSize;
                targetH = alignSize;
                return;
            }

            // std::round rounds halves away from zero; match that exactly.
            int RoundBy(double x) => (int)Math.Round(x / alignSize, MidpointRounding.AwayFromZero) * alignSize;
            int CeilBy(double x) => (int)Math.Ceiling(x / alignSize) * alignSize;
            int FloorBy(double x) => (int)Math.Floor(x / alignSize) * alignSize;

            // Always align up (round) first.
            int hBar = Math.Max(alignSize, RoundBy(height));
            int wBar = Math.Max(alignSize, RoundBy(width));

            long area = (long)hBar * wBar;
            if (maxPixels > 0 && area > maxPixels)
            {
                double beta = Math.Sqrt((double)height * width / maxPixels);
                hBar = Math.Max(alignSize, FloorBy(height / beta));
                wBar = Math.Max(alignSize, FloorBy(width / beta));
            }
            else if (minPixels > 0 && area < minPixels)
            {
                double beta = Math.Sqrt((double)minPixels / ((double)height * width));
                hBar = CeilBy(height * beta);
                wBar = CeilBy(width * beta);
            }

            targetW = wBar;
            targetH = hBar;
        }

        /// <summary>
        /// Resize the source image into a <paramref name="targetW"/> x <paramref name="targetH"/>
        /// canvas preserving aspect ratio (llama.cpp PAD_CEIL): the content is scaled down/up by
        /// the smaller of the per-axis scales, centred in the canvas, and the surrounding border is
        /// filled with the (normalized) padding colour black. The result is channel-first [C,H,W].
        /// </summary>
        private float[] BuildLetterboxedNormalized(byte[] rgba, int origW, int origH, int targetW, int targetH)
        {
            float scale = Math.Min((float)targetW / origW, (float)targetH / origH);
            int newW = Math.Min((int)Math.Ceiling(origW * (double)scale), targetW);
            int newH = Math.Min((int)Math.Ceiling(origH * (double)scale), targetH);
            newW = Math.Max(1, newW);
            newH = Math.Max(1, newH);
            int offsetX = (targetW - newW) / 2;
            int offsetY = (targetH - newH) / 2;

            bool hasMeanStd = _imageMean != null && _imageStd != null;

            // Resize the content preserving aspect ratio into [C, newH, newW].
            float[] content = hasMeanStd
                ? ImageProcessorUtils.ResizeRgbaToChannelFirstNormalized(rgba, origW, origH, newW, newH, _imageMean, _imageStd)
                : ImageProcessorUtils.ResizeRgbaToChannelFirstNormalized(rgba, origW, origH, newW, newH);

            int targetPixels = targetW * targetH;
            int contentPixels = newW * newH;
            float[] result = new float[3 * targetPixels];

            for (int c = 0; c < 3; c++)
            {
                // Normalized value of a black pad pixel (0/255 == 0) under the active scheme.
                float padVal = hasMeanStd ? (0f - _imageMean[c]) / _imageStd[c] : -1f;
                int cBase = c * targetPixels;

                for (int i = 0; i < targetPixels; i++)
                    result[cBase + i] = padVal;

                for (int y = 0; y < newH; y++)
                {
                    int srcRow = c * contentPixels + y * newW;
                    int dstRow = cBase + (y + offsetY) * targetW + offsetX;
                    Array.Copy(content, srcRow, result, dstRow, newW);
                }
            }

            return result;
        }

        public int ComputeOutputTokens(int imageWidth, int imageHeight)
        {
            int patchesX = imageWidth / PatchSize;
            int patchesY = imageHeight / PatchSize;
            int mergedX = patchesX / NMerge;
            int mergedY = patchesY / NMerge;
            return mergedX * mergedY;
        }
    }
}

