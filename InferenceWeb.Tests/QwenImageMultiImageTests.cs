// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Model-free unit tests for the multi-image Qwen-Image-Edit conditioning logic:
// the "Picture N" prompt template, per-image <|image_pad|> expansion, the
// Qwen2.5-VL M-RoPE position layout for several vision spans, and the DiT RoPE
// tables for several reference-latent streams.
using System;
using System.Linq;
using TensorSharp.Models.QwenImage;
using Xunit;

namespace InferenceWeb.Tests
{
    public class QwenImageMultiImageTests
    {
        // ---- prompt template -------------------------------------------------

        [Fact]
        public void BuildWithImages_SingleImage_MatchesEditPlusTemplate()
        {
            string s = QwenImagePrompt.BuildWithImages("add a hat", 1);
            Assert.Contains("Picture 1: <|vision_start|><|image_pad|><|vision_end|>add a hat", s);
            Assert.DoesNotContain("Picture 2:", s);
            // system prompt + chat scaffolding preserved
            Assert.StartsWith("<|im_start|>system\n", s);
            Assert.EndsWith("<|im_start|>assistant\n", s);
        }

        [Fact]
        public void BuildWithImages_TwoImages_NumbersContinuouslyInOrder()
        {
            string s = QwenImagePrompt.BuildWithImages("swap outfits", 2);
            int p1 = s.IndexOf("Picture 1: <|vision_start|><|image_pad|><|vision_end|>", StringComparison.Ordinal);
            int p2 = s.IndexOf("Picture 2: <|vision_start|><|image_pad|><|vision_end|>", StringComparison.Ordinal);
            Assert.True(p1 >= 0 && p2 > p1, "Picture blocks must appear in order with no separator");
            // the user prompt follows the last image block
            Assert.True(s.IndexOf("swap outfits", StringComparison.Ordinal) > p2);
            // exactly one <|image_pad|> per image before expansion
            Assert.Equal(2, CountOf(s, "<|image_pad|>"));
        }

        private static int CountOf(string s, string sub)
        {
            int n = 0, i = 0;
            while ((i = s.IndexOf(sub, i, StringComparison.Ordinal)) >= 0) { n++; i += sub.Length; }
            return n;
        }

        // ---- <|image_pad|> expansion ----------------------------------------

        [Fact]
        public void ExpandImagePads_TwoPads_ExpandsEachWithItsOwnCount()
        {
            const int pad = 151655;
            int[] tokens = { 10, pad, 11, 12, pad, 13 };
            int[] expanded = QwenImagePipeline.ExpandImagePads(tokens, pad, new[] { 3, 2 }, out int[] starts);

            Assert.Equal(new[] { 10, pad, pad, pad, 11, 12, pad, pad, 13 }, expanded);
            Assert.Equal(1, starts[0]);   // first span starts after token 10
            Assert.Equal(6, starts[1]);   // second span index is in the EXPANDED array
        }

        [Fact]
        public void ExpandImagePads_MorePadsThanCounts_LeavesExtraPadsUntouched()
        {
            const int pad = 7;
            int[] tokens = { pad, 1, pad };
            int[] expanded = QwenImagePipeline.ExpandImagePads(tokens, pad, new[] { 2 }, out int[] starts);
            Assert.Equal(new[] { pad, pad, 1, pad }, expanded);
            Assert.Equal(0, starts[0]);
        }

        [Fact]
        public void ExpandImagePads_MissingPad_ReportsMinusOne()
        {
            int[] tokens = { 1, 2, 3 };
            int[] expanded = QwenImagePipeline.ExpandImagePads(tokens, 99, new[] { 4 }, out int[] starts);
            Assert.Equal(tokens, expanded);
            Assert.Equal(-1, starts[0]);
        }

        // ---- Qwen2.5-VL M-RoPE positions (get_rope_index) --------------------

        [Fact]
        public void BuildPositions_TextOnly_AllAxesSequential()
        {
            int[] pos = QwenImageTextEncoder.BuildPositions(4, null);
            for (int s = 0; s < 4; s++)
            {
                Assert.Equal(s, pos[s]);          // t
                Assert.Equal(s, pos[4 + s]);      // h
                Assert.Equal(s, pos[8 + s]);      // w
            }
        }

        [Fact]
        public void BuildPositions_TwoImageSpans_EachGetsItsOwnGridAndAdvance()
        {
            // layout: [txt, txt, img0(2x2 grid -> 4 tokens), txt, img1(1x2 -> 2 tokens), txt]
            var imgs = new[]
            {
                new ImageCond { Start = 2, Count = 4, GridH = 4, GridW = 4 },   // llm grid 2x2
                new ImageCond { Start = 7, Count = 2, GridH = 2, GridW = 4 },   // llm grid 1x2
            };
            int seq = 10;
            int[] pos = QwenImageTextEncoder.BuildPositions(seq, imgs);

            int T(int s) => pos[s]; int H(int s) => pos[seq + s]; int W(int s) => pos[2 * seq + s];

            // leading text: 0,1
            Assert.Equal(0, T(0)); Assert.Equal(1, T(1));
            // image 0 at cur=2: t=2 for all, h=2+row, w=2+col over a 2x2 grid
            Assert.Equal(2, T(2)); Assert.Equal(2, H(2)); Assert.Equal(2, W(2));   // (0,0)
            Assert.Equal(2, T(3)); Assert.Equal(2, H(3)); Assert.Equal(3, W(3));   // (0,1)
            Assert.Equal(2, T(4)); Assert.Equal(3, H(4)); Assert.Equal(2, W(4));   // (1,0)
            Assert.Equal(2, T(5)); Assert.Equal(3, H(5)); Assert.Equal(3, W(5));   // (1,1)
            // after image 0: cur advanced by max(2,2)=2 -> text at 4
            Assert.Equal(4, T(6)); Assert.Equal(4, H(6)); Assert.Equal(4, W(6));
            // image 1 at cur=5: 1x2 grid
            Assert.Equal(5, T(7)); Assert.Equal(5, H(7)); Assert.Equal(5, W(7));   // (0,0)
            Assert.Equal(5, T(8)); Assert.Equal(5, H(8)); Assert.Equal(6, W(8));   // (0,1)
            // after image 1: cur advanced by max(1,2)=2 -> text at 7
            Assert.Equal(7, T(9)); Assert.Equal(7, H(9)); Assert.Equal(7, W(9));
        }

        // ---- DiT RoPE for several reference streams --------------------------

        [Fact]
        public void DitRope_MultiRef_FrameAxisSeparatesImages_SpatialAxesCentered()
        {
            // generated 2x2 + two refs 2x2 (frame index 0/1/2 per DitRope's idx scheme)
            var shapes = new (int f, int h, int w)[] { (1, 2, 2), (1, 2, 2), (1, 2, 2) };
            var rope = DitRope.Build(shapes, txtSeq: 3);

            int imgSeq = 12;                                     // 3 * 2*2
            Assert.Equal(imgSeq * 64, rope.ImgCos.Length);
            Assert.Equal(3 * 64, rope.TxtCos.Length);

            // Same in-grid token of gen/ref1/ref2 must agree on the h/w axis columns
            // (cols 8..63) and differ only on the frame-axis columns (cols 0..7).
            for (int tok = 0; tok < 4; tok++)
            {
                long g = (long)tok * 64, r1 = (long)(4 + tok) * 64, r2 = (long)(8 + tok) * 64;
                for (int c = 8; c < 64; c++)
                {
                    Assert.Equal(rope.ImgCos[g + c], rope.ImgCos[r1 + c], 6);
                    Assert.Equal(rope.ImgCos[g + c], rope.ImgCos[r2 + c], 6);
                }
            }
            // frame axis: gen has pos 0 (cos=1, sin=0); refs have pos 1 and 2 (differ from gen and each other)
            Assert.Equal(1f, rope.ImgCos[0], 6);
            Assert.Equal(0f, rope.ImgSin[0], 6);
            Assert.NotEqual(rope.ImgCos[4 * 64], rope.ImgCos[0]);
            Assert.NotEqual(rope.ImgCos[8 * 64], rope.ImgCos[4 * 64]);
        }

        [Fact]
        public void DitRope_MultiRef_DifferentRefGridSizes_ProduceCorrectLengths()
        {
            // gen 4x3, ref1 4x3, ref2 2x5 (a second image with a different aspect)
            var shapes = new (int f, int h, int w)[] { (1, 4, 3), (1, 4, 3), (1, 2, 5) };
            var rope = DitRope.Build(shapes, txtSeq: 5);
            Assert.Equal((12 + 12 + 10) * 64, rope.ImgCos.Length);
            Assert.Equal((12 + 12 + 10) * 64, rope.ImgSin.Length);
            Assert.Equal(5 * 64, rope.TxtCos.Length);
        }
    }
}
