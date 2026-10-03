// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    internal static partial class GgmlNative
    {
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35ArenaDecodeBatchedHidden(
            [In] Qwen35LayerDecodeArgs[] layers, int numLayers, int nSeqs,
            [In] int[] tokenIds, [In] int[] positions, [In] int[] ropePositions,
            [In] IntPtr[] kCaches, [In] IntPtr[] vCaches,
            [In] IntPtr[] convStates, [In] IntPtr[] deltaStates,
            [In] int[] gdnHostAuth, [In] int[] cacheSizes,
            int numHeads, int numKvHeads, int headDim,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm,
            IntPtr tokenEmbd, int tokenEmbdType,
            long tokenEmbdNe0, long tokenEmbdNe1, long tokenEmbdBytes,
            IntPtr sampled, int wantLogits, IntPtr embeddingRows);

        internal static int Qwen35ArenaDecodeBatchedHiddenStatus(
            Qwen35LayerDecodeArgs[] layers, int numLayers, int nSeqs,
            int[] tokenIds, int[] positions, int[] ropePositions,
            IntPtr[] kCaches, IntPtr[] vCaches,
            IntPtr[] convStates, IntPtr[] deltaStates,
            int[] gdnHostAuth, int[] cacheSizes,
            int numHeads, int numKvHeads, int headDim,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm,
            IntPtr tokenEmbd, int tokenEmbdType,
            long tokenEmbdNe0, long tokenEmbdNe1, long tokenEmbdBytes,
            IntPtr sampled, bool wantLogits, IntPtr embeddingRows)
        {
            // Supported device GET_ROWS types retain the original ABI. This also
            // allows updated managed code to run with an older native library.
            if (embeddingRows == IntPtr.Zero)
                return Qwen35ArenaDecodeBatchedStatus(layers, numLayers, nSeqs, tokenIds, positions, ropePositions,
                kCaches, vCaches, convStates, deltaStates, gdnHostAuth, cacheSizes,
                numHeads, numKvHeads, headDim, ropeNDims, ropeMode, kvCacheType,
                convKernel, headKDim, headVDim, numKHeads, numVHeads,
                eps, ropeBase, ropeFreqScale,
                numExperts, numExpertsUsed, expertFf, sharedFf, normTopk, expertWeightsScale,
                logits, vocabSize, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                finalNorm, tokenEmbd, tokenEmbdType, tokenEmbdNe0, tokenEmbdNe1, tokenEmbdBytes,
                sampled, wantLogits);

            return TSGgml_Qwen35ArenaDecodeBatchedHidden(layers, numLayers, nSeqs, tokenIds, positions, ropePositions,
                kCaches, vCaches, convStates, deltaStates, gdnHostAuth, cacheSizes,
                numHeads, numKvHeads, headDim, ropeNDims, ropeMode, kvCacheType,
                convKernel, headKDim, headVDim, numKHeads, numVHeads,
                eps, ropeBase, ropeFreqScale,
                numExperts, numExpertsUsed, expertFf, sharedFf, normTopk, expertWeightsScale,
                logits, vocabSize, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                finalNorm, tokenEmbd, tokenEmbdType, tokenEmbdNe0, tokenEmbdNe1, tokenEmbdBytes,
                sampled, wantLogits ? 1 : 0, embeddingRows);
        }
    }
}

