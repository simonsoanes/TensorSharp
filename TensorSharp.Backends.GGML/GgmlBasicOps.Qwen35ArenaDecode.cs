// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;

namespace TensorSharp.GGML
{
    public partial class GgmlBasicOps
    {
        /// <summary>Host embeddings are F32 [nSeqs, hiddenSize] in caller
        /// sequence order. Pass zero to use the original device GET_ROWS ABI.
        /// Returns 1 for success, 0 for a safe pre-compute decline, and -1 for
        /// potentially advanced recurrent state, which must fail the requests.</summary>
        public static int Qwen35ArenaDecodeBatchedHiddenStatus(
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
            => GgmlNative.Qwen35ArenaDecodeBatchedHiddenStatus(layers, numLayers, nSeqs, tokenIds, positions, ropePositions,
                kCaches, vCaches, convStates, deltaStates, gdnHostAuth, cacheSizes,
                numHeads, numKvHeads, headDim, ropeNDims, ropeMode, kvCacheType,
                convKernel, headKDim, headVDim, numKHeads, numVHeads,
                eps, ropeBase, ropeFreqScale,
                numExperts, numExpertsUsed, expertFf, sharedFf, normTopk, expertWeightsScale,
                logits, vocabSize, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                finalNorm, tokenEmbd, tokenEmbdType, tokenEmbdNe0, tokenEmbdNe1, tokenEmbdBytes,
                sampled, wantLogits, embeddingRows);
    }
}

