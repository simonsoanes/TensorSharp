// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>Qwen3.8-Flash-Next architecture plug-in: hyper-connections, PLE n-gram
    /// embeddings, Qwen Sparse Attention and Gated DeltaNet over a 512-expert MoE.</summary>
    internal static class Qwen4ExpArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = Qwen4ExpModel.ArchitectureId,
            DisplayName = "Qwen3.8-Flash-Next",
            Aliases = new[] { Qwen4ExpModel.ArchitectureId },
            // --backend cuda loads the direct-CUDA whole-model engine; the GGML backends the native
            // token spans.
            Factory = c => c.Backend == BackendType.Cuda
                ? new Qwen4ExpCudaModel(c.GgufPath, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath)
                : new Qwen4ExpModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree, c.DraftModelPath),
            ProjectorFileHints = new[] { "*mmproj*.gguf" },

            MultiGpu = MultiGpuMode.TensorParallel,
            SupportsLayerSplit = true,
            // The direct-CUDA engine places whole layers too.
            LayerSplitBackends = new[] { BackendType.GgmlCuda, BackendType.GgmlVulkan, BackendType.Cuda },
            SupportsDistributedTensorParallel = false,

        };
    }
}
