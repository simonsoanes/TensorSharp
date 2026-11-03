// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Liquid LFM3 (and the LFM3-VL text tower) architecture plug-in.</summary>
    internal static class Lfm3Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "lfm3",
            DisplayName = "LFM3",
            Aliases = ["lfm3"],
            Factory = c => new Lfm3Model(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup),
            // Heads, short-conv channels and the FFN shard across ranks; the KV-state
            // snapshot is mirrored across nodes (Lfm3Model.Snapshot.cs).
            MultiGpu = MultiGpuMode.TensorParallel,
        };
    }
}
