// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>V4.1 selects its own native graph; it cannot run the V4 graph.</summary>
    internal static class DeepSeek41Architecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "deepseek41",
            DisplayName = "DeepSeek V4.1 Flash",
            Aliases = new[] { "deepseek41" },
            MultiGpu = MultiGpuMode.LayerSplit,
            MultiGpuLimitation = "DeepSeek V4.1 (deepseek41) uses a single-process native executor with layer placement and optional routed-MoE tensor parallelism; full attention tensor parallelism and distributed groups are not implemented.",
            SupportsNativeTensorParallel = (degree, backend) => backend == BackendType.GgmlCuda &&
                degree is >= 2 and <= 8,
            DescribeMultiGpuPlacement = DescribePlacement,
            LayerSplitBackends = new[] { BackendType.GgmlCuda, BackendType.Cuda },
            ApplyNativeTunables = c => ValidateLoad(c.GgufPath, c.Backend, c.DraftModelPath, c.TpGroup, c.TpDegree),
            ProjectorFileHints = new[] { "deepseek41.vision.gguf" },
            Factory = c => new DeepSeek41Model(c.GgufPath, c.Backend,
                c.TpDegree, c.TpGroup, c.DraftModelPath, c.LayerSplitDegree),
        };

        internal static void ValidateLoad(string ggufPath, BackendType backend, string draftModelPath,
            ITensorParallelGroup tpGroup = null, int tpDegree = 1)
        {
            // Every backend that has a V4.1 implementation is allowed here:
            // ggml_cuda (the ggml graph), `cuda` (the direct-CUDA engine, which
            // owns its own kernels and does not go through ggml at all),
            // ggml_cpu (the ggml cpu_only branch, where the architecture's fused
            // ops fall back to their scalar implementations) and `cpu` (the pure
            // C# executor). Only ggml_cuda is a serving path; the other three
            // are correctness and portability paths (the direct-CUDA engine is
            // not yet held to a numerical gate).
            // A non-CUDA ggml GPU backend can run V4.1 the same way ggml_cpu
            // does -- its architecture-specific ops fall to the CPU backend's
            // scalar implementations -- but at a host round trip per occurrence.
            // It is opt-in for the same reason the native loader makes it
            // opt-in: what was originally refused was a SILENT fallback onto
            // whichever GPU enumerated first, not an explicit request.
            bool nonCudaGpuAllowed =
                Environment.GetEnvironmentVariable("TS_DSV41_ALLOW_NON_CUDA_GPU") == "1" &&
                (backend == BackendType.GgmlVulkan || backend == BackendType.GgmlMetal);
            if (backend != BackendType.GgmlCuda && backend != BackendType.Cuda &&
                backend != BackendType.GgmlCpu && backend != BackendType.Cpu && !nonCudaGpuAllowed)
                throw new NotSupportedException(
                    "DeepSeek V4.1 Flash requires --backend ggml_cuda (the serving backend), or " +
                    "--backend cuda (the direct-CUDA engine, not yet held to a numerical gate), --backend ggml_cpu " +
                    "or --backend cpu (correctness and portability paths only). " +
                    "TS_DSV41_ALLOW_NON_CUDA_GPU=1 additionally permits ggml_vulkan/ggml_metal, whose "
                    + "architecture-specific ops run on the CPU backend.");
            // A q8_0/q4_0 cache cannot be honoured on any V4.1 executor; refuse it
            // here, before the checkpoint is opened, instead of loading with F16
            // caches while reporting the requested dtype (which is what happened).
            DeepSeek4Architecture.RefuseBlockQuantizedKvCache("DeepSeek V4.1 Flash", v41: true);
            if (tpGroup != null)
                throw new NotSupportedException(
                    "DeepSeek V4.1 uses a single-process native executor and does not support distributed tensor-parallel groups. " +
                    "Start without --tp-node-id/--tp-peers.");
            if (!string.IsNullOrWhiteSpace(draftModelPath) &&
                backend != BackendType.GgmlCuda && backend != BackendType.GgmlCpu)
                throw new NotSupportedException(
                    "DeepSeek V4.1 DSpark requires the TensorSharp ggml executor (--backend ggml_cuda or ggml_cpu) " +
                    "and a matching deepseek41-dspark drafter. Other executors do not implement the V4.1 draft graph.");
            string draft = DeepSeek4Model.ResolveDsparkPath(draftModelPath);
            if (draft != null)
            {
                using var file = GgufFile.OpenWithoutSiblingShards(draft);
                ValidateDsparkArchitecture(file.GetString("general.architecture"));
            }

            // Routed-MoE TP shards expert dimensions across GPUs, so the native
            // loader refuses it under cpu_only. Say so here instead, before a
            // 246 GiB checkpoint is opened.
            if (ResolveTensorParallelRanks(tpDegree) > 0 && backend != BackendType.GgmlCuda)
                throw new NotSupportedException(
                    "DeepSeek V4.1 routed-MoE tensor parallelism requires --backend ggml_cuda. " +
                    "Select ggml_cuda, or omit --tp.");

            // TS_DSV41_ENGRAM_DEVICE picks where the Engram tables live. The
            // native loader reads it only off cpu_only, so on this backend `=1`
            // — which means "require GPU residency and fail if it does not
            // fit" — would be accepted and then ignored, and a malformed
            // value that ggml_cuda rejects would pass. Refuse both here, with
            // the loader's own wording for the malformed case. `=0` asks for
            // the host mappings this path already uses, so it is allowed.
            if ((backend == BackendType.GgmlCpu || backend == BackendType.Cpu) &&
                Environment.GetEnvironmentVariable("TS_DSV41_ENGRAM_DEVICE") is string engramDevice &&
                engramDevice != "0")
            {
                string name = backend == BackendType.Cpu ? "cpu" : "ggml_cpu";
                throw new NotSupportedException(engramDevice == "1"
                    ? "TS_DSV41_ENGRAM_DEVICE=1 needs a GPU to place the Engram tables on and cannot be combined " +
                      $"with --backend {name}, where they are always host mappings. Unset it or run on ggml_cuda."
                    : "TS_DSV41_ENGRAM_DEVICE must be 0 or 1 (unset selects automatically).");
            }


        }

        /// <summary>
        /// Null unless the run is about to put V4.1 on the CPU backend, in which
        /// case the operator gets told. Emitted by the DeepSeek4Model
        /// constructor rather than from ValidateLoad, because ValidateLoad runs
        /// twice per load (once as the descriptor's ApplyNativeTunables, once in
        /// the constructor) and a warning printed twice reads like two problems.
        ///
        /// <para>This exists because the server's default backend is
        /// <c>ggml_cpu</c> on everything but macOS, so forgetting
        /// <c>--backend ggml_cuda</c> on a GPU box used to be a loud refusal and
        /// is now a load that succeeds. Succeeding is the point of the CPU path,
        /// but succeeding into an unusable 246 GiB scalar decode without a word
        /// would be worse than the refusal it replaced.</para>
        /// </summary>
        internal static string DescribeCpuBackendChoice(BackendType backend) => backend switch
        {
            BackendType.GgmlCpu =>
                "[dsv41] --backend ggml_cpu: DeepSeek V4.1 will run on ONE CPU device, with the scalar CPU " +
                "implementations of its architecture-specific ops. That is a correctness and portability path, " +
                "not a serving path: every decoded token reads six of 384 routed experts in each layer. " +
                "Pass --backend ggml_cuda for the GPU executor (this backend is also the default when --backend " +
                "is omitted off macOS). TS_DSV4_THREADS sets the compute thread count; it defaults to at most 32.",
            BackendType.Cpu =>
                "[dsv41] --backend cpu: DeepSeek V4.1 will run on the pure C# executor, with no ggml and no GPU. " +
                "That is a correctness and portability path, not a serving path: every decoded token reads six of " +
                "384 routed experts in each layer, straight out of the memory-mapped shards. Pass --backend " +
                "ggml_cuda for the GPU executor. TS_DSV4_THREADS sets the compute thread count.",
            _ => null,
        };

        /// <summary>The routed-MoE tensor-parallel rank count the native loader is
        /// given: <c>--tp</c>'s degree, or 0 (layer placement) without one.</summary>
        internal static int ResolveTensorParallelRanks(int tpDegree)
        {
            if (tpDegree < 1) throw new ArgumentOutOfRangeException(nameof(tpDegree));
            if (tpDegree > 8)
                throw new NotSupportedException("DeepSeek V4.1 tensor parallelism supports 2 through 8 GPUs.");
            return tpDegree > 1 ? tpDegree : 0;
        }

        internal static void ValidateDsparkArchitecture(string architecture)
        {
            if (!string.Equals(architecture, "deepseek41-dspark", StringComparison.Ordinal))
                throw new NotSupportedException(
                    "DeepSeek V4.1 DSpark requires a deepseek41-dspark artifact; V4 and other draft architectures are incompatible.");
        }

        /// <summary>The <c>--layer-split N</c> placement line.</summary>
        private static string DescribePlacement(int layerSplitDegree)
            => $"  Multi-GPU: DeepSeek V4.1 uses {layerSplitDegree} GPUs by LAYER SPLIT (whole-layer placement). " +
               "Use --tp 2..8 for routed-MoE tensor parallelism.";

        /// <summary>The <c>--tp N</c> placement line.</summary>
        internal static string DescribeTensorParallelPlacement(int ranks)
            => $"  Multi-GPU: DeepSeek V4.1 routed-MoE tensor parallelism across {ranks} GPUs: " +
               "gate/up/down expert dimensions are sharded, with exact F32 activation/output gathers. " +
               "Attention and shared experts retain layer placement; CPU-offloaded layers retain whole CPU experts. " +
               "Full attention tensor parallelism and distributed groups are not supported.";
    }
}
