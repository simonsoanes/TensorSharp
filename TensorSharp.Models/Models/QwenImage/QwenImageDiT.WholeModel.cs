// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Whole-DiT forward in ONE resident-weight ggml graph (TSGgml_QwenImageForward).
//
// The per-block native path runs 60 separate device dispatches per forward, each
// preceded by a HOST-precomputed per-token modulation (PrecomputeMod expands temb
// into [seq,3072]x6 on the CPU) that is then uploaded and followed by a device
// sync — ~180 CPU<->GPU serialisations per forward that starve the GPU. This path
// hands the whole model to one kernel: weights stay resident (cached by their GGUF
// pointer) and the AdaLN modulation is computed in-graph, so a denoise step uploads
// only the small img/txt/rope inputs and does a single compute + sync. Mirrors
// stable-diffusion.cpp's QwenImageModel::forward_orig. Default-on for the GGML CUDA
// native path; TS_QWEN_DIT_WHOLE=0 forces the per-block path (e.g. for the
// First-Block-Cache, which needs block 0 computed separately).
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.Core;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    internal sealed partial class QwenImageDiT
    {
        // Whole-DiT single-graph forward. On CUDA the persistent CUDA-graph capture inside
        // TSGgml_QwenImageForward removes the per-op launch overhead; on Metal the SAME
        // export takes the non-persistent branch (one graph build + one graph_compute + one
        // synchronous readback, weights resident-cached by their GGUF pointer). Enabling it
        // on Metal replaces the 60 per-block dispatches — each with a full img/txt residual
        // host round-trip + device sync that leaves the GPU idle between blocks — with a
        // single pipelined submission (stable-diffusion.cpp's QwenImageModel::forward_orig
        // model). Metal never offloads (unified memory), so OffloadCpu is always false here.
        internal bool WholeModelOn =>
            NativeBlockOn && (_backend == BackendType.GgmlCuda || _backend == BackendType.GgmlMetal) && !OffloadCpu &&
            Environment.GetEnvironmentVariable("TS_QWEN_DIT_WHOLE") != "0";

        // CPU-offload fast path: run the 60 blocks as a FEW chunked whole-model graphs
        // (weights as per-call input slots in the shared reuse gallocr) instead of the
        // 60-dispatch per-block loop. Same in-graph AdaLN modulation as the resident
        // whole-model graph, so it eliminates the per-block host modulation expansion
        // (PrecomputeMod: ~6 x [seq,3072] arrays per stream per block, ~1 GB of PCIe
        // per block at 1 MP) and the per-block ModParams matmuls — the dominant cost
        // of the old offload path (measured ~75 s/step at 12.7k tokens; the chunked
        // path uploads only chunk weights + the img/txt streams per chunk boundary).
        // VRAM holds one chunk's weight slots (~totalWeights/numChunks) because the
        // reuse gallocr's buffer is re-planned (and thus re-used) chunk over chunk.
        internal static int OffloadChunkBlocks
        {
            get
            {
                var v = Environment.GetEnvironmentVariable("TS_QWEN_DIT_OFFLOAD_CHUNK");
                return v != null && int.TryParse(v, out var n) && n > 0 ? Math.Min(n, NumLayers) : 10;
            }
        }

        // Run blocks [start, NumLayers) in offload mode via chunked whole-model graphs.
        // Chunks that fail fall back to the per-block loop FROM THE FAILED CHUNK on
        // (earlier chunks already updated imgHost/txtHost in place). Returns true when
        // the layer range was fully applied by either route.
        private bool TryOffloadChunkedBlocks(float[] imgHost, int imgSeq, float[] txtHost, int txtSeq,
            Tensor temb, int[] modulateIndex, DitRope rope, int start)
        {
            if (_backend != BackendType.GgmlCuda) return false;
            int chunk = OffloadChunkBlocks;
            for (int s = start; s < NumLayers; s += chunk)
            {
                int count = Math.Min(chunk, NumLayers - s);
                bool ok = TryWholeBlocks(imgHost, imgSeq, txtHost, txtSeq, temb, modulateIndex, rope, s, count);
                if (!ok)
                {
                    // Finish the remaining layers on the per-block streaming path rather
                    // than corrupting the half-updated residual streams.
                    if (!FusedBlockOn) WarnLoraSkipped();
                    for (int layer = s; layer < NumLayers; layer++)
                        RunNativeLayer(layer, imgHost, imgSeq, txtHost, txtSeq, temb, modulateIndex, rope);
                    return true;
                }
            }
            return true;
        }

        // Run transformer blocks [layerStart, layerStart+layerCount) in ONE resident-weight
        // graph (in-graph AdaLN modulation, no per-block host-modulation upload + sync).
        // imgHost/txtHost are the post-prelude residual streams (img_in / txt_norm+txt_in
        // already applied by the shared C# prelude) and are updated IN PLACE. Returns false
        // if the native kernel declined (caller falls back to the per-block loop). The layer
        // range lets the First-Block-Cache run block 0 separately then this kernel for 1..N-1.
        private unsafe bool TryWholeBlocks(float[] imgHost, int imgSeq, float[] txtHost, int txtSeq,
            Tensor temb, int[] modulateIndex, DitRope rope, int layerStart, int layerCount)
        {
            float[] tembHost = TensorToHost(temb, 2L * Dim);

            // RoPE expanded to the interleaved-duplicated [head_dim, seq] layout.
            float[] iCos = CosFull(rope.ImgCos, imgSeq), iSin = CosFull(rope.ImgSin, imgSeq);
            float[] tCos = CosFull(rope.TxtCos, txtSeq), tSin = CosFull(rope.TxtSin, txtSeq);
            int[] modIdx = modulateIndex ?? new int[imgSeq];   // 0 = generated when null

            var blocks = new QImgBlockW[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                int l = layerStart + i;
                string b = $"transformer_blocks.{l}";
                blocks[i] = new QImgBlockW
                {
                    ImgMod = GgufW($"{b}.img_mod.1.weight", $"{b}.img_mod.1.bias"),
                    TxtMod = GgufW($"{b}.txt_mod.1.weight", $"{b}.txt_mod.1.bias"),
                    ToQ = GgufW($"{b}.attn.to_q.weight", $"{b}.attn.to_q.bias"),
                    ToK = GgufW($"{b}.attn.to_k.weight", $"{b}.attn.to_k.bias"),
                    ToV = GgufW($"{b}.attn.to_v.weight", $"{b}.attn.to_v.bias"),
                    ToOut = GgufW($"{b}.attn.to_out.0.weight", $"{b}.attn.to_out.0.bias"),
                    AddQ = GgufW($"{b}.attn.add_q_proj.weight", $"{b}.attn.add_q_proj.bias"),
                    AddK = GgufW($"{b}.attn.add_k_proj.weight", $"{b}.attn.add_k_proj.bias"),
                    AddV = GgufW($"{b}.attn.add_v_proj.weight", $"{b}.attn.add_v_proj.bias"),
                    ToAddOut = GgufW($"{b}.attn.to_add_out.weight", $"{b}.attn.to_add_out.bias"),
                    NormQ = GgufF32Ptr($"{b}.attn.norm_q.weight"),
                    NormK = GgufF32Ptr($"{b}.attn.norm_k.weight"),
                    NormAq = GgufF32Ptr($"{b}.attn.norm_added_q.weight"),
                    NormAk = GgufF32Ptr($"{b}.attn.norm_added_k.weight"),
                    INet0 = GgufW($"{b}.img_mlp.net.0.proj.weight", $"{b}.img_mlp.net.0.proj.bias"),
                    INet2 = GgufW($"{b}.img_mlp.net.2.weight", $"{b}.img_mlp.net.2.bias"),
                    TNet0 = GgufW($"{b}.txt_mlp.net.0.proj.weight", $"{b}.txt_mlp.net.0.proj.bias"),
                    TNet2 = GgufW($"{b}.txt_mlp.net.2.weight", $"{b}.txt_mlp.net.2.bias"),
                };
            }

            var handles = new List<GCHandle>(16);
            IntPtr Pin<T>(T[] a) where T : struct
            {
                var h = GCHandle.Alloc(a, GCHandleType.Pinned);
                handles.Add(h);
                return h.AddrOfPinnedObject();
            }
            try
            {
                var d = new QwenImageForwardArgs
                {
                    Img = Pin(imgHost), Txt = Pin(txtHost), Temb = Pin(tembHost),
                    ImgCos = Pin(iCos), ImgSin = Pin(iSin), TxtCos = Pin(tCos), TxtSin = Pin(tSin),
                    ModulateIndex = Pin(modIdx),
                    Blocks = Pin(blocks),
                    StructBytes = Marshal.SizeOf<QwenImageForwardArgs>(),
                    Dim = Dim, Heads = NumHeads, HeadDim = HeadDim, Ff = 12288,
                    ImgSeq = imgSeq, TxtSeq = txtSeq, NumLayers = layerCount, Eps = Eps,
                };
                return GgmlBasicOps.TryQwenImageForward(in d);
            }
            finally
            {
                foreach (var h in handles) h.Free();
            }
        }
    }
}
