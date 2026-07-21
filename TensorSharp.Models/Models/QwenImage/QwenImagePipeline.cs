// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Orchestrates the Qwen-Image-Edit denoising pipeline: preprocess the input image,
    /// VAE-encode it to a (normalized, packed) reference latent, build text conditioning,
    /// run the FlowMatch-Euler DiT denoise loop (true-CFG, reference-latent concatenation),
    /// then VAE-decode the result back to pixels.
    /// </summary>
    internal sealed class QwenImagePipeline : IDisposable
    {
        // VAE latent normalization (diffusers AutoencoderKLQwenImage config).
        private static readonly float[] LatentsMean = {
            -0.7571f,-0.7089f,-0.9113f,0.1075f,-0.1745f,0.9653f,-0.1517f,1.5508f,
            0.4134f,-0.0715f,0.5517f,-0.3632f,-0.1922f,-0.9497f,0.2503f,-0.2921f };
        private static readonly float[] LatentsStd = {
            2.8184f,1.4541f,2.3275f,2.6558f,1.2196f,1.7708f,2.6052f,2.0743f,
            3.2687f,2.1526f,2.8652f,1.5579f,1.6382f,1.1253f,2.8251f,1.916f };
        private const int C = QwenImageModel.VaeLatentChannels; // 16
        // Qwen-Image's native training/inference resolution (diffusers/sd.cpp size the edit to
        // ~1 MP). The VRAM area clamp targets this as the high-quality ceiling.
        private const long QwenImageNativeArea = 1024L * 1024;

        private readonly QwenImageModel _model;
        private QwenImageVae _vae;
        private QwenImageTextEncoder _te;
        private QwenImageDiT _dit;
        private QwenImageVisionEncoder _vision;

        public QwenImagePipeline(QwenImageModel model) { _model = model; }

        private QwenImageVae Vae => _vae ??= new QwenImageVae(_model);
        private QwenImageTextEncoder Te => _te ??= new QwenImageTextEncoder(_model.TePath, _model.Backend);
        private QwenImageDiT Dit => _dit ??= new QwenImageDiT(_model.DitGgufPath, _model.Backend);
        private QwenImageVisionEncoder Vision => _vision ??= new QwenImageVisionEncoder(_model.MmprojPath, _model.Backend);

        // Vision grounding (Stage 4): the Qwen2.5-VL vision tower + M-RoPE image conditioning are
        // VERIFIED correct against the real transformers model (block-0 cosine 0.99997; window_index,
        // M-RoPE section mapping, position_ids, tokenizer all match exactly). The remaining cosine
        // gap vs transformers is Q4_K-LLM quantization + fp32-vs-bf16 precision (same as the
        // text-only path), not a bug. This is the model's INTENDED path (the DiT was trained with
        // image-grounded conditioning), so it's on by default when the mmproj is present.
        // CAVEAT: the 264-token image conditioning makes each CPU denoise step ~128s and the grounded
        // generation needs more steps for clean output — use the CUDA path (Stage 7) for practical
        // full-quality edits, or TS_QWEN_IMAGE_NO_VISION=1 for the faster (ungrounded) text-only path.
        private bool UseVision =>
            _model.MmprojPath != null &&
            Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_NO_VISION") != "1";

        // The reference (condition) images carried into Edit so EncodePrompt can ground on them,
        // plus the per-request vision-tower outputs (cached so the negative-prompt CFG pass does
        // not re-encode the same images through the vision encoder).
        private RgbImage[] _conditionImages;
        private (int gridH, int gridW, float[] embeds)[] _visionCache;
        // Whether this request installed a device-copy residency cap (CPU offload); reset in
        // the Edit finally so a later non-offload request gets full residency back.
        private bool _offloadBudgetSet;

        public RgbImage Edit(string prompt, RgbImage input, QwenImageParams p) =>
            Edit(prompt, new[] { input }, p);

        /// <summary>
        /// Multi-image edit: <paramref name="inputs"/>[0] drives the output geometry; every
        /// input becomes a "Picture N" vision grounding + a VAE reference-latent stream the
        /// DiT attends to (diffusers <c>QwenImageEditPlusPipeline</c> semantics).
        /// </summary>
        public RgbImage Edit(string prompt, RgbImage[] inputs, QwenImageParams p)
        {
            try
            {
                return EditCore(prompt, inputs, p);
            }
            finally
            {
                // Hand back ALL device residency at request end (DiT weights + LoRA +
                // captured graphs + reuse gallocr + VAE uploads, ~10 GB on CUDA). The
                // captured DiT graph cannot survive into the next request anyway — the
                // next FreeEncoders' global cache clear resets it — but if it lingers
                // here, the next request's text-encoder upload + compute run against
                // ~10 GB of dead residency and spill into shared memory (measured
                // te.llm 1.1s -> 8.3s on a 16 GB card). Runs in a finally because a
                // cancelled stream (client disconnect throws OperationCanceledException
                // out of OnStep mid-denoise) or a decode failure must not strand that
                // residency either — both native frees are idempotent no-ops when
                // nothing is held. CUDA only: Metal's unified memory can't spill this
                // way, so keeping residency there is strictly better.
                if (_model.Backend is BackendType.GgmlCuda)
                {
                    GgmlBasicOps.ReleaseReuseComputeBuffers();
                    GgmlBasicOps.ClearHostBufferCache();
                }
                // Lift the offload residency cap so the next (possibly non-offload) request
                // binds weights resident again at full speed.
                if (_offloadBudgetSet)
                {
                    GgmlBasicOps.SetDeviceCopyBudget(0);
                    _offloadBudgetSet = false;
                }
            }
        }

        private RgbImage EditCore(string prompt, RgbImage[] inputs, QwenImageParams p)
        {
            if (inputs == null || inputs.Length == 0)
                throw new ArgumentException("Qwen-Image-Edit needs at least one input image.", nameof(inputs));
            var phase = System.Diagnostics.Stopwatch.StartNew();
            void Phase(string n) { Console.WriteLine($"  [pipe-timing] {n}: {phase.Elapsed.TotalMilliseconds:F0}ms"); phase.Restart(); }

            // 0. resolve auto sampling params. A Lightning step-distillation LoRA
            // (TS_QWEN_IMAGE_LORA, merged into the DiT at load) changes the sampling regime:
            // its trained step count, cfg 1.0 (no negative pass — the distillation bakes the
            // guidance in), and a FIXED timestep shift of 3 (lightx2v sets base_shift =
            // max_shift = log 3; the dynamic resolution shift would walk a different sigma
            // schedule than the distillation targets). Explicit caller values still win.
            int lightning = QwenImageDiT.LoraPath != null ? QwenImageLoraTable.ParseLightningSteps(QwenImageDiT.LoraPath) : 0;
            int steps = p.Steps > 0 ? p.Steps : (lightning > 0 ? lightning : 30);
            float cfgScale = p.CfgScale > 0f ? p.CfgScale : (lightning > 0 ? 1f : 2.5f);
            // Flow shift for the FlowMatch sigma schedule. Qwen-Image-Edit uses a FIXED shift of
            // 3: stable-diffusion.cpp sets default_flow_shift = 3 for every qwen_image version (its
            // docs use --flow-shift 3), and the Lightning distillation is trained at shift 3. The
            // diffusers *dynamic* resolution-dependent shift (calculate_shift) walks a much lower
            // effective shift (~1.8 at a ~0.4 MP edit) that under-noises the early trajectory and
            // visibly softens / degrades the edit — the "low quality without LoRA" the user hit.
            // Lightning always pins shift 3; the base path defaults to 3 too, overridable via
            // TS_QWEN_IMAGE_FLOW_SHIFT (<= 0 selects the old diffusers dynamic shift).
            float? muOverride;
            if (lightning > 0)
            {
                muOverride = MathF.Log(3f);
                Console.WriteLine($"  [pipe] Lightning LoRA active: steps={steps} cfg={cfgScale} shift=3 (fixed)");
            }
            else
            {
                float flowShift = 3f;
                var fsEnv = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_FLOW_SHIFT");
                if (fsEnv != null && float.TryParse(fsEnv, System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out var fsv))
                    flowShift = fsv;
                muOverride = flowShift > 0f ? MathF.Log(flowShift) : (float?)null;
                Console.WriteLine($"  [pipe] flow shift={(flowShift > 0f ? flowShift.ToString("0.##") : "dynamic")} steps={steps} cfg={cfgScale}");
            }

            _conditionImages = inputs;   // vision grounding uses the originals (resized to 384^2 internally)
            _visionCache = null;         // fresh per request (images differ between requests)
            // 1. preprocess (aspect-preserving, dims multiple of 16). The FIRST image drives the
            // output geometry (diffusers/sd.cpp Edit Plus semantics). Clamp the target area to
            // what the DiT attention will fit in device VRAM — with N reference streams the
            // token count grows ~(N+1)x, so the clamp accounts for the image count.
            // Explicit output size: from the params, or a server-global default set by the
            // server's --width/--height (TS_QWEN_IMAGE_WIDTH/HEIGHT). Snap to a /16 multiple
            // (VAE 8x downsample * 2x2 DiT patch).
            int reqW = p.Width, reqH = p.Height;
            if ((reqW <= 0 || reqH <= 0) &&
                int.TryParse(Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_WIDTH"), out int envW) && envW > 0 &&
                int.TryParse(Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_HEIGHT"), out int envH) && envH > 0)
            {
                reqW = envW; reqH = envH;
            }

            // CPU offload (sd.cpp --offload-to-cpu equivalent): stream the DiT weights from RAM
            // per block instead of holding ~7-9 GB resident in VRAM, freeing that VRAM for the
            // attention working set — the difference between a ~0.4 MP and a native ~1 MP edit
            // on a 16 GB card. TS_QWEN_IMAGE_OFFLOAD_CPU / --offload-cpu: "1" always streams,
            // "0" never (the resolution is clamped to the resident-weight ceiling instead),
            // unset = AUTO (engage exactly when the target resolution does not fit beside the
            // resident weights — quality wins over the streaming slowdown).
            string offloadEnv = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_OFFLOAD_CPU");
            bool offloadAllowed = offloadEnv != "0" && _model.Backend == BackendType.GgmlCuda;
            bool offload = offloadEnv == "1" && _model.Backend == BackendType.GgmlCuda;
            bool doCfgPass = cfgScale > 1f;
            // Reference-latent area (inputs[1..]): the detail source the DiT copies faces /
            // textures from. 0 = couple to the output area (legacy rule, kept for Metal/CPU
            // where the extra ref tokens have no offload path to absorb them).
            long refAreaPref = ResolveRefArea(_model.Backend);
            long refAreaForCeiling = refAreaPref > 0 ? refAreaPref : QwenImageNativeArea;

            long area;
            RgbImage img;
            if (reqW > 0 && reqH > 0)
            {
                reqW = Math.Max(16, reqW / 16 * 16);
                reqH = Math.Max(16, reqH / 16 * 16);
                // Honour the explicit size but never exceed the HARD VRAM ceiling — above it the DiT
                // OOMs and spills into GARBAGE (noise) output rather than erroring, which is what a
                // request like 2048x2048 on a 16 GB card would produce. When the size doesn't fit
                // beside the resident weights, stream the weights from RAM (offload) before giving
                // up resolution; only past the offload ceiling is the size actually clamped.
                long safe = MaxSafeArea(inputs.Length, doCfgPass, offload, refAreaForCeiling);
                if ((long)reqW * reqH > safe && offloadAllowed && !offload)
                {
                    offload = true;
                    safe = MaxSafeArea(inputs.Length, doCfgPass, offload: true, refAreaForCeiling);
                }
                if ((long)reqW * reqH > safe)
                {
                    double s = Math.Sqrt((double)safe / ((double)reqW * reqH));
                    int cw = Math.Max(16, (int)Math.Round(reqW * s / 16.0) * 16);
                    int ch = Math.Max(16, (int)Math.Round(reqH * s / 16.0) * 16);
                    Console.WriteLine($"  [pipe] requested output {reqW}x{reqH} ({(long)reqW * reqH} px) exceeds the " +
                        $"~{safe} px VRAM ceiling for {inputs.Length} input image(s) on this GPU; clamping to {cw}x{ch} " +
                        "to avoid an out-of-memory garbled result (use a smaller size, fewer input images, or more VRAM).");
                    reqW = cw; reqH = ch;
                }
                img = ImageIO.Resize(inputs[0], reqW, reqH);
                area = (long)reqW * reqH;
            }
            else
            {
                area = ResolveAutoArea(p.TargetArea, inputs.Length, steps, doCfgPass, offloadAllowed, ref offload, refAreaForCeiling);
                img = ImageIO.ResizeToArea(inputs[0], area);
            }
            Dit.OffloadCpu = offload;
            if (_model.Backend == BackendType.GgmlCuda)
                GgmlBasicOps.QwenImageSetOffload(offload);   // native: skip captured entries; stream weights
            if (offload)
                Console.WriteLine("  [pipe] CPU offload: DiT weights stream from RAM (VRAM goes to the " +
                                  "attention working set). Slower per step, but keeps the full target resolution.");

            // 2. VAE encode each reference -> normalize -> pack. The first reference is encoded
            // at the output geometry (its RoPE grid then aligns 1:1 with the generated tokens,
            // which helps unchanged regions reconstruct); additional references keep their own
            // aspect ratio at the REFERENCE area — the model's native ~1 MP by default
            // (diffusers Edit Plus VAE_IMAGE_SIZE), deliberately NOT scaled down by a smaller
            // output: this latent is where the DiT copies face/body detail from, so shrinking
            // it with the output silently threw that detail away. TS_QWEN_IMAGE_REF_AREA
            // overrides — each ref costs area/256 attention tokens.
            int Hl = 0, Wl = 0;                                  // gen latent dims (from inputs[0])
            var refPacked = new float[inputs.Length][];
            var refShapes = new (int f, int h, int w)[inputs.Length];
            long refArea = refAreaPref > 0 ? refAreaPref : Math.Min(QwenImageNativeArea, (long)img.Width * img.Height);
            var refLog = new System.Text.StringBuilder();
            for (int i = 0; i < inputs.Length; i++)
            {
                RgbImage r = i == 0 ? img : ImageIO.ResizeToArea(inputs[i], refArea);
                VaeLatent lat = Vae.Encode(r);
                NormalizeLatent(lat.Data, lat.Height, lat.Width);
                refPacked[i] = Pack(lat.Data, lat.Height, lat.Width);   // [refSeq_i, 64]
                refShapes[i] = (1, lat.Height / 2, lat.Width / 2);
                if (i == 0) { Hl = lat.Height; Wl = lat.Width; }
                if (refLog.Length > 0) refLog.Append(", ");
                refLog.Append($"[{i}] {inputs[i].Width}x{inputs[i].Height} -> {r.Width}x{r.Height}{(i == 0 ? " (output geometry)" : "")}");
            }
            Console.WriteLine($"  [pipe] reference latents: {refLog} (ref area {refArea} px; TS_QWEN_IMAGE_REF_AREA overrides)");
            Phase($"VAE encode ({string.Join(", ", Array.ConvertAll(refShapes, s => $"{s.w * 16}x{s.h * 16}"))})");
            int hp = Hl / 2, wp = Wl / 2, seq = hp * wp;

            // 3. text conditioning (prompt + negative for CFG)
            float[] cond = EncodePrompt(prompt, out int txtSeq);
            bool doCfg = cfgScale > 1f;
            float[] negCond = null; int negTxt = 0;
            if (doCfg) negCond = EncodePrompt(p.NegativePrompt ?? " ", out negTxt);
            _visionCache = null;   // release the per-request vision embeds
            Phase($"text encode (txtSeq={txtSeq})");

            // The text + vision encoders are finished for this request. On CUDA they hold
            // ~3.6 GB (Qwen2.5-VL-7B ~2.3 GB + mmproj ~1.3 GB) that is dead weight through
            // the entire denoise loop — leaving it resident pushes the DiT (7 GB) + its
            // O(n^2) attention scratch past 16 GB, spilling into shared VRAM (slow) and
            // then OOM at high resolution. Free it now; the lazy properties reload it on
            // the next Edit. Keep the VAE (small, needed again for the final decode).
            FreeEncoders();

            // CPU offload: cap the resident device-copy set for the denoise. New weight
            // uploads past the budget are DENIED residency and stream through the per-graph
            // upload path instead (sd.cpp --offload-to-cpu semantics: keep in VRAM what fits
            // after the activations are budgeted, stream the rest from RAM every forward).
            // Set AFTER FreeEncoders so the text-encoder pass ran at full speed and its
            // residency is already cleared; the Edit finally lifts the cap again.
            if (Dit.OffloadCpu)
            {
                int refTokens = 0;
                foreach (var s in refShapes) refTokens += s.h * s.w;
                long resBudget = OffloadResidencyBudget(seq + refTokens + Math.Max(txtSeq, negTxt));
                GgmlBasicOps.SetDeviceCopyBudget(resBudget);
                _offloadBudgetSet = true;
                Console.WriteLine($"  [pipe] offload: resident weight budget {resBudget >> 20} MiB; the rest streams from RAM per step.");
            }

            // 4. noise latents (packed) + reference concatenation layout
            var rng = new GaussianRng(p.Seed);
            float[] noise = new float[(long)C * Hl * Wl];
            for (long i = 0; i < noise.Length; i++) noise[i] = rng.Next();
            float[] latents = Pack(noise, Hl, Wl);               // [seq, 64], evolves

            int refSeqTotal = 0;
            foreach (var s in refShapes) refSeqTotal += s.h * s.w;
            int imgSeq = seq + refSeqTotal;                      // gen + all refs
            var modulateIndex = new int[imgSeq];
            for (int i = seq; i < imgSeq; i++) modulateIndex[i] = 1;

            // img_shapes: generated then each reference (frame index 0, 1, 2, ... in DitRope)
            var imgShapes = new (int f, int h, int w)[1 + refShapes.Length];
            imgShapes[0] = (1, hp, wp);
            Array.Copy(refShapes, 0, imgShapes, 1, refShapes.Length);
            DitRope rope = DitRope.Build(imgShapes, txtSeq);
            DitRope ropeNeg = doCfg ? DitRope.Build(imgShapes, negTxt) : null;

            // 5. scheduler
            var sched = new QwenImageScheduler(steps, seq, muOverride);

            // 6. denoise loop
            var imgTokens = new float[(long)imgSeq * 64];
            long refOff = (long)seq * 64;                        // ref parts fixed for the whole loop
            foreach (var rp in refPacked) { Array.Copy(rp, 0, imgTokens, refOff, rp.Length); refOff += rp.Length; }
            // CFG-batching (both branches in one captured dispatch) is a large win where the
            // combined block fits VRAM, but at very large token counts it falls back to a
            // non-persist path that holds both branches at once and can oversubscribe VRAM.
            // Gate it to the budget that fits; larger images keep the two-forward path.
            bool useCfgBatch = doCfg && Dit.UseCfgBatch &&
                               (imgSeq + Math.Max(txtSeq, negTxt)) <= Dit.CfgBatchMaxTokens;
            if (doCfg && Dit.UseCfgBatch && !useCfgBatch)
                Console.WriteLine($"  [pipe] CFG-batch disabled: {imgSeq + Math.Max(txtSeq, negTxt)} tokens > {Dit.CfgBatchMaxTokens} budget (set TS_QWEN_DIT_CFG_BATCH_MAXTOK or a smaller area).");

            Dit.ResetCache(sched.Steps);   // First-Block-Cache: fresh state per generation
            // Whole-step cache (EasyCache port, see QwenImageStepCache): skips the ENTIRE
            // DiT forward (both CFG branches) on low-change steps. OFF by default (quality
            // first, like sd.cpp); TS_QWEN_DIT_CACHE_MODE=easycache/fbc/both opts in.
            var stepCache = new QwenImageStepCache(sched.Steps);
            for (int step = 0; step < sched.Steps; step++)
            {
                Array.Copy(latents, 0, imgTokens, 0, (long)seq * 64);             // gen part = current latents
                float t01 = sched.Sigmas[step];   // FlowMatch: timestep == sigma
                var v = new float[(long)seq * 64];
                float[] velNeg = doCfg ? new float[(long)seq * 64] : null;

                if (stepCache.TrySkip(step, latents, v, velNeg))
                {
                    // Both branches reconstructed from the cached (output - input) diffs;
                    // no device work this step.
                    if (doCfg) TrueCfg(v, velNeg, seq, cfgScale);
                }
                else if (useCfgBatch)
                {
                    // CFG-batched: both branches in one launch-amortized fused pass (halves the
                    // per-block GPU sync + weight upload that starve the launch-bound DiT).
                    var (vc, vn) = Dit.PredictCfg(imgTokens, imgSeq, cond, txtSeq, rope, negCond, negTxt, ropeNeg,
                        t01, modulateIndex, step, sched.Steps, seq);
                    Array.Copy(vc, 0, v, 0, v.Length);
                    Array.Copy(vn, 0, velNeg, 0, velNeg.Length);
                    stepCache.AfterCompute(step, latents, v, velNeg);
                    TrueCfg(v, velNeg, seq, cfgScale);
                }
                else
                {
                    float[] vel = Dit.Predict(imgTokens, imgSeq, cond, txtSeq, t01, modulateIndex, rope,
                        stepIndex: step, totalSteps: sched.Steps, cfgBranch: 0, genTokens: seq);
                    // keep only the generated region
                    Array.Copy(vel, 0, v, 0, v.Length);

                    if (doCfg)
                    {
                        float[] velFull = Dit.Predict(imgTokens, imgSeq, negCond, negTxt, t01, modulateIndex, ropeNeg,
                            stepIndex: step, totalSteps: sched.Steps, cfgBranch: 1, genTokens: seq);
                        Array.Copy(velFull, 0, velNeg, 0, velNeg.Length);
                        stepCache.AfterCompute(step, latents, v, velNeg);
                        // true-CFG: comb = neg + scale*(cond-neg), then renorm to cond norm (per token row)
                        TrueCfg(v, velNeg, seq, cfgScale);
                    }
                    else
                    {
                        stepCache.AfterCompute(step, latents, v, null);
                    }
                }
                sched.Step(latents, v, step);
                Console.Write($"\r  denoise step {step + 1}/{sched.Steps}   ");

                // Live progress: a tick every step (so the UI never looks stuck) plus a decoded
                // preview of the current latent on evenly-spaced steps. Previews are decoded at
                // reduced resolution (cheap vs the full final decode, and keeps VRAM in budget
                // while the DiT weights are still resident). The final step is skipped here — the
                // caller renders the full-resolution result below.
                if (p.OnStep != null)
                {
                    // Ceiling division over PreviewCount+1 spaces the previews evenly across the
                    // loop while never emitting more than PreviewCount of them: floor division
                    // degenerates to interval=1 whenever Steps < 2*PreviewCount (14 decodes for
                    // steps=15, budget 8) and to interval=Steps for tiny runs (steps=2 with
                    // PreviewCount=1 emitted nothing).
                    int interval = p.PreviewCount > 0 ? Math.Max(1, (sched.Steps + p.PreviewCount) / (p.PreviewCount + 1)) : 0;
                    bool emit = interval > 0 && step < sched.Steps - 1 && (step + 1) % interval == 0;
                    RgbImage preview = null;
                    if (emit)
                    {
                        try { preview = DecodePreview(latents, Hl, Wl); }
                        catch (Exception ex) { Console.WriteLine($"  [pipe] preview decode skipped: {ex.Message}"); }
                    }
                    p.OnStep(step + 1, sched.Steps, preview);
                }
            }
            Console.WriteLine();
            if (stepCache.Enabled) Console.WriteLine($"  [pipe] DiT step-cache: {stepCache.Stats()}");
            if (Dit.CacheEnabled) Console.WriteLine($"  [pipe] DiT cache: {Dit.CacheStats()}");
            Phase($"denoise ({sched.Steps} steps)");

            // The DiT denoise loop packed every block into the persistent reuse gallocr
            // (the liveness-packing scratch). At high resolution that buffer holds a few GB
            // that is dead weight through the final VAE decode and would compete with the
            // decode's large im2col scratch for the working set (the 19 GB Metal budget at
            // 1 MP, or VRAM on CUDA). Hand it back now; the VAE decode (and the next Edit's
            // denoise) re-creates the gallocr on demand.
            if (_model.Backend is BackendType.GgmlMetal or BackendType.GgmlCuda or BackendType.GgmlCpu)
                GgmlBasicOps.ReleaseReuseComputeBuffers();

            // 7. unpack -> denormalize -> VAE decode
            float[] outLatent = Unpack(latents, Hl, Wl);
            DenormalizeLatent(outLatent, Hl, Wl);
            RgbImage result = Vae.Decode(new VaeLatent(C, Hl, Wl, outLatent));
            Phase("VAE decode");
            return result;
        }

        // Decode a low-resolution RGB preview of the current (partially denoised) latent for live
        // UI feedback. The latent is spatially average-pooled before the VAE decode so the preview
        // costs a fraction of the full decode (and its conv/im2col scratch stays small enough to fit
        // alongside the still-resident DiT weights). `packed` is the evolving [seq,64] latent.
        private RgbImage DecodePreview(float[] packed, int Hl, int Wl)
        {
            float[] lat = Unpack(packed, Hl, Wl);          // [C, Hl, Wl]
            DenormalizeLatent(lat, Hl, Wl);
            // Pool so the largest latent dim is ~48 (=> ~384 px preview), capping cost at any output
            // resolution; never below a 2x2 grid (the VAE conv stack needs a few cells).
            int f = Math.Max(1, (Math.Max(Hl, Wl) + 47) / 48);
            if (f > 1 && Hl / f >= 2 && Wl / f >= 2)
            {
                float[] small = DownsampleLatent(lat, Hl, Wl, f, out int Hs, out int Ws);
                return Vae.Decode(new VaeLatent(C, Hs, Ws, small));
            }
            return Vae.Decode(new VaeLatent(C, Hl, Wl, lat));
        }

        // Average-pool each latent channel by an integer factor (drops a partial trailing row/col).
        private static float[] DownsampleLatent(float[] lat, int H, int W, int f, out int Ho, out int Wo)
        {
            Ho = H / f; Wo = W / f;
            int hw = H * W, hwo = Ho * Wo;
            var outp = new float[(long)C * hwo];
            float inv = 1f / (f * f);
            for (int c = 0; c < C; c++)
                for (int hi = 0; hi < Ho; hi++)
                    for (int wi = 0; wi < Wo; wi++)
                    {
                        float sum = 0f;
                        for (int dh = 0; dh < f; dh++)
                            for (int dw = 0; dw < f; dw++)
                                sum += lat[(long)c * hw + (hi * f + dh) * W + (wi * f + dw)];
                        outp[(long)c * hwo + hi * Wo + wi] = sum * inv;
                    }
            return outp;
        }

        private float[] EncodePrompt(string prompt, out int condSeq)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            void Sub(string n) { Console.WriteLine($"  [pipe-timing]   te.{n}: {sw.Elapsed.TotalMilliseconds:F0}ms"); sw.Restart(); }

            int[] tokens;
            ImageCond[] imgConds = null;
            if (UseVision && _conditionImages != null && _conditionImages.Length > 0)
            {
                int nImg = _conditionImages.Length;
                string templated = QwenImagePrompt.BuildWithImages(prompt, nImg);
                int[] baseTokens = Te.Tokenizer.Encode(templated, addSpecial: false).ToArray();
                Sub("load+tokenize");
                // Vision-encode each image once per request (the negative-CFG pass reuses the cache).
                if (_visionCache == null)
                {
                    _visionCache = new (int, int, float[])[nImg];
                    for (int i = 0; i < nImg; i++)
                    {
                        float[] pv = QwenImageVisionProcessor.Preprocess(_conditionImages[i], out int gh, out int gw);
                        _visionCache[i] = (gh, gw, Vision.Encode(pv, gh, gw));    // [n_i, 3584]
                        Sub($"vision[{i}]({gh}x{gw})");
                    }
                }
                var counts = new int[nImg];
                for (int i = 0; i < nImg; i++)
                    counts[i] = (_visionCache[i].gridH / 2) * (_visionCache[i].gridW / 2);
                tokens = ExpandImagePads(baseTokens, QwenImagePrompt.ImagePadTokenId, counts, out int[] starts);
                var conds = new System.Collections.Generic.List<ImageCond>(nImg);
                for (int i = 0; i < nImg; i++)
                    if (starts[i] >= 0)
                        conds.Add(new ImageCond
                        {
                            Start = starts[i], Count = counts[i],
                            GridH = _visionCache[i].gridH, GridW = _visionCache[i].gridW,
                            Embeds = _visionCache[i].embeds,
                        });
                if (conds.Count > 0) imgConds = conds.ToArray();
            }
            else
            {
                tokens = Te.Tokenizer.Encode(QwenImagePrompt.Build(prompt), addSpecial: false).ToArray();
                Sub("load+tokenize");
            }

            float[] full = Te.EncodeHidden(tokens, imgConds);   // [seq, 3584]
            Sub($"llm({tokens.Length}tok)");
            int seq = tokens.Length;
            int hidden = Te.HiddenSize;
            int drop = Math.Min(QwenImagePrompt.DropIdx, seq - 1);
            condSeq = seq - drop;
            var cond = new float[(long)condSeq * hidden];
            Array.Copy(full, (long)drop * hidden, cond, 0, cond.Length);
            return cond;
        }

        // Replace the k-th <|image_pad|> placeholder with counts[k] copies (one per merged vision
        // patch of image k); starts[k] receives that span's index in the EXPANDED token array
        // (-1 if the template had fewer placeholders than images — should not happen).
        internal static int[] ExpandImagePads(int[] tokens, int padId, int[] counts, out int[] starts)
        {
            starts = new int[counts.Length];
            int total = tokens.Length;
            foreach (int c in counts) total += c - 1;
            var outp = new System.Collections.Generic.List<int>(total);
            int k = 0;
            foreach (int t in tokens)
            {
                if (t == padId && k < counts.Length)
                {
                    starts[k] = outp.Count;
                    for (int j = 0; j < counts[k]; j++) outp.Add(padId);
                    k++;
                }
                else outp.Add(t);
            }
            for (; k < counts.Length; k++) starts[k] = -1;
            return outp.ToArray();
        }

        // true-CFG over packed velocity rows [seq, 64]
        private static void TrueCfg(float[] cond, float[] neg, int rows, float scale)
        {
            for (int s = 0; s < rows; s++)
            {
                long o = (long)s * 64;
                double condNorm = 0, combNorm = 0;
                var comb = new float[64];
                for (int d = 0; d < 64; d++)
                {
                    float c = cond[o + d], n = neg[o + d];
                    float cb = n + scale * (c - n);
                    comb[d] = cb;
                    condNorm += (double)c * c; combNorm += (double)cb * cb;
                }
                float r = (float)(Math.Sqrt(condNorm) / (Math.Sqrt(combNorm) + 1e-12));
                for (int d = 0; d < 64; d++) cond[o + d] = comb[d] * r;
            }
        }

        // pack [C,H,W] -> [(H/2)*(W/2), C*4], token (hi,wi), channel c*4+ph*2+pw
        private static float[] Pack(float[] lat, int H, int W)
        {
            int hp = H / 2, wp = W / 2, hw = H * W;
            var outp = new float[(long)hp * wp * C * 4];
            for (int hi = 0; hi < hp; hi++)
                for (int wi = 0; wi < wp; wi++)
                {
                    long tok = (long)(hi * wp + wi) * (C * 4);
                    for (int c = 0; c < C; c++)
                        for (int ph = 0; ph < 2; ph++)
                            for (int pw = 0; pw < 2; pw++)
                                outp[tok + c * 4 + ph * 2 + pw] = lat[(long)c * hw + (hi * 2 + ph) * W + (wi * 2 + pw)];
                }
            return outp;
        }

        private static float[] Unpack(float[] packed, int H, int W)
        {
            int hp = H / 2, wp = W / 2, hw = H * W;
            var lat = new float[(long)C * hw];
            for (int hi = 0; hi < hp; hi++)
                for (int wi = 0; wi < wp; wi++)
                {
                    long tok = (long)(hi * wp + wi) * (C * 4);
                    for (int c = 0; c < C; c++)
                        for (int ph = 0; ph < 2; ph++)
                            for (int pw = 0; pw < 2; pw++)
                                lat[(long)c * hw + (hi * 2 + ph) * W + (wi * 2 + pw)] = packed[tok + c * 4 + ph * 2 + pw];
                }
            return lat;
        }

        private static void NormalizeLatent(float[] lat, int H, int W)
        {
            int hw = H * W;
            for (int c = 0; c < C; c++)
                for (int i = 0; i < hw; i++)
                    lat[(long)c * hw + i] = (lat[(long)c * hw + i] - LatentsMean[c]) / LatentsStd[c];
        }
        private static void DenormalizeLatent(float[] lat, int H, int W)
        {
            int hw = H * W;
            for (int c = 0; c < C; c++)
                for (int i = 0; i < hw; i++)
                    lat[(long)c * hw + i] = lat[(long)c * hw + i] * LatentsStd[c] + LatentsMean[c];
        }

        // Resolve the AUTO output area (no explicit --width/--height): QUALITY-FIRST. The target
        // is the full requested area (default: the model's native ~1 MP training resolution —
        // what diffusers QwenImageEditPlusPipeline and sd.cpp render), limited only by the hard
        // VRAM ceiling; when the target does not fit beside the resident DiT weights and offload
        // is allowed, the weights are streamed from RAM (offload=true on return) instead of
        // giving up resolution. The previous behavior scaled the area down by step count and
        // reference count as a SPEED budget — a 30-step 2-image edit rendered at ~0.4 MP with
        // visibly soft, low-resolution faces; speed is now the opt-in (smaller --width/--height
        // or TS_QWEN_IMAGE_MAX_AREA), not the default.
        private long ResolveAutoArea(long requestedArea, int imageCount, int steps, bool doCfg,
            bool offloadAllowed, ref bool offload, long refArea)
        {
            // Metal: no weight-streaming path; keep the established speed budget (a 1 MP
            // 30-step x 2-CFG run is tens of minutes there). TS_QWEN_IMAGE_MAX_AREA or an
            // explicit --width/--height still raise it.
            if (_model.Backend == BackendType.GgmlMetal)
            {
                const int baselineSteps = 12;
                double stepScale = Math.Sqrt((double)baselineSteps / Math.Max(1, steps));
                int streams = imageCount + 1;
                long cap = 512L * 1024;   // ~0.5 MP baseline (30-step) on a 19 GB M4 Pro
                var env = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_MAX_AREA");
                if (env != null && long.TryParse(env, out var e) && e > 0) cap = e;
                cap = (long)(cap * stepScale) * 2 / streams;  // step-scaled; total DiT work ~constant as refs are added
                cap = Math.Min(Math.Max(cap, 256L * 1024), QwenImageNativeArea);
                if (requestedArea > cap)
                {
                    Console.WriteLine($"  [pipe] target area {requestedArea} clamped to {cap} px on Metal " +
                                      "(set --width/--height or TS_QWEN_IMAGE_MAX_AREA to override).");
                    return cap;
                }
                return requestedArea;
            }
            if (_model.Backend != BackendType.GgmlCuda) return requestedArea;

            // Optional hard cap for users who WANT a faster/smaller default than native.
            long target = Math.Min(requestedArea > 0 ? requestedArea : QwenImageNativeArea, QwenImageNativeArea);
            var capEnv = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_MAX_AREA");
            if (capEnv != null && long.TryParse(capEnv, out var capV) && capV > 0)
                target = Math.Min(target, capV);

            long safe = MaxSafeArea(imageCount, doCfg, offload, refArea);
            if (target > safe && offloadAllowed && !offload)
            {
                offload = true;   // stream weights rather than shrink the image
                safe = MaxSafeArea(imageCount, doCfg, offload: true, refArea);
            }
            if (target > safe)
            {
                Console.WriteLine($"  [pipe] target area {target} clamped to {safe} px to fit VRAM " +
                                  "(fewer input images, a smaller TS_QWEN_IMAGE_REF_AREA, or more VRAM raise the ceiling).");
                return safe;
            }
            return target;
        }

        /// <summary>
        /// Reference-latent area for the extra input images (<c>inputs[1..]</c>). Default on
        /// CUDA: the model's native ~1 MP with each ref's own aspect (diffusers Edit Plus
        /// <c>VAE_IMAGE_SIZE</c>) — independent of the output size, because this latent is the
        /// detail source the DiT copies faces/textures from. Returns 0 on the other backends
        /// (= couple to the output area, the legacy rule; no offload path there to absorb the
        /// extra tokens). TS_QWEN_IMAGE_REF_AREA (pixels, clamped to [65536, 4 MP]) overrides
        /// on every backend — values above native ~1 MP keep more input detail but are outside
        /// the training distribution and cost area/256 attention tokens per reference.
        /// </summary>
        internal static long ResolveRefArea(BackendType backend)
        {
            var v = Environment.GetEnvironmentVariable("TS_QWEN_IMAGE_REF_AREA");
            if (v != null && long.TryParse(v, out var a) && a > 0)
                return Math.Clamp(a, 65536, 4L * 1024 * 1024);
            return backend == BackendType.GgmlCuda ? QwenImageNativeArea : 0;
        }

        /// <summary>
        /// The HARD VRAM ceiling for the output area, given the input-image count and whether a
        /// negative-CFG branch runs. Below this the DiT working set fits device memory; ABOVE it
        /// the persistent whole-model capture OOMs and its non-persist fallback spills into garbage
        /// output — so even an explicit --width/--height is capped here (only the softer auto SPEED
        /// default is bypassed, never this). <see cref="long.MaxValue"/> = no known limit (non-CUDA
        /// backend, or the device memory couldn't be queried). Calibrated + margined on the 16 GB
        /// target box (Q2_K DiT 7.1 GB + Lightning LoRA side-path 1.6 GB resident): a single-branch
        /// 2-image edit VERIFIED to fit ~0.6 MP (~7.7k tokens) and to OOM near ~0.9 MP (~10.9k).
        /// With <paramref name="offload"/> the DiT weights stream from RAM per block (nothing
        /// resident but one block's weights + the packed scratch), so nearly the whole budget goes
        /// to activations and the ceiling reaches the model's native ~1 MP on far smaller cards.
        /// <paramref name="refArea"/> is the FIXED per-reference latent area (see
        /// <see cref="ResolveRefArea"/>): each of the <paramref name="imageCount"/> references
        /// costs refArea/256 joint-attention tokens regardless of the output size, so the
        /// ceiling this returns is for the GENERATED stream only.
        /// </summary>
        internal long MaxSafeArea(int imageCount, bool doCfg, bool offload = false, long refArea = QwenImageNativeArea)
        {
            int txtEst = 400 + 250 * (imageCount - 1);
            long refTok = (long)imageCount * (refArea / 256);
            if (_model.Backend == BackendType.GgmlMetal) return QwenImageNativeArea;
            if (_model.Backend != BackendType.GgmlCuda) return long.MaxValue;
            if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out _, out long total) || total <= 0) return long.MaxValue;

            long budget = (long)(total * 0.85);
            const int heads = QwenImageModel.DitNumHeads;   // 24
            // Flash attention (native default) never materializes the [T,T] scores, so its memory
            // is ~LINEAR in the total joint-attention token count; only the explicit-scores path
            // (TS_QWEN_DIT_FLASH=0) needs the tighter quadratic bound.
            bool nativeFlash =
                Environment.GetEnvironmentVariable("TS_QWEN_DIT_NATIVE") != "0" &&
                Environment.GetEnvironmentVariable("TS_QWEN_DIT_FLASH") != "0";
            if (nativeFlash && offload)
            {
                // Weight streaming: resident is one chunk's weight slots + the device-copy
                // allowance (see OffloadResidencyBudget) + runtime overhead — reserve 3 GiB for
                // all of it. Activations use the same ~700 KiB/token envelope as the whole-model
                // graph (chunked graphs pack tighter through the reuse gallocr, so this stays
                // conservative); the CFG branches run as two sequential forwards over the SAME
                // reused scratch, so the bound is branch-count independent.
                const long GiB = 1L << 30;
                long actBudget = Math.Max(GiB, budget - 3L * GiB);
                long memTokens = actBudget / (700L * 1024);
                return Math.Min(Math.Max(256, memTokens - txtEst - refTok) * 256, QwenImageNativeArea);
            }
            if (nativeFlash)
            {
                const long GiB = 1L << 30;
                // Resident set = the DiT weights (actual GGUF size — a Q4_K_M is 13.2 GB where
                // the Q2_K this was first calibrated on is 7.1 GB) + the F32-expanded LoRA
                // factors (~1.6 GB for the Lightning checkpoints).
                long ditBytes = 71L * GiB / 10;
                try { ditBytes = new System.IO.FileInfo(_model.DitGgufPath).Length; } catch { /* keep estimate */ }
                long residentBytes = ditBytes + (QwenImageDiT.LoraPath != null ? 16L * GiB / 10 : 0);
                if (!doCfg)
                {
                    // Single CFG branch (Lightning few-step, or caller cfg<=1): the whole-model
                    // capture holds ONE branch's activations, ~linear in the token count. ~700 KiB
                    // per joint-attention token (measured, margined).
                    long actBudget = Math.Max(GiB, budget - residentBytes);
                    long memTokens = actBudget / (700L * 1024);
                    return Math.Min(Math.Max(256, memTokens - txtEst - refTok) * 256, QwenImageNativeArea);
                }
                // Two CFG branches (or the CFG-batched fused pass) roughly double the live
                // activation set; keep the proven-conservative quadratic-scores bound (this path's
                // high-resolution memory envelope is not empirically calibrated).
                long cfgScores = Math.Max(512L * 1024 * 1024, budget - 8L * GiB);
                double cfgMaxT = Math.Sqrt(cfgScores / (8.0 * heads));
                return Math.Min((long)Math.Max(256, cfgMaxT - txtEst - refTok) * 256, QwenImageNativeArea);
            }
            const long reserve = 12L * 1024 * 1024 * 1024;  // explicit-scores path (16 GB-calibrated)
            long scoresBudget = Math.Max(512L * 1024 * 1024, budget - reserve);
            double maxT = Math.Sqrt(scoresBudget / (8.0 * heads));   // 2*T^2*heads*4 <= scoresBudget
            return (long)Math.Max(256, maxT - txtEst - refTok) * 256;
        }

        /// <summary>
        /// How many bytes of device-resident weight copies to ALLOW during an offloaded denoise:
        /// current free VRAM minus the activation working set for <paramref name="totalTokens"/>
        /// joint-attention tokens (~700 KiB/token, the whole-model-graph envelope) minus fixed
        /// headroom for the streaming slots + runtime. Weights that fit the allowance stay
        /// resident (uploaded once); the rest streams from RAM every forward — the more VRAM,
        /// the less PCIe traffic, degrading gracefully down to full streaming. The floor covers
        /// the two flash-attention masks (cond + neg totals; each a padded SQUARE F16, ~350 MB
        /// at 1 MP) that the chunked offload forward binds resident FIRST — re-uploading those
        /// per chunk would dwarf the weight traffic they replace.
        /// </summary>
        private static long OffloadResidencyBudget(int totalTokens)
        {
            const long MiB = 1L << 20;
            // Two padded square F16 masks (cond/neg branch token totals differ).
            long pad = (totalTokens + 255) / 256 * 256 + 256;
            long maskFloor = 2 * pad * pad * 2 + 64 * MiB;
            if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) || free <= 0)
                return Math.Max(512 * MiB, maskFloor);
            long activations = (long)totalTokens * 700 * 1024;
            long headroom = 1536 * MiB;
            return Math.Max(maskFloor, free - activations - headroom);
        }

        // Release the text + vision encoders (reclaim their CUDA VRAM mid-request); the
        // lazy Te/Vision properties re-create them on demand for the next request.
        private void FreeEncoders()
        {
            _te?.Dispose(); _te = null;
            _vision?.Dispose(); _vision = null;
        }

        public void Dispose()
        {
            _vae?.Dispose();
            _te?.Dispose();
            _dit?.Dispose();
            _vision?.Dispose();
        }
    }

    /// <summary>Seeded Gaussian (Box-Muller) RNG for the initial noise latent.</summary>
    internal sealed class GaussianRng
    {
        private readonly Random _r;
        private double _spare; private bool _hasSpare;
        public GaussianRng(long seed) { _r = new Random((int)(seed ^ (seed >> 32))); }
        public float Next()
        {
            if (_hasSpare) { _hasSpare = false; return (float)_spare; }
            double u, v, s;
            do { u = _r.NextDouble() * 2 - 1; v = _r.NextDouble() * 2 - 1; s = u * u + v * v; }
            while (s >= 1 || s == 0);
            double m = Math.Sqrt(-2.0 * Math.Log(s) / s);
            _spare = v * m; _hasSpare = true;
            return (float)(u * m);
        }
    }
}
