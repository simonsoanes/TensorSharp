// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Liquid LFM3 / the LFM3-VL text tower (GGUF <c>lfm3</c>), following llama.cpp's
    /// <c>llama_model_lfm3</c> graph and Liquid's <c>modeling_lfm3.py</c>.
    ///
    /// Three layer kinds, all with an RMSNorm → mixer → residual → RMSNorm → SwiGLU →
    /// residual block, and optional "sandwich" norms on each branch's output:
    /// <list type="bullet">
    /// <item><b>conv</b> (<c>head_count_kv == 0</c>): a gated short convolution. in_proj
    /// gives [B | C | x]; the causal depthwise conv of width <c>shortconv.l_cache</c> runs
    /// over B·x and C gates its output before out_proj. Its recurrent state is the last
    /// <c>l_cache - 1</c> values of B·x per channel.</item>
    /// <item><b>full attention</b>: GQA with per-head Q/K RMSNorm BEFORE NeoX RoPE. The rope
    /// is "zero-tail" partial: the frequency progression spans the whole head, and
    /// <c>rope_freqs.weight</c> divides the trailing pairs by 1e30 so they do not rotate.</item>
    /// <item><b>sliding attention</b>: the same block over the last <c>sliding_window</c>
    /// positions, with its own rope base (<c>rope.freq_base_swa</c>) and no frequency mask.</item>
    /// </list>
    /// Both attention kinds scale each head's output by a sigmoid gate (<c>attn_gate</c>,
    /// one scalar per head) before attn_output.
    ///
    /// The conv state and the attention K/V are what a sequence carries, so the KV-state
    /// snapshot (the server's continuous-batching contract) holds both, the way Nemotron-H
    /// holds its Mamba2 state. Tensor parallelism shards heads, conv channels and the FFN
    /// (see <c>Lfm3Model.TensorParallel.cs</c>).
    /// </summary>
    public sealed partial class Lfm3Model : ModelBase
    {
        private enum LayerKind : byte { Conv, FullAttention, SlidingAttention }

        private LayerKind[] _layerKinds;
        private int _headDim;
        private int _ropeHalf;
        private float _ropeBaseSwa;
        private float[] _ropeFreqsFull;
        private float[] _ropeFreqsSwa;
        private int _slidingWindow;
        private int _convWidth;
        private bool _sandwichNorms;

        // Per conv layer, host copies: the kernel as [hidden, l_cache] (oldest tap first),
        // and the state as [hidden, l_cache - 1], channel-major, oldest value first.
        // Channel-major keeps every rank's channel slice contiguous under TP, so the
        // snapshot's conv section has the same layout sharded or not.
        private float[][] _convWeights;
        private float[][] _convState;

        // Attention layers only; null on conv layers.
        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        private int _kvCacheCapacity;

        public Lfm3Model(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null)
            : base(ggufPath, backend, tpDegree, tpGroup)
        {
            string arch = _gguf.GetString("general.architecture") ?? "lfm3";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();
            ParseLayerLayout(arch);
            ParseTokenizer();

            int convLayers = 0, fullLayers = 0, swaLayers = 0;
            foreach (LayerKind kind in _layerKinds)
            {
                if (kind == LayerKind.Conv) convLayers++;
                else if (kind == LayerKind.FullAttention) fullLayers++;
                else swaLayers++;
            }
            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers} (conv={convLayers}, full={fullLayers}, " +
                $"sliding={swaLayers}), Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, " +
                $"HeadDim={_headDim}, FFN={Config.IntermediateSize}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase} (full), {_ropeBaseSwa} (sliding), rotary dim={2 * _ropeHalf}, " +
                $"window={_slidingWindow}, conv width={_convWidth}");

            if (IsTensorParallel)
                ValidateTpConstraints();

            LoadWeights();
            // Before sharding moves the projections out of _quantWeights, so a TP load
            // picks the same cache dtype (and fingerprint) as a single-device one.
            ApplyModelAlignedKvCacheDefault(_quantWeights);
            ValidateAndPrepareLayers();
            PrecomputeRopeFrequencies();

            if (IsTensorParallel)
            {
                ShardLfm3WeightsForTP();
                PrepareCudaQuantizedWeightsForInferenceTP();
            }
            else
            {
                PrepareCudaQuantizedWeightsForInference();
            }

            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
            {
                Console.WriteLine(
                    $"Initial {_backend} KV cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");
            }

            if (IsTensorParallel)
                InitTpCaches(initialCacheLength, maxContextLength);
            else
                InitCaches(initialCacheLength, maxContextLength);
        }

        // The conv kernel is read on the host by the short-conv loop, so it is loaded as
        // F32 whatever the file stores it as.
        protected override bool IsQuantizedLinearWeight(GgufTensorInfo info)
            => !info.Name.EndsWith(".shortconv.conv.weight", StringComparison.Ordinal)
               && base.IsQuantizedLinearWeight(info);

        /// <summary>The conv state cannot be rewound, so a cache only ever grows or resets.</summary>
        public override bool SupportsKVCacheTruncation => false;

        public override void PrepareForPrefill(int requiredContextTokens)
        {
            if (IsTensorParallel)
                EnsureTpCacheCapacity(requiredContextTokens);
            else
                EnsureCacheCapacity(requiredContextTokens);
        }

        // ------------------------------------------------------------------ layout

        private void ParseLayerLayout(string arch)
        {
            int numLayers = Config.NumLayers;
            uint[] kvPerLayer = _gguf.GetUint32Array($"{arch}.attention.head_count_kv");
            bool[] swaPattern = _gguf.GetBoolArray($"{arch}.attention.sliding_window_pattern");
            if (kvPerLayer != null && kvPerLayer.Length != numLayers)
                throw new InvalidDataException(
                    $"lfm3: {arch}.attention.head_count_kv has {kvPerLayer.Length} entries for {numLayers} layers.");
            if (swaPattern != null && swaPattern.Length != numLayers)
                throw new InvalidDataException(
                    $"lfm3: {arch}.attention.sliding_window_pattern has {swaPattern.Length} entries for {numLayers} layers.");

            _layerKinds = new LayerKind[numLayers];
            int kvHeads = 0;
            for (int l = 0; l < numLayers; l++)
            {
                int layerKv = kvPerLayer != null ? (int)kvPerLayer[l] : Config.NumKVHeads;
                if (layerKv == 0)
                {
                    _layerKinds[l] = LayerKind.Conv;
                    continue;
                }
                if (kvHeads != 0 && layerKv != kvHeads)
                {
                    throw new NotSupportedException(
                        $"lfm3: attention layers with different KV head counts ({kvHeads} and {layerKv}) are not supported.");
                }
                kvHeads = layerKv;
                _layerKinds[l] = swaPattern != null && swaPattern[l] ? LayerKind.SlidingAttention : LayerKind.FullAttention;
            }
            if (kvHeads == 0)
                throw new InvalidDataException("lfm3: no attention layer (every head_count_kv entry is 0).");
            // ParseBaseConfig read the first array entry, which is 0 when layer 0 is a conv layer.
            Config.NumKVHeads = kvHeads;

            _headDim = Config.KeyLength > 0 ? Config.KeyLength : Config.HiddenSize / Config.NumHeads;
            int valueLength = Config.ValueLength > 0 ? Config.ValueLength : _headDim;
            if (valueLength != _headDim)
                throw new NotSupportedException($"lfm3 requires equal key and value head dims, got {_headDim} and {valueLength}.");
            if (Config.NumHeads % kvHeads != 0)
                throw new InvalidDataException($"lfm3: {Config.NumHeads} heads are not a multiple of {kvHeads} KV heads.");

            int rotaryDim = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", (uint)_headDim);
            if (rotaryDim <= 0 || rotaryDim > _headDim || rotaryDim % 2 != 0)
                throw new InvalidDataException($"lfm3: rope.dimension_count {rotaryDim} does not fit head dim {_headDim}.");
            _ropeHalf = rotaryDim / 2;
            _ropeBaseSwa = _gguf.GetFloat32($"{arch}.rope.freq_base_swa", Config.RopeBase);

            _slidingWindow = (int)_gguf.GetUint32($"{arch}.attention.sliding_window", 0);
            _convWidth = (int)_gguf.GetUint32($"{arch}.shortconv.l_cache", 0);
            if (Array.IndexOf(_layerKinds, LayerKind.SlidingAttention) >= 0 && _slidingWindow <= 0)
                throw new InvalidDataException($"lfm3: sliding-window layers but {arch}.attention.sliding_window is missing.");
            if (Array.IndexOf(_layerKinds, LayerKind.Conv) >= 0 && _convWidth < 1)
                throw new InvalidDataException($"lfm3: conv layers but {arch}.shortconv.l_cache is missing.");
            Config.SlidingWindow = _slidingWindow;
        }

        private bool HasWeight(string name)
            => _weights.ContainsKey(name) || _quantWeights.ContainsKey(name);

        private void RequireWeight(int layer, string name)
        {
            if (!HasWeight(name))
                throw new InvalidDataException($"lfm3 layer {layer} ({_layerKinds[layer]}) is missing '{name}'.");
        }

        /// <summary>Check every layer has the tensors its kind needs, decide the sandwich
        /// norms, and copy each conv kernel to the host.</summary>
        private void ValidateAndPrepareLayers()
        {
            int hidden = Config.HiddenSize;
            _convWeights = new float[Config.NumLayers][];
            _sandwichNorms = HasWeight("blk.0.post_attention_norm.weight");
            for (int l = 0; l < Config.NumLayers; l++)
            {
                string p = $"blk.{l}.";
                foreach (string name in new[] { "attn_norm", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down" })
                    RequireWeight(l, p + name + ".weight");
                bool attnPost = HasWeight(p + "post_attention_norm.weight");
                bool ffnPost = HasWeight(p + "post_ffw_norm.weight");
                if (attnPost != _sandwichNorms || ffnPost != _sandwichNorms)
                {
                    throw new InvalidDataException(
                        $"lfm3 layer {l}: the sandwich norms (post_attention_norm / post_ffw_norm) must be on every layer or none.");
                }

                if (_layerKinds[l] == LayerKind.Conv)
                {
                    foreach (string name in new[] { "shortconv.conv", "shortconv.in_proj", "shortconv.out_proj" })
                        RequireWeight(l, p + name + ".weight");
                    Tensor kernel = _weights[p + "shortconv.conv.weight"];
                    if (kernel.ElementCount() != (long)hidden * _convWidth)
                    {
                        throw new InvalidDataException(
                            $"lfm3 layer {l}: shortconv.conv.weight has {kernel.ElementCount()} values, expected {hidden}x{_convWidth}.");
                    }
                    _convWeights[l] = TensorToFloatArray(kernel);
                }
                else
                {
                    foreach (string name in new[] { "attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "attn_q_norm", "attn_k_norm" })
                        RequireWeight(l, p + name + ".weight");
                }
            }
        }

        private void PrecomputeRopeFrequencies()
        {
            int rotaryDim = 2 * _ropeHalf;
            float[] factors = _weights.TryGetValue("rope_freqs.weight", out Tensor factorTensor)
                ? TensorToFloatArray(factorTensor)
                : null;
            _ropeFreqsFull = new float[_ropeHalf];
            _ropeFreqsSwa = new float[_ropeHalf];
            for (int i = 0; i < _ropeHalf; i++)
            {
                // Same float32 arithmetic as the reference's inv_freq.
                float full = 1.0f / MathF.Pow(Config.RopeBase, 2.0f * i / rotaryDim);
                // llama.cpp applies rope_freqs to the full-attention layers only: it is the
                // zero-tail mask (1 for rotating pairs, 1e30 for the rest).
                if (factors != null && i < factors.Length)
                    full /= factors[i];
                _ropeFreqsFull[i] = full;
                _ropeFreqsSwa[i] = 1.0f / MathF.Pow(_ropeBaseSwa, 2.0f * i / rotaryDim);
            }
        }

        // ------------------------------------------------------------------ caches

        private void InitCaches(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            DType kvDtype = _kvCacheDtype.ToDType();
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerKinds[l] == LayerKind.Conv) continue;
                _kvCacheK[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, initialSeqLen, _headDim);
                _kvCacheV[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, initialSeqLen, _headDim);
                InitializeCacheTensor(_kvCacheK[l]);
                InitializeCacheTensor(_kvCacheV[l]);
            }
            AllocateConvState();
            _cacheSeqLen = 0;
        }

        /// <summary>Full-size conv state on every node, TP or not: each rank reads and
        /// writes only its own channels.</summary>
        private void AllocateConvState()
        {
            _convState = new float[Config.NumLayers][];
            int perLayer = Config.HiddenSize * Math.Max(0, _convWidth - 1);
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerKinds[l] == LayerKind.Conv)
                    _convState[l] = new float[perLayer];
            }
        }

        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
            {
                throw new InvalidOperationException(
                    $"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");
            }

            int newCapacity = Math.Max(_kvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvCacheK[l] == null) continue;
                _kvCacheK[l] = GrowCache(_kvCacheK[l], newCapacity);
                _kvCacheV[l] = GrowCache(_kvCacheV[l], newCapacity);
            }

            _kvCacheCapacity = newCapacity;
            Console.WriteLine($"Expanded lfm3 attention cache to {newCapacity} tokens.");
        }

        /// <summary>A cache of <paramref name="newCapacity"/> rows holding <paramref name="old"/>'s
        /// first <see cref="ModelBase._cacheSeqLen"/> rows; disposes <paramref name="old"/>.
        /// The rows move on the host, which is right for every cache dtype.</summary>
        private Tensor GrowCache(Tensor old, int newCapacity)
        {
            var grown = new Tensor(old.Storage.Allocator, old.ElementType, old.Sizes[0], newCapacity, old.Sizes[2]);
            InitializeCacheTensor(grown);
            if (_cacheSeqLen > 0)
            {
                int rowBytes = checked((int)CacheRowBytes(old));
                var rows = new byte[(long)old.Sizes[0] * _cacheSeqLen * rowBytes];
                if (!CopyCacheRowsOut(old, 0, _cacheSeqLen, rows, out _) || !CopyCacheRowsIn(grown, 0, _cacheSeqLen, rows, out _))
                    throw new InvalidOperationException("lfm3: could not move the attention cache into its larger buffer.");
                InvalidateTensorDeviceCache(grown);
            }
            old.Dispose();
            return grown;
        }

        protected override void ResetKVCacheCore()
        {
            _cacheSeqLen = 0;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
            if (_convState != null)
            {
                foreach (float[] state in _convState)
                {
                    if (state != null) Array.Clear(state);
                }
            }
            // Positions past _cacheSeqLen are never read, so the K/V rows need no clearing;
            // ResetCacheTensor also drops their device copies.
            if (_kvCacheK != null)
            {
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (_kvCacheK[l] == null) continue;
                    ResetCacheTensor(_kvCacheK[l]);
                    ResetCacheTensor(_kvCacheV[l]);
                }
            }
            ResetTpCaches();
        }

        /// <summary>Only an exact no-op or a full reset is reachable: the conv state cannot rewind.</summary>
        protected override bool TryTruncateKVCacheCore(int tokenCount)
        {
            if (tokenCount == _cacheSeqLen)
                return true;
            if (tokenCount != 0)
                return false;
            ResetKVCacheCore();
            return true;
        }

        protected override void TruncateKVCacheCore(int tokenCount)
        {
            if (!TryTruncateKVCacheCore(tokenCount))
            {
                throw new InvalidOperationException(
                    $"lfm3 cannot rewind its conv state from {_cacheSeqLen} to {tokenCount} tokens; reset and re-prefill instead.");
            }
        }

        // ------------------------------------------------------------------ forward

        protected override float[] ForwardCore(int[] tokens)
        {
            if (IsTensorParallel)
                return ForwardTP(tokens);

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            for (int layer = 0; layer < Config.NumLayers; layer++)
                DecoderLayer(hidden, layer, seqLen, startPos);

            ComputeLogits(hidden, seqLen);
            hidden.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        /// <summary>output_norm, the last row, the LM head (tied to the embedding when the
        /// file has no output.weight), into <see cref="ModelBase._logitsBuffer"/>.</summary>
        private void ComputeLogits(Tensor hidden, int seqLen)
        {
            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            Tensor lastHidden;
            if (seqLen > 1)
            {
                using Tensor narrowed = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            long t2 = Stopwatch.GetTimestamp();
            Tensor logitsTensor = LinearForward(lastHidden, "output.weight")
                ?? LinearForward(lastHidden, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHidden.Dispose();

            long t3 = Stopwatch.GetTimestamp();
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;
            logitsTensor.Dispose();
        }

        /// <summary>One decoder layer, in place on <paramref name="hidden"/>.</summary>
        private void DecoderLayer(Tensor hidden, int layer, int seqLen, int startPos)
        {
            string p = $"blk.{layer}.";
            Tensor normed = RMSNormOp(hidden, p + "attn_norm.weight");
            Tensor mixed = _layerKinds[layer] == LayerKind.Conv
                ? ShortConvBlock(normed, layer, seqLen)
                : AttentionBlock(normed, layer, seqLen, startPos);
            normed.Dispose();
            AddBranch(hidden, mixed, _sandwichNorms ? p + "post_attention_norm.weight" : null);

            Tensor ffnNormed = RMSNormOp(hidden, p + "ffn_norm.weight");
            Tensor gate = LinearForward(ffnNormed, p + "ffn_gate.weight");
            Tensor up = LinearForward(ffnNormed, p + "ffn_up.weight");
            ffnNormed.Dispose();
            Ops.SiLUMul(gate, gate, up);
            up.Dispose();
            Tensor down = LinearForward(gate, p + "ffn_down.weight");
            gate.Dispose();
            AddBranch(hidden, down, _sandwichNorms ? p + "post_ffw_norm.weight" : null);
        }

        /// <summary>hidden += norm(branch) (or the bare branch without a post norm); disposes branch.</summary>
        private void AddBranch(Tensor hidden, Tensor branch, string postNormName)
        {
            if (postNormName != null)
            {
                Tensor normed = RMSNormOp(branch, postNormName);
                branch.Dispose();
                branch = normed;
            }
            Ops.Add(hidden, hidden, branch);
            branch.Dispose();
        }

        private Tensor ShortConvBlock(Tensor input, int layer, int seqLen)
        {
            string p = $"blk.{layer}.";
            long t0 = Stopwatch.GetTimestamp();
            Tensor bcx = LinearForward(input, p + "shortconv.in_proj.weight");
            var gated = new Tensor(bcx.Storage.Allocator, DType.Float32, seqLen, Config.HiddenSize);
            RunShortConv(bcx, gated, layer, 0, Config.HiddenSize, seqLen);
            bcx.Dispose();
            _attnTicks += Stopwatch.GetTimestamp() - t0;

            Tensor output = LinearForward(gated, p + "shortconv.out_proj.weight");
            gated.Dispose();
            return output;
        }

        /// <summary>
        /// The gated short conv over <paramref name="channels"/> channels starting at global
        /// channel <paramref name="channelOffset"/>: <paramref name="bcx"/> is [seqLen, 3·channels]
        /// laid out [B | C | x] (a rank's slice of each segment under TP), and
        /// <paramref name="output"/> receives C·conv(B·x) as [seqLen, channels]. Advances
        /// the layer's conv state for those channels.
        /// </summary>
        private unsafe void RunShortConv(Tensor bcx, Tensor output, int layer, int channelOffset, int channels, int seqLen)
        {
            int width = _convWidth;
            int history = width - 1;
            int rowStride = 3 * channels;
            float* src = GetFloatPtr(bcx);
            float* dst = GetFloatPtr(output);
            nint srcAddress = (nint)src, dstAddress = (nint)dst;
            float[] kernel = _convWeights[layer];
            float[] state = _convState[layer];

            void Channels(int first, int end)
            {
                float* s = (float*)srcAddress;
                float* d = (float*)dstAddress;
                fixed (float* kernelBase = kernel, stateBase = state)
                {
                    for (int c = first; c < end; c++)
                    {
                        int global = channelOffset + c;
                        float* taps = kernelBase + (long)global * width;
                        float* window = stateBase + (long)global * history;
                        for (int t = 0; t < seqLen; t++)
                        {
                            float* row = s + (long)t * rowStride;
                            float bx = row[c] * row[2 * channels + c];
                            float acc = taps[history] * bx;
                            for (int j = 0; j < history; j++)
                                acc += taps[j] * window[j];
                            for (int j = 0; j + 1 < history; j++)
                                window[j] = window[j + 1];
                            if (history > 0)
                                window[history - 1] = bx;
                            d[(long)t * channels + c] = row[channels + c] * acc;
                        }
                    }
                }
            }

            const int channelsPerTask = 64;
            if ((long)channels * seqLen >= 65536)
            {
                int tasks = (channels + channelsPerTask - 1) / channelsPerTask;
                Parallel.For(0, tasks, task =>
                {
                    int first = task * channelsPerTask;
                    Channels(first, Math.Min(channels, first + channelsPerTask));
                });
            }
            else
            {
                Channels(0, channels);
            }
            InvalidateTensorDeviceCache(output);
        }

        private Tensor AttentionBlock(Tensor input, int layer, int seqLen, int startPos)
        {
            string p = $"blk.{layer}.";
            Tensor q = LinearForward(input, p + "attn_q.weight");
            Tensor k = LinearForward(input, p + "attn_k.weight");
            Tensor v = LinearForward(input, p + "attn_v.weight");
            Tensor gate = LinearForward(input, p + "attn_gate.weight");

            Tensor attn = GatedAttention(q, k, v, gate, _kvCacheK[layer], _kvCacheV[layer], layer, 0,
                Config.NumHeads, Config.NumKVHeads, seqLen, startPos);

            Tensor output = LinearForward(attn, p + "attn_output.weight");
            attn.Dispose();
            return output;
        }

        /// <summary>
        /// Q/K RMSNorm, RoPE, the cache append, attention over the causal (and, on sliding
        /// layers, windowed) keys, and the per-head sigmoid gate, for
        /// <paramref name="numHeads"/> heads whose K/V live in <paramref name="kCache"/> /
        /// <paramref name="vCache"/>. Consumes q, k, v and gate; returns [seqLen, numHeads·headDim].
        /// <paramref name="rank"/> selects the replicated norm weights' copy under TP.
        /// </summary>
        private Tensor GatedAttention(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor kCache, Tensor vCache,
            int layer, int rank, int numHeads, int numKVHeads, int seqLen, int startPos)
        {
            string p = $"blk.{layer}.";
            bool sliding = _layerKinds[layer] == LayerKind.SlidingAttention;
            int headDim = _headDim;
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);
            IAllocator alloc = q.Storage.Allocator;

            // Q/K norm BEFORE the rotation (llama.cpp's lfm3 graph and modeling_lfm3).
            ApplyHeadRmsNorm(q, TpReplicatedWeight(_weights[p + "attn_q_norm.weight"], rank), numHeads, seqLen);
            ApplyHeadRmsNorm(k, TpReplicatedWeight(_weights[p + "attn_k_norm.weight"], rank), numKVHeads, seqLen);
            float[] freqs = sliding ? _ropeFreqsSwa : _ropeFreqsFull;
            ApplyRope(q, numHeads, seqLen, startPos, freqs);
            ApplyRope(k, numKVHeads, seqLen, startPos, freqs);

            long t0 = Stopwatch.GetTimestamp();
            if (seqLen == 1)
            {
                CopyToCacheDecode(kCache, k, vCache, v, numKVHeads, headDim, startPos);
            }
            else
            {
                using Tensor kHeads = ReshapeToHeads(k, numKVHeads, seqLen, headDim);
                using Tensor vHeads = ReshapeToHeads(v, numKVHeads, seqLen, headDim);
                CopyToCache(kCache, kHeads, startPos, seqLen);
                CopyToCache(vCache, vHeads, startPos, seqLen);
            }
            k.Dispose();
            v.Dispose();

            // A sliding layer never looks further back than its window, so its keys start
            // at the first position the oldest query can see.
            int kvStart = sliding ? Math.Max(0, startPos - _slidingWindow + 1) : 0;
            Tensor result;
            if (seqLen == 1 && kvStart == 0)
            {
                result = new Tensor(alloc, DType.Float32, 1, numHeads * headDim);
                AttentionDecodePureCS(q, kCache, vCache, result, numHeads, numKVHeads, headDim, totalSeqLen, scale);
                q.Dispose();
            }
            else
            {
                int kvLen = totalSeqLen - kvStart;
                int groupSize = numHeads / numKVHeads;
                Tensor qHeads = ReshapeToHeads(q, numHeads, seqLen, headDim);
                q.Dispose();
                Tensor kExpanded = ExpandKVHeadsRange(kCache, groupSize, kvStart, kvLen);
                Tensor vExpanded = ExpandKVHeadsRange(vCache, groupSize, kvStart, kvLen);

                var scores = new Tensor(alloc, DType.Float32, numHeads, seqLen, kvLen);
                using (Tensor kT = kExpanded.Transpose(1, 2))
                    Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
                qHeads.Dispose();
                kExpanded.Dispose();

                MaskedSoftmax(scores, numHeads, seqLen, kvLen, startPos - kvStart, sliding ? _slidingWindow : 0);

                var heads = new Tensor(alloc, DType.Float32, numHeads, seqLen, headDim);
                Ops.AddmmBatch(heads, 0, heads, 1.0f, scores, vExpanded);
                scores.Dispose();
                vExpanded.Dispose();

                result = ReshapeFromHeads(heads, numHeads, seqLen, headDim);
                heads.Dispose();
            }
            _attnTicks += Stopwatch.GetTimestamp() - t0;

            ApplyHeadGate(result, gate, numHeads, seqLen);
            gate.Dispose();
            return result;
        }

        /// <summary>
        /// Causal (+ sliding-window) softmax over [heads, seqLen, kvLen] scores whose key
        /// 0 is <paramref name="queryOffset"/> positions before query 0: query i sees key j
        /// iff j &lt;= queryOffset + i and, with a window, queryOffset + i - j &lt; window.
        /// </summary>
        private unsafe void MaskedSoftmax(Tensor scores, int numHeads, int seqLen, int kvLen, int queryOffset, int window)
        {
            if (IsGgmlBackend)
            {
                // Same rule natively: key k is kept iff q_pos - window < k <= q_pos.
                GgmlBasicOps.AttentionSoftmaxWithSinks(scores, sinks: null,
                    numHeads: numHeads, seqLen: seqLen, kvLen: kvLen,
                    maskStartPos: queryOffset, slidingWindow: window, scale: 1.0f);
                return;
            }

            float* s = GetFloatPtr(scores);
            for (int h = 0; h < numHeads; h++)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    float* row = s + ((long)h * seqLen + i) * kvLen;
                    int queryPos = queryOffset + i;
                    int first = window > 0 ? Math.Max(0, queryPos - window + 1) : 0;
                    for (int j = 0; j < first; j++)
                        row[j] = float.NegativeInfinity;
                    for (int j = queryPos + 1; j < kvLen; j++)
                        row[j] = float.NegativeInfinity;
                }
            }
            Ops.Softmax(scores, scores);
        }

        /// <summary>Per-head RMSNorm of a [seqLen, heads·headDim] projection, in place.</summary>
        private void ApplyHeadRmsNorm(Tensor data, Tensor alpha, int numHeads, int seqLen)
        {
            using Tensor rows = data.View(seqLen * numHeads, _headDim);
            Ops.RMSNorm(rows, rows, alpha, null, Config.Eps);
        }

        // One cos/sin table per (frequency set, positions); the attention layers of a
        // forward share them.
        private readonly object _ropeTableGate = new();
        private float[] _ropeTableFreqs;
        private int _ropeTableStart = -1;
        private int _ropeTableLength;
        private float[] _ropeCos;
        private float[] _ropeSin;

        /// <summary>NeoX rotation of the first rotary pairs of every head, in place.
        /// Pair j is (j, j + rotaryDim/2), at angle position · freqs[j].</summary>
        private unsafe void ApplyRope(Tensor data, int numHeads, int seqLen, int startPos, float[] freqs)
        {
            int half = _ropeHalf;
            lock (_ropeTableGate)
            {
                if (!ReferenceEquals(_ropeTableFreqs, freqs) || _ropeTableStart != startPos || _ropeTableLength != seqLen)
                {
                    var cos = new float[seqLen * half];
                    var sin = new float[seqLen * half];
                    float positionScale = Config.RopeScale > 0f ? 1.0f / Config.RopeScale : 1.0f;
                    for (int s = 0; s < seqLen; s++)
                    {
                        float position = (startPos + s) * positionScale;
                        for (int j = 0; j < half; j++)
                        {
                            float angle = position * freqs[j];
                            cos[s * half + j] = MathF.Cos(angle);
                            sin[s * half + j] = MathF.Sin(angle);
                        }
                    }
                    _ropeCos = cos;
                    _ropeSin = sin;
                    _ropeTableFreqs = freqs;
                    _ropeTableStart = startPos;
                    _ropeTableLength = seqLen;
                }
                NeoXRopeCpu.Apply(GetFloatPtr(data), seqLen, numHeads, _headDim, _ropeCos, _ropeSin, half);
            }
            InvalidateTensorDeviceCache(data);
        }

        /// <summary>attn[t, h, :] *= sigmoid(gate[t, h]).</summary>
        private unsafe void ApplyHeadGate(Tensor attn, Tensor gate, int numHeads, int seqLen)
        {
            float* a = GetFloatPtr(attn);
            float* g = GetFloatPtr(gate);
            int headDim = _headDim;
            for (int t = 0; t < seqLen; t++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    float sigmoid = 1.0f / (1.0f + MathF.Exp(-g[(long)t * numHeads + h]));
                    float* head = a + ((long)t * numHeads + h) * headDim;
                    for (int d = 0; d < headDim; d++)
                        head[d] *= sigmoid;
                }
            }
            InvalidateTensorDeviceCache(attn);
        }

        /// <summary>
        /// Rows [<paramref name="start"/>, start + <paramref name="length"/>) of a K or V cache
        /// as a contiguous F32 [kvHeads·groupSize, length, headDim] tensor, GQA-broadcast.
        /// </summary>
        private unsafe Tensor ExpandKVHeadsRange(Tensor cache, int groupSize, int start, int length)
        {
            // ExpandKVHeads allocates on the primary device; a TP rank's cache stays on its own.
            if (start == 0 && ReferenceEquals(cache.Storage.Allocator, _allocator))
                return ExpandKVHeads(cache, groupSize, length);

            if (cache.ElementType == DType.Float32)
            {
                using Tensor active = cache.Narrow(1, start, length);
                return groupSize == 1 ? Ops.NewContiguous(active) : Ops.RepeatInterleave(null, active, groupSize, 0);
            }

            int numKVHeads = (int)cache.Sizes[0];
            int capacity = (int)cache.Sizes[1];
            int headDim = (int)cache.Sizes[2];
            long rowBytes = CacheRowBytes(cache);
            cache.Storage.EnsureHostReadable();
            var f32 = new Tensor(cache.Storage.Allocator, DType.Float32, numKVHeads * groupSize, length, headDim);
            float* dstBase = GetFloatPtr(f32);
            byte* srcBase = (byte*)TensorComputePrimitives.GetStoragePointer(cache);
            for (int h = 0; h < numKVHeads; h++)
            {
                byte* srcRows = srcBase + ((long)h * capacity + start) * rowBytes;
                float* first = dstBase + (long)h * groupSize * length * headDim;
                if (cache.ElementType == DType.Float16)
                    TensorComputePrimitives.F16ToF32(first, (ushort*)srcRows, length * headDim);
                else if (IsBlockQuantCacheDType(cache.ElementType))
                    ManagedQuantizedOps.DequantizeRowToFloat32(GgmlTypeForCacheDType(cache.ElementType),
                        (IntPtr)srcRows, first, (long)length * headDim);
                else
                    throw new NotSupportedException($"lfm3: no windowed read for a {cache.ElementType} KV cache.");
                for (int g = 1; g < groupSize; g++)
                {
                    Buffer.MemoryCopy(first, first + (long)g * length * headDim,
                        (long)length * headDim * sizeof(float), (long)length * headDim * sizeof(float));
                }
            }
            InvalidateTensorDeviceCache(f32);
            return f32;
        }

        // ------------------------------------------------------------------ host cache rows

        /// <summary>Bytes of one (head, position) row of a contiguous cache, for any dtype.</summary>
        private static long CacheRowBytes(Tensor cache)
            => cache.Storage.ByteLength / (cache.Sizes[0] * cache.Sizes[1]);

        /// <summary>Bytes a [heads, <paramref name="tokenCount"/>] section of <paramref name="cache"/> takes.</summary>
        private static long CacheSectionBytes(Tensor cache, int tokenCount)
            => cache.Sizes[0] * tokenCount * CacheRowBytes(cache);

        /// <summary>Rows [start, start+count) of every head, head-major, into <paramref name="destination"/>.</summary>
        private static unsafe bool CopyCacheRowsOut(Tensor cache, int start, int count, Span<byte> destination, out int written)
        {
            written = 0;
            long heads = cache.Sizes[0], capacity = cache.Sizes[1];
            long rowBytes = CacheRowBytes(cache);
            long perHead = count * rowBytes;
            if (start < 0 || start + count > capacity || destination.Length < heads * perHead)
                return false;
            cache.Storage.EnsureHostReadable();
            byte* src = (byte*)cache.Storage.PtrAtElement(0);
            fixed (byte* dst = destination)
            {
                for (long h = 0; h < heads; h++)
                    Buffer.MemoryCopy(src + (h * capacity + start) * rowBytes, dst + h * perHead, perHead, perHead);
            }
            written = checked((int)(heads * perHead));
            return true;
        }

        /// <summary>The inverse of <see cref="CopyCacheRowsOut"/>; the caller invalidates device copies.</summary>
        private static unsafe bool CopyCacheRowsIn(Tensor cache, int start, int count, ReadOnlySpan<byte> source, out int read)
        {
            read = 0;
            long heads = cache.Sizes[0], capacity = cache.Sizes[1];
            long rowBytes = CacheRowBytes(cache);
            long perHead = count * rowBytes;
            if (start < 0 || start + count > capacity || source.Length < heads * perHead)
                return false;
            cache.Storage.EnsureHostReadable();
            byte* dst = (byte*)cache.Storage.PtrAtElement(0);
            fixed (byte* src = source)
            {
                for (long h = 0; h < heads; h++)
                    Buffer.MemoryCopy(src + h * perHead, dst + (h * capacity + start) * rowBytes, perHead, perHead);
            }
            read = checked((int)(heads * perHead));
            return true;
        }

        public override void Dispose()
        {
            if (_kvCacheK != null)
            {
                foreach (Tensor t in _kvCacheK) t?.Dispose();
                foreach (Tensor t in _kvCacheV) t?.Dispose();
            }
            DisposeTpCaches();
            base.Dispose();
        }
    }
}
