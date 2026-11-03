// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// A tiny LFM3 GGUF laid out the way llama.cpp's converter (and AMQL's to-gguf) writes
// the lfm3 architecture - per-layer head_count_kv (0 on conv layers), the sliding
// window pattern, shortconv.l_cache, the conv kernel as [l_cache, hidden], and a
// rope_freqs tensor that masks the full-attention layers' trailing rotary pairs - and
// a float64 reference forward written from Liquid's modeling_lfm3.py (AMQL's
// scripts/lfm3-quant/lfm3_ref.py). The reference takes the zero-tail rope from the
// partial rotary factor itself, not from rope_freqs, so it checks the GGUF spelling
// as well as the engine.
using System;
using System.Collections.Generic;
using System.Linq;
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

internal sealed class Lfm3SyntheticModelBuilder
{
    public int Hidden { get; init; } = 64;
    public int Heads { get; init; } = 4;
    public int KvHeads { get; init; } = 2;
    public int HeadDim { get; init; } = 16;
    public int Ffn { get; init; } = 128;
    public int Vocab { get; init; } = 256;
    public int Window { get; init; } = 4;
    public int ConvWidth { get; init; } = 4;
    public float PartialRotaryFactor { get; init; } = 0.5f;
    public float RopeTheta { get; init; } = 1_000_000f;
    public float RopeThetaSwa { get; init; } = 10_000f;
    public bool SandwichNorms { get; init; } = true;
    public bool TiedOutput { get; init; } = true;
    /// <summary>Store every 2-D projection as Q8_0 (the reference then reads the
    /// dequantized values, exactly what the engine sees).</summary>
    public bool QuantizeProjections { get; init; }
    public string[] LayerTypes { get; init; } = { "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention" };

    private const float Eps = 1e-5f;
    private Dictionary<string, float[]> _values;

    public string Write(string path)
    {
        var tensors = BuildTensors();
        _values = tensors.ToDictionary(t => t.Name, StoredValues);
        SyntheticGguf.Write(path, BuildMetadata(), tensors);
        return path;
    }

    private List<Tensor> BuildTensors()
    {
        int qDim = Heads * HeadDim, kvDim = KvHeads * HeadDim;
        var t = new List<Tensor>
        {
            Gen("token_embd.weight", 0.5f, Hidden, Vocab),
            GenAround("output_norm.weight", 1f, 0.1f, Hidden),
        };
        if (!TiedOutput)
            t.Add(Gen("output.weight", 0.1f, Hidden, Vocab));
        int rotated = (int)(PartialRotaryFactor * HeadDim) / 2;
        t.Add(new Tensor
        {
            Name = "rope_freqs.weight",
            Dims = new[] { (ulong)(HeadDim / 2) },
            Data = Enumerable.Range(0, HeadDim / 2).Select(i => i < rotated ? 1f : 1e30f).ToArray(),
        });

        for (int l = 0; l < LayerTypes.Length; l++)
        {
            string p = $"blk.{l}.";
            t.Add(GenAround(p + "attn_norm.weight", 1f, 0.1f, Hidden));
            t.Add(GenAround(p + "ffn_norm.weight", 1f, 0.1f, Hidden));
            if (SandwichNorms)
            {
                t.Add(GenAround(p + "post_attention_norm.weight", 1f, 0.1f, Hidden));
                t.Add(GenAround(p + "post_ffw_norm.weight", 1f, 0.1f, Hidden));
            }
            t.Add(Projection(p + "ffn_gate.weight", Hidden, Ffn));
            t.Add(Projection(p + "ffn_up.weight", Hidden, Ffn));
            t.Add(Projection(p + "ffn_down.weight", Ffn, Hidden));
            if (LayerTypes[l] == "conv")
            {
                t.Add(Gen(p + "shortconv.conv.weight", 0.3f, ConvWidth, Hidden));
                t.Add(Projection(p + "shortconv.in_proj.weight", Hidden, 3 * Hidden));
                t.Add(Projection(p + "shortconv.out_proj.weight", Hidden, Hidden));
            }
            else
            {
                t.Add(Projection(p + "attn_q.weight", Hidden, qDim));
                t.Add(Projection(p + "attn_k.weight", Hidden, kvDim));
                t.Add(Projection(p + "attn_v.weight", Hidden, kvDim));
                t.Add(Projection(p + "attn_output.weight", qDim, Hidden));
                t.Add(Projection(p + "attn_gate.weight", Hidden, Heads));
                t.Add(GenAround(p + "attn_q_norm.weight", 1f, 0.1f, HeadDim));
                t.Add(GenAround(p + "attn_k_norm.weight", 1f, 0.1f, HeadDim));
            }
        }
        return t;
    }

    /// <summary>A [in → out] projection (GGUF dims {in, out}).</summary>
    private Tensor Projection(string name, int inDim, int outDim)
    {
        Tensor tensor = Gen(name, 1.0f / MathF.Sqrt(inDim), inDim, outDim);
        if (QuantizeProjections && inDim % 32 == 0)
            tensor.Type = GgmlType.Q8_0;
        return tensor;
    }

    /// <summary>The values the engine reads back for a tensor: Q8_0 blocks decoded.</summary>
    private static float[] StoredValues(Tensor tensor)
    {
        if (tensor.Type != GgmlType.Q8_0)
            return tensor.Data;
        byte[] raw = QuantizeQ8_0(tensor.Data, (int)tensor.Dims[0]);
        var values = new float[tensor.Data.Length];
        for (int block = 0; block < values.Length / 32; block++)
        {
            float d = (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(raw, block * 34));
            for (int i = 0; i < 32; i++)
                values[block * 32 + i] = d * (sbyte)raw[block * 34 + 2 + i];
        }
        return values;
    }

    private List<Kv> BuildMetadata()
    {
        const string a = "lfm3";
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = a },
            new U32 { Key = $"{a}.block_count", V = (uint)LayerTypes.Length },
            new U32 { Key = $"{a}.context_length", V = 512 },
            new U32 { Key = $"{a}.embedding_length", V = (uint)Hidden },
            new U32 { Key = $"{a}.feed_forward_length", V = (uint)Ffn },
            new U32 { Key = $"{a}.attention.head_count", V = (uint)Heads },
            new I32Arr { Key = $"{a}.attention.head_count_kv", V = LayerTypes.Select(t => t == "conv" ? 0 : KvHeads).ToArray() },
            new F32 { Key = $"{a}.rope.freq_base", V = RopeTheta },
            new F32 { Key = $"{a}.rope.freq_base_swa", V = RopeThetaSwa },
            new F32 { Key = $"{a}.attention.layer_norm_rms_epsilon", V = Eps },
            new U32 { Key = $"{a}.attention.key_length", V = (uint)HeadDim },
            new U32 { Key = $"{a}.attention.value_length", V = (uint)HeadDim },
            new U32 { Key = $"{a}.rope.dimension_count", V = (uint)HeadDim },
            new BoolArr { Key = $"{a}.attention.sliding_window_pattern", V = LayerTypes.Select(t => t == "sliding_attention").ToArray() },
            new U32 { Key = $"{a}.attention.sliding_window", V = (uint)Window },
            new U32 { Key = $"{a}.shortconv.l_cache", V = (uint)ConvWidth },
        };
        AddByteTokenizer(kv, Vocab, bos: 1, eos: 0);
        int pre = kv.FindIndex(e => e.Key == "tokenizer.ggml.pre");
        kv[pre] = new Str { Key = "tokenizer.ggml.pre", V = "lfm2" };
        return kv;
    }

    // ------------------------------------------------------------------ reference

    /// <summary>Last-position logits of a fresh forward over <paramref name="ids"/>.</summary>
    public float[] Reference(IReadOnlyList<int> ids) => ReferenceAll(ids)[^1];

    /// <summary>Logits at every position of a fresh forward over <paramref name="ids"/>.</summary>
    public float[][] ReferenceAll(IReadOnlyList<int> ids)
    {
        if (_values == null)
            throw new InvalidOperationException("Write the model first.");
        int T = ids.Count, H = Hidden, hd = HeadDim;
        var x = new double[T][];
        float[] emb = _values["token_embd.weight"];
        for (int t = 0; t < T; t++)
            x[t] = Enumerable.Range(0, H).Select(i => (double)emb[ids[t] * H + i]).ToArray();

        for (int l = 0; l < LayerTypes.Length; l++)
        {
            string p = $"blk.{l}.";
            double[][] h = x.Select(r => Rms(r, _values[p + "attn_norm.weight"])).ToArray();
            double[][] mixed = LayerTypes[l] == "conv" ? Conv(h, p) : Attention(h, p, LayerTypes[l] == "sliding_attention");
            if (SandwichNorms) mixed = mixed.Select(r => Rms(r, _values[p + "post_attention_norm.weight"])).ToArray();
            for (int t = 0; t < T; t++) for (int i = 0; i < H; i++) x[t][i] += mixed[t][i];

            for (int t = 0; t < T; t++)
            {
                double[] f = Rms(x[t], _values[p + "ffn_norm.weight"]);
                double[] g = MatVec(_values[p + "ffn_gate.weight"], f, Ffn);
                double[] u = MatVec(_values[p + "ffn_up.weight"], f, Ffn);
                for (int i = 0; i < Ffn; i++) g[i] = g[i] / (1 + Math.Exp(-g[i])) * u[i];
                double[] down = MatVec(_values[p + "ffn_down.weight"], g, H);
                if (SandwichNorms) down = Rms(down, _values[p + "post_ffw_norm.weight"]);
                for (int i = 0; i < H; i++) x[t][i] += down[i];
            }
        }

        float[] head = _values.TryGetValue("output.weight", out var o) ? o : emb;
        return x.Select(r => MatVec(head, Rms(r, _values["output_norm.weight"]), Vocab).Select(v => (float)v).ToArray()).ToArray();
    }

    private double[][] Conv(double[][] h, string p)
    {
        int T = h.Length, H = Hidden, K = ConvWidth;
        float[] w = _values[p + "shortconv.conv.weight"];
        var bcx = h.Select(r => MatVec(_values[p + "shortconv.in_proj.weight"], r, 3 * H)).ToArray();
        var gated = new double[T][];
        for (int t = 0; t < T; t++)
        {
            gated[t] = new double[H];
            for (int c = 0; c < H; c++)
            {
                // y_t = sum_k w[c, k] · (B·x)_{t-(K-1)+k}, zero before the first token.
                double y = 0;
                for (int k = 0; k < K; k++)
                {
                    int s = t - (K - 1) + k;
                    if (s >= 0) y += w[c * K + k] * bcx[s][c] * bcx[s][2 * H + c];
                }
                gated[t][c] = bcx[t][H + c] * y;
            }
        }
        return gated.Select(r => MatVec(_values[p + "shortconv.out_proj.weight"], r, H)).ToArray();
    }

    private double[][] Attention(double[][] h, string p, bool sliding)
    {
        int T = h.Length, hd = HeadDim, group = Heads / KvHeads;
        // modeling_lfm3: the first int(factor·head_dim)/2 pairs rotate at the full head
        // dim's frequencies, the rest have frequency zero. Sliding layers rotate fully.
        int angles = sliding ? hd / 2 : (int)(PartialRotaryFactor * hd) / 2;
        double theta = sliding ? RopeThetaSwa : RopeTheta;
        var q = new double[T][]; var k = new double[T][]; var v = new double[T][];
        for (int t = 0; t < T; t++)
        {
            q[t] = MatVec(_values[p + "attn_q.weight"], h[t], Heads * hd);
            k[t] = MatVec(_values[p + "attn_k.weight"], h[t], KvHeads * hd);
            v[t] = MatVec(_values[p + "attn_v.weight"], h[t], KvHeads * hd);
            HeadNormAndRope(q[t], Heads, _values[p + "attn_q_norm.weight"], t, angles, theta);
            HeadNormAndRope(k[t], KvHeads, _values[p + "attn_k_norm.weight"], t, angles, theta);
        }

        var outputs = new double[T][];
        for (int t = 0; t < T; t++)
        {
            double[] gate = MatVec(_values[p + "attn_gate.weight"], h[t], Heads);
            var attn = new double[Heads * hd];
            for (int head = 0; head < Heads; head++)
            {
                int kvHead = head / group;
                var scores = new List<(int S, double Score)>();
                for (int s = 0; s <= t; s++)
                {
                    if (sliding && t - s >= Window) continue;
                    double dot = 0;
                    for (int d = 0; d < hd; d++) dot += q[t][head * hd + d] * k[s][kvHead * hd + d];
                    scores.Add((s, dot / Math.Sqrt(hd)));
                }
                double max = scores.Max(e => e.Score);
                double sum = scores.Sum(e => Math.Exp(e.Score - max));
                double g = 1 / (1 + Math.Exp(-gate[head]));
                foreach (var (s, score) in scores)
                {
                    double weight = Math.Exp(score - max) / sum * g;
                    for (int d = 0; d < hd; d++) attn[head * hd + d] += weight * v[s][kvHead * hd + d];
                }
            }
            outputs[t] = MatVec(_values[p + "attn_output.weight"], attn, Hidden);
        }
        return outputs;
    }

    private void HeadNormAndRope(double[] data, int heads, float[] norm, int position, int angles, double theta)
    {
        int hd = HeadDim, half = hd / 2;
        for (int head = 0; head < heads; head++)
        {
            int o = head * hd;
            double ms = 0;
            for (int d = 0; d < hd; d++) ms += data[o + d] * data[o + d];
            double inv = 1 / Math.Sqrt(ms / hd + Eps);
            for (int d = 0; d < hd; d++) data[o + d] = data[o + d] * inv * norm[d];
            for (int j = 0; j < half; j++)
            {
                double freq = j < angles ? Math.Pow(theta, -2.0 * j / hd) : 0;
                double c = Math.Cos(position * freq), s = Math.Sin(position * freq);
                double a = data[o + j], b = data[o + j + half];
                data[o + j] = a * c - b * s;
                data[o + j + half] = a * s + b * c;
            }
        }
    }

    private static double[] Rms(double[] x, float[] w)
    {
        double ms = x.Sum(v => v * v) / x.Length;
        double inv = 1 / Math.Sqrt(ms + Eps);
        return x.Select((v, i) => v * inv * w[i]).ToArray();
    }

    /// <summary>y = W·x for a GGUF {in, out} weight stored row-major [out][in].</summary>
    private static double[] MatVec(float[] w, double[] x, int outDim)
    {
        int inDim = x.Length;
        var y = new double[outDim];
        for (int o = 0; o < outDim; o++)
        {
            double acc = 0;
            for (int i = 0; i < inDim; i++) acc += w[(long)o * inDim + i] * x[i];
            y[o] = acc;
        }
        return y;
    }
}
