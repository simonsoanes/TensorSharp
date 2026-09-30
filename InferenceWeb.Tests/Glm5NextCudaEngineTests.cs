// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using System.Linq;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// GLM-5.3-Flash on the direct-CUDA engine (<c>--backend cuda</c>), held to the native ggml executor
/// that <c>--backend ggml_cuda</c> runs, on a four-layer fixture with the production model's kernel
/// widths (<see cref="GlmDsaSyntheticModelBuilder.WriteGlm5NextCudaFixture"/>): KDA and MLA layers,
/// a dense and three MoE FFNs, hyper-connections, a prefill long enough for the grouped expert
/// kernels, then single-token decode.
/// </summary>
public sealed class Glm5NextCudaEngineTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-glm5next-cuda-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new EnvScope();

    public Glm5NextCudaEngineTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        _env.Set("MAX_CONTEXT", "512");
        _env.Set("TS_GLM_UBATCH", "64");
        _env.Set("TS_GLM_NATIVE", null);
        _env.Set("TS_GLM_NGPU", null);
        _env.ClearSpeculationVars();
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Fixture(int indexerTopK = 2048) => GlmDsaSyntheticModelBuilder.WriteGlm5NextCudaFixture(
        Path.Combine(_dir, $"glm5next-cuda-{indexerTopK}.gguf"), indexerTopK: indexerTopK);

    private static readonly int[] Prompt = Enumerable.Range(0, 40).Select(i => (i * 37 + 11) % 256).ToArray();
    private static readonly int[] Steps = { 17, 203, 99, 4, 150, 61 };

    /// <summary>The prompt's logits, then each decode step's.</summary>
    private static float[][] Run(ModelBase model)
    {
        var rows = new float[Steps.Length + 1][];
        rows[0] = (float[])model.ForwardRefill(Prompt).Clone();
        for (int i = 0; i < Steps.Length; i++)
            rows[i + 1] = (float[])model.Forward(new[] { Steps[i] }).Clone();
        return rows;
    }

    /// <summary>Dense attention: every cell sits below the indexer's top_k. The pooled sparse
    /// selection past it is held to the indexer's rules instead
    /// (<see cref="ThePooledSparseSelection_FollowsTheIndexerRules"/>): two correct engines that
    /// differ by rounding swap near-tied pools, so an end-to-end comparison cannot pin it down.</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void TheDirectCudaEngine_AnswersAsTheNativeExecutor()
    {
        string path = Fixture();
        float[][] native, cuda;
        using (ModelBase model = ModelBase.Create(path, BackendType.GgmlCuda))
            native = Run(model);
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
            cuda = Run(model);

        double worst = 0;
        for (int i = 0; i < native.Length; i++)
        {
            double err = RelativeError(cuda[i], native[i]);
            worst = Math.Max(worst, err);
            _output.WriteLine($"{(i == 0 ? "prefill" : $"decode {i}")}: relative error {err:E3}");
            Assert.Equal(ArgMax(native[i]), ArgMax(cuda[i]));
        }
        // Different kernels (F16 per-head MLA projections here, q8_1 there), same function.
        Assert.True(worst <= 2e-2, $"the direct-CUDA engine strayed {worst:E3} from the native executor");
    }

    /// <summary>Two devices answer as one: the hidden streams cross the boundary through pinned
    /// memory, and each layer keeps its caches and recurrent state on its own device.</summary>
    [MultiCudaFact(2)]
    public void ALayerSplit_AnswersAsOneDevice()
    {
        string path = Fixture();
        float[][] single, split;
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
            single = Run(model);
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda, layerSplitDegree: 2))
            split = Run(model);
        for (int i = 0; i < single.Length; i++)
            Assert.True(RelativeError(split[i], single[i]) <= 1e-5, $"forward {i} differs across the split");
    }

    /// <summary>A reset clears the recurrent state as well as the attention cache.</summary>
    [CudaFact]
    public void AReset_StartsAFreshSequence()
    {
        using ModelBase model = ModelBase.Create(Fixture(), BackendType.Cuda);
        float[][] first = Run(model);
        model.ResetKVCache();
        float[][] again = Run(model);
        for (int i = 0; i < first.Length; i++)
            Assert.True(RelativeError(again[i], first[i]) <= 1e-6, $"forward {i} differs after a reset");
    }

    // ---------------------------------------------------------------- pooled sparse selection

    /// <summary>
    /// Every sparse MLA step (top_k 16: each query keeps its best four pools of four cells from the
    /// first ubatch on) checked against the rules of the native executor's pooled indexer, computed
    /// here from the engine's own inputs:
    /// <list type="bullet">
    /// <item>a pool's key is its members' keys weighted by a per-channel softmax over their cached
    /// gates plus the slot embedding;</item>
    /// <item>the head weights are the layer input through indexer.proj over sqrt(D * H);</item>
    /// <item>a query scores only the pools whose last member it has seen, as the head-weighted sum
    /// of ReLU'd dot products;</item>
    /// <item>it keeps the best top_k / kpool of them, ties to the lower index, plus the cells of its
    /// own unfinished pool, and attends over exactly those cells.</item>
    /// </list>
    /// The end-to-end comparison with the native executor matched the selection at every step of
    /// this fixture but one, where the fourth and fifth pools' scores were 0.14% apart and the two
    /// engines' rounding ordered them differently.
    /// </summary>
    [CudaFact]
    public void ThePooledSparseSelection_FollowsTheIndexerRules()
    {
        string path = Fixture(indexerTopK: 16);
        var probes = new System.Collections.Generic.List<TensorSharp.Cuda.GlmCudaEngine.SparseLayerProbe>();
        using (var ex = OpenExecutor(path))
        {
            ex.Engine.SparseProbe = probes.Add;
            Step(ex, Prompt);
            foreach (int token in Steps)
                Step(ex, new[] { token });
        }
        // Two MLA layers, each sparse in the prompt and in every step.
        Assert.Equal(2 * (1 + Steps.Length), probes.Count);
        using var gguf = new GgufFile(path);
        foreach (var probe in probes)
            CheckSparseLayer(probe, gguf);
    }

    private static void CheckSparseLayer(TensorSharp.Cuda.GlmCudaEngine.SparseLayerProbe p, GgufFile gguf)
    {
        int d = p.IdxDim, h = p.IdxHeads, kp = p.Kpool, lat = p.Latent;
        string where = $"layer {p.Layer} at {p.P0}";
        float[] ape = F32(gguf, $"blk.{p.Layer}.indexer_compressor_ape.weight");   // [kpool][D]
        float[] proj = F32(gguf, $"blk.{p.Layer}.indexer.proj.weight");            // [H][NEmbd]

        // Pool keys: a per-channel softmax over the members' gates (plus the slot embedding)
        // weighting their keys.
        int pools = (p.P0 + p.Nt) / kp;
        var poolKeys = new float[pools * d];
        for (int b = 0; b < pools; b++)
            for (int c = 0; c < d; c++)
            {
                double max = double.NegativeInfinity;
                var logit = new double[kp];
                for (int j = 0; j < kp; j++)
                {
                    logit[j] = (float)p.IdxCache[(long)(b * kp + j) * 2 * d + d + c] + ape[j * d + c];
                    max = Math.Max(max, logit[j]);
                }
                double sum = 0, acc = 0;
                for (int j = 0; j < kp; j++)
                {
                    double e = Math.Exp(logit[j] - max);
                    sum += e;
                    acc += e * (float)p.IdxCache[(long)(b * kp + j) * 2 * d + c];
                }
                poolKeys[b * d + c] = (float)(acc / sum);
            }
        AssertClose(poolKeys, p.PoolKeys, 1e-5, $"pool keys, {where}");

        // Head weights, with both scale constants folded in.
        var w = new float[p.Nt * h];
        double wScale = 1.0 / Math.Sqrt((double)d * h);
        for (int t = 0; t < p.Nt; t++)
            for (int k = 0; k < h; k++)
            {
                double acc = 0;
                for (int i = 0; i < p.NEmbd; i++)
                    acc += (double)p.Cur[t * p.NEmbd + i] * proj[k * p.NEmbd + i];
                w[t * h + k] = (float)(acc * wScale);
            }
        AssertClose(w, p.IdxW, 1e-4, $"head weights, {where}");

        for (int t = 0; t < p.Nt; t++)
        {
            long q = (long)p.P0 + t;
            int nVis = (int)((q + 1) / kp);

            // Scores of the pools whose last member the query has seen: the ReLU sits between
            // each head's dot product and its weight.
            var want = new float[nVis];
            var got = new float[nVis];
            for (int b = 0; b < nVis; b++)
            {
                double score = 0;
                for (int k = 0; k < h; k++)
                {
                    double dot = 0;
                    for (int c = 0; c < d; c++)
                        dot += (double)p.IdxQ[((long)t * h + k) * d + c] * p.PoolKeys[b * d + c];
                    if (dot > 0)
                        score += dot * p.IdxW[t * h + k];
                }
                want[b] = (float)score;
                got[b] = p.Scores[(long)t * p.PoolStride + b];
            }
            AssertClose(want, got, 1e-4, $"pool scores, {where}, query {q}");

            // The best top_k / kpool of the ENGINE's scores (so the check is exact), in the
            // kernel's float order, ties to the lower pool index.
            int keep = Math.Min(nVis, p.SelectPools);
            int[] best = Enumerable.Range(0, nVis).OrderByDescending(b => OrderKey(got[b])).ThenBy(b => b).Take(keep).ToArray();
            Assert.Equal(keep, p.SelCnt[t]);
            int[] sel = p.Sel.Skip(t * p.SelectPools).Take(keep).ToArray();
            Assert.Equal(best.OrderBy(b => b), sel.OrderBy(b => b));

            // The selected pools' cells, then the query's own unfinished pool up to itself.
            int tailStart = (int)((q + 1) / kp * kp);
            int[] cells = best.SelectMany(b => Enumerable.Range(b * kp, kp))
                .Concat(Enumerable.Range(tailStart, (int)(q + 1 - tailStart))).OrderBy(c => c).ToArray();
            Assert.Equal(cells.Length, p.CellCnt[t]);
            Assert.Equal(cells, p.Cells.Skip(t * p.CellStride).Take(cells.Length).OrderBy(c => c));

            // Attention over exactly those cells of the latent cache (no sink weight).
            for (int k = 0; k < p.Heads; k++)
            {
                long qo = ((long)t * p.Heads + k) * lat;
                var logits = new double[cells.Length];
                double max = double.NegativeInfinity;
                for (int i = 0; i < cells.Length; i++)
                {
                    double dot = 0;
                    for (int l = 0; l < lat; l++)
                        dot += (double)p.QAbs[qo + l] * (float)p.KvCache[(long)cells[i] * lat + l];
                    logits[i] = dot * p.KqScale;
                    max = Math.Max(max, logits[i]);
                }
                double sum = logits.Sum(x => Math.Exp(x - max));
                var o = new float[lat];
                for (int l = 0; l < lat; l++)
                {
                    double acc = 0;
                    for (int i = 0; i < cells.Length; i++)
                        acc += Math.Exp(logits[i] - max) / sum * (float)p.KvCache[(long)cells[i] * lat + l];
                    o[l] = (float)acc;
                }
                AssertClose(o, p.AttnO.AsSpan((int)qo, lat).ToArray(), 1e-4, $"attention, {where}, query {q}, head {k}");
            }
        }
    }

    /// <summary>The top-k kernel's total order on floats (the IEEE bits made unsigned-comparable).</summary>
    private static uint OrderKey(float v)
    {
        uint k = BitConverter.SingleToUInt32Bits(v);
        return (k & 0x80000000u) != 0 ? ~k : k | 0x80000000u;
    }

    private static float[] F32(GgufFile gguf, string name)
    {
        byte[] raw = gguf.ReadTensorData(gguf.Tensors[name]);
        var f = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, f, 0, raw.Length);
        return f;
    }

    /// <summary>Worst absolute difference within <paramref name="tol"/> times the expected
    /// values' largest magnitude (at least 1).</summary>
    private static void AssertClose(float[] expected, float[] actual, double tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1.0, expected.Length == 0 ? 0 : expected.Max(v => Math.Abs((double)v)));
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
            worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]) / scale);
        Assert.True(worst <= tol, $"{what}: off by {worst:E3} (tolerance {tol:E1})");
    }

    // ---------------------------------------------------------------- sequence slots

    private static GlmDsaCudaExecutor OpenExecutor(string path) => new GlmDsaCudaExecutor(path, 512, 64, 1);

    private static float[] Step(GlmDsaCudaExecutor ex, int[] tokens)
    {
        var logits = new float[ex.VocabSize];
        ex.Forward(tokens, logits);
        return logits;
    }

    private static int NewActiveSlot(GlmDsaCudaExecutor ex)
    {
        int slot = ex.SlotAlloc();
        Assert.True(slot >= 0);
        Assert.True(ex.SetActiveSlot(slot));
        return slot;
    }

    /// <summary>Two sequences interleaved chunk by chunk answer exactly as each alone: every cache
    /// and recurrent state belongs to its slot.</summary>
    [CudaFact]
    public void InterleavedSlots_AnswerExactlyAsEachSequenceAlone()
    {
        using var ex = OpenExecutor(Fixture());
        int[][] a = { Prompt[..24], Steps[..1], Steps[1..2] };
        int[][] b = { Prompt[16..40], Steps[3..4], Steps[4..5] };
        var aloneA = a.Select(c => Step(ex, c)).ToList();
        ex.ResetChecked();
        var aloneB = b.Select(c => Step(ex, c)).ToList();

        int sa = NewActiveSlot(ex);
        int sb = NewActiveSlot(ex);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(ex.SetActiveSlot(sa));
            Assert.True(RelativeError(Step(ex, a[i]), aloneA[i]) <= 1e-6, $"sequence A chunk {i}");
            Assert.True(ex.SetActiveSlot(sb));
            Assert.True(RelativeError(Step(ex, b[i]), aloneB[i]) <= 1e-6, $"sequence B chunk {i}");
        }
    }

    /// <summary>A captured decode step answers bit for bit as the same step issued kernel by kernel:
    /// the token and the position are read at launch, pools complete inside the graph, and crossing
    /// the indexer's top_k (16) captures the sparse regime beside the dense one. A freed sequence
    /// takes its graphs with it.</summary>
    [CudaFact]
    public void CapturedDecodeSteps_AnswerAsUncapturedSteps()
    {
        string path = Fixture(indexerTopK: 16);
        // Positions 10..25: dense to 15, sparse from 16, four pools completing on the way.
        int[] tokens = Enumerable.Range(0, 16).Select(i => Steps[i % Steps.Length] + i).ToArray();
        float[][] Decode(GlmDsaCudaExecutor ex)
        {
            Step(ex, Prompt[..10]);
            return tokens.Select(t => Step(ex, new[] { t })).ToArray();
        }

        float[][] captured, plain;
        using (var ex = OpenExecutor(path))
        {
            captured = Decode(ex);
            Assert.Equal(2, ex.Engine.CapturedDecodeGraphs);

            int first = ex.ActiveSlot;
            int other = NewActiveSlot(ex);
            Step(ex, Prompt[..4]);
            Step(ex, new[] { 7 });
            Assert.Equal(3, ex.Engine.CapturedDecodeGraphs);
            Assert.True(ex.SetActiveSlot(first));
            Assert.True(ex.SlotFree(other));
            Assert.Equal(2, ex.Engine.CapturedDecodeGraphs);
        }
        _env.Set("TS_GLM_GRAPHS", "0");
        using (var ex = OpenExecutor(path))
        {
            plain = Decode(ex);
            Assert.Equal(0, ex.Engine.CapturedDecodeGraphs);
        }
        for (int i = 0; i < tokens.Length; i++)
            Assert.Equal(plain[i], captured[i]);
    }

    /// <summary>A batched decode step computes each sequence's row exactly as that sequence's own step
    /// does (the dense matmuls, the per-slot experts and the attention all run row-invariant
    /// kernels), over several steps and with one sequence past the indexer's top_k.</summary>
    [CudaFact]
    public void BatchedDecode_ComputesEverySequencesOwnStep()
    {
        using var ex = new GlmDsaCudaExecutor(Fixture(indexerTopK: 16), 512, 64, 1);
        int[] prompts = { 20, 13, 31 };
        int n = prompts.Length, steps = 4;
        var batched = new int[n];
        var serial = new int[n];
        for (int i = 0; i < n; i++)
        {
            batched[i] = NewActiveSlot(ex);
            Step(ex, Prompt[..prompts[i]]);
            serial[i] = NewActiveSlot(ex);
            Step(ex, Prompt[..prompts[i]]);
        }
        for (int step = 0; step < steps; step++)
        {
            int[] tokens = Enumerable.Range(0, n).Select(i => Steps[(i + step) % Steps.Length]).ToArray();
            int[] positions = prompts.Select(p => p + step).ToArray();
            var rows = new float[n * ex.VocabSize];
            Assert.True(ex.ForwardBatchedDecode(batched, tokens, positions, rows));
            for (int i = 0; i < n; i++)
            {
                Assert.True(ex.SetActiveSlot(serial[i]));
                float[] want = Step(ex, new[] { tokens[i] });
                Assert.Equal(want, rows.AsSpan(i * ex.VocabSize, ex.VocabSize).ToArray());
            }
        }
    }

    // ---------------------------------------------------------------- speculative verify

    /// <summary>
    /// The n-gram drafter's verify on this engine: a six-token window gives each row the logits
    /// plain decoding gives that token, and a rejected tail leaves nothing behind. After the KDA
    /// snapshot is restored (the head returns to where it was taken) and the two accepted tokens
    /// are re-forwarded, continuing with tokens OTHER than the rejected ones answers exactly as a
    /// sequence that never saw the tail: no recurrent state, latent row, indexer key or pool key of
    /// it survives. The indexer is sparse throughout (top_k 16).
    /// </summary>
    [CudaFact]
    public void AVerifyWindow_RollsBackToItsAcceptedPrefix()
    {
        using var ex = OpenExecutor(Fixture(indexerTopK: 16));
        int vocab = ex.VocabSize;
        int[] window = Steps;
        int accepted = 2;
        int[] other = { 33, 44, 55, 66 };

        Step(ex, Prompt);
        float[][] plainWindow = window.Select(t => Step(ex, new[] { t })).ToArray();
        Assert.True(ex.ResetChecked());
        Step(ex, Prompt);
        foreach (int t in window[..accepted])
            Step(ex, new[] { t });
        float[][] plainOther = other.Select(t => Step(ex, new[] { t })).ToArray();
        Assert.True(ex.ResetChecked());

        Step(ex, Prompt);
        Assert.True(ex.KdaStateCapture());
        var rows = new float[window.Length * vocab];
        var hidden = new float[window.Length * 256];   // the fixture's embedding width
        Assert.True(ex.SpecForward(window, hidden, rows, allLogitsRows: true));
        Assert.Equal(Prompt.Length + window.Length, ex.NPast);
        for (int i = 0; i < window.Length; i++)
        {
            float[] row = rows.AsSpan(i * vocab, vocab).ToArray();
            Assert.True(RelativeError(row, plainWindow[i]) <= 1e-6, $"verify row {i}");
        }
        Assert.Contains(hidden, v => v != 0f);

        Assert.Equal(Prompt.Length, ex.KdaStateRestore());
        Assert.Equal(Prompt.Length, ex.NPast);
        Assert.True(ex.Rewind(Prompt.Length));
        foreach (int t in window[..accepted])
            Step(ex, new[] { t });
        for (int i = 0; i < other.Length; i++)
            Assert.True(RelativeError(Step(ex, new[] { other[i] }), plainOther[i]) <= 1e-6, $"step {i} after the rollback");
    }

    /// <summary>The recurrent state cannot go back: a slot rewinds to its head or to 0, and refuses
    /// anything in between without touching its state.</summary>
    [CudaFact]
    public void Rewind_ReachesOnlyTheHeadOrZero()
    {
        using var ex = OpenExecutor(Fixture());
        float[] first = Step(ex, Prompt);
        Assert.True(ex.Rewind(Prompt.Length));
        Assert.False(ex.Rewind(Prompt.Length - 4));
        float[] next = Step(ex, Steps[..1]);
        Assert.True(ex.Rewind(0));
        Assert.Equal(first, Step(ex, Prompt));
        Assert.Equal(next, Step(ex, Steps[..1]));
    }

    private static double RelativeError(float[] actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1.0, expected.Max(v => Math.Abs((double)v)));
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
            worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]) / scale);
        return worst;
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
    }
}
