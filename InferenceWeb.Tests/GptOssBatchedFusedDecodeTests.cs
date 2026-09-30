// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// GPT-OSS's token-batched decode (one graph, one token for each of N sequences) never ran: its graph
/// context asked the native context pool for 64 MiB, the pool refused anything over its 32 MiB buffers, and
/// every concurrent step fell back to one sequence at a time ("failed to acquire ggml context"). Now that it
/// runs, it has to decode what each sequence decodes on its own.
///
/// <para>Within the backend's kernel differences: one token per sequence and three per graph run different
/// matmul and attention kernels. On Metal they agree to under 0.5; on ggml_cuda (A40) the batch moved the
/// logits by up to 0.83 over these 24 steps, and flipped the greedy token only where the solo top-2 margin was
/// inside that (0.16 and 0.03). A batch that read another sequence's cache or position would move them by
/// far more and flip clear winners, so the check is: the same token wherever the solo margin exceeds the
/// difference measured at that step, and the difference within the backend's bound (0.5 on Metal, 1.5
/// elsewhere: about twice the A40 measurement).</para>
/// </summary>
public sealed class GptOssBatchedFusedDecodeTests
{
    private readonly ITestOutputHelper _output;

    public GptOssBatchedFusedDecodeTests(ITestOutputHelper output) => _output = output;

    [ModelFact("TS_TEST_MODEL_DIR", "gpt-oss-20b")]
    public void TheBatchedDecodeRuns_AndDecodesWhatEachSequenceDecodesAlone()
    {
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable("TS_TEST_MODEL_DIR"), "gpt-oss-20b")!;
        using ModelBase model = ModelBase.Create(path, TestGates.PinnedGgmlBackend);
        var fused = (IBatchedPagedModel)model;
        string[] prompts =
        {
            "The capital of France is",
            "Water boils at sea level at a temperature of",
            "The largest planet in the solar system is",
        };
        int n = prompts.Length;
        int[][] prompt = prompts.Select(p => model.Tokenizer.Encode(p, addSpecial: false).ToArray()).ToArray();

        // Two copies of every sequence: one decodes alone, the other in the batch.
        var next = new int[n];
        for (int i = 0; i < n; i++)
        {
            foreach (string copy in new[] { $"solo{i}", $"batch{i}" })
            {
                Assert.True(fused.BindSequenceCache(copy));
                next[i] = ArgMax(model.ForwardRefill(prompt[i]));
            }
        }
        fused.RestorePrimaryCache();

        string[] batch = Enumerable.Range(0, n).Select(i => $"batch{i}").ToArray();
        int[] positions = prompt.Select(p => p.Length).ToArray();
        double worst = 0;
        var failures = new List<string>();
        for (int step = 0; step < 8; step++)
        {
            var solo = new float[n][];
            for (int i = 0; i < n; i++)
            {
                fused.BindSequenceCache($"solo{i}");
                solo[i] = (float[])model.Forward(new[] { next[i] }).Clone();
            }
            fused.RestorePrimaryCache();

            var batched = new float[n][];
            Assert.True(fused.TryForwardBatchedFusedDecode(batch, (int[])next.Clone(), (int[])positions.Clone(), batched),
                $"the batched decode declined at step {step}");
            for (int i = 0; i < n; i++)
            {
                double diff = 0;
                for (int v = 0; v < solo[i].Length; v++)
                    diff = Math.Max(diff, Math.Abs(solo[i][v] - batched[i][v]));
                worst = Math.Max(worst, diff);
                int alone = ArgMax(solo[i]), together = ArgMax(batched[i]);
                double margin = TopTwoMargin(solo[i]);
                if (alone != together)
                {
                    _output.WriteLine($"step {step} sequence {i}: {alone} alone, {together} batched; solo top-2 margin {margin:G4}, difference {diff:G4}");
                    if (margin > diff)
                        failures.Add($"step {step} sequence {i}: the batch changed a clear winner ({alone} -> {together}, margin {margin:G4} > difference {diff:G4})");
                }
                // Both copies continue with the solo token, so one tie cannot compound.
                next[i] = alone;
                positions[i]++;
            }
        }
        double bound = TestGates.PinnedGgmlBackend == BackendType.GgmlMetal ? 0.5 : 1.5;
        _output.WriteLine($"{n} sequences x 8 steps: max |logit difference| batched vs alone {worst:G4} (bound {bound})");
        if (worst >= bound)
            failures.Add($"the batched decode differs from decoding alone by {worst:G4} (bound {bound})");
        Assert.Empty(failures);

        for (int i = 0; i < n; i++)
        {
            fused.OnSequenceReleased($"solo{i}");
            fused.OnSequenceReleased($"batch{i}");
        }
    }

    private static double TopTwoMargin(float[] values)
    {
        float best = float.NegativeInfinity, second = float.NegativeInfinity;
        foreach (float v in values)
        {
            if (v > best) { second = best; best = v; }
            else if (v > second) second = v;
        }
        return best - second;
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
