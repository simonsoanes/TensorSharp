// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// GPT-OSS's harmony template drops the analysis channel when it re-renders a past answer, so a
/// conversation's next turn diverges right after the previous prompt, and continuing its retained holder
/// means rewinding the whole previous answer - hundreds of tokens, past the prefix cache's default 16-token
/// rewind cap. The sliding window masks a linear cache, so the rewind is exact: after rewinding past a long
/// answer, a holder's kept rows are bitwise the rows of a holder that never held the answer, and a
/// continuation reads nothing else.
///
/// <para>The contract is checked where it is exact on every backend: the kept K/V bytes, and a token-by-token
/// continuation, which reads only those rows. A multi-row suffix prefill is reported, not asserted: on
/// ggml_cuda (A40) its kernels depend on more than the rows it reads: a 40-row suffix after rewinding a
/// 220-row answer measured 0.17 max |dlogit| with bitwise-identical kept rows and the stale rows zeroed, and
/// prefilling the same 520 tokens as 300+220 rows instead of 520 moves the logits by 2.4. That is kernel
/// noise, not the rewind.</para>
/// </summary>
public sealed class GptOssHolderRewindTests
{
    private readonly ITestOutputHelper _output;

    public GptOssHolderRewindTests(ITestOutputHelper output) => _output = output;

    [ModelFact("TS_TEST_MODEL_DIR", "gpt-oss-20b")]
    public void AHolderRewoundPastALongAnswer_ContinuesExactlyAsOneThatNeverHeldIt()
    {
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable("TS_TEST_MODEL_DIR"), "gpt-oss-20b")!;
        using ModelBase model = ModelBase.Create(path, TestGates.PinnedGgmlBackend);
        var fused = (IBatchedPagedModel)model;
        Assert.True(fused.SupportsRetainedFusedCache);

        int[] prompt = Tokens(model, "The history of the printing press begins in the fifteenth century. ", 300);
        int[] answer = Tokens(model, "An analysis channel the next turn will not see again. ", 220);
        int[] nextTurn = Tokens(model, "Summarize that in two sentences. ", 40);

        // A holder that never held the answer: the prompt, then the next turn.
        Assert.True(fused.BindSequenceCache("reference"));
        model.ForwardRefill(prompt);
        byte[] expectedRows = KvRows(model, prompt.Length);
        List<float[]> expected = Continue(model, nextTurn);
        fused.RestorePrimaryCache();
        Assert.True(fused.BindSequenceCache("reference-suffix"));
        model.ForwardRefill(prompt);
        float[] expectedSuffix = (float[])model.Forward(nextTurn).Clone();
        fused.RestorePrimaryCache();

        // The conversation's own holder: the prompt prefilled, the answer generated token by token as a turn
        // generates it, retained, handed to the next turn and rewound past the whole answer.
        float[] actualSuffix = null;
        foreach (string mode in new[] { "tokens", "suffix" })
        {
            Assert.True(fused.BindSequenceCache($"turn1-{mode}"));
            model.ForwardRefill(prompt);
            foreach (int token in answer)
                model.Forward(new[] { token });
            fused.RestorePrimaryCache();
            Assert.True(fused.RetainSequenceCache($"turn1-{mode}"));
            Assert.True(fused.TryRebindRetainedCache($"turn1-{mode}", $"turn2-{mode}"));
            Assert.False(fused.BindSequenceCache($"turn2-{mode}"));
            Assert.True(model.CanTruncateKVCache(prompt.Length + answer.Length, prompt.Length));
            Assert.True(model.TryTruncateKVCache(prompt.Length));

            if (mode == "tokens")
            {
                Assert.True(expectedRows.AsSpan().SequenceEqual(KvRows(model, prompt.Length)),
                    $"the {prompt.Length} rows kept by a rewind past {answer.Length} tokens differ from a holder that never held them");
                List<float[]> actual = Continue(model, nextTurn);
                double worst = 0;
                for (int step = 0; step < expected.Count; step++)
                    worst = Math.Max(worst, MaxDiff(expected[step], actual[step]));
                _output.WriteLine($"rewound {answer.Length} tokens; {nextTurn.Length}-token continuation max |logit difference| {worst:G4}");
                Assert.True(worst == 0, $"a continuation after a rewind past {answer.Length} tokens changed the logits by {worst:G4}");
            }
            else
            {
                actualSuffix = (float[])model.Forward(nextTurn).Clone();
            }
            fused.RestorePrimaryCache();
        }
        _output.WriteLine($"{nextTurn.Length}-row suffix prefill after the rewind: max |logit difference| " +
                          $"{MaxDiff(expectedSuffix, actualSuffix):G4} (reported, see the class notes)");

        foreach (string key in new[] { "reference", "reference-suffix", "turn2-tokens", "turn2-suffix" })
            fused.OnSequenceReleased(key);
    }

    /// <summary>The bound cache's first <paramref name="rows"/> K/V rows, as bytes.</summary>
    private static byte[] KvRows(ModelBase model, int rows)
    {
        var bytes = new byte[model.ComputeKVBlockByteSize(rows)];
        Assert.True(model.TryExtractKVBlock(0, rows, bytes));
        return bytes;
    }

    /// <summary>Feed <paramref name="tokens"/> one at a time and keep every step's logits.</summary>
    private static List<float[]> Continue(ModelBase model, int[] tokens)
    {
        var logits = new List<float[]>();
        foreach (int token in tokens)
            logits.Add((float[])model.Forward(new[] { token }).Clone());
        return logits;
    }

    private static double MaxDiff(float[] a, float[] b)
    {
        double worst = 0;
        for (int v = 0; v < a.Length; v++)
            worst = Math.Max(worst, Math.Abs(a[v] - b[v]));
        return worst;
    }

    private static int[] Tokens(ModelBase model, string sentence, int count)
    {
        var tokens = new List<int>();
        while (tokens.Count < count)
            tokens.AddRange(model.Tokenizer.Encode(sentence, addSpecial: false));
        return tokens.Take(count).ToArray();
    }
}
