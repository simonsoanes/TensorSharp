// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

namespace InferenceWeb.Tests;

public class TokenSamplerTests
{
    [Fact]
    public void Sample_TopPNormalizesAfterTopK()
    {
        // Conditioned on the top two logits, token 0 has probability ~0.731,
        // so llama.cpp's top_p=.7 keeps only it. A full-vocabulary softmax has
        // token 0 below .7 and incorrectly leaves token 1 eligible.
        float[] logits = { 2f, 1f, 1f, 1f };
        for (int seed = 0; seed < 128; seed++)
        {
            var sampler = new TokenSampler(new SamplingConfig
            {
                Temperature = 1f,
                TopK = 2,
                TopP = 0.7f,
                MinP = 0f,
                RepetitionPenalty = 1f,
                Seed = seed,
            });
            Assert.Equal(0, sampler.Sample(logits));
        }
    }

    [Fact]
    public void Sample_AppliesTemperatureAfterNucleusFiltering()
    {
        // At T=1 the top-p nucleus contains tokens 0,1,2. Applying T=.7
        // before top-p (the old TensorSharp order) shrinks it to 0,1 and token
        // 2 can never be drawn. In llama order temperature only reshapes the
        // final three-token distribution.
        float[] logits = { 2.6f, 2.4f, 1.6f, 0.3f, -1.9f, -1.9f };
        var observed = new HashSet<int>();
        for (int seed = 0; seed < 256; seed++)
        {
            var sampler = new TokenSampler(new SamplingConfig
            {
                Temperature = 0.7f,
                TopK = 5,
                TopP = 0.8f,
                MinP = 0f,
                RepetitionPenalty = 1f,
                Seed = seed,
            });
            observed.Add(sampler.Sample(logits));
        }

        Assert.Subset(observed, new HashSet<int> { 0, 1, 2 });
        Assert.Contains(2, observed);
    }

    [Fact]
    public void ApplyPenalties_OnlyCountsConfiguredRecentWindow()
    {
        var sampler = new TokenSampler(new SamplingConfig
        {
            RepetitionPenalty = 2f,
            PenaltyLastN = 1,
        });
        float[] scores = { 10f, 10f, 10f };

        sampler.ApplyPenalties(scores, new[] { 0, 0, 1 });

        Assert.Equal(10f, scores[0]);
        Assert.Equal(5f, scores[1]);
        Assert.Equal(10f, scores[2]);
    }

    [Fact]
    public void ApplyPenalties_PendingDraftsConsumeNewestHistoryWindow()
    {
        var sampler = new TokenSampler(new SamplingConfig
        {
            RepetitionPenalty = 1f,
            FrequencyPenalty = 1f,
            PenaltyLastN = 2,
        });
        float[] scores = { 10f, 10f, 10f };

        sampler.ApplyPenalties(scores, new[] { 0, 0, 0 }, new[] { 1, 1 });

        Assert.Equal(10f, scores[0]);
        Assert.Equal(8f, scores[1]);
        Assert.Equal(10f, scores[2]);
    }
}
