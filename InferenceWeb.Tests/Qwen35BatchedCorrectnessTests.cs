// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Correctness: per-seq KV-swap vs batched paged-attention self-consistency
// on Qwen3.5. With greedy sampling (temperature=0, top-k=0)
// both paths must produce the same token stream for the same prompt because
// they're computing the same attention + GDN math; any divergence is either
// numerical drift or a structural bug in the batched implementation.
//
// We don't compare against vLLM directly here — vLLM requires HF safetensors
// (not GGUF) and a Python venv setup that's beyond an in-session test. The
// per-seq Forward path is the single-stream path (it produced the validated
// "Dragonfruit" image-aware output and coherent chat text), so
// self-consistency against it is the meaningful check for batched
// correctness.
//
// Opt-in via TS_TEST_MODEL_DIR pointing at the directory containing
// Qwen3.6-27B-IQ4_XS.gguf. Slow (model load + two forward runs per prompt).
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Qwen35BatchedCorrectnessTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    // --no-continuous-batching's variable: set to 1, every sequence takes the
    // per-sequence path instead of the batched one.
    private const string PerSequenceVar = "TS_SCHED_DISABLE_BATCHED";

    private readonly ITestOutputHelper _output;
    public Qwen35BatchedCorrectnessTests(ITestOutputHelper output) { _output = output; }

    // The actual correctness check: same prompt, greedy sampling, both paths.
    // Expectation: per-seq and batched produce identical token sequences for
    // at least the first several tokens (where numerical drift hasn't yet
    // accumulated enough to flip the argmax). We assert at least 4 of the
    // first 8 tokens match — relaxes for accumulating FP drift while still
    // catching structural divergence (which would diverge at token 0).
    [ModelFact("TS_TEST_MODEL_DIR", Qwen35Gguf)]
    public async Task Qwen35_Greedy_PerSequenceAndBatchedAgree()
    {
        var modelPath = FindQwen35();
        // The gate and this loader share Qwen35Gguf, so null here means they have
        // drifted apart again. Fail loudly rather than return into a green tick.
        Assert.True(modelPath != null,
            $"[ModelFact] admitted this test but no '{Qwen35Gguf}' GGUF was found under {EnvModelDir}; "
            + "the gate and the loader have drifted apart.");
        _output.WriteLine($"[qwen35-corr] loading {Path.GetFileName(modelPath)}");

        using var ctx = new CorrCtx(modelPath);

        string[] prompts =
        {
            "Q: What is two plus two?\nA:",
            "Q: Who wrote Hamlet?\nA:",
            "Q: Translate 'hello' to French.\nA:",
        };

        const int maxNewTokens = 8;
        int totalCompared = 0;
        int totalMatching = 0;

        foreach (var prompt in prompts)
        {
            // Per-seq first: ensures the model's GDN _convState/_deltaStateTensor
            // are zeroed by ResetKVCache before each path. Switching paths
            // without a reset would leak state from the previous run.
            ctx.Model.ResetKVCache();
            var perSeqTokens = await GenerateGreedy(ctx, prompt, maxNewTokens, optIn: false);

            ctx.Model.ResetKVCache();
            var batchedTokens = await GenerateGreedy(ctx, prompt, maxNewTokens, optIn: true);

            int matchPrefix = 0;
            int compareLen = Math.Min(perSeqTokens.Count, batchedTokens.Count);
            for (int i = 0; i < compareLen; i++)
            {
                if (perSeqTokens[i] == batchedTokens[i]) matchPrefix++;
                else break;
            }
            totalCompared += compareLen;
            totalMatching += matchPrefix;

            string perSeqText = ctx.Model.Tokenizer.Decode(perSeqTokens);
            string batchedText = ctx.Model.Tokenizer.Decode(batchedTokens);
            _output.WriteLine(
                $"[qwen35-corr] prompt=\"{Trim(prompt, 50)}\" " +
                $"perSeq=[{string.Join(",", perSeqTokens)}] " +
                $"batched=[{string.Join(",", batchedTokens)}] " +
                $"matchPrefix={matchPrefix}/{compareLen}");
            _output.WriteLine($"  per-seq: \"{Trim(perSeqText, 60)}\"");
            _output.WriteLine($"  batched: \"{Trim(batchedText, 60)}\"");
        }

        double matchRate = totalCompared > 0 ? (double)totalMatching / totalCompared : 0;
        _output.WriteLine($"[qwen35-corr] overall prefix-match {totalMatching}/{totalCompared} = {matchRate:P0}");

        // Threshold: at least 50% of compared tokens should be identical
        // prefix between paths. Random argmax overlap on a 150k-vocab model
        // is effectively 0%, so anything > random is signal. We aim for at
        // least HALF of the prefix to align — gracious to FP drift while
        // still failing on structural bugs (which would produce ~0% match).
        Xunit.Assert.True(matchRate >= 0.5,
            $"Per-sequence/batched prefix-match rate {matchRate:P0} below 50% — structural divergence suspected.");
    }

    private async Task<List<int>> GenerateGreedy(CorrCtx ctx, string prompt, int maxNewTokens, bool optIn)
    {
        Environment.SetEnvironmentVariable(PerSequenceVar, optIn ? "0" : "1");

        var history = new List<ChatMessage> { new() { Role = "user", Content = prompt } };
        var tokens = ctx.Renderer.RenderToTokens(
            ctx.Model.Tokenizer, ctx.Model.Config?.ChatTemplate, history,
            ctx.Model.Config?.Architecture ?? string.Empty,
            addGenerationPrompt: true, tools: null, enableThinking: false);

        // Unique request id per call so the engine doesn't think we're
        // re-submitting the same chat (which would re-use cached prefix).
        string reqId = $"corr-{(optIn ? "b" : "l")}-{Guid.NewGuid():N}";
        var seq = new SequenceState(reqId, tokens, maxNewTokens, ctx.BlockSize, SamplingConfig.Greedy);
        var handle = ctx.Engine.SubmitRequest(seq);
        var outs = new List<int>();
        try
        {
            await foreach (var tok in handle.Tokens.ReadAllAsync())
                outs.Add(tok);
        }
        catch { /* engine surfaces errors via Completion */ }
        await handle.Completion;
        return outs;
    }

    private static string Trim(string s, int len)
        => string.IsNullOrEmpty(s) ? string.Empty
           : (s.Length <= len ? s : s.Substring(0, len) + "...").Replace("\n", "\\n");

    /// <summary>
    /// The ONE pattern the [ModelFact] gate and the loader below both use.
    ///
    /// <para>
    /// They used to differ: the gate asked for "27b" while the loader accepted
    /// only "qwen3.6-27b" or "qwen3.5-27b". With Qwen3.8-27B on disk the gate
    /// matched, the loader did not, and the body took its `return` — which xUnit
    /// records as PASSED, not skipped. The correctness net for the qwen35 batched
    /// path was reporting green on a machine where it had loaded nothing. One
    /// pattern, shared, is the only arrangement that cannot drift.
    /// </para>
    /// </summary>
    private const string Qwen35Gguf = "qwen3.5-27b|qwen3.6-27b|qwen3.8-27b";

    private static string FindQwen35()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return null;
        return TestGates.FindSmallestGguf(dir, Qwen35Gguf);
    }

    private sealed class CorrCtx : IDisposable
    {
        public TensorSharp.Models.ModelBase Model { get; }
        public KVCachePromptRenderer Renderer { get; }
        public InferenceEngine Engine { get; }
        public int BlockSize { get; }

        public CorrCtx(string modelPath)
        {
            // The assembly pins ONE process-global GGML backend up front
            // (GgmlBackendTestInitializer), so picking Metal here purely because
            // the host is a Mac fought that pin and threw "a different GGML
            // backend was already initialized in this process" before a single
            // token was generated. Read the same switch the initializer reads.
            BackendType backend = TestGates.PinnedGgmlBackend;
            Model = TensorSharp.Models.ModelBase.Create(modelPath, backend);
            Renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
            BlockSize = 256;
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = 4096,
                MaxNumRunningSequences = 4,
                MaxPrefillChunkSize = 1024,
                NumBlocks = 64,
                BlockSize = BlockSize,
                EnablePrefixCaching = false, // disable so test runs see fresh KV every time
                DecodeQuantumTokens = BlockSize,
            };
            Engine = new InferenceEngine(Model, cfg, NullLogger.Instance);
        }

        public void Dispose()
        {
            Engine?.Dispose();
            Model?.Dispose();
        }
    }
}
