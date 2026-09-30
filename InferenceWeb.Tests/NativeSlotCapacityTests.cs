// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Concurrent requests on a model that keeps each one in its own device slot (DeepSeek V4.1, GLM 5.x) when
// device memory holds fewer slots than requests. GLM-5.3-Flash's loader sizes the context to fill the
// devices, so on six A40s only two slots fit: four concurrent chats returned "GLM sequence-slot allocation
// failed (device memory exhausted?)" for the third and fourth. A missing slot is a capacity limit: the
// request now waits for a running one to finish. OracleFakes.N has three native slots (the primary's
// included), and a fourth request used to fail the same way.
using System.Threading;
using System.Threading.Tasks;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

public sealed class NativeSlotCapacityTests
{
    private sealed class RecordingLogger : Microsoft.Extensions.Logging.ILogger
    {
        private readonly List<string> _messages = new();
        public IReadOnlyList<string> Messages { get { lock (_messages) return _messages.ToArray(); } }
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel logLevel) => true;
        public void Log<TState>(Microsoft.Extensions.Logging.LogLevel logLevel, Microsoft.Extensions.Logging.EventId eventId,
            TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            lock (_messages) _messages.Add(formatter(state, exception));
        }
    }

    private static SchedulerConfig Config(bool prefixCaching = true) => new()
    {
        BlockSize = 16, NumBlocks = 512, MaxNumRunningSequences = 8, MaxNumBatchedTokens = 128,
        SoloPrefillChunkSize = 128, EnablePrefixCaching = prefixCaching,
        StopRepetition = false,
    };

    private static List<int> Prompt(int i) => Enumerable.Range(1 + i * 23, 30).Select(t => t % 250 + 1).ToList();

    private static async Task<List<int>> Cold(List<int> prompt, int maxNew)
    {
        using var model = OracleFakes.N();
        using var engine = new InferenceEngine(model, Config(prefixCaching: false), NullLogger.Instance);
        var seq = new SequenceState("cold", prompt, maxNew, 16, SamplingConfig.Greedy);
        await engine.SubmitRequest(seq).Completion.WaitAsync(TimeSpan.FromSeconds(20));
        return seq.OutputTokens.ToList();
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    public async Task MoreConcurrentRequestsThanSlots_WaitForOne_AndAnswerLikeColdRuns(int requests)
    {
        var logger = new RecordingLogger();
        using var model = OracleFakes.N();
        using var engine = new InferenceEngine(model, Config(), logger);
        var gate = new ComputeGate();
        gate.Close();
        engine.ComputeGate = gate;
        SequenceState[] sequences = Enumerable.Range(0, requests)
            .Select(i => new SequenceState($"r{i}", Prompt(i), 20, 16, SamplingConfig.Greedy, cacheScope: $"s{i}")).ToArray();
        var handles = sequences.Select(s => engine.SubmitRequest(s)).ToArray();
        gate.Open();

        InferenceCompletion[] done = await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromSeconds(60));

        Assert.All(done, d => Assert.Equal(SequenceStatus.FinishedLengthCapped, d.Status));
        for (int i = 0; i < requests; i++)
            Assert.Equal(await Cold(Prompt(i), 20), sequences[i].OutputTokens);
        Assert.Contains(logger.Messages, m => m.Contains("has no device slot for another concurrent request", StringComparison.Ordinal));
    }
}
