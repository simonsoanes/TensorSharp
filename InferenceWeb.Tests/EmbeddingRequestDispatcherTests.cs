// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using Microsoft.AspNetCore.Http;
using TensorSharp.Models.Embeddings;
using TensorSharp.Server.ProtocolAdapters;

namespace InferenceWeb.Tests;

public sealed class EmbeddingRequestDispatcherTests
{
    private sealed class Call(int[][] inputs, CancellationToken token)
    {
        internal readonly int[][] Inputs = inputs;
        internal readonly CancellationToken Token = token;
        internal readonly TaskCompletionSource<EmbeddingBatchResult> Completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
        internal void Complete() => Completion.SetResult(Result(Inputs));
    }

    private static EmbeddingBatchResult Result(IReadOnlyList<int[]> inputs) => new(
        inputs.Select(input => new float[] { input[0], input.Length, input[^1] }).ToArray(), inputs.Sum(input => input.Length));

    private sealed class ControlledModel : IEmbeddingModel
    {
        internal readonly Channel<Call> Started = Channel.CreateUnbounded<Call>();
        internal readonly ConcurrentQueue<Call> Calls = new();
        internal Action? BeforeSubmit;
        internal bool CompleteSynchronously;
        internal bool Disposed;
        private int _active;
        public string ModelName => "test-encoder";
        public string Architecture => "bert";
        public int Dimensions => 3;
        public int MaxTokens => 8192;
        public int VocabularySize => 256;
        public int[] Tokenize(string text, bool truncate = false) => [text.Length, 9];
        public async Task<EmbeddingBatchResult> EmbedTokensAsync(IReadOnlyList<int[]> inputs, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Assert.Equal(1, Interlocked.Increment(ref _active));
            try
            {
                BeforeSubmit?.Invoke();
                var call = new Call(inputs.Select(input => input.ToArray()).ToArray(), cancellationToken);
                Calls.Enqueue(call);
                Started.Writer.TryWrite(call);
                if (CompleteSynchronously) call.Complete();
                return await call.Completion.Task.WaitAsync(cancellationToken);
            }
            finally { Interlocked.Decrement(ref _active); }
        }
        internal Task<Call> Next() => Started.Reader.ReadAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(5));
        public void Dispose() => Disposed = true;
    }

    [Fact]
    public async Task FirstRequestStartsImmediatelyAndPreservesCallerTokenAndModelOwnership()
    {
        var model = new ControlledModel();
        var dispatcher = new EmbeddingRequestDispatcher(model);
        using var cancellation = new CancellationTokenSource();
        Task<EmbeddingBatchResult> request = dispatcher.EmbedAsync([[3, 2]], cancellation.Token);
        Assert.True(model.Started.Reader.TryRead(out var call));
        Assert.Equal(cancellation.Token, call!.Token);
        call.Complete();
        Assert.Equal(2, (await request).PromptTokens);
        await dispatcher.DisposeAsync();
        dispatcher.Dispose();
        Assert.False(model.Disposed);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => dispatcher.EmbedAsync([[1]], default));
    }

    [Fact]
    public async Task AlreadyWaitingRequestsCombineAndKeepTheirOwnRowsAndUsage()
    {
        var model = new ControlledModel();
        await using var dispatcher = new EmbeddingRequestDispatcher(model);
        var first = dispatcher.EmbedAsync([[1]], default);
        var leader = await model.Next();
        var a = dispatcher.EmbedAsync([[2, 3], [4]], default);
        var b = dispatcher.EmbedAsync([[5, 6, 7]], default);
        var c = dispatcher.EmbedAsync([[8]], default);
        Assert.Single(model.Calls);
        leader.Complete();
        var combined = await model.Next();
        Assert.Equal(new[] { 2, 4, 5, 8 }, combined.Inputs.Select(input => input[0]));
        combined.Complete();
        Assert.Equal(1, (await first).PromptTokens);
        Assert.Equal(3, (await a).PromptTokens);
        Assert.Equal(new[] { 2f, 4f }, (await a).Embeddings.Select(vector => vector[0]));
        Assert.Equal(3, (await b).PromptTokens);
        Assert.Equal(5f, (await b).Embeddings[0][0]);
        Assert.Equal(1, (await c).PromptTokens);
        Assert.Equal(2, model.Calls.Count);
    }

    [Fact]
    public async Task CoalescingStopsAtSixtyFourSequences()
    {
        var model = new ControlledModel();
        await using var dispatcher = new EmbeddingRequestDispatcher(model);
        var first = dispatcher.EmbedAsync([[1]], default);
        var leader = await model.Next();
        var requests = Enumerable.Range(2, 65).Select(token => dispatcher.EmbedAsync([[token]], default)).ToArray();
        leader.Complete();
        var combined = await model.Next();
        Assert.Equal(64, combined.Inputs.Length);
        combined.Complete();
        var tail = await model.Next();
        Assert.Single(tail.Inputs);
        Assert.Equal(66, tail.Inputs[0][0]);
        tail.Complete();
        await Task.WhenAll(requests.Append(first));
    }

    [Fact]
    public async Task CoalescingStopsAtFourThousandNinetySixTokens()
    {
        var model = new ControlledModel();
        await using var dispatcher = new EmbeddingRequestDispatcher(model);
        var first = dispatcher.EmbedAsync([[1]], default);
        var leader = await model.Next();
        var requests = Enumerable.Range(2, 3).Select(token =>
            dispatcher.EmbedAsync([Enumerable.Repeat(token, 2048).ToArray()], default)).ToArray();
        leader.Complete();
        var combined = await model.Next();
        Assert.Equal(4096, combined.Inputs.Sum(input => input.Length));
        combined.Complete();
        var tail = await model.Next();
        Assert.Single(tail.Inputs);
        tail.Complete();
        Assert.All(await Task.WhenAll(requests), result => Assert.Equal(2048, result.PromptTokens));
        await first;
    }

    [Theory]
    [InlineData(65, 1)]
    [InlineData(1, 4097)]
    public async Task OversizedRequestRunsAloneWithoutSplittingItsInputs(int count, int tokens)
    {
        var model = new ControlledModel();
        await using var dispatcher = new EmbeddingRequestDispatcher(model);
        var first = dispatcher.EmbedAsync([[1]], default);
        var leader = await model.Next();
        var large = dispatcher.EmbedAsync(Enumerable.Range(2, count).Select(token => Enumerable.Repeat(token, tokens).ToArray()).ToArray(), default);
        var small = dispatcher.EmbedAsync([[100]], default);
        leader.Complete();
        var oversized = await model.Next();
        Assert.Equal(count, oversized.Inputs.Length);
        Assert.All(oversized.Inputs, input => Assert.Equal(tokens, input.Length));
        oversized.Complete();
        var tail = await model.Next();
        Assert.Equal(100, Assert.Single(tail.Inputs)[0]);
        tail.Complete();
        Assert.Equal(count * tokens, (await large).PromptTokens);
        await Task.WhenAll(first, small);
    }

    [Fact]
    public async Task QueuedCancellationCompletesPromptlyAndSkipsPhysicalInference()
    {
        var model = new ControlledModel();
        await using var dispatcher = new EmbeddingRequestDispatcher(model);
        var first = dispatcher.EmbedAsync([[1]], default);
        var leader = await model.Next();
        using var cancellation = new CancellationTokenSource();
        var canceled = dispatcher.EmbedAsync([[2]], cancellation.Token);
        var healthy = dispatcher.EmbedAsync([[3]], default);
        cancellation.Cancel();
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => canceled.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.False(first.IsCompleted);
        leader.Complete();
        var tail = await model.Next();
        Assert.Equal(3, Assert.Single(tail.Inputs)[0]);
        tail.Complete();
        await Task.WhenAll(first, healthy);
    }

    [Fact]
    public async Task OneCanceledCallerCannotCancelItsHealthyBatchPeer()
    {
        var model = new ControlledModel();
        await using var dispatcher = new EmbeddingRequestDispatcher(model);
        var first = dispatcher.EmbedAsync([[1]], default);
        var leader = await model.Next();
        using var cancellation = new CancellationTokenSource();
        var canceled = dispatcher.EmbedAsync([[2]], cancellation.Token);
        var healthy = dispatcher.EmbedAsync([[3]], default);
        leader.Complete();
        var combined = await model.Next();
        Assert.Equal(2, combined.Inputs.Length);
        cancellation.Cancel();
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => canceled.WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.False(combined.Token.IsCancellationRequested);
        Assert.False(healthy.IsCompleted);
        combined.Complete();
        Assert.Equal(3f, (await healthy).Embeddings[0][0]);
        await first;
    }

    [Fact]
    public async Task CancelingEveryCallerCancelsPhysicalBatchAndLeavesDispatcherUsable()
    {
        var model = new ControlledModel();
        await using var dispatcher = new EmbeddingRequestDispatcher(model);
        var first = dispatcher.EmbedAsync([[1]], default);
        var leader = await model.Next();
        using var aCancellation = new CancellationTokenSource();
        using var bCancellation = new CancellationTokenSource();
        var a = dispatcher.EmbedAsync([[2]], aCancellation.Token);
        var b = dispatcher.EmbedAsync([[3]], bCancellation.Token);
        leader.Complete();
        var combined = await model.Next();
        aCancellation.Cancel();
        bCancellation.Cancel();
        Assert.True(combined.Token.IsCancellationRequested);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => a);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => b);
        var recovered = dispatcher.EmbedAsync([[4]], default);
        (await model.Next()).Complete();
        Assert.Equal(4f, (await recovered).Embeddings[0][0]);
        await first;
    }

    [Fact]
    public async Task IsolatedCancellationStopsPhysicalCallAndPreservesTokenIdentity()
    {
        var model = new ControlledModel();
        await using var dispatcher = new EmbeddingRequestDispatcher(model);
        using var cancellation = new CancellationTokenSource();
        var first = dispatcher.EmbedAsync([[1]], cancellation.Token);
        var call = await model.Next();
        cancellation.Cancel();
        Assert.Equal(cancellation.Token, call.Token);
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => first);
        Assert.Equal(cancellation.Token, error.CancellationToken);
        var recovered = dispatcher.EmbedAsync([[2]], default);
        (await model.Next()).Complete();
        await recovered;
    }

    [Fact]
    public async Task DisposalCancelsOwnedPhysicalWorkAndFailsQueuedCallersWithoutOwningModel()
    {
        var model = new ControlledModel();
        var dispatcher = new EmbeddingRequestDispatcher(model);
        var first = dispatcher.EmbedAsync([[1]], default);
        var call = await model.Next();
        var queued = dispatcher.EmbedAsync([[2]], default);
        await dispatcher.DisposeAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(2));
        Assert.True(call.Token.IsCancellationRequested);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => first);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => queued);
        Assert.False(model.Disposed);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => dispatcher.EmbedAsync([[3]], default));
    }

    [Fact]
    public async Task DisposalWaitsForPhysicalCompletionWhenCallerOwnsCancellationSource()
    {
        var model = new ControlledModel();
        var dispatcher = new EmbeddingRequestDispatcher(model);
        using var cancellation = new CancellationTokenSource();
        var first = dispatcher.EmbedAsync([[1]], cancellation.Token);
        var call = await model.Next();
        var stopped = dispatcher.DisposeAsync().AsTask();
        Assert.False(stopped.IsCompleted);
        Assert.False(call.Token.IsCancellationRequested);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => first);
        call.Complete();
        await stopped.WaitAsync(TimeSpan.FromSeconds(2));
        dispatcher.Dispose();
    }

    [Fact]
    public async Task PhysicalFailureReachesEveryAffectedCallerAndPumpRecovers()
    {
        var model = new ControlledModel();
        await using var dispatcher = new EmbeddingRequestDispatcher(model);
        var first = dispatcher.EmbedAsync([[1]], default);
        var leader = await model.Next();
        var a = dispatcher.EmbedAsync([[2]], default);
        var b = dispatcher.EmbedAsync([[3]], default);
        leader.Complete();
        var combined = await model.Next();
        combined.Completion.SetException(new InvalidOperationException("encoder failure"));
        Assert.Equal("encoder failure", (await Assert.ThrowsAsync<InvalidOperationException>(() => a)).Message);
        Assert.Equal("encoder failure", (await Assert.ThrowsAsync<InvalidOperationException>(() => b)).Message);
        var recovered = dispatcher.EmbedAsync([[4]], default);
        (await model.Next()).Complete();
        await Task.WhenAll(first, recovered);
    }

    [Fact]
    public async Task ReentrantSubmitAndDisposeFailClearlyInsteadOfDeadlocking()
    {
        var model = new ControlledModel();
        await using var dispatcher = new EmbeddingRequestDispatcher(model);
        model.BeforeSubmit = () =>
        {
            Assert.Throws<InvalidOperationException>(() => { _ = dispatcher.EmbedAsync([[2]], default); });
            Assert.Throws<InvalidOperationException>(dispatcher.Dispose);
        };
        var first = dispatcher.EmbedAsync([[1]], default);
        (await model.Next()).Complete();
        await first;
    }

    [Fact]
    public async Task SynchronousCompletionPreservesPumpOwnershipAndDoesNotLeakExecutionContext()
    {
        var model = new ControlledModel { CompleteSynchronously = true };
        await using var dispatcher = new EmbeddingRequestDispatcher(model);
        for (int token = 1; token <= 20; ++token)
            Assert.Equal((float)token, (await dispatcher.EmbedAsync([[token]], default)).Embeddings[0][0]);
        var requests = Enumerable.Range(21, 128).Select(token => Task.Run(async () =>
        {
            var result = await dispatcher.EmbedAsync([[token]], default);
            Assert.Equal((float)token, result.Embeddings[0][0]);
        }));
        await Task.WhenAll(requests).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(149f, (await dispatcher.EmbedAsync([[149]], default)).Embeddings[0][0]);
    }

    [Fact]
    public async Task IncorrectEncoderResultCountFailsRequestAndPumpRecovers()
    {
        var model = new ControlledModel();
        await using var dispatcher = new EmbeddingRequestDispatcher(model);
        var first = dispatcher.EmbedAsync([[1]], default);
        (await model.Next()).Completion.SetResult(new EmbeddingBatchResult([], 1));
        var error = await Assert.ThrowsAsync<InvalidOperationException>(() => first);
        Assert.Contains("result count", error.Message);
        var recovered = dispatcher.EmbedAsync([[2]], default);
        (await model.Next()).Complete();
        await recovered;
    }

    [Fact]
    public async Task SharedProtocolDispatcherPreservesDimensionsEncodingAndEachRequestsUsage()
    {
        var model = new ControlledModel();
        await using var adapter = new EmbeddingAdapter(model, EmbeddingEndpointTests.Options());
        static DefaultHttpContext Context(string json)
        {
            var context = new DefaultHttpContext();
            context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(json));
            context.Response.Body = new MemoryStream();
            return context;
        }
        static JsonElement Response(HttpContext context)
        {
            context.Response.Body.Position = 0;
            using var document = JsonDocument.Parse(context.Response.Body);
            return document.RootElement.Clone();
        }
        var lead = Context("""{"model":"test-encoder","input":[1]}""");
        var first = adapter.OpenAIAsync(lead);
        var leader = await model.Next();
        var openAI = Context("""{"model":"test-encoder","input":[[3,2],[5,2]],"dimensions":2,"encoding_format":"base64"}""");
        var ollama = Context("""{"model":"test-encoder","input":"hi"}""");
        var ollamaSecond = Context("""{"model":"test-encoder","input":"a"}""");
        var a = adapter.OpenAIAsync(openAI);
        var b = adapter.OllamaAsync(ollama);
        var c = adapter.OllamaAsync(ollamaSecond);
        leader.Complete();
        var combined = await model.Next();
        Assert.Equal(4, combined.Inputs.Length);
        combined.Complete();
        await Task.WhenAll(first, a, b, c);
        var encoded = Response(openAI);
        Assert.Equal(4, encoded.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
        Assert.Equal(2, encoded.GetProperty("data").GetArrayLength());
        Assert.Equal(8, Convert.FromBase64String(encoded.GetProperty("data")[0].GetProperty("embedding").GetString()!).Length);
        Assert.Equal(2, Response(ollama).GetProperty("prompt_eval_count").GetInt32());
        Assert.Equal(3, Response(ollama).GetProperty("embeddings")[0].GetArrayLength());
        Assert.Equal(3, Response(ollamaSecond).GetProperty("embeddings")[0].GetArrayLength());
    }
}
