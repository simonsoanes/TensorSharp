// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Logging;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace TensorSharp.Server.Skills
{
    /// <summary>Runs one generation and reports it as raw text plus metrics.</summary>
    internal delegate IAsyncEnumerable<ChatStreamUpdate> SkillChatGeneration(
        List<ChatMessage> messages,
        List<ToolFunction> tools,
        CancellationToken cancellationToken);

    /// <summary>
    /// Wraps a chat generation in the progressive-disclosure loop: generate, answer any
    /// <c>skills_*</c> call in process, generate again, and stream only the round that
    /// finally answers.
    ///
    /// <para>
    /// <b>It streams.</b> The loop runs its own output parser — it has to, because it is
    /// looking for <c>skills_read</c> calls to answer — and forwards the SEPARATED
    /// pieces rather than the raw text: content as content, reasoning as reasoning, tool
    /// markup not at all. Every update it yields is marked
    /// <see cref="ChatStreamUpdate.IsParsed"/>, and an adapter that sees that must not
    /// parse again.
    /// </para>
    /// <para>
    /// That indirection is the whole design, and it was arrived at the hard way. The
    /// first version buffered each round and replayed only the last one, because
    /// forwarding a round's raw tokens would hand <c>skills_read</c> to a client with no
    /// implementation for it. It worked, and it silently cost every skills request its
    /// streaming: measured on the Web UI, 26 SSE content frames starting 0.17 s in
    /// became 3 frames all arriving at 3.28 s. Stopping the forward at the first tool
    /// token instead is not a fix either — it leaves the ADAPTER's parser inside a
    /// half-open tool-call span, and the next round's text is consumed as that call's
    /// arguments, so the answer vanishes rather than merely arriving late. Parsing once,
    /// here, and handing over the pieces avoids both: the markup never exists as far as
    /// the adapter is concerned, so there is no span to be caught inside.
    /// </para>
    /// <para>
    /// <b>Cost.</b> Every round goes through the same <see cref="ChatSession"/>, so the
    /// pipeline records each assistant turn's generated tokens in the session's tracked
    /// history and the next render splices them back rather than re-tokenizing —
    /// which is what keeps the rendered prefix byte-identical from round to round.
    /// </para>
    /// <para>
    /// A round therefore re-prefills only what it added. Measured on gemma-4-E4B
    /// (ggml_metal), a lookup that appends a 7.9 KB skill body reuses 1983 of the
    /// following round's 4197 prompt tokens (47%) — the whole of the previous round,
    /// with only the newly fetched file left to forward — and its time to first token
    /// drops from 2.3 s to 1.3 s. This needed an engine fix:
    /// <c>BatchExecutor.ComputeLiveContinuationLcp</c> used to require the live KV
    /// cache to be an EXACT prefix of the new prompt, and a turn that ends on a
    /// control token the chat template never re-renders (Gemma 4 answers a tool call by
    /// emitting <c>&lt;|tool_response&gt;</c>) failed that test by one token and
    /// re-prefilled everything. It now rewinds a bounded number of trailing tokens.
    /// </para>
    /// </summary>
    internal static class SkillChatLoop
    {
        /// <summary>
        /// Drive <paramref name="generate"/> until the model stops asking for skill
        /// content, then stream the final round.
        /// </summary>
        /// <param name="architecture">The loaded model's architecture, for the detection parser.</param>
        /// <param name="messages">The conversation, with the skill block already injected.</param>
        /// <param name="plan">The request's skills, tools and bounds.</param>
        /// <param name="enableThinking">Passed to the detection parser so reasoning is not mistaken for content.</param>
        /// <param name="generate">Runs one generation.</param>
        /// <param name="logger">Optional.</param>
        /// <param name="cancellationToken">Checked between rounds and before every tool call.</param>
        public static async IAsyncEnumerable<ChatStreamUpdate> RunAsync(
            string architecture,
            List<ChatMessage> messages,
            SkillRequestPlan plan,
            bool enableThinking,
            SkillChatGeneration generate,
            ILogger logger,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var working = new List<ChatMessage>(messages);
            int maxRounds = Math.Max(1, plan.LoopOptions.MaxRounds);

            int promptTokens = 0;
            int evalTokens = 0;
            int reusedTokens = 0;
            long promptNs = 0;
            long evalNs = 0;
            long totalNs = 0;

            for (int round = 1; round <= maxRounds; round++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // The loop parses the stream itself, because it has to: it is looking
                // for skills_read calls to answer. Having parsed, it forwards the
                // SEPARATED pieces rather than the raw text, so tool markup never
                // reaches the adapter and every round streams as it decodes.
                var parser = OutputParserFactory.Create(architecture);
                parser.Init(enableThinking, plan.Tools);

                var content = new StringBuilder();
                var thinking = new StringBuilder();
                var calls = new List<ToolCall>();
                ChatStreamUpdate terminal = default;
                string toolBeingWritten = null;

                await foreach (ChatStreamUpdate update in
                    generate(working, plan.Tools, cancellationToken).ConfigureAwait(false))
                {
                    if (update.Done)
                    {
                        terminal = update;
                        continue;
                    }
                    if (string.IsNullOrEmpty(update.Piece))
                        continue;

                    ParsedOutput delta = parser.Add(update.Piece, false);
                    Accumulate(delta, content, thinking, calls);

                    // Content and reasoning go out the moment they are decoded. A tool
                    // call does NOT: it is either ours to answer or the caller's to
                    // service, and forwarding it mid-round would commit to an answer the
                    // round has not finished giving.
                    if (!string.IsNullOrEmpty(delta.Content) || !string.IsNullOrEmpty(delta.Thinking))
                        yield return ChatStreamUpdate.Parsed(delta.Content, delta.Thinking, null);

                    // The call's BODY does stream — as progress, not as content. A
                    // shell call can be a whole heredoc written in silence otherwise;
                    // the UI shows "writing code" and the draft, and nothing here is
                    // handed to the client as an answer.
                    toolBeingWritten = delta.ToolCallName ?? toolBeingWritten;
                    if (!string.IsNullOrEmpty(delta.ToolCallText))
                        yield return ChatStreamUpdate.ToolProgress("writing", toolBeingWritten, delta.ToolCallText);
                }

                promptTokens += terminal.PromptTokens;
                evalTokens += terminal.EvalTokens;
                reusedTokens += terminal.KvCacheReusedTokens;
                promptNs += terminal.PromptNs;
                evalNs += terminal.EvalNs;
                totalNs += terminal.TotalNs;

                ParsedOutput flushed = parser.Add(string.Empty, true);
                Accumulate(flushed, content, thinking, calls);
                if (!string.IsNullOrEmpty(flushed.Content) || !string.IsNullOrEmpty(flushed.Thinking))
                    yield return ChatStreamUpdate.Parsed(flushed.Content, flushed.Thinking, null);

                // Three ways, not two. A call that is neither ours nor a tool the CLIENT
                // declared belongs to nobody, and forwarding it was a silent end to the
                // turn: the Web UI declares no tools and has no handler for the tool-call
                // frame, so the reply — thinking, then a tool call, and no content —
                // rendered as nothing at all. Answered here it costs one round.
                SkillTools.Partition(
                    calls, plan.ClientTools,
                    out List<ToolCall> skillCalls,
                    out List<ToolCall> clientCalls,
                    out List<ToolCall> unknownCalls);

                if (skillCalls.Count == 0 && unknownCalls.Count == 0)
                {
                    // Nothing more to fetch. Any tool calls left are the caller's, and
                    // only the caller knows what they do. The progress line still has to
                    // come down: it went up as the call was written, and the caller
                    // servicing it is not something this stream will see.
                    if (clientCalls.Count > 0)
                    {
                        foreach (ToolCall clientCall in clientCalls)
                            yield return ChatStreamUpdate.ToolProgress("finished", clientCall.Name);
                        yield return ChatStreamUpdate.Parsed(string.Empty, null, clientCalls);
                    }

                    yield return Combine(terminal, promptTokens, evalTokens, reusedTokens, promptNs, evalNs, totalNs);
                    yield break;
                }

                working.Add(new ChatMessage
                {
                    Role = "assistant",
                    Content = content.ToString(),
                    Thinking = thinking.Length == 0 ? null : thinking.ToString(),
                    ToolCalls = new List<ToolCall>(calls),
                    // The tokens as GENERATED, so the next round's render reproduces this
                    // one exactly and the live KV cache can be continued rather than
                    // rebuilt. Without it every round after the first re-prefilled the
                    // entire conversation; SkillAgentLoop has always recorded it, and this
                    // loop could not until the terminal update began carrying it.
                    RawOutputTokens = terminal.RawOutputTokens != null
                        ? new List<int>(terminal.RawOutputTokens)
                        : null,
                    RawPromptTrailingWhitespace = terminal.RawPromptTrailingWhitespace,
                });

                foreach (ToolCall unknownCall in unknownCalls)
                {
                    logger?.LogWarning(LogEventIds.SkillToolInvoked,
                        "skills.tool round={Round} tool={Tool} skill={SkillId} path={Path} ok={Ok} bytes={Bytes}",
                        round, unknownCall.Name ?? "-", "-", "-", false, 0);

                    string refusal = SkillTools.DescribeUnknownTool(unknownCall.Name, plan.Tools);
                    working.Add(BuildResult(plan, refusal, unknownCall.Name));

                    // Recorded as an invocation like any other, so the UI's trace shows a
                    // failed step rather than a gap: the round happened, it cost a
                    // generation, and the user watching the trace should see why.
                    lock (plan.Invocations)
                    {
                        plan.Invocations.Add(new SkillToolInvocation(
                            round, unknownCall.Name ?? string.Empty, null, null,
                            Ok: false, refusal.Length));
                    }

                    // The "writing <name>…" line went up while the call was being
                    // generated and only ever comes down on a "finished". Answering the
                    // call without one leaves the user watching a progress line for
                    // something that already happened.
                    yield return ChatStreamUpdate.ToolProgress("finished", unknownCall.Name);
                }

                int executed = 0;
                foreach (ToolCall call in skillCalls)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    if (executed >= plan.LoopOptions.MaxCallsPerRound)
                    {
                        working.Add(BuildResult(plan,
                            $"Error: too many tool calls in one turn; only the first {plan.LoopOptions.MaxCallsPerRound} "
                            + "were answered. Ask for one file at a time."));
                        break;
                    }

                    // Execution runs on a worker so this stream can keep breathing: a
                    // shell call that installs packages holds the request for a
                    // minute or more, and a synchronous call here meant the user's last
                    // sign of life was however much prose preceded it. The heartbeat
                    // updates carry the elapsed time; the adapter turns them into
                    // "running…" frames. Cancellation stops the WAIT — the process
                    // itself is bounded by the runner's own timeout, exactly as it was
                    // when this call blocked the loop.
                    string callDetail = DescribeCall(call);
                    yield return ChatStreamUpdate.ToolProgress("running", call.Name, detail: callDetail);

                    // The tool's own stdout/stderr, tapped live: a pip install's
                    // "Collecting reportlab", the program's prints as it runs. Lines
                    // arrive on the process's reader threads and drain into the
                    // heartbeat frames the user is already watching.
                    var liveOutput = new LiveOutputBuffer();
                    // Acquire BEFORE scheduling. Request cancellation can dispose this
                    // async iterator before the worker even starts; holding the operation
                    // here lets the request lease detach immediately while deferring
                    // workspace deletion until the worker's finally has run.
                    IDisposable workspaceOperation = plan.ToolContext?.Workspace?.BeginOperation();
                    Task<SkillToolResult> execution;
                    try
                    {
                        execution = Task.Run(() =>
                        {
                            using (workspaceOperation)
                                return SkillTools.Execute(call, plan.ToolContext, liveOutput.Add);
                        });
                    }
                    catch
                    {
                        workspaceOperation?.Dispose();
                        throw;
                    }
                    var executionClock = Stopwatch.StartNew();
                    while (await Task.WhenAny(execution, Task.Delay(1000, cancellationToken)).ConfigureAwait(false) != execution)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        yield return ChatStreamUpdate.ToolProgress(
                            "running", call.Name, piece: liveOutput.Drain(),
                            seconds: executionClock.Elapsed.TotalSeconds, detail: callDetail);
                    }
                    SkillToolResult result = await execution.ConfigureAwait(false);

                    // A run shorter than one heartbeat never hit the loop above; its
                    // output still deserves to reach the screen before "finished".
                    string trailingOutput = liveOutput.Drain();
                    if (!string.IsNullOrEmpty(trailingOutput))
                    {
                        yield return ChatStreamUpdate.ToolProgress(
                            "running", call.Name, piece: trailingOutput,
                            seconds: executionClock.Elapsed.TotalSeconds, detail: callDetail);
                    }
                    executed++;

                    var invocation = new SkillToolInvocation(
                        round, call.Name ?? string.Empty, result.SkillId, result.ResourcePath,
                        result.Ok, result.Content?.Length ?? 0)
                    { Files = result.Files };
                    lock (plan.Invocations)
                        plan.Invocations.Add(invocation);

                    logger?.LogInformation(LogEventIds.SkillToolInvoked,
                        "skills.tool round={Round} tool={Tool} skill={SkillId} path={Path} ok={Ok} bytes={Bytes}",
                        round, invocation.Tool, invocation.SkillId ?? "-", invocation.ResourcePath ?? "-",
                        invocation.Ok, invocation.ResultBytes);

                    working.Add(BuildResult(plan, result.Content ?? string.Empty, call.Name));

                    // Yielded AFTER the invocation is recorded, so the adapter's trace
                    // flush (which runs on every update it receives) emits the step's
                    // result frame before the UI is told the live progress line is done.
                    yield return ChatStreamUpdate.ToolProgress(
                        "finished", call.Name, seconds: executionClock.Elapsed.TotalSeconds);
                }
            }

            // Out of rounds. Tell the model so, in the conversation, and let it answer
            // from what it has — the alternative is returning a bare tool call to a
            // client that cannot service it, which shows the user nothing at all.
            logger?.LogWarning(LogEventIds.SkillLoopCapped,
                "skills.loop.capped rounds={Rounds} skills={Skills}", maxRounds, plan.DescribeSelection());

            working.Add(BuildResult(plan,
                "Error: the limit on tool calls for this turn has been reached. Answer now using what you have "
                + "already read, and say which part you could not check."));

            cancellationToken.ThrowIfCancellationRequested();

            // Parsed like every other round. If the model answers the "limit reached"
            // message with yet another skills_read, its markup must still not reach the
            // client — forwarding it raw here would surface a tool call the caller
            // cannot service, which is the exact stall this loop exists to prevent.
            var finalParser = OutputParserFactory.Create(architecture);
            finalParser.Init(enableThinking, plan.Tools);
            var finalCalls = new List<ToolCall>();
            var finalContent = new StringBuilder();
            var finalThinking = new StringBuilder();

            await foreach (ChatStreamUpdate update in
                generate(working, plan.Tools, cancellationToken).ConfigureAwait(false))
            {
                if (!update.Done)
                {
                    if (string.IsNullOrEmpty(update.Piece))
                        continue;
                    ParsedOutput delta = finalParser.Add(update.Piece, false);
                    Accumulate(delta, finalContent, finalThinking, finalCalls);
                    if (!string.IsNullOrEmpty(delta.Content) || !string.IsNullOrEmpty(delta.Thinking))
                        yield return ChatStreamUpdate.Parsed(delta.Content, delta.Thinking, null);
                    if (!string.IsNullOrEmpty(delta.ToolCallText))
                        yield return ChatStreamUpdate.ToolProgress("writing", delta.ToolCallName, delta.ToolCallText);
                    continue;
                }

                ParsedOutput last = finalParser.Add(string.Empty, true);
                Accumulate(last, finalContent, finalThinking, finalCalls);
                if (!string.IsNullOrEmpty(last.Content) || !string.IsNullOrEmpty(last.Thinking))
                    yield return ChatStreamUpdate.Parsed(last.Content, last.Thinking, null);

                // Only the client's own tools may be forwarded here — this is the last
                // round, so a name nobody declared would leave the turn empty with no
                // round left to recover in.
                SkillTools.Partition(
                    finalCalls, plan.ClientTools,
                    out List<ToolCall> stillOurs, out List<ToolCall> pending, out List<ToolCall> stillUnknown);
                foreach (ToolCall dropped in stillOurs.Concat(stillUnknown))
                    yield return ChatStreamUpdate.ToolProgress("finished", dropped.Name);
                if (pending.Count > 0)
                    yield return ChatStreamUpdate.Parsed(string.Empty, null, pending);

                // The model was told to answer and asked for another tool instead, so
                // its whole reply is markup this loop has just dropped and the user
                // would get an empty bubble — the same silence, arrived at from the
                // other end. Say what happened and what DID get made: a turn that
                // produced files is a partial success, and the user cannot tell that
                // from a crash unless someone says so.
                // APPENDED, not substituted, and no longer conditional on the content
                // being empty. Measured on this server's own logs: 6 of the 12 capped
                // turns ended with a NON-empty lead-in and a dropped tool call —
                // "Let me take a different approach and use a simpler Python script:"
                // followed by 6,069 tokens of markup nobody would run. The gate below
                // fired for none of them, so the user was handed a sentence promising
                // work, no work, and no notice that the budget had run out. Five
                // client-abort warnings the same day are consistent with people giving
                // up on exactly that.
                if (stillOurs.Count > 0 || stillUnknown.Count > 0)
                {
                    string exhausted = DescribeExhaustedTurn(plan, stillOurs.Concat(stillUnknown));
                    yield return ChatStreamUpdate.Parsed(
                        finalContent.Length == 0 ? exhausted : "\n\n" + exhausted, null, null);
                }

                yield return Combine(
                    update,
                    promptTokens + update.PromptTokens,
                    evalTokens + update.EvalTokens,
                    reusedTokens + update.KvCacheReusedTokens,
                    promptNs + update.PromptNs,
                    evalNs + update.EvalNs,
                    totalNs + update.TotalNs);
            }
        }

        /// <summary>
        /// Collects a running tool's stdout/stderr lines from the process reader
        /// threads until the loop's heartbeat drains them into a progress frame.
        ///
        /// <para>
        /// Bounded twice, because the lines come from code the MODEL wrote: the buffer
        /// stops accepting past 64 KB (a print loop must not grow the heap between
        /// drains), and <see cref="Drain"/> stops forwarding past 8 KB total — the live
        /// stream is a window onto the run, not a transcript; the full (32 KB-capped)
        /// output still arrives in the tool result.
        /// </para>
        /// </summary>
        private sealed class LiveOutputBuffer
        {
            private const int MaxBufferedChars = 64 * 1024;
            private const int MaxForwardedChars = 8 * 1024;

            private readonly StringBuilder _pending = new();
            private int _forwarded;
            private bool _truncationReported;

            public void Add(string line)
            {
                lock (_pending)
                {
                    if (_pending.Length < MaxBufferedChars)
                        _pending.Append(line).Append('\n');
                }
            }

            /// <summary>Everything buffered since the last drain, or null when quiet.</summary>
            public string Drain()
            {
                string chunk;
                lock (_pending)
                {
                    if (_pending.Length == 0)
                        return null;
                    chunk = _pending.ToString();
                    _pending.Clear();
                }

                if (_forwarded >= MaxForwardedChars)
                {
                    if (_truncationReported)
                        return null;
                    _truncationReported = true;
                    return "…[further live output not shown; the full output arrives with the result]\n";
                }

                if (_forwarded + chunk.Length > MaxForwardedChars)
                    chunk = chunk.Substring(0, MaxForwardedChars - _forwarded);
                _forwarded += chunk.Length;
                return chunk;
            }
        }

        /// <summary>
        /// One line saying what a tool call is about to do, for the live progress the
        /// user watches while it runs: the command of a <c>shell</c> call, the script and
        /// arguments of a <c>skills_run</c>, the file of a <c>skills_read</c>. Never the
        /// full payload — that streams separately as the call is written.
        ///
        /// <para>
        /// Keyed off the shared name constants rather than string literals. A literal
        /// here does not fail to compile when a tool is renamed; it just quietly stops
        /// matching, and the user is left watching a live line that never says what is
        /// running. That is exactly how this switch went stale once already.
        /// </para>
        /// </summary>
        private static string DescribeCall(ToolCall call)
        {
            if (call?.Arguments == null)
                return null;

            string Arg(string key) =>
                call.Arguments.TryGetValue(key, out object v) && v != null
                    ? (v as string ?? v.ToString())
                    : null;

            // The command itself is the label a user wants: "pip install pandas" says
            // more than "shell · 240 chars" ever could.
            if (string.Equals(call.Name, SkillToolNames.Shell, StringComparison.Ordinal))
            {
                // The RAW value, not Arg()'s string: a Codex-trained model sends command as
                // an argv array, and ToString() on the list yields "System.Object[]" — which
                // is what the user then watched for the whole minute the command ran.
                // ReadCommand understands every shape the runner accepts, so the live line
                // shows exactly what is running.
                object raw = call.Arguments.TryGetValue("command", out object v) ? v : null;
                return Truncate(ShellCommand.Summarize(ShellCommand.ReadCommand(raw)), 64);
            }

            if (SkillToolNames.IsApplyPatchAlias(call.Name)
                || string.Equals(call.Name, SkillToolNames.ApplyPatch, StringComparison.Ordinal))
            {
                int patchChars = Arg("patch")?.Length ?? 0;
                return patchChars.ToString(CultureInfo.InvariantCulture) + " chars";
            }

            switch (call.Name)
            {
                case "skills_run":
                    string script = Arg("path") ?? Arg("script");
                    string args = Arg("args");
                    return script == null ? null : script + (string.IsNullOrEmpty(args) ? "" : " " + args);

                case "skills_read":
                    string skill = Arg("skill");
                    string path = Arg("path") ?? "SKILL.md";
                    return skill == null ? path : skill + "/" + path;

                default:
                    return null;
            }
        }

        /// <summary>
        /// What to tell the USER when the turn ran out of tool calls with the model still
        /// working. Never empty, and never only an apology: the files the turn did
        /// produce are the part worth keeping, and naming them is what makes "ran out of
        /// budget" different from "nothing happened".
        /// </summary>
        private static string DescribeExhaustedTurn(SkillRequestPlan plan, IEnumerable<ToolCall> wanted)
        {
            var sb = new StringBuilder();
            sb.Append("I ran out of the tool-call budget for this turn while still working");

            string[] names;
            lock (plan.Invocations)
                names = plan.Invocations.SelectMany(i => i.Files).Select(f => f.Name).Distinct().ToArray();

            if (names.Length > 0)
            {
                sb.Append(". Finished so far: ").Append(string.Join(", ", names));
            }

            string[] next = wanted.Select(c => c.Name).Where(n => !string.IsNullOrEmpty(n)).Distinct().ToArray();
            if (next.Length > 0)
                sb.Append(". I was about to call ").Append(string.Join(", ", next)).Append(" again");

            sb.Append(". Send \"continue\" and I will pick up from here, or raise --skills-max-rounds to give a "
                      + "turn more steps.");
            return sb.ToString();
        }

        private static string Truncate(string text, int max) =>
            text.Length <= max ? text : text.Substring(0, max - 1) + "\u2026";

        /// <summary>Fold one parser delta into the round's running totals.</summary>
        private static void Accumulate(
            ParsedOutput delta, StringBuilder content, StringBuilder thinking, List<ToolCall> calls)
        {
            if (delta == null)
                return;
            if (!string.IsNullOrEmpty(delta.Content))
                content.Append(delta.Content);
            if (!string.IsNullOrEmpty(delta.Thinking))
                thinking.Append(delta.Thinking);
            if (delta.ToolCalls is { Count: > 0 })
                calls.AddRange(delta.ToolCalls);
        }

        private static ChatStreamUpdate Combine(
            ChatStreamUpdate terminal,
            int promptTokens,
            int evalTokens,
            int reusedTokens,
            long promptNs,
            long evalNs,
            long totalNs) =>
            new(string.Empty, true, promptTokens, evalTokens, reusedTokens,
                totalNs, promptNs, evalNs, terminal.FinishReason ?? "stop")
            {
                RawOutputTokens = terminal.RawOutputTokens,
                RawPromptTrailingWhitespace = terminal.RawPromptTrailingWhitespace,
            };

        /// <summary>
        /// Wrap a tool result in the message shape this model family renders. Mistral 3
        /// drops <c>role: "tool"</c> messages outright, so on that family the result is
        /// fed back as a user turn instead of vanishing from the prompt.
        /// </summary>
        private static ChatMessage BuildResult(SkillRequestPlan plan, string content, string tool = null)
        {
            if (plan.LoopOptions.ToolResultsAreRendered)
                return new ChatMessage { Role = "tool", Content = content };

            string prefix = tool == null
                ? "Result of the skill lookup you requested:"
                : $"Result of your {tool} call:";
            return new ChatMessage { Role = "user", Content = prefix + "\n\n" + content };
        }
    }
}
