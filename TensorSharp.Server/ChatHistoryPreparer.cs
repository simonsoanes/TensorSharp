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
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace TensorSharp.Server
{
    internal static class ChatHistoryPreparer
    {
        public static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch)
            => PrepareHistoryForInference(history, arch, NullLogger.Instance);

        public static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch, ILogger logger)
        {
            if (history == null || history.Count == 0)
                return history;

            List<ChatMessage> prepared = null;
            for (int i = 0; i < history.Count; i++)
            {
                var normalized = NormalizeMessageForInference(history[i], arch, logger);
                if (ReferenceEquals(normalized, history[i]))
                    continue;

                prepared ??= new List<ChatMessage>(history);
                prepared[i] = normalized;
            }

            return prepared ?? history;
        }

        /// <summary>
        /// Rebuild <paramref name="incoming"/> so its render reproduces the token
        /// sequence the live KV cache already holds.
        ///
        /// <para>
        /// Two mechanisms, walked with independent cursors over the two lists. A plain
        /// assistant turn gets the tracked <see cref="ChatMessage.RawOutputTokens"/>
        /// spliced onto it, exactly as before. A turn that ran the in-process tool loop
        /// is harder: the client sends ONE clean assistant message, but the cache holds
        /// the whole loop transcript — assistant round, tool result, assistant round,
        /// ... — so a positional walk breaks at the first tool result and the next
        /// render diverged thousands of tokens before the cache's end. The engine only
        /// rewinds a few trailing tokens (see BatchExecutor.MaxLiveContinuationRewindTokens),
        /// so that turn re-prefilled the entire conversation: measured on the pdf-skill
        /// incident, the follow-up turn re-prefilled 15.7k tokens (10.7s to first token)
        /// where 99% was reusable. The fix is EXPANSION: when the tracked history holds
        /// a tool transcript where the client sent one assistant message, substitute the
        /// transcript — raw tokens and all — so the rendered prefix stays byte-identical
        /// to the cache.
        /// </para>
        /// </summary>
        public static List<ChatMessage> AugmentWithCachedRawTokens(List<ChatMessage> incoming, IReadOnlyList<ChatMessage> trackedHistory)
        {
            if (incoming == null)
                return null;

            var result = new List<ChatMessage>(incoming.Count);
            int t = 0;
            bool diverged = trackedHistory == null || trackedHistory.Count == 0;

            for (int i = 0; i < incoming.Count; i++)
            {
                ChatMessage src = incoming[i];

                if (diverged || t >= trackedHistory.Count || src.Role != trackedHistory[t].Role)
                {
                    // Past the tracked prefix, or the conversation genuinely diverges
                    // here (an edited or regenerated turn): everything from this point
                    // renders from the client's own content.
                    diverged = true;
                    result.Add(src);
                    continue;
                }

                ChatMessage tracked = trackedHistory[t];

                if (src.Role != "assistant")
                {
                    // Compare on Content for non-assistant roles only. Assistant content
                    // can be legitimately altered by the streaming output parser between
                    // turns.
                    if (!string.Equals(src.Content ?? string.Empty, tracked.Content ?? string.Empty, StringComparison.Ordinal))
                    {
                        diverged = true;
                        result.Add(src);
                        continue;
                    }

                    result.Add(src);
                    t++;
                    continue;
                }

                string nextIncomingRole = i + 1 < incoming.Count ? incoming[i + 1].Role : null;
                if (TryMatchToolTranscript(trackedHistory, t, src, nextIncomingRole, out int runLength))
                {
                    var expanded = new List<ChatMessage>(runLength);
                    for (int k = 0; k < runLength; k++)
                        expanded.Add(CloneShallow(trackedHistory[t + k]));
                    PreserveCollapsedCacheMarkers(src, expanded);
                    result.AddRange(expanded);
                    t += runLength;
                    continue;
                }

                bool useTracked = tracked.RawOutputTokens is { Count: > 0 }
                    && (src.RawOutputTokens == null || src.RawOutputTokens.Count == 0);

                if (useTracked)
                {
                    result.Add(new ChatMessage
                    {
                        Role = src.Role,
                        Content = src.Content,
                        ImagePaths = src.ImagePaths,
                        AudioPaths = src.AudioPaths,
                        TextFilePaths = src.TextFilePaths,
                        TextFileNames = src.TextFileNames,
                        IsVideo = src.IsVideo,
                        ToolCalls = src.ToolCalls,
                        ToolCallId = src.ToolCallId,
                        Thinking = src.Thinking,
                        RawOutputTokens = tracked.RawOutputTokens,
                        CacheControl = src.CacheControl,
                        ContentCacheBreakpoints = src.ContentCacheBreakpoints,
                    });
                }
                else
                {
                    result.Add(src);
                }
                t++;
            }
            return result;
        }

        /// <summary>Map cache markers from a client-visible assistant message onto
        /// the tracked assistant/tool transcript that replaces it. A message-level
        /// marker belongs after the whole expansion. Content offsets are mapped over
        /// the intermediate assistant text that TryMatchToolTranscript verified; if
        /// an offset reaches the final raw model round, mark its start conservatively
        /// because parser-stripped thinking text makes the exact offset unknowable.</summary>
        private static void PreserveCollapsedCacheMarkers(
            ChatMessage source, List<ChatMessage> expanded)
        {
            if (expanded.Count == 0) return;

            int lastAssistant = expanded.FindLastIndex(m => m.Role == "assistant");
            if (lastAssistant < 0) return;

            if (source.CacheControl != null)
            {
                expanded[lastAssistant].CacheControl = new CacheControlMarker
                {
                    Type = source.CacheControl.Type,
                };
            }

            if (source.ContentCacheBreakpoints == null
                || source.ContentCacheBreakpoints.Count == 0)
                return;

            foreach (int rawOffset in source.ContentCacheBreakpoints)
            {
                int remaining = Math.Max(0, rawOffset);
                bool mapped = false;
                for (int i = 0; i < lastAssistant; i++)
                {
                    if (expanded[i].Role != "assistant") continue;
                    int length = expanded[i].Content?.Length ?? 0;
                    if (remaining <= length)
                    {
                        AddMappedContentCacheBreakpoint(expanded[i], remaining);
                        mapped = true;
                        break;
                    }
                    remaining -= length;
                }

                if (!mapped)
                    AddMappedContentCacheBreakpoint(expanded[lastAssistant], 0);
            }
        }

        private static void AddMappedContentCacheBreakpoint(ChatMessage message, int offset)
        {
            message.AddContentCacheBreakpoint(offset);
            message.ContentCacheBreakpoints.Sort();
        }

        /// <summary>
        /// Does the tracked history hold, at <paramref name="start"/>, an in-process
        /// tool-loop transcript that <paramref name="src"/> is the clean client-visible
        /// form of? A transcript is a contiguous run of assistant and <c>role: "tool"</c>
        /// messages ending on an assistant, with at least one tool result — the shape
        /// the skills/code loop leaves behind. Verified by content: each intermediate
        /// round's parsed content concatenates, in order, into a prefix of what the
        /// client sent back (the final round's tracked content is the RAW model text and
        /// is not compared, the same tolerance the plain splice has always had).
        ///
        /// <para>
        /// Families that feed tool results back as user turns (Mistral 3) never form
        /// this shape, so they keep today's behavior — a conservative miss, never a
        /// wrong splice.
        /// </para>
        /// </summary>
        private static bool TryMatchToolTranscript(
            IReadOnlyList<ChatMessage> trackedHistory, int start, ChatMessage src,
            string nextIncomingRole, out int runLength)
        {
            runLength = 0;

            // A client whose NEXT message is a tool result is carrying the transcript
            // itself (an OpenAI-style tool flow): every message lines up one-to-one,
            // and expanding here would insert the tracked rounds a second time.
            if (nextIncomingRole == "tool")
                return false;

            // The run: assistant, then tool results and further assistant rounds, up to
            // the last assistant before anything else (the next turn's user message).
            int end = start;
            int lastAssistant = start;
            bool sawTool = false;
            while (end < trackedHistory.Count)
            {
                string role = trackedHistory[end].Role;
                if (role == "assistant")
                    lastAssistant = end;
                else if (role == "tool")
                    sawTool = true;
                else
                    break;
                end++;
            }

            if (!sawTool || lastAssistant == start)
                return false;

            // The client message must not itself be a tool-aware transcript: a caller
            // that sends its own tool messages is matched positionally, not expanded.
            if (src.RawOutputTokens is { Count: > 0 })
                return false;

            // Sanity: the intermediate rounds' parsed content must lead the client's
            // concatenated text. This is what separates "the same turn, re-sent clean"
            // from "a different conversation that happens to align" (an edited turn
            // falls through to the plain splice and diverges there, as before).
            string clientContent = src.Content ?? string.Empty;
            int offset = 0;
            for (int k = start; k < lastAssistant; k++)
            {
                if (trackedHistory[k].Role != "assistant")
                    continue;
                string piece = trackedHistory[k].Content ?? string.Empty;
                if (piece.Length == 0)
                    continue;
                if (offset + piece.Length > clientContent.Length
                    || string.CompareOrdinal(clientContent, offset, piece, 0, piece.Length) != 0)
                    return false;
                offset += piece.Length;
            }

            runLength = lastAssistant - start + 1;
            return true;
        }

        public static void UpdateTrackedHistory(
            List<ChatMessage> trackedHistory,
            List<ChatMessage> incomingHistory,
            string assistantText,
            List<int> generatedTokens)
        {
            trackedHistory.Clear();
            if (incomingHistory != null)
            {
                for (int i = 0; i < incomingHistory.Count; i++)
                    trackedHistory.Add(CloneShallow(incomingHistory[i]));
            }

            trackedHistory.Add(new ChatMessage
            {
                Role = "assistant",
                Content = assistantText,
                RawOutputTokens = generatedTokens,
            });
        }

        public static bool HasMultimodalContent(ChatMessage msg)
        {
            if (msg == null) return false;
            return (msg.ImagePaths != null && msg.ImagePaths.Count > 0) ||
                   (msg.AudioPaths != null && msg.AudioPaths.Count > 0);
        }

        public static bool HasMultimodalContent(List<ChatMessage> history)
        {
            if (history == null || history.Count == 0)
                return false;

            return history.Any(HasMultimodalContent);
        }

        public static List<string> GetImagePathsInPromptOrder(List<ChatMessage> history)
        {
            var imagePaths = new List<string>();
            if (history == null)
                return imagePaths;

            foreach (var msg in history)
            {
                if (msg.ImagePaths == null)
                    continue;

                foreach (var path in msg.ImagePaths)
                {
                    if (!string.IsNullOrEmpty(path))
                        imagePaths.Add(path);
                }
            }

            return imagePaths;
        }

        private static ChatMessage NormalizeMessageForInference(ChatMessage msg, string arch, ILogger logger)
        {
            int maxVideoFrames = MediaHelper.GetConfiguredMaxVideoFrames();
            // maxVideoFrames <= 0 means "no cap" (pure time-based extraction); leave history untouched.
            // Whether a family expands a video into per-frame images (and so needs the
            // cap) is declared with the rest of its chat protocol, not matched on here.
            bool capsFrames = ChatProtocolRegistry.For(arch)?.CapsVideoFrames ?? false;
            if (!capsFrames || maxVideoFrames <= 0 || !msg.IsVideo || msg.ImagePaths == null || msg.ImagePaths.Count <= maxVideoFrames)
                return msg;

            var sampled = MediaHelper.SelectEvenlySpacedIndices(msg.ImagePaths.Count, maxVideoFrames)
                .Select(i => msg.ImagePaths[i])
                .ToList();

            // A Warning, not an Information: frames the user sent are being thrown away,
            // and the answer may miss what happened between the kept ones.
            (logger ?? NullLogger.Instance).LogWarning(LogEventIds.VideoFrameDownsample,
                "video.downsample originalFrames={OriginalFrames} sampledFrames={SampledFrames} architecture={Architecture}: " +
                "the video exceeds the per-message frame cap, so only the sampled, evenly spaced frames reach the model " +
                "and detail between them is lost. Raise VIDEO_MAX_FRAMES to keep more.",
                msg.ImagePaths.Count, sampled.Count, arch);

            return new ChatMessage
            {
                Role = msg.Role,
                Content = msg.Content,
                ImagePaths = sampled,
                AudioPaths = msg.AudioPaths != null ? new List<string>(msg.AudioPaths) : null,
                TextFilePaths = msg.TextFilePaths != null ? new List<string>(msg.TextFilePaths) : null,
                TextFileNames = msg.TextFileNames != null ? new List<string>(msg.TextFileNames) : null,
                IsVideo = msg.IsVideo,
                ToolCalls = msg.ToolCalls,
                ToolCallId = msg.ToolCallId,
                Thinking = msg.Thinking,
                RawOutputTokens = msg.RawOutputTokens,
                CacheControl = msg.CacheControl,
                ContentCacheBreakpoints = msg.ContentCacheBreakpoints != null
                    ? new List<int>(msg.ContentCacheBreakpoints)
                    : null,
            };
        }

        private static ChatMessage CloneShallow(ChatMessage src)
        {
            return new ChatMessage
            {
                Role = src.Role,
                Content = src.Content,
                ImagePaths = src.ImagePaths,
                AudioPaths = src.AudioPaths,
                TextFilePaths = src.TextFilePaths,
                TextFileNames = src.TextFileNames,
                IsVideo = src.IsVideo,
                ToolCalls = src.ToolCalls,
                ToolCallId = src.ToolCallId,
                Thinking = src.Thinking,
                RawOutputTokens = src.RawOutputTokens,
                CacheControl = src.CacheControl != null
                    ? new CacheControlMarker { Type = src.CacheControl.Type }
                    : null,
                ContentCacheBreakpoints = src.ContentCacheBreakpoints != null
                    ? new List<int>(src.ContentCacheBreakpoints)
                    : null,
            };
        }
    }
}
