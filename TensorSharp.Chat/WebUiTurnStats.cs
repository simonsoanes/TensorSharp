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

namespace TensorSharp.Chat
{
    /// <summary>
    /// The token count and speed a Web UI <c>done</c> frame reports for a finished turn.
    /// </summary>
    /// <remarks>
    /// The frame used to report the number of ANSWER pieces streamed, divided by the
    /// turn's wall-clock time. For a plain reply that is close enough. For a turn that
    /// used tools it is not: the reasoning and the tool calls never become answer
    /// pieces, while the wall clock includes every generation round and every tool run.
    /// Measured 2026-09-30 on Qwen3.8 27B (Metal) with --code-exec: the page reported
    /// "464 tokens · 226.7s · 2.0 tok/s" for a turn whose KV cache had grown by about
    /// 13,800 tokens. The engine already counts what it generated and how long it spent
    /// decoding (<see cref="TensorSharp.Server.ChatStreamUpdate.EvalTokens"/> and
    /// <see cref="TensorSharp.Server.ChatStreamUpdate.EvalNs"/>, summed across rounds by
    /// the skills loop), so the frame reports those.
    /// </remarks>
    internal static class WebUiTurnStats
    {
        /// <summary>
        /// What to report: every token the model generated this turn (reasoning, tool
        /// calls and answer, across all rounds), and its decode speed. Falls back to the
        /// streamed answer pieces over the wall clock when the engine's counts never
        /// arrived, as happens when a turn is aborted before its terminal update.
        /// </summary>
        /// <param name="evalTokens">Tokens the engine generated, summed over the turn's generations.</param>
        /// <param name="evalNs">Nanoseconds the engine spent decoding them.</param>
        /// <param name="streamedPieces">Answer pieces streamed to the page.</param>
        /// <param name="elapsedSeconds">The turn's wall-clock time.</param>
        public static (int Tokens, double TokensPerSecond) Summarize(
            long evalTokens, long evalNs, int streamedPieces, double elapsedSeconds)
        {
            if (evalTokens > 0)
            {
                int tokens = (int)Math.Min(evalTokens, int.MaxValue);
                if (evalNs > 0)
                    return (tokens, evalTokens / (evalNs / 1e9));
                return (tokens, elapsedSeconds > 0 ? evalTokens / elapsedSeconds : 0);
            }

            return (streamedPieces, streamedPieces > 0 && elapsedSeconds > 0 ? streamedPieces / elapsedSeconds : 0);
        }
    }
}
