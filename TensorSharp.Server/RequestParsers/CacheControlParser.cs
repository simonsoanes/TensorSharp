// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using TensorSharp.Runtime;

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Reads an explicit prompt-cache breakpoint off a message, a content part
    /// or a tool declaration. Two spellings are accepted in the same slot:
    /// <c>"cache_control": {"type": "ephemeral"}</c> (Anthropic/DashScope shape)
    /// and the bare <c>"prompt_cache_breakpoint": true</c> flag.
    /// <para>
    /// Like the rest of <c>RequestParsers</c>, nothing here throws: a field of an
    /// unexpected kind means "no marker", not a failed request.
    /// </para>
    /// </summary>
    internal static class CacheControlParser
    {
        /// <summary>
        /// True when <paramref name="el"/> carries a cache breakpoint, with
        /// <paramref name="marker"/> set to the parsed marker.
        /// </summary>
        public static bool TryParse(JsonElement el, out CacheControlMarker marker)
        {
            marker = null;
            if (el.ValueKind != JsonValueKind.Object)
                return false;

            if (el.TryGetProperty("cache_control", out var ccEl) && ccEl.ValueKind == JsonValueKind.Object)
            {
                marker = new CacheControlMarker { Type = ReadType(ccEl) };
                return true;
            }

            if (el.TryGetProperty("prompt_cache_breakpoint", out var flagEl) && flagEl.ValueKind == JsonValueKind.True)
            {
                marker = new CacheControlMarker();
                return true;
            }

            return false;
        }

        /// <summary>
        /// The marker's <c>type</c>, defaulting to <c>"ephemeral"</c> when absent
        /// or not a string — every currently specified marker is ephemeral, and a
        /// null <see cref="CacheControlMarker.Type"/> would only push the check
        /// onto every consumer.
        /// </summary>
        private static string ReadType(JsonElement ccEl)
        {
            if (ccEl.TryGetProperty("type", out var typeEl) &&
                typeEl.ValueKind == JsonValueKind.String)
            {
                string type = typeEl.GetString();
                if (!string.IsNullOrEmpty(type))
                    return type;
            }
            return "ephemeral";
        }
    }
}
