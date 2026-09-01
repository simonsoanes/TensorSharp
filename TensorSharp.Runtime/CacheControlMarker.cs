// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorSharp.Runtime
{
    /// <summary>
    /// An explicit prompt-cache breakpoint attached to a message, a content part
    /// or a tool declaration. Its presence is the whole signal — it marks the end
    /// of a prefix the client wants kept in the prefix cache.
    /// </summary>
    public class CacheControlMarker
    {
        /// <summary>
        /// The marker kind. Only <c>"ephemeral"</c> is currently specified, and
        /// unrecognised kinds are carried through verbatim rather than rejected
        /// so a newer client spelling degrades to a plain breakpoint.
        /// </summary>
        public string Type { get; set; } = "ephemeral";
    }
}
