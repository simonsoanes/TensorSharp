// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorAgent.Core.Hosting;

/// <summary>
/// What kind of machine a host runs on: the phone, whose memory budget was measured
/// against a jetsam limit, or a desktop with memory to spare.
/// </summary>
public enum DeviceClass
{
    /// <summary>
    /// The measured iPhone budget (<see cref="EngineMemoryPolicy"/>) and the phone's
    /// settings defaults. The default for every host, including the Mac-side validation
    /// launcher and the benchmarks that emulate the phone.
    /// </summary>
    Phone,

    /// <summary>
    /// The desktop app on macOS and Windows: the engine's own defaults, which are written
    /// for a machine with memory to spare, size the caches and decide what is kept for
    /// reuse, and a first launch starts from <see cref="Settings.AppSettings.DesktopDefaults"/>.
    /// </summary>
    Desktop,
}
