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
using System.Runtime.Versioning;

namespace TensorSharp.AgentHost
{
    /// <summary>
    /// The two host questions a single <see cref="OperatingSystem"/> check answers
    /// wrongly for a Mac Catalyst app. .NET reports Catalyst as iOS
    /// (<see cref="OperatingSystem.IsIOS"/> is true there) and not as macOS, yet the
    /// process is an ordinary macOS desktop process: it can start children, Seatbelt
    /// confines them and Darwin's ABI applies. A check written as <c>IsMacOS()</c> or
    /// <c>IsIOS()</c> therefore treats the desktop TensorAgent like a phone.
    /// </summary>
    internal static class HostOS
    {
        /// <summary>macOS, including a Mac Catalyst app: Seatbelt, posix_spawn, Darwin's ABI.</summary>
        [SupportedOSPlatformGuard("macos")]
        [SupportedOSPlatformGuard("maccatalyst")]
        public static bool IsMacDesktop => OperatingSystem.IsMacOS() || OperatingSystem.IsMacCatalyst();

        /// <summary>iPhone, iPad and Apple TV, where code runs inside the app and never in a child process.</summary>
        public static bool IsAppleMobile =>
            (OperatingSystem.IsIOS() && !OperatingSystem.IsMacCatalyst()) || OperatingSystem.IsTvOS();
    }
}
