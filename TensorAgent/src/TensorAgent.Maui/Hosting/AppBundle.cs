// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorAgent.Maui.Hosting;

/// <summary>
/// Where the files the project ships beside the app (the Web UI under <c>webui/</c>, the
/// skills under <c>skills/</c>) are found at run time.
/// </summary>
internal static class AppBundle
{
    /// <summary>
    /// The directory the csproj's bundled files land in: the root of an iOS bundle,
    /// <c>Contents/Resources</c> of a Mac one, and the output directory beside the
    /// executable on Windows. The bundle PATH is not it on a Mac, which is the
    /// difference that matters.
    /// </summary>
    public static string ResourceDirectory =>
#if IOS || MACCATALYST
        Foundation.NSBundle.MainBundle.ResourcePath ?? Foundation.NSBundle.MainBundle.BundlePath;
#else
        AppContext.BaseDirectory;
#endif
}
