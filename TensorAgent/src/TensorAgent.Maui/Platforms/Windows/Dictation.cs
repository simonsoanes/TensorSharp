// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Localization;

namespace TensorAgent.Maui.Services;

/// <summary>
/// Dictation on Windows: not offered by the app. Windows' own voice typing
/// (Windows+H) already writes into any focused text field, the message box included,
/// and the only recogniser Windows gives an app sends audio to a server unless the
/// user has installed offline speech packs -- which the Apple implementation refuses
/// to do on the user's behalf (see Services/Apple/Dictation.cs).
/// </summary>
internal sealed class Dictation : IDisposable
{
    /// <summary>The marker the Apple implementation puts on a refusal only Settings can lift.</summary>
    public const string DeniedMarker = "[denied]";

    public Dictation(string? language = null)
    {
    }

    /// <summary>Always false: the page is told to use Windows voice typing instead.</summary>
    public static bool IsSupported => false;

    /// <summary>What the page says instead of listening.</summary>
    public static string UnsupportedMessage => Loc.T("app.dictation.unsupportedWindows");

    public static Task<string?> RequestPermissionsAsync() => Task.FromResult<string?>(UnsupportedMessage);

    public Task<string> ListenAsync() => Task.FromResult(string.Empty);

    public void Stop()
    {
    }

    public void Dispose()
    {
    }
}
