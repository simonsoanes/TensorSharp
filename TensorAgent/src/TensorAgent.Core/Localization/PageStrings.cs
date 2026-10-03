// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text;
using System.Text.Json;

namespace TensorAgent.Core.Localization;

/// <summary>
/// The page's strings, as the script <c>/i18n.js</c>: the current language's table as one
/// line of data, then the small runtime (<c>WebUi/i18n.js</c>) that reads it.
///
/// <para>
/// A script rather than a fetch so the strings are there before the page's own script
/// runs: the markup is translated, and the first thing painted is in the right language.
/// The tag is spliced in ahead of every other script (see the loopback server), and its
/// URL carries the language, so a WebView reloaded after the user switches can never paint
/// with the table it cached for the previous one.
/// </para>
/// </summary>
internal static class PageStrings
{
    public const string ScriptName = "i18n.js";

    private static readonly Lazy<string> Runtime = new(() =>
    {
        using Stream? stream = typeof(PageStrings).Assembly.GetManifestResourceStream("TensorAgent.Core.WebUi.i18n.js");
        if (stream is null)
        {
            throw new InvalidOperationException(
                "TensorAgent.Core.WebUi.i18n.js is not embedded in the assembly; "
                + "check the EmbeddedResource item in TensorAgent.Core.csproj.");
        }
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    });

    /// <summary>The script tag the page gets, ahead of every other script.</summary>
    public static string Tag() =>
        "<script src=\"/" + ScriptName + "?lang=" + Uri.EscapeDataString(Loc.Language.Tag)
        + "\" onerror=\"document.getElementById('tensoragent-language-loading').remove()\"></script>";

    /// <summary>The script: the current language's strings, then the runtime that reads them.</summary>
    public static string Script()
    {
        var snapshot = Loc.Snapshot();
        var strings = new SortedDictionary<string, string>(StringComparer.Ordinal);
        foreach ((string key, string value) in snapshot.Strings)
            strings[key] = value;
        // Written by hand rather than serialized: nothing here depends on reflection, which
        // the phone's trimmed build is the place to discover is missing.
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteString("lang", snapshot.Language.Tag);
            writer.WriteStartObject("strings");
            foreach ((string key, string value) in strings)
                writer.WriteString(key, value);
            writer.WriteEndObject();
            writer.WriteEndObject();
        }
        return "window.TensorAgentI18n = " + Encoding.UTF8.GetString(buffer.ToArray()) + ";\n" + Runtime.Value;
    }
}
