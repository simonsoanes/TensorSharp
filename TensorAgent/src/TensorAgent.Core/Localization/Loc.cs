// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Globalization;
using TensorAgent.Core.Settings;
using TensorAgent.Sharing.Localization;

namespace TensorAgent.Core.Localization;

/// <summary>
/// The language of the app's interface, and its strings.
///
/// <para>
/// One language for the whole process, because the app has one interface: the native
/// pages, the messages the host hands to the page, and the page itself (served the same
/// strings as <c>/i18n.js</c>, see <see cref="PageStrings"/>) all read from here. The
/// tables are <c>Localization/&lt;tag&gt;/&lt;area&gt;.json</c> in this assembly; every key
/// in <c>&lt;area&gt;.json</c> begins with <c>&lt;area&gt;.</c>, English is the source, and
/// the tests keep every other language complete and every key in use.
/// </para>
/// <para>
/// Only the interface. What the model reads -- prompts, tool results, a refusal it is meant
/// to act on -- stays English whatever the user picks, and so does anything the page
/// compares against such text, and the logs.
/// </para>
/// </summary>
public static class Loc
{
    private static readonly StringCatalog Catalog = new(typeof(Loc).Assembly, "TensorAgent.Core.Localization.");

    /// <summary>
    /// The system's preferred languages, most preferred first.
    ///
    /// <para>
    /// The app head installs the device's list at startup (NSLocale on iOS and the Mac,
    /// the user's languages on Windows). Anywhere else -- the tests, the tools -- nothing is
    /// preferred, so a host nobody told otherwise speaks English rather than whatever the
    /// machine running it happens to be set to.
    /// </para>
    /// </summary>
    public static Func<IReadOnlyList<string>> SystemLanguages { get; set; } = () => Array.Empty<string>();

    /// <summary>Raised after the language changes, on the thread that changed it.</summary>
    public static event Action? Changed
    {
        add => Catalog.Changed += value;
        remove => Catalog.Changed -= value;
    }

    public static UiLanguage Language => Catalog.Language;

    /// <summary>The culture to format numbers and dates in for the interface. Never set as the
    /// thread's culture: the engine parses numbers, and must go on parsing them the same way.</summary>
    public static CultureInfo Culture => Catalog.Culture;

    /// <summary>Every key in the current language, English filling any gap.</summary>
    public static IReadOnlyDictionary<string, string> Strings => Catalog.Strings;

    /// <summary>The language and its strings from one table, for a consistent page response.</summary>
    internal static (UiLanguage Language, IReadOnlyDictionary<string, string> Strings) Snapshot() => Catalog.Snapshot();

    /// <summary>The catalog itself, for the tests that keep the translations complete.</summary>
    internal static StringCatalog Tables => Catalog;

    /// <summary>The text for a key, with its <c>{name}</c> placeholders filled.</summary>
    public static string T(string key, params (string Name, object? Value)[] args) => Catalog.T(key, args);

    /// <summary>Counted text: <c>key.one</c> or <c>key.other</c> by the language's rule, the count as <c>{count}</c>.</summary>
    public static string Plural(string key, long count, params (string Name, object? Value)[] args) =>
        Catalog.Plural(key, count, args);

    /// <summary>The language these settings ask for: the user's choice, else the system's, else English.</summary>
    public static UiLanguage Resolve(AppSettings settings) =>
        UiLanguages.Resolve(settings.UiLanguage, SystemLanguages());

    /// <summary>Switch the interface to the language these settings ask for.</summary>
    public static UiLanguage Apply(AppSettings settings)
    {
        UiLanguage language = Resolve(settings);
        Use(language);
        return language;
    }

    /// <summary>
    /// Switch to a language directly (the tests put English back afterwards). The share
    /// strings move first: the app writes the drafts of what was shared with them, and
    /// <see cref="Changed"/> repaints what shows those drafts.
    /// </summary>
    public static void Use(UiLanguage language)
    {
        ShareStrings.Use(language);
        Catalog.Use(language);
    }
}
