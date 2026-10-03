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

namespace TensorAgent.Sharing.Localization;

/// <summary>
/// The share extension's strings, and the language it shows them in.
///
/// <para>
/// The extension cannot read the app's settings -- they live in the app's own container --
/// and it cannot carry the app's string tables, which come with the inference engine's
/// assembly. So its few strings are filed here (<c>Localization/&lt;tag&gt;/share.json</c>),
/// and the app leaves the user's language choice where both can reach it: a one-line file
/// at the root of the App Group container (<see cref="ChoiceFileName"/>). The extension
/// resolves that choice exactly as the app does, against the same system languages, so
/// "follow the system" means the same in both.
/// </para>
/// </summary>
public static class ShareStrings
{
    /// <summary>The file, at the root of the App Group container, holding the user's choice: a tag, or nothing to follow the system.</summary>
    public const string ChoiceFileName = "ui-language.txt";

    private static readonly StringCatalog Catalog =
        new(typeof(ShareStrings).Assembly, "TensorAgent.Sharing.Localization.");

    public static UiLanguage Language => Catalog.Language;

    /// <summary>The culture to format numbers and sizes in, matching <see cref="Language"/>.</summary>
    public static CultureInfo Culture => Catalog.Culture;

    /// <summary>The catalog itself, for the tests that keep the translations complete.</summary>
    public static StringCatalog Strings => Catalog;

    public static string T(string key, params (string Name, object? Value)[] args) => Catalog.T(key, args);

    /// <summary>Counted text: <c>key.one</c> or <c>key.other</c> by the language's rule, the count as <c>{count}</c>.</summary>
    public static string Plural(string key, long count, params (string Name, object? Value)[] args) =>
        Catalog.Plural(key, count, args);

    /// <summary>
    /// Switch to a language directly. The app calls this through <c>Loc</c>: it composes the
    /// drafts of what was shared (<see cref="ShareComposer"/>) in its own process, so its
    /// copy of these strings follows the app's language, not the extension's file.
    /// </summary>
    public static void Use(UiLanguage language) => Catalog.Use(language);

    /// <summary>Show the extension in the language the app would show: the user's choice, else the system's.</summary>
    /// <param name="sharedContainer">The App Group container's directory, or null when it is unavailable.</param>
    /// <param name="preferred">The system's preferred languages, most preferred first.</param>
    public static UiLanguage UseChoiceIn(string? sharedContainer, IEnumerable<string?>? preferred)
    {
        UiLanguage language = UiLanguages.Resolve(ReadChoice(sharedContainer), preferred);
        Catalog.Use(language);
        return language;
    }

    /// <summary>The choice the app left, or null when it left none or the file cannot be read.</summary>
    public static string? ReadChoice(string? sharedContainer)
    {
        if (string.IsNullOrEmpty(sharedContainer))
            return null;
        try
        {
            string path = Path.Combine(sharedContainer, ChoiceFileName);
            return File.Exists(path) ? File.ReadAllText(path).Trim() : null;
        }
        catch (IOException)
        {
            return null;
        }
        catch (UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>
    /// Leave the user's choice for the extension, replacing the file whole so the extension
    /// never reads half of it.
    /// </summary>
    public static void WriteChoice(string sharedContainer, string? choice)
    {
        ArgumentException.ThrowIfNullOrEmpty(sharedContainer);
        string path = Path.Combine(sharedContainer, ChoiceFileName);
        string temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        File.WriteAllText(temporary, (choice ?? string.Empty).Trim());
        File.Move(temporary, path, overwrite: true);
    }
}
