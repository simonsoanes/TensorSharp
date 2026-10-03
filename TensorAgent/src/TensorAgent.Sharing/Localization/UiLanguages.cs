// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorAgent.Sharing.Localization;

/// <summary>A language TensorAgent's own interface is translated into.</summary>
/// <param name="Tag">The BCP-47 tag its string tables are filed under, and what the settings store.</param>
/// <param name="NativeName">The language's name in itself, which is how a language picker shows it.</param>
/// <param name="EnglishName">For logs and tests.</param>
public sealed record UiLanguage(string Tag, string NativeName, string EnglishName);

/// <summary>
/// The interface languages, and which of them a user gets.
///
/// <para>
/// Here rather than in the app because the share extension speaks the same languages and
/// must reach the same answer from the same inputs, and this assembly is the only code
/// both processes run.
/// </para>
/// </summary>
public static class UiLanguages
{
    public static UiLanguage English { get; } = new("en", "English", "English");

    /// <summary>Every language a string table exists for, English first.</summary>
    public static IReadOnlyList<UiLanguage> Supported { get; } = new[]
    {
        English,
        new UiLanguage("zh-Hans", "简体中文", "Chinese (Simplified)"),
        new UiLanguage("zh-Hant", "繁體中文", "Chinese (Traditional)"),
        new UiLanguage("ja", "日本語", "Japanese"),
        new UiLanguage("ko", "한국어", "Korean"),
        new UiLanguage("es", "Español", "Spanish"),
        new UiLanguage("fr", "Français", "French"),
        new UiLanguage("de", "Deutsch", "German"),
    };

    /// <summary>The supported language with exactly this tag, ignoring case.</summary>
    public static UiLanguage? Find(string? tag) =>
        string.IsNullOrWhiteSpace(tag)
            ? null
            : Supported.FirstOrDefault(l => string.Equals(l.Tag, tag.Trim(), StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// The supported language a BCP-47 or ICU locale identifier asks for, or null when
    /// none of them is that language.
    ///
    /// <para>
    /// Platforms spell the same choice several ways: iOS hands over "zh-Hans-CN" or
    /// "zh-Hant-TW", Windows and .NET "zh-CN" and "zh-TW", ICU "zh_CN", and Cantonese
    /// arrives as "yue-Hant-HK". Chinese is decided by script, then by the region a script
    /// implies, and plain "zh" is Simplified; any other language by its first subtag, so
    /// "es-MX" is Spanish and "pt-BR" is not one of these at all.
    /// </para>
    /// </summary>
    public static UiLanguage? Match(string? tag)
    {
        if (string.IsNullOrWhiteSpace(tag))
            return null;
        string[] parts = tag.Trim().Replace('_', '-').Split('-', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0)
            return null;

        string language = parts[0].ToLowerInvariant();
        if (language is "zh" or "yue")
        {
            foreach (string part in parts.Skip(1))
            {
                if (part.Equals("Hant", StringComparison.OrdinalIgnoreCase))
                    return Find("zh-Hant");
                if (part.Equals("Hans", StringComparison.OrdinalIgnoreCase))
                    return Find("zh-Hans");
            }
            foreach (string part in parts.Skip(1))
            {
                if (part.ToUpperInvariant() is "TW" or "HK" or "MO")
                    return Find("zh-Hant");
            }
            return Find(language == "yue" ? "zh-Hant" : "zh-Hans");
        }
        return Find(language);
    }

    /// <summary>
    /// The language to show: the user's choice when it is one this build has, otherwise the
    /// first of the system's preferred languages that is, otherwise English.
    /// </summary>
    /// <param name="choice">What the settings hold: a tag, or empty to follow the system.
    /// A tag this build does not know (a newer build may have written it) follows the
    /// system rather than failing.</param>
    /// <param name="preferred">The system's preferred languages, most preferred first.</param>
    public static UiLanguage Resolve(string? choice, IEnumerable<string?>? preferred)
    {
        if (Match(choice) is { } chosen)
            return chosen;
        foreach (string? tag in preferred ?? Array.Empty<string?>())
        {
            if (Match(tag) is { } match)
                return match;
        }
        return English;
    }
}
