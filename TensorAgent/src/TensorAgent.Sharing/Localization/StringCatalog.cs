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
using System.Reflection;
using System.Text;
using System.Text.Json;

namespace TensorAgent.Sharing.Localization;

/// <summary>
/// One set of translated strings, in the language the user gets.
///
/// <para>
/// The strings are JSON files embedded in an assembly, one directory per language and one
/// file per area (<c>&lt;prefix&gt;zh-Hans/settings.json</c>), each a flat object of
/// <c>"area.key": "text"</c>. English is the source: a key a translation lacks reads as the
/// English text rather than as nothing, and a key English lacks reads as the key itself, so
/// a gap is visible on screen instead of blank. The tests keep every language complete.
/// </para>
/// <para>
/// Text takes named placeholders, <c>{name}</c>, and <c>{{</c> / <c>}}</c> for literal
/// braces. Counted text is two keys, <c>key.one</c> and <c>key.other</c>, chosen by the
/// language's plural rule (<see cref="PluralCategory"/>), with the count as <c>{count}</c>.
/// </para>
/// <para>
/// The current table is swapped whole, so a reader on another thread sees either the old
/// language or the new one, never a mixture.
/// </para>
/// </summary>
public sealed class StringCatalog
{
    private readonly Assembly _assembly;
    private readonly string _prefix;
    private volatile Table _current;

    /// <param name="assembly">The assembly the JSON files are embedded in.</param>
    /// <param name="resourcePrefix">Their logical-name prefix; a file is <c>prefix + tag + "/" + area + ".json"</c>.</param>
    public StringCatalog(Assembly assembly, string resourcePrefix)
    {
        _assembly = assembly ?? throw new ArgumentNullException(nameof(assembly));
        _prefix = resourcePrefix ?? throw new ArgumentNullException(nameof(resourcePrefix));
        _current = Build(UiLanguages.English);
    }

    /// <summary>Raised after <see cref="Use"/> changes the language, on the thread that called it.</summary>
    public event Action? Changed;

    public UiLanguage Language => _current.Language;

    /// <summary>The culture numbers and dates are formatted in: the language's own, or the
    /// invariant culture where the runtime has no data for it.</summary>
    public CultureInfo Culture => _current.Culture;

    /// <summary>Every key in the current language, English filling any gap.</summary>
    public IReadOnlyDictionary<string, string> Strings => _current.Strings;

    /// <summary>The language and its strings from one table, even if another thread switches
    /// languages while the caller reads or serializes them. The captured table stays unchanged.</summary>
    public (UiLanguage Language, IReadOnlyDictionary<string, string> Strings) Snapshot()
    {
        Table table = _current;
        return (table.Language, table.Strings);
    }

    /// <summary>Switch to a language. A no-op, and no <see cref="Changed"/>, when it is already the one in use.</summary>
    public void Use(UiLanguage language)
    {
        ArgumentNullException.ThrowIfNull(language);
        if (string.Equals(_current.Language.Tag, language.Tag, StringComparison.Ordinal))
            return;
        _current = Build(language);
        Changed?.Invoke();
    }

    /// <summary>The text for a key, with its placeholders filled.</summary>
    public string T(string key, params (string Name, object? Value)[] args)
    {
        Table table = _current;
        string text = table.Strings.TryGetValue(key, out string? found) ? found : key;
        return Format(text, args, table.Culture);
    }

    /// <summary>
    /// Counted text: <c>key.one</c> or <c>key.other</c> by the language's rule, with
    /// <c>{count}</c> filled from <paramref name="count"/> in the language's number format.
    /// </summary>
    public string Plural(string key, long count, params (string Name, object? Value)[] args)
    {
        Table table = _current;
        string category = PluralCategory(table.Language.Tag, count);
        var all = new (string Name, object? Value)[args.Length + 1];
        all[0] = ("count", count);
        args.CopyTo(all, 1);
        string full = key + "." + category;
        string text = table.Strings.TryGetValue(full, out string? found) ? found : full;
        return Format(text, all, table.Culture);
    }

    /// <summary>
    /// The CLDR cardinal category of an integer in the shipped languages: Chinese, Japanese
    /// and Korean have one form, French counts 0 and 1 as singular, the rest only 1.
    /// </summary>
    public static string PluralCategory(string tag, long count) => tag switch
    {
        "zh-Hans" or "zh-Hant" or "ja" or "ko" => "other",
        "fr" => count is 0 or 1 ? "one" : "other",
        _ => count == 1 ? "one" : "other",
    };

    /// <summary>Fill <c>{name}</c> placeholders; an unknown name is left as written, so it shows.</summary>
    public static string Format(string text, IReadOnlyList<(string Name, object? Value)> args, IFormatProvider? culture)
    {
        if (text.IndexOf('{') < 0 && text.IndexOf('}') < 0)
            return text;
        var result = new StringBuilder(text.Length + 16);
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if ((c == '{' || c == '}') && i + 1 < text.Length && text[i + 1] == c)
            {
                result.Append(c);
                i++;
                continue;
            }
            if (c == '{')
            {
                int close = text.IndexOf('}', i + 1);
                if (close > i + 1)
                {
                    string name = text.Substring(i + 1, close - i - 1);
                    if (TryFind(args, name, out object? value))
                    {
                        result.Append(value is IFormattable formattable
                            ? formattable.ToString(null, culture)
                            : value?.ToString());
                        i = close;
                        continue;
                    }
                }
            }
            result.Append(c);
        }
        return result.ToString();
    }

    /// <summary>The placeholder names a text uses, for the tests that keep translations in step.</summary>
    public static IReadOnlySet<string> Placeholders(string text)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if ((c == '{' || c == '}') && i + 1 < text.Length && text[i + 1] == c)
            {
                i++;
                continue;
            }
            if (c != '{')
                continue;
            int close = text.IndexOf('}', i + 1);
            if (close > i + 1)
            {
                names.Add(text.Substring(i + 1, close - i - 1));
                i = close;
            }
        }
        return names;
    }

    /// <summary>
    /// One language's own strings, exactly as filed and without the English fallback, by
    /// area file name. For the tests and tools that check a translation is complete.
    /// </summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> LoadFiles(string tag)
    {
        var files = new SortedDictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal);
        // The directory separator is the one the build ran on: %(RecursiveDir) names the
        // resource "zh-Hans\settings.json" on Windows and "zh-Hans/settings.json" elsewhere.
        string[] directories = { _prefix + tag + "/", _prefix + tag + "\\" };
        foreach (string name in _assembly.GetManifestResourceNames())
        {
            string? directory = directories.FirstOrDefault(d => name.StartsWith(d, StringComparison.Ordinal));
            if (directory is null || !name.EndsWith(".json", StringComparison.Ordinal))
                continue;
            using Stream stream = _assembly.GetManifestResourceStream(name)!;
            files[name.Substring(directory.Length, name.Length - directory.Length - ".json".Length)] = Parse(stream, name);
        }
        return files;
    }

    private static bool TryFind(IReadOnlyList<(string Name, object? Value)> args, string name, out object? value)
    {
        for (int i = 0; i < args.Count; i++)
        {
            if (string.Equals(args[i].Name, name, StringComparison.Ordinal))
            {
                value = args[i].Value;
                return true;
            }
        }
        value = null;
        return false;
    }

    private Table Build(UiLanguage language)
    {
        var strings = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (IReadOnlyDictionary<string, string> file in LoadFiles(UiLanguages.English.Tag).Values)
        {
            foreach ((string key, string value) in file)
                strings[key] = value;
        }
        if (!string.Equals(language.Tag, UiLanguages.English.Tag, StringComparison.Ordinal))
        {
            foreach (IReadOnlyDictionary<string, string> file in LoadFiles(language.Tag).Values)
            {
                foreach ((string key, string value) in file)
                {
                    if (!string.IsNullOrEmpty(value))
                        strings[key] = value;
                }
            }
        }
        return new Table(language, CultureFor(language), strings);
    }

    private static CultureInfo CultureFor(UiLanguage language)
    {
        try
        {
            return CultureInfo.GetCultureInfo(language.Tag);
        }
        catch (CultureNotFoundException)
        {
            // An invariant-globalization runtime has no cultures at all; numbers then read
            // as they do everywhere else in the app rather than failing the language.
            return CultureInfo.InvariantCulture;
        }
    }

    /// <summary>A flat JSON object of strings; anything else in a table is a build mistake, so it throws.</summary>
    private static IReadOnlyDictionary<string, string> Parse(Stream stream, string name)
    {
        using JsonDocument document = JsonDocument.Parse(stream, new JsonDocumentOptions { AllowTrailingCommas = false, CommentHandling = JsonCommentHandling.Disallow });
        if (document.RootElement.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException(name + " must be a JSON object of strings.");
        var strings = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (JsonProperty property in document.RootElement.EnumerateObject())
        {
            if (property.Value.ValueKind != JsonValueKind.String)
                throw new InvalidDataException(name + ": \"" + property.Name + "\" must be a string.");
            strings[property.Name] = property.Value.GetString()!;
        }
        return strings;
    }

    private sealed record Table(UiLanguage Language, CultureInfo Culture, IReadOnlyDictionary<string, string> Strings);
}
