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

namespace TensorAgent.Maui.Services;

/// <summary>
/// The languages the user set on this device, most preferred first: what a first launch
/// shows the app in, and what "System" in the language setting follows.
///
/// <para>
/// The ordered list, not one culture, because the first preferred language may be one the
/// app has no strings for and the second one it does. On iOS and the Mac that is
/// <c>NSLocale.PreferredLanguages</c>, which also reflects a per-app language chosen in the
/// system's own Settings ("zh-Hans-CN", "en-US", …); on Windows, the user's language list.
/// </para>
/// </summary>
internal static class SystemLanguages
{
    public static IReadOnlyList<string> Preferred()
    {
        try
        {
#if IOS || MACCATALYST
            string[]? languages = Foundation.NSLocale.PreferredLanguages;
            if (languages is { Length: > 0 })
                return languages;
#elif WINDOWS
            IReadOnlyList<string> languages = global::Windows.System.UserProfile.GlobalizationPreferences.Languages;
            if (languages.Count > 0)
                return languages;
#endif
        }
        catch (Exception ex)
        {
            Console.WriteLine("TensorAgent: the system's preferred languages could not be read: " + ex.Message);
        }
        return new[] { CultureInfo.CurrentUICulture.Name };
    }
}
