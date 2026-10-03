// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using TensorAgent.Core.Localization;
using TensorAgent.Sharing.Localization;

namespace TensorAgent.Tests;

[Collection(UiLanguageCollection.Name)]
public sealed class PageStringsSnapshotTests
{
    [Fact]
    public async Task ASwitchDuringSerializationCannotMixTheLanguageAndItsStrings()
    {
        UiLanguage chinese = UiLanguages.Find("zh-Hans")!;
        var titles = new Dictionary<string, string>
        {
            ["en"] = Loc.Tables.LoadFiles("en")["settings"]["settings.title"],
            ["zh-Hans"] = Loc.Tables.LoadFiles("zh-Hans")["settings"]["settings.title"],
        };
        Loc.Use(UiLanguages.English);
        using var start = new Barrier(2);
        Task switching = Task.Run(() =>
        {
            start.SignalAndWait();
            for (int i = 0; i < 256; i++)
                Loc.Use(i % 2 == 0 ? chinese : UiLanguages.English);
        });

        try
        {
            start.SignalAndWait();
            for (int i = 0; i < 256; i++)
            {
                string script = PageStrings.Script();
                const string prefix = "window.TensorAgentI18n = ";
                int end = script.IndexOf(";\n", StringComparison.Ordinal);
                using JsonDocument data = JsonDocument.Parse(script[prefix.Length..end]);
                JsonElement page = data.RootElement;
                string tag = page.GetProperty("lang").GetString()!;
                Assert.Equal(titles[tag], page.GetProperty("strings").GetProperty("settings.title").GetString());
            }
        }
        finally
        {
            try { await switching; }
            finally { Loc.Use(UiLanguages.English); }
        }
    }
}
