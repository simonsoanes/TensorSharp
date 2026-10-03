// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Hosting;
using TensorAgent.Core.Localization;
using TensorAgent.Core.Settings;
using TensorAgent.Maui.Hosting;
using TensorAgent.Sharing.Localization;

namespace TensorAgent.Maui.Pages;

/// <summary>
/// The sandbox switches, and everything else the user gets to decide.
///
/// <para>
/// Two of these are the security surface of the whole app, so they are presented
/// as what they actually control rather than as feature names. "Run code" decides
/// whether the model may execute anything at all; with it off there is no shell,
/// no interpreter and no skill script, and the model is told so rather than
/// discovering it. "Allow network access" decides whether anything that runs may
/// open a socket. Both default to the safe answer — code on, because an agent that
/// cannot act is not an agent, and network off, because a model that can reach the
/// internet from inside a sandbox is a different risk entirely — and neither is
/// ever changed except from here.
/// </para>
/// <para>
/// A change takes effect on the next command the model runs. It used to take effect at
/// the next launch — the code runner and its policy were built once, at startup — and
/// the page said so, which was honest and useless: an iPhone app is not restarted by
/// leaving it, so the real instruction was "force-quit TensorAgent from the app
/// switcher" and nobody does that. The reported symptom was precisely what that
/// produces: network turned on, and <c>curl</c> still answering "network access is
/// disabled by the user". See <see cref="AgentAppHost.ApplySettings"/>.
/// </para>
/// </summary>
public sealed class SettingsPage : ContentPage
{
    private readonly AgentAppHost _app;
    private readonly VerticalStackLayout _body;
    private Label? _engine;
#if DEBUG
    private static int s_languagePicked;
#endif

    public SettingsPage(LoopbackWebHost host)
    {
        _app = host.App;
        Title = Loc.T("settings.title");
        BackgroundColor = Theme.Background;

        _body = new VerticalStackLayout { Spacing = 4, Padding = new Thickness(0, 8, 0, 24) };
        Content = new ScrollView { Content = _body, BackgroundColor = Theme.Background };

        // A new language repaints this screen at once: it is the one the user switched on.
        // Dispatched, so the picker that raised the change is not torn down inside its own event.
        Loc.Changed += () => Dispatcher.Dispatch(() =>
        {
            try
            {
                Build();
            }
            catch (Exception ex)
            {
                Console.WriteLine("TensorAgent: the settings screen failed to repaint in the new language: " + ex);
            }
        });
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();
        try
        {
            Build();
        }
        catch (Exception ex)
        {
            // Build reads the settings file, sizes the models directory and asks the
            // host to describe the engine. Any of those can throw, and an exception
            // raised here cancels the push this page is being appeared for -- so the
            // user taps Settings and stays on the chat, with nothing said anywhere.
            Console.WriteLine("TensorAgent: the settings screen failed to appear: " + ex);
        }
    }

    /// <summary>
    /// Save a change and hand it to the running app in the same breath.
    ///
    /// <para>
    /// Saving alone is what made "Allow network access" a switch with no effect: the
    /// code runner, the installer and the skill planner all read the file once, at
    /// startup, so the model went on being told "network access is disabled by the
    /// user" until the app was force-quit from the app switcher. The engine line under
    /// the switches is repainted from the host's own answer, so it is evidence rather
    /// than a promise.
    /// </para>
    /// </summary>
    private void Apply(Action<AppSettings> change)
    {
        AppSettings settings = _app.Settings.Update(s =>
        {
            change(s);
            return s;
        });
        _app.ApplySettings(settings);
        if (_engine is not null)
            _engine.Text = Loc.T("settings.sandbox.now", ("engine", _app.DescribeEngine()));
    }


    private void Build()
    {
        AppSettings settings = _app.Settings.Load();
        Title = Loc.T("settings.title");
        _body.Clear();

        _body.Add(Section(Loc.T("settings.language.section")));
        _body.Add(LanguagePicker(settings));

        _body.Add(Section(Loc.T("settings.sandbox.section")));
        _body.Add(Switch(Loc.T("settings.sandbox.runCode.title"), RunCodeDetail,
            settings.AllowCodeExecution,
            on => Apply(s => s.AllowCodeExecution = on)));
#if WINDOWS
        // Windows only, and off unless the user turns it on: the job object this platform
        // offers bounds a process tree but confines neither its files nor its network, so
        // without this the model is never offered the shell at all. The same explicit
        // choice as the server's --code-exec-unconfined.
        _body.Add(Switch(
            Loc.T("settings.sandbox.unconfined.title"),
            Loc.T("settings.sandbox.unconfined.detail"),
            settings.AllowUnconfinedExecution,
            on => Apply(s => s.AllowUnconfinedExecution = on)));
#endif

        _body.Add(Switch(
            Loc.T("settings.sandbox.network.title"),
            Loc.T("settings.sandbox.network.detail"),
            settings.AllowNetwork,
            on => Apply(s => s.AllowNetwork = on)));

        // Here rather than under Generation because it decides what the model may do,
        // like the two above. There was no switch at all: delegation was simply on, on
        // a phone where every sub-agent is another conversation held in memory.
        _body.Add(Switch(
            Loc.T("settings.sandbox.subAgents.title"),
            Loc.T("settings.sandbox.subAgents.detail"),
            settings.MultiAgentEnabled,
            on => Apply(s => s.MultiAgentEnabled = on)));

        // It used to say "the next time TensorAgent starts", which on a phone is not an
        // instruction anybody follows -- leaving an app does not restart it -- so the
        // switch read as one that did nothing. All three now take effect at once.
        _body.Add(Note(Loc.T("settings.sandbox.note")));
        _engine = new Label
        {
            Text = Loc.T("settings.sandbox.now", ("engine", DescribeEngineSafely())),
            FontSize = 12,
            TextColor = Theme.Muted,
            Padding = new Thickness(16, 2),
        };
        _body.Add(_engine);

        _body.Add(Section(Loc.T("settings.generation.section")));
        int loadedContext = _app.ModelService.ContextTokens;
        int modelContext = _app.ModelService.ModelContextTokens;
        string contextNote = modelContext > loadedContext && loadedContext > 0
            ? Loc.T("settings.generation.context.reduced",
                ("modelTokens", Describe(modelContext)), ("activeTokens", Describe(loadedContext)))
            : modelContext > 0 && loadedContext > modelContext
                ? Loc.T("settings.generation.context.configured",
                    ("modelTokens", Describe(modelContext)), ("activeTokens", Describe(loadedContext)))
                : modelContext > 0
                    ? Loc.T("settings.generation.context.model", ("modelTokens", Describe(modelContext)))
                    : loadedContext > 0
                        ? Loc.T("settings.generation.context.active", ("activeTokens", Describe(loadedContext)))
                        : Loc.T("settings.generation.context.unknown");
        _body.Add(Ladder(Loc.T("settings.generation.replyLimit.title"),
            Loc.T("settings.generation.replyLimit.detail", ("context", contextNote)),
            settings.MaxTokens, ReplyLengthRungs,
            v => Apply(s => s.MaxTokens = v)));
        _body.Add(Choice(Loc.T("settings.generation.kvCache.title"),
            Loc.T("settings.generation.kvCache.detail"),
            settings.KvCacheDtype, KvCacheRungs,
            v => Apply(s => s.KvCacheDtype = v)));
        _body.Add(Note(Loc.T("settings.generation.kvCache.note")));
        _body.Add(Stepper(Loc.T("settings.generation.toolTimeout.title"), Loc.T("settings.generation.toolTimeout.detail"),
            settings.ToolTimeoutSeconds, 10, 600, 10,
            v => Apply(s => s.ToolTimeoutSeconds = v)));
        _body.Add(Switch(Loc.T("settings.generation.reasoning.title"),
            Loc.T("settings.generation.reasoning.detail"),
            settings.ThinkByDefault,
            on => _app.Settings.Update(s => { s.ThinkByDefault = on; return s; })));
        // Through Apply like the sandbox switches. It used to save the file and nothing
        // else, so the engine that was standing kept the old policy until the next model
        // load, while AgentAppHost.ApplySpeculationSetting -- written to move the running
        // engine -- was only ever reached when some OTHER switch was flipped.
        _body.Add(Switch(Loc.T("settings.generation.speculative.title"),
            Loc.T("settings.generation.speculative.detail"),
            settings.SpeculativeDecoding,
            on => Apply(s => s.SpeculativeDecoding = on)));

        _body.Add(Section(Loc.T("settings.downloads.section")));
#if IOS
        _body.Add(Switch(Loc.T("settings.downloads.cellular.title"),
            Loc.T("settings.downloads.cellular.detail"),
            settings.AllowCellularDownloads,
            on => _app.Settings.Update(s => { s.AllowCellularDownloads = on; return s; })));
#endif
        _body.Add(Switch(Loc.T("settings.downloads.optional.title"),
            Loc.T("settings.downloads.optional.detail"),
            settings.DownloadOptionalFiles,
            on => _app.Settings.Update(s => { s.DownloadOptionalFiles = on; return s; })));
        // Said here because it is the thing people worry about while a download runs,
        // and the model list can only say it while they are looking at the model list.
        _body.Add(Note(Loc.T("settings.downloads.note")));

        _body.Add(Section(Loc.T("settings.storage.section")));
        _body.Add(ModelCacheDirectoryEditor(settings));
        Label modelSize = Note(Loc.T("settings.storage.models", ("size", "…")));
        _body.Add(modelSize);
        _ = UpdateModelDirectorySize(modelSize, _app.Models.Root);
        _body.Add(Note(Loc.T("settings.storage.chats", ("count", _app.Conversations.List().Count))));
        _body.Add(Note(Loc.T("settings.storage.skills", ("count", _app.Skills.Skills.Count))));

        var clear = new Button
        {
            Text = Loc.T("settings.storage.deleteAll"),
            BackgroundColor = Theme.Surface,
            TextColor = Theme.Danger,
            CornerRadius = 10,
            Margin = new Thickness(16, 12, 16, 0),
        };
        clear.Clicked += async (_, _) =>
        {
            if (!await DisplayAlert(Loc.T("settings.alert.deleteAll.title"), Loc.T("settings.alert.deleteAll.message"),
                    Loc.T("settings.alert.deleteAll.confirm"), Loc.T("common.cancel")))
                return;
            // includeEmpty: "delete all chats" has to mean all of them, including the
            // untouched one the current session is sitting in.
            foreach (var summary in _app.Conversations.List(includeEmpty: true))
                _app.Conversations.Delete(summary.Id);
            Build();
        };
        _body.Add(clear);
    }

    private View ModelCacheDirectoryEditor(AppSettings settings)
    {
        StringComparer pathComparer = OperatingSystem.IsWindows()
            ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
        var folder = new Entry
        {
            Text = _app.Models.Root,
            Placeholder = Loc.T("settings.storage.modelCache.placeholder"),
            TextColor = Theme.Text,
            PlaceholderColor = Theme.Muted,
            BackgroundColor = Theme.Surface,
            FontSize = 14,
            IsTextPredictionEnabled = false,
            IsSpellCheckEnabled = false,
            ReturnType = ReturnType.Done,
        };
        SemanticProperties.SetDescription(folder, Loc.T("settings.storage.modelCache.title"));
        var save = new Button
        {
            Text = Loc.T("settings.storage.modelCache.save"),
            BackgroundColor = Theme.Accent,
            TextColor = Colors.White,
            CornerRadius = 10,
            IsEnabled = false,
        };
        var restore = new Button
        {
            Text = Loc.T("settings.storage.modelCache.default"),
            BackgroundColor = Theme.Surface,
            TextColor = Theme.Text,
            CornerRadius = 10,
            IsEnabled = !string.IsNullOrWhiteSpace(settings.ModelCacheDirectory),
        };

        folder.TextChanged += (_, _) => save.IsEnabled =
            !pathComparer.Equals(folder.Text?.Trim(), _app.Models.Root);

        async Task SaveDirectory(string? directory)
        {
            folder.IsEnabled = save.IsEnabled = restore.IsEnabled = false;
            try
            {
                // The host validates the folder before persisting it. Saving
                // through Apply would keep an invalid path when the change is refused.
                await Task.Run(() => _app.SetModelCacheDirectory(directory));
            }
            catch (Exception ex) when (ex is ArgumentException or IOException or UnauthorizedAccessException or InvalidOperationException)
            {
                await DisplayAlert(Loc.T("settings.storage.modelCache.error.title"),
                    Loc.T("settings.storage.modelCache.error.message", ("error", ex.Message)), Loc.T("common.ok"));
                folder.IsEnabled = true;
                save.IsEnabled = !pathComparer.Equals(folder.Text?.Trim(), _app.Models.Root);
                restore.IsEnabled = !string.IsNullOrWhiteSpace(_app.Settings.Load().ModelCacheDirectory);
                return;
            }
            Build();
        }

        save.Clicked += async (_, _) => await SaveDirectory(folder.Text);
        restore.Clicked += async (_, _) => await SaveDirectory(string.Empty);
        folder.Completed += async (_, _) =>
        {
            if (save.IsEnabled)
                await SaveDirectory(folder.Text);
        };

        return new VerticalStackLayout
        {
            Spacing = 8,
            Padding = new Thickness(16, 10),
            Children =
            {
                new Label { Text = Loc.T("settings.storage.modelCache.title"), FontSize = 16, TextColor = Theme.Text },
                new Label { Text = Loc.T("settings.storage.modelCache.detail"), FontSize = 12, TextColor = Theme.Muted },
                folder,
                save,
                restore,
            },
        };
    }

    private async Task UpdateModelDirectorySize(Label label, string directory)
    {
        long bytes = await Task.Run(() => DirectorySize(directory));
        // A folder or language change rebuilds the page while the scan is running.
        if (_body.Children.Contains(label))
            label.Text = Loc.T("settings.storage.models", ("size", Gb(bytes)));
    }

    /// <summary>
    /// The engine line, or why there isn't one.
    ///
    /// <para>
    /// A live call into the host, made while this page is being appeared for a push
    /// that has not completed. AboutPage already wraps the identical call; unguarded
    /// here it could take the whole Settings screen down with it and leave the user on
    /// the chat, which reads as the menu having ignored them.
    /// </para>
    /// </summary>
    private string DescribeEngineSafely()
    {
        try { return _app.DescribeEngine(); }
        catch (Exception ex) { return Loc.T("settings.sandbox.engineUnavailable", ("error", ex.GetType().Name)); }
    }

    /// <summary>What "Run code" lets the model do here, which is not the same on every platform.</summary>
    private static string RunCodeDetail =>
#if IOS
        Loc.T("settings.sandbox.runCode.detail.ios");
#elif MACCATALYST
        Loc.T("settings.sandbox.runCode.detail.mac");
#else
        Loc.T("settings.sandbox.runCode.detail.windows", ("setting", Loc.T("settings.sandbox.unconfined.title")));
#endif

    /// <summary>
    /// The interface language. "System" says which language it means right now, and every
    /// other entry is written in its own language, so someone who cannot read the screen in
    /// front of them can still find theirs. A change is saved and applied at once: this
    /// screen repaints, and so do the page and every other screen (see <see cref="Loc.Changed"/>).
    /// </summary>
    private View LanguagePicker(AppSettings settings)
    {
        UiLanguage system = UiLanguages.Resolve(null, Loc.SystemLanguages());
        var options = new List<(string Tag, string Label)>
        {
            (string.Empty, Loc.T("settings.language.system", ("language", system.NativeName))),
        };
        options.AddRange(UiLanguages.Supported.Select(language => (language.Tag, language.NativeName)));

        // A tag this build does not know (a newer build may have saved it) shows as System,
        // which is also what it resolves to.
        string current = UiLanguages.Match(settings.UiLanguage)?.Tag ?? string.Empty;
        var picker = new Picker
        {
            ItemsSource = options.Select(o => o.Label).ToList(),
            SelectedIndex = Math.Max(0, options.FindIndex(o => o.Tag == current)),
            TextColor = Theme.Accent,
            FontSize = 15,
            HorizontalOptions = LayoutOptions.End,
            VerticalOptions = LayoutOptions.Center,
        };
        picker.SelectedIndexChanged += (_, _) =>
        {
            if (picker.SelectedIndex < 0 || picker.SelectedIndex >= options.Count)
                return;
            string tag = options[picker.SelectedIndex].Tag;
            if (string.Equals(tag, current, StringComparison.Ordinal))
                return;
            current = tag;
            Apply(s => s.UiLanguage = tag);
#if IOS
            // The share extension shows itself in the same language; it reads the choice
            // from the App Group, since it cannot read these settings.
            Platforms.iOS.SharedContainer.WriteLanguageChoice(tag);
#endif
        };

#if DEBUG
        // simctl cannot tap. TENSORAGENT_PICK_LANGUAGE=<tag> (or "system") picks that row once
        // per launch, through the handler above, so the simulator harness can switch the
        // language and relaunch to see it kept. Pair it with TENSORAGENT_START_PAGE=settings.
        string? pick = Environment.GetEnvironmentVariable("TENSORAGENT_PICK_LANGUAGE")?.Trim();
        if (!string.IsNullOrEmpty(pick) && Interlocked.Exchange(ref s_languagePicked, 1) == 0)
        {
            string wanted = pick.Equals("system", StringComparison.OrdinalIgnoreCase) ? string.Empty : pick;
            int index = options.FindIndex(o => string.Equals(o.Tag, wanted, StringComparison.OrdinalIgnoreCase));
            Console.WriteLine($"TensorAgent: languagecheck pick {pick} -> row {index}");
            if (index >= 0)
                Dispatcher.Dispatch(() => picker.SelectedIndex = index);
        }
#endif

        var grid = new Grid
        {
            ColumnDefinitions = { new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Auto) },
            Padding = new Thickness(16, 10),
            ColumnSpacing = 10,
        };
        grid.Add(new VerticalStackLayout
        {
            Spacing = 2,
            Children =
            {
                new Label { Text = Loc.T("settings.language.title"), FontSize = 16, TextColor = Theme.Text },
                new Label { Text = Loc.T("settings.language.detail"), FontSize = 12, TextColor = Theme.Muted },
            },
        }, 0, 0);
        grid.Add(picker, 1, 0);
        return grid;
    }

    private static View Section(string text) => new Label
    {
        Text = text.ToUpperInvariant(),
        FontSize = 12,
        TextColor = Theme.Muted,
        FontAttributes = FontAttributes.Bold,
        Padding = new Thickness(16, 20, 16, 6),
    };

    private static Label Note(string text) => new Label
    {
        Text = text,
        FontSize = 12,
        TextColor = Theme.Muted,
        Padding = new Thickness(16, 2),
    };

    private static View Switch(string title, string detail, bool value, Action<bool> onChanged)
    {
        var toggle = new Microsoft.Maui.Controls.Switch { IsToggled = value, OnColor = Theme.Accent, VerticalOptions = LayoutOptions.Center };
        toggle.Toggled += (_, e) => onChanged(e.Value);

        var grid = new Grid
        {
            ColumnDefinitions = { new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Auto) },
            Padding = new Thickness(16, 10),
        };
        grid.Add(new VerticalStackLayout
        {
            Spacing = 2,
            Children =
            {
                new Label { Text = title, FontSize = 16, TextColor = Theme.Text },
                new Label { Text = detail, FontSize = 12, TextColor = Theme.Muted },
            },
        }, 0, 0);
        grid.Add(toggle, 1, 0);
        return grid;
    }

    /// <summary>
    /// The reply-length rungs, up to 256K tokens.
    ///
    /// <para>
    /// A plain Stepper cannot express this range: 256 to 262,144 in steps of 256 is a
    /// thousand taps. The rungs double instead, so the whole range is eleven taps and
    /// the useful small values keep their resolution.
    /// </para>
    ///
    /// <para>
    /// The ceiling that actually applies is the CONTEXT, not this number:
    /// ChatGenerationPipeline.ClampGenerationReserve trims the reserve to what the
    /// window leaves after the prompt, so asking for 256K inside an 8,192-token context
    /// yields at most 8,192 minus the prompt. Raising this is what lets a long context
    /// be spent on one reply; it does not create context.
    /// </para>
    /// </summary>
    private static readonly int[] ReplyLengthRungs =
        { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144 };

    /// <summary>
    /// The K/V cache precisions, in the order the stepper walks them.
    ///
    /// <para>
    /// Widest first, so stepping right spends less memory — the same direction as every
    /// other stepper on this page. The stored values are the engine's own spellings;
    /// the labels are what the user is likely to have seen on a model card.
    /// </para>
    /// </summary>
    private static readonly (string Value, string Label)[] KvCacheRungs =
    {
        ("f16", "FP16"),
        ("q8_0", "Q8"),
        ("q4_0", "Q4"),
    };

    /// <summary>
    /// One of a short list of named values, on the same stepper the numbers use.
    ///
    /// <para>
    /// A Picker would open a modal wheel for three options. The stepper is already the
    /// page's idiom for "walk a small ordered range", and these ARE ordered: each step
    /// right halves the memory and loses a little precision.
    /// </para>
    /// </summary>
    private static View Choice(
        string title, string detail, string value,
        (string Value, string Label)[] options, Action<string> onChanged)
    {
        int index = 0;
        for (int i = 0; i < options.Length; i++)
        {
            if (string.Equals(options[i].Value, value, StringComparison.OrdinalIgnoreCase)) index = i;
        }

        var current = new Label
        {
            Text = options[index].Label,
            FontSize = 15,
            TextColor = Theme.Accent,
            VerticalOptions = LayoutOptions.Center,
        };
        var stepper = new Stepper(0, options.Length - 1, index, 1) { VerticalOptions = LayoutOptions.Center };
        stepper.ValueChanged += (_, e) =>
        {
            (string Value, string Label) picked = options[Math.Clamp((int)e.NewValue, 0, options.Length - 1)];
            current.Text = picked.Label;
            onChanged(picked.Value);
        };

        var grid = new Grid
        {
            ColumnDefinitions =
            {
                new ColumnDefinition(GridLength.Star),
                new ColumnDefinition(GridLength.Auto),
                new ColumnDefinition(GridLength.Auto),
            },
            Padding = new Thickness(16, 10),
            ColumnSpacing = 10,
        };
        grid.Add(new VerticalStackLayout
        {
            Spacing = 2,
            Children =
            {
                new Label { Text = title, FontSize = 16, TextColor = Theme.Text },
                new Label { Text = detail, FontSize = 12, TextColor = Theme.Muted },
            },
        });
        grid.Add(current, 1, 0);
        grid.Add(stepper, 2, 0);
        return grid;
    }

    private static View Ladder(string title, string detail, int value, int[] rungs, Action<int> onChanged)
    {
        int index = 0;
        for (int i = 0; i < rungs.Length; i++)
        {
            if (rungs[i] <= value) index = i;
        }

        var current = new Label
        {
            Text = Describe(rungs[index]),
            FontSize = 15,
            TextColor = Theme.Accent,
            VerticalOptions = LayoutOptions.Center,
        };
        var stepper = new Stepper(0, rungs.Length - 1, index, 1) { VerticalOptions = LayoutOptions.Center };
        stepper.ValueChanged += (_, e) =>
        {
            int chosen = rungs[Math.Clamp((int)e.NewValue, 0, rungs.Length - 1)];
            current.Text = Describe(chosen);
            onChanged(chosen);
        };

        var grid = new Grid
        {
            ColumnDefinitions =
            {
                new ColumnDefinition(GridLength.Star),
                new ColumnDefinition(GridLength.Auto),
                new ColumnDefinition(GridLength.Auto),
            },
            Padding = new Thickness(16, 10),
            ColumnSpacing = 10,
        };
        grid.Add(new VerticalStackLayout
        {
            Spacing = 2,
            Children =
            {
                new Label { Text = title, FontSize = 16, TextColor = Theme.Text },
                new Label { Text = detail, FontSize = 12, TextColor = Theme.Muted },
            },
        });
        grid.Add(current, 1, 0);
        grid.Add(stepper, 2, 0);
        return grid;
    }

    /// <summary>"1024" is harder to read at a glance than "1K"; the rungs are all
    /// powers of two, so the short form is exact rather than rounded.</summary>
    private static string Describe(int tokens) =>
        tokens >= 1024 && tokens % 1024 == 0 ? (tokens / 1024) + "K" : tokens.ToString();

    private static View Stepper(string title, string detail, int value, int min, int max, int step, Action<int> onChanged)
    {
        var current = new Label { Text = value.ToString(), FontSize = 15, TextColor = Theme.Accent, VerticalOptions = LayoutOptions.Center };
        var stepper = new Stepper(min, max, Math.Clamp(value, min, max), step) { VerticalOptions = LayoutOptions.Center };
        stepper.ValueChanged += (_, e) =>
        {
            int v = (int)e.NewValue;
            current.Text = v.ToString();
            onChanged(v);
        };

        var grid = new Grid
        {
            ColumnDefinitions =
            {
                new ColumnDefinition(GridLength.Star),
                new ColumnDefinition(GridLength.Auto),
                new ColumnDefinition(GridLength.Auto),
            },
            Padding = new Thickness(16, 10),
            ColumnSpacing = 10,
        };
        grid.Add(new VerticalStackLayout
        {
            Spacing = 2,
            Children =
            {
                new Label { Text = title, FontSize = 16, TextColor = Theme.Text },
                new Label { Text = detail, FontSize = 12, TextColor = Theme.Muted },
            },
        }, 0, 0);
        grid.Add(current, 1, 0);
        grid.Add(stepper, 2, 0);
        return grid;
    }

    private static long DirectorySize(string path)
    {
        try
        {
            long total = 0;
            foreach (string file in Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories))
                total += new FileInfo(file).Length;
            return total;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return 0;
        }
    }

    private static string Gb(long bytes) => (bytes / 1e9).ToString("0.00", Loc.Culture);
}
