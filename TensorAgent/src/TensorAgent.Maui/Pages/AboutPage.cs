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
using TensorAgent.Maui.Hosting;

namespace TensorAgent.Maui.Pages;

/// <summary>
/// What this app is, and what it is built on.
///
/// <para>
/// It exists because the Web UI's own header carried a "TensorSharp.ai" link banner
/// that made sense on a desktop page and not on a phone, where it spent a row of a
/// small screen on something a reader needs once. The link is removed from the chat
/// and the explanation lives here instead, where someone can go looking for it.
/// </para>
/// </summary>
public sealed class AboutPage : ContentPage
{
    private readonly LoopbackWebHost _host;

    public AboutPage(LoopbackWebHost host)
    {
        _host = host;
        BackgroundColor = Theme.Background;
        Build();

        // Rebuild the page's text when the language changes.
        Loc.Changed += () => Dispatcher.Dispatch(() =>
        {
            try
            {
                Build();
            }
            catch (Exception ex)
            {
                Console.WriteLine("TensorAgent: the about screen failed to repaint in the new language: " + ex);
            }
        });
    }

    private void Build()
    {
        Title = Loc.T("about.title");

        var stack = new VerticalStackLayout { Spacing = 0, Padding = new Thickness(20, 16, 20, 32) };

        // The project's own banner, whole rather than cropped: it carries the
        // TensorSharp wordmark, and AspectFill at any height a phone can spare would
        // cut it off. AspectFit against the page's content width lets the Image
        // measure its own height from the 1253x836 source, so the row costs about
        // two thirds of that width and no more.
        stack.Add(new Border
        {
            Margin = new Thickness(0, 0, 0, 18),
            Padding = new Thickness(0),
            BackgroundColor = Theme.Surface,
            StrokeThickness = 0,
            StrokeShape = new Microsoft.Maui.Controls.Shapes.RoundRectangle { CornerRadius = 12 },
            Content = new Image
            {
                Source = ImageSource.FromFile("banner_1.png"),
                Aspect = Aspect.AspectFit,
                HorizontalOptions = LayoutOptions.Fill,
            },
        });

        stack.Add(new Label
        {
            Text = "TensorAgent",
            FontSize = 30,
            FontAttributes = FontAttributes.Bold,
            TextColor = Theme.Text,
        });
        stack.Add(new Label
        {
            Text = Loc.T("about.tagline"),
            FontSize = 15,
            TextColor = Theme.Muted,
            Margin = new Thickness(0, 4, 0, 18),
        });

        stack.Add(Section(Loc.T("about.private.heading"), Loc.T("about.private.body")));

        stack.Add(Section(Loc.T("about.abilities.heading"), Loc.T("about.abilities.body")));

        stack.Add(Section(Loc.T("about.control.heading"), Loc.T("about.control.body")));

        stack.Add(Section(Loc.T("about.builtOn.heading"), Loc.T("about.builtOn.body")));

        var link = new Label
        {
            Text = "tensorsharp.ai",
            FontSize = 16,
            TextColor = Theme.Accent,
            Margin = new Thickness(0, 2, 0, 20),
        };
        link.GestureRecognizers.Add(new TapGestureRecognizer
        {
            Command = new Command(async () =>
            {
                try { await Launcher.OpenAsync("https://tensorsharp.ai"); }
                catch (Exception ex) { Console.WriteLine("TensorAgent: about link failed: " + ex.Message); }
            }),
        });
        stack.Add(link);

        var coffee = new Image
        {
            Source = ImageSource.FromUri(new Uri("https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png")),
            HeightRequest = 50,
            Aspect = Aspect.AspectFit,
            HorizontalOptions = LayoutOptions.Center,
        };
        SemanticProperties.SetDescription(coffee, "Buy Me A Coffee");
        coffee.GestureRecognizers.Add(new TapGestureRecognizer
        {
            Command = new Command(async () =>
            {
                try { await Launcher.OpenAsync("https://buymeacoffee.com/zhongkaifu"); }
                catch (Exception ex) { Console.WriteLine("TensorAgent: coffee link failed: " + ex.Message); }
            }),
        });
        stack.Add(coffee);
        stack.Add(new Label
        {
            Text = Loc.T("about.support.caption"),
            FontSize = 12,
            TextColor = Theme.Muted,
            HorizontalTextAlignment = TextAlignment.Center,
            LineBreakMode = LineBreakMode.WordWrap,
            Margin = new Thickness(0, 8, 0, 20),
        });

        stack.Add(new Label
        {
            Text = Engine(_host),
            FontSize = 12,
            TextColor = Theme.Muted,
            LineBreakMode = LineBreakMode.WordWrap,
        });

        Content = new ScrollView { BackgroundColor = Theme.Background, Content = stack };
    }

    private static View Section(string heading, string body) => new VerticalStackLayout
    {
        Spacing = 4,
        Margin = new Thickness(0, 0, 0, 18),
        Children =
        {
            new Label { Text = heading, FontSize = 17, FontAttributes = FontAttributes.Bold, TextColor = Theme.Text },
            new Label { Text = body, FontSize = 14, TextColor = Theme.Muted, LineBreakMode = LineBreakMode.WordWrap },
        },
    };

    /// <summary>
    /// The engine line the app already computes for its own diagnostics. It belongs on
    /// this page too: when something is wrong, this is the text a user can read out.
    /// </summary>
    private static string Engine(LoopbackWebHost host)
    {
        try
        {
            return host.App.DescribeEngine();
        }
        catch (Exception ex)
        {
            return Loc.T("about.engine.unavailable", ("error", ex.Message));
        }
    }
}
