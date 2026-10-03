// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.ObjectModel;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Localization;
using TensorAgent.Core.Sessions;
using TensorAgent.Maui.Hosting;

namespace TensorAgent.Maui.Pages;

/// <summary>
/// Saved chats: every conversation the app has kept, newest first, the way back into
/// any of them, and the way to remove one.
///
/// <para>
/// It is the only chat item in the flyout. "Chat" and "Chats" as two menu entries
/// asked the user to tell two words apart to discover that one of them was the page
/// they were already on; this list is what a chat menu is for — pick one to open it,
/// swipe it away to delete it, or start a new one from the top.
/// </para>
///
/// <para>
/// Picking one does not reconstruct the model's state — the KV cache did not
/// survive the app being closed, and pretending otherwise would give the user a
/// model that has forgotten a conversation it appears to be in the middle of.
/// What happens instead is what the Web UI already does on a page reload: the
/// transcript is re-rendered and the next turn is sent with the whole history, so
/// the model sees everything even though it is starting fresh.
/// </para>
/// </summary>
public sealed class SessionsPage : ContentPage
{
    private readonly AgentAppHost _app;
    private readonly MainPage _chat;
    private readonly ObservableCollection<ConversationSummary> _rows = new();

    public SessionsPage(LoopbackWebHost host, MainPage chat)
    {
        _app = host.App;
        _chat = chat;
        BackgroundColor = Theme.Background;
        Build();

        // A new language rebuilds the screen, cells and all; the rows' date and count lines
        // are formatted again as the new list binds them.
        Loc.Changed += () => Dispatcher.Dispatch(() =>
        {
            try
            {
                Build();
            }
            catch (Exception ex)
            {
                Console.WriteLine("TensorAgent: the chats list failed to repaint in the new language: " + ex);
            }
        });
    }

    private void Build()
    {
        Title = Loc.T("sessions.title");

        var list = new CollectionView
        {
            ItemsSource = _rows,
            ItemTemplate = new DataTemplate(BuildCell),
            SelectionMode = SelectionMode.None,
            BackgroundColor = Theme.Background,
        };

        var newChat = new Button
        {
            Text = Loc.T("sessions.newChat"),
            BackgroundColor = Theme.Accent,
            TextColor = Colors.White,
            CornerRadius = 10,
            Margin = new Thickness(12, 12, 12, 4),
        };
        newChat.Clicked += async (_, _) => await OpenAsync(null);

        var grid = new Grid
        {
            BackgroundColor = Theme.Background,
            RowDefinitions = { new RowDefinition(GridLength.Auto), new RowDefinition(GridLength.Star) },
        };
        grid.Add(newChat, 0, 0);
        grid.Add(list, 0, 1);
        Content = grid;
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();
        try
        {
            _rows.Clear();
            foreach (ConversationSummary summary in _app.Conversations.List())
                _rows.Add(summary);
        }
        catch (Exception ex)
        {
            // Listing the chats reads and parses every file in the conversation store.
            // If one of them cannot be read, an empty list is a page the user reached;
            // an exception here cancels the push instead and drops them on the chat,
            // which is indistinguishable from the menu item not working.
            Console.WriteLine("TensorAgent: the chats list failed to appear: " + ex);
        }
    }

    private View BuildCell()
    {
        var title = new Label { FontSize = 16, TextColor = Theme.Text, LineBreakMode = LineBreakMode.TailTruncation };
        title.SetBinding(Label.TextProperty, nameof(ConversationSummary.Title));

        var meta = new Label { FontSize = 12, TextColor = Theme.Muted };
        meta.SetBinding(Label.TextProperty, new Binding(".", converter: new SummaryLine()));

        var open = new TapGestureRecognizer();
        open.Tapped += async (s, _) =>
        {
            if ((s as View)?.BindingContext is ConversationSummary summary)
                await OpenAsync(summary.Id);
        };

        // Swipe left to delete, which is what a phone list means by "remove". The row
        // used to carry a Delete button and a confirmation dialog: a permanent target
        // for a rare action, sitting on the width the chat's own title needed, on every
        // row. A swipe reveals it, and revealing then tapping is already the two
        // deliberate acts the dialog was standing in for.
        var remove = new SwipeItem
        {
            Text = Loc.T("sessions.row.delete"),
            BackgroundColor = Theme.Danger,
            // Through the command and a bound parameter rather than off the item's own
            // BindingContext: a swipe item is not in the visual tree of the cell it
            // belongs to, and which row it ends up bound to has never been something to
            // rely on. The parameter says which chat, explicitly.
            Command = new Command(parameter =>
            {
                if (parameter is not ConversationSummary summary)
                    return;
                _app.Conversations.Delete(summary.Id);
                _rows.Remove(summary);
            }),
        };
        remove.SetBinding(MenuItem.CommandParameterProperty, ".");

        var body = new Grid();
        body.Add(new VerticalStackLayout { Spacing = 2, Children = { title, meta } }, 0, 0);
        body.GestureRecognizers.Add(open);

        var card = new Border
        {
            Padding = new Thickness(14, 12),
            BackgroundColor = Theme.Surface,
            StrokeThickness = 0,
            StrokeShape = new Microsoft.Maui.Controls.Shapes.RoundRectangle { CornerRadius = 12 },
            Content = body,
        };

        return new SwipeView
        {
            Margin = new Thickness(12, 4),
            // RightItems is what a LEFT swipe reveals: the finger travels left and the
            // item comes in from the right edge, as it does in Mail.
            RightItems = new SwipeItems(new[] { remove }) { Mode = SwipeMode.Reveal },
            Content = card,
        };
    }

    /// <summary>Open the chat page on a conversation, or on a brand new one.</summary>
    private async Task OpenAsync(string? conversationId)
    {
        // Back to the chat FIRST, and only then tell it which conversation. The other
        // order looks tidier and does not work: iOS suspends a WKWebView's content
        // process while its view is off the window, so the script asking the page to
        // switch would be handed to a page that is not running.
        await AppShell.BackToChatAsync();
        _chat.OpenConversation(conversationId);
    }

    private sealed class SummaryLine : IValueConverter
    {
        public object Convert(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo culture)
        {
            if (value is not ConversationSummary summary)
                return string.Empty;
            string when = When(summary.UpdatedAt.ToLocalTime());
            string count = Loc.Plural("sessions.row.messages", summary.MessageCount);
            return summary.ModelId is { Length: > 0 } model ? $"{when} · {count} · {model}" : $"{when} · {count}";
        }

        /// <summary>The date in the language's own pattern; a pattern a translation got wrong
        /// falls back to the English one rather than taking the list down.</summary>
        private static string When(DateTimeOffset local)
        {
            try { return local.ToString(Loc.T("sessions.row.datePattern"), Loc.Culture); }
            catch (FormatException) { return local.ToString("MMM d, HH:mm", Loc.Culture); }
        }

        public object ConvertBack(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo culture)
            => throw new NotSupportedException();
    }
}
