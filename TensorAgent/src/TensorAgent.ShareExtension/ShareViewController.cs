// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using CoreGraphics;
using Foundation;
using ObjCRuntime;
using TensorAgent.Sharing;
using TensorAgent.Sharing.Localization;
using UIKit;
using UserNotifications;

namespace TensorAgent.ShareExtension;

/// <summary>
/// "Ask TensorAgent": the sheet that appears when the user shares something into the
/// app from Safari, Mail, Messages, Notes or anywhere else.
///
/// <para>
/// A plain <see cref="UIViewController"/> rather than
/// <c>SLComposeServiceViewController</c>. The stock class would give the text box and
/// the Post button for free, and it shipped a crash in iOS 26.0 —
/// <c>NSInternalInconsistencyException</c> on <c>_UINavigationBarTitleControl</c>,
/// fixed only in 26.1 — that takes down any extension built on it. It also cannot show
/// the one-tap presets, which on a phone are most of the value: a person sharing an
/// article usually wants "summarise this" and should not have to type it.
/// </para>
/// <para>The extension atomically writes the complete share into the App Group before
/// it closes. iOS does not provide Share extensions an API to foreground their
/// containing app. If notification permission was already granted, a content-free
/// local notification gives the user a supported one-tap way to open TensorAgent;
/// otherwise the sheet clearly says that the item was saved and the app should be
/// opened. The app drains the durable inbox on every launch and foreground.</para>
/// </summary>
[Register("ShareViewController")]
public sealed class ShareViewController : UIViewController
{
    // The app's palette, so the sheet looks like the thing it is sharing into rather
    // than like a system dialog with a logo on it.
    private static readonly UIColor Background = UIColor.FromRGB(0x0b, 0x12, 0x20);
    private static readonly UIColor Card = UIColor.FromRGB(0x13, 0x1c, 0x2e);
    private static readonly UIColor Foreground = UIColor.FromRGB(0xe6, 0xed, 0xfb);
    private static readonly UIColor Muted = UIColor.FromRGB(0x8d, 0x9a, 0xb8);
    private static readonly UIColor Accent = UIColor.FromRGB(0x5b, 0x8c, 0xff);

    /// <summary>
    /// The one-tap prompts. First is the placeholder's own suggestion. Built with the sheet,
    /// once its language is chosen: the prompt becomes the user's own message.
    /// </summary>
    private static (string Label, string Prompt)[] Presets() => new[]
    {
        (ShareStrings.T("share.preset.summarize.label"), ShareStrings.T("share.preset.summarize.prompt")),
        (ShareStrings.T("share.preset.explain.label"), ShareStrings.T("share.preset.explain.prompt")),
        (ShareStrings.T("share.preset.keyPoints.label"), ShareStrings.T("share.preset.keyPoints.prompt")),
        (ShareStrings.T("share.preset.actionItems.label"), ShareStrings.T("share.preset.actionItems.prompt")),
        (ShareStrings.T("share.preset.translate.label"), ShareStrings.T("share.preset.translate.prompt")),
    };

    private UITextView _prompt = null!;
    private UILabel _placeholder = null!;
    private UILabel _summary = null!;
    private UIBarButtonItem _ask = null!;
    private NSLayoutConstraint _bottom = null!;
    private readonly List<NSObject> _observers = new();
    private int _finished;

    public ShareViewController(NativeHandle handle) : base(handle)
    {
    }

    public override void ViewDidLoad()
    {
        base.ViewDidLoad();
        UseAppLanguage();
        BuildUi();
        DescribeShare();

        _observers.Add(NSNotificationCenter.DefaultCenter.AddObserver(
            UIKeyboard.WillChangeFrameNotification, KeyboardChanged));
    }

    public override void ViewDidAppear(bool animated)
    {
        base.ViewDidAppear(animated);
        // The keyboard up front: the whole point of the sheet is the sentence the user
        // is about to type, and a preset is one tap away whether or not it is showing.
        _prompt.BecomeFirstResponder();
    }

    /// <summary>
    /// The language the app shows: the user's choice, which the app leaves in the App Group,
    /// else the system's. Chosen before any text is built; if it cannot be, the sheet is
    /// English rather than missing.
    /// </summary>
    private static void UseAppLanguage()
    {
        try
        {
            NSUrl? container = NSFileManager.DefaultManager.GetContainerUrl(ShareContainer.GroupIdentifier);
            ShareStrings.UseChoiceIn(container?.Path, NSLocale.PreferredLanguages);
        }
        catch (Exception ex)
        {
            Console.WriteLine("TensorAgent.Share: the interface language could not be chosen: " + ex.Message);
        }
    }

    // =====================================================================================
    // the sheet
    // =====================================================================================

    private void BuildUi()
    {
        View!.BackgroundColor = Background;

        var bar = new UINavigationBar { TranslatesAutoresizingMaskIntoConstraints = false };
        bar.BarTintColor = Background;
        bar.BackgroundColor = Background;
        bar.TintColor = Accent;
        bar.Translucent = false;
        bar.TitleTextAttributes = new UIStringAttributes { ForegroundColor = Foreground };

        var title = new UINavigationItem(ShareStrings.T("share.sheet.title"));
        title.LeftBarButtonItem = new UIBarButtonItem(UIBarButtonSystemItem.Cancel, (_, _) => Cancel());
        _ask = new UIBarButtonItem(ShareStrings.T("share.sheet.ask"), UIBarButtonItemStyle.Done, (_, _) => Send());
        title.RightBarButtonItem = _ask;
        bar.SetItems(new[] { title }, false);
        View.AddSubview(bar);

        _summary = new UILabel
        {
            TranslatesAutoresizingMaskIntoConstraints = false,
            TextColor = Muted,
            Font = UIFont.SystemFontOfSize(13)!,
            Lines = 3,
            LineBreakMode = UILineBreakMode.TailTruncation,
            Text = ShareStrings.T("share.sheet.reading"),
        };

        var promptCard = new UIView { TranslatesAutoresizingMaskIntoConstraints = false, BackgroundColor = Card };
        promptCard.Layer.CornerRadius = 12;
        _prompt = new UITextView
        {
            TranslatesAutoresizingMaskIntoConstraints = false,
            BackgroundColor = UIColor.Clear,
            TextColor = Foreground,
            Font = UIFont.SystemFontOfSize(17)!,
            // The composer in the app scrolls; here the sheet is short-lived and a
            // scrolling box inside a sheet inside a share sheet is one scroll too many.
            ScrollEnabled = true,
            KeyboardAppearance = UIKeyboardAppearance.Dark,
            Text = ShareStrings.T("share.sheet.defaultQuestion"),
        };
        _prompt.Changed += (_, _) => _placeholder.Hidden = (_prompt.Text?.Length ?? 0) > 0;
        _placeholder = new UILabel
        {
            TranslatesAutoresizingMaskIntoConstraints = false,
            TextColor = Muted,
            Font = UIFont.SystemFontOfSize(17)!,
            Text = ShareStrings.T("share.sheet.questionPlaceholder"),
        };
        _placeholder.Hidden = true;
        promptCard.AddSubview(_prompt);
        promptCard.AddSubview(_placeholder);

        UIScrollView presets = BuildPresets();

        var stack = new UIStackView(new UIView[] { _summary, promptCard, presets })
        {
            TranslatesAutoresizingMaskIntoConstraints = false,
            Axis = UILayoutConstraintAxis.Vertical,
            Spacing = 14,
        };
        stack.SetCustomSpacing(10, _summary);
        View.AddSubview(stack);

        _bottom = stack.BottomAnchor.ConstraintLessThanOrEqualTo(View.SafeAreaLayoutGuide.BottomAnchor, -12);
        NSLayoutConstraint.ActivateConstraints(new[]
        {
            bar.TopAnchor.ConstraintEqualTo(View.SafeAreaLayoutGuide.TopAnchor),
            bar.LeadingAnchor.ConstraintEqualTo(View.LeadingAnchor),
            bar.TrailingAnchor.ConstraintEqualTo(View.TrailingAnchor),

            stack.TopAnchor.ConstraintEqualTo(bar.BottomAnchor, 14),
            stack.LeadingAnchor.ConstraintEqualTo(View.LeadingAnchor, 16),
            stack.TrailingAnchor.ConstraintEqualTo(View.TrailingAnchor, -16),
            _bottom,

            promptCard.HeightAnchor.ConstraintGreaterThanOrEqualTo(96),
            _prompt.TopAnchor.ConstraintEqualTo(promptCard.TopAnchor, 6),
            _prompt.BottomAnchor.ConstraintEqualTo(promptCard.BottomAnchor, -6),
            _prompt.LeadingAnchor.ConstraintEqualTo(promptCard.LeadingAnchor, 10),
            _prompt.TrailingAnchor.ConstraintEqualTo(promptCard.TrailingAnchor, -10),
            _placeholder.TopAnchor.ConstraintEqualTo(_prompt.TopAnchor, 8),
            _placeholder.LeadingAnchor.ConstraintEqualTo(_prompt.LeadingAnchor, 5),
            _placeholder.TrailingAnchor.ConstraintEqualTo(_prompt.TrailingAnchor, -5),

            presets.HeightAnchor.ConstraintEqualTo(36),
        });
    }

    private UIScrollView BuildPresets()
    {
        var row = new UIStackView
        {
            TranslatesAutoresizingMaskIntoConstraints = false,
            Axis = UILayoutConstraintAxis.Horizontal,
            Spacing = 8,
        };
        foreach ((string label, string prompt) in Presets())
        {
            var chip = new UIButton(UIButtonType.System);
            chip.TranslatesAutoresizingMaskIntoConstraints = false;
            chip.SetTitle(label, UIControlState.Normal);
            chip.SetTitleColor(Foreground, UIControlState.Normal);
            chip.TitleLabel!.Font = UIFont.SystemFontOfSize(14)!;
            chip.BackgroundColor = Card;
            chip.Layer.CornerRadius = 16;
            // UIButton.ContentEdgeInsets is obsolete from iOS 15. A width tied to the
            // title expresses the same 14-point padding without opting the button into
            // a UIButtonConfiguration that would replace the custom colours above.
            chip.WidthAnchor.ConstraintEqualTo(chip.TitleLabel.WidthAnchor, 28).Active = true;
            chip.HeightAnchor.ConstraintEqualTo(32).Active = true;
            // Replaces rather than appends: the chips are alternatives to each other,
            // and tapping two of them should not build a sentence nobody wrote.
            chip.TouchUpInside += (_, _) =>
            {
                _prompt.Text = prompt;
                _placeholder.Hidden = true;
            };
            row.AddArrangedSubview(chip);
        }

        var scroll = new UIScrollView
        {
            TranslatesAutoresizingMaskIntoConstraints = false,
            ShowsHorizontalScrollIndicator = false,
        };
        scroll.AddSubview(row);
        NSLayoutConstraint.ActivateConstraints(new[]
        {
            row.TopAnchor.ConstraintEqualTo(scroll.ContentLayoutGuide.TopAnchor),
            row.BottomAnchor.ConstraintEqualTo(scroll.ContentLayoutGuide.BottomAnchor),
            row.LeadingAnchor.ConstraintEqualTo(scroll.ContentLayoutGuide.LeadingAnchor),
            row.TrailingAnchor.ConstraintEqualTo(scroll.ContentLayoutGuide.TrailingAnchor),
            row.HeightAnchor.ConstraintEqualTo(scroll.FrameLayoutGuide.HeightAnchor),
        });
        return scroll;
    }

    private void KeyboardChanged(NSNotification notification)
    {
        if (notification.UserInfo?[UIKeyboard.FrameEndUserInfoKey] is not NSValue value)
            return;
        CGRect frame = View!.ConvertRectFromView(value.CGRectValue, null);
        nfloat overlap = View.Bounds.GetMaxY() - frame.GetMinY();
        nfloat inset = overlap > 0 ? overlap - View.SafeAreaInsets.Bottom : 0;
        _bottom.Constant = -(12 + NMath.Max(0, inset));
        View.LayoutIfNeeded();
    }

    // =====================================================================================
    // what was shared
    // =====================================================================================

    /// <summary>
    /// A one-line description of what is about to be sent, so the user can see that the
    /// right thing was picked up before they commit to it.
    /// </summary>
    private void DescribeShare()
    {
        NSExtensionItem[] items = ExtensionContext?.InputItems ?? Array.Empty<NSExtensionItem>();
        var parts = new List<string>();
        int files = 0;
        int providersInspected = 0;
        bool moreFiles = false;
        int itemLimit = Math.Min(items.Length, SharePayload.MaxItems);
        for (int itemIndex = 0; itemIndex < itemLimit; itemIndex++)
        {
            NSExtensionItem item = items[itemIndex];
            if (parts.Count < 3)
            {
                string title = Preview(item.AttributedTitle);
                string text = Preview(item.AttributedContentText);
                if (title.Length > 0)
                    parts.Add(title);
                else if (text.Length > 0)
                    parts.Add(text);
            }

            // FILES, not attachments. Several attachments describing one thing is the
            // normal case — Safari sends the extracted page and a plain URL for the same
            // page, Firefox a URL and its title — so counting attachments told the user
            // they were sharing two things when they were sharing one.
            foreach (NSItemProvider provider in item.Attachments ?? Array.Empty<NSItemProvider>())
            {
                if (++providersInspected > SharePayload.MaxItems)
                {
                    moreFiles = true;
                    break;
                }
                if (provider.HasItemConformingTo("public.image")
                    || provider.HasItemConformingTo("public.movie")
                    || provider.HasItemConformingTo("public.audio")
                    || provider.HasItemConformingTo("com.adobe.pdf")
                    || provider.HasItemConformingTo("public.file-url"))
                {
                    files++;
                }
            }
            if (moreFiles)
                break;
        }
        if (items.Length > itemLimit)
            moreFiles = true;
        string what = parts.Count > 0 ? string.Join(" · ", parts) : ShareStrings.T("share.summary.fallback");
        if (moreFiles)
            _summary.Text = ShareStrings.T("share.summary.many", ("what", what));
        else
            _summary.Text = files > 0 ? ShareStrings.Plural("share.summary.files", files, ("what", what)) : what;
    }

    private static string Preview(NSAttributedString? value)
    {
        if (value is null || value.Length <= 0)
            return string.Empty;
        // Avoid Value here: it marshals the complete native string into managed memory
        // before a substring can be taken. The label has three lines and needs at most a
        // small prefix.
        nint length = value.Length < 121 ? value.Length : 121;
        using NSAttributedString prefix = value.Substring(0, length);
        return ShareText.FirstLine(prefix.Value, 120);
    }

    // =====================================================================================
    // sending
    // =====================================================================================

    private void Cancel()
    {
        if (Interlocked.Exchange(ref _finished, 1) == 1)
            return;
        ExtensionContext?.CompleteRequest(Array.Empty<NSExtensionItem>(), null);
    }

    private void Send()
    {
        if (Interlocked.Exchange(ref _finished, 1) == 1)
            return;
        _ask.Enabled = false;
        _prompt.ResignFirstResponder();
        string typed = _prompt.Text ?? string.Empty;
        _ = SendAsync(typed);
    }

    private async Task SendAsync(string typed)
    {
        string id;
        try
        {
            id = await BuildShareAsync(typed).ConfigureAwait(true);
        }
        catch (Exception ex)
        {
            Console.WriteLine("TensorAgent.Share: " + ex);
            ShowFailure(ex is ShareInboxFullException
                ? ShareStrings.T("share.error.inboxFull")
                : ShareStrings.T("share.error.saveFailed"));
            return;
        }

        await FinishPersistedShareAsync(id).ConfigureAwait(true);
    }

    /// <summary>
    /// Write the complete share to the App Group and return its durable envelope id.
    /// A missing container is a signing/configuration error, never a reason to put
    /// private shared content in a custom URL or to pretend the handoff succeeded.
    /// </summary>
    private async Task<string> BuildShareAsync(string typed)
    {
        string? inbox = SharedInbox();
        if (inbox is null)
            throw new InvalidOperationException("The TensorAgent App Group is not available in this build.");

        var store = new ShareEnvelopeStore(inbox);
        store.Purge(DateTimeOffset.UtcNow);
        ShareEnvelopeWriter? writer = null;
        try
        {
            writer = store.BeginWrite();
            long remaining = Math.Max(0, store.MaxTotalBytes - store.TotalBytes());
            SharePayload payload = await ShareItemReader
                .ReadAsync(
                    ExtensionContext?.InputItems ?? Array.Empty<NSExtensionItem>(),
                    writer,
                    sourceApp: string.Empty,
                    maxTotalFileBytes: remaining)
                .ConfigureAwait(true);
            payload.Prompt = typed;
            // Each committed envelope is a separate user action and maps to one fresh
            // conversation. There is intentionally no UI switch that can make two
            // independent shares accumulate in a single chat.
            payload.NewChat = true;
            payload.AutoSend = false;

            if (payload.Items.Count == 0 && payload.Notes.Count == 0)
            {
                throw new InvalidOperationException("The sharing app did not provide readable text, a URL, or a supported file.");
            }

            return writer.Commit(payload);
        }
        catch
        {
            writer?.Abandon();
            throw;
        }
    }

    private async Task FinishPersistedShareAsync(string id)
    {
        bool notified = false;
        try
        {
            UNUserNotificationCenter center = UNUserNotificationCenter.Current;
            UNNotificationSettings settings = await center.GetNotificationSettingsAsync().ConfigureAwait(true);
            if (settings.AuthorizationStatus is UNAuthorizationStatus.Authorized or UNAuthorizationStatus.Provisional
                && settings.AlertSetting == UNNotificationSetting.Enabled)
            {
                using var content = new UNMutableNotificationContent
                {
                    Title = ShareStrings.T("share.notification.title"),
                    Body = ShareStrings.T("share.notification.body"),
                };
                using UNTimeIntervalNotificationTrigger trigger =
                    UNTimeIntervalNotificationTrigger.CreateTrigger(1, false);
                center.RemoveDeliveredNotifications(new[] { ShareContainer.NotificationIdentifier });
                using UNNotificationRequest request = UNNotificationRequest.FromIdentifier(
                    ShareContainer.NotificationIdentifier, content, trigger);
                await center.AddNotificationRequestAsync(request).ConfigureAwait(true);
                notified = true;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("TensorAgent.Share: could not schedule the open-app notification: " + ex.Message);
        }

        var alert = UIAlertController.Create(
            ShareStrings.T("share.saved.title"),
            notified
                ? ShareStrings.T("share.saved.notified")
                : ShareStrings.T("share.saved.openApp"),
            UIAlertControllerStyle.Alert);
        alert.AddAction(UIAlertAction.Create(ShareStrings.T("share.saved.done"), UIAlertActionStyle.Default, _ =>
            ExtensionContext?.CompleteRequest(Array.Empty<NSExtensionItem>(), null)));
        PresentViewController(alert, true, null);
    }

    private void ShowFailure(string message)
    {
        Interlocked.Exchange(ref _finished, 0);
        _ask.Enabled = true;
        _summary.Text = message;
        _summary.TextColor = UIColor.SystemRed;
    }

    private static string? SharedInbox()
    {
        try
        {
            NSUrl? container = NSFileManager.DefaultManager.GetContainerUrl(ShareContainer.GroupIdentifier);
            if (container?.Path is not { Length: > 0 } root)
                return null;
            string inbox = Path.Combine(root, ShareContainer.InboxDirectoryName);
            Directory.CreateDirectory(inbox);
            using (NSUrl inboxUrl = NSUrl.FromFilename(inbox))
                inboxUrl.SetResource(NSUrl.IsExcludedFromBackupKey, NSNumber.FromBoolean(true), out _);
            return inbox;
        }
        catch (Exception ex)
        {
            Console.WriteLine("TensorAgent.Share: no shared container: " + ex.Message);
            return null;
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            foreach (NSObject observer in _observers)
                NSNotificationCenter.DefaultCenter.RemoveObserver(observer);
            _observers.Clear();
        }
        base.Dispose(disposing);
    }
}
