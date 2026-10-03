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
using System.Text;
using TensorAgent.Sharing.Localization;

namespace TensorAgent.Sharing;

/// <summary>
/// How much of a share the model is shown.
///
/// <para>
/// The numbers are a phone's, not a server's. The engine prefills in 1024-token
/// chunks here because a bigger chunk got the process killed on a 12 GB iPhone (see
/// <c>MauiProgram</c>), and every character of shared context is a character the user
/// did not choose to send: a page they wanted one question answered about can easily
/// be 200 kB. 24,000 characters is roughly six thousand tokens, which is under a
/// minute of prefill on a device and still enough for a long article's substance.
/// </para>
/// </summary>
public sealed record ShareCompositionOptions
{
    /// <summary>Total characters of shared text the message may carry.</summary>
    public int MaxTotalChars { get; init; } = 24_000;

    /// <summary>Characters any single shared item may carry, before the total is applied.</summary>
    public int MaxCharsPerItem { get; init; } = 12_000;

    /// <summary>
    /// Characters of a page SELECTION that are shown in full before it is shortened.
    ///
    /// <para>
    /// Higher than the page budget on purpose, and it is the most important number
    /// here. A selection is the one part of a share the user pointed at; shortening
    /// what they deliberately highlighted in order to make room for the page around it
    /// would be exactly backwards.
    /// </para>
    /// </summary>
    public int MaxSelectionChars { get; init; } = 16_000;

    /// <summary>
    /// Characters of the user's OWN question that are carried.
    ///
    /// <para>
    /// Separate from <see cref="MaxTotalChars"/>, which bounds the shared content, so
    /// that the two cannot take each other's room — but not unbounded, which is what it
    /// was: the share sheet's text view holds whatever is pasted into it, and a prompt
    /// was going into the message in full and charged against nothing at all.
    /// </para>
    /// </summary>
    public int MaxPromptChars { get; init; } = 4_000;

    /// <summary>
    /// Characters of a shared ADDRESS that are carried.
    ///
    /// <para>
    /// A URL longer than this is not something a model can use, and they get very long:
    /// a <c>data:</c> URL, or one of the tracking redirects that carry an entire encoded
    /// request. Before this existed a link item was appended whole and only then charged
    /// against the budget, so one share could produce a message twelve times the cap.
    /// </para>
    /// </summary>
    public int MaxUrlChars { get; init; } = 512;

    /// <summary>The defaults.</summary>
    public static ShareCompositionOptions Default { get; } = new();
}

/// <summary>The message a share becomes, and everything the caller must do with it.</summary>
/// <param name="Message">What goes into the composer.</param>
/// <param name="Files">The envelope items to attach, in order.</param>
/// <param name="Notices">Sentences to show the user once: what was shortened, what was dropped.</param>
/// <param name="SuggestedTitle">A name for the chat this share starts, better than its first line.</param>
/// <param name="UsedDefaultPrompt">Whether the user typed nothing and a default was supplied.</param>
/// <param name="HasSharedContent">
/// Whether any shared TEXT reached the message.
///
/// <para>
/// False is not the same as an empty message: the message still carries the question,
/// typed or default. It means the thing the question is ABOUT is not there — a page
/// that turned out to be blank, a selection that was whitespace.
/// </para>
/// <para>
/// Text only, deliberately. Whether the FILES arrived is not knowable here: they are
/// uploaded afterwards and any of them can be refused, so the caller tests this against
/// the attachments it actually got rather than against the ones the envelope promised.
/// </para>
/// </param>
public sealed record ShareComposition(
    string Message,
    IReadOnlyList<ShareItem> Files,
    IReadOnlyList<string> Notices,
    string SuggestedTitle,
    bool UsedDefaultPrompt,
    bool HasSharedContent);

/// <summary>
/// Turns a <see cref="SharePayload"/> into the one message the app sends.
///
/// <para>
/// Pure, and that is the point of it existing separately: this is where the quality of
/// the whole feature is decided — what the model is told it is looking at, in what
/// order, and how much of it — and none of that should need a device, a model or a
/// share sheet to check. Everything with a platform in it lives on the other side of
/// this call.
/// </para>
/// <para>
/// The output is an ORDINARY chat message with ordinary attachments. Nothing about a
/// share reaches the inference path: no new prompt role, no marker the pipeline has to
/// know about, no second code path through the chat service. A shared page is a
/// message a very fast typist could have written, which is why adding this feature
/// cannot regress a conversation that has nothing to do with it.
/// </para>
/// </summary>
public static class ShareComposer
{
    /// <summary>
    /// The least budget worth spending on one shared item. See the loop in
    /// <see cref="Compose"/>.
    /// </summary>
    private const int MinimumUsefulChars = 400;

    /// <summary>Whether an item fits in the floor regardless, so a short note at the
    /// end of a long share is not dropped for want of room it does not need.</summary>
    private static bool ItemIsShort(ShareItem item)
        => item.Text.Length + item.Selection.Length + item.Url.Length < MinimumUsefulChars;

    /// <summary>Compose the message for <paramref name="payload"/>.</summary>
    public static ShareComposition Compose(
        SharePayload payload,
        ShareCompositionOptions? options = null,
        IReadOnlyList<ShareItem>? filesToDescribe = null)
    {
        ArgumentNullException.ThrowIfNull(payload);
        payload.NormalizeNulls();
        ShareCompositionOptions opts = options ?? ShareCompositionOptions.Default;

        var notices = new List<string>(payload.Notes);
        // Importers pass only files that actually made it through the upload service.
        // Pure callers omit the argument and get the payload's complete file list.
        var files = filesToDescribe?.ToList() ?? payload.FileItems.ToList();
        List<ShareItem> textual = payload.TextualItems.ToList();

        string prompt = ShareText.Normalize(payload.Prompt).Trim();
        bool usedDefault = prompt.Length == 0;
        if (usedDefault)
            prompt = DefaultPromptFor(textual, files);
        ShareText.ShortenedText promptPart = ShareText.Shorten(prompt, opts.MaxPromptChars);
        if (promptPart.WasShortened)
            notices.Add(ShareStrings.T("share.notice.questionShortened", Fitted(promptPart)));

        var body = new StringBuilder();
        body.Append(promptPart.Text);
        bool anyContent = false;

        // The budget is spent item by item in the order the source app offered them,
        // which is the order the user sees in the share sheet's preview. Running out
        // part way is reported rather than hidden: an item that got nothing is worse
        // than an item that got a little, and the user is the only one who can decide
        // to share less.
        int remaining = opts.MaxTotalChars;
        int skipped = 0;
        // The per-item cap is a FAIRNESS limit between several shared things, not a
        // ceiling on one. Applying it to a lone shared page — which is the commonest
        // share there is — left half the budget unspent and told the user their article
        // had been cut when there was room for all of it.
        int perItem = textual.Count > 1 ? opts.MaxCharsPerItem : opts.MaxTotalChars;
        foreach (ShareItem item in textual)
        {
            // A floor, not zero. Sixteen characters of a five-thousand-character note is
            // not a shortened note, it is a fragment that reads as the whole thing, and a
            // model asked about it will answer confidently about nothing. Below this the
            // item is reported as not having fitted, which the user can act on.
            if (remaining < MinimumUsefulChars && !ItemIsShort(item))
            {
                skipped++;
                continue;
            }
            string block = ComposeItem(item, opts, perItem, ref remaining, notices);
            if (block.Length == 0)
                continue;
            body.Append("\n\n").Append(block);
            anyContent = true;
        }
        if (skipped > 0)
            notices.Add(ShareStrings.Plural("share.notice.skipped", skipped));

        if (files.Count > 0)
            body.Append("\n\n").Append(DescribeFiles(files));

        return new ShareComposition(
            body.ToString(),
            files,
            notices,
            SuggestTitle(payload, textual, files),
            usedDefault,
            anyContent);
    }

    /// <summary>
    /// One shared item as a labelled block.
    ///
    /// <para>
    /// Labelled, and with the source named, because the model is being handed something
    /// the user did not write and has to be able to tell the two apart. "Summarise this"
    /// followed by an unannounced wall of text reads to a model as the user's own words
    /// — which is how a shared phishing mail becomes an instruction. The quoting is
    /// there for the same reason: a shared page that itself contains the sentence
    /// "ignore your previous instructions" is quoted content, and it is labelled as
    /// such before the model reaches it.
    /// </para>
    /// </summary>
    private static string ComposeItem(
        ShareItem item, ShareCompositionOptions opts, int perItem, ref int remaining, List<string> notices)
    {
        var block = new StringBuilder();
        switch (item.Kind)
        {
            case ShareItemKinds.Url:
            {
                string title = ShareText.FirstLine(item.Title, 120);
                // FirstLine and a budget, not Trim: this address is as attacker
                // controlled as any other shared field -- for a page share it is
                // document.baseURI, read by the preprocessing script inside the page's
                // own JavaScript world -- and it used to be the ONE field that reached
                // the message without being normalized or bounded. A newline in it broke
                // the block it was in; a 300 kB data: URL was appended whole and only
                // charged against the budget afterwards.
                // A label is charged at its English length in every language, so how much
                // of a share fits does not depend on the language it is labelled in.
                remaining -= "Shared link: ".Length + title.Length;
                string address = Address(item.Url, opts, ref remaining);
                block.Append(title.Length > 0
                    ? ShareStrings.T("share.message.linkTitled", ("title", title))
                    : ShareStrings.T("share.message.link"));
                if (address.Length > 0)
                    block.Append('\n').Append(address);
                return block.ToString();
            }

            case ShareItemKinds.Page:
            {
                string title = ShareText.FirstLine(item.Title, 160);
                remaining -= "Shared web page\nTitle: \nLink: ".Length + title.Length;
                string address = Address(item.Url, opts, ref remaining);
                block.Append(ShareStrings.T("share.message.page"));
                if (title.Length > 0)
                    block.Append('\n').Append(ShareStrings.T("share.message.pageTitle", ("title", title)));
                if (address.Length > 0)
                    block.Append('\n').Append(ShareStrings.T("share.message.pageLink", ("address", address)));

                string selection = ShareText.Normalize(item.Selection);
                string page = ShareText.Normalize(item.Text);
                // The page body is innerText and CONTAINS the selection, so quoting both
                // sends the highlighted sentence twice -- paying twice for the scarcest
                // thing in the app, and telling the model something matters by repeating
                // it rather than by saying so.
                if (selection.Length > 0 && page.Length > 0)
                    page = WithoutSelection(page, selection);

                if (selection.Length > 0)
                {
                    ShareText.ShortenedText part = Take(selection, Math.Min(opts.MaxSelectionChars, remaining), ref remaining);
                    if (part.Text.Length > 0)
                    {
                        block.Append("\n\n").Append(ShareStrings.T("share.message.selection"))
                             .Append('\n').Append(Quote(part.Text));
                        if (part.WasShortened)
                            notices.Add(ShareStrings.T("share.notice.selectionShortened", Fitted(part)));
                    }
                }

                if (page.Length > 0 && remaining > 0)
                {
                    ShareText.ShortenedText part = Take(page, Math.Min(perItem, remaining), ref remaining);
                    if (part.Text.Length > 0)
                    {
                        block.Append("\n\n")
                             .Append(selection.Length > 0
                                 ? ShareStrings.T("share.message.restOfPage")
                                 : ShareStrings.T("share.message.pageText"))
                             .Append('\n').Append(Quote(part.Text));
                        if (part.WasShortened)
                            notices.Add(ShareStrings.T("share.notice.pageShortened", Fitted(part)));
                    }
                }
                else if (page.Length == 0 && selection.Length == 0)
                {
                    // A browser that offered a page but no text. Saying so is worth a
                    // line: it is the difference between "the model ignored my page"
                    // and "your browser did not hand the page over".
                    block.Append("\n\n").Append(ShareStrings.T("share.message.noPageText"));
                }
                return block.ToString();
            }

            default:
            {
                string text = ShareText.Normalize(item.Text);
                if (text.Length == 0 || remaining <= 0)
                    return string.Empty;
                string title = ShareText.FirstLine(item.Title, 120);
                remaining -= "Shared text -- :\n".Length + title.Length;
                ShareText.ShortenedText part = Take(text, Math.Min(perItem, remaining), ref remaining);
                if (part.Text.Length == 0)
                    return string.Empty;
                block.Append(title.Length > 0
                    ? ShareStrings.T("share.message.textTitled", ("title", title))
                    : ShareStrings.T("share.message.text"));
                block.Append('\n').Append(Quote(part.Text));
                if (part.WasShortened)
                    notices.Add(ShareStrings.T("share.notice.textShortened", Fitted(part)));
                return block.ToString();
            }
        }
    }

    /// <summary>
    /// A shared address, normalized to one line, bounded, and charged against the budget.
    /// </summary>
    private static string Address(string url, ShareCompositionOptions opts, ref int remaining)
    {
        string address = ShareText.FirstLine(url, Math.Max(0, Math.Min(opts.MaxUrlChars, remaining)));
        remaining -= address.Length;
        return address;
    }

    /// <summary>
    /// The page with the selected passage taken out, joined by a marker, or the page
    /// unchanged when the selection is not found in it.
    ///
    /// <para>
    /// Not found is a normal outcome rather than a failure: a selection that spans
    /// elements makes the browser's own <c>toString()</c> and its <c>innerText</c>
    /// disagree about the whitespace between them. Leaving the page whole is the right
    /// answer then — the cost is a duplicated passage, and a fuzzy match could remove
    /// something the user did not select.
    /// </para>
    /// </summary>
    private static string WithoutSelection(string page, string selection)
    {
        // Below this a "selection" is a word or two, and cutting every occurrence of it
        // out of the page would take sentences with it.
        if (selection.Length < 40)
            return page;
        int at = page.IndexOf(selection, StringComparison.Ordinal);
        if (at < 0)
            return page;
        string before = page[..at].TrimEnd();
        string after = page[(at + selection.Length)..].TrimStart();
        if (before.Length == 0)
            return after;
        if (after.Length == 0)
            return before;
        return before + "\n\n" + ShareStrings.T("share.message.selectionQuotedAbove") + "\n\n" + after;
    }

    private static ShareText.ShortenedText Take(string text, int budget, ref int remaining)
    {
        ShareText.ShortenedText part = ShareText.Shorten(text, Math.Max(0, budget));
        remaining -= part.Text.Length;
        return part;
    }

    /// <summary>The placeholders of an "only part of … fitted" notice: characters sent, of how many.</summary>
    private static (string Name, object? Value)[] Fitted(ShareText.ShortenedText part)
    {
        CultureInfo culture = ShareStrings.Culture;
        return new (string Name, object? Value)[]
        {
            ("sent", part.Text.Length.ToString("N0", culture)),
            ("total", part.OriginalLength.ToString("N0", culture)),
        };
    }

    /// <summary>
    /// Fence shared content so the model can see where it starts and stops.
    ///
    /// <para>
    /// A triple quote rather than a markdown code fence: the shared thing is prose far
    /// more often than code, and a code fence makes a model answer about the formatting.
    /// The fence is escaped out of the content, so text that contains one cannot close
    /// the block early.
    /// </para>
    /// </summary>
    private static string Quote(string text)
    {
        const string Fence = "\"\"\"";
        // A LOOP, not one Replace. Replace matches non-overlapping and left to right, so
        // six consecutive quotes came out as `""\u200b"` + `""\u200b"`, which contains a
        // bare triple quote at index three. Content that can close its own fence is
        // content a model reads as the user's own instructions, which is the single
        // thing this labelled-and-quoted arrangement exists to prevent.
        string safe = text;
        while (safe.Contains(Fence, StringComparison.Ordinal))
            safe = safe.Replace(Fence, "\"\"\u200b\"", StringComparison.Ordinal);
        return Fence + "\n" + safe + "\n" + Fence;
    }

    private static string DescribeFiles(IReadOnlyList<ShareItem> files)
    {
        if (files.Count == 1)
            return ShareStrings.T("share.message.file", ("file", Describe(files[0])));
        var lines = new StringBuilder(ShareStrings.T("share.message.files"));
        foreach (ShareItem file in files)
            lines.Append("\n- ").Append(Describe(file));
        return lines.ToString();

        static string Describe(ShareItem file)
        {
            string name = ShareEnvelopeWriter.SafeFileName(file.FileName);
            return file.Bytes > 0
                ? ShareStrings.T("share.message.fileWithSize", ("name", name), ("size", Bytes(file.Bytes)))
                : name;
        }
    }

    private static string Bytes(long bytes)
    {
        CultureInfo culture = ShareStrings.Culture;
        return bytes switch
        {
            >= 1024L * 1024 * 1024 => ShareStrings.T("share.message.size.gb",
                ("size", (bytes / (1024.0 * 1024 * 1024)).ToString("0.#", culture))),
            >= 1024 * 1024 => ShareStrings.T("share.message.size.mb",
                ("size", (bytes / (1024.0 * 1024)).ToString("0.#", culture))),
            >= 1024 => ShareStrings.T("share.message.size.kb",
                ("size", (bytes / 1024.0).ToString("0.#", culture))),
            _ => ShareStrings.T("share.message.size.bytes", ("size", bytes)),
        };
    }

    /// <summary>
    /// What to ask when the user asked nothing.
    ///
    /// <para>
    /// They tapped "Ask TensorAgent" and typed nothing, which is a request for the
    /// assistant to look at the thing and be useful about it — not for a blank turn.
    /// The wording differs by what was shared because the useful question does: a photo
    /// is not a page, and asking "summarise this" of an image reads as a mistake.
    /// </para>
    /// </summary>
    public static string DefaultPromptFor(IReadOnlyList<ShareItem> textual, IReadOnlyList<ShareItem> files)
    {
        bool hasPageText = textual.Any(i =>
            i.Kind == ShareItemKinds.Page && (i.Text.Trim().Length > 0 || i.Selection.Trim().Length > 0));
        bool hasText = textual.Any(i => i.Kind == ShareItemKinds.Text && i.Text.Trim().Length > 0);
        bool hasLinkOnly = textual.Any(i => i.Kind == ShareItemKinds.Url)
            || textual.Any(i => i.Kind == ShareItemKinds.Page && i.Text.Trim().Length == 0 && i.Selection.Trim().Length == 0);

        if (hasPageText)
            return ShareStrings.T("share.message.prompt.page");
        if (hasText)
            return ShareStrings.T("share.message.prompt.text");
        if (hasLinkOnly && files.Count == 0)
        {
            // NOT "what can you tell me about this link". Nothing in this app fetches a
            // URL — there is no such tool, and the network switch is off by default —
            // and every non-Safari browser lands here, because Chrome, Edge, Firefox and
            // Outlook vend the address and nothing else. Asking a small local model
            // about a page it cannot read is asking it to invent one.
            return ShareStrings.T("share.message.prompt.link");
        }
        if (files.Count > 0 && textual.Count == 0)
            return files.Count == 1
                ? ShareStrings.T("share.message.prompt.file")
                : ShareStrings.T("share.message.prompt.files");
        return ShareStrings.T("share.message.prompt.anything");
    }

    /// <summary>
    /// A name for the chat a share starts.
    ///
    /// <para>
    /// The chats list otherwise derives a title from the message's first line, which
    /// for a share with no typed prompt is the default instruction — so every share
    /// the user did not type into would be called the same thing. The shared thing's
    /// own title is what they will recognise.
    /// </para>
    /// </summary>
    private static string SuggestTitle(
        SharePayload payload, IReadOnlyList<ShareItem> textual, IReadOnlyList<ShareItem> files)
    {
        foreach (ShareItem item in textual)
        {
            string title = ShareText.FirstLine(item.Title, 60);
            if (title.Length > 0)
                return title;
        }
        foreach (ShareItem item in textual)
        {
            string first = ShareText.FirstLine(
                item.Selection.Trim().Length > 0 ? item.Selection : item.Text, 60);
            if (first.Length > 0)
                return first;
            string url = item.Url.Trim();
            if (url.Length > 0)
                return ShareText.FirstLine(url, 60);
        }
        if (files.Count > 0)
            return ShareEnvelopeWriter.SafeFileName(files[0].FileName);
        string source = ShareText.FirstLine(payload.SourceApp, 40);
        return source.Length > 0
            ? ShareStrings.T("share.chat.sharedFrom", ("app", source))
            : ShareStrings.T("share.chat.shared");
    }
}
