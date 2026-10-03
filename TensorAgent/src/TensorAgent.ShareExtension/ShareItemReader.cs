// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Foundation;
using TensorAgent.Sharing;
using TensorAgent.Sharing.Localization;
using UIKit;

namespace TensorAgent.ShareExtension;

/// <summary>
/// Turns whatever the sharing app handed over into a <see cref="SharePayload"/>.
///
/// <para>
/// This is the part of the feature that has to know about the world. The activation
/// rule in <c>Info.plist</c> is deliberately permissive — it has to be, or the extension
/// silently disappears from one app's share sheet and not another's — and the price of
/// a permissive rule is that this code is handed shapes it did not ask for and must
/// degrade rather than fail. What the source apps actually vend, from their own source
/// where it exists:
/// </para>
/// <list type="bullet">
/// <item>Safari, whole page: a <c>com.apple.property-list</c> item carrying the
/// dictionary <c>ExtensionPreprocessing.js</c> built inside the page — title, URL,
/// selection and the article's text. It is the ONLY app that runs that file.</item>
/// <item>Chrome and Edge: one <c>public.url</c> item, page title in
/// <c>attributedTitle</c>, built eagerly from the URL with no page access. Chromium's
/// <c>chrome_activity_url_source.mm</c> is explicit about this.</item>
/// <item>Firefox: TWO attachments, a <c>public.url</c> and a separate
/// <c>public.plain-text</c> holding the title.</item>
/// <item>Messages, Notes, Mail: selected text as <c>public.plain-text</c>, attachments
/// as their concrete type plus <c>public.file-url</c>, a mail message as an
/// <c>.eml</c> file URL, a note as <c>com.apple.rtfd</c>.</item>
/// <item>Photos: inconsistent BY DESIGN — sometimes a file URL, sometimes a
/// <c>UIImage</c>, sometimes <c>NSData</c>. All three are handled.</item>
/// </list>
/// <para>
/// Two rules govern every line here, and both are memory. A share extension runs under
/// roughly a 120 MB jetsam ceiling — a fraction of an app's — so a shared video is
/// copied file-to-file and never read into a buffer. And the temporary file
/// <c>loadFileRepresentation</c> hands over is deleted BY THE SYSTEM WHEN THE COMPLETION
/// HANDLER RETURNS, which makes the <c>Async</c> overload a trap in .NET: the await
/// continuation resumes after the handler returned, by which time the file is gone. The
/// callback overload is used, and the copy happens inside it.
/// </para>
/// </summary>
internal static class ShareItemReader
{
    // The uniform type identifiers, as strings. Written out rather than taken from
    // UTType so that the probe ORDER below is visible in one place: it is load-bearing,
    // because UTI conformance means a provider that registered a specific type will
    // happily answer a request for its ancestor and give back something less useful.
    private const string PropertyList = "com.apple.property-list";
    private const string Url = "public.url";
    private const string FileUrl = "public.file-url";
    private const string PlainText = "public.plain-text";
    private const string Text = "public.text";
    private const string Html = "public.html";
    private const string Rtf = "public.rtf";
    private const string Rtfd = "com.apple.rtfd";
    private const string Image = "public.image";
    private const string Movie = "public.movie";
    private const string Audio = "public.audio";
    private const string Pdf = "com.adobe.pdf";
    private const string Data = "public.data";
    private const string Zip = "public.zip-archive";

    /// <summary>The literal key Safari puts its preprocessing result under.</summary>
    private const string JavaScriptResults = "NSExtensionJavaScriptPreprocessingResultsKey";

    /// <summary>
    /// What <c>ExtensionPreprocessing.js</c> puts where it cut the middle out of a long page
    /// or selection: English, because the script runs inside the page, out of reach of the
    /// string tables. <see cref="WithOwnMarker"/> puts the interface language's in its place.
    /// </summary>
    private const string ScriptMiddleShortenedMarker = "\n\n… [middle shortened by TensorAgent] …\n\n";

    /// <summary>
    /// The biggest single file this will copy.
    ///
    /// <para>
    /// The copy itself costs no memory, but the app has to read the file afterwards and
    /// the model has to do something with it, and a 400 MB video is not a question
    /// anyone is asking a phone. Refused with a sentence the app shows, rather than
    /// dropped.
    /// </para>
    /// </summary>
    private const long MaxFileBytes = 128L * 1024 * 1024;

    /// <summary>Markup must be decoded whole; beyond this it is refused, not spliced.</summary>
    private const long MaxRichTextSourceBytes = 4L * 1024 * 1024;

    /// <summary>A file URL is a path reference, never a payload.</summary>
    private const ulong MaxFileUrlRepresentationBytes = 64UL * 1024;

    /// <summary>
    /// A URL representation is tiny in legitimate shares. UIKit sometimes writes
    /// <c>public.url</c> as a binary property-list scalar/array instead of raw UTF-8; that
    /// format must be read whole so Foundation can validate its trailer.
    /// </summary>
    private const long MaxUrlRepresentationBytes = 64L * 1024;

    /// <summary>
    /// A keyed archive has to be decoded whole. Legitimate archived shared text is
    /// only a small wrapper around the string; keep the allocation far below the
    /// extension's jetsam budget even when a provider lies about its type.
    /// </summary>
    private const long MaxArchivedTextRepresentationBytes = 4L * 1024 * 1024;

    /// <summary>A package walk is hostile input too; bound both CPU and path storage.</summary>
    private const int MaxPackageEntries = 1_024;

    /// <summary>Total across all files in one share.</summary>
    private const long MaxTotalBytes = 256L * 1024 * 1024;

    /// <summary>
    /// Read everything in <paramref name="items"/>, copying any files into
    /// <paramref name="writer"/>'s envelope.
    /// </summary>
    /// <param name="writer">Where files are copied, or null for a share that will travel in a URL.</param>
    public static async Task<SharePayload> ReadAsync(
        NSExtensionItem[] items, ShareEnvelopeWriter? writer, string sourceApp,
        long maxTotalFileBytes = MaxTotalBytes)
    {
        var payload = new SharePayload
        {
            SourceApp = sourceApp ?? string.Empty,
            CreatedAt = DateTimeOffset.UtcNow,
        };

        long total = 0;
        bool itemLimitSaid = false;
        NSExtensionItem[] inputItems = items ?? Array.Empty<NSExtensionItem>();
        int inputLimit = Math.Min(inputItems.Length, SharePayload.MaxItems);
        for (int itemIndex = 0; itemIndex < inputLimit; itemIndex++)
        {
            if (payload.Items.Count >= SharePayload.MaxItems)
            {
                if (!itemLimitSaid)
                {
                    payload.Notes.Add(ShareStrings.T("share.note.tooManyItems"));
                    itemLimitSaid = true;
                }
                break;
            }
            NSExtensionItem item = inputItems[itemIndex];
            // The item's own text, which some hosts use INSTEAD of an attachment. Read
            // before the attachments so a title arrives before the thing it titles. Use
            // native ranges rather than Value: Value first allocates the entire NSString,
            // defeating every managed bound for a hostile-sized attributed string.
            BoundedLoadedText titleLoaded = ReadAttributedText(item.AttributedTitle, SharePayload.MaxTitleChars);
            BoundedLoadedText contentLoaded = ReadAttributedText(item.AttributedContentText, 120_000);
            string title = titleLoaded.Text ?? string.Empty;
            string contentText = contentLoaded.Text ?? string.Empty;
            if (titleLoaded.Truncated)
                payload.Notes.Add(ShareStrings.T("share.note.titleShortened", ("characters", Chars(titleLoaded.OriginalChars))));
            if (contentLoaded.Truncated)
                payload.Notes.Add(ShareStrings.T("share.note.textShortened", ("characters", Chars(contentLoaded.OriginalChars))));

            NSItemProvider[] providers = item.Attachments ?? Array.Empty<NSItemProvider>();

            foreach (NSItemProvider provider in providers)
            {
                if (payload.Items.Count >= SharePayload.MaxItems)
                {
                    if (!itemLimitSaid)
                    {
                        payload.Notes.Add(ShareStrings.T("share.note.tooManyItems"));
                        itemLimitSaid = true;
                    }
                    break;
                }
                // The running total is passed IN so a file that would overrun the cap is
                // refused before it is copied. Checking afterwards, which is what this
                // did, wrote every byte into the envelope and then dropped the item that
                // named it — leaving the copy on the user's disk with nothing pointing at
                // it, and doing all the work anyway.
                ShareItem? read = await ReadProviderAsync(
                    provider, writer, title, payload, total,
                    Math.Clamp(maxTotalFileBytes, 0, MaxTotalBytes)).ConfigureAwait(false);
                if (read is null)
                    continue;
                total += read.Bytes;
                payload.Items.Add(read);
            }

            // Content text can be a caption accompanying a photo/file, not merely a
            // fallback when no attachment loaded. Preserve it unless an attachment
            // already supplied exactly the same text.
            if (payload.Items.Count < SharePayload.MaxItems
                && contentText.Trim().Length > 0
                && !payload.Items.Any(existing =>
                    existing.Kind == ShareItemKinds.Text
                    && string.Equals(existing.Text.Trim(), contentText.Trim(), StringComparison.Ordinal)))
                payload.Items.Add(ShareItem.ForText(contentText, title));
        }
        if (inputItems.Length > inputLimit && !itemLimitSaid)
            payload.Notes.Add(ShareStrings.T("share.note.tooManyItems"));

        // Several attachments describing ONE thing is the normal case, not the odd one:
        // Safari sends the extracted page AND a plain public.url for it, Firefox sends a
        // URL and a separate text item holding the title, and many apps add a text
        // preview beside a link. Left alone the model is handed the same page twice and
        // asked to summarise both.
        FoldDuplicateLinks(payload);
        FoldProvenDuplicateLinkText(payload);
        payload.ConstrainForWire();
        return payload;
    }

    private static async Task<ShareItem?> ReadProviderAsync(
        NSItemProvider provider, ShareEnvelopeWriter? writer, string title, SharePayload payload,
        long soFar, long totalLimit)
    {
        // ---- Safari's page extraction, first and always -------------------------
        // It is the richest thing any app hands over and it is only ever offered
        // alongside a plain public.url, so asking for the URL first would throw the
        // page away.
        if (LooksLikeSafariPreprocessingProvider(provider))
        {
            ShareItem? page = await ReadWebPageAsync(provider, payload).ConfigureAwait(false);
            if (page is not null)
                return page;
        }

        // ---- a link -------------------------------------------------------------
        // Checked before file-url because a web URL conforms to public.url and a FILE
        // url conforms to both; the file branch below re-checks and wins for files.
        if (provider.HasItemConformingTo(Url) && !provider.HasItemConformingTo(FileUrl))
        {
            BoundedLoadedText loaded = await LoadUrlBoundedAsync(provider).ConfigureAwait(false);
            if (loaded.Refusal is { Length: > 0 } refusal)
            {
                payload.Notes.Add(refusal);
                return null;
            }
            if (loaded.Text?.Trim() is { Length: > 0 } address)
            {
                if (loaded.Truncated)
                    payload.Notes.Add(ShareStrings.T("share.note.addressShortened", ("characters", Chars(SharePayload.MaxUrlChars))));
                return ShareItem.ForUrl(address, title);
            }
            payload.Notes.Add(ShareStrings.T("share.note.linkUnreadable"));
            return null;
        }

        // ---- a file whose kind is known ------------------------------------------
        // Everything that is a file goes down one path, because the app classifies an
        // upload by its file name alone and a second opinion here could only disagree
        // with it. public.data is DELIBERATELY not in this list: every text type
        // conforms to it — public.plain-text → public.text → public.data — so including
        // it here turned a selection shared out of Messages, Mail or Notes into a
        // "file", which then arrived as a .txt attachment at best and was dropped
        // entirely on a build with no shared container. Generic data is the LAST resort,
        // below, after the text probe has had its turn.
        // RTFD is a directory package. NSItemProvider advertises com.apple.rtfd for one,
        // but Foundation's loadFileRepresentation can fail with EISDIR. Its file-URL
        // representation remains usable and must be coordinated without ForUploading,
        // which would turn the package into a ZIP before ReadPackage can see it.
        if (provider.HasItemConformingTo(Rtfd) && provider.HasItemConformingTo(FileUrl))
        {
            return await CopyFileUrlOrRefuseAsync(
                provider, writer, payload, soFar, totalLimit, Rtfd).ConfigureAwait(false);
        }

        string? fileType = FirstConforming(provider, Image, Movie, Audio, Pdf, Rtfd);
        if (fileType is not null)
            return await CopyOrRefuseAsync(provider, fileType, writer, payload, soFar, totalLimit).ConfigureAwait(false);

        // A public.file-url is a REFERENCE to the shared file. Asking
        // LoadFileRepresentation for that UTI produces a tiny temporary file whose
        // contents are the literal "file:///…" address, not the target bytes. Resolve
        // it with LoadItem and copy the target while the callback/security lease lives.
        if (provider.HasItemConformingTo(FileUrl))
        {
            string concrete = Concrete(provider, Data);
            if (!string.Equals(concrete, Data, StringComparison.Ordinal))
            {
                // Prefer the content representation. Foundation coordinates/provider-
                // downloads it and gives this callback actual bytes, not the file URL.
                return await CopyOrRefuseAsync(
                    provider, concrete, writer, payload, soFar, totalLimit).ConfigureAwait(false);
            }
            return await CopyFileUrlOrRefuseAsync(
                provider, writer, payload, soFar, totalLimit).ConfigureAwait(false);
        }

        // ---- text, in decreasing order of how much it says -----------------------
        // plain-text before text: asking for public.text from a provider that only
        // registered public.plain-text fails at run time with "a string could not be
        // instantiated because of an unknown error".
        foreach (string type in new[] { PlainText, Html, Rtf, Text })
        {
            if (!provider.HasItemConformingTo(type))
                continue;
            BoundedLoadedText loaded = await LoadTextBoundedAsync(provider, type, 120_000).ConfigureAwait(false);
            if (loaded.Refusal is { Length: > 0 } refusal)
            {
                payload.Notes.Add(refusal);
                return null;
            }
            string? text = loaded.Text;
            if (text is null || text.Trim().Length == 0)
                continue;
            if (loaded.Truncated)
            {
                payload.Notes.Add(
                    ShareStrings.T("share.note.textShortened", ("characters", Chars(loaded.OriginalChars))));
            }
            return ShareItem.ForText(ShareText.Shorten(text, 120_000).Text, title);
        }

        // ---- anything else, as bytes ---------------------------------------------
        // A zip, a vCard, a document type nothing above recognised. Last, so that it
        // cannot claim something the branches above would have read properly.
        if (provider.HasItemConformingTo(Data))
            return await CopyOrRefuseAsync(provider, Data, writer, payload, soFar, totalLimit).ConfigureAwait(false);

        return null;
    }

    /// <summary>Copy a file, or say why it could not be one.</summary>
    private static async Task<ShareItem?> CopyOrRefuseAsync(
        NSItemProvider provider, string family, ShareEnvelopeWriter? writer, SharePayload payload,
        long soFar, long totalLimit)
    {
        if (writer is null)
        {
            payload.Notes.Add(ShareStrings.T("share.note.noContainer"));
            return null;
        }
        if (soFar >= totalLimit)
        {
            payload.Notes.Add(ShareStrings.T("share.note.tooMuch"));
            return null;
        }
        // The CONCRETE type the provider registered, not the family it conforms to:
        // "public.image" has no file extension and no MIME type, and both are what the
        // app downstream classifies an upload by. A HEIC has to arrive as
        // public.heic, not as the abstract ancestor that matched.
        return await CopyFileAsync(
            provider, Concrete(provider, family), writer, payload, soFar, totalLimit).ConfigureAwait(false);
    }

    private static Task<ShareItem?> CopyFileUrlOrRefuseAsync(
        NSItemProvider provider, ShareEnvelopeWriter? writer, SharePayload payload,
        long soFar, long totalLimit, string? forcedTypeIdentifier = null)
    {
        if (writer is null)
        {
            payload.Notes.Add(ShareStrings.T("share.note.noContainer"));
            return Task.FromResult<ShareItem?>(null);
        }
        if (soFar >= totalLimit && !string.Equals(forcedTypeIdentifier, Rtfd, StringComparison.Ordinal))
        {
            payload.Notes.Add(ShareStrings.T("share.note.tooMuch"));
            return Task.FromResult<ShareItem?>(null);
        }

        var done = new TaskCompletionSource<ShareItem?>(TaskCreationOptions.RunContinuationsAsynchronously);
        string suggested = provider.SuggestedName ?? string.Empty;
        string concrete = Concrete(provider, Data);
        try
        {
            provider.LoadItem(FileUrl, null, (item, _) =>
            {
                NSUrl? url = UrlOf(item);
                bool scoped = false;
                try
                {
                    if (url is null || !url.IsFileUrl || url.Path is not { Length: > 0 } source)
                    {
                        payload.Notes.Add(Named(suggested) is { } name
                            ? ShareStrings.T("share.note.fileNotOpened", ("name", name))
                            : ShareStrings.T("share.note.unnamedFileNotOpened"));
                        done.TrySetResult(null);
                        return;
                    }
                    try { scoped = url.StartAccessingSecurityScopedResource(); }
                    catch (Exception) { }
                    bool isRtfd = string.Equals(forcedTypeIdentifier, Rtfd, StringComparison.Ordinal)
                        || string.Equals(Path.GetExtension(source.TrimEnd(Path.DirectorySeparatorChar)), ".rtfd", StringComparison.OrdinalIgnoreCase);
                    bool isDirectory = isRtfd || Directory.Exists(source);
                    string effectiveType = isRtfd
                        ? Rtfd
                        : isDirectory ? Zip : forcedTypeIdentifier ?? concrete;
                    string effectiveName = isDirectory && !isRtfd
                        ? ZipName(suggested, source)
                        : suggested;
                    NSFileCoordinatorReadingOptions options = isRtfd
                        ? NSFileCoordinatorReadingOptions.WithoutChanges
                        : NSFileCoordinatorReadingOptions.ForUploading;

                    ShareItem? copied = null;
                    using var coordinator = new NSFileCoordinator();
                    coordinator.CoordinateRead(
                        url,
                        options,
                        out NSError? coordinationError,
                        coordinatedUrl =>
                        {
                            if (coordinatedUrl.Path is { Length: > 0 } coordinatedSource)
                            {
                                copied = CopySource(
                                    coordinatedSource, effectiveName, effectiveType,
                                    writer, payload, soFar, totalLimit);
                            }
                        });
                    if (coordinationError is not null)
                    {
                        payload.Notes.Add(ShareStrings.T("share.note.fileNotCoordinated",
                            ("name", Describe(suggested, source)), ("reason", coordinationError.LocalizedDescription)));
                    }
                    done.TrySetResult(copied);
                }
                catch (Exception ex)
                {
                    payload.Notes.Add(NotShared(suggested, ex.Message));
                    done.TrySetResult(null);
                }
                finally
                {
                    if (scoped)
                    {
                        try { url?.StopAccessingSecurityScopedResource(); }
                        catch (Exception) { }
                    }
                }
            });
        }
        catch (Exception ex)
        {
            payload.Notes.Add(NotShared(suggested, ex.Message));
            done.TrySetResult(null);
        }
        return done.Task;
    }

    /// <summary>
    /// The most specific type identifier the provider actually registered under
    /// <paramref name="family"/>, or the family itself when it registered nothing more
    /// specific.
    /// </summary>
    private static string Concrete(NSItemProvider provider, string family)
    {
        try
        {
            foreach (string registered in provider.RegisteredTypeIdentifiers ?? Array.Empty<string>())
            {
                if (string.Equals(registered, family, StringComparison.Ordinal)
                    || string.Equals(registered, FileUrl, StringComparison.Ordinal)
                    || string.Equals(registered, Url, StringComparison.Ordinal))
                    continue;
                UniformTypeIdentifiers.UTType? type = UniformTypeIdentifiers.UTType.CreateFromIdentifier(registered);
                UniformTypeIdentifiers.UTType? parent = UniformTypeIdentifiers.UTType.CreateFromIdentifier(family);
                if (type is not null && parent is not null && type.ConformsTo(parent))
                    return registered;
            }
        }
        catch (Exception)
        {
        }
        return family;
    }

    /// <summary>
    /// The dictionary <c>ExtensionPreprocessing.js</c> built inside the page.
    /// </summary>
    private static async Task<ShareItem?> ReadWebPageAsync(NSItemProvider provider, SharePayload payload)
    {
        NSObject? loaded = await LoadItemAsync(provider, PropertyList).ConfigureAwait(false);
        if (loaded is not NSDictionary outer)
            return null;

        // The results are nested one level down, under a key whose name is the literal
        // below. An empty dictionary here means the JS file did not run -- a host that
        // is not Safari -- and the caller falls through to the plain URL.
        if (outer[JavaScriptResults] is not NSDictionary results)
            return null;

        string url = StringOf(results, "URL");
        string title = StringOf(results, "title");
        string selection = StringOf(results, "selection");
        string text = StringOf(results, "text");
        if (results["truncated"] is NSNumber truncated && truncated.BoolValue)
        {
            payload.Notes.Add(ShareStrings.T("share.note.pageShortened"));
            text = WithOwnMarker(text);
        }
        if (results["selectionTruncated"] is NSNumber selectionTruncated && selectionTruncated.BoolValue)
        {
            payload.Notes.Add(ShareStrings.T("share.note.selectionShortened"));
            selection = WithOwnMarker(selection);
        }
        if (results["metadataTruncated"] is NSNumber metadataTruncated && metadataTruncated.BoolValue)
            payload.Notes.Add(ShareStrings.T("share.note.pageMetadataShortened"));
        if (url.Length == 0 && text.Length == 0 && selection.Length == 0)
            return null;

        return ShareItem.ForPage(url, title, text, selection);
    }

    /// <summary>
    /// Copy one shared file into the envelope, without ever holding it in memory.
    ///
    /// <para>
    /// <c>LoadFileRepresentation</c> rather than <c>LoadItem</c>: the latter coerces the
    /// item to a type, which for an image commonly materialises a <c>UIImage</c> or an
    /// <c>NSData</c> — a 40 MB photo becoming 40 MB of resident memory inside a process
    /// with about 120 MB to spend, before anything has been written. And the CALLBACK
    /// overload rather than the <c>Async</c> one, because iOS deletes the temporary file
    /// the moment the completion handler returns; an <c>await</c> resumes after that.
    /// </para>
    /// </summary>
    private static Task<ShareItem?> CopyFileAsync(
        NSItemProvider provider, string typeIdentifier, ShareEnvelopeWriter writer, SharePayload payload,
        long soFar, long totalLimit)
    {
        var done = new TaskCompletionSource<ShareItem?>(TaskCreationOptions.RunContinuationsAsynchronously);
        string suggested = provider.SuggestedName ?? string.Empty;

        provider.LoadFileRepresentation(typeIdentifier, (url, error) =>
        {
            try
            {
                if (url is null || url.Path is not { Length: > 0 } source)
                {
                    // loadDataRepresentation is not a bounded fallback: Foundation has
                    // already allocated the entire NSData before its callback can inspect
                    // Length. One large UIImage is enough to jetsam a Share extension.
                    // loadFileRepresentation also spills value-backed data to a temporary
                    // file, so if it failed there is no safe second representation to ask
                    // for here.
                    payload.Notes.Add(Named(suggested) is { } name
                        ? ShareStrings.T("share.note.noFileRepresentation", ("name", name))
                        : ShareStrings.T("share.note.unnamedNoFileRepresentation"));
                    done.TrySetResult(null);
                    return;
                }

                done.TrySetResult(CopySource(
                    source, suggested, typeIdentifier, writer, payload, soFar, totalLimit));
            }
            catch (Exception ex)
            {
                payload.Notes.Add(NotShared(suggested, ex.Message));
                done.TrySetResult(null);
            }
        });

        return done.Task;
    }

    private static ShareItem? CopySource(
        string source, string suggested, string typeIdentifier, ShareEnvelopeWriter writer,
        SharePayload payload, long soFar, long totalLimit)
    {
        if (Directory.Exists(source))
            return ReadPackage(source, suggested, typeIdentifier, payload);

        long bytes;
        try { bytes = new FileInfo(source).Length; }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            payload.Notes.Add(ShareStrings.T("share.note.fileNotMeasured",
                ("name", Describe(suggested, source)), ("reason", ex.Message)));
            return null;
        }
        if (bytes > MaxFileBytes)
        {
            payload.Notes.Add(ShareStrings.T("share.note.fileTooLarge",
                ("name", Describe(suggested, source)),
                ("size", (bytes / (1024.0 * 1024)).ToString("0.#", ShareStrings.Culture))));
            return null;
        }
        if (bytes > totalLimit - soFar)
        {
            payload.Notes.Add(ShareStrings.T("share.note.tooMuch"));
            return null;
        }

        string name = suggested.Length > 0 ? suggested : Path.GetFileName(source);
        if (Path.GetExtension(name).Length == 0 && Path.GetExtension(source).Length > 0)
            name += Path.GetExtension(source);

        (string absolute, string relative) = writer.ReserveFile(name);
        NSFileManager.DefaultManager.Copy(source, absolute, out NSError? copyError);
        if (copyError is not null)
        {
            payload.Notes.Add(Named(name) is { } shown
                ? ShareStrings.T("share.note.fileNotCopied", ("name", shown), ("reason", copyError.LocalizedDescription))
                : ShareStrings.T("share.note.unnamedFileNotCopied", ("reason", copyError.LocalizedDescription)));
            return null;
        }
        return ShareItem.ForFile(relative, name, bytes, typeIdentifier, MimeTypeFor(typeIdentifier));
    }

    private static ShareItem? ReadPackage(
        string source, string suggested, string typeIdentifier, SharePayload payload)
    {
        if (!string.Equals(typeIdentifier, Rtfd, StringComparison.Ordinal)
            && !string.Equals(Path.GetExtension(source), ".rtfd", StringComparison.OrdinalIgnoreCase))
        {
            payload.Notes.Add(ShareStrings.T("share.note.folder", ("name", Describe(suggested, source))));
            return null;
        }

        RtfdScan scan;
        try
        {
            scan = ScanRtfd(source);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            payload.Notes.Add(ShareStrings.T("share.note.fileNotRead",
                ("name", Describe(suggested, source)), ("reason", ex.Message)));
            return null;
        }
        string? rtf = scan.RtfPath;
        if (scan.LimitReached)
            payload.Notes.Add(ShareStrings.T("share.note.packageLimited"));
        if (rtf is null)
        {
            payload.Notes.Add(ShareStrings.T("share.note.noRichText", ("name", Describe(suggested, source))));
            return null;
        }
        var info = new FileInfo(rtf);
        if (info.Length > MaxRichTextSourceBytes)
        {
            payload.Notes.Add(ShareStrings.T("share.note.richTextTooLarge", ("name", Describe(suggested, source))));
            return null;
        }
        string plain;
        var attributes = new NSAttributedStringDocumentAttributes
        {
            DocumentType = NSDocumentType.RTF,
        };
        using (NSUrl url = NSUrl.FromFilename(rtf))
        using (NSAttributedString? rich = NSAttributedString.Create(url, attributes, out NSError? _))
        {
            if (rich is null)
            {
                payload.Notes.Add(ShareStrings.T("share.note.richTextUndecodable", ("name", Describe(suggested, source))));
                return null;
            }
            plain = rich.Value ?? string.Empty;
        }
        if (plain.Trim().Length == 0)
            return null;
        if (scan.FileCount > 1)
            payload.Notes.Add(ShareStrings.T("share.note.embeddedFilesOmitted"));
        ShareText.ShortenedText bounded = ShareText.Shorten(plain, 120_000);
        if (bounded.WasShortened)
            payload.Notes.Add(ShareStrings.T("share.note.noteShortened", ("characters", Chars(bounded.OriginalLength))));
        return ShareItem.ForText(bounded.Text, suggested);
    }

    private readonly record struct RtfdScan(string? RtfPath, int FileCount, bool LimitReached);

    private static RtfdScan ScanRtfd(string root)
    {
        string? firstRtf = null;
        string conventional = Path.Combine(root, "TXT.rtf");
        string? preferredRtf = File.Exists(conventional) ? conventional : null;
        int entries = 0;
        int files = 0;
        bool limited = false;
        var pending = new Stack<(string Directory, int Depth)>();
        pending.Push((root, 0));

        while (pending.Count > 0 && !limited)
        {
            (string directory, int depth) = pending.Pop();
            foreach (string entry in Directory.EnumerateFileSystemEntries(directory))
            {
                if (++entries > MaxPackageEntries)
                {
                    limited = true;
                    break;
                }

                FileAttributes attributes = File.GetAttributes(entry);
                if ((attributes & FileAttributes.ReparsePoint) != 0)
                    continue;
                if ((attributes & FileAttributes.Directory) != 0)
                {
                    if (depth < 8)
                        pending.Push((entry, depth + 1));
                    else
                        limited = true;
                    continue;
                }

                files++;
                if (!string.Equals(Path.GetExtension(entry), ".rtf", StringComparison.OrdinalIgnoreCase))
                    continue;
                firstRtf ??= entry;
                if (string.Equals(Path.GetFileName(entry), "TXT.rtf", StringComparison.OrdinalIgnoreCase))
                    preferredRtf ??= entry;
            }
        }

        return new RtfdScan(preferredRtf ?? firstRtf, files, limited);
    }

    /// <summary>
    /// Drop a bare link that names a page already read in full.
    ///
    /// <para>
    /// Safari's whole-page share carries both: the property-list item this extension's
    /// JavaScript produced, and an ordinary <c>public.url</c> beside it. The page is
    /// strictly more than the link, so the link is the one that goes.
    /// </para>
    /// </summary>
    private static void FoldDuplicateLinks(SharePayload payload)
    {
        var pages = payload.Items
            .Where(i => i.Kind == ShareItemKinds.Page && i.Url.Trim().Length > 0)
            .Select(i => i.Url.Trim())
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        if (pages.Count == 0)
            return;

        payload.Items.RemoveAll(i => i.Kind == ShareItemKinds.Url && pages.Contains(i.Url.Trim()));

        // A page that came with no title of its own can still take the one the host put
        // on the item, which is what a browser sets attributedTitle for.
        foreach (ShareItem page in payload.Items.Where(i => i.Kind == ShareItemKinds.Page))
        {
            if (page.Title.Trim().Length > 0)
                continue;
            ShareItem? titled = payload.Items.FirstOrDefault(
                i => i.Kind == ShareItemKinds.Text && i.Title.Trim().Length > 0);
            if (titled is not null)
                page.Title = titled.Title;
        }
    }

    /// <summary>
    /// Fold a text attachment that is really a link's title into the link.
    ///
    /// <para>
    /// Firefox sends the page title as a separate <c>public.plain-text</c> attachment
    /// beside the URL, and several apps add a short text preview to a link share. Left
    /// alone, the model is handed "Shared link: …" followed by "Shared text: …" holding
    /// the same sentence, and asked to summarise both.
    /// </para>
    /// </summary>
    private static void FoldProvenDuplicateLinkText(SharePayload payload)
    {
        ShareItem? link = payload.Items.FirstOrDefault(i => i.Kind == ShareItemKinds.Url);
        if (link is null)
            return;

        for (int i = payload.Items.Count - 1; i >= 0; i--)
        {
            ShareItem item = payload.Items[i];
            if (item.Kind != ShareItemKinds.Text)
                continue;
            string text = item.Text.Trim();
            if (text.Length == 0)
                continue;
            if (text == link.Url.Trim())
            {
                payload.Items.RemoveAt(i);
                continue;
            }
            // Only remove a title already present on the link. Inferring that any
            // short text beside an untitled URL is a title drops real captions and
            // selected quotes from Mail, Messages and Reddit.
            if (link.Title.Trim().Length > 0 && link.Title.Trim() == text)
                payload.Items.RemoveAt(i);
        }
    }

    // ---- the small awkward parts of NSItemProvider -------------------------------

    private static string? FirstConforming(NSItemProvider provider, params string[] types)
    {
        foreach (string type in types)
        {
            if (provider.HasItemConformingTo(type))
                return type;
        }
        return null;
    }

    private static bool LooksLikeSafariPreprocessingProvider(NSItemProvider provider)
    {
        if (!provider.HasItemConformingTo(PropertyList)
            || provider.HasItemConformingTo(FileUrl)
            || !string.IsNullOrEmpty(provider.SuggestedName))
        {
            return false;
        }

        // Safari's preprocessing attachment is an anonymous property-list value.
        // A document named foo.plist is content and must stay on the file-backed path;
        // LoadItem would otherwise materialise the entire document as NSData before any
        // extension memory/file-size guard could run.
        string[] registered = provider.RegisteredTypeIdentifiers ?? Array.Empty<string>();
        return registered.Any(type => string.Equals(type, PropertyList, StringComparison.Ordinal));
    }

    private static Task<NSObject?> LoadItemAsync(NSItemProvider provider, string typeIdentifier)
    {
        var done = new TaskCompletionSource<NSObject?>(TaskCreationOptions.RunContinuationsAsynchronously);
        try
        {
            provider.LoadItem(typeIdentifier, null, (item, _) => done.TrySetResult(item));
        }
        catch (Exception)
        {
            done.TrySetResult(null);
        }
        return done.Task;
    }

    private static NSUrl? UrlOf(NSObject? loaded)
    {
        return loaded switch
        {
            NSUrl url => url,
            NSString text when (ulong)text.Length <= MaxFileUrlRepresentationBytes =>
                NSUrl.FromString(text.ToString()),
            NSData data when data.Length <= MaxFileUrlRepresentationBytes =>
                NSUrl.FromString(data.ToString(NSStringEncoding.UTF8) ?? string.Empty),
            _ => null,
        };
    }

    private readonly record struct BoundedLoadedText(
        string? Text, bool Truncated, long OriginalChars, string? Refusal = null);

    private static BoundedLoadedText ReadAttributedText(NSAttributedString? attributed, int maxChars)
    {
        if (attributed is null || attributed.Length <= 0 || maxChars <= 0)
            return default;
        long original = (long)attributed.Length;
        if (original <= maxChars)
            return new BoundedLoadedText(AttributedSubstring(attributed, 0, attributed.Length), false, original);

        string marker = MiddleShortenedMarker();
        int contentBudget = Math.Max(0, maxChars - marker.Length);
        int headLimit = contentBudget * 2 / 3;
        int tailLimit = contentBudget - headLimit;
        string head = headLimit == 0 ? string.Empty : AttributedSubstring(attributed, 0, headLimit);
        if (head.Length > 0 && char.IsHighSurrogate(head[^1]))
            head = head[..^1];
        string tail = tailLimit == 0
            ? string.Empty
            : AttributedSubstring(attributed, (nint)(original - tailLimit), tailLimit);
        if (tail.Length > 0 && char.IsLowSurrogate(tail[0]))
            tail = tail[1..];
        return new BoundedLoadedText(head + marker + tail, true, original);
    }

    private static string AttributedSubstring(NSAttributedString attributed, nint start, nint length)
    {
        using NSAttributedString part = attributed.Substring(start, length);
        return part.Value ?? string.Empty;
    }

    private static Task<BoundedLoadedText> LoadUrlBoundedAsync(NSItemProvider provider)
    {
        var done = new TaskCompletionSource<BoundedLoadedText>(TaskCreationOptions.RunContinuationsAsynchronously);
        try
        {
            // Asking for a file representation makes Foundation spill even an
            // NSString/NSData-backed provider to disk; LoadItem would materialise an
            // attacker-sized NSData before a length check. Do not assume that file is
            // raw UTF-8, though. UIActivityViewController on physical iOS 26 serializes
            // an NSURL activity item as a binary property-list Foundation shape. Reading those
            // bytes with StreamReader visibly prepends "bplist00" and appends its
            // object-table/trailer bytes to the address in TensorAgent's composer.
            provider.LoadFileRepresentation(Url, (url, _) =>
            {
                try
                {
                    if (url?.Path is not { Length: > 0 } path || !File.Exists(path))
                    {
                        done.TrySetResult(default);
                        return;
                    }

                    long sourceBytes = new FileInfo(path).Length;
                    if (FileStartsWithBinaryPropertyList(path))
                    {
                        if (sourceBytes > MaxUrlRepresentationBytes)
                        {
                            done.TrySetResult(new BoundedLoadedText(
                                null, false, 0,
                                ShareStrings.T("share.note.linkTooLarge")));
                            return;
                        }

                        string? decoded = DecodeBinaryPropertyListString(path, allowUrl: true);
                        if (decoded is not null)
                        {
                            done.TrySetResult(BoundUrlString(decoded));
                            return;
                        }

                        // A provider claiming public.url handed us a plist of a
                        // different or invalid shape. Treat that as unreadable rather
                        // than showing its serialization or recursively guessing at it.
                        done.TrySetResult(default);
                        return;
                    }

                    int limit = SharePayload.MaxUrlChars;
                    var chars = new char[limit + 1];
                    int count = 0;
                    using var reader = new StreamReader(path, detectEncodingFromByteOrderMarks: true);
                    while (count < chars.Length)
                    {
                        int read = reader.Read(chars, count, chars.Length - count);
                        if (read == 0)
                            break;
                        count += read;
                    }
                    bool truncated = count > limit;
                    int kept = Math.Min(count, limit);
                    if (kept > 0 && kept < count && char.IsHighSurrogate(chars[kept - 1]))
                        kept--;
                    done.TrySetResult(new BoundedLoadedText(
                        new string(chars, 0, kept), truncated, count));
                }
                catch (Exception)
                {
                    done.TrySetResult(default);
                }
            });
        }
        catch (Exception)
        {
            done.TrySetResult(default);
        }
        return done.Task;
    }

    /// <summary>
    /// Decode the bounded binary-plist shapes NSItemProvider uses for value-backed
    /// objects: a direct NSString scalar, Foundation's exact three-part NSURL array,
    /// or an NSKeyedArchiver whose root is NSString (text) or NSURL (links). The secure unarchiver's allow-list is deliberately tiny;
    /// arbitrary classes from a sharing app are never instantiated.
    /// </summary>
    private static string? DecodeBinaryPropertyListString(string path, bool allowUrl)
    {
        using NSData? data = NSData.FromFile(path);
        if (data is null)
            return null;

        bool isKeyedArchive;
        using (NSObject? propertyList = ReadPropertyList(data))
        {
            if (propertyList is NSString directString)
                return directString.ToString();

            // Across the share-extension XPC boundary iOS 26 can encode an NSURL as
            // a three-part property-list array: relative string, base string (empty
            // for an absolute URL), and an options dictionary. This is not a keyed
            // archive, so accepting only a top-level NSString loses the link. Match
            // the complete Foundation shape and let NSUrl resolve a non-empty base;
            // never search arbitrary arrays/dictionaries for something URL-like.
            if (allowUrl
                && propertyList is NSArray urlParts
                && urlParts.Count == 3
                && urlParts.GetItem<NSObject>(0) is NSString relativeString
                && urlParts.GetItem<NSObject>(1) is NSString baseString
                && urlParts.GetItem<NSObject>(2) is NSDictionary)
            {
                string relative = relativeString.ToString();
                string baseAddress = baseString.ToString();
                if (relative.Length == 0)
                    return null;
                if (baseAddress.Length == 0)
                {
                    using NSUrl? absolute = NSUrl.FromString(relative);
                    return absolute?.AbsoluteString;
                }

                using NSUrl? baseUrl = NSUrl.FromString(baseAddress);
                if (baseUrl is null)
                    return null;
                using var resolved = new NSUrl(relative, baseUrl);
                return resolved.AbsoluteString;
            }

            // Only hand a recognized keyed archive to NSKeyedUnarchiver. Besides
            // making the accepted wire shapes explicit, this prevents an arbitrary
            // binary plist dictionary from becoming an object-decoding request.
            isKeyedArchive = propertyList is NSDictionary archive
                && archive["$archiver"] is NSString archiver
                && string.Equals(archiver.ToString(), "NSKeyedArchiver", StringComparison.Ordinal);
        }

        // Release the parsed plist graph before the unarchiver constructs a second
        // graph from the same (up to 4 MiB) input. Share extensions have a tight jetsam
        // budget, so retaining both representations at once is needless peak RSS.
        if (!isKeyedArchive)
            return null;

        Type[] allowed = allowUrl
            ? new[] { typeof(NSString), typeof(NSUrl) }
            : new[] { typeof(NSString) };
        using NSObject? decoded = NSKeyedUnarchiver.GetUnarchivedObject(
            allowed, data, out NSError? _);
        return decoded switch
        {
            NSString stringValue => stringValue.ToString(),
            NSUrl urlValue when allowUrl => urlValue.AbsoluteString,
            _ => null,
        };
    }

    private static NSObject? ReadPropertyList(NSData data)
    {
        NSPropertyListFormat format = default;
        return NSPropertyListSerialization.PropertyListWithData(
            data,
            NSPropertyListReadOptions.Immutable,
            ref format,
            out NSError? _);
    }

    private static BoundedLoadedText BoundUrlString(string? value)
    {
        string text = value ?? string.Empty;
        int limit = SharePayload.MaxUrlChars;
        if (text.Length <= limit)
            return new BoundedLoadedText(text, false, text.Length);
        int kept = limit;
        if (kept > 0 && char.IsHighSurrogate(text[kept - 1]))
            kept--;
        return new BoundedLoadedText(text[..kept], true, text.Length);
    }

    private static bool FileStartsWithBinaryPropertyList(string path)
    {
        ReadOnlySpan<byte> expected = "bplist00"u8;
        Span<byte> start = stackalloc byte[8];
        using var stream = File.OpenRead(path);
        int count = 0;
        while (count < start.Length)
        {
            int read = stream.Read(start[count..]);
            if (read == 0)
                break;
            count += read;
        }
        return count == start.Length && start.SequenceEqual(expected);
    }

    private static Task<BoundedLoadedText> LoadTextBoundedAsync(
        NSItemProvider provider, string typeIdentifier, int maxChars)
    {
        var done = new TaskCompletionSource<BoundedLoadedText>(TaskCreationOptions.RunContinuationsAsynchronously);
        try
        {
            // Request a file representation even for a value-backed NSString/NSData.
            // Foundation spills it to a temporary file, letting us stop after a bounded
            // number of characters instead of materialising hostile-sized text in the
            // extension. Read synchronously before the callback returns: iOS removes
            // the representation immediately afterwards.
            provider.LoadFileRepresentation(typeIdentifier, (url, _) =>
            {
                try
                {
                    if (url?.Path is not { Length: > 0 } path || !File.Exists(path))
                    {
                        done.TrySetResult(default);
                        return;
                    }
                    long sourceBytes = new FileInfo(path).Length;

                    // On physical iOS, any value-backed NSString activity item can
                    // arrive as an NSKeyedArchiver binary plist, including HTML/RTF.
                    // Decode before every raw/rich-text path, or archive keys and
                    // trailer bytes become visible text (or valid rich text is refused).
                    // AttributedContentText may contain the same clean value; ReadAsync's
                    // equality check folds that copy.
                    if (FileStartsWithBinaryPropertyList(path))
                    {
                        if (sourceBytes > MaxArchivedTextRepresentationBytes)
                        {
                            done.TrySetResult(new BoundedLoadedText(
                                null, false, 0,
                                ShareStrings.T("share.note.textTooLarge")));
                            return;
                        }

                        string? decoded = DecodeBinaryPropertyListString(path, allowUrl: false);
                        if (decoded is null)
                        {
                            done.TrySetResult(new BoundedLoadedText(
                                null, false, 0,
                                ShareStrings.T("share.note.textUndecodable")));
                            return;
                        }

                        if (string.Equals(typeIdentifier, Rtf, StringComparison.Ordinal))
                        {
                            using NSData encodedRtf = NSData.FromString(decoded, NSStringEncoding.UTF8);
                            var attributes = new NSAttributedStringDocumentAttributes
                            {
                                DocumentType = NSDocumentType.RTF,
                            };
                            using NSAttributedString? rich = NSAttributedString.Create(
                                encodedRtf, attributes, out NSError? _);
                            if (rich is null)
                            {
                                done.TrySetResult(new BoundedLoadedText(
                                    null, false, 0,
                                    ShareStrings.T("share.note.richTextItemUndecodable")));
                                return;
                            }
                            done.TrySetResult(ReadAttributedText(rich, maxChars));
                            return;
                        }

                        string plain = string.Equals(typeIdentifier, Html, StringComparison.Ordinal)
                            ? ShareText.FromHtml(decoded)
                            : decoded;
                        ShareText.ShortenedText decodedBounded = ShareText.Shorten(plain, maxChars);
                        done.TrySetResult(new BoundedLoadedText(
                            decodedBounded.Text,
                            decodedBounded.WasShortened,
                            decodedBounded.OriginalLength));
                        return;
                    }

                    bool isRtf = string.Equals(typeIdentifier, Rtf, StringComparison.Ordinal)
                        || FileStartsWithRtf(path);
                    if (isRtf)
                    {
                        if (sourceBytes > MaxRichTextSourceBytes)
                        {
                            done.TrySetResult(new BoundedLoadedText(
                                null, false, 0,
                                ShareStrings.T("share.note.richTextItemTooLarge")));
                            return;
                        }
                        var attributes = new NSAttributedStringDocumentAttributes
                        {
                            DocumentType = NSDocumentType.RTF,
                        };
                        using NSAttributedString? rich = NSAttributedString.Create(
                            url, attributes, out NSError? _);
                        if (rich is not null)
                        {
                            done.TrySetResult(ReadAttributedText(rich, maxChars));
                            return;
                        }
                        done.TrySetResult(new BoundedLoadedText(
                            null, false, 0,
                            ShareStrings.T("share.note.richTextItemUndecodable")));
                        return;
                    }
                    if (string.Equals(typeIdentifier, Html, StringComparison.Ordinal))
                    {
                        if (sourceBytes > MaxRichTextSourceBytes)
                        {
                            done.TrySetResult(new BoundedLoadedText(
                                null, false, 0,
                                ShareStrings.T("share.note.htmlTooLarge")));
                            return;
                        }
                        using var htmlReader = new StreamReader(path, detectEncodingFromByteOrderMarks: true);
                        string plain = ShareText.FromHtml(htmlReader.ReadToEnd());
                        ShareText.ShortenedText bounded = ShareText.Shorten(plain, maxChars);
                        done.TrySetResult(new BoundedLoadedText(
                            bounded.Text, bounded.WasShortened, bounded.OriginalLength));
                        return;
                    }
                    string marker = MiddleShortenedMarker();
                    int contentBudget = Math.Max(0, maxChars - marker.Length);
                    int headLimit = contentBudget * 2 / 3;
                    int tailLimit = contentBudget - headLimit;
                    var prefix = new System.Text.StringBuilder(maxChars + 1);
                    var tail = new char[tailLimit];
                    int tailCount = 0, tailPosition = 0;
                    long totalChars = 0;
                    using var reader = new StreamReader(path, detectEncodingFromByteOrderMarks: true);
                    var buffer = new char[8 * 1024];
                    int read;
                    while ((read = reader.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        totalChars += read;
                        int prefixRoom = maxChars + 1 - prefix.Length;
                        if (prefixRoom > 0)
                            prefix.Append(buffer, 0, Math.Min(read, prefixRoom));
                        if (tailLimit > 0)
                        {
                            for (int i = 0; i < read; i++)
                            {
                                tail[tailPosition] = buffer[i];
                                tailPosition = (tailPosition + 1) % tailLimit;
                                if (tailCount < tailLimit)
                                    tailCount++;
                            }
                        }
                    }
                    if (totalChars <= maxChars)
                    {
                        done.TrySetResult(new BoundedLoadedText(prefix.ToString(), false, totalChars));
                        return;
                    }

                    string head = prefix.ToString(0, Math.Min(headLimit, prefix.Length));
                    if (head.Length > 0 && char.IsHighSurrogate(head[^1]))
                        head = head[..^1];
                    var tailText = new System.Text.StringBuilder(tailCount);
                    int start = tailCount == tailLimit ? tailPosition : 0;
                    for (int i = 0; i < tailCount; i++)
                        tailText.Append(tail[(start + i) % tailLimit]);
                    if (tailText.Length > 0 && char.IsLowSurrogate(tailText[0]))
                        tailText.Remove(0, 1);
                    done.TrySetResult(new BoundedLoadedText(
                        head + marker + tailText, true, totalChars));
                }
                catch (Exception)
                {
                    done.TrySetResult(default);
                }
            });
        }
        catch (Exception)
        {
            done.TrySetResult(default);
        }
        return done.Task;
    }

    private static bool FileStartsWithRtf(string path)
    {
        Span<char> start = stackalloc char[5];
        using var reader = new StreamReader(path, detectEncodingFromByteOrderMarks: true);
        int count = 0;
        while (count < start.Length)
        {
            int read = reader.Read(start[count..]);
            if (read == 0)
                break;
            count += read;
        }
        return count == start.Length && start.SequenceEqual("{\\rtf");
    }

    private static string StringOf(NSDictionary dictionary, string key)
        => dictionary[key] is NSString value ? value.ToString() : string.Empty;

    /// <summary>The shared file's name as the user knows it: the one the sharing app gave, else the file's own.</summary>
    private static string Describe(string name, string path) =>
        ShareEnvelopeWriter.SafeFileName(name.Length > 0 ? name : Path.GetFileName(path));

    /// <summary>
    /// The name the sharing app gave the file, or null when it gave none: each note about a
    /// file is a whole sentence of its own for an unnamed one.
    /// </summary>
    private static string? Named(string name) =>
        name.Length > 0 ? ShareEnvelopeWriter.SafeFileName(name) : null;

    private static string NotShared(string name, string reason) => Named(name) is { } shown
        ? ShareStrings.T("share.note.fileNotShared", ("name", shown), ("reason", reason))
        : ShareStrings.T("share.note.unnamedFileNotShared", ("reason", reason));

    /// <summary>A number of characters, written the way the interface writes numbers.</summary>
    private static string Chars(long count) => count.ToString("N0", ShareStrings.Culture);

    /// <summary>What stands where the middle of a very long shared text was cut out.</summary>
    private static string MiddleShortenedMarker() =>
        "\n\n" + ShareStrings.T("share.text.middleShortened") + "\n\n";

    /// <summary>
    /// A page or selection the script cut, marked as the extension marks its own cuts. The
    /// app's shorter cut usually drops the script's marker with the middle around it, but
    /// not when a long highlighted passage taken out of the page moves it into the part kept.
    /// </summary>
    private static string WithOwnMarker(string text)
    {
        int at = text.IndexOf(ScriptMiddleShortenedMarker, StringComparison.Ordinal);
        return at < 0
            ? text
            : text[..at] + MiddleShortenedMarker() + text[(at + ScriptMiddleShortenedMarker.Length)..];
    }

    private static string ZipName(string suggested, string source)
    {
        string name = suggested;
        if (name.Length == 0)
            name = Path.GetFileName(source.TrimEnd(Path.DirectorySeparatorChar));
        if (!name.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
            name += ".zip";
        return name;
    }

    /// <summary>
    /// The MIME type for a uniform type identifier, for the app's upload naming.
    ///
    /// <para>
    /// Only the families matter, because the app uses this ONLY when the file arrived
    /// with no usable extension, and then only to pick one. Anything not listed falls
    /// through to the app's own content sniffing, which reads the first bytes.
    /// </para>
    /// </summary>
    private static string MimeTypeFor(string typeIdentifier)
    {
        try
        {
            UniformTypeIdentifiers.UTType? type = UniformTypeIdentifiers.UTType.CreateFromIdentifier(typeIdentifier);
            return type?.PreferredMimeType ?? string.Empty;
        }
        catch (Exception)
        {
            return string.Empty;
        }
    }
}
