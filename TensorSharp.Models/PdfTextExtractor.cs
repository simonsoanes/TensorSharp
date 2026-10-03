// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.IO;
using System.Text;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;

namespace TensorSharp.Models
{
    /// <summary>
    /// Text extracted from a PDF document, ready to be inlined into an LLM prompt.
    /// </summary>
    public sealed class PdfTextResult
    {
        /// <summary>The concatenated text of the extracted pages (reading order, one blank line between pages).</summary>
        public string Text { get; init; }

        /// <summary>Total number of pages in the source document.</summary>
        public int PageCount { get; init; }

        /// <summary>Number of pages actually read (equals <see cref="PageCount"/> unless a page cap was applied or a page failed to parse).</summary>
        public int ExtractedPageCount { get; init; }

        /// <summary>Count of non-whitespace characters in <see cref="Text"/> (used to detect image-only PDFs).</summary>
        public int NonWhitespaceCharCount { get; init; }

        /// <summary>
        /// Whether extraction stopped because its caller's character budget was reached.
        /// This is distinct from <see cref="ExtractedPageCount"/>: the latter also reflects
        /// a page limit or an unreadable page.
        /// </summary>
        public bool TextTruncated { get; init; }

        /// <summary>
        /// True when the document has (almost) no selectable text — the tell-tale of a
        /// scanned or image-only PDF (each page is a picture with no embedded text layer).
        /// Such a PDF cannot be handed to a text model as-is; its pages must instead be
        /// read as images by a vision model (see <see cref="PdfPageImageExtractor"/>).
        /// The threshold scales with page count so a genuine document with a couple of
        /// sparse pages is not misclassified, while a stray page number or watermark on a
        /// scan still reads as textless.
        /// </summary>
        public bool LooksTextless => NonWhitespaceCharCount < Math.Max(16, PageCount * 4);
    }

    /// <summary>
    /// Extracts plain text from PDF documents so they can be uploaded and sent to a
    /// (text-only) LLM for inference. Extraction is a one-time preprocessing step: the
    /// resulting text flows through the model's normal (already optimized) prefill path,
    /// so the LLM — not this extractor — dominates end-to-end latency. We nonetheless keep
    /// extraction cheap: a single streaming pass over the pages with a bounded
    /// <see cref="StringBuilder"/> and an optional page cap for very large documents.
    ///
    /// Backed by PdfPig (pure-managed, cross-platform, no native dependency). Text is
    /// pulled in content/reading order via <c>ContentOrderTextExtractor</c>, which
    /// reconstructs word and line spacing far better than a raw glyph dump.
    /// </summary>
    public static class PdfTextExtractor
    {
        /// <summary>Returns true when <paramref name="path"/> has a <c>.pdf</c> extension (case-insensitive).</summary>
        public static bool IsPdfFile(string path)
        {
            if (string.IsNullOrEmpty(path))
                return false;
            return string.Equals(Path.GetExtension(path), ".pdf", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Extracts text from a PDF on disk. See <see cref="ExtractFromBytes"/> for behavior.
        /// </summary>
        /// <param name="pdfPath">Path to the PDF file.</param>
        /// <param name="maxPages">Optional cap on the number of pages read (<c>&lt;= 0</c> = all pages).</param>
        /// <param name="password">Optional password for an encrypted PDF.</param>
        public static PdfTextResult ExtractFromFile(string pdfPath, int maxPages = 0, string password = null)
        {
            return ExtractFromFile(pdfPath, maxPages, password, maxTextCharacters: 0);
        }

        /// <summary>
        /// Extracts text from a PDF on disk without first copying the complete file into
        /// a managed byte array. A positive <paramref name="maxTextCharacters"/> bounds
        /// the aggregate string built while pages are visited; zero keeps the
        /// unlimited-text contract.
        /// </summary>
        public static PdfTextResult ExtractFromFile(
            string pdfPath, int maxPages, string password, int maxTextCharacters)
        {
            if (string.IsNullOrEmpty(pdfPath))
                throw new ArgumentNullException(nameof(pdfPath));
            if (!File.Exists(pdfPath))
                throw new FileNotFoundException("PDF file not found.", pdfPath);
            if (maxTextCharacters < 0)
                throw new ArgumentOutOfRangeException(nameof(maxTextCharacters));

            // PdfPig accepts a seekable stream and does not take ownership of it. Keeping
            // that stream alive for the document lifetime avoids the previous 128 MiB file
            // -> 128 MiB byte[] duplication on a memory-constrained phone.
            using var stream = new FileStream(
                pdfPath, FileMode.Open, FileAccess.Read, FileShare.Read,
                bufferSize: 64 * 1024, options: FileOptions.SequentialScan);
            return ExtractFromStream(stream, maxPages, password, maxTextCharacters);
        }

        /// <summary>
        /// Extracts text from PDF bytes (e.g. an uploaded file held in memory).
        /// </summary>
        /// <param name="pdfBytes">The raw PDF file contents.</param>
        /// <param name="maxPages">Optional cap on the number of pages read (<c>&lt;= 0</c> = all pages).</param>
        /// <param name="password">Optional password for an encrypted PDF.</param>
        /// <exception cref="InvalidDataException">
        /// The bytes are not a usable PDF (corrupt, or encrypted with a password we were not given).
        /// </exception>
        public static PdfTextResult ExtractFromBytes(byte[] pdfBytes, int maxPages = 0, string password = null)
        {
            return ExtractFromBytes(pdfBytes, maxPages, password, maxTextCharacters: 0);
        }

        /// <summary>
        /// Byte-array extraction with an optional aggregate text bound. The input bytes
        /// already belong to the caller; unlike <see cref="ExtractFromFile(string,int,string,int)"/>,
        /// this overload does not make another copy of them.
        /// </summary>
        public static PdfTextResult ExtractFromBytes(
            byte[] pdfBytes, int maxPages, string password, int maxTextCharacters)
        {
            if (pdfBytes == null || pdfBytes.Length == 0)
                throw new ArgumentException("Empty PDF data.", nameof(pdfBytes));
            if (maxTextCharacters < 0)
                throw new ArgumentOutOfRangeException(nameof(maxTextCharacters));

            using var stream = new MemoryStream(pdfBytes, writable: false);
            return ExtractFromStream(stream, maxPages, password, maxTextCharacters);
        }

        private static PdfTextResult ExtractFromStream(
            Stream pdfStream, int maxPages, string password, int maxTextCharacters)
        {
            if (pdfStream == null)
                throw new ArgumentNullException(nameof(pdfStream));

            // Lenient parsing lets PdfPig recover from the many real-world PDFs with a
            // broken xref table or minor spec violations. SkipMissingFonts keeps a page
            // whose font can't be loaded readable (glyphs fall back) instead of aborting.
            var options = new ParsingOptions
            {
                UseLenientParsing = true,
                SkipMissingFonts = true,
            };
            if (!string.IsNullOrEmpty(password))
                options.Password = password;

            PdfDocument document;
            try
            {
                document = PdfDocument.Open(pdfStream, options);
            }
            catch (Exception ex)
            {
                // Encrypted-without-password, truncated, or otherwise unreadable file.
                throw new InvalidDataException(
                    "Could not open the PDF (it may be corrupt or password-protected): " + ex.Message, ex);
            }

            using (document)
            {
                int total = document.NumberOfPages;
                int limit = maxPages > 0 ? Math.Min(maxPages, total) : total;

                var sb = maxTextCharacters > 0
                    ? new StringBuilder(Math.Min(maxTextCharacters, 16 * 1024))
                    : new StringBuilder();
                int extracted = 0;
                int nonWhitespace = 0;
                bool textTruncated = false;
                for (int i = 1; i <= limit; i++)
                {
                    string pageText;
                    try
                    {
                        Page page = document.GetPage(i);
                        // ContentOrderTextExtractor reconstructs reading order + spacing;
                        // fall back to the raw glyph stream if it trips on an odd page.
                        try { pageText = ContentOrderTextExtractor.GetText(page, true); }
                        catch { pageText = page.Text; }
                    }
                    catch
                    {
                        // A single unparseable page shouldn't sink the whole document.
                        continue;
                    }

                    extracted++;
                    if (string.IsNullOrWhiteSpace(pageText))
                        continue;

                    // Avoid pageText.TrimEnd(): on a pathological single page it creates
                    // another full-size string before the aggregate bound can help.
                    int pageLength = pageText.Length;
                    while (pageLength > 0 && char.IsWhiteSpace(pageText[pageLength - 1]))
                        pageLength--;
                    if (pageLength == 0)
                        continue;

                    for (int c = 0; c < pageLength; c++)
                    {
                        if (!char.IsWhiteSpace(pageText[c]))
                            nonWhitespace++;
                    }

                    if (maxTextCharacters <= 0)
                    {
                        if (sb.Length > 0)
                            sb.Append("\n\n");
                        sb.Append(pageText, 0, pageLength);
                        continue;
                    }

                    int separatorLength = sb.Length > 0 ? 2 : 0;
                    int remaining = maxTextCharacters - sb.Length - separatorLength;
                    if (remaining <= 0)
                    {
                        textTruncated = true;
                        break;
                    }

                    if (separatorLength > 0)
                        sb.Append("\n\n");
                    int take = Math.Min(pageLength, remaining);
                    // Never leave an isolated high surrogate at the end of the bounded
                    // result. The next page/file layer can safely add its truncation note.
                    if (take > 0 && take < pageLength && char.IsHighSurrogate(pageText[take - 1]))
                        take--;
                    if (take > 0)
                        sb.Append(pageText, 0, take);
                    if (take < pageLength)
                    {
                        textTruncated = true;
                        break;
                    }
                }

                string text = sb.ToString();

                return new PdfTextResult
                {
                    Text = text,
                    PageCount = total,
                    ExtractedPageCount = extracted,
                    NonWhitespaceCharCount = nonWhitespace,
                    TextTruncated = textTruncated,
                };
            }
        }
    }
}
