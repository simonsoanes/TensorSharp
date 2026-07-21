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
            if (string.IsNullOrEmpty(pdfPath))
                throw new ArgumentNullException(nameof(pdfPath));
            if (!File.Exists(pdfPath))
                throw new FileNotFoundException("PDF file not found.", pdfPath);

            byte[] bytes = File.ReadAllBytes(pdfPath);
            return ExtractFromBytes(bytes, maxPages, password);
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
            if (pdfBytes == null || pdfBytes.Length == 0)
                throw new ArgumentException("Empty PDF data.", nameof(pdfBytes));

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
                document = PdfDocument.Open(pdfBytes, options);
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

                var sb = new StringBuilder();
                int extracted = 0;
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

                    if (sb.Length > 0)
                        sb.Append("\n\n");
                    sb.Append(pageText.TrimEnd());
                }

                string text = sb.ToString();
                int nonWhitespace = 0;
                for (int c = 0; c < text.Length; c++)
                {
                    if (!char.IsWhiteSpace(text[c]))
                        nonWhitespace++;
                }

                return new PdfTextResult
                {
                    Text = text,
                    PageCount = total,
                    ExtractedPageCount = extracted,
                    NonWhitespaceCharCount = nonWhitespace,
                };
            }
        }
    }
}
