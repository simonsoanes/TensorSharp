using System;
using System.IO;
using TensorSharp.Models;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.Fonts.Standard14Fonts;
using UglyToad.PdfPig.Writer;

namespace InferenceWeb.Tests;

public class PdfTextExtractorTests
{
    // Builds a small, well-formed PDF in memory with one text line per page using a
    // Standard-14 font (Helvetica), whose metrics PdfPig knows without embedding — so
    // the round-trip (build -> extract) exercises the real parse + text-extraction path.
    private static byte[] BuildPdf(params string[] pageTexts)
    {
        var builder = new PdfDocumentBuilder();
        PdfDocumentBuilder.AddedFont font = builder.AddStandard14Font(Standard14Font.Helvetica);
        foreach (string text in pageTexts)
        {
            PdfPageBuilder page = builder.AddPage(PageSize.A4);
            page.AddText(text, 12m, new PdfPoint(25, 800), font);
        }
        return builder.Build();
    }

    [Theory]
    [InlineData("document.pdf", true)]
    [InlineData("DOCUMENT.PDF", true)]
    [InlineData("report.final.Pdf", true)]
    [InlineData("notes.txt", false)]
    [InlineData("image.png", false)]
    [InlineData("", false)]
    [InlineData(null, false)]
    public void IsPdfFile_DetectsExtensionCaseInsensitively(string path, bool expected)
    {
        Assert.Equal(expected, PdfTextExtractor.IsPdfFile(path));
    }

    [Fact]
    public void ExtractFromBytes_ReturnsTextAndPageCount()
    {
        byte[] pdf = BuildPdf("HelloTensorSharpPdf");

        PdfTextResult result = PdfTextExtractor.ExtractFromBytes(pdf);

        Assert.Equal(1, result.PageCount);
        Assert.Equal(1, result.ExtractedPageCount);
        Assert.Contains("HelloTensorSharpPdf", result.Text.Replace(" ", string.Empty));
    }

    [Fact]
    public void ExtractFromBytes_ConcatenatesAllPagesByDefault()
    {
        byte[] pdf = BuildPdf("AlphaPageOne", "BravoPageTwo", "CharliePageThree");

        PdfTextResult result = PdfTextExtractor.ExtractFromBytes(pdf);
        string flat = result.Text.Replace(" ", string.Empty);

        Assert.Equal(3, result.PageCount);
        Assert.Equal(3, result.ExtractedPageCount);
        Assert.Contains("AlphaPageOne", flat);
        Assert.Contains("BravoPageTwo", flat);
        Assert.Contains("CharliePageThree", flat);
    }

    [Fact]
    public void ExtractFromBytes_MaxPagesCapsExtractionButReportsTotal()
    {
        byte[] pdf = BuildPdf("AlphaPageOne", "BravoPageTwo", "CharliePageThree");

        PdfTextResult result = PdfTextExtractor.ExtractFromBytes(pdf, maxPages: 2);
        string flat = result.Text.Replace(" ", string.Empty);

        Assert.Equal(3, result.PageCount);          // total pages in the document
        Assert.Equal(2, result.ExtractedPageCount); // only the first two were read
        Assert.Contains("AlphaPageOne", flat);
        Assert.Contains("BravoPageTwo", flat);
        Assert.DoesNotContain("CharliePageThree", flat);
    }

    [Fact]
    public void ExtractFromBytes_RejectsEmptyData()
    {
        Assert.Throws<ArgumentException>(() => PdfTextExtractor.ExtractFromBytes(Array.Empty<byte>()));
    }

    [Fact]
    public void ExtractFromBytes_ThrowsInvalidDataForNonPdfBytes()
    {
        byte[] garbage = System.Text.Encoding.ASCII.GetBytes("this is not a pdf file at all");

        Assert.Throws<InvalidDataException>(() => PdfTextExtractor.ExtractFromBytes(garbage));
    }

    [Fact]
    public void ExtractFromFile_RoundTripsThroughDisk()
    {
        byte[] pdf = BuildPdf("DiskRoundTripToken");
        string tmp = Path.Combine(Path.GetTempPath(), $"ts-pdf-test-{Guid.NewGuid():N}.pdf");
        try
        {
            File.WriteAllBytes(tmp, pdf);
            PdfTextResult result = PdfTextExtractor.ExtractFromFile(tmp);
            Assert.Equal(1, result.PageCount);
            Assert.Contains("DiskRoundTripToken", result.Text.Replace(" ", string.Empty));
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    [Fact]
    public void ExtractFromFile_ThrowsWhenMissing()
    {
        string missing = Path.Combine(Path.GetTempPath(), $"ts-missing-{Guid.NewGuid():N}.pdf");
        Assert.Throws<FileNotFoundException>(() => PdfTextExtractor.ExtractFromFile(missing));
    }
}
