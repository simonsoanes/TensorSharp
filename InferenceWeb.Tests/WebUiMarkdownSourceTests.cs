// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace InferenceWeb.Tests;

/// <summary>
/// The bundled Web UI renders the model's Markdown (it used to print it raw: tables as
/// rows of pipes). What these pin is the property that makes that safe to do with
/// text a model wrote -- it becomes DOM nodes and text nodes, never parsed HTML -- and
/// that the streaming path really uses it. Source checks, like
/// WebUiLiveActivitySourceTests, so the suite needs no browser; the renderer's output
/// was checked in Chromium when it was written.
/// </summary>
public class WebUiMarkdownSourceTests
{
    private static string ReadWebUi()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "wwwroot", "index.html");
        Assert.True(File.Exists(path),
            $"The TensorSharp.Server Web UI was not copied to the test output: {path}");
        return File.ReadAllText(path);
    }

    private static string Between(string source, string startMarker, string endMarker)
    {
        int start = source.IndexOf(startMarker, StringComparison.Ordinal);
        Assert.True(start >= 0, $"Could not find Web UI marker: {startMarker}");
        int end = source.IndexOf(endMarker, start + startMarker.Length, StringComparison.Ordinal);
        Assert.True(end > start, $"Could not find Web UI marker after {startMarker}: {endMarker}");
        return source.Substring(start, end - start);
    }

    // From the first line of code: the comment above it names what the code avoids.
    private static string Renderer(string html) => Between(html, "const MD_FENCE", "async function sendMessage(");

    [Fact]
    public void TheRenderer_NeverHandsModelTextToAnHtmlParser()
    {
        string renderer = Renderer(ReadWebUi());

        Assert.Contains("function renderMarkdown(el, text)", renderer, StringComparison.Ordinal);
        foreach (string parser in new[]
        {
            "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write",
            "DOMParser", "createContextualFragment", "srcdoc",
        })
        {
            Assert.DoesNotContain(parser, renderer, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void Links_KeepOnlySafeSchemes_AndOnlyArtifactsBecomeImages()
    {
        string renderer = Renderer(ReadWebUi());

        Assert.Contains(@"const MD_LINK_OK = /^(https?:\/\/|mailto:)/i;", renderer, StringComparison.Ordinal);

        // Anything else -- javascript:, data:, a relative path -- stays the text it was.
        string link = Between(renderer, "function mdLink(", "const MD_INLINE");
        Assert.Contains("host.appendChild(document.createTextNode(raw));", link, StringComparison.Ordinal);

        // An <img> is made only for this server's own artifact route: a remote image
        // would be fetched the moment the answer rendered.
        int artifactBranch = link.IndexOf("url.startsWith(ARTIFACT_URL_PREFIX)", StringComparison.Ordinal);
        int img = link.IndexOf("mdNode('img'", StringComparison.Ordinal);
        int external = link.IndexOf("MD_LINK_OK.test(url)", StringComparison.Ordinal);
        Assert.True(artifactBranch >= 0 && img > artifactBranch && img < external,
            "the only <img> the renderer creates must be inside the artifact branch");
        Assert.Equal(1, CountOf(renderer, "mdNode('img'"));
    }

    [Fact]
    public void StreamedAnswers_AreRenderedAsMarkdown()
    {
        string html = ReadWebUi();
        string send = Between(html, "async function sendMessage(", "function addUserBubble(");

        string token = Between(send, "if (data.token) {", "if (data.replace !== undefined)");
        Assert.Contains("renderLiveAnswer();", token, StringComparison.Ordinal);
        Assert.DoesNotContain("textContent = fullText", token, StringComparison.Ordinal);

        // At the end, the whole answer once more -- after any frame still pending is
        // cancelled, so a late frame cannot overwrite an error with the partial text.
        // (Anchored on the chat stream: the image and video flows have done frames too.)
        string done = Between(send, "if (data.replace !== undefined)", "let stats =");
        Assert.True(done.IndexOf("cancelLiveAnswer();", StringComparison.Ordinal)
                    < done.IndexOf("renderMarkdown(bubbleText, fullText)", StringComparison.Ordinal));
        Assert.Contains("renderMarkdown(bubbleText, fullText)", done, StringComparison.Ordinal);
    }

    /// <summary>
    /// Every run publishes its files under its own URL, so chips keyed by URL gave a
    /// program the model ran three times three chips for one file (seen 2026-09-30:
    /// four copies of mortgage.py under one answer). One chip per name, latest run.
    /// </summary>
    [Fact]
    public void GeneratedFiles_GetOneChipPerName_PointingAtTheLatestRun()
    {
        string chips = Between(ReadWebUi(), "function appendArtifactFiles(", "// ---- Markdown");

        Assert.Contains("x.dataset.name === name", chips, StringComparison.Ordinal);
        Assert.DoesNotContain("a[href=", chips, StringComparison.Ordinal);
        // An existing chip is re-pointed, not duplicated.
        Assert.True(chips.IndexOf("a.href = url;", StringComparison.Ordinal)
                    > chips.IndexOf("if (!a) {", StringComparison.Ordinal));
    }

    [Fact]
    public void TheAnswerContainer_IsABlockElement()
    {
        // A rendered answer holds paragraphs, lists and tables, which a <span> may not.
        string bubble = Between(ReadWebUi(), "function addAssistantBubble()", "function toggleThinking(");
        Assert.Contains("<div class=\"bubble-text\"></div>", bubble, StringComparison.Ordinal);
    }

    private static int CountOf(string text, string needle)
    {
        int count = 0;
        for (int at = text.IndexOf(needle, StringComparison.Ordinal); at >= 0;
             at = text.IndexOf(needle, at + needle.Length, StringComparison.Ordinal))
            count++;
        return count;
    }
}
