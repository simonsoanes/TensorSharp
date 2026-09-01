// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.RegularExpressions;

namespace InferenceWeb.Tests;

/// <summary>
/// Guards the bundled Web UI's live-activity policy without introducing a browser or
/// JavaScript test dependency. TensorSharp.Server copies <c>wwwroot</c> into the test
/// output through the project reference, so these assertions inspect the exact asset a
/// built server ships rather than reaching back into the source tree.
/// </summary>
public class WebUiLiveActivitySourceTests
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

    private static string WindowAfter(string source, string marker, int length)
    {
        int start = source.IndexOf(marker, StringComparison.Ordinal);
        Assert.True(start >= 0, $"Could not find Web UI marker: {marker}");
        return source.Substring(start, Math.Min(length, source.Length - start));
    }

    [Fact]
    public void CompletedOperationHistorySurface_IsNotShipped()
    {
        string html = ReadWebUi();

        // These were the append-only renderer and CSS hooks for the old completed
        // skills/tool transcript. Their return would make prior processing visible
        // again even if the new current-activity panel kept working.
        foreach (string retired in new[]
        {
            "appendSkillStep(",
            "ensureSkillBlock(",
            "skill-step-line",
            "skill-block-title",
            "tool-draft",
            "tool-output",
            "tool-live-preview",
        })
        {
            Assert.DoesNotContain(retired, html, StringComparison.Ordinal);
        }

        Assert.Contains("function ensureCurrentActivity(", html, StringComparison.Ordinal);
        Assert.Contains("function clearCurrentActivity(", html, StringComparison.Ordinal);
    }

    [Fact]
    public void CurrentActivity_IsClearedForFinishedDoneAndFinally()
    {
        string html = ReadWebUi();
        string progress = Between(
            html, "function updateToolProgress(", "const ARTIFACT_URL_PREFIX");

        // Whitespace and comments may evolve; pin the semantic relationship within
        // the focused function instead of snapshotting its formatting.
        Assert.Matches(new Regex(
            "phase\\s*===\\s*['\"]finished['\"][\\s\\S]{0,240}?clearCurrentActivity\\s*\\(\\s*assistDiv\\s*\\)",
            RegexOptions.CultureInvariant), progress);

        string request = Between(
            html, "async function requestAssistantResponse()", "function revertFrom(");
        string done = WindowAfter(request, "if (data.done)", 320);
        Assert.Contains("clearCurrentActivity(assistDiv)", done, StringComparison.Ordinal);

        int finallyIndex = request.LastIndexOf("finally", StringComparison.Ordinal);
        Assert.True(finallyIndex >= 0, "requestAssistantResponse has no terminal finally block");
        string finallyHead = request.Substring(
            finallyIndex, Math.Min(700, request.Length - finallyIndex));
        Assert.Contains("clearCurrentActivity(assistDiv)", finallyHead, StringComparison.Ordinal);
    }

    [Fact]
    public void SkillStep_PreservesArtifactsWithoutRenderingCompletedStatus()
    {
        string html = ReadWebUi();
        string request = Between(
            html, "async function requestAssistantResponse()", "function revertFrom(");
        string skillStep = Between(request, "if (data.skill_step)", "if (data.tool_progress)");

        Assert.Contains("appendArtifactFiles(assistDiv, data.files)", skillStep, StringComparison.Ordinal);

        // The completion frame may expose durable output files, but it must not
        // create or populate a completed-step/status node.
        Assert.DoesNotContain("appendSkillStep", skillStep, StringComparison.Ordinal);
        Assert.DoesNotContain("ensureCurrentActivity", skillStep, StringComparison.Ordinal);
        Assert.DoesNotContain("document.createElement", skillStep, StringComparison.Ordinal);
        Assert.DoesNotContain("textContent", skillStep, StringComparison.Ordinal);
    }
}
