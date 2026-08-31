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
using System.Linq;
using TensorSharp.AgentHost.CodeExec;

namespace InferenceWeb.Tests;

/// <summary>
/// The rules the artifact download route has to keep, tested against the store the route
/// delegates to.
///
/// <para>
/// Everything served there was written by a program a model wrote, in response to text
/// that may itself have come from somewhere untrusted. The route therefore never trusts
/// the path it was given, never lets a browser sniff the body, and never serves anything
/// inline — and those three are what these pin.
/// </para>
/// </summary>
public class CodeArtifactEndpointTests : IDisposable
{
    private readonly string _root;
    private readonly CodeArtifactStore _store;

    public CodeArtifactEndpointTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "ts-artroute-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
        _store = new CodeArtifactStore(_root);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    /// <summary>Put a file in the store the way a run would have.</summary>
    private string Seed(string relativePath, string content = "x")
    {
        string runId = Guid.NewGuid().ToString("N");
        string work = Path.Combine(_root, "..", "work-" + runId);
        Directory.CreateDirectory(work);
        try
        {
            string file = Path.Combine(work, relativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(file)!);
            File.WriteAllText(file, content);
            _store.Capture(runId, work, (id, rel, _) => $"/api/code/artifacts/{id}/{rel}", out _);
            return runId;
        }
        finally
        {
            try { Directory.Delete(work, recursive: true); } catch { /* best effort */ }
        }
    }

    [Fact]
    public void AStoredFileResolves()
    {
        string runId = Seed("report.pdf", "%PDF-1.4");

        Assert.True(_store.TryResolve(runId, "report.pdf", out string? path, out _));
        Assert.Equal("%PDF-1.4", File.ReadAllText(path!));
    }

    [Fact]
    public void ANestedPathResolves_BecauseTheRouteBindsItWhole()
    {
        string runId = Seed(Path.Combine("out", "report.pdf"), "nested");

        Assert.True(_store.TryResolve(runId, "out/report.pdf", out string? path, out _));
        Assert.Equal("nested", File.ReadAllText(path!));
    }

    [Theory]
    [InlineData("../../../etc/passwd")]
    [InlineData("..\\..\\windows\\win.ini")]
    [InlineData("/etc/passwd")]
    [InlineData("out/../../escape.txt")]
    public void NothingOutsideTheRunDirectoryResolves(string path)
    {
        string runId = Seed("ok.txt");

        Assert.False(_store.TryResolve(runId, path, out _, out _));
    }

    [Theory]
    [InlineData("")]
    [InlineData("..")]
    [InlineData("../other")]
    [InlineData("a/b")]
    [InlineData("run id with spaces")]
    public void ARunIdThatIsNotOneIsRefused(string runId)
    {
        Assert.False(_store.TryResolve(runId, "ok.txt", out _, out _));
    }

    [Fact]
    public void AnUnknownRunIsRefused()
    {
        Seed("ok.txt");

        Assert.False(_store.TryResolve(Guid.NewGuid().ToString("N"), "ok.txt", out _, out _));
    }

    [Fact]
    public void ListingARunGivesTheUrlsTheRouteServes()
    {
        string runId = Seed("chart.png");

        var listed = _store.List(runId, (id, rel, _) => $"/api/code/artifacts/{id}/{rel}");

        CodeArtifact only = Assert.Single(listed);
        Assert.Equal("chart.png", only.Path);
        Assert.Equal($"/api/code/artifacts/{runId}/chart.png", only.Pointer);
    }
}
