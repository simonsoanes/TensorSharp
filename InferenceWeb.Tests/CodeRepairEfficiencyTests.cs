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
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Regression tests for the economic and correctness properties of localized repair.
/// A small repair must produce the exact bytes a full rewrite would have produced,
/// without requiring the model to emit all of the bytes that were already correct.
/// </summary>
public sealed class CodeRepairEfficiencyTests : IDisposable
{
    private readonly string _base;
    private readonly SessionWorkspaceManager _workspaces;
    private readonly SessionWorkspace _workspace;
    private readonly ShellRunner _runner;

    public CodeRepairEfficiencyTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-repair-efficiency-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
        _workspaces = new SessionWorkspaceManager(Path.Combine(_base, "sessions"));
        _workspace = _workspaces.GetOrCreate("s");
        _runner = new ShellRunner(new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            ScratchDirectory = _base,
        });
    }

    public void Dispose()
    {
        _runner.Dispose();
        try { _workspaces.Release("s"); } catch (Exception ex) when (ex is IOException) { }
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void LocalizedEditAndPatch_MatchAFullRewrite_AndPreserveAllOtherBytes()
    {
        RepairCase repair = CodeRepairCorpus.Create(lineCount: 2_000);
        string editPath = Write("edit.cs", repair.Before);
        string patchPath = Write("patch.cs", repair.Before);

        CodeExecResult edit = _runner.EditFile(
            new ShellTools.EditRequest("edit.cs", repair.OldLine, repair.NewLine, ReplaceAll: false),
            _workspace);
        CodeExecResult patch = _runner.ApplyPatch(
            repair.Patch.Replace(CodeRepairCorpus.RelativePath, "patch.cs", StringComparison.Ordinal),
            _workspace);

        Assert.True(edit.Ok, edit.Content);
        Assert.True(patch.Ok, patch.Content);

        string edited = File.ReadAllText(editPath);
        string patched = File.ReadAllText(patchPath);
        Assert.Equal(repair.After, edited);
        Assert.Equal(repair.After, patched);

        // Equality with After proves the complete result. These two assertions pin the
        // more useful invariant explicitly: neither localized path may manufacture a
        // change before or after the named span.
        Assert.Equal(repair.Before[..repair.ChangeOffset], edited[..repair.ChangeOffset]);
        Assert.Equal(
            repair.Before[(repair.ChangeOffset + repair.OldLine.Length)..],
            edited[(repair.ChangeOffset + repair.NewLine.Length)..]);
        Assert.Equal(repair.Before[..repair.ChangeOffset], patched[..repair.ChangeOffset]);
        Assert.Equal(
            repair.Before[(repair.ChangeOffset + repair.OldLine.Length)..],
            patched[(repair.ChangeOffset + repair.NewLine.Length)..]);
    }

    [Fact]
    public void LocalizedToolCalls_UseLessThanFivePercentOfTheRewritePayload()
    {
        RepairCase repair = CodeRepairCorpus.Create(lineCount: 2_000);

        int rewriteBytes = Encoding.UTF8.GetByteCount(repair.RewritePayload);
        int editBytes = Encoding.UTF8.GetByteCount(repair.EditPayload);
        int patchBytes = Encoding.UTF8.GetByteCount(repair.PatchPayload);
        int rewriteTokens = CodeRepairCorpus.CountByteTokens(repair.RewritePayload);
        int editTokens = CodeRepairCorpus.CountByteTokens(repair.EditPayload);
        int patchTokens = CodeRepairCorpus.CountByteTokens(repair.PatchPayload);

        // This deterministic proxy drives the production tokenizer implementation with
        // one vocabulary entry per normalized byte and no merges. It is deliberately
        // conservative and avoids pretending that chars/4 is a tokenizer measurement.
        Assert.Equal(rewriteBytes, rewriteTokens);
        Assert.Equal(editBytes, editTokens);
        Assert.Equal(patchBytes, patchTokens);

        Assert.True(editBytes * 20 < rewriteBytes,
            $"edit payload was {editBytes:N0} bytes versus {rewriteBytes:N0} for a rewrite");
        Assert.True(patchBytes * 20 < rewriteBytes,
            $"patch payload was {patchBytes:N0} bytes versus {rewriteBytes:N0} for a rewrite");
        Assert.True(editTokens * 20 < rewriteTokens,
            $"edit payload was {editTokens:N0} byte-fallback tokens versus {rewriteTokens:N0} for a rewrite");
        Assert.True(patchTokens * 20 < rewriteTokens,
            $"patch payload was {patchTokens:N0} byte-fallback tokens versus {rewriteTokens:N0} for a rewrite");

        // The localized calls must not smuggle the surrounding source back into their
        // arguments. This catches a superficially small API whose serializer quietly
        // includes the full document as context.
        Assert.Contains("Value00001", repair.RewritePayload, StringComparison.Ordinal);
        Assert.DoesNotContain("Value00001", repair.EditPayload, StringComparison.Ordinal);
        Assert.DoesNotContain("Value00001", repair.PatchPayload, StringComparison.Ordinal);
        Assert.Contains("Value02000", repair.RewritePayload, StringComparison.Ordinal);
        Assert.DoesNotContain("Value02000", repair.EditPayload, StringComparison.Ordinal);
        Assert.DoesNotContain("Value02000", repair.PatchPayload, StringComparison.Ordinal);
    }

    private string Write(string name, string content)
    {
        string path = Path.Combine(_workspace.WorkDirectory, name);
        File.WriteAllText(path, content);
        return path;
    }
}

/// <summary>
/// Model-free micro-benchmark for localized repair. Serialized bytes are the stable
/// payload measure. The one-token-per-byte vocabulary is explicitly a conservative
/// token proxy, not a claim about a particular model's tokenizer. Timings cover only
/// the in-memory edit/patch transforms, not model inference or end-to-end file I/O.
/// </summary>
[Trait("Category", "Bench")]
public sealed class CodeRepairEfficiencyBenchmark
{
    [Fact]
    public void SingleLineRepair_PayloadAndApplicationCost_ByFileSize()
    {
        Console.WriteLine(
            "[CodeRepair] serialized tool-call payload; byte-token is a deterministic conservative proxy");

        foreach ((int lines, int iterations) in new[]
        {
            (100, 500),
            (1_000, 150),
            (10_000, 20),
        })
        {
            RepairCase repair = CodeRepairCorpus.Create(lines);
            int rewriteBytes = Encoding.UTF8.GetByteCount(repair.RewritePayload);
            int editBytes = Encoding.UTF8.GetByteCount(repair.EditPayload);
            int patchBytes = Encoding.UTF8.GetByteCount(repair.PatchPayload);
            int rewriteTokens = CodeRepairCorpus.CountByteTokens(repair.RewritePayload);
            int editTokens = CodeRepairCorpus.CountByteTokens(repair.EditPayload);
            int patchTokens = CodeRepairCorpus.CountByteTokens(repair.PatchPayload);

            MeasureStats edit = Measure(
                () => CodeRepairCorpus.ApplyEdit(repair), iterations, repair.After);
            MeasureStats patch = Measure(
                () => CodeRepairCorpus.ApplyPatch(repair), iterations, repair.After);

            Console.WriteLine(
                $"[CodeRepair] lines={lines,6:N0} "
                + $"rewrite={rewriteBytes,9:N0} B/{rewriteTokens,9:N0} byte-token  "
                + $"edit={editBytes,5:N0} B/{editTokens,5:N0} byte-token "
                + $"({100.0 * editBytes / rewriteBytes,6:F2}%)  "
                + $"patch={patchBytes,5:N0} B/{patchTokens,5:N0} byte-token "
                + $"({100.0 * patchBytes / rewriteBytes,6:F2}%)");
            Console.WriteLine(
                $"[CodeRepair] lines={lines,6:N0} "
                + $"edit in-memory p50={edit.P50Us,9:F1} us p95={edit.P95Us,9:F1} us "
                + $"alloc={edit.BytesAllocated / 1024.0,9:F1} KiB/op; "
                + $"patch in-memory p50={patch.P50Us,9:F1} us p95={patch.P95Us,9:F1} us "
                + $"alloc={patch.BytesAllocated / 1024.0,9:F1} KiB/op");

            Assert.True(editBytes < rewriteBytes);
            Assert.True(patchBytes < rewriteBytes);
        }
    }

    private static MeasureStats Measure(Func<string> operation, int iterations, string expected)
    {
        for (int i = 0; i < 8; i++)
        {
            if (!string.Equals(expected, operation(), StringComparison.Ordinal))
                throw new InvalidOperationException("localized repair benchmark produced incorrect output");
        }

        var ticks = new long[iterations];
        long allocationStart = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < iterations; i++)
        {
            long start = Stopwatch.GetTimestamp();
            string actual = operation();
            ticks[i] = Stopwatch.GetTimestamp() - start;
            if (!string.Equals(expected, actual, StringComparison.Ordinal))
                throw new InvalidOperationException("localized repair benchmark produced incorrect output");
            GC.KeepAlive(actual);
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - allocationStart;

        Array.Sort(ticks);
        return new MeasureStats(
            ToMicroseconds(ticks[(iterations - 1) / 2]),
            ToMicroseconds(ticks[(int)Math.Floor((iterations - 1) * 0.95)]),
            allocated / iterations);
    }

    private static double ToMicroseconds(long ticks) =>
        ticks * 1_000_000.0 / Stopwatch.Frequency;

    private readonly record struct MeasureStats(double P50Us, double P95Us, long BytesAllocated);
}

internal static class CodeRepairCorpus
{
    internal const string RelativePath = "src/LargeCalculator.cs";

    private static readonly BpeTokenizer ByteTokenizer = CreateByteTokenizer();

    internal static RepairCase Create(int lineCount)
    {
        if (lineCount < 2)
            throw new ArgumentOutOfRangeException(nameof(lineCount));

        int target = lineCount / 2;
        var source = new StringBuilder(lineCount * 48);
        source.Append("namespace RepairFixture;\n\ninternal static class LargeCalculator\n{\n");
        for (int i = 1; i <= lineCount; i++)
            source.Append("    public static int Value").Append(i.ToString("D5"))
                .Append(" => ").Append(i).Append(";\n");
        source.Append("}\n");

        string before = source.ToString();
        string oldLine = $"    public static int Value{target:D5} => {target};";
        string newLine = $"    public static int Value{target:D5} => -{target};";
        int offset = before.IndexOf(oldLine, StringComparison.Ordinal);
        if (offset < 0)
            throw new InvalidOperationException("repair fixture target was not generated");
        string after = before.Remove(offset, oldLine.Length).Insert(offset, newLine);
        string patch = "*** Begin Patch\n"
                     + $"*** Update File: {RelativePath}\n"
                     + "@@\n"
                     + $"-{oldLine}\n"
                     + $"+{newLine}\n"
                     + "*** End Patch";

        return new RepairCase(
            before,
            after,
            oldLine,
            newLine,
            offset,
            patch,
            ToolPayload(ShellTools.WriteToolName, new Dictionary<string, object?>
            {
                ["path"] = RelativePath,
                ["content"] = after,
            }),
            ToolPayload(ShellTools.EditToolName, new Dictionary<string, object?>
            {
                ["path"] = RelativePath,
                ["old_string"] = oldLine,
                ["new_string"] = newLine,
                ["replace_all"] = false,
            }),
            ToolPayload(ShellTools.PatchToolName, new Dictionary<string, object?>
            {
                ["patch"] = patch,
            }));
    }

    internal static string ApplyEdit(RepairCase repair)
    {
        FileEdit.MatchResult match = FileEdit.Find(repair.Before, repair.OldLine);
        if (!match.Found || match.Count != 1)
            throw new InvalidOperationException("localized edit fixture stopped matching uniquely");
        return repair.Before.Remove(match.Index, match.Search.Length).Insert(match.Index, repair.NewLine);
    }

    internal static string ApplyPatch(RepairCase repair)
    {
        string body = $"-{repair.OldLine}\n+{repair.NewLine}\n";
        V4ADiff.DiffResult result = V4ADiff.Update(repair.Before, V4ADiff.SplitDiffLines(body));
        if (!result.Ok)
            throw new InvalidOperationException(result.Error);
        return result.Text;
    }

    internal static int CountByteTokens(string payload) =>
        ByteTokenizer.Encode(payload, addSpecial: false).Count;

    private static string ToolPayload(string name, Dictionary<string, object?> arguments) =>
        JsonSerializer.Serialize(new { name, arguments });

    private static BpeTokenizer CreateByteTokenizer()
    {
        // BpeTokenizer's default pre-tokenizer maps every UTF-8 byte to one Unicode
        // vocabulary character before merging. Supplying exactly those 256 characters
        // and no merges makes token count equal serialized UTF-8 bytes, deterministically.
        string[] vocab = Enumerable.Range(0, 256).Select(i => ByteVocabularyChar((byte)i).ToString()).ToArray();
        return new BpeTokenizer(
            vocab,
            new int[vocab.Length],
            Array.Empty<string>(),
            bosTokenId: -1,
            eosTokenIds: Array.Empty<int>(),
            addBos: false,
            addEos: false);
    }

    private static char ByteVocabularyChar(byte value)
    {
        char normalized = (char)value;
        if (normalized == 0x00ad)
            return (char)0x0143;
        if (normalized <= 0x0020)
            return (char)(normalized + 0x0100);
        if (normalized >= 0x007f && normalized <= 0x00a0)
            return (char)(normalized + 0x00a2);
        return normalized;
    }
}

internal readonly record struct RepairCase(
    string Before,
    string After,
    string OldLine,
    string NewLine,
    int ChangeOffset,
    string Patch,
    string RewritePayload,
    string EditPayload,
    string PatchPayload);
