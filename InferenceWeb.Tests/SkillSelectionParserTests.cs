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
using System.Text.Json;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the request field, including every malformed shape it must survive.
///
/// <para>
/// <c>skills</c> is spelled identically on all four surfaces — <c>/v1/chat/completions</c>,
/// <c>/v1/responses</c>, Ollama's <c>/api/chat</c> and the Web UI's — so one reader
/// serves them all, and it sees whatever a third-party client happens to send. Nothing
/// here may throw: a malformed corner of a request has to fail that corner and not the
/// whole completion (issue #142), because an exception thrown while reading an optional
/// field turns a request that was perfectly answerable into a 500 the caller cannot
/// diagnose. Every case below therefore asserts a value or a null, never an exception.
/// </para>
/// <para>
/// The distinction between "absent" and "present and empty" is the one piece of real
/// logic. A server started with <c>--skill pdf</c> applies that selection to any request
/// that does not carry its own, so a client needs a way to say "none for this one" —
/// and <c>"skills": []</c> is the only spelling available to it. Collapsing an empty
/// array to null would silently ignore the opt-out and hand the model a skill the caller
/// explicitly declined.
/// </para>
/// </summary>
public class SkillSelectionParserTests
{
    /// <summary>
    /// A detached element, so it stays valid after the document that produced it is
    /// gone — the same shape the request pipeline hands the parser.
    /// </summary>
    private static JsonElement Body(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        return document.RootElement.Clone();
    }

    // ---- the selection -----------------------------------------------------

    [Fact]
    public void Parse_AnArrayOfStrings_IsReadInOrder()
    {
        List<string> names = SkillSelectionParser.Parse(Body("""{"skills": ["pdf", "xlsx"]}"""));

        Assert.Equal(new[] { "pdf", "xlsx" }, names);
    }

    [Fact]
    public void Parse_ASingleBareString_IsAcceptedAsAOneSkillSelection()
    {
        // A natural thing to write, and it means exactly one thing, so accepting it
        // costs nothing and spares the caller a silently ignored field.
        Assert.Equal(new[] { "pdf" }, SkillSelectionParser.Parse(Body("""{"skills": "pdf"}""")));
    }

    [Fact]
    public void Parse_ObjectsCarryingANameOrAnId_AreRead()
    {
        // What a client that models skills as objects sends. Reading the name out
        // spares the user a baffling "no skill called '{...}'".
        Assert.Equal(
            new[] { "pdf", "xlsx" },
            SkillSelectionParser.Parse(Body("""{"skills": [{"name": "pdf"}, {"id": "xlsx"}]}""")));
    }

    [Fact]
    public void Parse_AnAbsentField_IsNull()
    {
        Assert.Null(SkillSelectionParser.Parse(Body("""{"model": "m", "messages": []}""")));
    }

    [Theory]
    [InlineData("""{"skills": 5}""")]
    [InlineData("""{"skills": true}""")]
    [InlineData("""{"skills": null}""")]
    [InlineData("""{"skills": {"name": "pdf"}}""")]
    public void Parse_AFieldThatIsNotAnArrayOrAString_IsNull(string body)
    {
        // Null rather than an empty list: "the caller said nothing usable" and "the
        // caller asked for no skills" are different requests, and only the second one
        // may override a server-wide --skill selection.
        Assert.Null(SkillSelectionParser.Parse(Body(body)));
    }

    [Fact]
    public void Parse_APresentButEmptyArray_IsAnEmptySelectionNotSilence()
    {
        List<string> names = SkillSelectionParser.Parse(Body("""{"skills": []}"""));

        Assert.NotNull(names);
        Assert.Empty(names);
    }

    [Fact]
    public void Parse_AMixedArray_KeepsWhatIsUsableAndSkipsTheRest()
    {
        // One unusable entry must not cost the caller the skills it did spell correctly.
        List<string> names = SkillSelectionParser.Parse(
            Body("""{"skills": ["pdf", 7, null, {"description": "no name here"}, {"name": "xlsx"}, ""]}"""));

        Assert.Equal(new[] { "pdf", "xlsx" }, names);
    }

    [Fact]
    public void Parse_TrimsSurroundingWhitespace()
    {
        // The name is looked up by exact match against the registry, so a stray space
        // pasted in from a UI would otherwise read as an unknown skill.
        Assert.Equal(new[] { "pdf" }, SkillSelectionParser.Parse(Body("""{"skills": ["  pdf  "]}""")));
    }

    [Theory]
    [InlineData("""[1, 2, 3]""")]
    [InlineData("\"a string, not an object\"")]
    [InlineData("""5""")]
    [InlineData("""null""")]
    public void Parse_ABodyThatIsNotAnObject_IsNull(string body)
    {
        Assert.Null(SkillSelectionParser.Parse(Body(body)));
    }

    // ---- the discovery override ---------------------------------------------

    [Fact]
    public void ParseDiscovery_ReadsBothBooleans()
    {
        Assert.True(SkillSelectionParser.ParseDiscovery(Body("""{"skills_discovery": true}""")));
        Assert.False(SkillSelectionParser.ParseDiscovery(Body("""{"skills_discovery": false}""")));
    }

    [Theory]
    [InlineData("""{}""")]
    [InlineData("""{"skills_discovery": "yes"}""")]
    [InlineData("""{"skills_discovery": 1}""")]
    [InlineData("""{"skills_discovery": null}""")]
    [InlineData("""[]""")]
    public void ParseDiscovery_AnythingElse_LeavesTheServerDefaultInForce(string body)
    {
        // Null, not false: a value the parser cannot read must not silently turn a
        // feature off that the operator configured on.
        Assert.Null(SkillSelectionParser.ParseDiscovery(Body(body)));
    }

    // ---- nothing throws -----------------------------------------------------

    [Theory]
    [InlineData("""{"skills": [[["pdf"]]]}""")]
    [InlineData("""{"skills": [{"name": 5}]}""")]
    [InlineData("""{"skills": [{"name": null}]}""")]
    [InlineData("""{"skills": [{}]}""")]
    [InlineData("""{"skills": ["   "]}""")]
    [InlineData("""{"skills": {"skills": {"skills": []}}}""")]
    [InlineData("""{"SKILLS": ["pdf"]}""")]
    public void Parse_EveryMalformedShape_AnswersInsteadOfThrowing(string body)
    {
        // The contract this class exists for: reading an optional field must never be
        // able to fail a completion the model could otherwise have answered.
        Exception? thrown = Record.Exception(() =>
        {
            SkillSelectionParser.Parse(Body(body));
            SkillSelectionParser.ParseDiscovery(Body(body));
        });

        Assert.Null(thrown);
    }
}
