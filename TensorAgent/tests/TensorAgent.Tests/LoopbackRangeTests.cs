// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Net;
using System.Net.Sockets;
using System.Text;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Hosting;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;

namespace TensorAgent.Tests;

/// <summary>
/// Byte ranges and HEAD on the files the loopback server hands the WebView.
///
/// <para>
/// A generated clip is useless if its <c>&lt;video&gt;</c> will not play. WebKit gives
/// media to AVFoundation, which opens a URL with <c>Range: bytes=0-1</c>, takes the size
/// from the 206's Content-Range and seeks with more ranges; the server used to answer
/// every one of those with 200 and the whole file, never said Accept-Ranges, and sent
/// a 404 -- body and all -- to a HEAD. The desktop never had the problem: ASP.NET's
/// static files do ranges. These go through the real <c>/uploads</c> route, the real
/// static root and the real code-artifact route with a real HttpClient, because what
/// breaks is the status line and the headers.
/// </para>
/// </summary>
public sealed class LoopbackRangeTests : IDisposable
{
    /// <summary>About the size of a short generated clip, and big enough that a slice from its middle is a seek.</summary>
    private const int ClipBytes = 3 * 1024 * 1024;

    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-range-" + Guid.NewGuid().ToString("N"));
    private readonly string _uploads;
    private readonly string _web;
    private readonly CodeArtifactStore _artifacts;
    private readonly LoopbackServer _server;
    private readonly HttpClient _client;
    private readonly byte[] _clip = new byte[ClipBytes];

    public LoopbackRangeTests()
    {
        _uploads = Path.Combine(_root, "uploads");
        _web = Path.Combine(_root, "webui");
        Directory.CreateDirectory(_uploads);
        Directory.CreateDirectory(_web);

        // Random bytes, so a slice from the wrong offset cannot match by accident.
        new Random(20260930).NextBytes(_clip);
        File.WriteAllBytes(Path.Combine(_uploads, "clip.mp4"), _clip);

        var options = new ServerHostingOptions(
            startupModelPath: Path.Combine(_root, "models", "none.gguf"),
            startupMmProjPath: null,
            defaultBackend: "ggml_cpu",
            supportedBackends: new[] { new BackendOption("ggml_cpu", "GGML CPU") },
            defaultMaxTokens: 256,
            maxTokensPinned: false,
            defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0,
            defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: null,
            uploadDirectory: _uploads,
            logDirectory: Path.Combine(_root, "logs"),
            fileLoggingEnabled: false,
            samplingDefaults: null);

        var chat = new WebUiChatService(
            new ModelService(), new SessionManager(), options,
            new UploadStoragePolicy(_uploads), new SkillRegistry(new SkillRegistryOptions()),
            codeRunner: null, workspaces: null, codeArtifacts: null,
            NullLoggerFactory.Instance);

        _artifacts = new CodeArtifactStore(Path.Combine(_root, "artifacts"));
        _server = new LoopbackServer(NullLogger.Instance) { StaticRoot = _web };
        _server.MapWebUi(chat, _uploads);
        _server.MapCodeArtifacts(_artifacts);
        _server.Start();

        _client = new HttpClient { BaseAddress = new Uri(_server.BaseUrl), Timeout = TimeSpan.FromMinutes(1) };
        _client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={_server.Token}");
    }

    public void Dispose()
    {
        _client.Dispose();
        _server.Dispose();
        try { Directory.Delete(_root, true); } catch (Exception) { /* scratch */ }
    }

    /// <summary>One request, with the Range header exactly as given -- malformed ones included.</summary>
    private async Task<(HttpResponseMessage Reply, byte[] Body)> FetchAsync(
        HttpMethod method, string url, string? range = null, HttpClient? client = null, string? ifRange = null)
    {
        using var request = new HttpRequestMessage(method, url);
        if (range is not null)
            Assert.True(request.Headers.TryAddWithoutValidation("Range", range));
        if (ifRange is not null)
            Assert.True(request.Headers.TryAddWithoutValidation("If-Range", ifRange));
        HttpResponseMessage reply = await (client ?? _client).SendAsync(request);
        return (reply, await reply.Content.ReadAsByteArrayAsync());
    }

    /// <summary>A header as it arrived, wherever HttpClient filed it; null when it was not sent.</summary>
    private static string? Header(HttpResponseMessage reply, string name) =>
        reply.Headers.TryGetValues(name, out IEnumerable<string>? values) || reply.Content.Headers.TryGetValues(name, out values)
            ? string.Join(", ", values)
            : null;

    private byte[] Slice(long first, long last) => _clip[(int)first..(int)(last + 1)];

    private static void AssertPartial(HttpResponseMessage reply, byte[] body, long first, long last, long size, byte[] expected)
    {
        Assert.Equal(HttpStatusCode.PartialContent, reply.StatusCode);
        Assert.Equal($"bytes {first}-{last}/{size}", Header(reply, "Content-Range"));
        Assert.Equal(last - first + 1, reply.Content.Headers.ContentLength);
        Assert.Equal("bytes", Header(reply, "Accept-Ranges"));
        Assert.Equal(expected, body);
    }

    // ---- the range forms ------------------------------------------------------------

    [Fact]
    public async Task APlainGetSaysRangesAreAcceptedAndStillSendsTheWholeFile()
    {
        (HttpResponseMessage reply, byte[] body) = await FetchAsync(HttpMethod.Get, "/uploads/clip.mp4");

        Assert.Equal(HttpStatusCode.OK, reply.StatusCode);
        Assert.Equal("bytes", Header(reply, "Accept-Ranges"));
        Assert.Null(Header(reply, "Content-Range"));
        Assert.Equal(ClipBytes, reply.Content.Headers.ContentLength);
        Assert.Equal(_clip, body);
        // What the file reply already said stays exactly as it was.
        Assert.Equal("video/mp4", reply.Content.Headers.ContentType?.MediaType);
        Assert.Equal("nosniff", Header(reply, "X-Content-Type-Options"));
        Assert.Equal("no-cache", Header(reply, "Cache-Control"));
        Assert.Null(Header(reply, "Content-Disposition"));
    }

    [Fact]
    public async Task TheProbeAVFoundationOpensWithAndASeekMidFileGetExactlyTheirBytes()
    {
        // The first request a media element's loader makes: two bytes, to learn the size
        // from Content-Range and that ranges work at all.
        (HttpResponseMessage probe, byte[] two) = await FetchAsync(HttpMethod.Get, "/uploads/clip.mp4", "bytes=0-1");
        AssertPartial(probe, two, 0, 1, ClipBytes, Slice(0, 1));
        Assert.Equal("video/mp4", probe.Content.Headers.ContentType?.MediaType);

        // Then a seek: one mebibyte from the middle of the clip.
        (HttpResponseMessage seek, byte[] middle) = await FetchAsync(HttpMethod.Get, "/uploads/clip.mp4", "bytes=1048576-2097151");
        AssertPartial(seek, middle, 1048576, 2097151, ClipBytes, Slice(1048576, 2097151));
    }

    [Theory]
    [InlineData("bytes=0-0", 0, 0)]
    [InlineData("bytes=100-199", 100, 199)]
    [InlineData("bytes=0-", 0, ClipBytes - 1)]
    [InlineData("bytes=3145000-", 3145000, ClipBytes - 1)]
    [InlineData("bytes=-500", ClipBytes - 500, ClipBytes - 1)]
    // A suffix longer than the file is the whole file, still as a 206.
    [InlineData("bytes=-9999999", 0, ClipBytes - 1)]
    // The unit is case-insensitive, and whitespace around '=' and '-' is tolerated as
    // ASP.NET tolerates it on the desktop.
    [InlineData("Bytes = 10 - 20", 10, 20)]
    public async Task EveryRangeFormIsA206WithExactlyItsBytes(string range, long first, long last)
    {
        (HttpResponseMessage reply, byte[] body) = await FetchAsync(HttpMethod.Get, "/uploads/clip.mp4", range);
        AssertPartial(reply, body, first, last, ClipBytes, Slice(first, last));
    }

    [Fact]
    public async Task AnEndPastTheEndOfTheFileIsClampedToItsLastByte()
    {
        (HttpResponseMessage reply, byte[] body) = await FetchAsync(HttpMethod.Get, "/uploads/clip.mp4", "bytes=3145700-99999999");
        AssertPartial(reply, body, 3145700, ClipBytes - 1, ClipBytes, Slice(3145700, ClipBytes - 1));
        Assert.Equal(28, body.Length);
    }

    [Theory]
    [InlineData("bytes=3145728-")]
    [InlineData("bytes=3145728-3145800")]
    [InlineData("bytes=99999999-")]
    // The last zero bytes: the one suffix RFC 9110 calls unsatisfiable.
    [InlineData("bytes=-0")]
    public async Task ARangeThatStartsPastTheEndIs416WithTheSizeAndNoBody(string range)
    {
        (HttpResponseMessage reply, byte[] body) = await FetchAsync(HttpMethod.Get, "/uploads/clip.mp4", range);

        Assert.Equal(HttpStatusCode.RequestedRangeNotSatisfiable, reply.StatusCode);
        Assert.Equal($"bytes */{ClipBytes}", Header(reply, "Content-Range"));
        Assert.Equal(0, reply.Content.Headers.ContentLength);
        Assert.Empty(body);
    }

    [Theory]
    [InlineData("bytes=0-1,4-5")]
    [InlineData("bytes=0-1, 4-5")]
    [InlineData("items=0-1")]
    [InlineData("bytes 0-1")]
    [InlineData("bytes=abc")]
    [InlineData("bytes=5-3")]
    [InlineData("bytes=-")]
    [InlineData("bytes=")]
    [InlineData("bytes=1-2-3")]
    [InlineData("bytes=+1-2")]
    [InlineData("bytes=99999999999999999999-")]
    public async Task AHeaderThatIsNotOneByteRangeIsIgnoredAndTheWholeFileSent(string range)
    {
        // Several ranges, or something that does not parse: RFC 9110 lets a server ignore
        // the header, and ASP.NET does. A unit it does not know it MUST ignore (§14.2),
        // which is the one case here the desktop gets wrong: it serves items=0-1 as bytes.
        (HttpResponseMessage reply, byte[] body) = await FetchAsync(HttpMethod.Get, "/uploads/clip.mp4", range);

        Assert.Equal(HttpStatusCode.OK, reply.StatusCode);
        Assert.Null(Header(reply, "Content-Range"));
        Assert.Equal("bytes", Header(reply, "Accept-Ranges"));
        Assert.Equal(_clip, body);
    }

    [Fact]
    public async Task AnIfRangeTheServerCannotMatchGetsTheWholeFile()
    {
        // This server sends no ETag and no Last-Modified, so an If-Range has nothing to
        // match, and RFC 9110 §13.1.5 then requires the whole file rather than a slice
        // that might belong to some other version of it.
        (HttpResponseMessage reply, byte[] body) = await FetchAsync(
            HttpMethod.Get, "/uploads/clip.mp4", "bytes=0-1", ifRange: "\"some-etag\"");

        Assert.Equal(HttpStatusCode.OK, reply.StatusCode);
        Assert.Null(Header(reply, "Content-Range"));
        Assert.Equal(_clip, body);
    }

    [Fact]
    public async Task AnEmptyFileIsNeverAnsweredWithAPartialResponse()
    {
        await File.WriteAllBytesAsync(Path.Combine(_uploads, "silence.wav"), []);

        (HttpResponseMessage whole, byte[] nothing) = await FetchAsync(HttpMethod.Get, "/uploads/silence.wav");
        Assert.Equal(HttpStatusCode.OK, whole.StatusCode);
        Assert.Equal("bytes", Header(whole, "Accept-Ranges"));
        Assert.Equal(0, whole.Content.Headers.ContentLength);
        Assert.Empty(nothing);

        // Every first byte is past the end of nothing.
        (HttpResponseMessage start, _) = await FetchAsync(HttpMethod.Get, "/uploads/silence.wav", "bytes=0-");
        Assert.Equal(HttpStatusCode.RequestedRangeNotSatisfiable, start.StatusCode);
        Assert.Equal("bytes */0", Header(start, "Content-Range"));

        // A non-zero suffix is satisfiable by the RFC's definition, but there is no byte
        // range of an empty file to put in a 206, so it gets the (empty) whole.
        (HttpResponseMessage suffix, byte[] empty) = await FetchAsync(HttpMethod.Get, "/uploads/silence.wav", "bytes=-1");
        Assert.Equal(HttpStatusCode.OK, suffix.StatusCode);
        Assert.Null(Header(suffix, "Content-Range"));
        Assert.Empty(empty);
    }

    // ---- HEAD -----------------------------------------------------------------------

    [Theory]
    [InlineData(null)]
    [InlineData("bytes=0-1")]
    [InlineData("bytes=1048576-")]
    [InlineData("bytes=-0")]
    [InlineData("bytes=0-1,4-5")]
    public async Task AHeadIsAnsweredWithTheStatusAndHeadersItsGetGetsAndNoBody(string? range)
    {
        // A Range on a HEAD is ignored (RFC 9110 §14.2 defines ranges for GET alone, and
        // ASP.NET ignores it too), so every HEAD describes the whole file an unranged GET gets.
        (HttpResponseMessage get, byte[] getBody) = await FetchAsync(HttpMethod.Get, "/uploads/clip.mp4");
        (HttpResponseMessage head, byte[] headBody) = await FetchAsync(HttpMethod.Head, "/uploads/clip.mp4", range);

        Assert.Equal(get.StatusCode, head.StatusCode);
        foreach (string name in new[]
        {
            "Content-Length", "Content-Range", "Content-Type", "Accept-Ranges",
            "X-Content-Type-Options", "Cache-Control",
        })
            Assert.Equal(Header(get, name), Header(head, name));
        Assert.Equal((long)getBody.Length, head.Content.Headers.ContentLength);
        Assert.Empty(headBody);
    }

    [Fact]
    public async Task AHeadForAFileThatIsNotThereIsA404WithNoBody()
    {
        (HttpResponseMessage reply, byte[] body) = await FetchAsync(HttpMethod.Head, "/uploads/missing.mp4");
        Assert.Equal(HttpStatusCode.NotFound, reply.StatusCode);
        Assert.Empty(body);
    }

    /// <summary>
    /// The wire under the HEAD tests, read by hand.
    ///
    /// <para>
    /// HttpClient reads no body after a HEAD whatever arrives, and quietly drops a
    /// connection that holds bytes it did not expect, so it cannot see the failure that
    /// matters: the managed HttpListener writes whatever it is given after a HEAD's
    /// headers, and those bytes are the first thing the NEXT response on the connection
    /// appears to say. The server did exactly that with its 404 before this change. So
    /// every kind of reply a HEAD can get goes down one connection -- a file, the page,
    /// a 404, a 405 and a 403, which are three different reply classes -- and each must
    /// be followed directly by the next status line.
    /// </para>
    /// </summary>
    [Fact]
    public async Task AHeadLeavesItsConnectionReadyForTheNextRequest()
    {
        await File.WriteAllTextAsync(Path.Combine(_web, "index.html"), "<html><body></body></html>");
        string host = $"Host: 127.0.0.1:{_server.Port}\r\n";
        string cookie = $"Cookie: {LoopbackServer.TokenCookie}={_server.Token}\r\n";
        (string Request, string Status)[] heads =
        {
            ($"HEAD /uploads/clip.mp4 HTTP/1.1\r\n{host}{cookie}\r\n", "200"),
            ($"HEAD / HTTP/1.1\r\n{host}{cookie}\r\n", "200"),
            ($"HEAD /uploads/missing.mp4 HTTP/1.1\r\n{host}{cookie}\r\n", "404"),
            ($"HEAD /api/models HTTP/1.1\r\n{host}{cookie}\r\n", "405"),
            ($"HEAD /uploads/clip.mp4 HTTP/1.1\r\n{host}\r\n", "403"),
        };

        using var tcp = new TcpClient();
        await tcp.ConnectAsync(IPAddress.Loopback, _server.Port);
        NetworkStream stream = tcp.GetStream();
        var received = new MemoryStream();
        for (int i = 0; i < heads.Length; i++)
        {
            // One at a time, so whatever follows a HEAD's headers can only be what the
            // server sent for that HEAD.
            await stream.WriteAsync(Encoding.ASCII.GetBytes(heads[i].Request));
            int blocks = i + 1;
            await ReadAsync(stream, received, until: bytes => HeaderBlocks(bytes) >= blocks);
        }
        await stream.WriteAsync(Encoding.ASCII.GetBytes(
            $"GET /uploads/clip.mp4 HTTP/1.1\r\n{host}{cookie}Range: bytes=0-1\r\nConnection: close\r\n\r\n"));
        await ReadAsync(stream, received, until: _ => false);

        byte[] wire = received.ToArray();
        int at = 0;
        foreach ((string request, string status) in heads)
        {
            int end = Find(wire, at);
            Assert.True(end >= 0, "the wire ran out before the answer to " + request);
            string headers = Encoding.ASCII.GetString(wire, at, end - at);
            Assert.True(headers.StartsWith("HTTP/1.1 " + status, StringComparison.Ordinal),
                $"after the previous HEAD, the next bytes were not the answer to {request.Split('\r')[0]}:\n{Printable(headers)}");
            at = end + 4;
        }
        Assert.Contains($"Content-Length: {ClipBytes}", Encoding.ASCII.GetString(wire, 0, Find(wire, 0)), StringComparison.Ordinal);

        int getEnd = Find(wire, at);
        Assert.True(getEnd >= 0, "no answer to the GET arrived");
        string get = Encoding.ASCII.GetString(wire, at, getEnd - at);
        Assert.StartsWith("HTTP/1.1 206", get, StringComparison.Ordinal);
        Assert.Contains($"Content-Range: bytes 0-1/{ClipBytes}", get, StringComparison.Ordinal);
        Assert.Equal(Slice(0, 1), wire[(getEnd + 4)..]);

        // Where the header block that starts at or after `from` ends, or -1.
        static int Find(byte[] bytes, int from)
        {
            int at = bytes.AsSpan(from).IndexOf("\r\n\r\n"u8);
            return at < 0 ? -1 : from + at;
        }

        static int HeaderBlocks(byte[] bytes)
        {
            int count = 0;
            for (int end = Find(bytes, 0); end >= 0; end = Find(bytes, end + 4))
                count++;
            return count;
        }

        // A stray body can be the whole 3 MB clip; the start of it says enough.
        static string Printable(string text) =>
            new(text.Take(300).Select(c => c is '\r' or '\n' || (c >= ' ' && c < '\x7f') ? c : '.').ToArray());
    }

    /// <summary>Read into <paramref name="into"/> until <paramref name="until"/> holds or the server closes; bounded so a hang fails fast.</summary>
    private static async Task ReadAsync(NetworkStream stream, MemoryStream into, Func<byte[], bool> until)
    {
        byte[] buffer = new byte[64 * 1024];
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        while (!until(into.ToArray()))
        {
            int read = await stream.ReadAsync(buffer, deadline.Token);
            if (read == 0)
                return;
            into.Write(buffer, 0, read);
        }
    }

    // ---- what did not change --------------------------------------------------------

    [Fact]
    public async Task RangedAndHeadRequestsWithoutTheLaunchCookieAreStillRefused()
    {
        using var bare = new HttpClient { BaseAddress = new Uri(_server.BaseUrl) };

        (HttpResponseMessage ranged, byte[] refusal) = await FetchAsync(HttpMethod.Get, "/uploads/clip.mp4", "bytes=0-1", bare);
        Assert.Equal(HttpStatusCode.Forbidden, ranged.StatusCode);
        Assert.Equal("{\"error\":\"forbidden\"}", Encoding.UTF8.GetString(refusal));
        Assert.Null(Header(ranged, "Content-Range"));

        (HttpResponseMessage head, byte[] body) = await FetchAsync(HttpMethod.Head, "/uploads/clip.mp4", client: bare);
        Assert.Equal(HttpStatusCode.Forbidden, head.StatusCode);
        Assert.Empty(body);

        using var wrong = new HttpClient { BaseAddress = new Uri(_server.BaseUrl) };
        wrong.DefaultRequestHeaders.Add("Cookie", LoopbackServer.TokenCookie + "=not-the-token");
        (HttpResponseMessage forged, _) = await FetchAsync(HttpMethod.Get, "/uploads/clip.mp4", "bytes=0-1", wrong);
        Assert.Equal(HttpStatusCode.Forbidden, forged.StatusCode);
    }

    [Fact]
    public async Task RangedAndHeadRequestsStillCannotLeaveTheUploadDirectory()
    {
        await File.WriteAllTextAsync(Path.Combine(_root, "secret.txt"), "not yours");
        foreach (string escape in new[] { "/uploads/../secret.txt", "/uploads/..%2Fsecret.txt" })
        {
            (HttpResponseMessage ranged, byte[] body) = await FetchAsync(HttpMethod.Get, escape, "bytes=0-3");
            Assert.True(ranged.StatusCode is HttpStatusCode.NotFound or HttpStatusCode.BadRequest,
                $"{escape} answered {(int)ranged.StatusCode}");
            Assert.DoesNotContain("not yours", Encoding.UTF8.GetString(body), StringComparison.Ordinal);

            (HttpResponseMessage head, _) = await FetchAsync(HttpMethod.Head, escape);
            Assert.True(head.StatusCode is HttpStatusCode.NotFound or HttpStatusCode.BadRequest,
                $"HEAD {escape} answered {(int)head.StatusCode}");
        }
    }

    // ---- the other file routes ------------------------------------------------------

    [Fact]
    public async Task TheStaticBundleAnswersRangesAndHeadToo()
    {
        byte[] intro = _clip[..65536];
        Directory.CreateDirectory(Path.Combine(_web, "media"));
        await File.WriteAllBytesAsync(Path.Combine(_web, "media", "intro.mp4"), intro);
        await File.WriteAllTextAsync(Path.Combine(_web, "index.html"), "<html><body><div id=\"chat\"></div></body></html>");

        (HttpResponseMessage ranged, byte[] slice) = await FetchAsync(HttpMethod.Get, "/media/intro.mp4", "bytes=1000-1999");
        AssertPartial(ranged, slice, 1000, 1999, intro.Length, intro[1000..2000]);

        (HttpResponseMessage head, byte[] none) = await FetchAsync(HttpMethod.Head, "/media/intro.mp4");
        Assert.Equal(HttpStatusCode.OK, head.StatusCode);
        Assert.Equal(intro.Length, head.Content.Headers.ContentLength);
        Assert.Equal("bytes", Header(head, "Accept-Ranges"));
        Assert.Empty(none);

        // The page is not a file reply: the companion script is appended on the way out,
        // so it ignores a Range and does not claim to take one. Its HEAD still carries
        // the length its GET sends, script tag included.
        (HttpResponseMessage page, byte[] html) = await FetchAsync(HttpMethod.Get, "/", "bytes=0-1");
        Assert.Equal(HttpStatusCode.OK, page.StatusCode);
        Assert.Null(Header(page, "Accept-Ranges"));
        Assert.Contains("tensoragent.js", Encoding.UTF8.GetString(html), StringComparison.Ordinal);

        (HttpResponseMessage pageHead, byte[] pageBody) = await FetchAsync(HttpMethod.Head, "/");
        Assert.Equal(HttpStatusCode.OK, pageHead.StatusCode);
        Assert.Equal("text/html", pageHead.Content.Headers.ContentType?.MediaType);
        Assert.Equal(html.Length, pageHead.Content.Headers.ContentLength);
        Assert.Empty(pageBody);
    }

    [Fact]
    public async Task ACodeArtifactHonoursARangeAndKeepsItsAttachmentHeader()
    {
        string work = Path.Combine(_root, "work-run1");
        Directory.CreateDirectory(work);
        byte[] report = _clip[..20000];
        await File.WriteAllBytesAsync(Path.Combine(work, "report.pdf"), report);
        string url = _artifacts.Capture("run1", work, (id, rel, _) => CodeArtifactStore.UrlFor("/api/code/artifacts", id, rel), out _)
            .Single().Pointer;

        (HttpResponseMessage reply, byte[] body) = await FetchAsync(HttpMethod.Get, url, "bytes=19990-");
        AssertPartial(reply, body, 19990, 19999, report.Length, report[19990..]);
        Assert.Equal("attachment", reply.Content.Headers.ContentDisposition?.DispositionType);
        Assert.Equal("nosniff", Header(reply, "X-Content-Type-Options"));

        // Its route is an ordinary GET, as the desktop's MapGet endpoint is, so a HEAD
        // is refused there exactly as it is on the desktop.
        (HttpResponseMessage head, byte[] none) = await FetchAsync(HttpMethod.Head, url);
        Assert.Equal(HttpStatusCode.MethodNotAllowed, head.StatusCode);
        Assert.Equal("GET", Header(head, "Allow"));
        Assert.Empty(none);
    }

    // ---- HEAD everywhere else -------------------------------------------------------

    [Fact]
    public async Task AHeadForAnApiRouteIsRefusedWithoutRunningItsHandler()
    {
        // A GET handler can have side effects or open a stream, so a HEAD is never
        // allowed to run one it was not opened to (LoopbackServer.MapFiles).
        int runs = 0;
        using var server = new LoopbackServer(NullLogger.Instance) { RequireToken = false };
        server.MapGet("/api/count", (_, _) =>
        {
            Interlocked.Increment(ref runs);
            return Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { runs }));
        });
        server.MapPost("/api/count", (_, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { ok = true })));
        server.Start();
        using var client = new HttpClient { BaseAddress = new Uri(server.BaseUrl) };

        (HttpResponseMessage head, byte[] body) = await FetchAsync(HttpMethod.Head, "/api/count", client: client);
        Assert.Equal(HttpStatusCode.MethodNotAllowed, head.StatusCode);
        Assert.Equal("GET, POST", Header(head, "Allow"));
        Assert.Empty(body);
        Assert.Equal(0, Volatile.Read(ref runs));

        (HttpResponseMessage unknown, byte[] nothing) = await FetchAsync(HttpMethod.Head, "/api/nothing-here", client: client);
        Assert.Equal(HttpStatusCode.NotFound, unknown.StatusCode);
        Assert.Empty(nothing);

        // And GET is untouched.
        (HttpResponseMessage get, _) = await FetchAsync(HttpMethod.Get, "/api/count", client: client);
        Assert.Equal(HttpStatusCode.OK, get.StatusCode);
        Assert.Equal(1, Volatile.Read(ref runs));
    }

    [Fact]
    public async Task TheAppsOwnApiRoutesRefuseAHeadAndNameWhatTheyTake()
    {
        // The answer ASP.NET's endpoint routing gives the same HEAD on the desktop: 405,
        // and an Allow line in the same form ("GET, POST" for a path that takes both).
        (HttpResponseMessage models, byte[] none) = await FetchAsync(HttpMethod.Head, "/api/models");
        Assert.Equal(HttpStatusCode.MethodNotAllowed, models.StatusCode);
        Assert.Equal("GET", Header(models, "Allow"));
        Assert.Empty(none);

        (HttpResponseMessage chat, _) = await FetchAsync(HttpMethod.Head, "/api/chat");
        Assert.Equal(HttpStatusCode.MethodNotAllowed, chat.StatusCode);
        Assert.Equal("POST", Header(chat, "Allow"));
    }
}
