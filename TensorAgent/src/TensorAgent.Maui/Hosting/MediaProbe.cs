// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

#if IOS || MACCATALYST
using AVFoundation;
using CoreGraphics;
using CoreImage;
using Foundation;
using ImageIO;
using TensorSharp.Models.Media;
using TensorSharp.Models.Media.Apple;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;

namespace TensorAgent.Maui.Hosting;

/// <summary>One named check the media probe ran.</summary>
/// <param name="Name">Stable identifier; <c>scripts/verify-sim.sh</c> matches on these.</param>
/// <param name="Ok">Whether the check passed.</param>
/// <param name="Detail">What it measured, or the exception that stopped it.</param>
public sealed record MediaCheck(string Name, bool Ok, string Detail);

/// <param name="Providers"><see cref="MediaCodecs.Describe"/> — which provider owns each slot.</param>
/// <param name="AllPassed">Every check in <paramref name="Checks"/> passed.</param>
public sealed record MediaProbeResult(string Providers, bool AllPassed, IReadOnlyList<MediaCheck> Checks);

/// <summary>
/// Runs the iOS media provider against real files, on the device, at startup.
///
/// <para><b>Why this exists rather than a unit test:</b> <c>TensorSharp.Models</c>'s
/// <c>Media/Apple/</c> sources compile only for the <c>net10.0-ios</c> target framework, and
/// the repo's xunit suite is a net10.0 host that cannot load an iOS assembly. The desktop
/// suite pins the CONTRACT — <c>InferenceWeb.Tests/MediaProviderParityTests</c> runs the same
/// assertions against the managed and Magick.NET providers, and
/// <c>AppleProviderMathTests</c> covers the alpha and display-matrix arithmetic that was
/// deliberately kept platform-neutral for that purpose. What none of it can do is execute
/// ImageIO or AVFoundation. That is this file's job, and
/// <c>TensorAgent/scripts/verify-sim.sh</c> fails the simulator E2E run when any check here
/// reports false.</para>
///
/// <para>The two checks that justify the whole provider are <c>heic-decode</c> and
/// <c>exif-orientation</c>: HEIC is what the iPhone camera writes and the managed decoder
/// cannot read it at all, and a stored orientation ignored on decode silently rotates every
/// photo a user ever attaches. Both fixtures are produced here by ImageIO's own encoder rather
/// than checked in, so the bytes are what this device actually writes.</para>
/// </summary>
public static class MediaProbe
{
    private const int Width = 64;
    private const int Height = 48;

    /// <summary>Top-left, top-right, bottom-left, bottom-right of the fixture image.</summary>
    private static readonly (byte R, byte G, byte B)[] Quadrants =
    {
        (220, 40, 40), (40, 200, 80), (50, 90, 230), (240, 220, 60),
    };

    /// <summary>Per-channel slack for a lossy codec sampled in the middle of a flat block.</summary>
    private const int LossyTolerance = 12;

    public static MediaProbeResult Run()
    {
        var checks = new List<MediaCheck>();
        string workspace = Path.Combine(Path.GetTempPath(), "ts-media-probe-" + Guid.NewGuid().ToString("N"));

        try
        {
            Directory.CreateDirectory(workspace);
            checks.Add(Check("providers", CheckProviders));
            checks.Add(Check("png-roundtrip", CheckPngRoundTrip));
            checks.Add(Check("png-straight-alpha", CheckStraightAlpha));
            checks.Add(Check("png-transparent-colour", CheckTransparentColour));
            checks.Add(Check("heic-decode", CheckHeicDecode));
            checks.Add(Check("exif-orientation", CheckExifOrientation));
            checks.Add(Check("mp4-roundtrip", () => CheckMp4RoundTrip(workspace)));
            checks.Add(Check("mp4-soundtrack", () => CheckMp4Soundtrack(workspace)));
            checks.Add(Check("audio-decode", () => CheckAudioDecode(workspace)));
        }
        catch (Exception ex)
        {
            checks.Add(new MediaCheck("probe", false, $"{ex.GetType().Name}: {ex.Message}"));
        }
        finally
        {
            try { Directory.Delete(workspace, recursive: true); } catch { /* best effort */ }
        }

        return new MediaProbeResult(MediaCodecs.Describe(), checks.TrueForAll(c => c.Ok), checks);
    }

    private static MediaCheck Check(string name, Func<string> body)
    {
        try
        {
            return new MediaCheck(name, true, body());
        }
        catch (Exception ex)
        {
            return new MediaCheck(name, false, $"{ex.GetType().Name}: {ex.Message}");
        }
    }

    // ------------------------------------------------------------------
    // Checks
    // ------------------------------------------------------------------

    /// <summary>All four slots are the Apple provider. A slot left on the managed default is
    /// the failure mode this catches: it works for PNG and WAV and then throws the first time
    /// a user attaches anything a phone actually produces.</summary>
    private static string CheckProviders()
    {
        Require(ReferenceEquals(MediaCodecs.Image, AppleMediaProvider.Instance), "MediaCodecs.Image is not the Apple provider");
        Require(ReferenceEquals(MediaCodecs.Video, AppleMediaProvider.Instance), "MediaCodecs.Video is not the Apple provider");
        Require(ReferenceEquals(MediaCodecs.VideoEncoder, AppleMediaProvider.Instance), "MediaCodecs.VideoEncoder is not the Apple provider");
        Require(ReferenceEquals(MediaCodecs.Audio, AppleMediaProvider.Instance), "MediaCodecs.Audio is not the Apple provider");
        return MediaCodecs.Describe();
    }

    private static string CheckPngRoundTrip()
    {
        byte[] rgba = QuadrantRgba();
        byte[] png = MediaCodecs.Image.EncodePng(rgba, Width, Height, 4);
        Require(MediaCodecs.Image.CanDecode(png), "CanDecode said no to a PNG the provider wrote");

        byte[] back = MediaCodecs.Image.DecodeRgba(png, out int w, out int h);
        Require(w == Width && h == Height, $"decoded {w}x{h}, expected {Width}x{Height}");
        Require(back.Length == rgba.Length, $"decoded {back.Length} bytes, expected {rgba.Length}");
        for (int i = 0; i < rgba.Length; i++)
            Require(back[i] == rgba[i], $"PNG is lossless but byte {i} came back {back[i]}, not {rgba[i]}");

        (int dw, int dh) = MediaCodecs.Image.ReadDimensions(png);
        Require(dw == Width && dh == Height, $"ReadDimensions said {dw}x{dh}");
        return $"{Width}x{Height} exact";
    }

    /// <summary>Alpha comes back straight, not premultiplied. Core Graphics has no
    /// straight-alpha RGBA layout, so the provider recovers it arithmetically; a regression
    /// here darkens every semi-transparent image a model is shown, with nothing in the log.</summary>
    private static string CheckStraightAlpha()
    {
        byte[] alphas = { 255, 192, 128, 64 };
        var rgba = new byte[alphas.Length * 4];
        for (int i = 0; i < alphas.Length; i++)
        {
            rgba[i * 4] = 200; rgba[i * 4 + 1] = 128; rgba[i * 4 + 2] = 64; rgba[i * 4 + 3] = alphas[i];
        }

        byte[] png = MediaCodecs.Image.EncodePng(rgba, alphas.Length, 1, 4);
        byte[] back = MediaCodecs.Image.DecodeRgba(png, out _, out _);

        int worst = 0;
        for (int i = 0; i < alphas.Length; i++)
        {
            Require(back[i * 4 + 3] == alphas[i], $"alpha {back[i * 4 + 3]} != {alphas[i]}");
            for (int c = 0; c < 3; c++)
                worst = Math.Max(worst, Math.Abs(back[i * 4 + c] - rgba[i * 4 + c]));
        }
        Require(worst <= 3, $"straight-alpha recovery is off by {worst} levels (premultiplied output would be off by ~100)");
        return $"worst channel error {worst} at alphas 255/192/128/64";
    }

    /// <summary>
    /// Decode HEIC — the format the camera writes, and the reason a platform image codec
    /// exists at all. The fixture is encoded here by ImageIO, so the bytes are what this
    /// device produces rather than a checked-in blob from another machine.
    /// </summary>
    private static string CheckHeicDecode()
    {
        byte[]? heic = TryEncode("public.heic");
        if (heic == null)
            throw new InvalidOperationException("ImageIO has no HEIC encoder here, so the decode path could not be exercised");

        Require(MediaCodecs.Image.CanDecode(heic), "CanDecode said no to a HEIC file");
        (int dw, int dh) = MediaCodecs.Image.ReadDimensions(heic);
        Require(dw == Width && dh == Height, $"ReadDimensions said {dw}x{dh}");

        byte[] pixels = MediaCodecs.Image.DecodeRgba(heic, out int w, out int h);
        Require(w == Width && h == Height, $"decoded {w}x{h}");
        int worst = AssertQuadrants(pixels, w, h, orientation: 1);
        return $"{heic.Length} bytes, {w}x{h}, worst channel error {worst}";
    }

    /// <summary>
    /// A stored orientation is applied on decode.
    ///
    /// <para>The fixture is a JPEG written by ImageIO with an APP1 Exif segment spliced in
    /// carrying orientation 6 — exactly the shape the camera produces for a portrait photo. If
    /// the provider ignores it the image comes back landscape and every photo the user sends
    /// is rotated a quarter turn, which looks like the model being bad at images.</para>
    /// </summary>
    private static string CheckExifOrientation()
    {
        byte[]? jpeg = TryEncode("public.jpeg");
        if (jpeg == null)
            throw new InvalidOperationException("ImageIO has no JPEG encoder here");

        byte[] tagged = InjectJpegOrientation(jpeg, orientation: 6);

        // ReadDimensions reports the STORED geometry, orientation not applied.
        (int dw, int dh) = MediaCodecs.Image.ReadDimensions(tagged);
        Require(dw == Width && dh == Height, $"ReadDimensions said {dw}x{dh}, expected the stored {Width}x{Height}");

        byte[] pixels = MediaCodecs.Image.DecodeRgba(tagged, out int w, out int h);
        Require(w == Height && h == Width, $"decoded {w}x{h}; orientation 6 must transpose {Width}x{Height} to {Height}x{Width}");
        int worst = AssertQuadrants(pixels, w, h, orientation: 6);
        return $"{Width}x{Height} stored, {w}x{h} upright, worst channel error {worst}";
    }

    /// <summary>
    /// Encode an MP4 and read its frames back.
    ///
    /// <para>Frames carry their index as the position of a bright block, so an off-by-one in
    /// either access path is unambiguous rather than hidden inside a colour tolerance. Both
    /// paths are exercised: a dense request (stepped) and a sparse one with a repeat (sought,
    /// and the repeat is how a slow clip is held onto a faster timeline).</para>
    /// </summary>
    private static string CheckMp4RoundTrip(string workspace)
    {
        const int count = 20, fps = 10;
        string path = Path.Combine(workspace, "probe.mp4");

        var frames = new RgbImage[count];
        for (int i = 0; i < count; i++)
            frames[i] = MarkerFrame(i);

        string codec = MediaCodecs.VideoEncoder.SaveMp4(path, frames, fps);
        Require(codec == "h264", $"encoder reported '{codec}', expected h264");
        Require(File.Exists(path) && new FileInfo(path).Length > 0, "the encoder wrote nothing");

        VideoInfo info = MediaCodecs.Video.Probe(path);
        Require(info.Width == Width && info.Height == Height, $"probe reported {info.Width}x{info.Height}");
        Require(Math.Abs(info.Fps - fps) <= 0.5, $"probe reported {info.Fps} fps");
        Require(Math.Abs(info.FrameCount - count) <= 2, $"probe reported {info.FrameCount} frames");

        ReadAndVerify(path, new[] { 0, 1, 2, 3 });
        ReadAndVerify(path, new[] { 0, 15, 15, 19 });
        return $"{new FileInfo(path).Length} bytes, {info.FrameCount} frames at {info.Fps:0.##} fps, stepped and sought";
    }

    /// <summary>
    /// Write a clip WITH a soundtrack, the way a video model's output is saved, and read it
    /// back through AVFoundation: one H.264 track and one AAC track at the soundtrack's own
    /// rate and channel count, lasting as long as the frames, with the index ahead of the
    /// media so a web view can play it while it loads, and a sound that decodes to
    /// something audible rather than to silence.
    /// </summary>
    private static string CheckMp4Soundtrack(string workspace)
    {
        const int count = 24, fps = 24, rate = 32000;
        string path = Path.Combine(workspace, "probe-av.mp4");

        var frames = new RgbImage[count];
        for (int i = 0; i < count; i++)
            frames[i] = MarkerFrame(i % 20);
        var left = new float[rate];
        var right = new float[rate];
        for (int i = 0; i < rate; i++)
        {
            left[i] = (float)Math.Sin(2 * Math.PI * 440 * i / rate) * 0.5f;
            right[i] = (float)Math.Sin(2 * Math.PI * 660 * i / rate) * 0.25f;
        }
        var audio = new GeneratedVideoAudio { Channels = new[] { left, right }, SampleRate = rate };

        string codec = MediaCodecs.VideoEncoder.SaveMp4(path, frames, fps, audio, out bool muxed);
        Require(codec == "h264", $"encoder reported '{codec}', expected h264");
        Require(muxed, "the encoder wrote the frames without the soundtrack");

        byte[] head = File.ReadAllBytes(path);
        int moov = TopLevelBox(head, "moov"), mdat = TopLevelBox(head, "mdat");
        Require(moov >= 0 && mdat >= 0 && moov < mdat, $"the index is not ahead of the media (moov at {moov}, mdat at {mdat})");

        using NSUrl url = NSUrl.FromFilename(path);
        using var asset = new AVUrlAsset(url, new AVUrlAssetOptions { PreferPreciseDurationAndTiming = true });
        AVAssetTrack[] video = asset.GetTracks(AVMediaTypes.Video) ?? Array.Empty<AVAssetTrack>();
        AVAssetTrack[] sound = asset.GetTracks(AVMediaTypes.Audio) ?? Array.Empty<AVAssetTrack>();
        Require(video.Length == 1 && sound.Length == 1, $"{video.Length} video and {sound.Length} audio tracks, expected one of each");
        AudioToolbox.AudioStreamBasicDescription? format = sound[0].FormatDescriptions.FirstOrDefault()?.AudioStreamBasicDescription;
        Require(format is { } f && f.Format == AudioToolbox.AudioFormatType.MPEG4AAC,
            $"the soundtrack is {format?.Format.ToString() ?? "unreadable"}, not AAC");
        Require(format!.Value.SampleRate == rate && format.Value.ChannelsPerFrame == 2,
            $"the soundtrack is {format.Value.SampleRate} Hz x{format.Value.ChannelsPerFrame}, not {rate} Hz stereo");
        double seconds = asset.Duration.Seconds;
        Require(Math.Abs(seconds - 1.0) <= 0.05, $"the clip lasts {seconds:0.###} s, not 1 s");

        DecodedAudio decoded = MediaCodecs.Audio.Decode(path);
        int half = decoded.SampleCount / 2;
        double rms = Rms(decoded.Channels[0], 0, decoded.SampleCount);
        // The left tone's RMS is 0.5/sqrt(2) = 0.354; AAC keeps a pure tone close to it.
        Require(rms > 0.25 && rms < 0.45,
            $"the decoded left channel's RMS is {rms:0.###}, not the tone's 0.354 ({decoded.SampleCount} samples x"
            + $"{decoded.ChannelCount} at {decoded.SampleRate} Hz; first half {Rms(decoded.Channels[0], 0, half):0.###}, "
            + $"second half {Rms(decoded.Channels[0], half, decoded.SampleCount):0.###}; right {Rms(decoded.Channels[^1], 0, decoded.SampleCount):0.###})");
        return $"{head.Length} bytes, h264 + AAC {format.Value.SampleRate:0} Hz x{format.Value.ChannelsPerFrame}, " +
               $"{seconds:0.###} s, moov before mdat, decoded RMS {rms:0.###}";
    }

    private static double Rms(float[] samples, int from, int until)
    {
        double sum = 0;
        for (int i = from; i < until; i++)
            sum += samples[i] * (double)samples[i];
        return Math.Sqrt(sum / Math.Max(1, until - from));
    }

    /// <summary>
    /// A transparent pixel keeps the colour it was stored with. The pipelines drop alpha and
    /// take the colour as stored, and a premultiplied decode has none to give back where
    /// alpha is 0 and a few levels where it is low: a picture with a mostly transparent
    /// background reached MiniMax-H3 as black blotches. Exact, because a PNG with
    /// transparency is read as stored rather than drawn.
    /// </summary>
    private static string CheckTransparentColour()
    {
        byte[] alphas = { 0, 1, 8, 32 };
        var rgba = new byte[alphas.Length * 4];
        for (int i = 0; i < alphas.Length; i++)
        {
            rgba[i * 4] = 40; rgba[i * 4 + 1] = 120; rgba[i * 4 + 2] = 230; rgba[i * 4 + 3] = alphas[i];
        }

        byte[] png = MediaCodecs.Image.EncodePng(rgba, alphas.Length, 1, 4);
        byte[] back = MediaCodecs.Image.DecodeRgba(png, out _, out _);
        for (int i = 0; i < back.Length; i++)
            Require(back[i] == rgba[i], $"pixel {i / 4} channel {i % 4} came back {back[i]}, stored {rgba[i]} (alpha {alphas[i / 4]})");
        return "colour exact at alphas 0/1/8/32";
    }

    /// <summary>Offset of a top-level MP4 box, or -1.</summary>
    private static int TopLevelBox(byte[] file, string type)
    {
        long at = 0;
        while (at + 8 <= file.Length)
        {
            long size = ((long)file[at] << 24) | ((long)file[at + 1] << 16) | ((long)file[at + 2] << 8) | file[at + 3];
            string name = System.Text.Encoding.ASCII.GetString(file, (int)at + 4, 4);
            if (name == type)
                return (int)at;
            if (size == 1 && at + 16 <= file.Length)
            {
                size = 0;
                for (int i = 8; i < 16; i++)
                    size = (size << 8) | file[at + i];
            }
            if (size < 8)
                return -1;
            at += size;
        }
        return -1;
    }

    private static void ReadAndVerify(string path, int[] requested)
    {
        var seen = new List<int>();
        MediaCodecs.Video.ReadFrames(path, requested, (index, pixels, w, h, stride, layout) =>
        {
            Require(w == Width && h == Height, $"frame {index} came back {w}x{h}");
            int bytesPerPixel = layout is PixelLayout.Bgra or PixelLayout.Rgba ? 4 : 3;
            Require(stride >= w * bytesPerPixel, $"frame {index} stride {stride} is shorter than a row");
            int marker = ReadMarker(pixels, w, h, stride, bytesPerPixel);
            Require(marker == index, $"asked for frame {index}, got the frame carrying marker {marker}");
            seen.Add(index);
        });
        Require(seen.Count == requested.Length, $"asked for {requested.Length} frames, received {seen.Count}");
    }

    /// <summary>Decode audio at the file's own rate and channel count. WAV is the format both
    /// the managed and the AVFoundation decoders read, which is what makes it comparable — and
    /// on this provider it is AVAudioFile doing the reading.</summary>
    private static string CheckAudioDecode(string workspace)
    {
        const int rate = 22050, frames = 2205;
        var left = new float[frames];
        var right = new float[frames];
        for (int i = 0; i < frames; i++)
        {
            left[i] = (float)Math.Sin(2 * Math.PI * 440 * i / rate) * 0.5f;
            right[i] = (float)Math.Sin(2 * Math.PI * 660 * i / rate) * 0.25f;
        }

        string wav = Path.Combine(workspace, "tone.wav");
        WavWriter.Write(wav, new[] { left, right }, rate);

        DecodedAudio decoded = MediaCodecs.Audio.Decode(wav);
        Require(decoded.SampleRate == rate, $"decoded at {decoded.SampleRate} Hz, not the file's {rate}");
        Require(decoded.ChannelCount == 2, $"decoded {decoded.ChannelCount} channels, not the file's 2");
        Require(decoded.SampleCount == frames, $"decoded {decoded.SampleCount} frames, not {frames}");

        float worst = 0;
        for (int i = 0; i < frames; i++)
        {
            worst = Math.Max(worst, Math.Abs(decoded.Channels[0][i] - left[i]));
            worst = Math.Max(worst, Math.Abs(decoded.Channels[1][i] - right[i]));
        }
        // WavWriter stores 16-bit PCM, so the round trip is exact to within one LSB.
        Require(worst < 2e-4f, $"samples differ by {worst}, more than one 16-bit step");
        return $"{decoded.SampleCount} frames, {decoded.ChannelCount} ch at {decoded.SampleRate} Hz, worst sample error {worst:e2}";
    }

    // ------------------------------------------------------------------
    // Fixtures
    // ------------------------------------------------------------------

    /// <summary>A 64x48 RGBA image of four flat quadrants: flat blocks survive a lossy codec,
    /// and four different ones make a wrong channel order or an upside-down blit obvious.</summary>
    private static byte[] QuadrantRgba()
    {
        var rgba = new byte[Width * Height * 4];
        for (int y = 0; y < Height; y++)
            for (int x = 0; x < Width; x++)
            {
                (byte r, byte g, byte b) = Quadrants[QuadrantOf(x, y, Width, Height)];
                int p = (y * Width + x) * 4;
                rgba[p] = r; rgba[p + 1] = g; rgba[p + 2] = b; rgba[p + 3] = 255;
            }
        return rgba;
    }

    private static int QuadrantOf(int x, int y, int w, int h) => (y < h / 2 ? 0 : 2) + (x < w / 2 ? 0 : 1);

    /// <summary>Encode the fixture with ImageIO, or null when this device has no encoder for
    /// the type (which is a reportable outcome, not a crash).</summary>
    private static byte[]? TryEncode(string typeIdentifier)
    {
        byte[] rgba = QuadrantRgba();
        using var provider = new CGDataProvider(rgba);
        using CGColorSpace colorSpace = CGColorSpace.CreateSrgb()
            ?? throw new InvalidOperationException("no sRGB colour space");
        using var image = new CGImage(
            Width, Height, 8, 32, Width * 4, colorSpace, CGImageAlphaInfo.NoneSkipLast,
            provider, null, false, CGColorRenderingIntent.Default);

        using var data = new NSMutableData();
        using CGImageDestination? destination = CGImageDestination.Create(
            data, typeIdentifier, 1, new CGImageDestinationOptions { LossyCompressionQuality = 1f });
        if (destination == null)
            return null;

        destination.AddImage(image, new CGImageDestinationOptions { LossyCompressionQuality = 1f });
        return destination.Close() ? data.ToArray() : null;
    }

    /// <summary>Splice an APP1 Exif segment carrying just IFD0 { Orientation } straight after
    /// the JPEG's SOI marker. Byte surgery rather than re-encoding with properties, so the
    /// image data is untouched and the tag is the only difference.</summary>
    private static byte[] InjectJpegOrientation(byte[] jpeg, int orientation)
    {
        byte[] tiff =
        {
            (byte)'I', (byte)'I', 0x2A, 0x00,           // little-endian, 42
            0x08, 0x00, 0x00, 0x00,                     // IFD0 at offset 8
            0x01, 0x00,                                 // one entry
            0x12, 0x01, 0x03, 0x00,                     // tag 0x0112 (Orientation), type SHORT
            0x01, 0x00, 0x00, 0x00,                     // count 1
            (byte)orientation, 0x00, 0x00, 0x00,        // value, left-justified in four bytes
            0x00, 0x00, 0x00, 0x00,                     // no next IFD
        };

        byte[] payload = new byte[6 + tiff.Length];
        "Exif\0\0"u8.CopyTo(payload);
        tiff.CopyTo(payload, 6);

        int length = payload.Length + 2;
        var outp = new List<byte>(jpeg.Length + length + 4);
        outp.AddRange(jpeg.Take(2));                    // SOI
        outp.Add(0xFF);
        outp.Add(0xE1);                                 // APP1
        outp.Add((byte)(length >> 8));
        outp.Add((byte)length);
        outp.AddRange(payload);
        outp.AddRange(jpeg.Skip(2));
        return outp.ToArray();
    }

    /// <summary>Frame <paramref name="index"/> of the synthetic clip: a mid-grey field with one
    /// bright 8x8 block whose cell position spells the index.</summary>
    private static RgbImage MarkerFrame(int index)
    {
        const int cell = 8;
        int columns = Width / cell;
        int bx = (index % columns) * cell;
        int by = (index / columns) * cell;

        var pixels = new float[Width * Height * 3];
        for (int y = 0; y < Height; y++)
            for (int x = 0; x < Width; x++)
            {
                bool marked = x >= bx && x < bx + cell && y >= by && y < by + cell;
                float v = marked ? 0.92f : 0.24f;
                int p = (y * Width + x) * 3;
                pixels[p] = v; pixels[p + 1] = v; pixels[p + 2] = v;
            }
        return new RgbImage(Width, Height, pixels);
    }

    /// <summary>The index a decoded frame carries, found as the brightest 8x8 cell.</summary>
    private static int ReadMarker(byte[] pixels, int width, int height, int stride, int bytesPerPixel)
    {
        const int cell = 8;
        int columns = width / cell;
        int rows = height / cell;

        int best = -1;
        long bestSum = -1;
        for (int cy = 0; cy < rows; cy++)
            for (int cx = 0; cx < columns; cx++)
            {
                long sum = 0;
                for (int y = cy * cell; y < (cy + 1) * cell; y++)
                    for (int x = cx * cell; x < (cx + 1) * cell; x++)
                    {
                        int p = y * stride + x * bytesPerPixel;
                        sum += pixels[p] + pixels[p + 1] + pixels[p + 2];
                    }
                if (sum > bestSum) { bestSum = sum; best = cy * columns + cx; }
            }

        Require(bestSum > 3L * cell * cell * 160, "no bright marker cell in the decoded frame");
        return best;
    }

    /// <summary>Check the four quadrant colours in a decoded image, following
    /// <paramref name="orientation"/> back to the source pixel each output sample came from.
    /// Returns the worst per-channel difference.</summary>
    private static int AssertQuadrants(byte[] pixels, int w, int h, int orientation)
    {
        int worst = 0;
        for (int q = 0; q < 4; q++)
        {
            // The middle of each output quadrant, away from the block edges where a lossy
            // codec spends its error budget.
            int x = (q % 2) * (w / 2) + w / 4;
            int y = (q / 2) * (h / 2) + h / 4;

            // EXIF 6 means the stored pixels are rotated 90 degrees clockwise to display, so
            // output (x,y) came from source (y, storedHeight - 1 - x).
            (int sx, int sy) = orientation == 6 ? (y, Height - 1 - x) : (x, y);
            (byte r, byte g, byte b) = Quadrants[QuadrantOf(sx, sy, Width, Height)];

            int p = (y * w + x) * 4;
            worst = Math.Max(worst, Math.Abs(pixels[p] - r));
            worst = Math.Max(worst, Math.Abs(pixels[p + 1] - g));
            worst = Math.Max(worst, Math.Abs(pixels[p + 2] - b));
            Require(
                Math.Abs(pixels[p] - r) <= LossyTolerance &&
                Math.Abs(pixels[p + 1] - g) <= LossyTolerance &&
                Math.Abs(pixels[p + 2] - b) <= LossyTolerance,
                $"output ({x},{y}) is ({pixels[p]},{pixels[p + 1]},{pixels[p + 2]}), expected source ({sx},{sy}) = ({r},{g},{b})");
            Require(pixels[p + 3] == 255, $"output ({x},{y}) has alpha {pixels[p + 3]}, expected an opaque image");
        }
        return worst;
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
            throw new InvalidOperationException(message);
    }
}
#endif
