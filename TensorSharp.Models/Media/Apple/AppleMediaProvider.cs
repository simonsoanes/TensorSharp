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
using System;
using System.IO;
using System.Runtime.CompilerServices;
using CoreGraphics;
using CoreImage;
using Foundation;
using ImageIO;

#nullable enable

namespace TensorSharp.Models.Media.Apple;

/// <summary>
/// Puts the Apple stack on <see cref="MediaCodecs"/> before any user code runs, the way
/// <c>Media/Desktop/DesktopMediaRegistration</c> does for OpenCV and Magick.NET.
///
/// <para>A module initializer rather than a call each host has to remember: the app, the
/// loopback API and any on-device test all decode media, and one of them forgetting the call
/// would surface as the managed provider's "HEIC needs a platform image provider" the first
/// time someone attaches a photo from Photos. The host still calls
/// <see cref="AppleMediaProvider.Register"/> explicitly at startup (see
/// <c>TensorAgent.Maui.MauiProgram</c>) so the wiring is visible where the app is assembled;
/// the call is idempotent.</para>
/// </summary>
internal static class AppleMediaRegistration
{
    // CA2255 warns that module initializers belong in application code. Same documented
    // exception as the desktop registration: a library wiring its own default provider
    // before any of its callers can observe the unwired state.
#pragma warning disable CA2255
    [ModuleInitializer]
    internal static void Register() => AppleMediaProvider.Register();
#pragma warning restore CA2255
}

/// <summary>
/// The media stack for iPhone and iPad: ImageIO for stills, AVFoundation for video frames,
/// MP4 export and audio.
///
/// <para>It exists because neither of the desktop libraries can run here — Magick.NET and
/// OpenCvSharp ship no ios-arm64 binaries and are dynamic libraries an app could not link
/// anyway — and because the managed fallback cannot read the two formats a phone produces by
/// default: HEIC stills and H.264/HEVC video. <see cref="ManagedMediaProvider"/> stays
/// underneath for the pieces that need no native code (PNG encode, both resize filters), which
/// is not laziness: those are the pieces whose exact numbers the generative pipelines were
/// validated against, and re-deriving them from a platform API would change them.</para>
///
/// <para><b>Where this cannot match the desktop provider exactly</b>, both places documented
/// at the code:</para>
/// <list type="bullet">
/// <item><description>An image <i>with</i> an alpha channel comes back through a premultiplied
/// Core Graphics context and is un-premultiplied by <see cref="StraightAlpha"/>, because Core
/// Graphics has no straight-alpha 8-bit RGBA bitmap layout at all - except a PNG without an
/// embedded colour profile, which the managed PNG codec reads exactly as stored. Opaque
/// images — every JPEG, every HEIC from the camera, every video frame — are exact.</description></item>
/// <item><description>MP4 export goes through VideoToolbox at a chosen bit rate rather than
/// ffmpeg at <c>-crf 17</c>; see <c>AppleMediaProvider.Mp4.cs</c>.</description></item>
/// </list>
/// </summary>
public sealed partial class AppleMediaProvider : IImageCodec, IVideoDecoder, IVideoEncoder, IAudioDecoder
{
    /// <summary>The instance the module initializer registers. Stateless.</summary>
    public static AppleMediaProvider Instance { get; } = new AppleMediaProvider();

    /// <summary>Register this provider for images, video decode, MP4 encode and audio decode.
    /// Idempotent; a host that swapped in something else can call it to get the Apple stack
    /// back. Unlike the desktop registration this does claim
    /// <see cref="MediaCodecs.Audio"/> — AVFoundation is exactly what makes .m4a / .aac /
    /// .caf / .flac (Voice Memos, and everything the Files app hands over) readable, and the
    /// managed WAV/MP3/OGG decoders still take those formats first.</summary>
    public static void Register()
    {
        MediaCodecs.Image = Instance;
        MediaCodecs.Video = Instance;
        MediaCodecs.VideoEncoder = Instance;
        MediaCodecs.Audio = Instance;
    }

    // ------------------------------------------------------------------
    // IImageCodec — ImageIO / Core Graphics
    // ------------------------------------------------------------------

    /// <summary>ImageIO decodes more than the sniffer names (RAW, ICNS, ...), but the contract
    /// is "false means DecodeRgba throws", so the answer is tied to what is sniffed and
    /// <see cref="DecodeRgba"/> refuses the unrecognised rather than guessing — the same
    /// bargain the desktop provider strikes with ImageMagick.</summary>
    public bool CanDecode(ReadOnlySpan<byte> header) =>
        ImageFormatSniffer.Detect(header) != ImageFormat.Unknown;

    /// <summary>
    /// Decode to straight (not premultiplied) RGBA8 with the stored orientation applied.
    ///
    /// <para>Orientation is read from <c>kCGImagePropertyOrientation</c> and applied by
    /// <see cref="ExifOrientation.Apply"/> rather than by drawing through a rotated CTM. That
    /// is the deliberate choice: the managed and desktop providers both produce their upright
    /// pixels with that exact transform, so sharing it makes the three providers agree to the
    /// byte, while a rotated <c>CGContext</c> would put a resampler in the path of what has to
    /// be a pure permutation. It matters — the common case is a photo straight out of the
    /// iPhone camera, which is stored on its side with an orientation tag, and getting this
    /// wrong rotates every photo a user ever sends without any visible error.</para>
    /// </summary>
    public byte[] DecodeRgba(byte[] file, out int width, out int height)
    {
        if (file == null) throw new ArgumentNullException(nameof(file));

        ImageFormat format = ImageFormatSniffer.Detect(file);
        if (format == ImageFormat.Unknown)
        {
            throw new NotSupportedException(
                "Unrecognised image format: the Apple image codec decodes PNG, JPEG, HEIC/HEIF, GIF, BMP, " +
                "WebP, PSD and TIFF through ImageIO, identified by their magic bytes.");
        }

        // A PNG with transparency is read by the managed codec, as stored. Core Graphics can
        // only draw it premultiplied, and 8-bit premultiplied keeps alpha/255 of a pixel's
        // colour: none at all where alpha is 0, a few levels where it is low. The pipelines
        // drop alpha and take the colour as stored (ImageIO.Decode), so a picture whose
        // background is mostly transparent reached MiniMax-H3 as black blotches - MEASURED on
        // a 1253x836 PNG whose alpha averages 0.43: the clip's background came out burnt
        // where the desktop's, from the same file, kept its starfield. A PNG that embeds a
        // colour profile still goes through Core Graphics, which converts it to sRGB.
        if (format == ImageFormat.Png && PngCodec.CarriesAlpha(file) && !PngCodec.HasChunkBeforeImageData(file, "iCCP"))
        {
            byte[] stored = PngCodec.Decode(file, out width, out height);
            return ExifOrientation.Apply(stored, ref width, ref height, PngCodec.ReadExifOrientation(file));
        }

        using NSData data = NSData.FromArray(file);
        using CGImageSource? source = CGImageSource.FromData(data);
        if (source == null || source.ImageCount == IntPtr.Zero)
        {
            throw new InvalidDataException(
                $"ImageIO could not open the {format} data ({file.Length} bytes); the file is truncated or corrupt.");
        }

        int orientation = ReadOrientation(source);
        using CGImage? image = source.CreateImage(0, new CGImageOptions { ShouldCache = false });
        if (image == null)
            throw new InvalidDataException($"ImageIO opened the {format} container but could not decode its first image.");

        byte[] rgba = DrawStraightRgba(image, format, out int storedWidth, out int storedHeight);
        width = storedWidth;
        height = storedHeight;
        return ExifOrientation.Apply(rgba, ref width, ref height, orientation);
    }

    /// <summary>The dimensions as STORED, from <c>CGImageSourceCopyPropertiesAtIndex</c> — the
    /// header only, no pixels decoded, and no orientation applied (a caller estimating vision
    /// tokens wants the same number the desktop provider's <c>MagickImageInfo</c> gives).</summary>
    public (int width, int height) ReadDimensions(byte[] file)
    {
        if (file == null) throw new ArgumentNullException(nameof(file));

        ImageFormat format = ImageFormatSniffer.Detect(file);
        if (format == ImageFormat.Unknown)
        {
            throw new NotSupportedException(
                "Unrecognised image format: the Apple image codec reads PNG, JPEG, HEIC/HEIF, GIF, BMP, " +
                "WebP, PSD and TIFF through ImageIO, identified by their magic bytes.");
        }

        using NSData data = NSData.FromArray(file);
        using CGImageSource? source = CGImageSource.FromData(data);
        if (source == null || source.ImageCount == IntPtr.Zero)
            throw new InvalidDataException($"ImageIO could not open the {format} data ({file.Length} bytes).");

        CoreGraphics.CGImageProperties? properties = source.GetProperties(0, new CGImageOptions { ShouldCache = false });
        int? width = properties?.PixelWidth;
        int? height = properties?.PixelHeight;
        if (width is not > 0 || height is not > 0)
            throw new InvalidDataException($"ImageIO reported no pixel dimensions for the {format} data.");

        return (width.Value, height.Value);
    }

    /// <summary>Straight to the managed encoder. Nothing about a PNG file needs a platform
    /// library, and this is the encoder every extracted video frame already goes through, so
    /// a phone and a server write byte-identical files.</summary>
    public byte[] EncodePng(byte[] rgb8, int width, int height, int channels) =>
        ManagedMediaProvider.Instance.EncodePng(rgb8, width, height, channels);

    /// <summary>
    /// Straight to the managed resampler, for both filters.
    ///
    /// <para>Bilinear is managed on every provider by design (it is the cheap filter the chat
    /// vision processors define for themselves). Lanczos stays managed for a sharper reason:
    /// the generative pipelines were validated against ImageMagick's Lanczos-3, the managed
    /// path is a faithful Pillow Lanczos-3 port that agrees with it to within a few levels, and
    /// nothing Apple ships is that filter. <c>CGContext</c> at
    /// <c>kCGInterpolationHigh</c> is an unspecified quality tier, and vImage's scaler is its
    /// own kernel; either would silently move every resized pixel a model sees. Replacing this
    /// needs a measurement against the MiniMax-H3 parity fixtures first, which is what
    /// <see cref="MediaCodecs"/> asks for.</para>
    /// </summary>
    public byte[] ResizeRgb8(byte[] rgb8, int srcW, int srcH, int dstW, int dstH, ResizeFilter filter) =>
        ManagedMediaProvider.Instance.ResizeRgb8(rgb8, srcW, srcH, dstW, dstH, filter);

    /// <summary>The stored orientation as an EXIF value in 1..8; 1 when the file does not say.
    /// <c>CIImageOrientation</c> is the EXIF numbering (<c>TopLeft</c> = 1 ... <c>LeftBottom</c>
    /// = 8), and ImageIO synthesises it for HEIF's <c>irot</c> box too, so one code path covers
    /// JPEG's EXIF tag, PNG's <c>eXIf</c> chunk and HEIC alike.</summary>
    private static int ReadOrientation(CGImageSource source)
    {
        CoreGraphics.CGImageProperties? properties = source.GetProperties(0, new CGImageOptions { ShouldCache = false });
        CIImageOrientation? orientation = properties?.Orientation;
        if (orientation == null)
            return 1;

        int value = (int)orientation.Value;
        return value is >= 1 and <= 8 ? value : 1;
    }

    /// <summary>
    /// Draw a decoded image into a known byte order and return straight RGBA8, unpadded.
    ///
    /// <para>An opaque source is drawn with <c>kCGImageAlphaNoneSkipLast</c>, which touches
    /// only the three colour bytes and is exact; its alpha plane is then filled with 255 so
    /// callers always get a four-channel buffer (the desktop provider's
    /// <c>Alpha(AlphaOption.Set)</c>). A source that really has alpha has to go through
    /// <c>kCGImageAlphaPremultipliedLast</c> — Core Graphics has no straight-alpha 8-bit RGBA
    /// bitmap layout — and is un-premultiplied afterwards, which is lossy in the low bits of
    /// semi-transparent pixels and is the one place this provider cannot match Magick.NET.</para>
    ///
    /// <para>The destination context is sRGB, matching the desktop provider's explicit
    /// conversion, so a Display P3 photo from a recent iPhone is brought into the same space
    /// the pipelines were validated in rather than being read as if its wide-gamut numbers were
    /// sRGB.</para>
    /// </summary>
    private static byte[] DrawStraightRgba(CGImage image, ImageFormat format, out int width, out int height)
    {
        width = (int)image.Width;
        height = (int)image.Height;
        if (width <= 0 || height <= 0)
            throw new InvalidDataException($"ImageIO decoded a {width}x{height} {format} image.");

        long bytes = (long)width * height * 4;
        if (bytes > int.MaxValue)
            throw new NotSupportedException($"a {width}x{height} image exceeds the 2 GB buffer limit");

        bool hasAlpha = image.AlphaInfo is not (CGImageAlphaInfo.None or CGImageAlphaInfo.NoneSkipLast or CGImageAlphaInfo.NoneSkipFirst);
        byte[] rgba = new byte[bytes];

        // Core Graphics owns the backing store (the null-data form of CGBitmapContextCreate).
        // Handing it a managed array instead would leave the collector free to move a buffer
        // the context still holds a raw pointer to, and the bug that produces is a rare
        // corrupted decode rather than a crash anyone could trace.
        using (CGColorSpace colorSpace = CGColorSpace.CreateSrgb()
                   ?? throw new InvalidOperationException("Core Graphics could not create the sRGB colour space."))
        using (var context = new CGBitmapContext(
                   IntPtr.Zero,
                   (nint)width,
                   (nint)height,
                   (nint)8,
                   (nint)(width * 4),
                   colorSpace,
                   hasAlpha ? CGImageAlphaInfo.PremultipliedLast : CGImageAlphaInfo.NoneSkipLast))
        {
            // A 1:1 blit must be a copy, not a resample: no antialiasing, no interpolation.
            context.SetShouldAntialias(false);
            context.InterpolationQuality = CGInterpolationQuality.None;
            context.DrawImage(new CGRect(0, 0, width, height), image);

            IntPtr pixels = context.Data;
            if (pixels == IntPtr.Zero)
                throw new InvalidDataException($"Core Graphics allocated no backing store for a {width}x{height} bitmap context.");

            // CGBitmapContextCreate honours the requested bytesPerRow, but read it back rather
            // than assume: a padded row copied as if it were tight shears the whole image.
            int stride = (int)context.BytesPerRow;
            if (stride == width * 4)
            {
                System.Runtime.InteropServices.Marshal.Copy(pixels, rgba, 0, (int)bytes);
            }
            else
            {
                for (int y = 0; y < height; y++)
                {
                    System.Runtime.InteropServices.Marshal.Copy(
                        pixels + y * stride, rgba, y * width * 4, width * 4);
                }
            }
        }

        if (hasAlpha)
            StraightAlpha.Unpremultiply(rgba);
        else
            StraightAlpha.SetOpaque(rgba);

        return rgba;
    }
}
#endif
