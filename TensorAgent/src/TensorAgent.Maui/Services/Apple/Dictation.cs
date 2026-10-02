// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AVFoundation;
using Foundation;
using Speech;

namespace TensorAgent.Maui.Services;

/// <summary>
/// Voice input, straight onto Apple's own speech recogniser.
///
/// <para>
/// This is deliberately not a toolkit package. The app is fully ahead-of-time
/// compiled and aggressively trimmed, and the whole feature is one recogniser, one
/// audio tap and one request object — sixty lines against a dependency tree, for
/// something that has to keep working when the trimmer is at its most aggressive.
/// </para>
/// <para>
/// It asks for on-device recognition where the device offers it. That is not a
/// performance choice: dictation for a private, offline chat app must not send the
/// user's speech to a server, and a recogniser that quietly falls back to the
/// network would do exactly that. Where on-device recognition is unavailable the
/// caller is told, rather than being upgraded to a server round trip it did not ask
/// for.
/// </para>
/// </summary>
internal sealed class Dictation : IDisposable
{
    private readonly AVAudioEngine _engine = new();
    private SFSpeechRecognizer? _recognizer;
    private readonly string _language;

    /// <param name="language">BCP-47 tag such as "en-US" or "zh-CN"; empty follows the device.</param>
    public Dictation(string? language = null) => _language = language ?? string.Empty;

    private SFSpeechAudioBufferRecognitionRequest? _request;
    private SFSpeechRecognitionTask? _task;
    private TaskCompletionSource<string>? _completion;

    /// <summary>Whether this device has a recogniser for the language dictation would use.</summary>
    public static bool IsSupported => new SFSpeechRecognizer(ResolveLocale(string.Empty)) is { Available: true };

    /// <summary>What the page says when <see cref="IsSupported"/> is false.</summary>
    public const string UnsupportedMessage = "Speech recognition is not available on this device.";

    /// <summary>
    /// The locale to recognise in: the caller's explicit choice, or — for "Auto" — the
    /// language the USER chose, not the region their phone formats dates in.
    ///
    /// <para>
    /// This used to be <c>NSLocale.CurrentLocale</c>, and that is a different thing
    /// from what anyone means by "Auto". CurrentLocale is the region and formatting
    /// locale; a phone set to the United States reports en_US however many languages
    /// its owner has added. Speaking Chinese into an English recogniser does not fail —
    /// it succeeds, and hands back the sounds romanised. Reported from a phone as
    /// "Auto gives me Pinyin instead of Chinese characters", which is exactly what an
    /// English recogniser does with Mandarin.
    /// </para>
    /// <para>
    /// <c>PreferredLanguages</c> is the ordered list the user actually set in
    /// Settings › General › Language &amp; Region, so a bilingual user's first language
    /// wins. It is intersected with what this device can actually recognise, because a
    /// preferred language with no recogniser has to fall through to the next one rather
    /// than fail. iOS still recognises ONE language per session and cannot detect which
    /// is being spoken — that is why the chips beside the button exist — but Auto now
    /// means something true.
    /// </para>
    /// </summary>
    internal static NSLocale ResolveLocale(string? language)
    {
        if (!string.IsNullOrWhiteSpace(language))
            return new NSLocale(language);

        try
        {
            NSLocale[] supported = SFSpeechRecognizer.SupportedLocales?.ToArray() ?? Array.Empty<NSLocale>();
            if (supported.Length == 0)
                return NSLocale.CurrentLocale;

            foreach (string preferred in NSLocale.PreferredLanguages ?? Array.Empty<string>())
            {
                if (Match(supported, preferred) is { } exact)
                    return exact;
            }
            // Nothing the user listed can be recognised here; the region locale is the
            // next most likely thing to be right, and after that anything at all beats
            // refusing to listen.
            return Match(supported, NSLocale.CurrentLocale.Identifier) ?? NSLocale.CurrentLocale;
        }
        catch (Exception ex)
        {
            Console.WriteLine("TensorAgent: could not resolve a dictation locale: " + ex.Message);
            return NSLocale.CurrentLocale;
        }
    }

    /// <summary>
    /// The supported locale that best answers <paramref name="wanted"/>: the same tag,
    /// or failing that the same language. "zh-Hans-CN" has to find "zh-CN", and
    /// identifiers arrive in both ICU ("zh_CN") and BCP-47 ("zh-CN") spellings.
    /// </summary>
    private static NSLocale? Match(IReadOnlyList<NSLocale> supported, string wanted)
    {
        string want = Normalize(wanted);
        foreach (NSLocale locale in supported)
        {
            if (string.Equals(Normalize(locale.Identifier), want, StringComparison.OrdinalIgnoreCase))
                return locale;
        }

        string language = want.Split('-')[0];
        foreach (NSLocale locale in supported)
        {
            if (string.Equals(Normalize(locale.Identifier).Split('-')[0], language, StringComparison.OrdinalIgnoreCase))
                return locale;
        }
        return null;
    }

    private static string Normalize(string identifier) =>
        (identifier ?? string.Empty).Replace('_', '-');

    /// <summary>Ask for the two permissions this needs, and say which one was refused.</summary>
    /// <summary>
    /// Marker the caller can look for to know the refusal is permanent and only the
    /// user can lift it, so it can offer Settings instead of repeating itself.
    /// </summary>
    public const string DeniedMarker = "[denied]";

    /// <summary>
    /// Ask for the two permissions dictation needs, or say why it cannot have them.
    ///
    /// <para>
    /// Two, not one: speech recognition and the microphone are separate grants, and a
    /// build that asks for only the first fails later inside the audio session with an
    /// error that says nothing about permissions.
    /// </para>
    /// <para>
    /// A permission already DENIED is reported differently from one not yet asked for.
    /// iOS shows its prompt once; after that RequestAuthorization returns the stored
    /// answer without showing anything, so a user who said no by reflex sees the
    /// button fail forever with no way back. That case names Settings, and carries
    /// <see cref="DeniedMarker"/> so the caller can offer to open it.
    /// </para>
    /// </summary>
    public static async Task<string?> RequestPermissionsAsync()
    {
        // RunContinuationsAsynchronously on both: iOS delivers these callbacks on its
        // own threads, and without it everything after the await -- including starting
        // the AVAudioSession -- continues on that thread instead of the main one.
        if (SFSpeechRecognizer.AuthorizationStatus is SFSpeechRecognizerAuthorizationStatus.Denied
            or SFSpeechRecognizerAuthorizationStatus.Restricted)
        {
            return "Speech recognition is turned off for TensorAgent. Turn it on in "
                + SettingsPath("Speech Recognition") + ". " + DeniedMarker;
        }

        var speech = new TaskCompletionSource<SFSpeechRecognizerAuthorizationStatus>(
            TaskCreationOptions.RunContinuationsAsynchronously);
        SFSpeechRecognizer.RequestAuthorization(speech.SetResult);
        if (await speech.Task != SFSpeechRecognizerAuthorizationStatus.Authorized)
            return "Dictation needs permission to use speech recognition. " + DeniedMarker;

        if (AVAudioApplication.SharedInstance.RecordPermission == AVAudioApplicationRecordPermission.Denied)
        {
            return "The microphone is turned off for TensorAgent. Turn it on in "
                + SettingsPath("Microphone") + ". " + DeniedMarker;
        }

        var microphone = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        AVAudioApplication.RequestRecordPermission(microphone.SetResult);
        return await microphone.Task ? null : "Dictation needs permission to use the microphone. " + DeniedMarker;
    }

    /// <summary>Where the user turns a refused permission back on, on this platform.</summary>
    private static string SettingsPath(string permission) =>
#if MACCATALYST
        "System Settings › Privacy & Security › " + permission;
#else
        "Settings › TensorAgent › " + permission;
#endif

    /// <summary>
    /// Listen until <see cref="Stop"/> is called, then return everything recognised.
    /// Partial results are discarded on purpose: appending them to the composer would
    /// rewrite the user's own text while they were still speaking.
    /// </summary>
    public async Task<string> ListenAsync()
    {
        // One locale per session: iOS does not detect the spoken language, so the
        // caller's choice decides. Empty means Auto, which resolves to the user's own
        // preferred language rather than to their region -- see ResolveLocale.
        NSLocale locale = ResolveLocale(_language);
        Console.WriteLine($"TensorAgent: dictating in {locale.Identifier}"
            + (_language.Length == 0 ? " (auto)" : string.Empty));
        _recognizer = new SFSpeechRecognizer(locale)
            ?? throw new InvalidOperationException(
                $"This device has no speech recogniser for {locale.Identifier}.");
        if (!_recognizer.Available)
            throw new InvalidOperationException("The speech recogniser is not available right now.");

        var session = AVAudioSession.SharedInstance();
        session.SetCategory(AVAudioSessionCategory.Record, AVAudioSessionCategoryOptions.DuckOthers);
        session.SetActive(true, AVAudioSessionSetActiveOptions.NotifyOthersOnDeactivation, out NSError? sessionError);
        if (sessionError is not null)
            throw new InvalidOperationException("The microphone could not be started: " + sessionError.LocalizedDescription);

        _request = new SFSpeechAudioBufferRecognitionRequest
        {
            ShouldReportPartialResults = false,
            // Keep the audio on the device. See the class summary: a chat app that
            // runs its model locally must not ship the microphone to a server.
            RequiresOnDeviceRecognition = _recognizer.SupportsOnDeviceRecognition,
        };

        _completion = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);

        AVAudioFormat format = _engine.InputNode.GetBusOutputFormat(0);
        _engine.InputNode.InstallTapOnBus(0, 1024, format, (buffer, _) => _request?.Append(buffer));
        _engine.Prepare();
        _engine.StartAndReturnError(out NSError? engineError);
        if (engineError is not null)
            throw new InvalidOperationException("The microphone could not be started: " + engineError.LocalizedDescription);

        _task = _recognizer.GetRecognitionTask(_request, (result, error) =>
        {
            if (error is not null)
            {
                _completion?.TrySetException(new InvalidOperationException(error.LocalizedDescription));
                return;
            }
            if (result is { Final: true })
                _completion?.TrySetResult(result.BestTranscription?.FormattedString ?? string.Empty);
        });

        try
        {
            return await _completion.Task;
        }
        finally
        {
            Teardown();
        }
    }

    /// <summary>End the session; the recogniser then delivers its final transcription.</summary>
    public void Stop()
    {
        try
        {
            _engine.InputNode.RemoveTapOnBus(0);
            _engine.Stop();
            _request?.EndAudio();
        }
        catch (Exception)
        {
            // Stopping twice, or stopping a session that never started, is not worth
            // an error the user has to read.
        }
    }

    private void Teardown()
    {
        try
        {
            _engine.InputNode.RemoveTapOnBus(0);
            if (_engine.Running)
                _engine.Stop();
            _task?.Cancel();
            _task?.Dispose();
            _request?.Dispose();
            AVAudioSession.SharedInstance().SetActive(false, AVAudioSessionSetActiveOptions.NotifyOthersOnDeactivation, out _);
        }
        catch (Exception) { /* teardown is best effort */ }
        finally
        {
            _task = null;
            _request = null;
        }
    }

    public void Dispose()
    {
        Teardown();
        _engine.Dispose();
        _recognizer?.Dispose();
    }
}
