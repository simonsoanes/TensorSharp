// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Foundation;

namespace TensorAgent.Maui.Platforms.iOS;

/// <summary>
/// The one directory the app and its share extension can both see.
///
/// <para>
/// An app extension runs in its own process with its own sandbox; it cannot read the
/// app's Application Support or Caches directory and the app cannot read its temporary
/// one. An App Group container is the only shared filesystem the two have, which is why
/// the whole hand-off is a directory of envelope files rather than any kind of message.
/// </para>
/// <para>
/// iOS returns null silently when either bundle lacks the App Groups entitlement. That
/// is reported as a configuration error; shared content is never placed in an
/// unauthenticated custom-scheme URL as a fallback.
/// </para>
/// </summary>
internal static class SharedContainer
{
    /// <summary>The App Group both bundles are entitled to. See <see cref="Core"/>.</summary>
    private const string GroupIdentifier = TensorAgent.Sharing.ShareContainer.GroupIdentifier;

    /// <summary>The subdirectory the extension drops envelopes into.</summary>
    private const string InboxDirectoryName = TensorAgent.Sharing.ShareContainer.InboxDirectoryName;

    /// <summary>
    /// Where the extension leaves shares, or an empty string when this build has no
    /// shared container.
    /// </summary>
    public static string InboxDirectory()
    {
        try
        {
            NSUrl? container = NSFileManager.DefaultManager
                .GetContainerUrl(GroupIdentifier);
            if (container?.Path is not { Length: > 0 } root)
            {
                Console.WriteLine(
                    $"TensorAgent: no App Group container for {GroupIdentifier}; sharing is unavailable. "
                    + "Enable App Groups on both App IDs and regenerate both provisioning profiles.");
                return string.Empty;
            }

            string inbox = Path.Combine(root, InboxDirectoryName);
            Directory.CreateDirectory(inbox);
            using (NSUrl inboxUrl = NSUrl.FromFilename(inbox))
                inboxUrl.SetResource(NSUrl.IsExcludedFromBackupKey, NSNumber.FromBoolean(true), out _);
            Console.WriteLine($"TensorAgent: share inbox {inbox}");
            return inbox;
        }
        catch (Exception ex)
        {
            // Never fatal. A share that cannot be picked up is a feature that does
            // nothing; an app that will not start is everything.
            Console.WriteLine("TensorAgent: could not open the App Group container: " + ex.Message);
            return string.Empty;
        }
    }

    /// <summary>
    /// Leave the user's interface-language choice (a tag, or empty for the system's) where
    /// the share extension reads it, since the extension cannot read the app's settings.
    /// See <see cref="TensorAgent.Sharing.Localization.ShareStrings"/>.
    /// </summary>
    public static void WriteLanguageChoice(string? choice)
    {
        try
        {
            NSUrl? container = NSFileManager.DefaultManager.GetContainerUrl(GroupIdentifier);
            if (container?.Path is { Length: > 0 } root)
                TensorAgent.Sharing.Localization.ShareStrings.WriteChoice(root, choice);
        }
        catch (Exception ex)
        {
            // The extension then follows the system's language: a mismatch, not a failure.
            Console.WriteLine("TensorAgent: could not leave the language choice for the share extension: " + ex.Message);
        }
    }
}
