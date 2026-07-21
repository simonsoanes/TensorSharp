namespace TensorSharp.Server
{
    internal static class TextUploadHelper
    {
        /// <summary>
        /// Normalizes an uploaded text payload without applying an upload-time
        /// character or token budget. Context validation belongs to the fully
        /// rendered prompt, where message/template/generation overhead is known;
        /// cutting a file here loses source data and needlessly rejects documents
        /// that fit comfortably in the model's actual context window.
        /// </summary>
        internal static string PreserveFullText(string textContent)
        {
            return textContent ?? string.Empty;
        }
    }
}
