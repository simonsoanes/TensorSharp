// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// The Qwen-Image-Edit-2511 prompt template + the template-prefix drop count, matching
    /// diffusers <c>QwenImageEditPlusPipeline</c>. With vision grounding each input image
    /// contributes a <c>Picture N: &lt;|vision_start|&gt;&lt;|image_pad|&gt;&lt;|vision_end|&gt;</c>
    /// prefix (1-based, in input order) whose pad expands to one token per merged vision patch.
    /// </summary>
    internal static class QwenImagePrompt
    {
        public const int DropIdx = 64;
        public const int ImagePadTokenId = 151655;   // <|image_pad|>

        public static string BuildWithImages(string prompt, int imageCount)
        {
            var sb = new System.Text.StringBuilder();
            for (int i = 1; i <= imageCount; i++)
                sb.Append($"Picture {i}: <|vision_start|><|image_pad|><|vision_end|>");
            sb.Append(prompt ?? string.Empty);
            return string.Format(Template, sb.ToString());
        }

        private const string Template =
            "<|im_start|>system\nDescribe the key features of the input image (color, shape, " +
            "size, texture, objects, background), then explain how the user's text instruction " +
            "should alter or modify the image. Generate a new image that meets the user's " +
            "requirements while maintaining consistency with the original input where appropriate." +
            "<|im_end|>\n<|im_start|>user\n{0}<|im_end|>\n<|im_start|>assistant\n";

        public static string Build(string prompt) => string.Format(Template, prompt ?? string.Empty);
    }
}
