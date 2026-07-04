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
using System.Text.Json;

namespace TensorSharp.Server
{
    public static class OpenAIResponseFormatParser
    {
        public static bool TryParse(JsonElement body, out StructuredOutputFormat format, out string error)
        {
            format = null;
            error = null;

            if (!body.TryGetProperty("response_format", out var responseFormatEl) ||
                responseFormatEl.ValueKind == JsonValueKind.Null ||
                responseFormatEl.ValueKind == JsonValueKind.Undefined)
            {
                return true;
            }

            return TryParseFormatElement(responseFormatEl, "response_format", out format, out error);
        }

        /// <summary>
        /// Responses API equivalent of <see cref="TryParse"/>: the same
        /// <c>{type, json_schema}</c> shape lives under <c>text.format</c>
        /// instead of a top-level <c>response_format</c>.
        /// </summary>
        public static bool TryParseResponsesText(JsonElement body, out StructuredOutputFormat format, out string error)
        {
            format = null;
            error = null;

            if (!body.TryGetProperty("text", out var textEl) || textEl.ValueKind != JsonValueKind.Object ||
                !textEl.TryGetProperty("format", out var formatEl) ||
                formatEl.ValueKind == JsonValueKind.Null || formatEl.ValueKind == JsonValueKind.Undefined)
            {
                return true;
            }

            return TryParseFormatElement(formatEl, "text.format", out format, out error);
        }

        private static bool TryParseFormatElement(JsonElement formatEl, string fieldName, out StructuredOutputFormat format, out string error)
        {
            format = null;
            error = null;

            if (formatEl.ValueKind != JsonValueKind.Object)
            {
                error = $"{fieldName} must be an object.";
                return false;
            }

            if (!formatEl.TryGetProperty("type", out var typeEl) || typeEl.ValueKind != JsonValueKind.String)
            {
                error = $"{fieldName}.type is required.";
                return false;
            }

            string type = typeEl.GetString();
            switch (type)
            {
                case "text":
                    return true;

                case "json_object":
                    format = StructuredOutputFormat.JsonObject();
                    return true;

                case "json_schema":
                    return TryParseJsonSchema(formatEl, out format, out error);

                default:
                    error = $"Unsupported {fieldName}.type '{type}'.";
                    return false;
            }
        }

        private static bool TryParseJsonSchema(JsonElement responseFormatEl, out StructuredOutputFormat format, out string error)
        {
            format = null;
            error = null;

            JsonElement schemaWrapper = responseFormatEl;
            if (responseFormatEl.TryGetProperty("json_schema", out var nestedSchemaEl))
            {
                if (nestedSchemaEl.ValueKind != JsonValueKind.Object)
                {
                    error = "response_format.json_schema must be an object.";
                    return false;
                }

                schemaWrapper = nestedSchemaEl;
            }

            if (!schemaWrapper.TryGetProperty("name", out var nameEl) || nameEl.ValueKind != JsonValueKind.String ||
                string.IsNullOrWhiteSpace(nameEl.GetString()))
            {
                error = "response_format.json_schema.name is required.";
                return false;
            }

            if (!schemaWrapper.TryGetProperty("schema", out var schemaEl) || schemaEl.ValueKind != JsonValueKind.Object)
            {
                error = "response_format.json_schema.schema must be an object.";
                return false;
            }

            bool strict = schemaWrapper.TryGetProperty("strict", out var strictEl) &&
                          strictEl.ValueKind is JsonValueKind.True or JsonValueKind.False &&
                          strictEl.GetBoolean();

            string description = schemaWrapper.TryGetProperty("description", out var descriptionEl) &&
                                 descriptionEl.ValueKind == JsonValueKind.String
                ? descriptionEl.GetString()
                : null;

            format = StructuredOutputFormat.JsonSchema(
                nameEl.GetString(),
                schemaEl.GetRawText(),
                strict,
                description);

            return true;
        }
    }
}


