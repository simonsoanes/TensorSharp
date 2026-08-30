# Explicit Prompt Cache Markers - TensorSharp2 Implementation Design

This document outlines the design for implementing explicit prompt-cache markers in TensorSharp2, ensuring full compatibility with explicit caching specifications based on `cache_control` markers.

## 1. Overview

TensorSharp2 currently relies on implicit, automatic prefix caching via `PagedKvCacheManager`, which splits prompts into fixed-size blocks and hashes them. This design introduces support for **explicit caching**, where the client dictates the exact boundaries of cacheable segments using `cache_control` markers.

The implementation will span the API layer (headers and JSON parsing), the prompt rendering layer (tracking markers to exact token indices without leaking them into the prompt), the caching layer (prioritizing/capturing based on markers), and the response serialization layer (reporting `cached_tokens`).

## 2. API & Data Model Changes

### 2.1 Request Parsing
* **Headers**: Update `SessionEndpoints.cs` and `OpenAIChatAdapter.cs` to check for the presence of the `X-Prompt-Cache-Control` or `X-DashScope-CacheControl` headers.
* **DTOs**: Update `ChatMessage.cs` (or the underlying DTOs) and `ToolFunction.cs` to include a `CacheControl` property:
  ```csharp
  public class CacheControlMarker
  {
      public string Type { get; set; } // "ephemeral"
  }
  ```
* **JSON Parsers**: Update `ChatMessageParser.cs` and `ToolFunctionParser.cs` to deserialize the `cache_control` block. Since markers can be on content parts, the `ChatMessage` parser must inspect `type: "text"` parts for `cache_control` objects. Support is also provided for `prompt_cache_breakpoint: true` in these same locations to remain forward-compatible with emerging endpoint proposals.

### 2.2 Opt-In Semantics
* A request is considered to have explicit caching enabled if the header is present (`enable`), OR if any `cache_control` or `prompt_cache_breakpoint` marker is found during request parsing.
* If explicit caching is enabled, TensorSharp2 will transition from *automatic* caching (capturing everything) to *explicit* caching (capturing only up to marked boundaries) for that request.

## 3. Prompt Rendering & Marker Tracking

The spec mandates that markers must not affect the rendered prompt (token sequence must remain identical). However, the engine needs to know the exact token offsets of these markers.

### 3.1 Tracking in `KVCachePromptRenderer`
`KVCachePromptRenderer` currently uses a sentinel strategy (`PlaceholderSentinel`) to splice raw tokens. We will extend this strategy to track explicit breakpoints:
1. **Injection**: When formatting the prompt, if a content part or tool has a `cache_control` marker, insert a unique, invisible breakpoint sentinel (e.g., `\uE001B{index}\uE001`) at the exact location of the marker.
2. **Tokenization**: Tokenize the entire text string.
3. **Extraction & Stripping**: Iterate through the resulting tokens. Locate the breakpoint sentinels. Record their token indices, and then **remove** the sentinel tokens from the sequence so they never reach the model.
4. **Result**: The renderer outputs the clean `List<int>` of input tokens, plus a `List<int>` of token indices representing the explicit breakpoints (`[m1, m2, m3]`).

## 4. Cache Manager & Engine Integration

### 4.1 Capturing Segments
Currently, `PagedKvCacheManager.Capture` tries to snapshot all available tokens rounded down to a block boundary.
* With explicit markers, we pass the breakpoint indices `[m1, m2, m3]` to the engine/cache manager.
* `PagedKvCacheManager.Capture` will capture prefixes **up to the provided breakpoints**. 
* **Block Alignment**: Because `PagedKvCacheManager` uses fixed blocks (e.g., 256 tokens), a marker at token `M` might not align with a block boundary. 
  * *Option A (Strict)*: We only capture full blocks up to `M` (e.g., `M / 256` blocks). This loses the tail end of the segment but maintains fixed-block simplicity.
  * *Option B (Padded/Variable)*: We store the partial block. However, GPU kernels heavily rely on uniform block sizes.
  * *Decision*: Proceed with Option A for phase 1. The spec acknowledges: *"block-based caches naturally floor at their block size... losing the tail of a matching prefix"*. Capturing `Math.Floor(M / BlockSize)` blocks is acceptable and safe.

### 4.2 Retention Priority (Future Enhancement)
The spec suggests giving retention priority to explicitly marked segments.
* We can augment `PagedKvBlockStore` to distinguish between "automatic" blocks and "explicit" blocks.
* Explicit blocks can bypass standard LRU eviction until all automatic blocks have been evicted, protecting long-lived agent conversations from background noise.

## 5. Usage Reporting (Mandatory)

Client applications and agents rely on `cached_tokens` to measure cache effectiveness. If it's missing, they may assume the cache is unmeasurable.

### 5.1 Telemetry
* `InferenceCompletion.PrefixCacheReusedTokens` already records how many tokens were restored from the cache.
* This is piped through `ChatGenerationPipeline` to `kvCacheReusedTokens`.

### 5.2 Response Serializers
* Update `OpenAIResponseFactory.cs` and `OpenAIResponsesFactory.cs`.
* In the `usage` block of the JSON output, add the `prompt_tokens_details` (or `input_tokens_details` for Responses API) object:
  ```json
  "usage": {
    "prompt_tokens": 1000,
    "completion_tokens": 50,
    "total_tokens": 1050,
    "prompt_tokens_details": {
      "cached_tokens": 768
    }
  }
  ```
* Ensure this is populated for **both** the standard (non-streaming) response and the final chunk of a streaming response.
* Always emit the `prompt_tokens_details` object, even if `cached_tokens` is 0.

## 6. Execution Plan

1. **Parser Layer**: Update JSON parsers in `RequestParsers/` to extract `cache_control` and populate the DTOs.
2. **Rendering Layer**: Modify `KVCachePromptRenderer.cs` to inject, locate, and strip breakpoint sentinels, returning a list of breakpoint indices.
3. **Engine Layer**: Update `SequenceState` to carry `CacheBreakpoints`. Modify `InferenceEngine` to pass these breakpoints to `PagedKvCacheManager.Capture`.
4. **Serialization Layer**: Update `OpenAIResponseFactory` and streaming writers to format the `prompt_tokens_details.cached_tokens` block accurately.
5. **Tests**: Add unit tests replicating the conformance vectors (V1-V8) from the specification.