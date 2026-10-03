This metadata-only diagnostic checks whether a Qwen follow-up prompt reproduces
the token prefix of a previous request. It loads the GGUF vocabulary and template,
not the model weights, and does not need a GPU.

Run the server with `TS_CB_DEBUG=1`, greedy sampling, repetition penalty 1 and
speculation off. Pass the completed request's ID and original user text:

```powershell
dotnet run --project eng/validation/QwenCacheReplayProbe -c Release `
  -p:TensorSharpSkipGgmlNative=true -- MODEL.gguf DEBUG_LOG REQUEST_ID 'USER TEXT'
```

The diagnostic reconstructs generated IDs from the debug log, renders a text
follow-up and reports the matching prefix length. A nonzero exit code means the
rendered follow-up differs from the cache. This isolates template/tokenizer
problems from cache retention problems; it does not validate model state or
multimodal expansion. Sampling and speculative decoding must stay disabled
because debug `top1` is the unmodified greedy logit choice.

Save generated reports under ignored `docs/validation/` or `artifacts/`.
