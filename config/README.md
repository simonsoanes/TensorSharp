# Configuration files

Both `TensorSharp.Cli` and `TensorSharp.Server` can read their startup options
from a JSON file passed with `--config`, in addition to the command line:

```bash
TensorSharp.Server --config config/server-basic.json
TensorSharp.Cli    --config config/cli-basic.json
```

**Command-line options always win.** File values are applied first, then anything
you also pass on the command line overrides them — so one config file can be
reused across machines while you override just what differs:

```bash
# Use the file, but force the CPU backend for this run:
TensorSharp.Server --config config/server-basic.json --backend ggml_cpu
```

You can pass `--config` more than once; later files override earlier ones (and
the command line still wins over all of them).

## File format

The file is a JSON object whose keys are the same long option names each host
already accepts (listed in `--help`), with or without the leading `--`. Comments
(`//`, `/* */`) and trailing commas are allowed.

| JSON value          | Becomes                          | Example |
|---------------------|----------------------------------|---------|
| string / number     | `--key value`                    | `"max-tokens": 4096` → `--max-tokens 4096` |
| `true`              | the bare switch `--key`          | `"continuous-batching": true` → `--continuous-batching` |
| `false` / `null`    | nothing (omit the option)        | to turn something off, use its negation key, e.g. `"no-continuous-batching": true` |
| array               | a repeated flag                  | `"stop": ["</s>", "<\|eot\|>"]` → `--stop </s> --stop <\|eot\|>` |
| object              | a downloadable file (see below)  | `{ "path": "...", "urls": ["..."] }` |

## Variables

Define shared values once under `"variables"` and reference them with `${name}`
in any string value. A `${name}` that is not defined there falls back to an
environment variable of the same name, and variables may reference other
variables. See [`variables.json`](variables.json).

```json
{
  "variables": { "modelRoot": "C:/models" },
  "model": "${modelRoot}/gemma-4-E4B-it-Q8_0.gguf",
  "mmproj": "${modelRoot}/mmproj-gemma-4-E4B-it-Q8_0.gguf"
}
```

You can declare **as many variables (and root paths) as you need** — models that
live in different folders each get their own root:

```json
{
  "variables": {
    "ditRoot": "C:/models/qwen-image-edit",
    "companionRoot": "C:/Works/models"
  },
  "model": { "path": "${ditRoot}/qwen-image-edit-2511-Q4_K_M.gguf", "urls": ["..."] },
  "qwen-image-vae": "${companionRoot}/Qwen_Image-VAE.safetensors"
}
```

## Auto-download

Any file option can be an object with a local `path` and one or more `urls`
instead of a plain string. On startup:

- if `path` already exists, it is used as-is (no network access);
- otherwise the file is downloaded from the first working URL, saved to `path`,
  and reused on every later run.

List several mirrors under `urls` for automatic fallback — if the first URL
fails, the next is tried. A relative `path` resolves next to the config file. An
optional `sha256` (lowercase hex) verifies a freshly downloaded file. Download
progress is printed to standard error so you can see what is being fetched and
how far along it is. See [`auto-download.json`](auto-download.json).

```json
{
  "model": {
    "path": "C:/models/Qwen3.5-9B-Q8_0.gguf",
    "urls": [
      "https://primary-mirror.example.com/Qwen3.5-9B-Q8_0.gguf",
      "https://backup-mirror.example.com/Qwen3.5-9B-Q8_0.gguf"
    ],
    "sha256": "0123...optional..."
  }
}
```

`url` (singular) is accepted as a shorthand for a single-entry `urls`.

## Examples in this folder

Every example uses **real, public, ungated** GGUF URLs, so each one works on a
fresh machine: the files auto-download to their local `path` on the first run and
are reused afterward. If you already have a file at that `path`, it is used as-is
(no download).

| File | Model(s) | Shows |
|------|----------|-------|
| [`cli-basic.json`](cli-basic.json) | Qwen3.5-9B (~8.9 GB) | Minimal CLI config, auto-download |
| [`server-basic.json`](server-basic.json) | Gemma-4 E4B model + vision projector | Multimodal server, sampling defaults, auto-download |
| [`variables.json`](variables.json) | Gemma-4 26B-A4B: model + mmproj + MTP draft | One shared root/repo reused across three related files |
| [`auto-download.json`](auto-download.json) | Qwen3-0.6B (~640 MB) | Small, fast download demo with mirror fallback |
| [`qwen-image-edit.json`](qwen-image-edit.json) | Qwen-Image-Edit 2511: DiT + VAE + text encoder + mmproj + Lightning LoRA | Multi-file image pipeline, all auto-downloaded |

`server-basic.json` uses the standard `gemma-4-E4B-it` build — point its `path` at
your own file to host a different variant.
