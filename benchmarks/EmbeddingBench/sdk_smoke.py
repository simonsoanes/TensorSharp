#!/usr/bin/env python3
"""Verify running embedding services with the official OpenAI and Ollama SDKs.

Example using the validation virtual environment:
  /tmp/tensorsharp-embedding-venv/bin/python benchmarks/EmbeddingBench/sdk_smoke.py \
    --service http://127.0.0.1:18400 ggml_metal \
    --service http://127.0.0.1:18401 cpu \
    --output docs/validation/embeddings-2026-09/sdk-smoke.json

This script sends local inference requests; it does not start or stop services.
"""
import argparse
import asyncio
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

from openai import AsyncOpenAI, OpenAI
import ollama

import embedding_bench as bench
from final_verify import preserve, sha256


TEXTS = ["Hello world", "Searching documents using semantic similarity.", "检索多语言文档。"]


def rows(response):
    assert [item.index for item in response.data] == list(range(len(response.data)))
    return [item.embedding for item in response.data]


def maximum_error(first, second):
    assert len(first) == len(second)
    return max(abs(a - b) for x, y in zip(first, second) for a, b in zip(x, y))


async def concurrent_sdk_requests(url, model, reference):
    async with AsyncOpenAI(base_url=url + "/v1", api_key="local-embedding-service") as client:
        results = await asyncio.gather(*(client.embeddings.create(model=model, input=TEXTS[index % len(TEXTS)])
                                         for index in range(8)))
    similarities = []
    for index, result in enumerate(results):
        actual = rows(result)
        bench.assert_vectors(actual, len(reference[0]))
        assert len(actual) == 1
        similarity = bench.cosine(actual[0], reference[index % len(TEXTS)])
        assert similarity > .9999, similarity
        similarities.append(similarity)
    return dict(requests=len(results), minimum_cosine_to_reference=min(similarities),
                prompt_tokens=[result.usage.prompt_tokens for result in results])


def verify(url, expected_backend, model):
    url = url.rstrip("/")
    status, health = bench.request(url, "/health")
    assert status == 200, health
    status, metadata = bench.request(url, "/api/models")
    assert status == 200 and metadata["loadedBackend"] == expected_backend, metadata
    with OpenAI(base_url=url + "/v1", api_key="local-embedding-service") as client:
        discovery = client.models.list()
        assert model in [item.id for item in discovery.data]
        # The official SDK requests base64 by default and decodes it to floats.
        automatic = client.embeddings.create(model=model, input=TEXTS)
        explicit = client.embeddings.create(model=model, input=TEXTS, encoding_format="float")
        automatic_rows, float_rows = rows(automatic), rows(explicit)
        dimensions = len(float_rows[0])
        assert len(float_rows) == len(TEXTS)
        bench.assert_vectors(automatic_rows, dimensions)
        bench.assert_vectors(float_rows, dimensions)
        assert maximum_error(automatic_rows, float_rows) < 1e-7
        reduced = rows(client.embeddings.create(model=model, input=TEXTS, dimensions=256))
        bench.assert_vectors(reduced, 256)
        assert min(bench.cosine(small, full[:256]) for small, full in zip(reduced, float_rows)) > .99999
        assert model.startswith("snowflake-arctic-embed-l-v2.0"), "The raw-token fixture is pinned to Snowflake v2."
        token_response = client.embeddings.create(model=model, input=[0, 10, 2])
        text_response = client.embeddings.create(model=model, input="a")
        assert token_response.usage.prompt_tokens == text_response.usage.prompt_tokens == 3
        token_error = maximum_error(rows(token_response), rows(text_response))
        assert token_error < 1e-7, token_error
    ollama_client = ollama.Client(host=url)
    modern = ollama_client.embed(model=model, input=TEXTS, truncate=False)
    bench.assert_vectors(modern.embeddings, dimensions)
    modern_error = maximum_error(float_rows, modern.embeddings)
    assert modern_error < 1e-6, modern_error
    assert automatic.usage.prompt_tokens == explicit.usage.prompt_tokens == modern.prompt_eval_count
    assert automatic.usage.total_tokens == automatic.usage.prompt_tokens
    concurrent = asyncio.run(concurrent_sdk_requests(url, model, float_rows))
    return dict(url=url, expected_backend=expected_backend, loaded_backend=metadata["loadedBackend"],
                model=model, passed=True, dimensions=dimensions, inputs=TEXTS,
                prompt_tokens=automatic.usage.prompt_tokens, ollama_prompt_eval_count=modern.prompt_eval_count,
                default_openai_encoding_sdk_decoded=True, explicit_float_max_error=maximum_error(automatic_rows, float_rows),
                reduced_dimensions=256, vector_norms=[bench.norm(vector) for vector in float_rows],
                ollama_max_error=modern_error,
                raw_token_fixture=dict(tokens=[0, 10, 2], text="a", prompt_tokens=3, max_error=token_error),
                concurrent=concurrent, health=health)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", nargs=2, action="append", metavar=("URL", "BACKEND"), required=True)
    parser.add_argument("--model", default="snowflake-arctic-embed-l-v2.0-q8_0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  command=[sys.executable, *sys.argv], sdk_versions={"openai": version("openai"), "ollama": version("ollama")},
                  preserved_evidence=preserve(args.output), services=[], services_left_running=True)
    anchor = args.output.parent / "dispatcher-focused-tests.json"
    if anchor.exists():
        result["binary_anchor"] = dict(path=str(anchor), sha256=sha256(anchor),
                                       host_binaries=json.loads(anchor.read_text())["host_binaries"])
    try:
        for url, backend in args.service:
            print("Checking official SDKs:", url, backend, flush=True)
            result["services"].append(verify(url, backend, args.model))
        result["passed"] = True
        print("Official OpenAI/Ollama SDK smoke passed for all running services.", flush=True)
    except BaseException as error:
        result.update(passed=False, error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
