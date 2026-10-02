#!/usr/bin/env python3
"""Check V4.1 complete image spans and mixed text/image inference against oracles.

Generate vision fixtures with eng/dsv41-vision-fixture.py. Its expected spans
come directly from the pinned official vision.py. Optional --text-fixture runs
the independent GGUF language oracle, including visual router biases, masked
Engram injection, history barriers, chunking, slots, and rejected input state.
"""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


def bind(library, name, arguments, result=ctypes.c_int):
    function = getattr(library, "TSGgml_" + name)
    function.argtypes, function.restype = arguments, result
    return function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--text-fixture", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", default="CPU")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpu-moe", type=int, default=0)
    parser.add_argument("--ubatch", type=int, default=32)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--text-atol", type=float, default=2e-5)
    parser.add_argument("--text-rtol", type=float, default=2e-5)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--native-output-dir", type=Path, help="Preserve native complete spans for numerical diagnostics")
    args = parser.parse_args()
    manifest = json.loads((args.fixture_dir / "manifest.json").read_text())
    library = ctypes.CDLL(str(args.library.resolve()))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    load = bind(library, "Dsv41VisionLoad", [ctypes.c_char_p, ctypes.c_char_p, integer, integer], ptr)
    free = bind(library, "Dsv41VisionFree", [ptr], None)
    info = bind(library, "Dsv41VisionInfo", [ptr, ptr, integer])
    encode = bind(library, "Dsv41VisionEncode", [ptr, ptr, integer, integer, ptr, integer])
    vision = load(str(args.fixture_dir / "deepseek41.vision.gguf").encode(), args.backend.encode(), 0, 2)
    if not vision:
        raise RuntimeError("Native vision fixture load failed")
    checks = []

    def invariant(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        assert passed, name

    def compare(name, actual, expected, atol=None, rtol=None):
        atol = args.atol if atol is None else atol
        rtol = args.rtol if rtol is None else rtol
        checks.append(dict(name=name, max_absolute_error=float(np.max(np.abs(actual - expected))),
                           relative_l2=float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)),
                           argmax=int(actual.argmax()), reference_argmax=int(expected.argmax()),
                           atol=atol, rtol=rtol,
                           passed=bool(np.allclose(actual, expected, atol=atol, rtol=rtol))))

    spans, native_spans = {}, {}
    try:
        values = np.zeros(8, dtype=np.int32)
        invariant("vision_info", info(vision, values.ctypes.data, len(values)) == 0)
        dim = manifest["config"]["text_config"]["hidden_size"]
        invariant("vision_metadata", values[2] == dim and values[6] == manifest["config"]["image_token_id"])
        scratch = np.zeros(dim * 4, dtype=np.float32)
        patches = np.zeros(3 * 14 * 14, dtype=np.float32)
        for height, width in ((0, 1), (1, 0), (-1, 1), (2**30, 2**30)):
            invariant(f"reject_grid_{height}_{width}",
                      encode(vision, patches.ctypes.data, height, width, scratch.ctypes.data, scratch.size) < 0)
        for case in manifest["cases"]:
            patches = np.fromfile(args.fixture_dir / case["patches"], dtype="<f4")
            expected = np.fromfile(args.fixture_dir / case["expected"], dtype="<f4").reshape(case["tokens"], dim)
            actual = np.empty_like(expected)
            rows = encode(vision, patches.ctypes.data, *case["patch_grid"], actual.ctypes.data, actual.size)
            invariant(case["name"] + "_span_rows", rows == case["tokens"])
            compare(case["name"] + "_complete_span", actual, expected)
            features = np.array(case["token_types"]) == 1
            checks[-1].update(feature_relative_l2=float(np.linalg.norm(actual[features] - expected[features]) /
                                 max(np.linalg.norm(expected[features]), 1e-30)),
                              feature_max_absolute_error=float(np.max(np.abs(actual[features] - expected[features]))),
                              sentinel_max_absolute_error=float(np.max(np.abs(actual[~features] - expected[~features]))))
            if args.native_output_dir:
                args.native_output_dir.mkdir(parents=True, exist_ok=True)
                actual.tofile(args.native_output_dir / (case["name"] + ".span.f32"))
            spans[case["name"]], native_spans[case["name"]] = expected, actual
            invariant(case["name"] + "_reject_small_output",
                      encode(vision, patches.ctypes.data, *case["patch_grid"], actual.ctypes.data, actual.size - 1) < 0)
        if args.text_fixture:
            spec = importlib.util.spec_from_file_location("dsv41_reference", Path(__file__).parents[1] / "dsv41-reference.py")
            reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference)
            torch.set_num_threads(2)
            metadata = json.loads((args.text_fixture / "deepseek41.config.json").read_text())
            if not metadata.get("fixture"):
                raise ValueError("Mixed-state tests require a small synthetic text fixture")
            weights = reference.GgufWeights(args.text_fixture / "deepseek41-fixture.gguf")
            visual_weights = reference.GgufWeights(args.fixture_dir / "deepseek41.vision.gguf")
            engram = reference.load_engram(weights)
            config = metadata["config"]
            image_id = manifest["config"]["image_token_id"]
            vocab = config["text_config"]["vocab_size"]
            model_load = bind(library, "Dsv4LoadModel",
                              [ctypes.c_char_p] + [integer] * 4 + [ctypes.c_char_p, integer, ctypes.c_char_p, integer], ptr)
            model_free = bind(library, "Dsv4Free", [ptr], None)
            reset_checked = bind(library, "Dsv4ResetChecked", [ptr])
            def reset(handle):
                if reset_checked(handle) != 1:
                    raise RuntimeError("native reset was refused")
            past = bind(library, "Dsv4NPast", [ptr])
            attach = bind(library, "Dsv41AttachVision", [ptr, ptr])
            forward = bind(library, "Dsv41ForwardVision", [ptr, ptr, ptr, ptr, integer, integer, ptr])
            slot_alloc = bind(library, "Dsv4SlotAlloc", [ptr])
            slot_select = bind(library, "Dsv4SetActiveSlot", [ptr, integer])
            slot_free = bind(library, "Dsv4SlotFree", [ptr, integer])
            model = model_load(str(args.text_fixture / "deepseek41-fixture.gguf").encode(),
                               args.gpus, 256, args.ubatch, 2, None, args.cpu_moe, args.backend.encode(), 0)
            if not model:
                raise RuntimeError("Native text fixture load failed")
            try:
                invariant("attach_vision", attach(model, vision) == 0)
                invariant("reject_duplicate_attach", attach(model, vision) < 0)
                free(vision)
                vision = None  # The text model must retain the shared encoder owner.

                def prompt(parts):
                    tokens, masks, embeddings = [], [], []
                    for part in parts:
                        if isinstance(part, str):
                            span = spans[part]
                            tokens.extend([image_id] * len(span))
                            masks.extend([1] * len(span))  # Includes START/NEWLINE/END.
                            embeddings.append(span)
                        else:
                            tokens.extend(part)
                            masks.extend([0] * len(part))
                    return (np.array(tokens, dtype=np.int32), np.array(masks, dtype=np.uint8),
                            np.concatenate(embeddings).astype(np.float32))

                streams = [prompt([[0, 15], "grid_4x5", [32, 64, 128, 13, 254, 18]]),
                           prompt([[0], "grid_3x3", [9, 21], "grid_3x3", [85, 11, 19, 6]])]

                def oracle(stream):
                    instance = reference.Reference(weights, config, engram, "model")
                    ids, mask, embedding = stream
                    result = instance.forward(ids.tolist(), embedding, mask, visual_weights).cpu().numpy()
                    invariant("oracle_negative_image_history", all(token == -1 for token, visual in zip(instance.history, mask) if visual))
                    return result

                targets = [oracle(stream) for stream in streams]

                # Demonstrate that the fixture exercises each new mechanism:
                # these intentionally wrong oracles must produce different
                # logits. A passing comparison with inactive routing/hash
                # differences would provide no coverage of those mechanisms.
                class TextBiasOnly:
                    def weight(self, name):
                        layer = int(name.split(".")[1])
                        return weights.weight(f"blk.{layer}.exp_probs_b.bias")

                class MissingHistoryBarrier(reference.Reference):
                    def eng_inject(self, layer, hidden, tokens):
                        saved = self.history
                        self.history = [image_id if token < 0 else token for token in saved]
                        try:
                            return super().eng_inject(layer, hidden, tokens)
                        finally:
                            self.history = saved

                class MissingInjectionMask(reference.Reference):
                    def eng_inject(self, layer, hidden, tokens):
                        return super().eng_inject(layer, hidden, [image_id if token < 0 else token for token in tokens])

                for name, constructor, bias in (("visual_router_bias", reference.Reference, TextBiasOnly()),
                        ("engram_history_barrier", MissingHistoryBarrier, visual_weights),
                        ("engram_injection_mask", MissingInjectionMask, visual_weights)):
                    wrong = constructor(weights, config, engram, "model")
                    ids, mask, embedding = streams[0]
                    logits = wrong.forward(ids.tolist(), embedding, mask, bias).cpu().numpy()
                    invariant("fixture_exercises_" + name, float(np.max(np.abs(logits - targets[0]))) > 1e-3)

                def run(stream, start, stop):
                    ids, mask, embedding = stream
                    ids = np.ascontiguousarray(ids[start:stop])
                    visual_start = int(mask[:start].sum())
                    selected_mask = np.ascontiguousarray(mask[start:stop])
                    count = int(selected_mask.sum())
                    rows = np.ascontiguousarray(embedding[visual_start:visual_start + count])
                    output = np.empty(vocab, dtype=np.float32)
                    status = forward(model, ids.ctypes.data, selected_mask.ctypes.data, rows.ctypes.data,
                                     len(ids), count, output.ctypes.data)
                    invariant(f"forward_status_{start}_{stop}", status == 0)
                    return output

                for chunk in (len(streams[0][0]), 1, 3, 5):
                    reset(model)
                    for start in range(0, len(streams[0][0]), chunk):
                        stop = min(start + chunk, len(streams[0][0]))
                        compare(f"mixed_chunk_{chunk}_position_{stop}", run(streams[0], start, stop), targets[0][stop - 1], args.text_atol, args.text_rtol)
                        invariant("mixed_position", past(model) == stop)

                # Reject every malformed request before cache/history mutation.
                reset(model)
                run(streams[0], 0, 2)
                ids = np.array([image_id, image_id], dtype=np.int32)
                mask = np.array([1, 1], dtype=np.uint8)
                rows = spans["grid_3x3"][:2].copy()
                output = np.empty(vocab, dtype=np.float32)
                for name, bad_ids, bad_mask, bad_rows, count in (
                    ("mask_value", ids, np.array([2, 0], dtype=np.uint8), rows, 2),
                    ("mask_count", ids, mask, rows, 1),
                    ("image_id", np.array([image_id, 1], dtype=np.int32), mask, rows, 2),
                    ("nonfinite_embedding", ids, mask, np.full_like(rows, np.nan), 2),
                ):
                    result = forward(model, bad_ids.ctypes.data, bad_mask.ctypes.data, bad_rows.ctypes.data, 2, count, output.ctypes.data)
                    invariant("reject_" + name, result < 0 and past(model) == 2)
                compare("rejected_input_continuation", run(streams[0], 2, len(streams[0][0])), targets[0][-1], args.text_atol, args.text_rtol)

                # Alternate independent request states with image spans split
                # across calls; reset/free must not leak negative hash barriers.
                reset(model)
                slot = slot_alloc(model)
                invariant("allocate_second_slot", slot > 0)
                positions = [0, 0]
                while any(positions[i] < len(streams[i][0]) for i in range(2)):
                    for index, slot_id in enumerate((0, slot)):
                        start = positions[index]
                        stop = min(start + 3 + index, len(streams[index][0]))
                        if stop == start:
                            continue
                        invariant("select_slot", slot_select(model, slot_id) == 0)
                        compare(f"mixed_slot_{index}_position_{stop}", run(streams[index], start, stop), targets[index][stop - 1], args.text_atol, args.text_rtol)
                        invariant("slot_position", past(model) == stop)
                        positions[index] = stop
                invariant("restore_slot_zero", slot_select(model, 0) == 0)
                invariant("free_second_slot", slot_free(model, slot) == 0)
                reset(model)
                compare("reset_visual_history", run(streams[1], 0, len(streams[1][0])), targets[1][-1], args.text_atol, args.text_rtol)
            finally:
                model_free(model)
    finally:
        if vision:
            free(vision)
    report = dict(backend=args.backend, gpus=args.gpus, cpu_moe=args.cpu_moe, ubatch=args.ubatch,
                  fixture_dtype=manifest["dtype"], reference_revision=manifest["reference_revision"],
                  oracle_sdpa_backend=manifest.get("sdpa_backend", "auto"),
                  oracle_linear_f32=manifest.get("linear_f32", False),
                  environment={key: os.getenv(key) for key in
                  ("TS_DSV4_FA", "TS_DSV4_GATHER", "TS_DSV41_VISION_FA", "TS_DSV41_VISION_BF16_GEMM")}, checks=checks)
    path = args.report or args.fixture_dir / f"validation-vision-{args.backend.lower()}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    passed = sum(item["passed"] for item in checks)
    print(f"Passed {passed}/{len(checks)} vision checks; {path}")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
