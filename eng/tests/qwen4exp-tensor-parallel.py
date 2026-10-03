#!/usr/bin/env python3
"""Qwen4Exp tensor-sliced FFN span parity, replay and multi-rank rollback.

Two synthetic layers exercise GDN, PLE, dense/QSA attention, multi-axis RoPE,
all-logit speculative taps and two FFN reductions. This fixture does not claim
full-model quality or multi-device performance. CPU/Metal loopback is explicitly
correctness-only. --devices selects actual CUDA devices for device collectives.
"""
import argparse
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


targets = module("q4e_targets", "qwen4exp-target-snapshot.py")
qsa_module = module("q4e_qsa", "qwen4exp-qsa.py")
op = targets.op
P, I, F = C.c_void_p, C.c_int, C.c_float


def sample_from_source():
    source = (Path(__file__).resolve().parents[2] / "InferenceWeb.Tests/Qwen4ExpMtpSample.cs").read_text()
    arrays = {}
    for name, key in (("EmbeddingNorm", "embedding_norm"), ("HiddenNorm", "hidden_norm"),
                      ("Eh", "eh"), ("HeadNorm", "head_norm"), ("HeadDown", "head_down"), ("HeadUp", "head_up")):
        match = re.search(r"float\[\] " + name + r" = new float\[\]\s*\{(.*?)\};", source, re.S)
        arrays[key] = [float(x.strip().removesuffix("f")) for x in match[1].split(",") if x.strip()]
    return {"epsilon": 1e-6, "arrays": arrays}


def quantized_layout_sample(hidden, low):
    rng = np.random.default_rng(9351)
    hc = op.Fixture.HC
    shapes = {"embedding_norm": (hidden,), "hidden_norm": (hc,hidden),
              "eh": (hidden,2*hidden), "head_norm": (hc,hidden),
              "head_down": (low,hc*hidden), "head_up": (hc*hidden,low)}
    return {"epsilon": 1e-6, "arrays": {name: (np.ones(shape) if name.endswith("norm")
        else rng.normal(0,.2/np.sqrt(shape[-1]),shape)).reshape(-1).tolist() for name,shape in shapes.items()}}


def main():
    global _library
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backend", choices=("CPU", "CUDA", "Metal"), default="CPU")
    parser.add_argument("--devices", help="Actual device indices; omit for two-rank loopback")
    parser.add_argument("--degree", type=int, default=2, help="Loopback rank count")
    parser.add_argument("--tokens", default="1,3,17", help="Comma-separated prefill/replay widths")
    parser.add_argument("--rollback-width", type=int, default=3, help="Verification width after the five-token snapshot prefix")
    parser.add_argument("--quantized-ffn", action="store_true", help="Public-admissible quantized FFNs: H=256*degree,FF640,IQ3_S/IQ4_NL and sharedQ8_0")
    args = parser.parse_args()
    widths = [int(value) for value in args.tokens.split(",")]
    if not widths or any(count < 1 for count in widths): raise ValueError("Token widths must be positive")
    if args.rollback_width < 1: raise ValueError("Rollback width must be positive")
    degree = len(args.devices.split(",")) if args.devices else args.degree
    if args.quantized_ffn:
        op.Fixture.H, op.Fixture.LOW = 256 * degree, 16
        op.Fixture.FF = op.Fixture.SH = 640
        op.Fixture.EXP, op.Fixture.USED = 8, 4
        sample = quantized_layout_sample(op.Fixture.H, op.Fixture.LOW)
    else:
        op.Fixture.FF = op.Fixture.SH = 4 * degree
        sample = sample_from_source()
    t = targets.Target(sample, "gdn32" if args.backend == "CUDA" else "tiny")
    b = t.base
    dll = C.CDLL(str(args.library.resolve()))
    _library = dll
    error = dll.TSGgml_GetLastError; error.restype = C.c_char_p
    backend = {"CPU": 2, "CUDA": 3, "Metal": 1}[args.backend]
    if args.devices:
        init = dll.TSGgml_TensorParallelInit; init.argtypes = [I, P, I, I]; init.restype = I
        devices = (I * degree)(*map(int, args.devices.split(",")))
        assert init(backend, devices, degree, 0), error()
    else:
        init = dll.TSGgml_TensorParallelInitLoopback; init.argtypes = [I, I]; init.restype = I
        assert init(backend, degree), error()
    if args.quantized_ffn:
        quantize = dll.ggml_quantize_chunk
        quantize.argtypes = [I,P,P,C.c_int64,C.c_int64,C.c_int64,P];quantize.restype = C.c_size_t
        row_size = dll.ggml_row_size;row_size.argtypes = [I,C.c_int64];row_size.restype = C.c_size_t
        for field,kind in (("gate_exps",21),("up_exps",21),("down_exps",20),
                           ("sh_gate",8),("sh_up",8),("sh_down",8)):
            source = b.arrays[field]
            packed = np.empty((*source.shape[:-1],int(row_size(kind,source.shape[-1]))),np.uint8)
            assert quantize(kind,source.ctypes.data,packed.ctypes.data,0,source.size//source.shape[-1],source.shape[-1],None)==packed.nbytes
            b.arrays[field] = packed
            setattr(b.f,field,packed.ctypes.data);setattr(b.f,field+"_type",kind);setattr(b.f,field+"_bytes",packed.nbytes)
        t.ffn = (op.Ffn*2)(b.f,b.f)
    signature = [P, P, P, P, I, I, P, P] + [I] * 16 + [F] * 3 + [I] * 4 + [F, I, I, P, P, P, I, P, P, P, I, I]
    normal = dll.TSGgml_Qwen4ExpTokenSpanQsa; normal.argtypes = signature + [P, I, P, P, I]; normal.restype = I
    tp = dll.TSGgml_Qwen4ExpTokenSpanTp; tp.argtypes = normal.argtypes + [C.POINTER(P)]; tp.restype = I
    execute = dll.TSGgml_TensorParallelExecutePlans; execute.argtypes = [P, I]; execute.restype = I
    release = dll.TSGgml_Qwen4ExpReleaseAllSeqState; release.argtypes = []; release.restype = None
    clear = dll.TSGgml_ClearHostBufferCache; clear.argtypes = []; clear.restype = None
    create = dll.TSGgml_Qwen4ExpStateSnapshotCreate; create.argtypes = [P, P, I, P, P, P]; create.restype = P
    capture = dll.TSGgml_Qwen4ExpStateSnapshotCapture; capture.argtypes = [P]; capture.restype = I
    restore = dll.TSGgml_Qwen4ExpStateSnapshotRestore; restore.argtypes = [P]; restore.restype = I
    free = dll.TSGgml_Qwen4ExpStateSnapshotFree; free.argtypes = [P]; free.restype = None
    ggml = Path(__file__).resolve().parents[2] / "ExternalProjects/ggml"
    revision = subprocess.run(["git", "-C", str(ggml), "rev-parse", "HEAD"], capture_output=True, text=True)
    report = dict(backend=args.backend, degree=degree, devices=args.devices, tokens=args.tokens,
                  unexecuted_default_widths=[count for count in (1, 3, 17) if count not in widths],
                  weight_type="IQ3_S/IQ4_NL routed, Q8_0 shared" if args.quantized_ffn else "F32 synthetic direct ABI; public quantized-model qualification is separate",
                  hidden=b.H, ff=b.FF, rollback_width=args.rollback_width,
                  native_sha256=op.sha(args.library),
                  ggml_revision=revision.stdout.strip() if revision.returncode == 0 else "unavailable",
                  loopback_correctness_only=not bool(args.devices), checks=[])

    def check(name, actual, expected):
        delta = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        ok = bool(np.isfinite(actual).all() and np.allclose(actual, expected, atol=2e-5, rtol=2e-5))
        report["checks"].append(dict(name=name, passed=ok, max_abs=delta))
        if not ok: report["status"] = "failed"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        assert ok, (name, delta)

    shard_arrays, descriptors = [], []
    for rank in range(degree):
        desc = op.Ffn.from_buffer_copy(b.f)
        for field, axis in (("gate_exps", 1), ("up_exps", 1), ("down_exps", 1),
                            ("sh_gate", 0), ("sh_up", 0), ("sh_down", 0)):
            source = b.arrays[field];rows = source.shape[axis]
            first,end = rank*rows//degree,(rank+1)*rows//degree
            if args.quantized_ffn and field in ("gate_exps","up_exps","sh_gate","sh_up"):
                first,end = first//128*128,(end+127)//128*128
            selection = [slice(None)]*source.ndim;selection[axis] = slice(first,end)
            value = np.ascontiguousarray(source[tuple(selection)])
            shard_arrays.append(value)
            setattr(desc, field, value.ctypes.data)
            setattr(desc, field + "_bytes", value.nbytes)
        descriptors.append((op.Ffn * 2)(desc, desc))

    qsa = (qsa_module.Args * 2)()
    d = qsa[1]
    qsa_arrays = {}
    for name, shape in (("k_proj", (8, b.H)), ("q_proj", (2 * 8, b.H)),
                        ("k_norm", (8,)), ("q_norm", (8,)), ("cache", (b.capacity, 8))):
        value = np.zeros(shape, np.float32) if name == "cache" else np.asarray(b.rng.normal(0, .3, shape), np.float32)
        if name.endswith("norm"): value += 1
        qsa_arrays[name] = value
        setattr(d, name, value.ctypes.data)
    d.k_bytes, d.q_bytes, d.cache_bytes = [qsa_arrays[n].nbytes for n in ("k_proj", "q_proj", "cache")]
    d.head_dim, d.heads, d.ratio, d.top_k = 8, 2, 2, 2
    d.rope_sections[:] = (2, 1, 1, 0)

    def reset():
        release(); clear()
        for key in ("k_cache", "v_cache", "gdn.conv_state", "gdn.ssm_state", "ple.conv_state"):
            b.arrays[key].fill(0)
        qsa_arrays["cache"].fill(0)

    def run(parallel, count, position, seed, sparse=False, media=False):
        e, hidden = b.inputs(count, seed)
        residual = hidden.copy()
        mask = np.full((count, position + count), -np.inf, np.float16)
        for row in range(count): mask[row, :position + row + 1] = 0
        out = np.full((count, b.VOCAB), np.nan, np.float32)
        wide = np.full_like(hidden, np.nan)
        coords = np.repeat(np.arange(position + count, dtype=np.int32)[:, None], 3, axis=1)
        if media:
            # Repeated time with varying H/W exercises multimodal QSA ordering.
            coords[:, 0] //= 3
            coords[:, 1] %= 3
        mrope = np.ascontiguousarray(coords[position:])
        sections = np.array([2, 1, 1, 0], np.int32)
        values = [C.addressof(t.ffn), C.addressof(t.gdn), C.addressof(t.attn), C.addressof(t.kinds),
            0, 2, residual.ctypes.data, mask.ctypes.data, b.H, b.HC, b.LOW, count,
            t.GK, t.GV, t.GKH, t.GVH, t.CONV, b.HD, b.NH, b.NK, b.capacity, position + count, position,
            b.c.n_rot, b.c.rope_base, b.c.rope_scale, b.c.attn_scale, b.EXP, b.USED,
            b.FF // degree if parallel else b.FF, b.SH // degree if parallel else b.SH,
            b.eps, 17, 0, C.addressof(b.o), out.ctypes.data, C.addressof(t.p), 0, e.ctypes.data,
            mrope.ctypes.data if media else None, sections.ctypes.data if media else None, position, 0,
            wide.ctypes.data, count, C.addressof(qsa) if sparse else None, coords.ctypes.data if sparse else None,
            position + count if sparse else 0]
        if parallel:
            plans = (P * degree)()
            for rank in range(degree):
                values[0] = C.addressof(descriptors[rank]); values[len(signature) - 1] = rank
                # Production computes/downloads the head and speculative taps only
                # on rank zero; the final segment is intentionally asymmetric.
                values[34] = C.addressof(b.o) if rank == 0 else None
                values[35] = out.ctypes.data if rank == 0 else None
                values[len(signature)] = wide.ctypes.data if rank == 0 else None
                plan = P()
                assert tp(*values, C.byref(plan)), error()
                plans[rank] = plan
            assert execute(plans, degree), error()
        else:
            assert normal(*values), error()
        return wide, out

    for sparse, media in ((False, False), (True, False), (True, True)):
        label = f"qsa{int(sparse)}_media{int(media)}"
        for count in widths:
            reset(); sh, sl = run(False, count, 0, 700 + count, sparse, media)
            reset(); ph, pl = run(True, count, 0, 700 + count, sparse, media)
            check(f"{label}_prefill{count}_hidden", ph, sh)
            check(f"{label}_prefill{count}_logits", pl, sl)
            # Same topology at a new position exercises cached-plan replay.
            ah, al = run(True, count, count, 800 + count, sparse, media)
            reset(); run(False, count, 0, 700 + count, sparse, media)
            rh, rl = run(False, count, count, 800 + count, sparse, media)
            check(f"{label}_replay{count}_hidden", ah, rh)
            check(f"{label}_replay{count}_logits", al, rl)

        reset(); run(True, 5, 0, 901, sparse, media)
        cold_h, cold_l = run(True, args.rollback_width, 5, 902, sparse, media)
        reset(); run(True, 5, 0, 901, sparse, media)
        keys = (P * (2 * degree))(*[key for rank in range(degree) for key in t.keys])
        ranks = (I * (2 * degree))(*[rank for rank in range(degree) for key in t.keys])
        snapshot = create(keys, ranks, len(keys), C.addressof(t.attn), C.addressof(t.gdn), C.addressof(t.p))
        assert snapshot, error()
        try:
            assert capture(snapshot), error()
            run(True, args.rollback_width, 5, 999, sparse, media)
            assert restore(snapshot), error()
            restored_h, restored_l = run(True, args.rollback_width, 5, 902, sparse, media)
            check(label + "_rollback_hidden", restored_h, cold_h)
            check(label + "_rollback_logits", restored_l, cold_l)
        finally:
            free(snapshot)
    reset()
    report["status"] = "passed"
    args.report.write_text(json.dumps(report, indent=2))
    print(f"PASS: {len(report['checks'])} parity/replay/rollback checks; {args.backend}, {degree} ranks; loopback={not bool(args.devices)}")


if __name__ == "__main__":
    _library = None
    try:
        main()
    finally:
        if _library is not None:
            _library.TSGgml_Shutdown.argtypes = []
            _library.TSGgml_Shutdown.restype = None
            _library.TSGgml_Shutdown()
