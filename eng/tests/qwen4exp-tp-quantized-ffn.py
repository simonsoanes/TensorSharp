#!/usr/bin/env python3
"""Tensor-sliced Qwen FFN quantized-kernel probe with independent FP64 arithmetic.

Uses the real intermediate width640, IQ4_XS gate/up and Q8_0 down. Slices bytes
of the once-quantized source, never requantizes shards. Reports both native
unsharded-versus-TP error and each path's error against dequantized FP64 weights.
The FP64 oracle does not model CUDA activation quantization; its errors are
reported separately, never used to excuse a sharding discrepancy.
"""
import argparse
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import numpy as np

spec = importlib.util.spec_from_file_location("op", Path(__file__).with_name("qwen4exp-mtp-operator.py"))
op = importlib.util.module_from_spec(spec); spec.loader.exec_module(op)
P,I,L,F = C.c_void_p,C.c_int,C.c_int64,C.c_float
_library = None


def main():
    global _library
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--backend", choices=("CPU","CUDA"), default="CPU")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--tokens", default="1,2,3,4,5,6,7,8,17,31,128", help="Comma-separated batch widths, including every supported verify width")
    ap.add_argument("--float", action="store_true")
    ap.add_argument("--require-bitwise", action="store_true",
        help="Fail on any F32 bit difference between the unsharded and TP outputs")
    ap.add_argument("--gate-type", type=int, choices=(21,23), default=23,
        help="Routed gate/up type: 21 IQ3_S or23 IQ4_XS")
    ap.add_argument("--down-type", type=int, choices=(8,20), default=8,
        help="Routed down type: 8 Q8_0 or20 IQ4_NL")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--used", type=int, default=4)
    ap.add_argument("--shared-q8", action="store_true", help="Use Q8_0 for all shared FFN matrices")
    args = ap.parse_args()
    degree = args.degree
    H,HC,LOW,FF,EXPERTS,USED = args.hidden,4,16,640,args.experts,args.used
    if EXPERTS < 1 or USED < 1 or USED > EXPERTS:
        ap.error("Use 1 <= --used <= --experts")
    lib = C.CDLL(str(args.library.resolve()))
    _library = lib
    err=lib.TSGgml_GetLastError; err.restype=C.c_char_p
    if args.backend == "CPU":
        init=lib.TSGgml_TensorParallelInitLoopback;init.argtypes=[I,I];init.restype=I
        assert init(2,degree),err()
    else:
        init=lib.TSGgml_TensorParallelInit;init.argtypes=[I,P,I,I];init.restype=I
        assert init(3,(I*degree)(*map(int,args.devices.split(','))),degree,0),err()
        if not args.float:
            capability=lib.TSGgml_Qwen4ExpTpWeightSupported
            capability.argtypes=[I,I,I,I,I,I];capability.restype=I
            assert capability(args.gate_type,H,FF,EXPERTS,degree,1),err()
            assert capability(args.down_type,FF,H,EXPERTS,degree,0),err()
            assert not capability(13,H,FF,EXPERTS,degree,1), "Unqualified Q5_K must refuse before data loading"
            assert not capability(8,H,641,EXPERTS,degree,1), "Nondivisible row tail must refuse before data loading"
    quant=lib.ggml_quantize_chunk;quant.argtypes=[I,P,P,L,L,L,P];quant.restype=C.c_size_t
    row_size=lib.ggml_row_size;row_size.argtypes=[I,L];row_size.restype=C.c_size_t
    dequant=lib.TSGgml_DequantizeToF32;dequant.argtypes=[I,P,L,P];dequant.restype=I
    signature=[P,P,P,P,I,I,P,P]+[I]*16+[F]*3+[I]*4+[F,I,I,P,P,P,I,P,P,P,I,I]
    normal=lib.TSGgml_Qwen4ExpTokenSpanQsa;normal.argtypes=signature+[P,I,P,P,I];normal.restype=I
    tp=lib.TSGgml_Qwen4ExpTokenSpanTp;tp.argtypes=normal.argtypes+[C.POINTER(P)];tp.restype=I
    execute=lib.TSGgml_TensorParallelExecutePlans;execute.argtypes=[P,I];execute.restype=I
    release=lib.TSGgml_Qwen4ExpReleaseAllSeqState;release.argtypes=[];release.restype=None
    rng=np.random.default_rng(93017)
    desc=op.Ffn();pinned=[];weights={};raw={};types={}
    matrices={"hc_down":(LOW,HC*H),"hc_up":(HC*H,LOW),"hc_inject":(HC,HC*H),
        "router":(EXPERTS,H),"gate_exps":(EXPERTS,FF,H),"up_exps":(EXPERTS,FF,H),
        "down_exps":(EXPERTS,H,FF),"sh_gate":(FF,H),"sh_up":(FF,H),"sh_down":(H,FF)}
    for field,shape in matrices.items():
        value=np.asarray(rng.normal(0,1/np.sqrt(shape[-1]),shape),np.float32)
        kind=0 if args.float else (args.gate_type if field in ("gate_exps","up_exps")
            else (8 if args.shared_q8 else 23) if field in ("sh_gate","sh_up")
            else args.down_type if field == "down_exps"
            else 8 if field == "sh_down" else 0)
        size=int(row_size(kind,shape[-1])); packed=np.empty((*shape[:-1],size),np.uint8)
        assert quant(kind,value.ctypes.data,packed.ctypes.data,0,value.size//shape[-1],shape[-1],None)==packed.nbytes
        decoded=np.empty_like(value);assert dequant(kind,packed.ctypes.data,value.size,decoded.ctypes.data)==0
        weights[field]=decoded.astype(np.float64);raw[field]=packed;types[field]=kind;pinned.append(packed)
        setattr(desc,field,packed.ctypes.data);setattr(desc,field+'_bytes',packed.nbytes);setattr(desc,field+'_type',kind)
    for field,shape in (("hc_norm",(HC,H)),("sh_gate_inp",(H,))):
        value=np.ones(shape,np.float32) if field=="hc_norm" else np.asarray(rng.normal(0,1/np.sqrt(H),shape),np.float32)
        weights[field]=value.astype(np.float64);pinned.append(value);setattr(desc,field,value.ctypes.data)
    full=(op.Ffn*1)(desc); shards=[];shard_weights=[]
    for rank in range(degree):
        sd=op.Ffn.from_buffer_copy(desc);sw={}
        for field,axis in (("gate_exps",1),("up_exps",1),("down_exps",1),("sh_gate",0),("sh_up",0),("sh_down",0)):
            rows=raw[field].shape[axis];first=rank*rows//degree;end=(rank+1)*rows//degree
            # The production loader retains complete MMQ output tiles around
            # each logical gate/up slice; native execution crops the overlap.
            if not args.float and field in ("gate_exps","up_exps","sh_gate","sh_up"):
                first=first//128*128;end=(end+127)//128*128
            selection=[slice(None)]*raw[field].ndim;selection[axis]=slice(first,end)
            value=np.ascontiguousarray(raw[field][tuple(selection)]);pinned.append(value)
            setattr(sd,field,value.ctypes.data);setattr(sd,field+'_bytes',value.nbytes)
            sw[field]=np.split(weights[field],degree,axis=axis)[rank]
        shards.append((op.Ffn*1)(sd));shard_weights.append(sw)
    kinds=(C.c_uint8*1)(1);dummy=C.create_string_buffer(1024)
    sigmoid=lambda x:1/(1+np.exp(-x))
    def oracle(res, partition=False):
        T=len(res);xn=res/np.sqrt(np.mean(res*res,axis=2,keepdims=True)+1e-6)*weights['hc_norm'];xn=xn.reshape(T,HC*H)
        lo=xn@weights['hc_down'].T/HC;lo=lo*sigmoid(lo)
        mixed=(xn*sigmoid(lo@weights['hc_up'].T)).reshape(T,HC,H).mean(axis=1)
        injection=2*sigmoid(xn@weights['hc_inject'].T/HC)
        router=mixed@weights['router'].T; probs=np.exp(router-router.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True)
        ids=np.argsort(-probs,axis=1)[:,:USED];scores=np.take_along_axis(probs,ids,axis=1);scores/=scores.sum(axis=1,keepdims=True)
        def partial(w):
            routed=np.zeros((T,w['down_exps'].shape[1]))
            for t in range(T):
                for k,e in enumerate(ids[t]):
                    gate=w['gate_exps'][e]@mixed[t];up=w['up_exps'][e]@mixed[t]
                    routed[t]+=scores[t,k]*(w['down_exps'][e]@(gate*sigmoid(gate)*up))
            gate=mixed@w['sh_gate'].T;up=mixed@w['sh_up'].T
            return routed+(gate*sigmoid(gate)*up)@w['sh_down'].T*sigmoid(mixed@weights['sh_gate_inp'])[:,None]
        # The rank's down rows consume the gathered, full-width activation.
        ffn=np.concatenate([partial(dict(weights,down_exps=w['down_exps'],sh_down=w['sh_down']))
            for w in shard_weights],axis=1) if partition else partial(weights)
        return res+injection[:,:,None]*ffn[:,None,:]
    def run(parallel,res):
        T=len(res);out=res.copy();plans=(P*degree)()
        values=[C.addressof(full),C.addressof(dummy),C.addressof(dummy),C.addressof(kinds),0,1,out.ctypes.data,None,
            H,HC,LOW,T,4,4,1,2,3,8,4,2,256,T,0,4,10000.,1.,1.,EXPERTS,USED,FF//degree if parallel else FF,FF//degree if parallel else FF,
            1e-6,3,1,None,None,None,-1,None,None,None,0,0,None,1,None,None,0]
        if parallel:
            for rank in range(degree):
                values[0]=C.addressof(shards[rank]);values[len(signature)-1]=rank;plan=P()
                assert tp(*values,C.byref(plan)),err();plans[rank]=plan
            assert execute(plans,degree),err()
            if os.environ.get("TS_Q4E_NODE_DUMP"):
                dump = lib.TSGgml_Qwen4ExpTestDumpTpPlans
                dump.argtypes = [P,I,I,I,I,I,I,I]; dump.restype = None
                dump(plans,degree,0,1,T,0,1,T)
        else: assert normal(*values),err()
        return out
    report=dict(hidden=H,ff=FF,experts=EXPERTS,used=USED,degree=degree,types=types,backend=args.backend,
        native_sha256=op.sha(args.library),gate=dict(require_bitwise=args.require_bitwise,atol=2e-5,rtol=2e-5),checks=[])
    def stats(x,y):
        diff=x.astype(np.float64)-y.astype(np.float64)
        return dict(max_abs=float(np.max(np.abs(diff))),relative_l2=float(np.linalg.norm(diff)/np.linalg.norm(y)))
    passed=True
    for T in map(int,args.tokens.split(',')):
        residual=np.asarray(rng.normal(0,.4,(T,HC,H)),np.float32)
        plain=run(False,residual);parallel=run(True,residual)
        bitwise_equal = bool(np.array_equal(plain.view(np.uint32), parallel.view(np.uint32)))
        reference=oracle(residual.astype(np.float64));partitioned=oracle(residual.astype(np.float64),True)
        passed &= bool(all(np.isfinite(x).all() for x in (plain,parallel,reference,partitioned))
            and np.allclose(plain,parallel,atol=2e-5,rtol=2e-5)
            and (not args.require_bitwise or bitwise_equal)
            and np.allclose(reference,partitioned,atol=1e-12,rtol=1e-12))
        report['checks'].append(dict(tokens=T,plain_vs_tp=dict(stats(parallel,plain),bitwise_equal=bitwise_equal),plain_vs_f64=stats(plain,reference),tp_vs_f64=stats(parallel,reference),f64_partition=stats(partitioned,reference)))
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2))
        print(json.dumps(report['checks'][-1]),flush=True)
    release();report['passed']=passed;args.report.write_text(json.dumps(report,indent=2))
    return 0 if passed else 1

if __name__=='__main__':
    try:
        result = main()
    finally:
        if _library is not None:
            _library.TSGgml_Shutdown.argtypes = []
            _library.TSGgml_Shutdown.restype = None
            _library.TSGgml_Shutdown()
    raise SystemExit(result)
