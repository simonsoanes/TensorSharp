#!/usr/bin/env python3
"""Compare named FFN nodes from a reference and two TP TS_Q4E_NODE_DUMP files."""
import argparse
import json
from pathlib import Path
import struct
import numpy as np


def load(path):
    data = path.read_bytes()
    offset = 0
    nodes = {}
    values_by_index = {}
    while offset < len(data):
        magic, index, op, kind = struct.unpack_from("<4i", data, offset)
        assert magic == 0x444E3451
        offset += 16
        shape = struct.unpack_from("<4q", data, offset); offset += 32
        sources = struct.unpack_from("<10i", data, offset); offset += 40
        name = data[offset:offset+64].split(b"\0")[0].decode(); offset += 64
        count, = struct.unpack_from("<q", data, offset); offset += 8
        values = None
        if count:
            values = np.frombuffer(data, dtype="<f4", count=count, offset=offset)
            values_by_index[index] = values
        elif name.startswith("q4e.ffn.shared_") and sources[0] in values_by_index:
            # Dense TP strips end in a contiguous reshape. The dump omits
            # view payloads, so read the identical bytes from their producer.
            parent = values_by_index[sources[0]]
            if parent.size == np.prod(shape): values = parent
        if values is not None and name.startswith("q4e.ffn."):
            values = values.reshape(shape[::-1])
            nodes[name] = values.astype(np.float64)
        offset += 4*count
    return nodes


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path)
parser.add_argument("rank0", type=Path)
parser.add_argument("rank1", type=Path, nargs="?", help="Omit to compare two unsharded executions")
args = parser.parse_args()
reference, rank0 = [load(path) for path in (args.reference,args.rank0)]
rank1 = load(args.rank1) if args.rank1 else None
for name, expected in reference.items():
    x = rank0[name]
    y = rank1[name] if rank1 else None
    if rank1 is None:
        actual, mode = x, "unsharded"
    elif x.shape != expected.shape:
        actual = np.concatenate((x,y),axis=-1)
        mode = "concat_channels"
    elif name.endswith("_down"):
        actual = x+y
        mode = "sum_partials"
    else:
        actual = x
        mode = "replicated_rank0"
    assert expected.shape == actual.shape, (name,expected.shape,actual.shape)
    delta = actual-expected
    print(json.dumps(dict(node=name,mode=mode,shape=expected.shape,
        max_abs=float(np.max(np.abs(delta))),
        relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(expected),1e-300)),
        bit_equal=bool(np.array_equal(actual,expected)))))
