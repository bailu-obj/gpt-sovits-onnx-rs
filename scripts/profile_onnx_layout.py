#!/usr/bin/env python3
"""Read-only ONNX bundle profiler for layout/copy-heavy operators.

Reports Transpose/Reshape/Concat/Split/Slice/Gather/Expand/Tile/Cast counts,
repeated transpose pairs, initializer duplication, and dynamic-shape boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
from onnx import numpy_helper, shape_inference

LAYOUT_OPS = {
    "Transpose",
    "Reshape",
    "Concat",
    "Split",
    "Slice",
    "Gather",
    "Expand",
    "Tile",
    "Squeeze",
    "Unsqueeze",
    "Flatten",
    "Shape",
    "Cast",
    "ScatterND",
    "Resize",
}


def tensor_nbytes(tensor: onnx.TensorProto) -> int:
    try:
        arr = numpy_helper.to_array(tensor)
        return int(arr.nbytes)
    except Exception:
        return 0


def tensor_fingerprint(tensor: onnx.TensorProto) -> str:
    try:
        arr = numpy_helper.to_array(tensor)
        return hashlib.sha1(arr.tobytes()).hexdigest()
    except Exception:
        return f"unreadable:{tensor.name}"


def dims_of(value_info: onnx.ValueInfoProto) -> list[Any]:
    dims: list[Any] = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.dim_value:
            dims.append(int(d.dim_value))
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims


def analyze_model(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # pragma: no cover - best effort
        shape_err = str(exc)
    else:
        shape_err = None

    graph = model.graph
    op_counts = Counter(n.op_type for n in graph.node)
    layout_counts = {op: op_counts[op] for op in sorted(LAYOUT_OPS) if op_counts[op]}

    producer_of: dict[str, onnx.NodeProto] = {}
    consumers_of: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in graph.node:
        for out in node.output:
            producer_of[out] = node
        for inp in node.input:
            if inp:
                consumers_of[inp].append(node)

    transpose_pairs: list[dict[str, str]] = []
    for node in graph.node:
        if node.op_type != "Transpose":
            continue
        for out in node.output:
            for cons in consumers_of.get(out, []):
                if cons.op_type == "Transpose":
                    transpose_pairs.append(
                        {
                            "a": node.name or node.output[0],
                            "b": cons.name or cons.output[0],
                        }
                    )

    init_bytes = 0
    fingerprints: dict[str, list[str]] = defaultdict(list)
    for init in graph.initializer:
        nbytes = tensor_nbytes(init)
        init_bytes += nbytes
        fingerprints[tensor_fingerprint(init)].append(init.name)
    duplicated = {
        fp: names for fp, names in fingerprints.items() if len(names) > 1
    }
    dup_bytes = 0
    for fp, names in duplicated.items():
        # count all but one as duplicate cost
        sample = next(i for i in graph.initializer if i.name == names[0])
        dup_bytes += tensor_nbytes(sample) * (len(names) - 1)

    dynamic_ios: list[dict[str, Any]] = []
    for vi in list(graph.input) + list(graph.output):
        dims = dims_of(vi)
        if any(not isinstance(d, int) for d in dims):
            dynamic_ios.append({"name": vi.name, "dims": dims})

    return {
        "path": str(path),
        "size_mib": round(path.stat().st_size / (1024 * 1024), 2),
        "nodes": len(graph.node),
        "op_counts_top": dict(op_counts.most_common(20)),
        "layout_ops": layout_counts,
        "layout_op_total": int(sum(layout_counts.values())),
        "transpose_pairs": transpose_pairs[:50],
        "transpose_pair_count": len(transpose_pairs),
        "initializer_bytes": init_bytes,
        "initializer_mib": round(init_bytes / (1024 * 1024), 2),
        "duplicated_initializer_groups": len(duplicated),
        "duplicated_initializer_extra_mib": round(dup_bytes / (1024 * 1024), 2),
        "dynamic_ios": dynamic_ios,
        "shape_inference_error": shape_err,
    }


def analyze_bundle(bundle: Path) -> dict[str, Any]:
    models = sorted(bundle.rglob("*.onnx"))
    per_model = [analyze_model(p) for p in models]
    layout_totals: Counter[str] = Counter()
    for m in per_model:
        layout_totals.update(m["layout_ops"])
    return {
        "bundle": str(bundle),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_count": len(per_model),
        "total_size_mib": round(sum(m["size_mib"] for m in per_model), 2),
        "layout_ops_total": dict(layout_totals),
        "models": per_model,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_bundle(args.bundle.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(
        f"models={report['model_count']} size_mib={report['total_size_mib']} "
        f"layout={report['layout_ops_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
