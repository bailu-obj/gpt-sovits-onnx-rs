#!/usr/bin/env python3
"""ORT node profiler for GPT-SoVITS ONNX stages (BERT / T2S FS / T2S AR / VITS).

Ranks nodes by wall time and groups layout/copy-family ops separately from compute.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit("onnxruntime required") from exc

REPO = Path(__file__).resolve().parents[1]
LAYOUT_HINTS = (
    "/transpose",
    "transpose_",
    "_transpose",
    "/reshape",
    "reshape_",
    "/concat",
    "concat_",
    "/slice",
    "slice_",
    "/gather",
    "gather_",
    "/expand",
    "/tile",
    "/squeeze",
    "/unsqueeze",
    "/cast",
    "memcpy",
)


def is_layout_node(name: str) -> bool:
    n = name.lower().replace("_kernel_time", "")
    # Avoid matching compute ops like ConvTranspose.
    if "convtranspose" in n or "transposec" in n:
        return False
    return any(h in n for h in LAYOUT_HINTS)


def profile_session(
    model_path: Path,
    feeds: dict[str, np.ndarray],
    *,
    warmup: int,
    top_k: int,
) -> dict[str, Any]:
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    for _ in range(warmup):
        sess.run(None, feeds)
    t0 = time.perf_counter()
    sess.run(None, feeds)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    profile_path = Path(sess.end_profiling())
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    nodes: list[dict[str, Any]] = []
    layout_ms = 0.0
    compute_ms = 0.0
    for ev in events:
        if ev.get("cat") != "Node":
            continue
        dur_ms = float(ev.get("dur", 0.0)) / 1000.0
        name = ev.get("name", "")
        layout = is_layout_node(name)
        if layout:
            layout_ms += dur_ms
        else:
            compute_ms += dur_ms
        nodes.append({"name": name, "ms": round(dur_ms, 4), "layout_family": layout})
    nodes.sort(key=lambda x: x["ms"], reverse=True)
    return {
        "model": str(model_path),
        "wall_ms": round(wall_ms, 3),
        "layout_ms": round(layout_ms, 3),
        "compute_ms": round(compute_ms, 3),
        "top_nodes": nodes[:top_k],
        "profile_file": str(profile_path),
    }


def find_prefix(bundle: Path) -> str:
    for p in bundle.glob("*_vits.onnx"):
        return p.name[: -len("_vits.onnx")]
    # split-only bundles may expose *_vits_decode.onnx
    for p in bundle.glob("*_vits_decode.onnx"):
        return p.name[: -len("_vits_decode.onnx")]
    raise SystemExit(f"No vits model in {bundle}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--pred-len", type=int, default=80)
    ap.add_argument("--text-len", type=int, default=40)
    ap.add_argument("--x-len", type=int, default=60)
    ap.add_argument("--kv-len", type=int, default=80)
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    prefix = find_prefix(bundle)

    results: dict[str, Any] = {
        "bundle": str(bundle),
        "prefix": prefix,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }

    # Prefer split decode (no STFT/ref) when present; also profile mono for comparison.
    decode = bundle / f"{prefix}_vits_decode.onnx"
    if decode.is_file():
        import onnx

        g = onnx.load(str(decode), load_external_data=False).graph
        ge_shape = [1, 256, 1]
        for inp in g.input:
            if inp.name == "ge":
                dims = [d.dim_value or 0 for d in inp.type.tensor_type.shape.dim]
                ge_shape = [d if d else 1 for d in dims] or ge_shape
                break
        feeds = {
            "text_seq": np.random.randint(0, 300, size=(1, args.text_len), dtype=np.int64),
            "pred_semantic": np.random.randint(0, 1024, size=(1, 1, args.pred_len), dtype=np.int64),
            "ge": np.random.randn(*ge_shape).astype(np.float32),
            "noise_scale": np.array([0.5], dtype=np.float32),
            "speed": np.array([1.0], dtype=np.float32),
        }
        results["stages"]["vits_decode"] = profile_session(
            decode, feeds, warmup=args.warmup, top_k=args.top_k
        )

    vits = bundle / f"{prefix}_vits.onnx"
    if vits.is_file():
        feeds = {
            "text_seq": np.random.randint(0, 300, size=(1, args.text_len), dtype=np.int64),
            "pred_semantic": np.random.randint(0, 1024, size=(1, 1, args.pred_len), dtype=np.int64),
            "ref_audio": np.random.randn(1, 147_200).astype(np.float32),
        }
        results["stages"]["vits"] = profile_session(
            vits, feeds, warmup=args.warmup, top_k=args.top_k
        )

    fs = bundle / f"{prefix}_t2s_fs_decoder.onnx"
    if fs.is_file():
        # Support both [1,1024,T] legacy and [1,T,1024] native layouts.
        import onnx

        g = onnx.load(str(fs), load_external_data=False).graph
        bert_dims = None
        for inp in g.input:
            if inp.name == "bert":
                bert_dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
                break
        if bert_dims and len(bert_dims) >= 3 and bert_dims[1] == 1024:
            bert = np.random.randn(1, 1024, args.x_len).astype(np.float32)
        else:
            bert = np.random.randn(1, args.x_len, 1024).astype(np.float32)
        feeds = {
            "x": np.random.randint(0, 300, size=(1, args.x_len), dtype=np.int64),
            "prompts": np.random.randint(0, 1024, size=(1, 20), dtype=np.int64),
            "bert": bert,
        }
        results["stages"]["t2s_fs"] = profile_session(
            fs, feeds, warmup=args.warmup, top_k=args.top_k
        )

    sd = bundle / f"{prefix}_t2s_s_decoder.onnx"
    if sd.is_file():
        import onnx

        g = onnx.load(str(sd), load_external_data=False).graph
        n_layers = sum(1 for i in g.input if i.name.startswith("ik_cache_"))
        # Detect legacy [B,T,H] vs head-major [B,H,T,D] from ik_cache_0 rank.
        kv_rank = 3
        n_heads, head_dim = 16, 32
        for inp in g.input:
            if inp.name != "ik_cache_0":
                continue
            dims = [d.dim_value or 0 for d in inp.type.tensor_type.shape.dim]
            kv_rank = len(dims)
            if kv_rank >= 4:
                n_heads = dims[1] or 16
                head_dim = dims[3] or 32
            break
        feeds: dict[str, Any] = {
            "iy": np.random.randint(0, 1024, size=(1, args.kv_len + 1), dtype=np.int64),
            "y_len": np.array([20], dtype=np.int64),
            "idx": np.array([args.kv_len - 20], dtype=np.int64),
        }
        for i in range(n_layers):
            if kv_rank >= 4:
                shape = (1, n_heads, args.kv_len, head_dim)
            else:
                shape = (1, args.kv_len, 512)
            feeds[f"ik_cache_{i}"] = np.random.randn(*shape).astype(np.float32)
            feeds[f"iv_cache_{i}"] = np.random.randn(*shape).astype(np.float32)
        results["stages"]["t2s_s"] = profile_session(
            sd, feeds, warmup=args.warmup, top_k=args.top_k
        )
        results["t2s_s_kv_layout"] = (
            "head_major" if kv_rank >= 4 else "legacy_seq_major"
        )

    bert = bundle / "bert.onnx"
    if bert.is_file():
        feeds = {
            "input_ids": np.random.randint(0, 1000, size=(1, 32), dtype=np.int64),
            "attention_mask": np.ones((1, 32), dtype=np.int64),
            "token_type_ids": np.zeros((1, 32), dtype=np.int64),
        }
        results["stages"]["bert"] = profile_session(
            bert, feeds, warmup=args.warmup, top_k=args.top_k
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    for stage, data in results["stages"].items():
        print(
            f"  {stage}: wall={data['wall_ms']:.1f}ms layout={data['layout_ms']:.1f}ms "
            f"compute={data['compute_ms']:.1f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
