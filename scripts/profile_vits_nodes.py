#!/usr/bin/env python3
"""Profile GPT-SoVITS VITS ONNX node times with ORT session profiling.

Groups node kernel times into reference conditioning (STFT / ref_enc / ge path)
vs semantic encode / flow / vocoder. Writes a JSON summary under target/ or tmp/.

Requires onnxruntime (see scripts/requirements.txt). Falls back to documenting a
Rust re-run path when ORT is unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - env without ORT
    ort = None  # type: ignore[assignment,misc]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models/gpt-sovits-onnx-custom/quant/custom_vits.onnx"
DEFAULT_REF = REPO_ROOT / "models/gpt-sovits-onnx-custom/quant/ref.wav"
DEFAULT_OUT_DIR = REPO_ROOT / "target/vits_profile"

# Fixed ~4.6 s reference @ 32 kHz (matches Rust ref_audio_32k resample target).
DEFAULT_REF_SAMPLES_32K = 147_200


@dataclass
class ScenarioResult:
    pred_semantic_len: int
    text_seq_len: int
    ref_audio_samples: int
    wall_ms: float
    node_sum_ms: float
    groups_ms: dict[str, float] = field(default_factory=dict)
    groups_pct_of_wall: dict[str, float] = field(default_factory=dict)
    groups_pct_of_nodes: dict[str, float] = field(default_factory=dict)
    op_family_ms: dict[str, float] = field(default_factory=dict)
    top_nodes: list[dict[str, float | str]] = field(default_factory=list)
    profile_file: str | None = None


@dataclass
class ProfileReport:
    model_path: str
    model_inputs: list[str]
    has_sv_emb: bool
    ref_audio_source: str
    scenarios: list[ScenarioResult]
    decision_threshold_pct: float
    recommendation: str
    generated_at: str


def classify_node(name: str) -> str:
    """Map ORT node names to coarse VITS stage buckets."""
    n = name.replace("_kernel_time", "").lower()
    if any(k in n for k in ("stft", "spectrogram", "/ref_enc", "/sv_emb", "ge_to512", "/prelu")):
        return "ref_conditioning"
    if "/flow/" in n or n.startswith("/vq_model/flow"):
        return "flow"
    if "/dec/" in n or n.startswith("/vq_model/dec"):
        return "vocoder"
    if "quantizer" in n or "/enc_p/" in n:
        return "semantic_encode"
    return "other"


def load_ref_audio(path: Path | None, ref_samples: int) -> tuple[np.ndarray, str]:
    if path is None:
        return np.random.randn(1, ref_samples).astype(np.float32), "random"
    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit(
            "soundfile is required to load --ref-audio; omit the flag to use random noise."
        ) from exc

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Bundled ref.wav is 16 kHz; VITS expects model-rate audio (32 kHz in Rust bundles).
    if sr != 32_000:
        try:
            import torchaudio
        except ImportError as exc:
            raise SystemExit(
                f"ref audio is {sr} Hz; install torchaudio or resample to 32 kHz offline."
            ) from exc
        import torch

        wav = torch.from_numpy(audio).unsqueeze(0)
        audio = (
            torchaudio.functional.resample(wav, sr, 32_000).squeeze(0).numpy().astype(np.float32)
        )
    if len(audio) > ref_samples:
        audio = audio[:ref_samples]
    elif len(audio) < ref_samples:
        pad = np.zeros(ref_samples - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, pad])
    return audio.reshape(1, -1), str(path)


def model_input_names(model_path: Path) -> list[str]:
    import onnx

    graph = onnx.load(str(model_path), load_external_data=False).graph
    return [i.name for i in graph.input]


def model_input_shape(model_path: Path, name: str, default: list[int]) -> list[int]:
    import onnx

    graph = onnx.load(str(model_path), load_external_data=False).graph
    for inp in graph.input:
        if inp.name != name:
            continue
        out: list[int] = []
        for i, d in enumerate(inp.type.tensor_type.shape.dim):
            if d.dim_value:
                out.append(int(d.dim_value))
            elif i < len(default):
                out.append(default[i])
            else:
                out.append(1)
        return out or list(default)
    return list(default)


def _op_family(node_name: str) -> str:
    """Coarse kernel family from ORT profile node name."""
    n = node_name.replace("_kernel_time", "").lower()
    if "convtranspose" in n:
        return "ConvTranspose"
    if "conv" in n:
        return "Conv"
    if "matmul" in n or "gemm" in n:
        return "MatMul"
    if "transpose" in n:
        return "Transpose"
    if "reshape" in n or "squeeze" in n or "unsqueeze" in n:
        return "Reshape"
    if "concat" in n:
        return "Concat"
    if "slice" in n:
        return "Slice"
    return "Other"


def aggregate_profile(
    profile_path: Path,
) -> tuple[dict[str, float], float, dict[str, float], list[dict[str, float | str]]]:
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    groups: dict[str, float] = defaultdict(float)
    by_op: dict[str, float] = defaultdict(float)
    nodes: list[tuple[float, str]] = []
    for event in data:
        if event.get("cat") != "Node":
            continue
        dur_us = float(event.get("dur", 0.0))
        name = event.get("name", "")
        groups[classify_node(name)] += dur_us
        by_op[_op_family(name)] += dur_us
        nodes.append((dur_us, name))
    total_us = sum(groups.values())
    top_nodes = [
        {"ms": round(us / 1000.0, 3), "name": name, "op_family": _op_family(name)}
        for us, name in sorted(nodes, key=lambda x: -x[0])[:25]
    ]
    return (
        {k: v / 1000.0 for k, v in groups.items()},
        total_us / 1000.0,
        {k: v / 1000.0 for k, v in by_op.items()},
        top_nodes,
    )


def run_scenario(
    model_path: Path,
    *,
    pred_len: int,
    text_len: int,
    ref_audio: np.ndarray | None,
    has_sv_emb: bool,
    warmup: int,
    profile_file_dir: Path | None,
    is_decode: bool = False,
    ge: np.ndarray | None = None,
) -> ScenarioResult:
    if ort is None:
        raise SystemExit(
            "onnxruntime is not installed. Install scripts/requirements.txt or use:\n"
            "  cargo run --release --example gpt_sovits_demo -- --model-path <bundle> --run-count 10"
        )

    text_seq = np.random.randint(0, 300, size=(1, text_len), dtype=np.int64)
    pred_sem = np.random.randint(0, 1024, size=(1, 1, pred_len), dtype=np.int64)
    feeds: dict[str, Any] = {
        "text_seq": text_seq,
        "pred_semantic": pred_sem,
    }
    if is_decode:
        feeds["ge"] = ge if ge is not None else np.random.randn(1, 512, 1).astype(np.float32)
        feeds["noise_scale"] = np.array([0.5], dtype=np.float32)
        feeds["speed"] = np.array([1.0], dtype=np.float32)
        ref_samples = 0
    else:
        assert ref_audio is not None
        feeds["ref_audio"] = ref_audio
        ref_samples = int(ref_audio.shape[1])
        if has_sv_emb:
            feeds["sv_emb"] = np.random.randn(1, 20_480).astype(np.float32)

    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_file_dir is not None:
        profile_file_dir.mkdir(parents=True, exist_ok=True)
        opts.profile_file_prefix = str(profile_file_dir / "ort_vits")

    session = ort.InferenceSession(
        str(model_path),
        opts,
        providers=["CPUExecutionProvider"],
    )

    for _ in range(warmup):
        session.run(None, feeds)

    t0 = time.perf_counter()
    session.run(None, feeds)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    raw_profile = session.end_profiling()
    groups_ms, node_sum_ms, op_family_ms, top_nodes = aggregate_profile(Path(raw_profile))

    def pct(part: float, whole: float) -> float:
        return round(100.0 * part / whole, 2) if whole > 0 else 0.0

    return ScenarioResult(
        pred_semantic_len=pred_len,
        text_seq_len=text_len,
        ref_audio_samples=ref_samples,
        wall_ms=round(wall_ms, 2),
        node_sum_ms=round(node_sum_ms, 2),
        groups_ms={k: round(v, 3) for k, v in sorted(groups_ms.items())},
        groups_pct_of_wall={
            k: pct(groups_ms.get(k, 0.0), wall_ms) for k in sorted(groups_ms)
        },
        groups_pct_of_nodes={
            k: pct(groups_ms.get(k, 0.0), node_sum_ms) for k in sorted(groups_ms)
        },
        op_family_ms={k: round(v, 3) for k, v in sorted(op_family_ms.items())},
        top_nodes=top_nodes,
        profile_file=raw_profile,
    )


def decide(report: ProfileReport, threshold_pct: float) -> str:
    ref_pcts = [
        s.groups_pct_of_wall.get("ref_conditioning", 0.0)
        for s in report.scenarios
    ]
    max_ref = max(ref_pcts) if ref_pcts else 0.0
    if max_ref >= threshold_pct:
        return (
            f"split_recommended: ref_conditioning reached {max_ref:.1f}% of wall time "
            f"(threshold {threshold_pct:.0f}%)"
        )
    return (
        f"defer_split: ref_conditioning peaked at {max_ref:.1f}% of wall time "
        f"(threshold {threshold_pct:.0f}%)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile VITS ONNX node times")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to *_vits.onnx (default: quantized custom bundle)",
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=DEFAULT_REF if DEFAULT_REF.exists() else None,
        help="Reference wav for ref_audio input (resampled to 32 kHz)",
    )
    parser.add_argument(
        "--ref-samples",
        type=int,
        default=DEFAULT_REF_SAMPLES_32K,
        help="ref_audio length at 32 kHz when using random noise or after trim/pad",
    )
    parser.add_argument(
        "--pred-lens",
        type=int,
        nargs="+",
        default=[50, 150, 300],
        help="pred_semantic sequence lengths to benchmark",
    )
    parser.add_argument(
        "--text-len",
        type=int,
        default=40,
        help="text_seq length (phoneme tokens)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs before the profiled inference",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON report path (default: target/vits_profile/report-<timestamp>.json)",
    )
    parser.add_argument(
        "--keep-ort-profiles",
        action="store_true",
        help="Keep raw ORT profile JSON files under target/vits_profile/raw/",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=15.0,
        help="ref_conditioning %% of wall time that triggers split recommendation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    inputs = model_input_names(model_path)
    has_sv_emb = "sv_emb" in inputs
    is_decode = "ge" in inputs and "ref_audio" not in inputs
    if is_decode:
        ref_audio, ref_source = None, "cached_ge"
        ge_shape = model_input_shape(model_path, "ge", [1, 512, 1])
        ge = np.random.randn(*ge_shape).astype(np.float32)
        print(f"decode profile: ge shape={ge_shape}", flush=True)
    else:
        ref_audio, ref_source = load_ref_audio(args.ref_audio, args.ref_samples)
        ge = None

    raw_dir = REPO_ROOT / "target/vits_profile/raw" if args.keep_ort_profiles else None
    scenarios: list[ScenarioResult] = []
    for pred_len in args.pred_lens:
        scenarios.append(
            run_scenario(
                model_path,
                pred_len=pred_len,
                text_len=args.text_len,
                ref_audio=ref_audio,
                has_sv_emb=has_sv_emb,
                warmup=args.warmup,
                profile_file_dir=raw_dir,
                is_decode=is_decode,
                ge=ge,
            )
        )

    report = ProfileReport(
        model_path=str(model_path),
        model_inputs=inputs,
        has_sv_emb=has_sv_emb,
        ref_audio_source=ref_source,
        scenarios=scenarios,
        decision_threshold_pct=args.threshold_pct,
        recommendation="",
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    if is_decode:
        # Decode-only: report vocoder share instead of split recommendation.
        voc_pcts = [s.groups_pct_of_wall.get("vocoder", 0.0) for s in scenarios]
        max_voc = max(voc_pcts) if voc_pcts else 0.0
        report.recommendation = (
            f"decode_profile: vocoder peaked at {max_voc:.1f}% of wall "
            f"(semantic_encode/flow are the other buckets)"
        )
    else:
        report.recommendation = decide(report, args.threshold_pct)

    out_path = args.out
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = DEFAULT_OUT_DIR / f"report-{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Model: {model_path}")
    print(f"Inputs: {inputs}")
    print(f"Recommendation: {report.recommendation}")
    for s in scenarios:
        if is_decode:
            voc = s.groups_ms.get("vocoder", 0.0)
            enc = s.groups_ms.get("semantic_encode", 0.0)
            flow = s.groups_ms.get("flow", 0.0)
            print(
                f"  pred_semantic={s.pred_semantic_len:4d}: wall={s.wall_ms:7.1f} ms, "
                f"vocoder={voc:6.2f} enc={enc:6.2f} flow={flow:6.2f}"
            )
        else:
            ref_ms = s.groups_ms.get("ref_conditioning", 0.0)
            ref_pct = s.groups_pct_of_wall.get("ref_conditioning", 0.0)
            print(
                f"  pred_semantic={s.pred_semantic_len:4d}: wall={s.wall_ms:7.1f} ms, "
                f"ref_conditioning={ref_ms:6.2f} ms ({ref_pct:.1f}% of wall)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
