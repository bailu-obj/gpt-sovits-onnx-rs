#!/usr/bin/env python3
"""Compare Rust ONNX vs Python PyTorch for v2Pro and v2ProPlus."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from sampling_defaults import DEFAULT_SAMPLING


def audio_stats(path: Path) -> dict:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    rms = math.sqrt(float((audio**2).mean())) if len(audio) else 0.0
    peak = float(abs(audio).max()) if len(audio) else 0.0
    return {
        "path": str(path),
        "sr": sr,
        "samples": len(audio),
        "duration_sec": round(duration, 3),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "audio": audio,
    }


def waveform_similarity(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    if n == 0:
        return {"aligned_samples": 0, "corr": 0.0, "mse": 0.0}
    x = np.asarray(a[:n], dtype=np.float64)
    y = np.asarray(b[:n], dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    corr = float(x @ y / denom) if denom > 1e-12 else 0.0
    mse = float(((np.asarray(a[:n], np.float64) - np.asarray(b[:n], np.float64)) ** 2).mean())
    return {
        "aligned_samples": n,
        "corr": round(corr, 6),
        "mse": round(mse, 8),
    }


VERSION_BUNDLES = {
    "v2Pro": "custom",
    "v2ProPlus": "custom_v2proplus",
}


def _append_params(cmd: list[str], params_path: Path | None) -> None:
    if params_path is not None:
        cmd.extend(["--params", str(params_path)])


def run_python(
    repo: Path,
    upstream: Path,
    version: str,
    params_path: Path | None,
    ref_audio: Path,
    out_dir: Path,
    text: str,
    ref_text: str,
) -> Path:
    cmd = [
        str(upstream / ".venv/bin/python"),
        str(repo / "scripts/run_python_baseline.py"),
        "--output-dir",
        str(out_dir),
        "--versions",
        version,
        "--ref-audio",
        str(ref_audio),
        "--text",
        text,
        "--ref-text",
        ref_text,
    ]
    _append_params(cmd, params_path)
    subprocess.run(cmd, check=True, cwd=repo)
    return out_dir / f"python_{version.lower()}.wav"


def run_rust(
    repo: Path,
    bundle: Path,
    params_path: Path | None,
    out_path: Path,
    text: str,
    ref_text: str,
) -> None:
    cmd = [
        "cargo",
        "run",
        "--release",
        "--example",
        "gpt_sovits_demo",
        "--",
        "--model-path",
        str(bundle),
        "--text",
        text,
        "--ref-text",
        ref_text,
        "--output",
        str(out_path),
    ]
    _append_params(cmd, params_path)
    subprocess.run(cmd, check=True, cwd=repo)


def run_t2s_snapshot(
    repo: Path,
    bundle: Path,
    params_path: Path | None,
    text: str,
    ref_text: str,
) -> dict:
    cmd = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--example",
        "dump_rust_t2s",
        "--",
        "--model-path",
        str(bundle),
        "--text",
        text,
        "--ref-text",
        ref_text,
    ]
    _append_params(cmd, params_path)
    out = subprocess.check_output(cmd, cwd=repo, text=True)
    return json.loads(out)


def resolve_params_path(explicit: Path | None, bundle: Path) -> Path | None:
    if explicit is not None:
        return explicit
    bundle_params = bundle / "infer_params.json"
    if bundle_params.is_file():
        return bundle_params
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ONNX Rust vs PyTorch for v2Pro / v2ProPlus")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Optional JSON file to override built-in sampling defaults",
    )
    parser.add_argument(
        "--text",
        default="今天天气真不错。",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--ref-text",
        default="格式化，可以给自家的奶带来大量的。",
        help="Reference prompt text",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v2Pro", "v2ProPlus"],
        choices=list(VERSION_BUNDLES),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    upstream = repo / "gpt-sovits-upstream"
    report = {
        "sampling_defaults": DEFAULT_SAMPLING,
        "text": args.text,
        "ref_text": args.ref_text,
        "versions": {},
    }

    for version in args.versions:
        bundle_name = VERSION_BUNDLES[version]
        bundle = upstream / "onnx-patched" / bundle_name
        if not bundle.is_dir():
            print(f"Bundle missing: {bundle}", file=sys.stderr)
            return 1

        use_params = resolve_params_path(args.params, bundle)

        py_out = run_python(
            repo, upstream, version, use_params, bundle / "ref.wav", repo, args.text, args.ref_text
        )
        rs_out = repo / f"output_{version.lower()}.wav"
        run_rust(repo, bundle, use_params, rs_out, args.text, args.ref_text)

        py_stats = audio_stats(py_out)
        rs_stats = audio_stats(rs_out)
        sim = waveform_similarity(py_stats["audio"], rs_stats["audio"])

        try:
            t2s = run_t2s_snapshot(repo, bundle, use_params, args.text, args.ref_text)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            t2s = {"error": str(exc)}

        report["versions"][version] = {
            "bundle": str(bundle),
            "params_file": str(use_params) if use_params else None,
            "python": {k: v for k, v in py_stats.items() if k != "audio"},
            "rust": {k: v for k, v in rs_stats.items() if k != "audio"},
            "waveform_similarity": sim,
            "duration_match": py_stats["samples"] == rs_stats["samples"],
            "t2s_inputs": t2s,
        }

    reports_dir = repo / "scripts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "compare_onnx_versions.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
