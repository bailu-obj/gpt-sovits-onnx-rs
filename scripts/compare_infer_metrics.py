#!/usr/bin/env python3
"""Compare Rust vs Python inference metrics on fixed inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from sampling_defaults import COMPARE_SEED, DEFAULT_SAMPLING


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


def dump_python_sv(upstream: Path, ref_audio: Path) -> dict:
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "GPT_SoVITS"))
    import numpy as np
    import onnxruntime as ort
    import torch
    import torchaudio
    from torchaudio.compliance import kaldi as Kaldi

    wav, sr = torchaudio.load(str(ref_audio))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    audio_16k = wav.squeeze(0)
    fbank = Kaldi.fbank(
        audio_16k.unsqueeze(0),
        num_mel_bins=80,
        sample_frequency=16000,
        dither=0,
    ).numpy()
    bundle = ref_audio.parent
    sv_onnx = bundle / "sv.onnx"
    sess = ort.InferenceSession(str(sv_onnx), providers=["CPUExecutionProvider"])
    emb = sess.run(None, {"audio_feature": fbank.astype(np.float32)})[0]
    arr = emb.reshape(-1)
    return {
        "fbank_shape": list(fbank.shape),
        "shape": list(emb.shape),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "head": arr[:8].tolist(),
    }


def dump_rust_sv(repo: Path, model_path: Path, ref_text: str) -> dict:
    out = subprocess.check_output(
        [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--example",
            "dump_rust_sv",
            "--",
            "--model-path",
            str(model_path),
            "--ref-text",
            ref_text,
        ],
        cwd=repo,
        text=True,
    )
    return json.loads(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="onnx-patched bundle for Rust demo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=COMPARE_SEED,
        help="Fixed seed for reproducible Rust vs Python comparison",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Optional JSON file to override built-in sampling defaults",
    )
    parser.add_argument(
        "--text",
        default="你好啊，这是一个测试。吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。This demo is only for test  usage. If you find any 问题, 请修复它。",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--ref-text",
        default="格式化，可以给自家的奶带来大量的。",
        help="Reference prompt text",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    upstream = repo / "gpt-sovits-upstream"
    model_path = args.model_path or (upstream / "onnx-patched/custom_v2proplus")
    params_path = args.params
    if params_path is None:
        bundle_params = model_path / "infer_params.json"
        if bundle_params.is_file():
            params_path = bundle_params

    ref_text = args.ref_text
    text = args.text
    ref_audio = model_path / "ref.wav"

    py_out = repo / "python_v2proplus.wav"
    rs_out = repo / "output.wav"
    reports_dir = repo / "scripts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "compare_infer_metrics.json"

    py_cmd = [
        str(upstream / ".venv/bin/python"),
        str(repo / "scripts/run_python_baseline.py"),
        "--output-dir",
        str(repo),
        "--versions",
        "v2ProPlus",
        "--ref-audio",
        str(ref_audio),
        "--text",
        text,
        "--ref-text",
        ref_text,
        "--seed",
        str(args.seed),
    ]
    if params_path is not None:
        py_cmd.extend(["--params", str(params_path)])

    subprocess.run(py_cmd, check=True, cwd=repo)

    rs_cmd = [
        "cargo",
        "run",
        "--release",
        "--example",
        "gpt_sovits_demo",
        "--",
        "--model-path",
        str(model_path),
        "--text",
        text,
        "--ref-text",
        ref_text,
        "--seed",
        str(args.seed),
    ]
    if params_path is not None:
        rs_cmd.extend(["--params", str(params_path)])

    subprocess.run(rs_cmd, check=True, cwd=repo)

    py_stats = audio_stats(py_out) if py_out.exists() else None
    rs_stats = audio_stats(rs_out) if rs_out.exists() else None

    report = {
        "inputs": {
            "params_file": str(params_path) if params_path else None,
            "sampling_defaults": DEFAULT_SAMPLING,
            "seed": args.seed,
            "ref_audio": str(ref_audio),
            "ref_text": ref_text,
            "text": text,
        },
        "python_v2proplus": {k: v for k, v in (py_stats or {}).items() if k != "audio"},
        "rust_output": {k: v for k, v in (rs_stats or {}).items() if k != "audio"},
    }

    if py_stats and rs_stats:
        sim = waveform_similarity(py_stats["audio"], rs_stats["audio"])
        report["waveform_similarity"] = sim
        if py_stats["samples"] != rs_stats["samples"]:
            report["duration_match"] = False
            report["note"] = (
                "Duration mismatch alone does not imply quality parity; "
                "check waveform_similarity.corr."
            )
        else:
            report["duration_match"] = True

    if ref_audio.exists():
        report["python_sv_emb"] = dump_python_sv(upstream, ref_audio)
        try:
            rust_sv = dump_rust_sv(repo, model_path, ref_text)
            report["rust_sv_emb"] = rust_sv
            py = np.array(report["python_sv_emb"]["head"])
            rs = np.array(rust_sv["head"])
            report["sv_head_max_abs_diff"] = float(abs(py - rs).max())
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            report["rust_sv_emb_error"] = str(exc)

    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
