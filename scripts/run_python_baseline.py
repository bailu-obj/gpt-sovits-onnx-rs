#!/usr/bin/env python3
"""Run upstream GPT-SoVITS PyTorch inference for v2 / v2Pro / v2ProPlus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import soundfile as sf

from sampling_defaults import DEFAULT_SAMPLING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-SoVITS PyTorch baseline inference")
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "gpt-sovits-upstream",
        help="GPT-SoVITS upstream clone root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory to write wav files",
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=None,
        help="Reference wav path (default: onnx-patched bundle ref.wav)",
    )
    parser.add_argument(
        "--ref-text",
        default="格式化，可以给自家的奶带来大量的。",
        help="Reference prompt text",
    )
    parser.add_argument(
        "--text",
        default="今天天气真不错。",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v2", "v2Pro", "v2ProPlus"],
        help="Model versions to run",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLING["seed"])
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Optional JSON file to override built-in sampling defaults",
    )
    return parser.parse_args()


def build_sampling(args: argparse.Namespace) -> dict:
    sampling = dict(DEFAULT_SAMPLING)
    sampling["seed"] = args.seed
    if args.params is not None:
        with args.params.open(encoding="utf-8") as f:
            overrides = json.load(f)
        sampling.update(overrides)
    return sampling


def main() -> int:
    args = parse_args()
    upstream = args.upstream_dir.resolve()
    if not upstream.is_dir():
        print(f"Upstream directory not found: {upstream}", file=sys.stderr)
        return 1

    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "GPT_SoVITS"))

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    ref_audio = args.ref_audio
    if ref_audio is None:
        ref_audio = upstream / "onnx-patched/custom_v2proplus/ref.wav"
    ref_audio = ref_audio.resolve()
    if not ref_audio.is_file():
        print(f"Reference audio not found: {ref_audio}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampling = build_sampling(args)

    req = {
        "text": args.text,
        "text_lang": "zh",
        "ref_audio_path": str(ref_audio),
        "prompt_text": args.ref_text,
        "prompt_lang": "zh",
        "top_k": sampling["top_k"],
        "top_p": sampling["top_p"],
        "temperature": sampling["temperature"],
        "repetition_penalty": sampling["repetition_penalty"],
        "text_split_method": "cut0",
        "batch_size": 1,
        "split_bucket": False,
        "parallel_infer": False,
        "streaming_mode": False,
        "seed": sampling["seed"],
    }

    for version in args.versions:
        if version not in TTS_Config.default_configs:
            print(f"Unknown version: {version}", file=sys.stderr)
            return 1

        print(f"\n=== Running PyTorch inference: {version} ===")
        version_cfg = deepcopy(TTS_Config.default_configs[version])
        version_cfg["device"] = "cpu"
        version_cfg["is_half"] = False
        tts_config = TTS_Config({"custom": version_cfg})
        tts = TTS(tts_config)

        sr = None
        audio = None
        for chunk_sr, chunk_audio in tts.run(req):
            sr = chunk_sr
            if audio is None:
                audio = chunk_audio
            else:
                import numpy as np

                audio = np.concatenate([audio, chunk_audio])

        out_path = args.output_dir / f"python_{version.lower()}.wav"
        sf.write(out_path, audio, sr)
        print(f"Saved {out_path} ({len(audio) / sr:.2f}s @ {sr} Hz)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
