#!/usr/bin/env python3
"""Compare Python Kaldi fbank vs exported SV ONNX input stats."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--sv-onnx", type=Path, required=True)
    args = parser.parse_args()

    upstream = args.upstream_dir.resolve()
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "GPT_SoVITS"))

    import numpy as np
    import onnxruntime as ort
    import torch
    import torchaudio
    from torch import nn

    import kaldi as Kaldi

    wav, sr = torchaudio.load(str(args.ref_audio))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    audio_16k = wav.squeeze(0).numpy().astype(np.float32)

    fbank = Kaldi.fbank(
        torch.from_numpy(audio_16k),
        num_mel_bins=80,
        sample_frequency=16000,
        dither=0,
    ).numpy()

    sess = ort.InferenceSession(str(args.sv_onnx), providers=["CPUExecutionProvider"])
    sv_emb = sess.run(None, {"audio_feature": fbank})[0]

    report = {
        "fbank_shape": list(fbank.shape),
        "fbank_mean": float(fbank.mean()),
        "fbank_std": float(fbank.std()),
        "sv_emb_shape": list(sv_emb.shape),
        "sv_emb_mean": float(sv_emb.mean()),
        "sv_emb_std": float(sv_emb.std()),
        "sv_emb_head": sv_emb.reshape(-1)[:8].tolist(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
