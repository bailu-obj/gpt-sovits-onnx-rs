# v2ProPlus Run Validation

Validated on 2026-07-09 (macOS).

## Conclusion

`v2ProPlus` is already supported end-to-end in this repository. Export, artifact layout, and Rust inference all work without architecture changes.

## Commands Used

```bash
./scripts/export/04_download_models.sh --version v2ProPlus
./scripts/export/05_export.sh --version v2ProPlus --export-name custom_v2proplus --no-quant

./scripts/export/validate_bundle.sh \
  --bundle-dir gpt-sovits-upstream/onnx-patched/custom_v2proplus \
  --expect-v2pro

RUST_LOG=Debug cargo run --release --example gpt_sovits_demo -- \
  --model-path gpt-sovits-upstream/onnx-patched/custom_v2proplus \
  --ref-text "格式化，可以给自家的奶带来大量的。" \
  --text "今天天气真不错。"
```

## Export Results

- Output directory: `gpt-sovits-upstream/onnx-patched/custom_v2proplus/`
- Metadata (`gpt-sovits-upstream/onnx/custom_v2proplus.json`):
  - `Version`: `v2ProPlus`
  - `IsV2Pro`: `true`
- Required artifacts present: `ssl.onnx`, `bert.onnx`, `sv.onnx`, `g2pW.onnx`, `g2p_en/`, `{prefix}_t2s_*.onnx`, `{prefix}_vits.onnx`
- VITS inputs: `text_seq`, `pred_semantic`, `ref_audio`, `sv_emb`

## Inference Results

- Demo loaded `sv.onnx` automatically
- Logged `SV embedding shape: [1, 20480]`
- Generated `output.wav` (mono 32000 Hz)
- Synchronous inference: ~1447 ms on validation machine

## Audio Quality Parity (2026-07-09)

After runtime fixes (T2S sampler, semantic extraction, EOS stop, VITS output crop, Kaldi fbank for SV):

| Metric | Python `python_v2proplus.wav` | Rust `output.wav` |
|--------|-------------------------------|-------------------|
| Duration | 2.38 s (76160 samples) | 2.38 s (76160 samples) |
| RMS | 0.00308 | 0.00283 |
| Peak | 0.026 | 0.027 |
| SV emb head max-abs diff vs Python | — | < 0.00001 |

Reproduce:

```bash
gpt-sovits-upstream/.venv/bin/python scripts/compare_infer_metrics.py --seed 42
```

Key runtime behaviors aligned with upstream `TTS.py`:

- Semantic tokens: `y[-idx:]` after EOS pop
- VITS crop: `semantic_len * 2 * 640` samples before postprocess (ONNX VITS returns a fixed max buffer)
- SV fbank: Kaldi-compatible fbank without `knf_rs` per-bin mean normalization

## Notes

- `v2ProPlus` shares the same Pro-family runtime path as `v2Pro`; Rust detects Pro-family models by the presence of `sv.onnx`.
- Use `--export-name custom` if you want filenames like `custom_vits.onnx`, or any export name with the updated demo auto-discovery (`*_vits.onnx` prefix detection).
