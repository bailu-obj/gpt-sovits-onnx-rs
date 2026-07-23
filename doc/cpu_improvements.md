# CPU improvements (notes)

Short notes from the V2 CPU work. Defaults stay conservative.

**Current Apple Silicon latency / RSS numbers:** see [`doc/onnx_export_cpu_optimization.md`](onnx_export_cpu_optimization.md) (`fast` / `quality` / `fp32` / `fp16` presets).

## What shipped

- **Streaming**: `synthesize` yields each fragment after VITS (better TTFA on multi-sentence text).
- **Adaptive KV**: reusable T2S cache workspace (less short-request memory).
- **G2PW batch**: one ONNX call for all polyphone queries in a batch.
- **ORT profiles**: `configure_ort_runtime` / demo `--ort-profile latency|low-power`.
  - **Latency (default)**: independent per-session pools + spinning (best wall time).
  - **LowPower**: shared global pool + spinning off (lower idle CPU / fewer threads).
- **Exclusive VITS ownership**: load either monolithic `*_vits.onnx` **or** split `{stem}_ref.onnx` + `{stem}_decode.onnx`, never both.
- **Split VITS**: export default via `05_export.sh`; if split graphs exist at load time, cache `ge` at reference. Decode inputs include `noise_scale` and `speed`.
- **Hot-path**: FS-decoder KV written once into workspace; AR input vecs pre-sized; top-k uses partial select (seed-stable).
- **Precision presets** (export-time): `--precision fp32|fp16|quality|fast` — see the canonical export doc.

## Defaults that stay off / serial

| Feature | Status | Ship gate |
|---------|--------|-----------|
| **T2S batching** | Prototype in `src/t2s_batch.rs`; export via `GSV_EXPORT_T2S_BATCH=1` | Enable only if multi-fragment A/B shows **≥10% E2E** after padding; VITS stays serial (~1.4× theoretical) |
| **XNNPACK EP** | Cargo feature `xnnpack` (off) | Android/ARM only after device A/B **beats CPU EP** |
| **NNAPI** | Not used | Do not pursue unless new measurements reverse `doc/perf_record.md` slowdown |
| **Selective VITS quant** | Not done | Profile with `scripts/profile_vits_nodes.py` first; quality risk > T2S int8 |
| **fp16 preset** | Selective VITS + native FP16 I/O | BERT/T2S FP16 still slower than FP32 on Apple CPU EP (even native I/O); do not ship for latency — use `fast` |

## ORT runtime knobs

```bash
# Latency profile (default): independent pools, spinning on
cargo run --release --example gpt_sovits_demo -- \
  --model-path models/cpu_opt_v2_fast \
  --ort-profile latency --benchmark --run-count 3 --seed 42

# Low-power: shared pool, no spinning
cargo run --release --example gpt_sovits_demo -- \
  --model-path models/cpu_opt_v2_fast \
  --ort-profile low-power --ort-threads 4 --benchmark --run-count 3

# Optional XNNPACK (ARM/Android builds that include the EP)
cargo run --release --features xnnpack --example gpt_sovits_demo -- ...
```

From Rust:

```rust
use gpt_sovits_onnx_rs::{OrtConfig, OrtRuntimeProfile, configure_ort_runtime};
configure_ort_runtime(OrtConfig {
    profile: OrtRuntimeProfile::LowPower,
    intra_threads: Some(4),
    use_xnnpack: false,
    shared_thread_pool: Some(true),
});
```

## Export

```bash
scripts/export/05_export.sh --version v2 --export-name custom_v2 --precision fast
# or
python GPT_SoVITS/export_onnx_v2.py --model_path … --export_name custom_v2 --version v2 --split-vits-ref
python scripts/optimize_aio.py --input-dir onnx/custom_v2 --output-dir onnx-patched/custom_v2 --precision fast
```

## Quick checks

```bash
cargo test --release --lib
cargo test preprocess_parity --release
cargo test --release t2s_batch

cargo run --release --example gpt_sovits_demo -- \
  --model-path models/cpu_opt_v2_fast \
  --benchmark --run-count 3 --seed 42
```

## Acceptance targets

- No token/waveform parity regression under fixed-seed validation (`--precision fp32` or `quality` for AR-sensitive checks).
- ≥15% lower peak RSS when split VITS replaces monolithic (exclusive ownership), or document that session weights dominate (typical with mono-only bundles).
- ≤5% cached-ref latency regression in low-power vs latency profile on the same seed; target 5–10% lower short-input latency from copy/thread tuning where measurable.
- T2S batching / XNNPACK ship by default only when measured E2E improvement exceeds 10%.

## Profiler (optional)

```bash
gpt-sovits-upstream/.venv/bin/python scripts/profile_vits_nodes.py \
  --model models/cpu_opt_v2_fast/cpu_opt_v2_layout_vits.onnx \
  --ref-audio models/cpu_opt_v2_fast/ref.wav \
  --out target/vits_profile/report.json
```
