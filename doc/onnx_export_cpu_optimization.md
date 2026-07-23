# ONNX CPU Export Optimization (2026-07-23)

Canonical Apple Silicon (arm64) benchmark for layout + precision presets. Focus: **latency**, **RSS**, **bundle size**.

**Hardware for numbers below:** Apple M4 Pro, 14 cores, macOS 26.5.2, ORT CPU EP (`ort` 2.0.0-rc.10 / Python ORT 1.23.2 for stage probes).

## Precision presets

Set at export time via `--precision` on [`scripts/export/05_export.sh`](../scripts/export/05_export.sh) / [`scripts/optimize_aio.py`](../scripts/optimize_aio.py). Runtime selects a prepared bundle with `--model-path`; `--ort-profile` is independent.

| Preset | BERT / g2pW | T2S decoders | SSL / g2p_en | VITS | Role |
|--------|-------------|--------------|--------------|------|------|
| **`fast`** (default) | INT4 | INT8 | FP32 slim | FP32 slim | Best measured CPU latency / RSS |
| `quality` | INT4 | FP32 | FP32 slim | FP32 slim | Smaller frontend; AR stays FP32 |
| `fp32` | FP32 | FP32 | FP32 slim | FP32 slim | Parity / debug |
| `fp16` | FP32 slim | FP32 slim | FP32 slim | **FP16 native I/O** | Selective: VITS only (see diagnosis) |

### FP16 diagnosis (why BERT/T2S stay FP32)

Blanket weight-FP16 with FP32 I/O (`keep_io_types=True`) was **~2× slower** than `fp32` E2E. Root causes on Apple CPU EP:

1. **Boundary Casts** — Cast count 87 → 251 (T2S S-decoder alone 16 → 113).
2. **Weak FP16 MatMul** — even with **native FP16 I/O** (`keep_io_types=False`), stage microbench still loses to FP32:

| Stage | FP32 | FP16 keep-IO | FP16 native-IO |
|-------|------|--------------|----------------|
| T2S S (AR step) | 3.17 ms | 4.63 ms | 3.98 ms |
| T2S FS | 17.7 ms | 21.2 ms | 20.8 ms |
| BERT | 30.0 ms | 38.7 ms | 37.8 ms |
| VITS decode | 371 ms | 577 ms | **272 ms** |

**Conclusion:** BERT/T2S cannot be fixed for Apple CPU latency by “full FP16 calc” — ORT CPU EP FP16 kernels are slower than FP32. Those stages stay FP32. VITS benefits from **native FP16 I/O**, so the `fp16` preset converts only `vits` / `vits_decode` / `vits_ref`.

Override (experiments only): `--fp16-kinds all` and/or `--fp16-keep-io`.

## Current bundles (same raw layout/split-VITS source)

| Preset | Path | Size |
|--------|------|------|
| **fast** | `models/cpu_opt_v2_fast` | ~1.1 GB |
| quality | `models/cpu_opt_v2_quality` | ~1.5 GB |
| fp32 | `models/cpu_opt_v2_fp32` | ~3.0 GB |
| fp16 | `models/cpu_opt_v2_fp16` | ~2.9 GB (VITS half; rest FP32) |

Shared export features: native BERT `[B,T,1024]`, KV-delta stage outputs, split VITS ref/decode.

## Measured results (2026-07-23 evening, refreshed)

**Method:** seed `42`, `--ort-profile latency`, `--run-count 5`, texts from [`scripts/bench_cpu_bundles.py`](../scripts/bench_cpu_bundles.py):

- short: `你好，这是一个测试。`
- multi: `你好啊。这是一个测试。吃葡萄不吐葡萄皮。`

Report: [`target/bench/cpu_opt_v2_precision_final.json`](../target/bench/cpu_opt_v2_precision_final.json)

| Preset | Case | E2E median | TTFA | RSS final | Size | exit |
|--------|------|------------|------|-----------|------|------|
| **fast** | short | **320.9 ms** | 325.7 ms | 2.11 GB | 1104 MiB | 0 |
| **fast** | multi | **846.0 ms** | 222.0 ms | 2.20 GB | 1104 MiB | 0 |
| quality | short | 412.6 ms | 386.8 ms | 2.95 GB | 1549 MiB | 0 |
| quality | multi | 1019.7 ms | 242.5 ms | 3.04 GB | 1549 MiB | 0 |
| fp32 | short | 465.6 ms | 464.9 ms | 5.84 GB | 3046 MiB | 0 |
| fp32 | multi | 1033.6 ms | 235.9 ms | 5.80 GB | 3046 MiB | 0 |
| fp16 (selective VITS) | short | 462.3 ms | 460.5 ms | 5.80 GB | 2892 MiB | 0 |
| fp16 (selective VITS) | multi | 1070.0 ms | 247.2 ms | 5.77 GB | 2892 MiB | 0 |

**Legacy blanket FP16** (all graphs, FP32 I/O): short **~924 ms** / multi **~2056 ms** — do not regenerate.

**fp16 vs fp32:** selective native-IO VITS removes the ~2× regression; E2E is **~parity** with `fp32` (not a reliable beat). Use **`fast`** for latency. Fixed-seed T2S dumps match `fp32` (identical AR graphs); VITS FP16 changes waveform numerics (corr can be low under stochastic noise).

## What changed (layout / runtime)

### Export ([`patch/GPT_SoVITS/`](../patch/GPT_SoVITS/))

- Native FS BERT layout (default): `[B,T,1024]`. Legacy `[B,1024,T]` via `GSV_EXPORT_BERT_BFT=1`.
- KV delta outputs (default): stage decoder emits `[B,1,H]` K/V (`GSV_EXPORT_KV_DELTA=1`).
- Split VITS ref/decode default in `05_export.sh` (`--no-split-vits-ref` to disable).

### Optimize ([`scripts/optimize_aio.py`](../scripts/optimize_aio.py))

- `--precision fp16` converts only `--fp16-kinds` (default VITS).
- Default **native FP16 I/O** (`keep_io_types=False`); `--fp16-keep-io` restores legacy FP32 I/O boundaries.

### Runtime

- Autodetect BERT layout and KV-delta from ORT metadata.
- Autodetect VITS native FP16 I/O; feed/extract `f16` for `ge` / `noise_scale` / `speed` / `audio` / `ref_audio` when needed ([`src/ort_dtype.rs`](../src/ort_dtype.rs)).

## Recommendations

1. **Ship `fast`** for CPU latency / RSS.
2. Use **`quality`** when you want INT4 frontend savings but FP32 T2S logits.
3. Use **`fp32`** for parity/debug only.
4. **`fp16`** = selective VITS FP16 (native I/O). Fixes the old 2× regression vs blanket FP16; **does not beat `fp32` E2E** on Apple CPU. Do **not** FP16 BERT/T2S for CPU latency.
5. Keep Rust layout autodetection for older bundles.
6. Do not enable T2S batching / head-major KV / selective VITS INT8 without new ≥10% E2E proof.

## Reproduce

```bash
bash scripts/export/02_patch.sh

# Selective FP16 (default kinds = VITS, native I/O)
gpt-sovits-upstream/.venv/bin/python scripts/optimize_aio.py \
  --input-dir gpt-sovits-upstream/onnx/cpu_opt_v2_layout \
  --output-dir models/cpu_opt_v2_fp16 --precision fp16

# Legacy blanket experiment (slow on Apple CPU):
#   ... --precision fp16 --fp16-kinds all --fp16-keep-io

python3 scripts/bench_cpu_bundles.py \
  --preset-bundles \
    fp32=models/cpu_opt_v2_fp32 \
    fp16=models/cpu_opt_v2_fp16 \
    quality=models/cpu_opt_v2_quality \
    fast=models/cpu_opt_v2_fast \
  --out target/bench/cpu_opt_v2_precision.json
```

## Env knobs (export)

| Variable | Default | Effect |
|----------|---------|--------|
| `GSV_EXPORT_KV_DELTA` | `1` | Stage decoder emits single-row K/V |
| `GSV_EXPORT_BERT_BFT` | `0` | `1` → legacy `[B,1024,T]` BERT |
| `GSV_EXPORT_T2S_BATCH` | `0` | Batched T2S axes (Rust path still prototype) |

## Phase 2 notes (historical)

VITS decode is vocoder Conv/ConvTranspose-bound; selective MatMul/Gather INT8 and head-major KV did **not** clear ≥10% E2E / ≥15% VITS gates. Decode stays FP32 slim under `fast`/`quality`/`fp32`; seq-major KV remains the export default. Selective **FP16** VITS is a separate size/experiment path, not a replacement for `fast`.
