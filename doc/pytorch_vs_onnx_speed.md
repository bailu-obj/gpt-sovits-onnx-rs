# PyTorch vs Rust ONNX Speed Comparison

Validated on **2026-07-09** (macOS, CPU only).

> **Note:** This page uses the **pre-layout** v2Pro / v2ProPlus patched bundles and the **long demo mixed CN/EN** text. It is **not** comparable to the short/multi Apple Silicon table in [`onnx_export_cpu_optimization.md`](onnx_export_cpu_optimization.md) (`fast` / `fp32` presets, seed 42).

## Summary

On the same fixed inputs (built-in sampling defaults, text via CLI), **Rust ONNX inference is faster than upstream PyTorch** for both v2Pro and v2ProPlus when reference audio is cached (typical chat / multi-utterance use).

| Version | PyTorch `tts.run()` | Rust `synthesize_sync` | Speedup |
|---------|---------------------|-------------------------|---------|
| **v2Pro** | **910 ms** (median 910) | **400 ms** (median 400) | **~2.3×** |
| **v2ProPlus** | **992 ms** (median 992) | **548 ms** (median 547) | **~1.8×** |

PyTorch one-time model load: ~760 ms (v2Pro), ~823 ms (v2ProPlus).

Rust pays a larger one-time cost on first `cargo run` (ONNX session creation + `process_reference` including SSL/SV). That cost is **not** included in the synthesis numbers above.

---

## Test setup

| Item | Value |
|------|--------|
| Text | `你好啊，这是一个测试。吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。This demo is only for test  usage. If you find any 问题, 请修复它。` |
| Ref text | `格式化，可以给自家的奶带来大量的。` |
| Sampling | `top_k=4`, `top_p=0.9`, `temperature=1.0`, `repetition_penalty=1.35`; `seed=42` only in compare scripts |
| PyTorch device | CPU (`is_half=False`) |
| Rust | Release build, ONNX Runtime CPU EP |
| Timed runs | 10 (Python: 2 warmup discarded) |
| Reference | Cached after first run (HuBERT / ref spec not re-encoded) |

Bundles:

- v2Pro → `gpt-sovits-upstream/onnx-patched/custom/`
- v2ProPlus → `gpt-sovits-upstream/onnx-patched/custom_v2proplus/`

**Fairness note:** PyTorch `tts.run()` still re-runs text frontend (BERT/G2P) on every call even when `prompt_cache` holds reference audio. Rust times `synthesize_sync` only, after `process_reference_sync` — matching “reference fixed, synthesize many sentences” usage.

---

## Per-stage breakdown (single run)

### v2ProPlus

| Stage | PyTorch (cached ref)* | Rust ONNX |
|-------|----------------------|-----------|
| Text preprocess | ~61 ms | ~20 ms |
| T2S semantic | ~586–592 ms | ~241 ms (encoder 2.5 + fs 46 + s-decoder 193 ms) |
| VITS decode | ~234–239 ms | ~289 ms |
| **Synthesis total** | **~910–992 ms** | **~548 ms** |

### v2Pro

| Stage | PyTorch (cached ref)* | Rust ONNX |
|-------|----------------------|-----------|
| T2S semantic | ~586 ms | ~238 ms |
| VITS decode | ~157 ms | ~149 ms |
| **Synthesis total** | **~910 ms** | **~410 ms** |

\*From upstream log after reference cache, e.g. `0.000 0.060 0.588 0.236` → ref skipped, BERT ≈ 60 ms, T2S ≈ 588 ms, VITS ≈ 236 ms.

---

## Takeaways

1. **End-to-end synthesis:** Rust ONNX wins on CPU for both Pro-family models (~1.8–2.3×).
2. **Largest gain is T2S:** PyTorch ~590 ms vs Rust ~190–240 ms per short utterance.
3. **VITS is mixed:** Rust is slightly faster on v2Pro; on v2ProPlus VITS alone can be a bit slower than PyTorch, but T2S savings dominate total latency.
4. **Text frontend:** Rust preprocess (~20 ms) is faster than PyTorch BERT pass (~60 ms) on cached-ref runs.

Audio **quality** parity is separate from speed; see `doc/v2proplus_validation.md` and `scripts/compare_onnx_versions.py`.

---

## Reproduce

### PyTorch

```bash
cd gpt-sovits-upstream
../.venv/bin/python ../scripts/run_python_baseline.py \
  --versions v2Pro v2ProPlus
```

Upstream prints per-stage seconds on the last line of each run (ref, BERT, T2S, VITS).

### Rust ONNX

```bash
# v2Pro — 10-run median/avg
cargo run --release --example gpt_sovits_demo -- \
  --model-path gpt-sovits-upstream/onnx-patched/custom \
  --run-count 10 \
  --output /tmp/output_v2pro.wav

# v2ProPlus
cargo run --release --example gpt_sovits_demo -- \
  --model-path gpt-sovits-upstream/onnx-patched/custom_v2proplus \
  --run-count 10 \
  --output /tmp/output_v2proplus.wav
```

Per-stage Rust timings (debug):

```bash
RUST_LOG=gpt_sovits_onnx_rs=debug cargo run --release --example gpt_sovits_demo -- \
  --model-path gpt-sovits-upstream/onnx-patched/custom_v2proplus \
  --run-count 1
```

---

## Related

- Historical platform benchmarks (quantization, Android): [`doc/perf_record.md`](perf_record.md)
- v2ProPlus validation: [`doc/v2proplus_validation.md`](v2proplus_validation.md)