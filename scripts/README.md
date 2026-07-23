# 使用说明

本目录包含 GPT-SoVITS → ONNX 的转换与优化脚本。上游 ONNX 覆盖文件位于仓库根目录的 [`patch/GPT_SoVITS/`](../patch/GPT_SoVITS/)（对上游的 patch overlay，不是独立 Python 包）。

下载预训练包与 demo 运行说明见根目录 [`Readme.md`](../Readme.md)。

## 目录结构

| 路径 | 说明 |
|------|------|
| [`patch/GPT_SoVITS/`](../patch/GPT_SoVITS/) | 必须覆盖到上游的 ONNX 导出 patch（KV-cache T2S、V2Pro SV 等） |
| [`scripts/export/`](export/) | macOS 分阶段导出脚本（按顺序逐步执行） |
| [`scripts/optimize_aio.py`](optimize_aio.py) | ONNX 后期优化（`--precision` 预设） |
| [`scripts/bench_cpu_bundles.py`](bench_cpu_bundles.py) | Apple Silicon 多 bundle 延迟 / RSS 基准 |
| [`scripts/profile_onnx_layout.py`](profile_onnx_layout.py) | Transpose/Reshape 等布局热点统计 |
| [`scripts/profile_onnx_stages.py`](profile_onnx_stages.py) | 分阶段 ORT 耗时 |
| [`scripts/profile_vits_nodes.py`](profile_vits_nodes.py) | VITS 节点级 profiling |
| [`scripts/requirements.txt`](requirements.txt) | `optimize_aio.py` 的 Python 依赖 |
| [`scripts/run_python_baseline.py`](run_python_baseline.py) | PyTorch 基线推理 |
| [`scripts/compare_infer_metrics.py`](compare_infer_metrics.py) | Rust vs Python 时长/RMS/SV/波形（v2ProPlus） |
| [`scripts/compare_onnx_versions.py`](compare_onnx_versions.py) | Rust vs Python（v2Pro + v2ProPlus） |
| [`scripts/sampling_defaults.py`](sampling_defaults.py) | 内置采样默认值（与 Rust `InferParams::default()` 一致） |
| [`scripts/reports/`](reports/) | 对比脚本 JSON（gitignore） |

## 重要：必须覆盖上游 ONNX 模块

最新版 GPT-SoVITS 自带的 `t2s_model_onnx.py` 使用内部 cache 字典，与 Rust 推理所需的 **外部 KV-cache** 设计不兼容。导出前**必须**将 `patch/GPT_SoVITS/` 覆盖到上游项目的 `GPT_SoVITS/` 目录。

## Precision 预设

| `--precision` | BERT / g2pW | T2S | SSL | VITS | 用途 |
|---------------|-------------|-----|-----|------|------|
| **`fast`**（默认） | INT4 | INT8 | FP32 slim | FP32 slim | 推荐 CPU 延迟 / RSS |
| `quality` | INT4 | FP32 | FP32 slim | FP32 slim | 前端压缩，AR 保持 FP32 |
| `fp32` | FP32 | FP32 | FP32 slim | FP32 slim | 对齐 / 调试 |
| `fp16` | FP32 slim | FP32 slim | FP32 slim | **FP16 原生 I/O** | 仅 VITS；修复旧 blanket ~2× 回退；Apple CPU 上不优于 `fp32` 延迟 |

Apple Silicon 上 BERT/T2S 即使全 FP16 I/O 仍慢于 FP32（ORT CPU EP MatMul）；勿对它们开 FP16 追延迟。实验：`--fp16-kinds all` / `--fp16-keep-io`。

HF 发布目录：`quant/` ≈ `fast`，`unquant/` ≈ `fp32`。完整测速见 [`doc/onnx_export_cpu_optimization.md`](../doc/onnx_export_cpu_optimization.md)。

## macOS 分阶段导出

仅支持 **macOS**。按顺序逐步运行：

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 1 | [`01_clone.sh`](export/01_clone.sh) | 克隆/更新 GPT-SoVITS → `gpt-sovits-upstream/` |
| 2 | [`02_patch.sh`](export/02_patch.sh) | 应用 `patch/GPT_SoVITS/` |
| 3 | [`03_setup_env.sh`](export/03_setup_env.sh) | conda（优先）或 uv 环境 |
| 4 | [`04_download_models.sh`](export/04_download_models.sh) | 下载权重；含 G2PW、g2p_en |
| 5 | [`05_export.sh`](export/05_export.sh) | `export_onnx_v2.py` + `optimize_aio.py` |

```bash
./scripts/export/01_clone.sh
./scripts/export/02_patch.sh
./scripts/export/03_setup_env.sh --source HF
./scripts/export/04_download_models.sh --version v2Pro
./scripts/export/05_export.sh --version v2Pro --export-name custom --precision fast

# 对齐用 fp32；上线用 fast（默认）
./scripts/export/04_download_models.sh --version v2ProPlus
./scripts/export/05_export.sh --version v2ProPlus --export-name custom_v2proplus --precision fp32
./scripts/export/validate_bundle.sh --bundle-dir gpt-sovits-upstream/onnx-patched/custom_v2proplus --expect-v2pro
```

输出：`gpt-sovits-upstream/onnx-patched/{export_name}/`

常用参数：

- `03_setup_env.sh`：`--source HF|HF-Mirror|ModelScope`，`--skip-env`
- `04_download_models.sh`：`--version v2|v2Pro|v2ProPlus`
- `05_export.sh`：`--precision …`，`--output-dir PATH`，`--no-split-vits-ref`，`--smoke-test`

### 环境管理器

| 检测顺序 | 行为 |
|----------|------|
| 有 conda | `conda create -n GPTSoVits python=3.10` + 上游 `install.sh --device MPS` |
| 仅有 uv | `.venv` + `uv pip install`；`04` 负责模型与 NLTK/OpenJTalk |
| 都没有 | 报错并提示安装 Miniconda 或 uv |

## 手动导出（任意平台）

1. 克隆 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)，安装依赖并下载模型。
2. 将 `patch/GPT_SoVITS/` **整体覆盖**到上游 `GPT_SoVITS/`。
3. 准备 `{model_path}/gpt.ckpt` + `sovits.pth`。
4. 导出：

```bash
python GPT_SoVITS/export_onnx_v2.py --model_path ./models/v2 --export_name custom --version v2 --split-vits-ref
```

V2Pro / Plus 还需 SV 权重：`GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt`。

5. 拷贝 `g2pW.onnx` 与 `g2p_en/` 到 `onnx/{export_name}/`。
6. 优化：

```bash
python scripts/optimize_aio.py --input-dir onnx/custom --output-dir onnx-patched/custom --precision fast
python scripts/optimize_aio.py --input-dir onnx/custom --output-dir onnx-patched/custom_fp32 --precision fp32
```

## 导出产物

| 文件 | V2 | V2Pro / V2ProPlus |
|------|----|-------------------|
| `ssl.onnx` | ✓ | ✓ |
| `bert.onnx` | ✓ | ✓ |
| `{name}_t2s_encoder.onnx` | ✓ | ✓ |
| `{name}_t2s_fs_decoder.onnx` | ✓ | ✓ |
| `{name}_t2s_s_decoder.onnx` | ✓ | ✓ |
| `{name}_vits.onnx` | 3 输入 | 4 输入（+ `sv_emb`） |
| `{name}_vits_ref.onnx` / `{name}_vits_decode.onnx` | 默认导出（可用 `--no-split-vits-ref` 关闭） | 同左 |
| `sv.onnx` | — | ✓ |
| `{name}.json` | 元数据 | 同上 |
| `g2pW.onnx` | ✓ | ✓ |
| `g2p_en/` | ✓ | ✓ |
| `ref.wav` | 推荐 | 推荐 |

## 推理参数（内置默认值）

| 参数 | 默认值 |
|------|--------|
| `top_k` | 4 |
| `top_p` | 0.9 |
| `temperature` | 1.0 |
| `repetition_penalty` | 1.35 |
| `seed` | random（对比脚本用 `COMPARE_SEED=42`） |

Rust：`InferParams::default()`；demo 可用 `--top-k` 等覆盖。Python：[`sampling_defaults.py`](sampling_defaults.py)。

## 基准与对比

```bash
# 四预设延迟 / RSS（seed 42，short + multi）
python3 scripts/bench_cpu_bundles.py \
  --preset-bundles \
    fp32=models/cpu_opt_v2_fp32 \
    fp16=models/cpu_opt_v2_fp16 \
    quality=models/cpu_opt_v2_quality \
    fast=models/cpu_opt_v2_fast \
  --out target/bench/cpu_opt_v2_precision.json

# Rust vs Python（报告在 scripts/reports/）
gpt-sovits-upstream/.venv/bin/python scripts/compare_onnx_versions.py
gpt-sovits-upstream/.venv/bin/python scripts/compare_infer_metrics.py \
  --model-path gpt-sovits-upstream/onnx-patched/custom_v2proplus
```

对齐验证请用 `--precision fp32`（见 [`doc/v2proplus_validation.md`](../doc/v2proplus_validation.md)）。跨框架历史对比见 [`doc/pytorch_vs_onnx_speed.md`](../doc/pytorch_vs_onnx_speed.md)。

```bash
cargo run --release --example dump_rust_t2s -- \
  --model-path gpt-sovits-upstream/onnx-patched/custom_v2proplus \
  --text "你好啊，这是一个测试。" \
  --ref-text "格式化，可以给自家的奶带来大量的。"
```

## 预处理一致性测试

```bash
GPT_SOVITS_ROOT=~/gpt-sovits-upstream python3 scripts/preprocess_parity.py > /tmp/python_preprocess.json
PYTHON_PREPROCESS_JSON=/tmp/python_preprocess.json cargo test preprocess_parity_vs_python -- --ignored
```

语料：`resource/preprocess_corpus.json`。
