# 使用说明

本目录包含 GPT-SoVITS → ONNX 的转换与优化脚本。上游 ONNX 覆盖文件位于仓库根目录的 [`patch/GPT_SoVITS/`](../patch/GPT_SoVITS/)（对上游的 patch overlay，不是独立 Python 包）。

## 目录结构

| 路径 | 说明 |
|------|------|
| [`patch/GPT_SoVITS/`](../patch/GPT_SoVITS/) | 必须覆盖到上游的 ONNX 导出 patch（KV-cache T2S、V2Pro SV 等） |
| [`scripts/export/`](export/) | macOS 分阶段导出脚本（按顺序逐步执行） |
| [`scripts/optimize_aio.py`](optimize_aio.py) | ONNX 后期优化（量化、simplify） |
| [`scripts/requirements.txt`](requirements.txt) | `optimize_aio.py` 的 Python 依赖 |
| [`scripts/run_python_baseline.py`](run_python_baseline.py) | PyTorch 基线推理（v2 / v2Pro / v2ProPlus） |
| [`scripts/compare_infer_metrics.py`](compare_infer_metrics.py) | Rust vs Python 时长/RMS/SV/波形相似度（v2ProPlus） |
| [`scripts/compare_onnx_versions.py`](compare_onnx_versions.py) | Rust vs Python 对比 **v2Pro + v2ProPlus**（T2S 输入 + 波形） |
| [`scripts/sampling_defaults.py`](sampling_defaults.py) | 内置采样默认值（与 Rust `InferParams::default()` 一致） |
| [`scripts/reports/`](reports/) | 对比脚本输出的 JSON 报告（已 gitignore，勿放仓库根目录） |
| [`scripts/sv_fbank_parity.py`](sv_fbank_parity.py) | Python Kaldi fbank → SV ONNX 统计 |

## 重要：必须覆盖上游 ONNX 模块

最新版 GPT-SoVITS 自带的 `t2s_model_onnx.py` 使用内部 cache 字典，与 Rust 推理所需的 **外部 KV-cache** 设计不兼容。导出前**必须**将 `patch/GPT_SoVITS/` 覆盖到上游项目的 `GPT_SoVITS/` 目录。

## macOS 分阶段导出

仅支持 **macOS**。请**按顺序**逐步运行各脚本（每步独立、可重复执行，避免长时间一键等待）：

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 1 | [`01_clone.sh`](export/01_clone.sh) | 克隆/更新 GPT-SoVITS → `gpt-sovits-upstream/` |
| 2 | [`02_patch.sh`](export/02_patch.sh) | 应用 `patch/GPT_SoVITS/` |
| 3 | [`03_setup_env.sh`](export/03_setup_env.sh) | conda（优先）或 uv 环境 |
| 4 | [`04_download_models.sh`](export/04_download_models.sh) | 下载权重并 staging `gpt.ckpt` / `sovits.pth`；含 G2PW、g2p_en |
| 5 | [`05_export.sh`](export/05_export.sh) | `export_onnx_v2.py` + `optimize_aio.py` |

### 示例（V2Pro / V2ProPlus）

```bash
# 1. 克隆上游（只需一次，或更新时重跑）
./scripts/export/01_clone.sh

# 2. 打 patch（patch 或 onnx-rs 更新后重跑）
./scripts/export/02_patch.sh

# 3. 安装环境（conda 优先；两者都有时用 conda）
./scripts/export/03_setup_env.sh --source HF

# 4. 下载并 staging 权重（--version 必填）
./scripts/export/04_download_models.sh --version v2Pro

# 5. 导出 ONNX（--version 与 --export-name 必填）
./scripts/export/05_export.sh --version v2Pro --export-name custom --no-quant

# V2ProPlus
./scripts/export/04_download_models.sh --version v2ProPlus
./scripts/export/05_export.sh --version v2ProPlus --export-name custom_v2proplus --no-quant
./scripts/export/validate_bundle.sh --bundle-dir gpt-sovits-upstream/onnx-patched/custom_v2proplus --expect-v2pro
```

输出目录：`gpt-sovits-upstream/onnx-patched/{export_name}/`

常用可选参数：

- `03_setup_env.sh`：`--source HF|HF-Mirror|ModelScope`，`--skip-env`（环境已就绪时跳过）
- `04_download_models.sh`：`--source …`，`--version v2|v2Pro|v2ProPlus`
- `05_export.sh`：`--output-dir PATH`，`--quant` / `--no-quant`（默认不量化），`--smoke-test`

某步已完成时可**只重跑后续步骤**，无需从头再来。

### 环境管理器

| 检测顺序 | 行为 |
|----------|------|
| 有 conda | `conda create -n GPTSoVits python=3.10` + 上游 `install.sh --device MPS` |
| 仅有 uv | `.venv` + `uv pip install` torch/requirements；`04` 负责模型与 NLTK/OpenJTalk |
| 都没有 | 报错并提示安装 Miniconda 或 uv |

## 手动导出流程（任意平台）

1. 克隆 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)，按上游 README 安装依赖并下载模型。
2. 将 `patch/GPT_SoVITS/` **整体覆盖**到上游 `GPT_SoVITS/`。
3. 准备模型目录 `{model_path}/gpt.ckpt` + `sovits.pth`。
4. 在上游根目录执行：

```bash
# V2
python GPT_SoVITS/export_onnx_v2.py --model_path ./models/v2 --export_name custom --version v2

# V2Pro
python GPT_SoVITS/export_onnx_v2.py --model_path ./models/v2pro --export_name custom --version v2Pro

# V2ProPlus
python GPT_SoVITS/export_onnx_v2.py --model_path ./models/v2proplus --export_name custom --version v2ProPlus

# 自动检测版本
python GPT_SoVITS/export_onnx_v2.py --model_path ./models/v2pro --export_name custom --auto-version
```

V2Pro 还需 SV 权重：`GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt`。

5. 拷贝 `GPT_SoVITS/text/G2PWModel/g2pW.onnx` 到 `onnx/{export_name}/`。
6. 运行优化（建议首次 `--no-quant`）：

```bash
python scripts/optimize_aio.py --input-dir onnx/custom --output-dir onnx-patched/custom --no-quant
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
| `sv.onnx` | — | ✓ |
| `{name}.json` | 元数据（含 `NumLayers`、`IsV2Pro`） | 同上 |
| `g2pW.onnx` | ✓ | ✓ |
| `g2p_en/` | ✓ | ✓ |
| `ref.wav` | 推荐 | 推荐 |

## 推理参数（内置默认值）

采样参数内置在程序中，**无需** JSON 配置文件：

| 参数 | 默认值 |
|------|--------|
| `top_k` | 4 |
| `top_p` | 0.9 |
| `temperature` | 1.0 |
| `repetition_penalty` | 1.35 |
| `seed` | 42 |

- Rust：`InferParams::default()`；demo 可用 `--top-k` 等 CLI 覆盖
- Python：[`scripts/sampling_defaults.py`](sampling_defaults.py)；`run_python_baseline.py` 内置相同默认值
- 可选：通过 `--params path/to.json` 覆盖部分采样字段（仅当你需要非默认配置时）
- 文本：`--text` / `--ref-text` 始终由 CLI 传入（有内置默认值）

## Rust 推理验证

将 [示例参考音频 ref.wav](https://huggingface.co/mikv39/gpt-sovits-onnx-custom/blob/main/ref.wav) 放到模型目录（与 ONNX 文件同级）：

```bash
MODEL_DIR=/path/to/onnx-patched/custom
curl -fL -o "${MODEL_DIR}/ref.wav" \
  https://huggingface.co/mikv39/gpt-sovits-onnx-custom/resolve/main/ref.wav
```

```bash
cargo run --release --example gpt_sovits_demo -- \
  --model-path "${MODEL_DIR}" \
  --text "今天天气真不错。" \
  --ref-text "格式化，可以给自家的奶带来大量的。" \
  --output output.wav
```

可用 CLI 覆盖采样参数：

```bash
cargo run --release --example gpt_sovits_demo -- \
  --model-path "${MODEL_DIR}" \
  --ref-text "格式化，可以给自家的奶带来大量的。" \
  --text "今天天气真不错。" \
  --top-k 4 --top-p 0.9 --temperature 1.0 --repetition-penalty 1.35 --seed 42
```

V2Pro / V2ProPlus 目录需包含 `sv.onnx`；demo 会自动检测。`05_export.sh --smoke-test` 也需要模型目录下已有 `ref.wav`。

### Rust vs Python 对比

报告写入 **`scripts/reports/`**（不在仓库根目录）：

```bash
# v2Pro + v2ProPlus 一键对比（T2S 输入、时长、波形 corr）
gpt-sovits-upstream/.venv/bin/python scripts/compare_onnx_versions.py

# 仅 v2ProPlus + SV 嵌入对比
gpt-sovits-upstream/.venv/bin/python scripts/compare_infer_metrics.py \
  --model-path gpt-sovits-upstream/onnx-patched/custom_v2proplus
```

输出示例：

| 脚本 | 报告路径 | WAV 输出 |
|------|----------|----------|
| `compare_onnx_versions.py` | `scripts/reports/compare_onnx_versions.json` | `output_v2pro.wav`, `output_v2proplus.wav`, `python_v2pro.wav`, `python_v2proplus.wav` |
| `compare_infer_metrics.py` | `scripts/reports/compare_infer_metrics.json` | `output.wav`, `python_v2proplus.wav` |

速度与 PyTorch 对比（CPU、v2Pro / v2ProPlus）：[`doc/pytorch_vs_onnx_speed.md`](../doc/pytorch_vs_onnx_speed.md)

调试 T2S 输入（phones / prompts / fs_decoder argmax）：

```bash
cargo run --release --example dump_rust_t2s -- \
  --model-path gpt-sovits-upstream/onnx-patched/custom_v2proplus \
  --text "今天天气真不错。" \
  --ref-text "格式化，可以给自家的奶带来大量的。"
```

导出后可用 [`validate_bundle.sh`](export/validate_bundle.sh) 检查产物是否完整；`v2ProPlus` 端到端验证记录见 [`doc/v2proplus_validation.md`](../doc/v2proplus_validation.md)。

## 预处理一致性测试

```bash
GPT_SOVITS_ROOT=~/gpt-sovits-upstream python3 scripts/preprocess_parity.py > /tmp/python_preprocess.json
PYTHON_PREPROCESS_JSON=/tmp/python_preprocess.json cargo test preprocess_parity_vs_python -- --ignored
```

语料：`resource/preprocess_corpus.json`。
