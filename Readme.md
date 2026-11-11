# GPT-SoVITS ONNX RS

[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 是一个少样本语音克隆项目。推理时，
GPT 根据待合成文本和参考音生成语义 token，SoVITS 再根据 token 与参考音条件生成波形。
提供一段清晰的参考录音和准确转录，即可用相应音色合成新文本。

GPT-SoVITS ONNX RS 将官方预训练权重导出为 ONNX，提供面向 x86-64 和 arm64 CPU 的
Rust 推理库。项目源于 Bailu 的桌面和 Android 语音对话需求，目标是在设备本地完成合成，
让应用拿到首块 PCM 就能开播，同时支持按需切换参考音。它可用于语音助手、聊天朗读和离线
旁白，也提供模型导出、校验脚本和生成 WAV 的 CLI。

**Android CPU 实时性优化已在 Galaxy S25 Ultra 上验证。** macOS 和 Android 对话推荐
**v2Pro + CPU Adaptive 句内流式**。需要生成完整音频时，可以使用同权重导出的普通 VITS 包。

## 快速开始

### 下载预导出模型（推荐）

[Hugging Face 模型仓](https://huggingface.co/mikv39/gpt-sovits-onnx-custom) 提供在 macOS
导出的 v2Pro `quality` CPU 模型。根目录的 `vits.onnx` 是推荐的 Adaptive 流式图。
用 `uv` 下载后，直接跳到 [macOS 构建与使用](#3-macos-构建与使用)。已有 Hugging Face CLI
时也可以直接运行 `hf download`。

```bash
STREAM_DIR=models/v2pro-quality-stream
uvx --from huggingface_hub hf download mikv39/gpt-sovits-onnx-custom \
  --local-dir "$STREAM_DIR" --exclude vits_non_streaming.onnx
```

整段合成使用同仓库的 `vits_non_streaming.onnx`。它与流式图共用其余模型文件，运行前在普通
模型目录中将其保存为 `vits.onnx`。[非流式下载与运行示例](https://huggingface.co/mikv39/gpt-sovits-onnx-custom#download-and-run)

自定义权重、量化或 Android `mobile` 模型包可按下面的步骤导出。

### 1. 在 macOS 导出模型

在仓库根目录执行。准备 Git、Rust、CMake 和 FFmpeg。首次导出推荐用 `uv` 准备 Python
环境。后续导出命令在同一终端执行。

```bash
export ENV_MANAGER=uv
./scripts/export/01_clone.sh --upstream-ref 04a44eb977586645f33594b72af13a035a2056c6
./scripts/export/02_patch.sh
./scripts/export/03_setup_env.sh --source HF
./scripts/export/04_download_models.sh --version v2Pro
```

已有 `gpt-sovits-upstream/.venv` 且装齐上游依赖和 [ONNX 工具](scripts/requirements.txt) 时，
可以跳过步骤 03，复用本地已安装的包。

根据运行平台选择一组命令。两个模型包都在 macOS 导出。

**macOS 使用 `quality`。**

```bash
./scripts/export/05_export.sh --version v2Pro --export-name v2pro-macos \
  --quant --profile quality --shared-kv --preoptimize-cpu
MODEL_DIR=gpt-sovits-upstream/onnx-patched/v2pro-macos
```

**Android 使用 `mobile`。**

```bash
./scripts/export/05_export.sh --version v2Pro --export-name v2pro-android \
  --quant --profile mobile --shared-kv --preoptimize-cpu
MODEL_DIR=gpt-sovits-upstream/onnx-patched/v2pro-android
```

### 2. 生成流式模型包

`05_export.sh` 产出可直接用于整段推理的普通 VITS。推荐的流式流程继续拆分参考音图，再生成
带 `stream_start_frame` 的流式 VITS。可以用 `uv` 准备的 Python，也可以把 `PYTHON` 指向
本地已安装 `onnx` 和 `onnxruntime` 的 Python。

```bash
PYTHON=gpt-sovits-upstream/.venv/bin/python

CACHED_DIR="${MODEL_DIR}-cached"
STREAM_DIR="${MODEL_DIR}-stream"
"$PYTHON" scripts/vits_dynamic.py "$MODEL_DIR/vits.onnx" --split-reference-dir "$CACHED_DIR"
rsync -a --exclude='/vits.onnx' --exclude='/vits_ref.onnx' \
  --exclude='/ref.wav' --exclude='/ref.txt' --exclude='/reference.wav' --exclude='/reference.txt' \
  "$MODEL_DIR/" "$CACHED_DIR/"
mkdir -p "$STREAM_DIR"
rsync -a --exclude='/vits.onnx' "$CACHED_DIR/" "$STREAM_DIR/"
"$PYTHON" scripts/vits_streaming.py "$CACHED_DIR/vits.onnx" "$STREAM_DIR/vits.onnx"
./scripts/export/validate_bundle.sh --python "$PYTHON" \
  --bundle-dir "$STREAM_DIR" --expect-v2pro --expect-streaming
```

### 3. macOS 构建与使用

准备稳定版 Rust、CMake 和 Xcode Command Line Tools。`cargo run` 会构建并运行示例。
参考音与模型包分开保存，运行时用 `--ref-audio` 指定 WAV，用 `--ref-text` 传入准确转录。
省略 `--ref-audio` 时，CLI 读取当前工作目录的 `ref.wav`。

```bash
cargo run --locked --release --example gpt_sovits_demo -- \
  --model-path "$STREAM_DIR" --stream \
  --ref-audio /path/to/ref.wav \
  --ref-text '<参考音频的准确转录>' --text '等雨停下来以后，我们可以沿着河边慢慢走到灯塔，再去那家小店坐一会儿。' \
  --output stream.wav
```

### 4. Android 构建与使用

在 macOS 或 Linux 上准备 Android NDK、CMake、`curl` 和 `unzip`。设备端验证使用
NDK r29（29.0.14206865）和 Android SDK Platform-Tools 的 `adb`。

```bash
rustup target add aarch64-linux-android
export ANDROID_NDK_ROOT=/path/to/android-ndk
./build_for_android.sh

DEVICE_DIR=/data/local/tmp/gpt-sovits
adb shell "mkdir -p '$DEVICE_DIR'"
adb push target/android/gpt_sovits_demo "$DEVICE_DIR/gpt_sovits_demo"
adb push target/android/lib "$DEVICE_DIR/lib"
adb push "$STREAM_DIR" "$DEVICE_DIR/models"
adb push /path/to/ref.wav "$DEVICE_DIR/ref.wav"
adb shell "chmod 700 '$DEVICE_DIR/gpt_sovits_demo'"
adb shell "LD_LIBRARY_PATH='$DEVICE_DIR/lib' '$DEVICE_DIR/gpt_sovits_demo' \
  --ort-library '$DEVICE_DIR/lib/libonnxruntime.so' --model-path '$DEVICE_DIR/models' \
  --ref-audio '$DEVICE_DIR/ref.wav' \
  --stream --ref-text '<参考音频的准确转录>' --text '等雨停下来以后，我们可以沿着河边慢慢走到灯塔，再去那家小店坐一会儿。' \
  --output '$DEVICE_DIR/stream.wav'"
adb pull "$DEVICE_DIR/stream.wav" target/android/stream.wav
```

设备上使用新的测试目录。构建产物包含 CLI、运行库和演示整段推理的 JNI 示例。

## 支持范围

| 能力 | 当前支持 |
| --- | --- |
| 模型 | v2、v2Pro、v2ProPlus 整段推理。v2Pro 流式模型在 macOS 导出，已在 macOS 和 Android 实测 |
| 语言 | 普通话和英文自动处理，支持中英混读。粤语使用 CLI 的 `--lang yue` 或 Rust 的 `LangId::AutoYue` |
| 音色 | WAV 和准确转录作为参考输入。Pro 系列使用 `sv.onnx` 提取说话人特征，可拆分参考图并缓存 `ge` |
| 部署 | Rust 库、命令行示例和 Android JNI 示例。下文的性能数据来自 macOS Apple Silicon 和 Android arm64 |

推荐的 v2Pro 模型包使用官方 `s1v3.ckpt` 和 `v2Pro/s2Gv2Pro.pth`。参考音建议选用
3–10 秒的清晰录音。

## 推理流程

1. **文本前端**先断句和规范化文本，再结合 BERT、G2PW 和英文 G2P 生成发音和文本特征。
2. **T2S** 根据文本特征和参考语义逐步生成语义 token。
3. **VITS** 根据语义 token 和参考音条件生成 PCM。流式模式在 T2S 继续解码时就开始输出音频。

### CPU 推理优化

- **共享 KV 缓冲区**减少 T2S 逐 token 解码时的缓存分配与拷贝。
- **参考音特征缓存**通过 `vits_ref.onnx` 在设置或切换参考音时计算 `ge`，后续合成直接复用。
- **选择性预融合与量化**在导出时完成适合 CPU 的 ONNX Runtime 图融合，减少加载时的优化开销。
  `quality` 对 BERT 和 G2PW 的 MatMul 权重做 INT8 量化，保留浮点语音生成。`mobile` 对逐 token
  解码使用 INT4 投影，可按需要校准部分 VITS 卷积。[模型导出与量化配置](scripts/README.md)
- **Adaptive 句内流式**让 VITS 从指定帧开始生成当前块及拼接所需上下文，运行时按播放余量和
  合成速度调整块长。应用可在收到首块 PCM 后开始播放。

在同一模型实例上重新调用 `process_reference_sync`，即可用新的 WAV 路径和转录切换音色。

## 性能

v2Pro CPU Adaptive 以 8 个语义 token 起步，使用相同的 25 条文本做热机测试。
RTF 为合成耗时除以音频时长。

| 设备 | 平均首包（TTFA） | 平均 RTF |
| --- | ---: | ---: |
| Mac M4 Pro | 175 ms | 0.108 |
| Galaxy S25 Ultra | 386 ms | 0.271 |

两组测试的模拟断供均为 0 次。S25 Ultra 另一次合成了 72 秒的多句音频，RTF 为 0.219。
更长的单句实测出现过重复和漏读，长文可按自然句提交。
[测试口径、分场景结果与长语音记录](doc/perf_record.md#cpu-句内流式)

## CPU 推理接口

| 用途 | 模型包 | CLI | Rust |
| --- | --- | --- | --- |
| 流式，边合成边播放 | 下载或步骤 2 生成的 `$STREAM_DIR`，VITS 含 `stream_start_frame` | `--stream` | `synthesize_token_chunks_sync` 回调逐块交付 PCM |
| 非流式，生成完整音频 | 下载普通图或使用步骤 1 的 `$MODEL_DIR` | 默认模式 | `synthesize_sync` 返回 `(WavSpec, Vec<f32>)` |

整段推理将上面命令的 `--model-path` 指向使用普通 VITS 的模型目录，并省略 `--stream`。
已有模型包可用 `--expect-ordinary` 或 `--expect-streaming` 校验。

Rust 调用方用 `TTSModel::from_bundle` 加载模型，再按上表选择接口。流式接口缺少对应的
VITS 图时返回错误。[Rust 命令行示例](examples/gpt_sovits_demo.rs) · [Android JNI 示例](examples/gpt_sovits_demo_jni.rs)

## 更新记录

### 2026-11

- Android CPU 实时性优化加入共享 KV、参考音缓存和选择性 CPU 预融合。
- 普通 VITS 通过 `synthesize_sync` 整段交付，流式 VITS 通过 `synthesize_token_chunks_sync`
  使用 Adaptive。旧异步 `synthesize` 调用方可按交付方式迁移。
- 增加独立模型校验和 Android 构建，公开 Mac M4 Pro 与 S25 Ultra 实测数据。
- 增加手动加载的 Qualcomm QNN pipeline，支持 128 帧 W8A16 HTP 逐窗口交付和设备端 `EPContext`。

### 2026-07

- 对齐 GPT-SoVITS Python 的中文预处理、参考音填充和 T2S 采样。
- 验证 v2Pro 与 v2ProPlus 的导出和推理。[v2ProPlus 验证](doc/v2proplus_validation.md)

## 可选的 Qualcomm QNN pipeline

在支持 Qualcomm HTP 的 Android 设备上用 `./build_for_android.sh --qnn` 准备运行库。
运行时用 `--model-path` 指定独立 QNN 包，用 `--qnn-libraries` 显式启用。
[QNN 模型准备与真机运行](scripts/README.md#qualcomm-qnn-pipeline) · [QNN 性能记录](doc/perf_record.md#qualcomm-qnn-pipeline)
