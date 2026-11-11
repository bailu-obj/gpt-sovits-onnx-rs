# GPT-SoVITS ONNX RS

GPT-SoVITS ONNX RS 是面向 x86-64 和 arm64 CPU 的本地语音合成引擎。项目源于 Bailu 的
桌面和 Android 语音对话需求，用 Rust 和 ONNX Runtime 加载 GPT-SoVITS 模型，根据一段
参考音和对应转录合成同音色语音。推理引擎可作为 Rust 库嵌入应用。仓库也提供模型导出与校验
脚本，以及生成 WAV 的命令行示例。

**Android CPU 实时性优化已在 Galaxy S25 Ultra 上验证。** macOS 和 Android 对话推荐
**v2Pro + CPU Adaptive 句内流式**。需要生成完整音频时，可以使用同权重导出的普通 VITS 包。

## 快速开始

### 1. 在 macOS 导出模型

在仓库根目录执行。准备 Git、Rust、CMake、FFmpeg，以及 conda 或 uv。首次导出先准备上游代码和权重。

```bash
./scripts/export/01_clone.sh --upstream-ref 04a44eb977586645f33594b72af13a035a2056c6
./scripts/export/02_patch.sh
./scripts/export/03_setup_env.sh --source HF
./scripts/export/04_download_models.sh --version v2Pro
```

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
带 `stream_start_frame` 的流式 VITS。先在当前终端激活导出环境。

```bash
conda activate GPTSoVits
# 如果通过 uv 准备环境，改用 source gpt-sovits-upstream/.venv/bin/activate

CACHED_DIR="${MODEL_DIR}-cached"
STREAM_DIR="${MODEL_DIR}-stream"
python scripts/vits_dynamic.py "$MODEL_DIR/vits.onnx" --split-reference-dir "$CACHED_DIR"
rsync -a --exclude='/vits.onnx' --exclude='/vits_ref.onnx' \
  --exclude='/ref.wav' --exclude='/ref.txt' --exclude='/reference.wav' --exclude='/reference.txt' \
  "$MODEL_DIR/" "$CACHED_DIR/"
mkdir -p "$STREAM_DIR"
rsync -a --exclude='/vits.onnx' "$CACHED_DIR/" "$STREAM_DIR/"
python scripts/vits_streaming.py "$CACHED_DIR/vits.onnx" "$STREAM_DIR/vits.onnx"
./scripts/export/validate_bundle.sh --bundle-dir "$STREAM_DIR" --expect-v2pro --expect-streaming
```

### 3. macOS 构建与使用

`cargo run` 会构建并运行示例。参考音与模型包分开保存，运行时用 `--ref-audio` 指定 WAV，
用 `--ref-text` 传入准确转录。省略 `--ref-audio` 时，CLI 读取当前工作目录的 `ref.wav`。

```bash
cargo run --release --example gpt_sovits_demo -- \
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

推荐的 v2Pro 模型包使用官方 `s1v3.ckpt` 和 `v2Pro/s2Gv2Pro.pth`。`quality` 保留
T2S 和 VITS 的浮点计算，将 BERT 和 G2PW 的 MatMul 权重转为 INT8。`mobile` 对
逐 token 解码采用 INT4 投影，可按需要校准部分 VITS 卷积。建议使用 3–10 秒的清晰参考音。
[模型导出与量化配置](scripts/README.md)

## 推理流程

1. **文本前端**先断句和规范化文本，再结合 BERT、G2PW 和英文 G2P 生成发音和文本特征。
2. **T2S** 根据文本特征和参考语义生成语义 token。支持共享 KV 布局的模型会在逐 token 解码期间复用缓存。
3. **VITS** 根据语义 token 和参考音条件生成 PCM。整段模式汇总完整音频。流式模式在 T2S
   继续解码时开始生成可播放的 PCM，Adaptive 根据播放余量和近期合成耗时调整后续块长。

在同一模型实例上重新调用 `process_reference_sync`，即可用新的 WAV 路径和转录切换音色。
使用 `vits_ref.onnx` 的模型包也会在这一步缓存 `ge`，后续合成复用。

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
| 流式，边合成边播放 | 步骤 2 的 `$STREAM_DIR`，VITS 含 `stream_start_frame` | `--stream` | `synthesize_token_chunks_sync` 回调逐块交付 PCM |
| 非流式，生成完整音频 | 步骤 1 的 `$MODEL_DIR`，普通 VITS | 默认模式 | `synthesize_sync` 返回 `(WavSpec, Vec<f32>)` |

整段推理直接使用步骤 1 的模型包。把上面命令的 `--model-path` 指向普通包，并省略
`--stream`。已有模型包可用 `--expect-ordinary` 或 `--expect-streaming` 校验。

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
