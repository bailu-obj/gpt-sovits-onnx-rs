# 模型导出与部署

## 快速开始

macOS 和 Android 的对话场景都推荐 v2Pro CPU Adaptive。先在 macOS 导出模型，再在目标平台构建和运行。

| 目标平台 | 推荐导出参数 | 构建与运行 |
| --- | --- | --- |
| macOS | `05_export.sh --quant --profile quality --shared-kv --preoptimize-cpu` | [导出命令](../Readme.md#1-在-macos-导出模型) · [macOS 运行](../Readme.md#3-macos-构建与使用) |
| Android | `05_export.sh --quant --profile mobile --shared-kv --preoptimize-cpu` | [导出命令](../Readme.md#1-在-macos-导出模型) · [Android 构建与运行](../Readme.md#4-android-构建与使用) |

两个导出命令都先生成普通模型包。对话部署继续[生成流式模型包](../Readme.md#2-生成流式模型包)，运行时使用 `--stream`。

## 流式 VITS 导出

普通包和流式包都以 `vits.onnx` 为主图文件名，图的输入和调用入口如下。

| 模型包 | 生成步骤 | VITS 输入 | 使用方式 |
| --- | --- | --- | --- |
| 普通包 | `05_export.sh` | 文本、语义 token、参考音和可选说话人特征，拆分参考音后使用 `ge` | 非流式。CLI 默认模式或 Rust `synthesize_sync` 返回完整 PCM |
| 流式包 | 参考音拆分后运行 `vits_streaming.py` | 在普通图基础上增加 `stream_start_frame` | CLI `--stream` 或 Rust `synthesize_token_chunks_sync`，按 Adaptive 逐块返回 PCM |

校验脚本用 `--expect-ordinary` 或 `--expect-streaming` 检查图类型，`--expect-v2pro` 检查
Pro 系列模型接口和说话人模型。已有模型包可用 `--python /path/to/python` 指定装有
`onnx` 和 `onnxruntime` 的独立环境。缺少流式图时，流式接口返回错误。

```bash
./scripts/export/validate_bundle.sh --bundle-dir "$MODEL_DIR" --expect-v2pro --expect-ordinary
./scripts/export/validate_bundle.sh --bundle-dir "$STREAM_DIR" --expect-v2pro --expect-streaming
```

流式图保留语义前缀，按起始帧裁切 flow 和生成器的计算范围。Adaptive 根据播放余量和近期
合成耗时调整后续块长。默认前瞻 6 个 token，拼接重叠 2 个 token。命令行的
`--block-tokens` 默认 8，控制首块和后续块长下限。使用 `vits_streaming.py --latent-history`
可导出实验性的 flow 历史复用图，由 Rust 调用方通过 `TokenStreamOptions` 设置左侧历史。

## 参考音特征缓存与 CPU 预融合

v2Pro 和 v2ProPlus 可以用 `vits_dynamic.py --split-reference-dir` 将 VITS 拆成
`vits_ref.onnx` 和使用 `ge` 的 `vits.onnx`。引擎在加载或切换参考音时计算一次
`ge`，后续合成复用。拆图输出目录同时补齐其他模型后可用于普通推理，也是快速开始里
`$CACHED_DIR` 的来源。

`--preoptimize-cpu` 在量化之后对 BERT、中英文 G2P 和适用的 T2S 图做 ORT Extended
预融合。已有模型包可用 `python scripts/preoptimize_cpu.py "$CACHED_DIR"` 更新。
`--shared-kv` 为 T2S 逐 token 解码启用可复用的 KV 缓冲区。运行时通过
`past_length` 输入识别共享布局，并按长句需要扩容。[CPU 优化实测](../doc/perf_record.md#cpu-优化)

## Android CPU 低延迟导出

快速开始的 `mobile` 用于移动端部署。需要提高精度时，可以选择 `quality`。
`05_export.sh` 默认输出浮点图，传入 `--quant` 才应用所选配置。

| 配置 | BERT / G2PW | T2S 首步 | T2S 逐 token | VITS |
| --- | --- | --- | --- | --- |
| `quality` | 分块 INT8 权重，FP32 激活 | FP32 | FP32 | FP32 |
| `mobile` | 动态 INT8 投影 | 动态 INT8 投影 | 分块 INT4 投影，FP32 词表输出层 | 默认 FP32，可选卷积 INT8 校准 |
| `rtf` | 同 `quality` | FP32 | 分块 INT4 投影，FP32 词表输出层 | FP32 |
| `compact` | 分块 INT4 权重 | 动态 INT8 | 动态 INT8 | FP32 |

已有代表性输入时，给 `--quant --profile mobile` 加上
`--vits-calibration-dir /path/to/calibration`，校准 VITS 前九个残差块的卷积。
每个 `.npz` 包含 `text_seq`、`pred_semantic`、`ref_audio` 和 Pro 系列的 `sv_emb`。
更低位宽的 INT2、INT3 属于实验候选。评估时可对照图大小、首包和连续合成耗时、
语义 token、PCM、字错率及音色相似度。

## 运行参数与验证

- **采样。** 默认 `top_k=4`、`top_p=0.9`、`temperature=1.0`、`repetition_penalty=1.35`。
  命令行可覆盖这些值及 `--seed`，也可用 `--params` 指定 JSON。默认读取模型目录的
  `infer_params.json`，种子未设时随机采样。`--lang yue` 选择粤语模式。
- **参考音。** WAV 和转录由运行时提供。用 `--ref-audio` 指定外部 WAV，或使用当前工作目录的
  `ref.wav`。Pro 系列模型包包含 `sv.onnx`。导出自测通过
  `05_export.sh --smoke-test --ref-audio … --ref-text … --text …` 直接读取外部参考音。
- **线程。** 默认每个 ORT 会话使用独立线程池。宿主可启用 `shared-ort-pool`，并在加载模型前
  用 `GlobalThreadPoolOptions` 配置共享池。Android CLI 使用 4 个 ORT 池线程。
  `configure_onnx_sessions` 可在加载模型前设置等待自旋。
- **回归。** `cargo test --release --lib` 验证 Rust 逻辑。
  `python -m unittest discover -s scripts -p test_vits_dynamic.py` 验证动态 VITS 图。
  `compare_onnx_versions.py` 和 `compare_infer_metrics.py` 用同一个 `--ref-audio` 对照两端，
  输出位于 `scripts/reports/`。
  [性能记录](../doc/perf_record.md) · [v2ProPlus 验证](../doc/v2proplus_validation.md)

## 其他模型与自定义导出

分阶段脚本在 macOS 上运行。步骤 02 将子仓库的 `patch/GPT_SoVITS/` 覆盖到上游，以导出 Rust 所需的外部 KV。
步骤 03 会为 conda 或 uv 环境安装 [ONNX 工具](requirements.txt)。后续 Python 图处理命令使用同一个环境。
步骤 04 支持 `v2`、`v2Pro`、`v2ProPlus`。官方 v2Pro 权重为 `s1v3.ckpt` 和
`v2Pro/s2Gv2Pro.pth`。自定义检查点可通过 `05_export.sh --model-path PATH` 指定目录。

`--vits-only` 可用同权重刷新已有普通包的 VITS。旧 VITS 图可用
`vits_dynamic.py INPUT --output OUTPUT` 修复固定长度掩码。自定义或手动导出的包可用
`validate_bundle.sh` 核对 VITS、T2S、参考音和中英文 G2P 等必需文件。

## Qualcomm QNN pipeline

QNN 是 Android 上的可选后端。它使用普通 CPU VITS 包及配套的 128 帧 W8A16 HTP 生成器。
CPU 生成整句语义 token 和 VITS 潜变量，HTP 每完成一个窗口便交付裁边后的 PCM。
Bailu 的对话路径使用 CPU Adaptive。独立 CLI 通过 `--model-path` 和 `--qnn-libraries` 显式启用
QNN，Rust 调用方使用 `enable_qnn_pipeline` 后调用 `synthesize_qnn_pipeline`。

### 下载运行库并构建

```bash
rustup target add aarch64-linux-android
export ANDROID_NDK_ROOT=/path/to/android-ndk
./build_for_android.sh --qnn
```

脚本从 Maven Central 获取并校验 ORT、QNN provider 和 QAIRT 归档。使用
`--output-dir` 指定产物目录，`--archive-dir` 复用下载缓存。只准备运行库时可以运行
`./scripts/prepare_android_runtime.sh --output-dir target/android/lib --qnn`。
S25 Ultra 实测环境为 ORT 1.30.0、QNN EP 2.7.0、QAIRT 2.51.0 和 burst 模式。

### 准备同权重模型和校准输入

使用快速开始的 `$CACHED_DIR` 作为普通 CPU 包。用同一套权重另导出完整浮点 VITS，
保留 `ConvTranspose`，作为 HTP 生成器的输入。准备多条代表性中长句，每个参与校准的模型段
至少需要 64 个语义 token。

```bash
./scripts/export/05_export.sh --version v2Pro --export-name v2pro-qnn-fp32 \
  --no-quant --vits-conv-transpose
QNN_SOURCE=gpt-sovits-upstream/onnx-patched/v2pro-qnn-fp32/vits.onnx

mkdir -p target
cargo run --release --example gpt_sovits_demo -- \
  --model-path "$CACHED_DIR" --ref-audio /path/to/ref.wav --ref-text '<参考音频的准确转录>' \
  --text '<多条代表性中长句>' --seed 42 --dump-qnn-inputs target/qnn-calibration.json
python scripts/vits_generator.py "$QNN_SOURCE" target/qnn-generator \
  --frames 128 --htp-calibration-inputs target/qnn-calibration.json --htp-activation-bits 16

QNN_DIR=target/qnn-bundle
mkdir -p "$QNN_DIR"
rsync -a "$CACHED_DIR/" "$QNN_DIR/"
cp target/qnn-generator/vits_prefix.onnx target/qnn-generator/vits_generator_htp.onnx "$QNN_DIR/"
./scripts/export/validate_bundle.sh --bundle-dir "$QNN_DIR" --expect-qnn
```

可重复传入 `--htp-calibration-inputs` 合并多份 JSON，或用 `--htp-calibration-dir` 指定
现成的 `z`、`ge` 校准窗口。已验证的模型转换环境为 ONNX 1.23.2 和 ONNX Runtime 1.30.0。
`--expect-qnn` 校验普通 CPU VITS、HTP 边界及已有的 `EPContext`。

### 真机运行并生成 EPContext

```bash
DEVICE_DIR=/data/local/tmp/gpt-sovits-qnn
adb shell "mkdir -p '$DEVICE_DIR'"
adb push target/android/gpt_sovits_demo "$DEVICE_DIR/gpt_sovits_demo"
adb push target/android/lib "$DEVICE_DIR/lib"
adb push "$QNN_DIR" "$DEVICE_DIR/models"
adb push /path/to/ref.wav "$DEVICE_DIR/ref.wav"
adb shell "chmod 700 '$DEVICE_DIR/gpt_sovits_demo'"
adb shell "LD_LIBRARY_PATH='$DEVICE_DIR/lib' ADSP_LIBRARY_PATH='$DEVICE_DIR/lib' \
  '$DEVICE_DIR/gpt_sovits_demo' --model-path '$DEVICE_DIR/models' --qnn-libraries '$DEVICE_DIR/lib' \
  --ref-audio '$DEVICE_DIR/ref.wav' \
  --ref-text '<参考音频的准确转录>' --text '<至少一条中长句>' \
  --seed 42 --run-count 4 --output '$DEVICE_DIR/qnn.wav'"
adb pull "$DEVICE_DIR/qnn.wav" target/qnn.wav
```

首次运行可加 `--compile-qnn-context` 生成设备缓存。后续运行会优先加载该缓存。
`GptSovitsQnn` 日志的 `htp_ready` 和 `htp_sentence` 记录 HTP 启用情况和窗口计时。
复现性能表时丢弃预热轮，取后三轮中位数。PCM 对照需固定 `z`、`ge`，T2S 的 `--seed`
只固定语义采样。[QNN 实测与测试语料](../doc/perf_record.md#qualcomm-qnn-pipeline)
