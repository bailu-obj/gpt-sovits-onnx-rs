# GPT-SOVITS-ONNX-RS

一个基于 **Rust** 和 **ONNX Runtime** 的轻量级、跨平台 GPT-SoVITS TTS 推理引擎，专为在 **x86/ARM** 架构的 **CPU** 上运行而设计。

## 项目简介

本项目旨在将 GPT-SoVITS (V2) 模型通过 ONNX Runtime 部署到各类 CPU 设备上，以实现低延迟、高可用的本地文本转语音（TTS）能力。它最初是为个人全平台 Chatbot 项目（尚未开源）的 Android 和 PC 端提供 TTS 支持而开发的。

将在保证可接受精度的前提下，对实时性进行持续优化。

目前的实时性优化已经接近尾声，后续考虑中英文混合场景的优化。

之前的账号由于一些原因被删除，导致原项目删除。

-----

## 核心特性

* **跨平台推理核心**：使用 Rust 编写，确保了内存安全与高性能，可轻松编译到 x86 和 ARM 平台（Linux, Android 等）。
* **一键式模型转换**：提供位于 `scripts` 目录的 Python 脚本，用于一键导出和优化 SoVITS 模型。**注意：** 优化后的模型结构与官方不兼容，转换时需使用本项目提供的完整脚本并按照文档操作。
* **完整的 Android 构建支持**：提供了 `build_for_android.sh` 脚本，自动化处理 ONNX Runtime 的源码下载、编译及项目构建，解决了官方 `ort-rs` 缺少 Android 预构建包的问题。


* 详细性能数据请参考 [**性能记录 (perf\_record)**](doc/perf_record.md)。
* 通过在运行时设置lang_id为LangId::AutoYue，可以启用粤语模式。

-----

## 项目状态与已知问题

2026-07-08: 重构推理预处理管线，对齐 Python TextPreprocessor/chinese2：tone sandhi、erhua、分段切句、短句补齐、英文 num2en、parity 测试，增强多语言输出效果。更新导出代码,方便自行导出模型。

2026-03-21: 优化了中英文混合效果（借助Cursor自动化编写）

2026-01-25: 测试了ort_rc11,但是在mac（arm）上性能更差（-10%），目前将deps固定在了ort_rc10。参见[update/ort_rc_11](https://github.com/bailu-obj/gpt-sovits-onnx-rs/tree/update/ort_rc_11)分支  

2025-11-06: 初步支持V2Pro模型，详情请见转换脚本。目前V2Pro的精度和速度仍未优化，且android平台仍未验证。

2025-07-12: 将部分onnx模型代码同步到和pytorch一致，简化了一部分模型逻辑，并修复了library中由于缺少空白声音导致的吸气问题，如果提示找不到输出，请更新hf上的新模型，或者重新转换自己的模型。

2025-06-30: 提取sampler到rust层，并更新了模型转换脚本和demo模型，如果提示找不到输出，请更新模型。


-----

## 方案对比

为了帮助您选择最适合的方案，我将其与社区主流项目进行了对比。

| 方案 | TTS 效果 | 性能 | 平台兼容性 | 易用性 |
| :--- | :--- | :--- | :--- | :--- |
| **sherpa-onnx** | ★★☆☆☆ (情感稍弱) | ★★★★★ (模型小，实时性强) | ★★★★★ (全平台) | ★★★★★ (官方预构建) |
| **[GPT-SoVITS-RS](https://github.com/second-state/gpt_sovits_rs)** | ★★☆★☆ (接近原版) | ★★★★☆ (依赖 Torch) | ★★☆☆☆ (Android 支持不佳) | ★★★☆☆ (需手动配置) |
| **本项目** | ★★☆☆☆ (不稳定) | ★★★★☆ (ONNX 优化) | ★★★☆☆ (支持 ARM/x86) | ★★★★☆ (Android 需手动执行构建脚本) |


其他TTS方案
| 方案 | TTS 效果 | 性能 | 平台兼容性 | 易用性 |
| :--- | :--- | :--- | :--- | :--- |
| **[Qwen3TTS](https://github.com/predict-woo/qwen3-tts.cpp)** | ★★★★★ (效果优秀) | ★★☆☆☆ (GGML优化，但是模型本身开销大) | ★★★★☆ (理论支持全平台，但平台性能要求高) | ★★★★☆ (需要转换模型) |

-----

## 使用建议

根据您的具体需求，推荐如下：

* **追求极致性能和易用性的 Android 平台**：
  * ✅ **推荐使用 `sherpa-onnx`**
* **追求高拟真度且在 x86 平台（Linux/Windows, CUDA/CPU）**：
  * ✅ **推荐使用 `GPT-SoVITS-RS`**
* **追求高拟真度且需要在 Android 和 x86 CPU 上运行**：
  * ✅ **可以尝试本项目**，并欢迎帮助改进！

-----

## 模型下载

如果您不想自行训练和导出模型，可以使用预训练模型进行快速体验。

* **主模型下载地址**：[huggingface.co/mikv39/gpt-sovits-onnx-custom](https://huggingface.co/mikv39/gpt-sovits-onnx-custom)
* 该模型可直接在 [gpt-sovits-android-demo](https://github.com/null-define/gpt-sovits-android-demo/tree/master) 中加载使用，或者直接替换examples下的gpt_sovits_demo中的模型地址。

> **版权声明**：此模型使用了受版权保护的音视频素材进行微调，请勿用于任何商业用途。

**gp2en模型下载** 建议下载，参见[cisco-ai/mini-bart-g2p](https://huggingface.co/cisco-ai/mini-bart-g2p/tree/main/onnx),下载完成后可以把模型目录文件夹设置为TTSModel的g2p_en_path参数，启用gp2 en模型支持。默认的demo和JNI都启用了gp2 en模型，需要在原来的目录下新建一个g2p_en文件夹，把下载的模型放进去。（macOS 导出流程会在 `04_download_models.sh` 自动下载 g2p_en。）

### 参考音频（ref.wav）

`gpt_sovits_demo` 会从模型目录读取 `ref.wav` 作为参考音色。可使用我们之前配套的示例音频：

- 文件：[mikv39/gpt-sovits-onnx-custom — ref.wav](https://huggingface.co/mikv39/gpt-sovits-onnx-custom/blob/main/ref.wav)
- 建议参考文本（与示例音频匹配）：`格式化，可以给自家的奶带来大量的。`

下载到模型目录（与 `custom_vits.onnx` 等同级）：

```bash
MODEL_DIR=/path/to/onnx-patched/custom
curl -fL -o "${MODEL_DIR}/ref.wav" \
  https://huggingface.co/mikv39/gpt-sovits-onnx-custom/resolve/main/ref.wav
```

运行 demo 时指定对应 `--ref-text`：

```bash
cargo run --release --example gpt_sovits_demo -- \
  --model-path "${MODEL_DIR}" \
  --ref-text "格式化，可以给自家的奶带来大量的。" \
  --text "今天天气真不错。"
```

参考音频建议 **3–10 秒**；过长可能导致上游训练/推理报错。

### 预处理说明

| 组件 | 路径参数 | 作用 |
|------|----------|------|
| `bert.onnx` | `bert_path` | 中文 BERT 特征（缺失时用零向量，语速/韵律下降） |
| `g2pW.onnx` | `g2pw_path` | 多音字 G2PW（缺失时用字典 fallback） |
| `g2p_en/` | `g2p_en_path` | 英文 G2P ONNX（缺失时用 CMUdict） |

- `LangId::Auto`：普通话 + 英文自动分词
- `LangId::AutoYue`：粤语模式
- 预处理与 Python 版对照：见 [scripts/README.md](scripts/README.md) 中的 parity 测试说明

-----

## 构建指南

### 1\. 模型转换

请参考 `scripts` 目录下的说明文档：[scripts/README.md](scripts/README.md)。

**macOS 分阶段导出**（逐步运行，避免长时间一键等待）：

```bash
./scripts/export/01_clone.sh
./scripts/export/02_patch.sh
./scripts/export/03_setup_env.sh --source HF
./scripts/export/04_download_models.sh --version v2Pro
./scripts/export/05_export.sh --version v2Pro --export-name custom --no-quant
```

详见 [scripts/README.md — macOS 分阶段导出](scripts/README.md#macos-分阶段导出)。

### 2\. x86 平台构建和运行 (Linux/Windows/macOS)

直接使用 Cargo 即可完成编译：

```bash
cargo build --release
```

使用如下命令可以运行命令行demo，该demo会自动根据模型路径下转换的模型文件，启用v2或v2Pro模型。需先将 [ref.wav](https://huggingface.co/mikv39/gpt-sovits-onnx-custom/blob/main/ref.wav) 下载到模型目录，见上文「参考音频」一节。

```bash
RUST_LOG=Debug cargo run --release --example gpt_sovits_demo -- \
  --model-path /path/to/onnx-patched/custom \
  --ref-text "格式化，可以给自家的奶带来大量的。" \
  --text "你好啊，我最喜欢你了"
```

### 3\. Android 平台构建

> ⚠️ **重要提示**
>
> 为确保 ONNX Runtime 版本的灵活性与及时更新，**本项目不提供预构建的二进制文件**。您需要根据以下步骤自行构建。

1. **环境准备**:
      * 安装 CMake ≥ 3.28 (推荐使用 Conda 安装以避免系统版本限制)。
      * 下载并配置 Android NDK 与 SDK，并设置好相关环境变量。
      * 在~/.cargo/config.toml中设置好`[target.aarch64-linux-android]`的linker和ar,注意androidN-clang的N最好>=28，
        * 最好保证build_for_android.sh中的android_api参数一致
        * 如果是较新版本的android系统，请使用16k page size `rustflags = ["-C", "link-arg=-Wl,-z,max-page-size=16384"]`

      ```toml
      [target.aarch64-linux-android]
      linker = "/android-ndk-r27c//toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android32-clang"
      ar = "/android-ndk-r27c//toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
      rustflags = ["-C", "link-arg=-Wl,-z,max-page-size=16384"]
      ```

2. **首次构建**:
      * 运行一键式脚本，该脚本将自动完成 ONNX Runtime 源码下载、编译，并构建适用于 Android 的可执行文件和动态库。
    <!-- end list -->
    ```bash
    ./build_for_android.sh
    ```

3. **后续增量构建**:
      * 如果仅修改了 Rust 代码，可直接使用 Cargo 命令进行编译。
    <!-- end list -->
    ```bash
    cargo build --target aarch64-linux-android --release --features jni --examples
    ```
