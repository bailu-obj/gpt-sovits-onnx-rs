# v2ProPlus 导出与推理验证

2026 年 7 月 9 日在 macOS 验证了 v2ProPlus 的普通 VITS 导出、模型包校验和整段推理。
首次导出环境见[快速开始](../Readme.md#快速开始)。

```bash
./scripts/export/04_download_models.sh --version v2ProPlus
./scripts/export/05_export.sh --version v2ProPlus --export-name v2proplus --no-quant
MODEL_DIR=gpt-sovits-upstream/onnx-patched/v2proplus

./scripts/export/validate_bundle.sh --bundle-dir "$MODEL_DIR" --expect-v2pro --expect-ordinary
cargo run --release --example gpt_sovits_demo -- \
  --model-path "$MODEL_DIR" --ref-audio /path/to/ref.wav --ref-text '<参考音频的准确转录>' \
  --text '<待合成文本>' --output v2proplus.wav
```

普通 `vits.onnx` 接受 `text_seq`、`pred_semantic`、`ref_audio` 和 `sv_emb`。
Rust 加载包内的 `sv.onnx`，说话人嵌入形状为 `[1, 20480]`。验证时生成了 32 kHz
单声道 WAV，整段推理约 1,447 ms。

## 输出对照

| 指标 | Python | Rust |
| --- | ---: | ---: |
| 音频时长 | 2.38 s（76,160 样本） | 2.38 s（76,160 样本） |
| RMS | 0.00308 | 0.00283 |
| 峰值 | 0.026 | 0.027 |
| 说话人嵌入前部最大绝对差 | — | 小于 0.00001 |

```bash
python scripts/compare_infer_metrics.py --model-path "$MODEL_DIR" --seed 42 \
  --ref-audio /path/to/ref.wav --ref-text '<参考音频的准确转录>' --text '<待合成文本>'
```

对照包含 T2S 采样及停止行为、VITS 输出长度裁切和说话人模型的 Kaldi 滤波器组特征。
[整段推理耗时](pytorch_vs_onnx_speed.md)
