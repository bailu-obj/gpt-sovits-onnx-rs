# PyTorch 与 Rust ONNX 的整段推理耗时

2026 年 7 月 9 日在 macOS CPU 上对照 v2Pro 和 v2ProPlus。两端使用同一文本、同一参考音
和默认采样配置，模型及参考音就绪后计时。表中为 10 轮结果，Python 的两轮预热单独排除。

| 模型 | PyTorch `tts.run()` | Rust `synthesize_sync` | 加速倍数 |
| --- | ---: | ---: | ---: |
| v2Pro | 910 ms | 400 ms | 约 2.3 倍 |
| v2ProPlus | 992 ms | 548 ms | 约 1.8 倍 |

PyTorch 单次模型加载约 760 ms（v2Pro）和 823 ms（v2ProPlus）。上表计时从模型加载、
参考音处理完成后开始。Python 每次仍执行 BERT 和 G2P，Rust 在 `process_reference_sync`
完成后开始测量 `synthesize_sync`。

## 单轮阶段耗时

| 模型与阶段 | PyTorch | Rust ONNX |
| --- | ---: | ---: |
| v2Pro 文本到语义 | 586 ms | 238 ms |
| v2Pro VITS | 157 ms | 149 ms |
| v2Pro 合成总耗时 | 910 ms | 410 ms |
| v2ProPlus 文本预处理 | 61 ms | 20 ms |
| v2ProPlus 文本到语义 | 586–592 ms | 241 ms |
| v2ProPlus VITS | 234–239 ms | 289 ms |
| v2ProPlus 合成总耗时 | 910–992 ms | 548 ms |

Rust v2ProPlus 文本到语义阶段约由编码器 2.5 ms、首步 46 ms 和逐 token 解码 193 ms 组成。

## 复现

在仓库根目录使用导出环境的 Python，给两端传入同一参考 WAV 和转录。

```bash
REF_WAV=/path/to/ref.wav
REF_TEXT='<参考音频的准确转录>'
TEXT='<待合成文本>'
MODEL_DIR=/path/to/ordinary-model-bundle

python scripts/run_python_baseline.py --versions v2Pro v2ProPlus \
  --ref-audio "$REF_WAV" --ref-text "$REF_TEXT" --text "$TEXT"

cargo run --release --example gpt_sovits_demo -- \
  --model-path "$MODEL_DIR" --ref-audio "$REF_WAV" --ref-text "$REF_TEXT" --text "$TEXT" \
  --run-count 10 --output output.wav
```

加入 `RUST_LOG=gpt_sovits_onnx_rs=debug` 可查看 Rust 分阶段耗时。
[近期性能记录](perf_record.md) · [v2ProPlus 输出对照](v2proplus_validation.md)
