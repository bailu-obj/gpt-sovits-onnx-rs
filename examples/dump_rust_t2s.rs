use clap::Parser;
use gpt_sovits_onnx_rs::*;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long)]
    params: Option<PathBuf>,
    #[arg(long, default_value = "格式化，可以给自家的奶带来大量的。")]
    ref_text: String,
    #[arg(
        long,
        default_value = "你好啊，这是一个测试。吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。This demo is only for test  usage. If you find any 问题, 请修复它。"
    )]
    text: String,
}

fn find_model_prefix(assets_dir: &PathBuf) -> Result<String, GSVError> {
    for entry in std::fs::read_dir(assets_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(prefix) = name.strip_suffix("_vits.onnx") {
            return Ok(prefix.to_string());
        }
    }
    Err(GSVError::from("no *_vits.onnx in model path"))
}

fn resolve_infer_params(args: &Args) -> Result<InferParams, GSVError> {
    if let Some(path) = &args.params {
        InferParams::from_file(path)
    } else {
        Ok(InferParams::default())
    }
}

fn main() -> Result<(), GSVError> {
    let args = Args::parse();
    let _infer = resolve_infer_params(&args)?;
    let prefix = find_model_prefix(&args.model_path)?;
    let mut model = TTSModel::new(
        args.model_path.join(format!("{prefix}_vits.onnx")),
        args.model_path.join("ssl.onnx"),
        args.model_path.join(format!("{prefix}_t2s_encoder.onnx")),
        args.model_path
            .join(format!("{prefix}_t2s_fs_decoder.onnx")),
        args.model_path.join(format!("{prefix}_t2s_s_decoder.onnx")),
        Some(args.model_path.join("bert.onnx")),
        Some(args.model_path.join("g2pW.onnx")),
        None,
        Some(args.model_path.join("sv.onnx")),
    )?;
    model.process_reference_sync(
        args.model_path.join("ref.wav"),
        &args.ref_text,
        LangId::Auto,
    )?;
    let report = model.debug_t2s_snapshot(&args.text, LangId::Auto)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| GSVError::from(e.to_string()))?
    );
    Ok(())
}
