use clap::Parser;
use gpt_sovits_onnx_rs::*;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long)]
    params: Option<PathBuf>,
    #[arg(long)]
    ref_text: String,
    #[arg(long)]
    text: String,
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
    let mut model = TTSModel::new(
        args.model_path.join("vits.onnx"),
        args.model_path.join("ssl.onnx"),
        args.model_path.join("t2s_encoder.onnx"),
        args.model_path.join("t2s_fs_decoder.onnx"),
        args.model_path.join("t2s_s_decoder.onnx"),
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
