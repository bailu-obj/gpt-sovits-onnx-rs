use clap::Parser;
use gpt_sovits_onnx_rs::*;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long, default_value = "格式化，可以给自家的奶带来大量的。")]
    ref_text: String,
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

fn main() -> Result<(), GSVError> {
    let args = Args::parse();
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

    let sv_emb = model
        .sv_embedding()
        .ok_or_else(|| GSVError::from("sv embedding missing"))?;
    let flat: Vec<f32> = sv_emb.iter().copied().collect();
    let mean = flat.iter().sum::<f32>() / flat.len() as f32;
    let var = flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / flat.len() as f32;
    let report = serde_json::json!({
        "shape": sv_emb.shape(),
        "mean": mean,
        "std": var.sqrt(),
        "min": flat.iter().copied().fold(f32::INFINITY, f32::min),
        "max": flat.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        "head": &flat[..8.min(flat.len())],
    });
    println!("{}", report);
    Ok(())
}
