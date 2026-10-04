use clap::Parser;
use gpt_sovits_onnx_rs::*;
use hound::{WavSpec, WavWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "gpt-sovits-upstream/onnx-patched/v2pro-quality")]
    model_path: PathBuf,
    /// Optional JSON file to override built-in sampling defaults.
    #[arg(long)]
    params: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    run_count: usize,
    #[arg(long)]
    text: String,
    #[arg(long, default_value = "zh")]
    lang: String,
    #[arg(long)]
    ref_text: String,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    repetition_penalty: Option<f32>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value = "output.wav")]
    output: String,
}

struct TimingStats {
    avg: f64,
    median: f64,
    max: f64,
    min: f64,
}

impl TimingStats {
    fn new(times: &[f64]) -> Self {
        let count = times.len() as f64;
        let sum: f64 = times.iter().sum();
        let avg = sum / count;
        let mut sorted = times.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        Self {
            avg,
            median,
            max: *sorted.last().unwrap_or(&0.0),
            min: *sorted.first().unwrap_or(&0.0),
        }
    }

    fn print(&self, mode: &str, runs: usize) {
        println!("{} Inference ({} runs):", mode, runs);
        println!("  Average: {:.2} ms", self.avg);
        println!("  Median: {:.2} ms", self.median);
        println!("  Max: {:.2} ms", self.max);
        println!("  Min: {:.2} ms", self.min);
    }
}

fn resolve_infer_params(args: &Args) -> Result<InferParams, GSVError> {
    let mut params = if let Some(path) = &args.params {
        InferParams::from_file(path)?
    } else {
        InferParams::default()
    };

    if let Some(top_k) = args.top_k {
        params.top_k = top_k;
    }
    if let Some(top_p) = args.top_p {
        params.top_p = top_p;
    }
    if let Some(temperature) = args.temperature {
        params.temperature = temperature;
    }
    if let Some(repetition_penalty) = args.repetition_penalty {
        params.repetition_penalty = repetition_penalty;
    }
    if let Some(seed) = args.seed {
        params.seed = Some(seed);
    }

    Ok(params)
}

fn create_model(assets_dir: &Path) -> Result<TTSModel, GSVError> {
    if !assets_dir.exists() {
        return Err(GSVError::FileNotFound(format!(
            "Assets directory not found: {:?}",
            assets_dir
        )));
    }
    TTSModel::new(
        assets_dir.join("vits.onnx"),
        assets_dir.join("ssl.onnx"),
        assets_dir.join("t2s_encoder.onnx"),
        assets_dir.join("t2s_fs_decoder.onnx"),
        assets_dir.join("t2s_s_decoder.onnx"),
        Some(assets_dir.join("bert.onnx")),
        Some(assets_dir.join("g2pW.onnx")),
        Some(assets_dir.join("g2p_en")),
        match assets_dir.join("sv.onnx").exists() {
            true => Some(assets_dir.join("sv.onnx")),
            false => None,
        },
    )
}

fn write_wav(spec: WavSpec, samples: &[f32], filename: &str) -> Result<(), GSVError> {
    let mut writer = WavWriter::create(filename, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

fn run_sync_inference(
    model: &mut TTSModel,
    infer: &InferParams,
    text: &str,
    lang: &str,
    runs: usize,
    output_file: &str,
) -> Result<TimingStats, GSVError> {
    let mut times = Vec::with_capacity(runs);
    let mut lang_id = LangId::Auto;
    if lang == "yue" {
        lang_id = LangId::AutoYue;
    }
    let sampling = infer.to_sampling_params();
    for i in 0..runs {
        let start = Instant::now();
        let (spec, samples) =
            model.synthesize_sync(text, sampling, lang_id, PostprocessParams::default())?;
        if i == runs - 1 {
            write_wav(spec, &samples, output_file)?;
        }
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(TimingStats::new(&times))
}

fn main() -> Result<(), GSVError> {
    env_logger::init();
    let args = Args::parse();
    let infer = resolve_infer_params(&args)?;

    let mut model = create_model(&args.model_path)?;
    model.process_reference_sync(
        args.model_path.join("ref.wav"),
        &args.ref_text,
        LangId::Auto,
    )?;

    println!("text: {:?} ref_text: {:?}", args.text, args.ref_text);
    println!(
        "infer params: top_k={} top_p={} temperature={} repetition_penalty={} seed={:?}",
        infer.top_k, infer.top_p, infer.temperature, infer.repetition_penalty, infer.seed
    );

    let stats = run_sync_inference(
        &mut model,
        &infer,
        &args.text,
        &args.lang,
        args.run_count,
        &args.output,
    )?;
    stats.print("Synchronous", args.run_count);
    println!("Wrote {}", args.output);

    Ok(())
}
