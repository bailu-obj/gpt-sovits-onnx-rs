use clap::Parser;
use gpt_sovits_onnx_rs::*;
use hound::{WavSpec, WavWriter};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "gpt-sovits-upstream/onnx-patched/v2pro-quality")]
    model_path: PathBuf,
    /// Override the model bundle's infer_params.json.
    #[arg(long)]
    params: Option<PathBuf>,
    #[arg(long, default_value = "1")]
    run_count: NonZeroUsize,
    #[arg(long)]
    text: String,
    #[arg(long, default_value = "zh")]
    lang: String,
    /// Reference WAV path. The default is ref.wav in the current directory.
    #[arg(long, default_value = "ref.wav")]
    ref_audio: PathBuf,
    /// Accurate transcript of the reference WAV.
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
    /// Stream PCM with adaptive blocks. Requires a streaming VITS graph.
    #[arg(long)]
    stream: bool,
    /// First adaptive block in semantic tokens.
    #[arg(long, default_value_t = 8)]
    block_tokens: usize,
    /// Android ONNX Runtime shared library. Defaults to libonnxruntime.so.
    #[cfg(target_os = "android")]
    #[arg(long)]
    ort_library: Option<PathBuf>,
    #[arg(long, default_value = "output.wav")]
    output: String,
    /// Write representative VITS inputs for QNN calibration and exit.
    #[arg(long, conflicts_with = "stream", help_heading = "Optional QNN pipeline")]
    dump_qnn_inputs: Option<PathBuf>,
    /// Directory containing the Qualcomm QNN provider and runtime libraries.
    #[cfg(target_os = "android")]
    #[arg(long, conflicts_with_all = ["stream", "dump_qnn_inputs"], help_heading = "Optional QNN pipeline")]
    qnn_libraries: Option<PathBuf>,
    /// Compile an EPContext beside the generator before running inference.
    #[cfg(target_os = "android")]
    #[arg(long, requires = "qnn_libraries", help_heading = "Optional QNN pipeline")]
    compile_qnn_context: bool,
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
        InferParams::from_bundle(&args.model_path)?
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
    TTSModel::from_bundle(assets_dir)
}

fn write_wav(spec: WavSpec, samples: &[f32], filename: &str) -> Result<(), GSVError> {
    let mut writer = WavWriter::create(filename, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Synchronous,
    Streaming,
    #[cfg(target_os = "android")]
    QnnPipeline,
}

fn run_inference(
    model: &mut TTSModel,
    infer: &InferParams,
    args: &Args,
    lang: LangId,
    mode: Mode,
) -> Result<(), GSVError> {
    let runs = args.run_count.get();
    let mut times = Vec::with_capacity(runs);
    let sampling = infer.to_sampling_params();
    let postprocess = PostprocessParams::default();
    for i in 0..runs {
        let start = Instant::now();
        let mut spec = None;
        let mut samples = Vec::new();
        let mut first_ms = 0.0;
        let mut playback_end = 0.0_f64;
        let mut underruns = 0;
        let mut chunks = 0;
        let mut receive = |format: WavSpec, chunk: Vec<f32>| {
            if chunk.is_empty() {
                return true;
            }
            let now_ms = start.elapsed().as_secs_f64() * 1000.0;
            if chunks == 0 {
                first_ms = now_ms;
                spec = Some(format);
            } else if now_ms > playback_end {
                underruns += 1;
            }
            playback_end = playback_end.max(now_ms)
                + 1000.0 * chunk.len() as f64 / f64::from(format.sample_rate);
            chunks += 1;
            samples.extend(chunk);
            true
        };
        match mode {
            Mode::Synchronous => {
                let (spec, pcm) = model.synthesize_sync(&args.text, sampling, lang, postprocess)?;
                receive(spec, pcm);
            }
            Mode::Streaming => model.synthesize_token_chunks_sync(
                &args.text,
                sampling,
                lang,
                postprocess,
                TokenStreamOptions {
                    block_tokens: args.block_tokens,
                    vits_left_context_tokens: None,
                },
                || false,
                receive,
            )?,
            #[cfg(target_os = "android")]
            Mode::QnnPipeline => {
                model.synthesize_qnn_pipeline(&args.text, sampling, lang, || false, receive)?;
                if !model.has_qnn_pipeline() {
                    return Err(GSVError::from("QNN generator failed during synthesis"));
                }
            }
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let spec = spec.ok_or_else(|| GSVError::from("Synthesis returned no PCM"))?;
        let duration_ms = 1000.0 * samples.len() as f64 / f64::from(spec.sample_rate);
        println!(
            "run={} chunks={chunks} ttfa_ms={first_ms:.2} synthesis_ms={elapsed:.2} audio_ms={duration_ms:.2} rtf={:.4} simulated_underruns={underruns}",
            i + 1,
            elapsed / duration_ms,
        );
        if i == runs - 1 {
            write_wav(spec, &samples, &args.output)?;
        }
        times.push(elapsed);
    }
    TimingStats::new(&times).print(&format!("{mode:?}"), runs);
    Ok(())
}

#[cfg(target_os = "android")]
fn init_android_ort(args: &Args) -> Result<(), GSVError> {
    use ort::environment::Environment;
    use std::sync::{Arc, OnceLock};

    static PROCESS_ORT: OnceLock<Arc<Environment>> = OnceLock::new();
    let library = args.ort_library.clone().unwrap_or_else(|| {
        args.qnn_libraries.as_ref().map_or_else(
            || PathBuf::from("libonnxruntime.so"),
            |dir| dir.join("libonnxruntime.so"),
        )
    });
    let ort = ort::init_from(&library).map_err(|error| {
        GSVError::from(format!("Load Android ORT {}: {error}", library.display()))
    })?;
    #[cfg(feature = "shared-ort-pool")]
    let ort = ort.with_global_thread_pool(
        ort::environment::GlobalThreadPoolOptions::default()
            .with_intra_threads(4)?
            .with_inter_threads(1)?
            .with_spin_control(false)?,
    );
    ort.commit();
    // The Android dynamic runtime can destroy its C++ mutex before ort's
    // process-exit hook. Keep the CLI's environment alive until OS cleanup.
    let _ = PROCESS_ORT.set(Environment::current()?);
    Ok(())
}

fn main() -> Result<(), GSVError> {
    env_logger::init();
    let args = Args::parse();
    if !args.ref_audio.is_file() {
        return Err(GSVError::FileNotFound(format!(
            "Reference WAV missing at {}. Specify it with --ref-audio PATH.",
            args.ref_audio.display()
        )));
    }
    let infer = resolve_infer_params(&args)?;
    let lang = if args.lang == "yue" {
        LangId::AutoYue
    } else {
        LangId::Auto
    };

    #[cfg(target_os = "android")]
    init_android_ort(&args)?;

    let mut model = create_model(&args.model_path)?;
    model.process_reference_sync(&args.ref_audio, &args.ref_text, LangId::Auto)?;

    if let Some(path) = &args.dump_qnn_inputs {
        let inputs = model.qnn_calibration_inputs(&args.text, infer.to_sampling_params(), lang)?;
        let output = std::fs::File::create_new(path)?;
        serde_json::to_writer(output, &inputs).map_err(anyhow::Error::from)?;
        println!(
            "Wrote {} calibration inputs to {}",
            inputs.len(),
            path.display()
        );
        return Ok(());
    }

    let mode = if args.stream {
        Mode::Streaming
    } else {
        Mode::Synchronous
    };
    #[cfg(target_os = "android")]
    let mode = if let Some(libraries) = &args.qnn_libraries {
        if args.compile_qnn_context {
            let bundle = args.model_path.canonicalize()?;
            let context = bundle.join("vits_generator_context.onnx");
            TTSModel::compile_qnn_context(&bundle, libraries, &context)?;
            println!("Wrote {}", context.display());
        }
        model.enable_qnn_pipeline(&args.model_path, libraries)?;
        Mode::QnnPipeline
    } else {
        mode
    };
    println!(
        "mode={mode:?} block_tokens={} top_k={} top_p={} temperature={} repetition_penalty={} seed={:?}",
        args.block_tokens,
        infer.top_k,
        infer.top_p,
        infer.temperature,
        infer.repetition_penalty,
        infer.seed,
    );

    run_inference(&mut model, &infer, &args, lang, mode)?;
    println!("Wrote {}", args.output);

    Ok(())
}
