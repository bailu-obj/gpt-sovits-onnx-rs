use clap::Parser;
use futures::StreamExt;
use gpt_sovits_onnx_rs::*;
use hound::{WavSpec, WavWriter};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;
use tokio::runtime::Runtime;

#[derive(Parser, Debug)]
struct Args {
    #[arg(
        long,
        default_value = "/home/qiang/projects/GPT-SoVITS/onnx-patched/custom"
    )]
    model_path: PathBuf,
    /// Optional JSON file to override built-in sampling defaults.
    #[arg(long)]
    params: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    run_count: usize,
    #[arg(
        long,
        default_value = "你好啊，这是一个测试。吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。This demo is only for test  usage. If you find any 问题, 请修复它。"
    )]
    text: String,
    #[arg(long, default_value = "zh")]
    lang: String,
    #[arg(long, default_value = "格式化，可以给自家的奶带来大量的。")]
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
    /// Print cold-init / reference / TTFA / RSS / p95 timing breakdown.
    #[arg(long, default_value_t = false)]
    benchmark: bool,
    /// Drop SSL session after reference to measure RSS reclaim (synthesize-only path).
    #[arg(long, default_value_t = false)]
    release_ssl: bool,
    /// ORT runtime profile: `latency` (default) or `low-power`.
    #[arg(long, default_value = "latency")]
    ort_profile: String,
    /// Override ORT intra-op thread count.
    #[arg(long)]
    ort_threads: Option<usize>,
}

struct TimingStats {
    avg: f64,
    median: f64,
    p95: f64,
    max: f64,
    min: f64,
}

impl TimingStats {
    fn new(times: &[f64]) -> Self {
        let count = times.len() as f64;
        let sum: f64 = times.iter().sum();
        let avg = if count > 0.0 { sum / count } else { 0.0 };
        let mut sorted = times.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.is_empty() {
            0.0
        } else if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        let p95 = if sorted.is_empty() {
            0.0
        } else {
            let idx = ((sorted.len() as f64) * 0.95).ceil() as usize;
            sorted[idx.saturating_sub(1).min(sorted.len() - 1)]
        };
        Self {
            avg,
            median,
            p95,
            max: *sorted.last().unwrap_or(&0.0),
            min: *sorted.first().unwrap_or(&0.0),
        }
    }

    fn print(&self, mode: &str, runs: usize) {
        println!("{} Inference ({} runs):", mode, runs);
        println!("  Average: {:.2} ms", self.avg);
        println!("  Median: {:.2} ms", self.median);
        println!("  P95: {:.2} ms", self.p95);
        println!("  Max: {:.2} ms", self.max);
        println!("  Min: {:.2} ms", self.min);
    }
}

fn demo_runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| Runtime::new().expect("failed to create tokio runtime"))
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

fn find_model_prefix(assets_dir: &Path) -> Result<String, GSVError> {
    if assets_dir.join("custom_vits.onnx").exists() {
        return Ok("custom".to_string());
    }

    for entry in std::fs::read_dir(assets_dir).map_err(|e| {
        GSVError::FileNotFound(format!(
            "Failed to read model directory {:?}: {}",
            assets_dir, e
        ))
    })? {
        let entry = entry.map_err(|e| GSVError::from(e.to_string()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(prefix) = name.strip_suffix("_vits.onnx") {
            return Ok(prefix.to_string());
        }
    }

    Err(GSVError::FileNotFound(format!(
        "No *_vits.onnx found in {:?}",
        assets_dir
    )))
}

fn configure_ort_from_args(args: &Args) {
    let profile = match args.ort_profile.to_ascii_lowercase().as_str() {
        "low-power" | "low_power" | "lowpower" => OrtRuntimeProfile::LowPower,
        _ => OrtRuntimeProfile::Latency,
    };
    configure_ort_runtime(OrtConfig {
        profile,
        intra_threads: args.ort_threads,
        use_xnnpack: cfg!(feature = "xnnpack"),
        shared_thread_pool: None,
    });
}

fn create_model(assets_dir: &Path) -> Result<TTSModel, GSVError> {
    if !assets_dir.exists() {
        return Err(GSVError::FileNotFound(format!(
            "Assets directory not found: {:?}",
            assets_dir
        )));
    }
    let prefix = find_model_prefix(assets_dir)?;
    let vits_path = assets_dir.join(format!("{prefix}_vits.onnx"));
    let mut model = TTSModel::new(
        vits_path.clone(),
        assets_dir.join("ssl.onnx"),
        assets_dir.join(format!("{prefix}_t2s_encoder.onnx")),
        assets_dir.join(format!("{prefix}_t2s_fs_decoder.onnx")),
        assets_dir.join(format!("{prefix}_t2s_s_decoder.onnx")),
        Some(assets_dir.join("bert.onnx")),
        Some(assets_dir.join("g2pW.onnx")),
        Some(assets_dir.join("g2p_en")),
        match assets_dir.join("sv.onnx").exists() {
            true => Some(assets_dir.join("sv.onnx")),
            false => None,
        },
    )?;
    let _ = model.try_load_split_vits_beside(&vits_path)?;
    Ok(model)
}

fn write_wav(spec: WavSpec, samples: &[f32], filename: &str) -> Result<(), GSVError> {
    let mut writer = WavWriter::create(filename, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

fn current_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        use std::mem::MaybeUninit;
        #[repr(C)]
        struct RUsage {
            ru_utime: [i64; 2],
            ru_stime: [i64; 2],
            ru_maxrss: i64,
            _rest: [i64; 14],
        }
        unsafe extern "C" {
            fn getrusage(who: i32, usage: *mut RUsage) -> i32;
        }
        const RUSAGE_SELF: i32 = 0;
        let mut usage = MaybeUninit::<RUsage>::uninit();
        let rc = unsafe { getrusage(RUSAGE_SELF, usage.as_mut_ptr()) };
        if rc == 0 {
            // macOS reports ru_maxrss in bytes.
            return Some(unsafe { usage.assume_init() }.ru_maxrss as u64);
        }
        None
    }
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

fn format_bytes(bytes: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else {
        format!("{:.1} MB", b / MB)
    }
}

fn lang_id_from(lang: &str) -> LangId {
    if lang == "yue" {
        LangId::AutoYue
    } else {
        LangId::Auto
    }
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
    let lang_id = lang_id_from(lang);
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

/// Measure time-to-first-audio by consuming the streaming API.
fn run_streaming_ttfa(
    model: &mut TTSModel,
    infer: &InferParams,
    text: &str,
    lang: &str,
) -> Result<(f64, f64, usize), GSVError> {
    let lang_id = lang_id_from(lang);
    let sampling = infer.to_sampling_params();
    demo_runtime().block_on(async {
        let start = Instant::now();
        let (_spec, stream) = model
            .synthesize(text, sampling, lang_id, PostprocessParams::default())
            .await?;
        futures::pin_mut!(stream);
        let mut ttfa_ms = None;
        let mut samples = 0usize;
        while let Some(sample) = stream.next().await {
            let _ = sample?;
            samples += 1;
            if ttfa_ms.is_none() {
                ttfa_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok((ttfa_ms.unwrap_or(total_ms), total_ms, samples))
    })
}

fn main() -> Result<(), GSVError> {
    env_logger::init();
    let args = Args::parse();
    configure_ort_from_args(&args);
    let infer = resolve_infer_params(&args)?;

    let init_start = Instant::now();
    let mut model = create_model(&args.model_path)?;
    let init_ms = init_start.elapsed().as_secs_f64() * 1000.0;
    let rss_after_init = current_rss_bytes();

    let ref_start = Instant::now();
    model.process_reference_sync(
        args.model_path.join("ref.wav"),
        &args.ref_text,
        LangId::Auto,
    )?;
    let ref_ms = ref_start.elapsed().as_secs_f64() * 1000.0;
    let rss_after_ref = current_rss_bytes();
    if args.release_ssl {
        model.release_ssl_after_reference();
    }
    let rss_after_ssl_release = if args.release_ssl {
        current_rss_bytes()
    } else {
        None
    };

    println!("text: {:?} ref_text: {:?}", args.text, args.ref_text);
    println!(
        "infer params: top_k={} top_p={} temperature={} repetition_penalty={} seed={:?}",
        infer.top_k, infer.top_p, infer.temperature, infer.repetition_penalty, infer.seed
    );
    println!(
        "split_vits={} ort_profile={} ort_config={:?}",
        model.uses_split_vits(),
        args.ort_profile,
        ort_config()
    );

    if args.benchmark {
        println!("Cold init: {:.2} ms", init_ms);
        println!("Reference: {:.2} ms", ref_ms);
        if let Some(b) = rss_after_init {
            println!("RSS after init: {}", format_bytes(b));
        }
        if let Some(b) = rss_after_ref {
            println!("RSS after reference: {}", format_bytes(b));
        }
        if let Some(b) = rss_after_ssl_release {
            println!("RSS after SSL release: {}", format_bytes(b));
        }
        let (ttfa, total, n_samples) =
            run_streaming_ttfa(&mut model, &infer, &args.text, &args.lang)?;
        println!(
            "TTFA (first audio sample): {:.2} ms | stream total: {:.2} ms | samples: {}",
            ttfa, total, n_samples
        );
    }

    let stats = run_sync_inference(
        &mut model,
        &infer,
        &args.text,
        &args.lang,
        args.run_count,
        &args.output,
    )?;
    stats.print("Synchronous", args.run_count);
    if let Some(b) = current_rss_bytes() {
        println!("RSS (max/current): {}", format_bytes(b));
    }
    println!("Wrote {}", args.output);

    Ok(())
}
