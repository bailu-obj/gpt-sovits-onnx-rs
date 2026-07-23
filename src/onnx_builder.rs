use crate::cpu_info::{default_intra_threads, get_hw_big_cores};
use lazy_static::lazy_static;
use log::info;
use std::path::Path;
use std::sync::OnceLock;

use ort::{
    environment::GlobalThreadPoolOptions,
    execution_providers::CPUExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
};

use crate::error::GSVError;

lazy_static! {
    /// Prefer big cores on Linux/Android; elsewhere use a capped P-core estimate.
    pub static ref BIG_CORES: Vec<(usize, u64)> =
        get_hw_big_cores().unwrap_or_else(|_| {
            let n = default_intra_threads();
            (0..n).map(|id| (id, 0)).collect()
        });
}

/// Runtime profile controlling ORT thread spinning and pool sharing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OrtRuntimeProfile {
    /// Prefer lower latency: per-session pools + spinning.
    #[default]
    Latency,
    /// Prefer lower RAM / idle CPU: shared global pool, spinning off.
    LowPower,
}

/// Global ONNX Runtime configuration. Call [`configure_ort_runtime`] before the first session.
#[derive(Clone, Copy, Debug)]
pub struct OrtConfig {
    pub profile: OrtRuntimeProfile,
    /// Override intra-op thread count; `None` uses [`default_intra_threads`].
    pub intra_threads: Option<usize>,
    /// Use XNNPACK EP when the `xnnpack` Cargo feature is enabled (ARM/Android).
    pub use_xnnpack: bool,
    /// Share one process-wide ORT thread pool. Defaults from [`OrtRuntimeProfile`]:
    /// Latency → false (independent pools), LowPower → true (shared).
    pub shared_thread_pool: Option<bool>,
}

impl Default for OrtConfig {
    fn default() -> Self {
        Self {
            profile: OrtRuntimeProfile::Latency,
            intra_threads: None,
            use_xnnpack: cfg!(feature = "xnnpack"),
            shared_thread_pool: None,
        }
    }
}

impl OrtConfig {
    pub fn uses_shared_thread_pool(&self) -> bool {
        self.shared_thread_pool.unwrap_or(matches!(
            self.profile,
            OrtRuntimeProfile::LowPower
        ))
    }
}

static ORT_CONFIG: OnceLock<OrtConfig> = OnceLock::new();
static ORT_ENV_READY: OnceLock<()> = OnceLock::new();

/// Configure ORT before any session is created. Subsequent calls are ignored.
pub fn configure_ort_runtime(config: OrtConfig) {
    let _ = ORT_CONFIG.set(config);
}

pub fn ort_config() -> OrtConfig {
    *ORT_CONFIG.get_or_init(OrtConfig::default)
}

fn ensure_ort_environment() -> Result<(), GSVError> {
    if ORT_ENV_READY.get().is_some() {
        return Ok(());
    }
    let config = ort_config();
    let threads = config
        .intra_threads
        .unwrap_or_else(default_intra_threads)
        .max(1);
    let spin = matches!(config.profile, OrtRuntimeProfile::Latency);
    let shared = config.uses_shared_thread_pool();

    info!(
        "ORT env: threads={} profile={:?} shared_pool={} xnnpack={} spin={}",
        threads, config.profile, shared, config.use_xnnpack, spin
    );

    if shared {
        let pool = GlobalThreadPoolOptions::default()
            .with_intra_threads(threads)?
            .with_inter_threads(1)?
            .with_spin_control(spin)?;
        ort::init()
            .with_name("gpt-sovits-onnx-rs")
            .with_global_thread_pool(pool)
            .commit();
    } else {
        ort::init().with_name("gpt-sovits-onnx-rs").commit();
    }

    let _ = ORT_ENV_READY.set(());
    Ok(())
}

/// Create an optimized CPU (or optional XNNPACK) session.
pub fn create_onnx_cpu_session<P: AsRef<Path>>(path: P) -> Result<Session, GSVError> {
    ensure_ort_environment()?;
    let config = ort_config();
    let threads = config
        .intra_threads
        .unwrap_or_else(default_intra_threads)
        .max(1);
    let spin = matches!(config.profile, OrtRuntimeProfile::Latency);
    let shared = config.uses_shared_thread_pool();

    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads)?
        .with_prepacking(true)?
        .with_config_entry("session.enable_mem_reuse", "1")?
        .with_intra_op_spinning(spin)?;

    if !shared {
        // Independent pools: best latency for AR decode / VITS; higher RAM.
        builder = builder.with_independent_thread_pool()?;
    }

    #[cfg(feature = "xnnpack")]
    {
        use ort::execution_providers::xnnpack::XNNPACKExecutionProvider;
        use std::num::NonZeroUsize;
        if config.use_xnnpack {
            let n = NonZeroUsize::new(threads).unwrap_or(NonZeroUsize::new(1).unwrap());
            return Ok(builder
                .with_execution_providers([
                    XNNPACKExecutionProvider::default()
                        .with_intra_op_num_threads(n)
                        .build(),
                    CPUExecutionProvider::default()
                        .with_arena_allocator(true)
                        .build(),
                ])?
                .commit_from_file(path)?);
        }
    }
    let _ = config.use_xnnpack;

    Ok(builder
        .with_execution_providers([CPUExecutionProvider::default()
            .with_arena_allocator(true)
            .build()])?
        .commit_from_file(path)?)
}
