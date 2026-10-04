use crate::cpu_info::get_hw_big_cores;
use lazy_static::lazy_static;
use std::path::Path;
use std::sync::OnceLock;

use ort::{
    ep::CPU,
    session::{Session, builder::GraphOptimizationLevel},
};

use crate::error::GSVError;
lazy_static! {
    pub static ref BIG_CORES: Vec<(usize, u64)> =
        get_hw_big_cores().unwrap_or((0..8).map(|id| (id, 0)).collect());
}

/// Process-wide defaults for sessions created by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnnxSessionOptions {
    pub spinning: bool,
}

impl Default for OnnxSessionOptions {
    fn default() -> Self {
        Self {
            spinning: !cfg!(feature = "shared-ort-pool"),
        }
    }
}

static SESSION_OPTIONS: OnceLock<OnnxSessionOptions> = OnceLock::new();

/// Configure before loading any model. Options freeze on the first session
/// creation; changing them afterwards would leave models with mixed policies.
/// For a shared pool, also set the host's GlobalThreadPoolOptions spin control.
pub fn configure_onnx_sessions(options: OnnxSessionOptions) -> Result<(), &'static str> {
    SESSION_OPTIONS
        .set(options)
        .map_err(|_| "ONNX session options already initialized")
}

/// Create a CPU session. With `shared-ort-pool`, the embedding application
/// must initialize an ORT environment with a global thread pool first.
/// Existing callers retain independent pools unless they opt into the feature.
pub fn create_onnx_cpu_session<P: AsRef<Path>>(path: P) -> Result<Session, GSVError> {
    let options = SESSION_OPTIONS.get_or_init(OnnxSessionOptions::default);
    let builder = Session::builder()?
        .with_execution_providers([CPU::default().with_arena_allocator(true).build()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_prepacking(true)?
        .with_config_entry("session.enable_mem_reuse", "1")?
        .with_intra_op_spinning(options.spinning)?
        .with_inter_op_spinning(options.spinning)?;
    #[cfg(feature = "shared-ort-pool")]
    let mut builder = builder;
    #[cfg(not(feature = "shared-ort-pool"))]
    let mut builder = builder
        .with_intra_threads(BIG_CORES.len())?
        .with_independent_thread_pool()?;
    Ok(builder.commit_from_file(path)?)
}
