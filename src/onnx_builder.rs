use crate::cpu_info::get_hw_big_cores;
use lazy_static::lazy_static;
use std::path::Path;

use ort::{
    ep::CPU,
    session::{Session, builder::GraphOptimizationLevel},
};

use crate::error::GSVError;
lazy_static! {
    pub static ref BIG_CORES: Vec<(usize, u64)> =
        get_hw_big_cores().unwrap_or((0..8).map(|id| (id, 0)).collect());
}

/// Create a CPU session. With `shared-ort-pool`, the embedding application
/// must initialize an ORT environment with a global thread pool first.
/// Existing callers retain independent pools unless they opt into the feature.
pub fn create_onnx_cpu_session<P: AsRef<Path>>(path: P) -> Result<Session, GSVError> {
    let builder = Session::builder()?
        .with_execution_providers([CPU::default().with_arena_allocator(true).build()])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_prepacking(true)?
        .with_config_entry("session.enable_mem_reuse", "1")?;
    #[cfg(feature = "shared-ort-pool")]
    let mut builder = builder.with_intra_op_spinning(false)?;
    #[cfg(not(feature = "shared-ort-pool"))]
    let mut builder = builder
        .with_intra_threads(BIG_CORES.len())?
        .with_independent_thread_pool()?
        .with_intra_op_spinning(true)?;
    Ok(builder.commit_from_file(path)?)
}
