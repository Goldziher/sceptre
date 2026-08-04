//! Concurrency / thread-budget configuration.
//!
//! Native readers resolve one `max_threads` budget for their private Rayon pool
//! and the inference backend's intra-op threads, preventing nested parallelism.
//! Browser-WASM readers use the same API with sequential execution.

use serde::{Deserialize, Serialize};

#[cfg(not(target_arch = "wasm32"))]
use crate::error::OcrError;
use crate::error::Result;

/// Per-reader execution context: a private Rayon pool on native targets and a
/// sequential adapter on browser WASM.
pub(crate) struct WorkerPool {
    #[cfg(not(target_arch = "wasm32"))]
    inner: rayon::ThreadPool,
}

impl WorkerPool {
    pub(crate) fn install<T: Send>(&self, operation: impl FnOnce() -> T + Send) -> T {
        #[cfg(not(target_arch = "wasm32"))]
        return self.inner.install(operation);

        #[cfg(target_arch = "wasm32")]
        operation()
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    fn current_num_threads(&self) -> usize {
        self.inner.current_num_threads()
    }
}

/// Caps the worker and inference thread pools within each reader.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConcurrencyConfig {
    /// Maximum threads for all internal pools. `None` = auto (`num_cpus`, capped at 8).
    pub max_threads: Option<usize>,
}

/// Resolve the effective thread budget: user value, else `num_cpus` capped at 8.
pub(crate) fn resolve_thread_budget(config: Option<&ConcurrencyConfig>) -> usize {
    if let Some(n) = config.and_then(|c| c.max_threads) {
        return n.max(1);
    }
    #[cfg(not(target_arch = "wasm32"))]
    return num_cpus::get().min(8);

    #[cfg(target_arch = "wasm32")]
    1
}

/// Build the target-appropriate execution context for one reader.
pub(crate) fn build_thread_pool(budget: usize) -> Result<WorkerPool> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let inner = rayon::ThreadPoolBuilder::new()
            .num_threads(budget)
            .build()
            .map_err(|source| OcrError::Config {
                message: format!("failed to initialize the OCR worker pool with {budget} threads"),
                source: Some(Box::new(source)),
            })?;
        Ok(WorkerPool { inner })
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = budget;
        Ok(WorkerPool {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_prefers_user_value() {
        let cfg = ConcurrencyConfig { max_threads: Some(3) };
        assert_eq!(resolve_thread_budget(Some(&cfg)), 3);
    }

    #[test]
    fn budget_auto_is_sane() {
        let budget = resolve_thread_budget(None);
        assert!((1..=8).contains(&budget));
    }

    #[test]
    fn budget_clamps_to_one() {
        let cfg = ConcurrencyConfig { max_threads: Some(0) };
        assert_eq!(resolve_thread_budget(Some(&cfg)), 1);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn private_pool_uses_resolved_budget() {
        let pool = build_thread_pool(2).expect("the private pool should build");
        assert_eq!(pool.current_num_threads(), 2);
    }
}
