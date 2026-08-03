//! Concurrency / thread-budget configuration.
//!
//! Each reader resolves one `max_threads` budget for its private Rayon pool and
//! the inference backend's intra-op threads, preventing nested parallelism from
//! oversubscribing the CPU within that reader.

use rayon::ThreadPool;
use serde::{Deserialize, Serialize};

use crate::error::{OcrError, Result};

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
    num_cpus::get().min(8)
}

/// Build an isolated Rayon pool for one reader.
pub(crate) fn build_thread_pool(budget: usize) -> Result<ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(budget)
        .build()
        .map_err(|source| OcrError::Config {
            message: format!("failed to initialize the OCR worker pool with {budget} threads"),
            source: Some(Box::new(source)),
        })
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
    fn private_pool_uses_resolved_budget() {
        let pool = build_thread_pool(2).expect("the private pool should build");
        assert_eq!(pool.current_num_threads(), 2);
    }
}
