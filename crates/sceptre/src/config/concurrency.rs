//! Concurrency / thread-budget configuration.
//!
//! A single `max_threads` value caps every internal pool (Rayon, and the ONNX
//! Runtime intra-op threads once the backend is wired) so nested parallelism
//! cannot oversubscribe the CPU.

use std::sync::Once;

use serde::{Deserialize, Serialize};

/// Caps all internal thread pools to a single budget.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConcurrencyConfig {
    /// Maximum threads for all internal pools. `None` = auto (`num_cpus`, capped at 8).
    pub max_threads: Option<usize>,
}

static POOL_INIT: Once = Once::new();

/// Resolve the effective thread budget: user value, else `num_cpus` capped at 8.
pub(crate) fn resolve_thread_budget(config: Option<&ConcurrencyConfig>) -> usize {
    if let Some(n) = config.and_then(|c| c.max_threads) {
        return n.max(1);
    }
    num_cpus::get().min(8)
}

/// Initialize the global Rayon pool with `budget` threads. Idempotent.
pub(crate) fn init_thread_pools(budget: usize) {
    POOL_INIT.call_once(|| {
        if let Err(_err) = rayon::ThreadPoolBuilder::new().num_threads(budget).build_global() {
            tracing::debug!(budget, "global rayon pool already initialized; reusing existing pool");
        }
    });
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
}
