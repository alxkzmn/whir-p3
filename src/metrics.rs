//! Lightweight hash-call counters used by EVM-oriented benchmarking.
//!
//! Counters are feature-gated behind `hash_count`. When disabled, APIs are no-ops.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HashCountSnapshot {
    pub leaf_hash_calls: u64,
    pub node_hash_calls: u64,
}

#[cfg(feature = "hash_count")]
mod enabled {
    use core::sync::atomic::{AtomicU64, Ordering};

    use super::HashCountSnapshot;

    static LEAF_HASH_CALLS: AtomicU64 = AtomicU64::new(0);
    static NODE_HASH_CALLS: AtomicU64 = AtomicU64::new(0);

    #[inline]
    pub fn add_leaf_hash_call() {
        LEAF_HASH_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_node_hash_call() {
        NODE_HASH_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn reset_hash_counters() {
        LEAF_HASH_CALLS.store(0, Ordering::Relaxed);
        NODE_HASH_CALLS.store(0, Ordering::Relaxed);
    }

    #[inline]
    pub fn snapshot_hash_counters() -> HashCountSnapshot {
        HashCountSnapshot {
            leaf_hash_calls: LEAF_HASH_CALLS.load(Ordering::Relaxed),
            node_hash_calls: NODE_HASH_CALLS.load(Ordering::Relaxed),
        }
    }
}

#[cfg(not(feature = "hash_count"))]
mod enabled {
    use super::HashCountSnapshot;

    #[inline]
    pub fn add_leaf_hash_call() {}

    #[inline]
    pub fn add_node_hash_call() {}

    #[inline]
    pub fn reset_hash_counters() {}

    #[inline]
    pub fn snapshot_hash_counters() -> HashCountSnapshot {
        HashCountSnapshot::default()
    }
}

pub use enabled::{
    add_leaf_hash_call, add_node_hash_call, reset_hash_counters, snapshot_hash_counters,
};
