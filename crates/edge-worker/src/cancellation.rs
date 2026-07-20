//! Cancellation of proofs already dispatched to this worker.
//!
//! Canceling drops the proof from the manager's state at once, but the jobs it
//! already dispatched keep running here. Recording the cancellation lets the
//! proving loops stop at their next segment instead of running the whole job
//! out and spending a full app-prove phase of GPU time on a discarded proof.

use std::collections::{HashSet, VecDeque};
use std::sync::{Mutex, OnceLock};

/// How many canceled proofs to remember. This only has to outlive the jobs
/// still running for them, and the manager proves one at a time.
const REMEMBERED_CANCELLATIONS: usize = 64;

#[derive(Default)]
struct Cancellations {
    /// Cancellation order, used to drop the oldest once the set is full.
    order: VecDeque<String>,
    proofs: HashSet<String>,
}

fn cancellations() -> &'static Mutex<Cancellations> {
    static CANCELLATIONS: OnceLock<Mutex<Cancellations>> = OnceLock::new();
    CANCELLATIONS.get_or_init(Mutex::default)
}

/// Record that `proof_uuid` is canceled.
pub fn cancel(proof_uuid: String) {
    let mut cancellations = cancellations().lock().expect("cancellations poisoned");
    if !cancellations.proofs.insert(proof_uuid.clone()) {
        return;
    }
    cancellations.order.push_back(proof_uuid);
    while cancellations.order.len() > REMEMBERED_CANCELLATIONS {
        let evicted = cancellations.order.pop_front().expect("non-empty above");
        cancellations.proofs.remove(&evicted);
    }
}

/// Whether `proof_uuid` is canceled. Read once per segment, so the lock costs
/// nothing against a segment that takes seconds to prove.
pub fn is_cancelled(proof_uuid: &str) -> bool {
    cancellations()
        .lock()
        .expect("cancellations poisoned")
        .proofs
        .contains(proof_uuid)
}

#[cfg(test)]
mod tests {
    use super::{cancel, is_cancelled, REMEMBERED_CANCELLATIONS};

    /// The registry is global, so each test uses uuids of its own.
    fn uuid(test: &str, n: usize) -> String {
        format!("{test}-{n}")
    }

    #[test]
    fn records_a_cancellation_once() {
        assert!(!is_cancelled(&uuid("once", 0)));
        cancel(uuid("once", 0));
        cancel(uuid("once", 0));
        assert!(is_cancelled(&uuid("once", 0)));
    }

    #[test]
    fn forgets_the_oldest_beyond_the_bound() {
        cancel(uuid("bound", 0));
        for n in 1..=REMEMBERED_CANCELLATIONS {
            cancel(uuid("bound", n));
        }
        assert!(!is_cancelled(&uuid("bound", 0)));
        assert!(is_cancelled(&uuid("bound", REMEMBERED_CANCELLATIONS)));
    }
}
