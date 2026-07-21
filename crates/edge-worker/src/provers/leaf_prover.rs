//! Leaf Prover implementation.
//!
//! This prover aggregates app proofs into leaf proofs in the recursion tree.
//!
//! # Architecture
//!
//! The `LeafProverInstance` struct holds the reusable leaf prover that is expensive
//! to create. Worker threads create one instance at startup and reuse it across all
//! jobs via `prove_leaf_with_prover`.
//!
//! The `prove_leaf` function is used in mock mode and creates instances internally.

use eyre::Result;
#[cfg(feature = "mock-provers")]
use std::time::Duration;
#[cfg(feature = "mock-provers")]
use std::time::Instant;
use tracing::{info, instrument};

#[cfg(feature = "mock-provers")]
use proof::{ProofWithPublicValue, F};
use protocol::ProofResult;
#[cfg(feature = "mock-provers")]
use protocol::{LeafProof, LeafProofState};

use super::{LeafProverJob, ProverResult};

/// Execute leaf proving to aggregate app proofs.
#[instrument(skip_all, fields(
    proof_id = %job.context.proof_uuid,
    segment_start = job.segment_start,
    segment_end = job.segment_end,
    num_app_proofs = job.app_proofs.len()
))]
pub fn prove_leaf(job: LeafProverJob) -> ProverResult {
    info!(
        "Starting leaf prove: segment_start={}, segment_end={}, app_proofs={}",
        job.segment_start,
        job.segment_end,
        job.app_proofs.len()
    );

    match prove_leaf_impl(job) {
        Ok(results) => ProverResult::Success(results),
        Err(e) => ProverResult::Error(format!("Leaf prove failed: {}", e)),
    }
}

#[cfg(feature = "mock-provers")]
fn prove_leaf_impl(job: LeafProverJob) -> Result<Vec<ProofResult>> {
    // Validate inputs
    if job.app_proofs.is_empty() {
        return Err(eyre::eyre!("Cannot aggregate empty app_proofs list"));
    }
    if job.segment_start > job.segment_end {
        return Err(eyre::eyre!(
            "Invalid segment range: start {} > end {}",
            job.segment_start,
            job.segment_end
        ));
    }

    // Mock implementation - simulate proving time
    let start = Instant::now();
    std::thread::sleep(Duration::from_millis(200));
    let prove_time_ms = start.elapsed().as_millis() as u64;

    // Generate mock leaf proof — wire-side payload is bincode-encoded.
    let mock_proof = ProofWithPublicValue::<F> {
        proof: vec![0u8; 512],
        public_values: vec![F::default(); 8],
    };
    let proof = LeafProof {
        context: job.context.clone(),
        state: LeafProofState {
            proof: Some(proof::encode_proof(&mock_proof)?),
            segment_start: job.segment_start,
            segment_end: job.segment_end,
            prove_time_ms,
            sub_metrics: std::collections::HashMap::new(),
        },
    };

    info!(
        "Generated mock leaf proof for segments [{}, {}) ({}ms)",
        job.segment_start, job.segment_end, prove_time_ms
    );

    Ok(vec![ProofResult::Leaf(proof)])
}

// ============================================================================
// Real prover implementation (not mock-provers)
// ============================================================================

#[cfg(not(feature = "mock-provers"))]
mod real_impl {
    use super::super::real_prover_types::{ChildVkKind, LeafProver, RecursionEngine};
    use super::*;
    use crate::artifacts::ArtifactStore;
    use proof::ProofWithPublicValue;
    use protocol::{LeafProof, LeafProofState};
    use sdk_v2::config::MAX_NUM_CHILDREN_LEAF;
    use std::sync::Arc;

    /// Leaf prover instance that holds reusable prover state.
    ///
    /// This struct is created once per worker thread and reused across jobs.
    pub struct LeafProverInstance {
        /// The leaf prover for aggregation. Carries `def_hook_commit =
        /// Some(...)` iff the worker is in deferral mode.
        pub prover: LeafProver,
    }

    impl LeafProverInstance {
        /// Create a new leaf prover instance from global artifacts.
        ///
        /// This should be called once per worker thread at startup.
        pub fn new() -> Result<Self> {
            let artifact_store = ArtifactStore::global()
                .ok_or_else(|| eyre::eyre!("Artifact store not initialized"))?;
            // Parks on a registration-driven worker, which has no artifacts
            // until the first `/register_program` publishes them.
            let edge_artifacts = artifact_store.wait_for_edge_artifacts();

            // Get the app verifying key (child_vk for leaf prover). In
            // deferral mode `edge_artifacts.app_pk` already points at
            // `cached_pk.app_pk`, so this is the deferral app_vk
            // automatically.
            let app_vk = Arc::new(edge_artifacts.app_pk.get_app_vk().vk);
            // `def_hook_cached_commit` is `Some(...)` iff the worker is
            // in deferral mode — bakes the deferral hook AIR into the
            // leaf circuit (gating #3). `None` on the default path.
            let def_hook_cached_commit = edge_artifacts.def_hook_cached_commit();

            info!(
                "Creating LeafProverInstance (deferral_mode={})",
                def_hook_cached_commit.is_some()
            );
            // is_recursive = false because leaf proofs aggregate app proofs (not recursive)
            let prover: LeafProver = LeafProver::from_pk::<RecursionEngine>(
                app_vk,
                edge_artifacts.agg_stark_pk.prefix.leaf.clone(),
                false,
                def_hook_cached_commit,
            );
            info!("LeafProverInstance created successfully");

            Ok(Self { prover })
        }
    }

    /// Execute leaf proving using a provided prover instance.
    ///
    /// This is the main entry point for real proving with prover reuse.
    #[instrument(skip_all, fields(
        proof_id = %job.context.proof_uuid,
        segment_start = job.segment_start,
        segment_end = job.segment_end,
        num_app_proofs = job.app_proofs.len()
    ))]
    pub fn prove_leaf_with_prover(
        job: LeafProverJob,
        prover_instance: &LeafProverInstance,
    ) -> ProverResult {
        info!(
            "Starting leaf prove (with prover): segment_start={}, segment_end={}, app_proofs={}",
            job.segment_start,
            job.segment_end,
            job.app_proofs.len()
        );

        match prove_leaf_impl_with_prover(job, prover_instance) {
            Ok(results) => ProverResult::Success(results),
            Err(e) => ProverResult::Error(format!("Leaf prove failed: {}", e)),
        }
    }

    fn prove_leaf_impl_with_prover(
        job: LeafProverJob,
        prover_instance: &LeafProverInstance,
    ) -> Result<Vec<ProofResult>> {
        // Validate inputs
        if job.app_proofs.is_empty() {
            return Err(eyre::eyre!("Cannot aggregate empty app_proofs list"));
        }
        if job.segment_start > job.segment_end {
            return Err(eyre::eyre!(
                "Invalid segment range: start {} > end {}",
                job.segment_start,
                job.segment_end
            ));
        }
        // Validate proof count matches segment range (inclusive)
        let expected_count = job.segment_end - job.segment_start + 1;
        if job.app_proofs.len() != expected_count {
            return Err(eyre::eyre!(
                "Proof count {} does not match segment range [{}, {}] (expected {})",
                job.app_proofs.len(),
                job.segment_start,
                job.segment_end,
                expected_count
            ));
        }
        // Validate max children to avoid panic in agg_prove
        if job.app_proofs.len() > MAX_NUM_CHILDREN_LEAF {
            return Err(eyre::eyre!(
                "Too many app proofs: {} exceeds MAX_NUM_CHILDREN_LEAF={}",
                job.app_proofs.len(),
                MAX_NUM_CHILDREN_LEAF
            ));
        }

        let leaf_prover = &prover_instance.prover;

        // Get user public values from the last proof if present
        // (only the terminal segment should have user public values)
        let user_public_values = job
            .app_proofs
            .last()
            .and_then(|p| p.user_public_values.clone());

        // Extract the raw proofs from ProofWithPublicValue, consuming the job to avoid cloning
        let proofs: Vec<_> = job.app_proofs.into_iter().map(|p| p.proof).collect();

        info!("Running leaf agg_prove with {} proofs", proofs.len(),);

        // Run the leaf aggregation proof
        // ChildVkKind::App because leaf proofs aggregate app proofs
        let _ = telemetry::span_timing::drain_span_timings(); // clear stale timings
        let start = std::time::Instant::now();
        let leaf_proof =
            leaf_prover.agg_prove_no_def::<RecursionEngine>(&proofs, ChildVkKind::App)?;
        let prove_time_ms = start.elapsed().as_millis() as u64;
        let sub_metrics = telemetry::span_timing::drain_span_timings();

        info!(
            "Generated leaf proof for segments [{}, {}] ({}ms), spans={}",
            job.segment_start, job.segment_end, prove_time_ms,
            telemetry::span_timing::format_span_timings(&sub_metrics)
        );

        // Create the result structure - propagate user_public_values from app proofs.
        // The wire-side proof field is bincode-encoded.
        let leaf_proof_typed = ProofWithPublicValue {
            proof: leaf_proof,
            user_public_values, // Propagate from terminal app proof
        };
        let result = LeafProof {
            context: job.context,
            state: LeafProofState {
                proof: Some(proof::encode_proof(&leaf_proof_typed)?),
                segment_start: job.segment_start,
                segment_end: job.segment_end,
                prove_time_ms,
                sub_metrics,
            },
        };

        Ok(vec![ProofResult::Leaf(result)])
    }

    /// Legacy implementation that creates its own prover instance.
    ///
    /// This is kept for backward compatibility but should be avoided in favor
    /// of `prove_leaf_with_prover` which reuses prover instances.
    pub fn prove_leaf_impl(job: LeafProverJob) -> Result<Vec<ProofResult>> {
        // Create a temporary prover instance (inefficient - avoid in production)
        let prover_instance = LeafProverInstance::new()?;
        prove_leaf_impl_with_prover(job, &prover_instance)
    }
}

#[cfg(not(feature = "mock-provers"))]
pub use real_impl::{prove_leaf_with_prover, LeafProverInstance};

#[cfg(not(feature = "mock-provers"))]
fn prove_leaf_impl(job: LeafProverJob) -> Result<Vec<ProofResult>> {
    real_impl::prove_leaf_impl(job)
}

#[cfg(all(test, feature = "mock-provers"))]
mod tests {
    use super::*;
    use protocol::ProofContext;

    #[test]
    fn test_leaf_prove_mock() {
        // Create mock app proofs
        let app_proofs: Vec<ProofWithPublicValue<F>> = (0..4)
            .map(|_| ProofWithPublicValue {
                proof: vec![0u8; 256],
                public_values: vec![F::default(); 4],
            })
            .collect();

        let job = LeafProverJob {
            context: ProofContext::new(
                "test-proof-id".to_string(),
                protocol::ProgramRef::new("test-program", 1),
                Default::default(),
            ),
            app_proofs,
            segment_start: 0,
            segment_end: 4,
        };

        let result = prove_leaf(job);

        match result {
            ProverResult::Success(proofs) => {
                assert_eq!(proofs.len(), 1);
                match &proofs[0] {
                    ProofResult::Leaf(leaf_proof) => {
                        assert_eq!(leaf_proof.state.segment_start, 0);
                        assert_eq!(leaf_proof.state.segment_end, 4);
                    }
                    _ => panic!("Expected Leaf proof"),
                }
            }
            ProverResult::Error(e) => panic!("Unexpected error: {}", e),
            ProverResult::Canceled => panic!("Unexpected cancellation"),
        }
    }

    #[test]
    fn test_leaf_prove_empty_app_proofs() {
        let job = LeafProverJob {
            context: ProofContext::new(
                "test-proof-id".to_string(),
                protocol::ProgramRef::new("test-program", 1),
                Default::default(),
            ),
            app_proofs: vec![], // Empty!
            segment_start: 0,
            segment_end: 0,
        };

        let result = prove_leaf(job);

        match result {
            ProverResult::Error(e) => {
                assert!(
                    e.contains("empty"),
                    "Error should mention empty list: {}",
                    e
                );
            }
            ProverResult::Success(_) => panic!("Should have failed with empty app_proofs"),
            ProverResult::Canceled => panic!("Unexpected cancellation"),
        }
    }

    #[test]
    fn test_leaf_prove_invalid_segment_range() {
        let app_proofs: Vec<ProofWithPublicValue<F>> = (0..2)
            .map(|_| ProofWithPublicValue {
                proof: vec![0u8; 256],
                public_values: vec![F::default(); 4],
            })
            .collect();

        let job = LeafProverJob {
            context: ProofContext::new(
                "test-proof-id".to_string(),
                protocol::ProgramRef::new("test-program", 1),
                Default::default(),
            ),
            app_proofs,
            segment_start: 10, // Invalid: start > end
            segment_end: 5,
        };

        let result = prove_leaf(job);

        match result {
            ProverResult::Error(e) => {
                assert!(
                    e.contains("Invalid segment range"),
                    "Error should mention invalid range: {}",
                    e
                );
            }
            ProverResult::Success(_) => panic!("Should have failed with invalid segment range"),
            ProverResult::Canceled => panic!("Unexpected cancellation"),
        }
    }
}
