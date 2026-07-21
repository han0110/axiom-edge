//! Internal Prover implementation.
//!
//! This prover handles internal node proving in the recursion tree,
//! aggregating leaf proofs or other internal proofs.
//!
//! # Architecture
//!
//! The `InternalProverInstance` struct holds the reusable internal prover components
//! that are expensive to create. Worker threads create one instance at startup and
//! reuse it across all jobs via `prove_internal_with_prover`.
//!
//! The internal prover handles multiple layer types:
//! - Layer 0: Aggregates leaf proofs
//! - Layer 1+: Recursively aggregates internal proofs
//! - Final proofs include compression
//!
//! The `prove_internal` function is used in mock mode and creates instances internally.

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
use protocol::{InternalProof, InternalProofState};

use super::{InternalProverJob, ProverResult};

/// Execute internal proving to aggregate child proofs.
#[instrument(skip_all, fields(
    proof_id = %job.context.proof_uuid,
    layer_idx = job.layer_idx,
    segment_start = job.segment_start,
    segment_end = job.segment_end,
    is_final = job.is_final_proof,
    num_child_proofs = job.child_proofs.len()
))]
pub fn prove_internal(job: InternalProverJob) -> ProverResult {
    info!(
        "Starting internal prove: layer={}, segment=[{}, {}), is_final={}, children={}",
        job.layer_idx,
        job.segment_start,
        job.segment_end,
        job.is_final_proof,
        job.child_proofs.len()
    );

    match prove_internal_impl(job) {
        Ok(results) => ProverResult::Success(results),
        Err(e) => ProverResult::Error(format!("Internal prove failed: {}", e)),
    }
}

#[cfg(feature = "mock-provers")]
fn prove_internal_impl(job: InternalProverJob) -> Result<Vec<ProofResult>> {
    // Validate inputs
    if job.child_proofs.is_empty() {
        return Err(eyre::eyre!("Cannot aggregate empty child_proofs list"));
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
    let prove_duration = Duration::from_millis(300);
    std::thread::sleep(prove_duration);
    let prove_time_ms = start.elapsed().as_millis() as u64;

    // Final proofs take longer due to compression
    let compression_time_ms = if job.is_final_proof {
        let compress_start = Instant::now();
        std::thread::sleep(Duration::from_millis(200));
        compress_start.elapsed().as_millis() as u64
    } else {
        0
    };

    // Generate mock internal proof — wire-side payload is bincode-encoded.
    let mock_proof = ProofWithPublicValue::<F> {
        proof: vec![0u8; 1024],
        public_values: vec![F::default(); 16],
    };
    let proof = InternalProof {
        context: job.context.clone(),
        state: InternalProofState {
            proof: Some(proof::encode_proof(&mock_proof)?),
            layer_idx: job.layer_idx,
            segment_start: job.segment_start,
            segment_end: job.segment_end,
            prove_time_ms,
            compression_time_ms,
            sub_metrics: std::collections::HashMap::new(),
            deferral_merkle_proofs_bytes: None,
            // The prover emits the raw internal proof; the dedicated-mode
            // ready-for-evm flag is set later by the handler, not here.
            ready_for_evm: false,
        },
    };

    info!(
        "Generated mock internal proof: layer={}, segment=[{}, {}), is_final={}, prove={}ms, compress={}ms",
        job.layer_idx, job.segment_start, job.segment_end, job.is_final_proof,
        prove_time_ms, compression_time_ms
    );

    Ok(vec![ProofResult::Internal(proof)])
}

// ============================================================================
// Real prover implementation (not mock-provers)
// ============================================================================

#[cfg(not(feature = "mock-provers"))]
mod real_impl {
    use super::super::real_prover_types::{ChildVkKind, InternalProver, RecursionEngine};
    use super::*;
    use crate::artifacts::ArtifactStore;
    use proof::ProofWithPublicValue;
    use protocol::{InternalProof, InternalProofState};
    use sdk_v2::config::MAX_NUM_CHILDREN_INTERNAL;
    use std::sync::Arc;

    /// Internal prover instance that holds reusable prover state.
    ///
    /// This struct is created once per worker thread and reused across jobs.
    /// It contains all the prover configurations needed for different layers.
    pub struct InternalProverInstance {
        /// Prover for layer 0 (aggregates leaf proofs).
        pub internal_for_leaf_prover: InternalProver,
        /// Prover for layer 1+ (recursive aggregation).
        pub internal_recursive_prover: InternalProver,
    }

    impl InternalProverInstance {
        /// Create a new internal prover instance from global artifacts.
        ///
        /// This should be called once per worker thread at startup.
        pub fn new() -> Result<Self> {
            let artifact_store = ArtifactStore::global()
                .ok_or_else(|| eyre::eyre!("Artifact store not initialized"))?;
            let edge_artifacts = artifact_store
                .get_edge_artifacts()
                .ok_or_else(|| eyre::eyre!("Edge artifacts not loaded"))?;

            // `Some(def_hook_cached_commit)` iff the worker is in
            // deferral mode. `EdgeArtifacts.agg_stark_pk` already points
            // at the deferral keyset's `agg_pk` in that case, so
            // everything else is unchanged.
            let def_hook_cached_commit = edge_artifacts.def_hook_cached_commit();
            let is_deferral_deployment = edge_artifacts.is_deferral_deployment();

            info!(
                "Creating InternalProverInstance (deferral_mode={})",
                is_deferral_deployment
            );

            // Layer 0 prover (aggregates leaf proofs).
            let leaf_vk = Arc::new(edge_artifacts.agg_stark_pk.prefix.leaf.get_vk());
            let internal_for_leaf_prover: InternalProver = InternalProver::from_pk::<RecursionEngine>(
                leaf_vk,
                edge_artifacts.agg_stark_pk.prefix.internal_for_leaf.clone(),
                false,
                def_hook_cached_commit,
            );
            info!("Created internal_for_leaf_prover");

            // Layer 1+ prover (recursive, can verify itself).
            let internal_for_leaf_vk = Arc::new(
                edge_artifacts
                    .agg_stark_pk
                    .prefix
                    .internal_for_leaf
                    .get_vk(),
            );
            let internal_recursive_prover: InternalProver =
                InternalProver::from_pk::<RecursionEngine>(
                    internal_for_leaf_vk,
                    edge_artifacts.agg_stark_pk.internal_recursive.clone(),
                    true,
                    def_hook_cached_commit,
                );
            info!("Created internal_recursive_prover");

            info!("InternalProverInstance created successfully");

            Ok(Self {
                internal_for_leaf_prover,
                internal_recursive_prover,
            })
        }
    }

    /// Execute internal proving using a provided prover instance.
    ///
    /// This is the main entry point for real proving with prover reuse.
    #[instrument(skip_all, fields(
        proof_id = %job.context.proof_uuid,
        layer_idx = job.layer_idx,
        segment_start = job.segment_start,
        segment_end = job.segment_end,
        is_final = job.is_final_proof,
        num_child_proofs = job.child_proofs.len()
    ))]
    pub fn prove_internal_with_prover(
        job: InternalProverJob,
        prover_instance: &InternalProverInstance,
    ) -> ProverResult {
        info!(
            "Starting internal prove (with prover): layer={}, segment=[{}, {}), is_final={}, children={}",
            job.layer_idx,
            job.segment_start,
            job.segment_end,
            job.is_final_proof,
            job.child_proofs.len()
        );

        match prove_internal_impl_with_prover(job, prover_instance) {
            Ok(results) => ProverResult::Success(results),
            Err(e) => ProverResult::Error(format!("Internal prove failed: {}", e)),
        }
    }

    fn prove_internal_impl_with_prover(
        job: InternalProverJob,
        prover_instance: &InternalProverInstance,
    ) -> Result<Vec<ProofResult>> {
        // Validate inputs
        if job.child_proofs.is_empty() {
            return Err(eyre::eyre!("Cannot aggregate empty child_proofs list"));
        }
        if job.segment_start > job.segment_end {
            return Err(eyre::eyre!(
                "Invalid segment range: start {} > end {}",
                job.segment_start,
                job.segment_end
            ));
        }
        // Validate max children to avoid panic in agg_prove
        if job.child_proofs.len() > MAX_NUM_CHILDREN_INTERNAL {
            return Err(eyre::eyre!(
                "Too many child proofs: {} exceeds MAX_NUM_CHILDREN_INTERNAL={}",
                job.child_proofs.len(),
                MAX_NUM_CHILDREN_INTERNAL
            ));
        }

        // Extract user_public_values from child proofs (find the last one that has it)
        // This propagates user PVs from the terminal segment up through the recursion tree
        let user_public_values = job
            .child_proofs
            .iter()
            .rev()
            .find_map(|p| p.user_public_values.as_ref())
            .cloned();

        // Extract proofs, consuming the job to avoid cloning
        let proofs: Vec<_> = job.child_proofs.into_iter().map(|p| p.proof).collect();

        let _ = telemetry::span_timing::drain_span_timings(); // clear stale timings
        let agg_start = std::time::Instant::now();
        let internal_proof = if job.layer_idx == 0 {
            info!(
                "Running internal agg_prove (layer 0) with {} leaf proofs",
                proofs.len()
            );
            // ChildVkKind::Standard because children are leaf proofs.
            prover_instance
                .internal_for_leaf_prover
                .agg_prove_no_def::<RecursionEngine>(&proofs, ChildVkKind::Standard)?
        } else if job.layer_idx == 1 {
            info!(
                "Running internal agg_prove (layer 1) with {} internal proofs",
                proofs.len()
            );
            // ChildVkKind::Standard because children are internal_for_leaf
            // (NOT internal_recursive).
            prover_instance
                .internal_recursive_prover
                .agg_prove_no_def::<RecursionEngine>(&proofs, ChildVkKind::Standard)?
        } else {
            info!(
                "Running internal agg_prove (layer {}) with {} internal proofs, recursive_self",
                job.layer_idx,
                proofs.len()
            );
            // ChildVkKind::RecursiveSelf because children are from
            // internal_recursive_pk.
            prover_instance
                .internal_recursive_prover
                .agg_prove_no_def::<RecursionEngine>(&proofs, ChildVkKind::RecursiveSelf)?
        };

        let prove_time_ms = agg_start.elapsed().as_millis() as u64;
        let mut sub_metrics = telemetry::span_timing::drain_span_timings();

        // Skip the `is_final_proof` wrap only when THIS proof carries a
        // deferral tail — the tail merge's `wrap_proof` supplies the
        // canonical `ADDITIONAL_INTERNAL_RECURSIVE_LAYERS=1` layer instead.
        //
        // This is per-proof, NOT deployment-wide. The SDK's
        // `StarkProver::prove` runs the final wrap unconditionally and only
        // gates `prove_def`/`prove_mixed` on `!def_inputs.is_empty()`. So a
        // no-deferral proof on a deferral deployment (`is_deferral_deployment`
        // true but no tail attached) MUST fall through to the normal wrap
        // below — otherwise the terminal proof is one internal-recursive layer
        // short of the canonical shape and fails to verify.
        //
        // `agg_prove_no_def` hard-codes `ProofsType::Vm` (its contract — see
        // `openvm/crates/continuations/src/prover/inner/mod.rs:99–109`), which
        // is exactly right for the non-deferral wrap.
        let skip_wrap_for_deferral = job.is_final_proof && job.proof_has_deferral;
        let (final_proof, compression_time_ms) = if job.is_final_proof && !skip_wrap_for_deferral {
            info!("Wrapping final internal proof...");
            assert!(
                job.layer_idx >= 1,
                "Internal for leaf should not be final proof!!"
            );

            let wrap_start = std::time::Instant::now();

            // Single wrap: aggregate internal_proof into internal_recursive.
            info!("Wrap: agg_prove with RecursiveSelf");
            let wrapped_proof = prover_instance
                .internal_recursive_prover
                .agg_prove_no_def::<RecursionEngine>(
                    &[internal_proof],
                    ChildVkKind::RecursiveSelf,
                )?;
            let wrap_time_ms = wrap_start.elapsed().as_millis() as u64;
            // Merge wrap sub-step timings into sub_metrics
            for (k, v) in telemetry::span_timing::drain_span_timings() {
                *sub_metrics.entry(k).or_insert(0.0) += v;
            }
            info!(
                "Final internal proof wrapped successfully ({}ms)",
                wrap_time_ms
            );
            (wrapped_proof, wrap_time_ms)
        } else {
            if skip_wrap_for_deferral {
                info!(
                    "Skipping pre-merge final-internal wrap (deferral deployment) — \
                     the tail-merge wrap_proof in run_evm_prove is the canonical \
                     ADDITIONAL_INTERNAL_RECURSIVE_LAYERS=1 equivalent for the deferral path."
                );
            }
            (internal_proof, 0)
        };

        info!(
            "Generated internal proof: layer={}, segment=[{}, {}], is_final={}, prove={}ms, compress={}ms, spans={}",
            job.layer_idx, job.segment_start, job.segment_end, job.is_final_proof,
            prove_time_ms, compression_time_ms,
            telemetry::span_timing::format_span_timings(&sub_metrics)
        );

        // Create the result structure - propagate user_public_values from child proofs.
        // The wire-side proof field is bincode-encoded.
        let internal_proof_typed = ProofWithPublicValue {
            proof: final_proof,
            user_public_values, // Propagate from terminal segment
        };
        let result = InternalProof {
            context: job.context,
            state: InternalProofState {
                proof: Some(proof::encode_proof(&internal_proof_typed)?),
                layer_idx: job.layer_idx,
                segment_start: job.segment_start,
                segment_end: job.segment_end,
                prove_time_ms,
                compression_time_ms,
                sub_metrics,
                // The internal prover never merges deferral tails; for a
                // stark-mode deferral job the handler runs the merge on the
                // final internal proof and sets this field there.
                deferral_merkle_proofs_bytes: None,
                // Raw internal proof; the ready-for-evm flag (and the
                // post-merge proof/merkle it carries) is set by the handler.
                ready_for_evm: false,
            },
        };

        Ok(vec![ProofResult::Internal(result)])
    }

    /// Legacy implementation that creates its own prover instances.
    ///
    /// This is kept for backward compatibility but should be avoided in favor
    /// of `prove_internal_with_prover` which reuses prover instances.
    pub fn prove_internal_impl(job: InternalProverJob) -> Result<Vec<ProofResult>> {
        // Create a temporary prover instance (inefficient - avoid in production)
        let prover_instance = InternalProverInstance::new()?;
        prove_internal_impl_with_prover(job, &prover_instance)
    }
}

#[cfg(not(feature = "mock-provers"))]
pub use real_impl::{prove_internal_with_prover, InternalProverInstance};

#[cfg(not(feature = "mock-provers"))]
fn prove_internal_impl(job: InternalProverJob) -> Result<Vec<ProofResult>> {
    real_impl::prove_internal_impl(job)
}

#[cfg(all(test, feature = "mock-provers"))]
mod tests {
    use super::*;
    use protocol::ProofContext;

    #[test]
    fn test_internal_prove_mock() {
        // Create mock child proofs (could be leaf or internal proofs)
        let child_proofs: Vec<ProofWithPublicValue<F>> = (0..2)
            .map(|_| ProofWithPublicValue {
                proof: vec![0u8; 512],
                public_values: vec![F::default(); 8],
            })
            .collect();

        let job = InternalProverJob {
            context: ProofContext::new(
                "test-proof-id".to_string(),
                protocol::ProgramRef::new("test-program", 1),
                Default::default(),
            ),
            child_proofs,
            layer_idx: 0,
            segment_start: 0,
            segment_end: 8,
            is_final_proof: false,
            proof_has_deferral: false,
        };

        let result = prove_internal(job);

        match result {
            ProverResult::Success(proofs) => {
                assert_eq!(proofs.len(), 1);
                match &proofs[0] {
                    ProofResult::Internal(internal_proof) => {
                        assert_eq!(internal_proof.state.layer_idx, 0);
                        assert_eq!(internal_proof.state.segment_start, 0);
                        assert_eq!(internal_proof.state.segment_end, 8);
                    }
                    _ => panic!("Expected Internal proof"),
                }
            }
            ProverResult::Error(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_final_internal_prove_mock() {
        // Create mock child proofs
        let child_proofs: Vec<ProofWithPublicValue<F>> = (0..2)
            .map(|_| ProofWithPublicValue {
                proof: vec![0u8; 1024],
                public_values: vec![F::default(); 16],
            })
            .collect();

        let job = InternalProverJob {
            context: ProofContext::new(
                "test-proof-id".to_string(),
                protocol::ProgramRef::new("test-program", 1),
                Default::default(),
            ),
            child_proofs,
            layer_idx: 2,
            segment_start: 0,
            segment_end: 16,
            is_final_proof: true,
            proof_has_deferral: false,
        };

        let result = prove_internal(job);

        match result {
            ProverResult::Success(proofs) => {
                assert_eq!(proofs.len(), 1);
                match &proofs[0] {
                    ProofResult::Internal(internal_proof) => {
                        assert_eq!(internal_proof.state.layer_idx, 2);
                    }
                    _ => panic!("Expected Internal proof"),
                }
            }
            ProverResult::Error(e) => panic!("Unexpected error: {}", e),
        }
    }
}
