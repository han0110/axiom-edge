//! On-disk persistence for completed proofs and failure snapshots.
//!
//! - [`ProofState::persist_final_proof_to_disk`] writes the final STARK
//!   proof as an openvm-codec `verify_stark::VmStarkProof` (the proof, its
//!   user public values, and any deferral merkle proofs; optionally
//!   zstd-compressed for external upload flows).
//! - [`ProofState::persist_leaf_failure_app_proofs_to_disk`] snapshots the
//!   accumulated app proofs when leaf proving has hit the known
//!   `LogupZerocheck` nonzero-root-sum failure, so the failure can be
//!   reproduced offline.
//!
//! These are blocking I/O operations. They're called while the per-proof
//! `Mutex<ProofState>` is held, from `finalize_proof`'s `spawn_blocking`
//! section (so they don't stall the async runtime, but they do extend the
//! window during which other tasks block on this proof's mutex).

use chrono::{DateTime, Utc};
use eyre::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use protocol::{AppProofState, ErrorResult, ProofContext, ProofType};

use super::state::{ProofState, ProofStatus};

const FINAL_PROOF_ZSTD_LEVEL: i32 = 19;

const LEAF_LOGUP_NONZERO_ROOT_SUM_ERROR: &str =
    "LogupZerocheck: Fractional sumcheck: nonzero root sum";

/// Range of segments covered by a fully-aggregated leaf proof, used in the
/// [`PersistedLeafFailureAppProofs`] snapshot to record which leaves had
/// already completed before the failure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompletedLeafBatch {
    pub segment_start: usize,
    pub segment_end: usize,
}

/// On-disk snapshot written when leaf proving hits the known logup
/// nonzero-root-sum failure. Captures everything needed to reproduce or
/// inspect the failure offline.
#[derive(Clone, Serialize, Deserialize)]
pub struct PersistedLeafFailureAppProofs {
    pub persisted_at: DateTime<Utc>,
    pub context: ProofContext,
    pub error: ErrorResult,
    pub num_segments: Option<usize>,
    pub leaf_arity: usize,
    pub app_proofs: Vec<AppProofState>,
    pub completed_leaf_batches: Vec<CompletedLeafBatch>,
}

fn is_leaf_logup_nonzero_root_sum_error(result: &ErrorResult) -> bool {
    let step = result.step.to_ascii_lowercase();
    let is_leaf_step = step.contains("leaf") || result.error.contains("Leaf prove failed:");
    is_leaf_step && result.error.contains(LEAF_LOGUP_NONZERO_ROOT_SUM_ERROR)
}

impl ProofState {
    /// Persist the completed final proof to disk as bincode, optionally
    /// zstd-compressed for external upload flows.
    ///
    /// Branches on `context.proof_type`:
    /// - `Stark` ⇒ writes `{uuid}.proof.bin`: a `verify_stark::VmStarkProof`
    ///   encoded with the openvm codec (`encode_to_vec`) — the proof, its user
    ///   public values, and (for deferral jobs) the `DeferralMerkleProofs`.
    ///   This is exactly the format `verify_edge_final_proof`'s
    ///   `load_final_proof` decodes.
    /// - `Evm`   ⇒ writes `{uuid}.evm.bin` (raw bincode-encoded evm proof
    ///   bytes — the same wire payload the worker posted, kept opaque so
    ///   this crate doesn't have to decode the halo2 proof).
    pub fn persist_final_proof_to_disk(
        &mut self,
        output_dir: &Path,
        compress: bool,
    ) -> Result<Option<PathBuf>> {
        if !matches!(self.status, ProofStatus::Completed) {
            return Ok(None);
        }

        if let Some(path) = &self.persisted_final_proof_path {
            return Ok(Some(PathBuf::from(path)));
        }

        let (bytes, suffix) = match self.context.proof_type {
            ProofType::Stark => {
                let final_state = match self.final_internal_state() {
                    Some(state) => state,
                    None => return Ok(None),
                };
                let proof_bytes = match &final_state.proof {
                    Some(bytes) => bytes,
                    None => return Ok(None),
                };
                // Reconstruct a `VmStarkProof` (proof + user public values +
                // any deferral merkle proofs) and encode it with the openvm
                // codec — the format `load_final_proof` decodes. For a
                // deferral job the merkle proofs are required to verify; a
                // non-deferral job encodes `deferral_merkle_proofs = None`.
                #[cfg(not(feature = "mock-provers"))]
                let bytes = {
                    // `Encode` provides `VmStarkProof::encode_to_vec`;
                    // `DeferralMerkleProofs` has an inherent `decode(reader)`.
                    use openvm_stark_backend::codec::Encode;

                    let pwpv = proof::decode_proof(proof_bytes)?;
                    let user_pvs_proof = pwpv.user_public_values.ok_or_else(|| {
                        eyre::eyre!(
                            "stark proof for {} has no user_public_values; cannot build VmStarkProof",
                            self.context.proof_uuid
                        )
                    })?;
                    // A real deferral proof carries its merkle proofs on the
                    // final internal state (the tail merge set them). A
                    // no-deferral proof on a deferral deployment has none there
                    // — fall back to the depth-0 proofs the terminal app worker
                    // buffered on this `ProofState`. On a non-deferral
                    // deployment both are `None` (the VK has no deferral hook).
                    let merkle_bytes = final_state
                        .deferral_merkle_proofs_bytes
                        .as_ref()
                        .or(self.deferral_depth0_merkle_proofs_bytes.as_ref());
                    let deferral_merkle_proofs = match merkle_bytes {
                        Some(mb) => {
                            let mut reader = std::io::Cursor::new(mb.as_slice());
                            Some(
                                verify_stark::deferral::DeferralMerkleProofs::decode(&mut reader)
                                    .map_err(|e| {
                                    eyre::eyre!(
                                        "failed to decode deferral merkle proofs for {}: {e}",
                                        self.context.proof_uuid
                                    )
                                })?,
                            )
                        }
                        None => None,
                    };
                    verify_stark::VmStarkProof {
                        inner: pwpv.proof,
                        user_pvs_proof,
                        deferral_merkle_proofs,
                    }
                    .encode_to_vec()
                    .map_err(|e| {
                        eyre::eyre!(
                            "failed to encode VmStarkProof for {}: {e}",
                            self.context.proof_uuid
                        )
                    })?
                };
                // Mock builds can't construct a real `VmStarkProof` (the mock
                // `ProofWithPublicValue` holds raw bytes, not a `Proof<SC>`);
                // persist the raw proof bytes. Not exercised at runtime — mock
                // e2e only persists Evm proofs.
                #[cfg(feature = "mock-provers")]
                let bytes = proof_bytes.clone();

                (bytes, "proof.bin")
            }
            ProofType::Evm => {
                let evm_bytes = match self.get_evm_proof() {
                    Some(b) => b,
                    None => return Ok(None),
                };
                (evm_bytes, "evm.bin")
            }
        };

        std::fs::create_dir_all(output_dir)?;

        let final_path = output_dir.join(format!("{}.{}", self.context.proof_uuid, suffix));
        let temp_path = output_dir.join(format!("{}.{}.tmp", self.context.proof_uuid, suffix));
        let bytes = if compress {
            zstd::encode_all(&bytes[..], FINAL_PROOF_ZSTD_LEVEL)?
        } else {
            bytes
        };

        std::fs::write(&temp_path, bytes)?;
        std::fs::rename(&temp_path, &final_path)?;

        self.persisted_final_proof_path = Some(final_path.display().to_string());
        Ok(Some(final_path))
    }

    fn should_persist_leaf_failure_app_proofs(&self) -> bool {
        matches!(
            self.status,
            ProofStatus::Failed(_) | ProofStatus::Failing(_)
        ) && self
            .last_error_result
            .as_ref()
            .is_some_and(is_leaf_logup_nonzero_root_sum_error)
    }

    /// Persist the current app proofs when the proof failed with the known
    /// leaf logup nonzero-root-sum error.
    pub fn persist_leaf_failure_app_proofs_to_disk(
        &mut self,
        output_dir: &Path,
    ) -> Result<Option<PathBuf>> {
        if !self.should_persist_leaf_failure_app_proofs() {
            return Ok(None);
        }

        if let Some(path) = &self.persisted_leaf_failure_app_proofs_path {
            return Ok(Some(PathBuf::from(path)));
        }

        let Some(error) = self.last_error_result.clone() else {
            return Ok(None);
        };

        std::fs::create_dir_all(output_dir)?;

        let final_path = output_dir.join(format!(
            "{}.leaf-logup.app-proofs.bin",
            self.context.proof_uuid
        ));
        let temp_path = output_dir.join(format!(
            "{}.leaf-logup.app-proofs.bin.tmp",
            self.context.proof_uuid
        ));

        let mut app_proofs: Vec<_> = self.app_proofs.values().cloned().collect();
        app_proofs.sort_by_key(|state| state.segment_idx);

        let mut completed_leaf_batches: Vec<_> = self
            .leaf_proofs
            .values()
            .map(|state| CompletedLeafBatch {
                segment_start: state.segment_start,
                segment_end: state.segment_end,
            })
            .collect();
        completed_leaf_batches.sort_by_key(|batch| batch.segment_start);

        let snapshot = PersistedLeafFailureAppProofs {
            persisted_at: Utc::now(),
            context: self.context.clone(),
            error,
            num_segments: self.num_segments,
            leaf_arity: self.leaf_arity,
            app_proofs,
            completed_leaf_batches,
        };
        let bytes = bincode::serialize(&snapshot)?;

        std::fs::write(&temp_path, bytes)?;
        std::fs::rename(&temp_path, &final_path)?;

        self.persisted_leaf_failure_app_proofs_path = Some(final_path.display().to_string());
        Ok(Some(final_path))
    }
}

#[cfg(all(test, feature = "mock-provers"))]
mod tests {
    use super::*;
    use proof::{ProofWithPublicValue, F};
    use protocol::{ErrorResult, ProofContext, ProofResult};
    use std::collections::HashMap;

    fn make_context() -> ProofContext {
        ProofContext::new(
            "test-proof".to_string(),
            protocol::ProgramRef::new("test-program", 1),
            Default::default(),
        )
    }

    fn make_mock_proof() -> Vec<u8> {
        proof::encode_proof(&ProofWithPublicValue::<F> {
            proof: vec![0u8; 256],
            public_values: vec![F::default(); 4],
        })
        .expect("mock proof encodes")
    }

    #[test]
    fn test_persist_leaf_failure_app_proofs_to_disk() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);
        state.num_segments = Some(4);

        for segment_idx in 0..4 {
            state.app_proofs.insert(
                segment_idx,
                AppProofState {
                    proof: Some(make_mock_proof()),
                    segment_idx,
                    prove_time_ms: 10,
                    fastfwd_time_ms: 3,
                    stark_prove_time_ms: 7,
                    queue_wait_ms: 0,
                    metered_time_ms: 0,
                    sub_metrics: HashMap::new(),
                    final_merkle_path_bytes: None,
                    deferral_merkle_proofs_bytes: None,
                    worker_id: 0,
                    completed_at_ms: 0,
                },
            );
        }

        state
            .handle_proof_result(ProofResult::Error(ErrorResult {
                context: context.clone(),
                step: "unknown".to_string(),
                error: "Leaf prove failed: LogupZerocheck: Fractional sumcheck: nonzero root sum"
                    .to_string(),
            }))
            .unwrap();

        let output_dir = tempfile::tempdir().unwrap();
        let path = state
            .persist_leaf_failure_app_proofs_to_disk(output_dir.path())
            .unwrap()
            .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let snapshot: PersistedLeafFailureAppProofs = bincode::deserialize(&bytes).unwrap();

        assert_eq!(snapshot.context.proof_uuid, context.proof_uuid);
        assert_eq!(snapshot.num_segments, Some(4));
        assert_eq!(snapshot.leaf_arity, 4);
        assert_eq!(snapshot.app_proofs.len(), 4);
        assert_eq!(snapshot.app_proofs[0].segment_idx, 0);
        assert_eq!(snapshot.app_proofs[3].segment_idx, 3);
        assert_eq!(snapshot.completed_leaf_batches.len(), 0);
        assert_eq!(
            state.persisted_leaf_failure_app_proofs_path,
            Some(path.display().to_string())
        );
    }

    #[test]
    fn test_unrelated_failure_does_not_persist_leaf_app_proofs() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);

        state
            .handle_proof_result(ProofResult::Error(ErrorResult {
                context,
                step: "LeafProve".to_string(),
                error: "Leaf prove failed: unrelated".to_string(),
            }))
            .unwrap();

        let output_dir = tempfile::tempdir().unwrap();
        assert!(state
            .persist_leaf_failure_app_proofs_to_disk(output_dir.path())
            .unwrap()
            .is_none());
    }
}
