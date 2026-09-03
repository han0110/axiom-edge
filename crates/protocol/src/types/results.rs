//! Result types for Edge mode.
//!
//! Proof and segment payloads are carried as opaque `Vec<u8>` (bincode-encoded
//! `proof::ProofWithPublicValue<F>` / `proof::Segment`). Decoders that need to
//! inspect proof internals add the `proof` crate.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::requests::{ProofBytes, SegmentBytes};
use super::{MessageEnvelope, ProofContext, WithProofContext};

/// Result types returned from prove-service to prove-manager.
#[derive(Clone, Serialize, Deserialize)]
pub enum ProofResult {
    ExecuteE2(ExecuteE2Result),
    App(AppProof),
    Leaf(LeafProof),
    Internal(InternalProof),
    Evm(EvmProof),
    Error(ErrorResult),
}

impl ProofResult {
    pub fn kind(&self) -> &str {
        match self {
            ProofResult::ExecuteE2(_) => "execute_e2",
            ProofResult::App(_) => "app",
            ProofResult::Leaf(_) => "leaf",
            ProofResult::Internal(_) => "internal",
            ProofResult::Evm(_) => "evm",
            ProofResult::Error(_) => "error",
        }
    }

    /// Record the worker and the receipt time on the carried state.
    pub fn stamp(&mut self, worker_id: usize, completed_at_ms: u64) {
        let (state_worker_id, state_completed_at_ms) = match self {
            ProofResult::App(r) => (&mut r.state.worker_id, &mut r.state.completed_at_ms),
            ProofResult::Leaf(r) => (&mut r.state.worker_id, &mut r.state.completed_at_ms),
            ProofResult::Internal(r) => (&mut r.state.worker_id, &mut r.state.completed_at_ms),
            ProofResult::ExecuteE2(_) | ProofResult::Evm(_) | ProofResult::Error(_) => return,
        };
        *state_worker_id = worker_id;
        *state_completed_at_ms = completed_at_ms;
    }
}

impl WithProofContext for ProofResult {
    fn proof_uuid(&self) -> &str {
        &self.context().proof_uuid
    }

    fn context(&self) -> &ProofContext {
        match self {
            ProofResult::ExecuteE2(r) => &r.context,
            ProofResult::App(r) => &r.context,
            ProofResult::Leaf(r) => &r.context,
            ProofResult::Internal(r) => &r.context,
            ProofResult::Evm(r) => &r.context,
            ProofResult::Error(r) => &r.context,
        }
    }
}

impl std::fmt::Display for ProofResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofResult::ExecuteE2(r) => {
                write!(
                    f,
                    "ExecuteE2Result(num_segments: {}, cost: {})",
                    r.state.num_segments, r.state.cost,
                )
            }
            ProofResult::App(r) => {
                write!(
                    f,
                    "AppProof(segment_idx: {}, has_proof: {})",
                    r.state.segment_idx,
                    r.state.proof.is_some()
                )
            }
            ProofResult::Leaf(r) => {
                write!(
                    f,
                    "LeafProof(segments: [{}-{}], has_proof: {})",
                    r.state.segment_start,
                    r.state.segment_end,
                    r.state.proof.is_some()
                )
            }
            ProofResult::Internal(r) => {
                write!(
                    f,
                    "InternalProof(layer_idx: {}, segments: [{}-{}], has_proof: {})",
                    r.state.layer_idx,
                    r.state.segment_start,
                    r.state.segment_end,
                    r.state.proof.is_some()
                )
            }
            ProofResult::Evm(r) => {
                write!(f, "EvmProof(has_proof: {})", r.state.proof.is_some())
            }
            ProofResult::Error(r) => {
                write!(
                    f,
                    "ErrorResult(proof_uuid: {}, step: {}, error: {})",
                    r.context.proof_uuid, r.step, r.error
                )
            }
        }
    }
}

/// Execute E2 result containing segment information.
#[derive(Clone, Serialize, Deserialize)]
pub struct ExecuteE2Result {
    pub context: ProofContext,
    pub state: ExecuteE2State,
}

/// State from E2 execution.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ExecuteE2State {
    pub num_segments: usize,
    /// Each entry is a bincode-encoded `proof::Segment`.
    pub segments: Vec<SegmentBytes>,
    pub cost: u64,
    /// Wall-clock time for metered execution (segment discovery), in milliseconds.
    #[serde(default)]
    pub execute_time_ms: u64,
}

/// App proof for a single segment.
#[derive(Clone, Serialize, Deserialize)]
pub struct AppProof {
    pub context: ProofContext,
    pub state: AppProofState,
}

/// App proof state.
#[derive(Clone, Serialize, Deserialize)]
pub struct AppProofState {
    /// Bincode-encoded `proof::ProofWithPublicValue<F>` when present.
    pub proof: Option<ProofBytes>,
    pub segment_idx: usize,
    /// Wall-clock time for this segment's app proof, in milliseconds.
    #[serde(default)]
    pub prove_time_ms: u64,
    /// Time spent fast-forwarding from snapshot to segment start, in milliseconds.
    #[serde(default)]
    pub fastfwd_time_ms: u64,
    /// Time spent generating the STARK proof, in milliseconds.
    #[serde(default)]
    pub stark_prove_time_ms: u64,
    /// Time the segment sat in the channel between the executor and the prover, in milliseconds.
    #[serde(default)]
    pub queue_wait_ms: u64,
    /// Time of the executor of this worker between the end of the previous send of this worker and
    /// this send, in milliseconds. It covers every segment executed since that send.
    #[serde(default)]
    pub metered_time_ms: u64,
    /// STARK sub-step timings captured from tracing spans (e.g., trace_gen_time_ms).
    #[serde(default)]
    pub sub_metrics: HashMap<String, f64>,
    /// Terminal-segment-only: depth-independent `(DEFERRAL_AS, 0)`
    /// authentication path extracted from the FINAL memory merkle tree.
    /// Encoded via `proof::encode_deferral_auth_path` (length-prefixed
    /// digest slice, `overall_height()` entries). The path is "depth-0":
    /// the tail worker zero-pads its first `depth` entries once `depth`
    /// is known from `DeferralPvs`. `None` on non-deferral jobs and on
    /// every non-terminal segment; opaque on the wire.
    #[serde(default)]
    pub final_merkle_path_bytes: Option<Vec<u8>>,
    /// Terminal-segment-only: the COMPLETE, already-finalized
    /// `DeferralMerkleProofs` (depth 0) for a proof that made no deferred
    /// calls while running on a deferral deployment. A deferral-configured
    /// verifying key demands `DeferralMerkleProofs` on every proof, even
    /// one with an empty accumulator, so the terminal app worker — the only
    /// stage holding both the exe and the final memory when there is no tail
    /// worker — builds it here. Encoded via `DeferralMerkleProofs::encode`.
    /// Mutually exclusive with `final_merkle_path_bytes`: set only when the
    /// deployment is deferral-enabled AND this proof has no deferral input.
    /// `None` otherwise; opaque on the wire.
    #[serde(default)]
    pub deferral_merkle_proofs_bytes: Option<Vec<u8>>,
    /// Worker that produced the result, set by the manager on receipt.
    #[serde(default)]
    pub worker_id: usize,
    /// Manager clock at receipt, in milliseconds since the epoch.
    #[serde(default)]
    pub completed_at_ms: u64,
}

/// Leaf proof aggregating multiple app proofs.
#[derive(Clone, Serialize, Deserialize)]
pub struct LeafProof {
    pub context: ProofContext,
    pub state: LeafProofState,
}

/// Leaf proof state.
#[derive(Clone, Serialize, Deserialize)]
pub struct LeafProofState {
    /// Bincode-encoded `proof::ProofWithPublicValue<F>` when present.
    pub proof: Option<ProofBytes>,
    pub segment_start: usize,
    /// Inclusive end segment index
    pub segment_end: usize,
    /// Wall-clock time for leaf aggregation, in milliseconds.
    #[serde(default)]
    pub prove_time_ms: u64,
    /// STARK sub-step timings captured from tracing spans.
    #[serde(default)]
    pub sub_metrics: HashMap<String, f64>,
    /// Worker that produced the result, set by the manager on receipt.
    #[serde(default)]
    pub worker_id: usize,
    /// Manager clock at receipt, in milliseconds since the epoch.
    #[serde(default)]
    pub completed_at_ms: u64,
}

/// Internal proof aggregating leaf or other internal proofs.
#[derive(Clone, Serialize, Deserialize)]
pub struct InternalProof {
    pub context: ProofContext,
    pub state: InternalProofState,
}

/// Internal proof state.
#[derive(Clone, Serialize, Deserialize)]
pub struct InternalProofState {
    /// Bincode-encoded `proof::ProofWithPublicValue<F>` when present.
    pub proof: Option<ProofBytes>,
    /// Layer index (0 = bottom layer aggregating leaf proofs)
    pub layer_idx: usize,
    pub segment_start: usize,
    /// Inclusive end segment index
    pub segment_end: usize,
    /// Wall-clock time for internal aggregation, in milliseconds.
    #[serde(default)]
    pub prove_time_ms: u64,
    /// Compression time (non-zero only for is_final_proof), in milliseconds.
    #[serde(default)]
    pub compression_time_ms: u64,
    /// STARK sub-step timings captured from tracing spans.
    #[serde(default)]
    pub sub_metrics: HashMap<String, f64>,
    /// Span durations of the final proof wrap, empty when no wrap ran on this task.
    #[serde(default)]
    pub wrap_sub_metrics: HashMap<String, f64>,
    /// Deferral-only, final-internal-only: the `DeferralMerkleProofs` for the
    /// merged final internal proof, encoded via
    /// `verify_stark::deferral::DeferralMerkleProofs::encode` (the stark-backend
    /// digest codec; opaque here).
    ///
    /// A `proof_type=stark` deferral job's completion artifact is the merged
    /// internal proof, which only verifies together with these merkle proofs.
    /// The tail worker attaches them here when it merges the deferral tail for
    /// a stark job. `None` on non-deferral jobs, on every non-final internal
    /// proof, and on `proof_type=evm` jobs (where the merkle proofs are
    /// consumed in-process by root prove and never shipped).
    #[serde(default)]
    pub deferral_merkle_proofs_bytes: Option<Vec<u8>>,
    /// Set on the **post-merge** final internal proof of an `Evm` proof to
    /// signal the manager to dispatch the
    /// [`Step::EvmProve`](super::Step::EvmProve) step (root → halo2). The
    /// `proof` bytes carried alongside are the finished inputs — never the raw
    /// unmerged internal, so the manager routes only on this ready-for-evm
    /// message. The manager dispatches `EvmProve` to any eligible
    /// `runs_evm_prove()` worker: a `Full` worker in the default deployment
    /// (possibly the same one that produced the internal), or the
    /// `EvmDedicated` worker in dedicated-halo2 mode.
    ///
    /// `false` on every non-final internal proof and on `Stark` jobs; a
    /// `false` value means no `EvmProve` is dispatched.
    #[serde(default)]
    pub ready_for_evm: bool,
    /// Worker that produced the result, set by the manager on receipt.
    #[serde(default)]
    pub worker_id: usize,
    /// Manager clock at receipt, in milliseconds since the epoch.
    #[serde(default)]
    pub completed_at_ms: u64,
}

/// Root proof state — the worker-internal payload of the root prove stage.
///
/// Root proofs are a transient intermediate consumed by the worker's halo2
/// stage and are never posted to the manager, so there is no `ProofResult`
/// variant for them; this state travels only on the worker's local root
/// channel (see `edge-worker`'s `run_evm_prove`).
#[derive(Clone, Serialize, Deserialize)]
pub struct RootProofState {
    /// Bincode-encoded `proof::RootProof` when present.
    pub proof: Option<ProofBytes>,
    /// Wall-clock time for root proving, in milliseconds.
    #[serde(default)]
    pub prove_time_ms: u64,
    /// Root prover sub-step timings captured from tracing spans.
    #[serde(default)]
    pub sub_metrics: HashMap<String, f64>,
}

/// EVM proof wrapping the root proof for on-chain verification.
#[derive(Clone, Serialize, Deserialize)]
pub struct EvmProof {
    pub context: ProofContext,
    pub state: EvmProofState,
}

/// EVM proof state.
#[derive(Clone, Serialize, Deserialize)]
pub struct EvmProofState {
    /// Bincode-encoded `proof::EvmProof` when present.
    pub proof: Option<ProofBytes>,
    /// Wall-clock time for Halo2 proving, in milliseconds.
    #[serde(default)]
    pub prove_time_ms: u64,
    /// Wall-clock time for root proving, in milliseconds. Root is the EVM
    /// tail's first stage; its timing is folded into this (the only reported)
    /// result, since the root proof itself is a worker-internal intermediate
    /// and is not reported on its own.
    #[serde(default)]
    pub root_prove_time_ms: u64,
    /// Root + Halo2 prover sub-step timings captured from tracing spans,
    /// prefixed `root_` / `halo2_` to distinguish the two stages.
    #[serde(default)]
    pub sub_metrics: HashMap<String, f64>,
}

/// Error result when a proving step fails.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrorResult {
    pub context: ProofContext,
    pub step: String,
    pub error: String,
}

/// Payload sent from worker to manager's result endpoint.
#[derive(Serialize, Deserialize, Clone)]
pub struct ResultPayload {
    pub worker_id: usize,
    pub proof_uuid: String,
    pub result: MessageEnvelope<ProofResult>,
}

impl std::fmt::Debug for ResultPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResultPayload")
            .field("worker_id", &self.worker_id)
            .field("proof_uuid", &self.proof_uuid)
            .field("result_kind", &self.result.message.kind())
            .finish()
    }
}
