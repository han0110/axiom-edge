//! Central data definitions and read-only projections: `ProofState`,
//! `ProofStatus`, `InternalProofIndex`, `LightweightProofState`.
//!
//! Mutating behavior lives in sibling modules (`result_handler`,
//! `persistence`, `metrics_report`); the pure recursion-tree math lives in
//! `recursion`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Notify;

use proof::{ProofWithPublicValue, F};
use protocol::{
    AppProofState, ErrorResult, EvmProofState, ExecuteE2State, InternalProofState, LeafProofState,
    ProofContext, ProofType,
};

use super::recursion::num_internal_layers_for_leaf_count;

fn new_completion_notifier() -> Arc<Notify> {
    Arc::new(Notify::new())
}

fn default_leaf_arity() -> usize {
    4
}

fn default_internal_arity() -> usize {
    3
}

fn default_timeout_secs() -> u64 {
    300
}

/// Index for internal proofs in the recursion tree.
#[derive(Copy, Clone, Hash, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct InternalProofIndex {
    pub layer_idx: usize,
    pub idx: usize,
}

impl fmt::Display for InternalProofIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.layer_idx, self.idx)
    }
}

impl FromStr for InternalProofIndex {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (layer_str, idx_str) = s.split_once(':').ok_or("expected format 'layer:idx'")?;
        Ok(Self {
            layer_idx: layer_str.parse().map_err(|_| "invalid layer idx")?,
            idx: idx_str.parse().map_err(|_| "invalid idx")?,
        })
    }
}

/// Proof status.
///
/// `Failing` is a transient state: the proof has hit a failure condition
/// (worker error, timeout, partial-dispatch abort, etc.) but workers may
/// still be physically running shards for it. Manager treats `Failing` as
/// terminal for result *accumulation* (the accumulator drops further
/// results) but keeps the proof in `proof_states` so workers can drain:
/// late results still decrement the scheduler's per-worker busy slots
/// (`EdgeStateStore::worker_drained`, which never dispatches new work).
/// Once every worker's slot has drained — or the drain TTL fires as a
/// backstop for a worker that never reports — the proof transitions
/// `Failing` → `Failed` and is removed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ProofStatus {
    InProgress,
    Completed,
    /// Failure has been declared but workers may still be draining. Holds
    /// the human-readable reason that will surface in the final `Failed`.
    Failing(String),
    Failed(String),
    Canceled,
}

impl ProofStatus {
    pub fn as_str(&self) -> &str {
        match self {
            ProofStatus::InProgress => "in_progress",
            ProofStatus::Completed => "completed",
            ProofStatus::Failing(_) => "failing",
            ProofStatus::Failed(_) => "failed",
            ProofStatus::Canceled => "canceled",
        }
    }

    /// Whether this is the proof's last status. `Failing` is not, since it
    /// still becomes `Failed` once the workers drain.
    pub fn is_settled(&self) -> bool {
        matches!(
            self,
            ProofStatus::Completed | ProofStatus::Failed(_) | ProofStatus::Canceled
        )
    }
}

/// Complete state of an Edge proof, including the recursion tree.
///
/// This struct is the central data type the manager mutates as worker
/// results arrive. The behavior is split across sibling modules:
/// - [`super::result_handler`] handles incoming results and emits follow-up requests
/// - [`super::recursion`] computes recursion-tree shape (pure)
/// - [`super::persistence`] writes completed proofs / failure snapshots to disk
/// - [`super::metrics_report`] generates the per-proof report and OTel metrics
#[serde_as]
#[derive(Serialize, Deserialize, Clone)]
pub struct ProofState {
    pub context: ProofContext,
    pub status: ProofStatus,
    pub cost_limit: u64,
    pub num_workers: usize,
    pub num_segments: Option<usize>,
    pub num_instructions: Option<u64>,

    /// Leaf-circuit fan-in: number of app proofs aggregated into one leaf
    /// proof (from manager config).
    #[serde(default = "default_leaf_arity")]
    pub leaf_arity: usize,

    /// Internal-circuit fan-in: number of child proofs aggregated into one
    /// internal proof (from manager config).
    #[serde(default = "default_internal_arity")]
    pub internal_arity: usize,

    /// Watchdog deadline for this proof (in seconds, wall-clock from
    /// `proof_start_time`). Resolved at `start_proof` from request override
    /// > manager config; stored per-proof so the watchdog can honor
    /// per-request overrides.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// Original execution info for debugging
    pub execute_e2_state: Option<ExecuteE2State>,

    /// Timing information
    pub proof_start_time: DateTime<Utc>,
    pub e2e_latency_ms: Option<u64>,
    /// Wall-clock instant at which the manager finished uploading the input to
    /// all workers and dispatched the proving work. Anchors `proving_latency_ms`
    /// so it excludes the `/start_proof` admission + input read + worker upload
    /// fan-out (the "submitting proof" overhead) and measures only distributed
    /// proving wall-clock.
    pub proving_started_at: Option<DateTime<Utc>>,
    pub proving_latency_ms: Option<u64>,

    /// Wall-clock phase timestamps (set when first/last result of each phase arrives)
    pub app_prove_started_at: Option<DateTime<Utc>>,
    pub app_prove_ended_at: Option<DateTime<Utc>>,
    pub leaf_prove_started_at: Option<DateTime<Utc>>,
    pub leaf_prove_ended_at: Option<DateTime<Utc>>,
    pub internal_prove_started_at: Option<DateTime<Utc>>,
    pub internal_prove_ended_at: Option<DateTime<Utc>>,

    /// Recursion tree
    pub app_proofs: HashMap<usize, AppProofState>,
    pub leaf_proofs: HashMap<usize, LeafProofState>,
    #[serde_as(as = "HashMap<DisplayFromStr, _>")]
    pub internal_proofs: HashMap<InternalProofIndex, InternalProofState>,

    /// EVM (halo2-wrapped) proof artifact. Presence of this field is what
    /// flips an `Evm`-typed proof to `Completed` (see [`Self::is_completed`]).
    #[serde(default)]
    pub evm_proof: Option<EvmProofState>,

    /// Timestamp for cleanup
    pub last_updated: DateTime<Utc>,

    /// Completion notifier (kept private — accessed only via accessor methods).
    #[serde(skip, default = "new_completion_notifier")]
    completion_notifier: Arc<Notify>,

    /// Optional on-disk path where the completed final proof was persisted.
    pub persisted_final_proof_path: Option<String>,

    /// Optional on-disk path where app proofs were snapshotted after a leaf
    /// logup nonzero-root-sum failure.
    pub persisted_leaf_failure_app_proofs_path: Option<String>,

    /// Most recent error result, retained for failure handling and used by
    /// [`super::persistence`] to decide whether to snapshot app proofs.
    #[serde(skip, default)]
    pub(super) last_error_result: Option<ErrorResult>,

    /// Whether [`super::result_handler`] has already emitted the
    /// "dropping late results" notice for this proof.
    /// A "late result" is a worker result that arrives for a proof that has
    /// already reached a terminal state (i.e. Failed). It can happen if worker
    /// 1 reports failure, but other workers continue to prove and send results.
    #[serde(skip, default)]
    pub(super) late_result_notice_emitted: bool,

    /// Number of deferral circuits for this proof, inferred at `start_proof`
    /// from the deferral artifacts the caller staged on the manager (`0` =
    /// non-deferral). Drives whether the manager attaches a `DeferralTailDispatch`
    /// to the final `InternalProveRequest`; the tail worker reconstructs the
    /// per-circuit `DeferralInput` paths itself from the deterministic staging
    /// convention + its loaded keyset, so no paths cross the wire.
    #[serde(default)]
    pub deferral_circuit_count: usize,

    /// Manager-buffered depth-independent `(DEFERRAL_AS, 0)`
    /// authentication path from the FINAL memory merkle tree, captured
    /// on the terminal app worker (`sharded_app_prover.rs`) and shipped
    /// in the terminal `AppProof.state.final_merkle_path_bytes`. Forwarded
    /// to the tail worker in `DeferralTailDispatch.final_merkle_path_bytes`
    /// so it can finalize with `depth` and attach
    /// `DeferralMerkleProofs` before root. `None` on non-deferral jobs and
    /// until the terminal `AppProof` arrives (the dispatch helper handles
    /// the absent-yet case explicitly).
    #[serde(default)]
    pub deferral_final_merkle_path_bytes: Option<Vec<u8>>,

    /// Manager-buffered COMPLETE depth-0 `DeferralMerkleProofs` for a proof
    /// that made no deferred calls while running on a deferral deployment.
    /// The terminal app worker builds it (there is no tail worker to finalize
    /// it) and ships it in `AppProof.state.deferral_merkle_proofs_bytes`. A
    /// deferral verifying key rejects a proof carrying no merkle proofs, so
    /// this is attached to the terminal `VmStarkProof`: for `proof_type=stark`
    /// at persist time (onto the final `InternalProofState`), and for
    /// `proof_type=evm` forwarded to the worker's `run_evm_prove` on the final
    /// `InternalProveRequest`. `None` on real deferral jobs (the tail worker
    /// produces the merkle proofs) and on non-deferral deployments.
    #[serde(default)]
    pub deferral_depth0_merkle_proofs_bytes: Option<Vec<u8>>,
}

impl ProofState {
    /// Create a new proof state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: ProofContext,
        cost_limit: u64,
        num_workers: usize,
        leaf_arity: usize,
        internal_arity: usize,
        timeout_secs: u64,
    ) -> Self {
        let now = Utc::now();
        Self {
            context,
            status: ProofStatus::InProgress,
            cost_limit,
            num_workers,
            num_segments: None,
            num_instructions: None,
            leaf_arity,
            internal_arity,
            timeout_secs,
            execute_e2_state: None,
            proof_start_time: now,
            e2e_latency_ms: None,
            proving_started_at: None,
            proving_latency_ms: None,
            app_prove_started_at: None,
            app_prove_ended_at: None,
            leaf_prove_started_at: None,
            leaf_prove_ended_at: None,
            internal_prove_started_at: None,
            internal_prove_ended_at: None,
            app_proofs: HashMap::new(),
            leaf_proofs: HashMap::new(),
            internal_proofs: HashMap::new(),
            evm_proof: None,
            last_updated: now,
            completion_notifier: new_completion_notifier(),
            persisted_final_proof_path: None,
            persisted_leaf_failure_app_proofs_path: None,
            last_error_result: None,
            late_result_notice_emitted: false,
            deferral_circuit_count: 0,
            deferral_final_merkle_path_bytes: None,
            deferral_depth0_merkle_proofs_bytes: None,
        }
    }

    /// Get the completion notifier.
    pub fn completion_notifier(&self) -> Arc<Notify> {
        self.completion_notifier.clone()
    }

    /// Notify waiters that the proof is complete.
    pub fn notify_completion(&self) {
        self.completion_notifier.notify_waiters();
    }

    /// Whether the proof has reached a terminal state from the accumulator's
    /// perspective — incoming results are dropped. Includes `Failing`, which
    /// is awaiting worker drain but no longer accepts recursion-tree updates.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            ProofStatus::Completed
                | ProofStatus::Failing(_)
                | ProofStatus::Failed(_)
                | ProofStatus::Canceled
        )
    }

    /// Whether the watchdog should mark this proof as timed out: still
    /// in-progress and past its `timeout_secs` deadline.
    pub fn is_timed_out(&self, now: DateTime<Utc>) -> bool {
        if self.is_terminal() {
            return false;
        }
        let elapsed = (now - self.proof_start_time).num_seconds();
        elapsed > 0 && (elapsed as u64) > self.timeout_secs
    }

    /// Mark this proof as timed out. Sets `Failing` (not terminal yet —
    /// awaits worker drain). Records `last_error_result` so failure-snapshot
    /// persistence still gates correctly. Idempotent on terminal states.
    pub fn mark_timed_out(&mut self) {
        if self.is_terminal() {
            return;
        }
        let reason = format!("timed out after {}s", self.timeout_secs);
        let error = ErrorResult {
            context: self.context.clone(),
            step: "timeout".to_string(),
            error: reason.clone(),
        };
        self.last_error_result = Some(error);
        self.status = ProofStatus::Failing(reason);
        self.notify_completion();
    }

    /// Mark this proof as failing with the given reason. Transient state —
    /// the manager will transition to `Failed` once workers drain (or TTL).
    /// No-op on already-terminal proofs.
    pub fn mark_failing(&mut self, reason: String) {
        if self.is_terminal() {
            return;
        }
        let error = ErrorResult {
            context: self.context.clone(),
            step: "failing".to_string(),
            error: reason.clone(),
        };
        self.last_error_result = Some(error);
        self.status = ProofStatus::Failing(reason);
        self.notify_completion();
    }

    /// Transition from `Failing` to terminal `Failed`. Called by the
    /// drain orchestrator once all workers have reported completion (or
    /// when the drain TTL has expired). No-op if not in `Failing`.
    pub fn transition_failing_to_failed(&mut self) {
        if let ProofStatus::Failing(reason) = &self.status {
            self.status = ProofStatus::Failed(reason.clone());
            // notify_completion was already called when entering Failing.
        }
    }

    /// Whether the proof has reached its final artifact.
    ///
    /// Branches on `context.proof_type`:
    /// - `Stark` ⇒ the final internal proof is present in the recursion tree.
    /// - `Evm`   ⇒ the worker's in-process EVM prove has posted the final
    ///   `Evm` artifact (the root proof on its own is only an intermediate
    ///   step, so the final-internal arrival is no longer "done").
    pub fn is_completed(&self) -> bool {
        match self.context.proof_type {
            ProofType::Stark => self.final_internal_proof_present(),
            ProofType::Evm => self
                .evm_proof
                .as_ref()
                .and_then(|s| s.proof.as_ref())
                .is_some(),
        }
    }

    /// Whether the final-internal slot in the recursion tree is occupied.
    /// Shared helper between `is_completed` (for Stark) and `get_stark_proof`.
    fn final_internal_proof_present(&self) -> bool {
        if let Some(num_segments) = self.num_segments {
            let num_leaf_proofs = num_segments.div_ceil(self.leaf_arity);
            let num_internal_layers =
                num_internal_layers_for_leaf_count(num_leaf_proofs, self.internal_arity);
            let effective_final_layer = num_internal_layers.max(2) - 1;
            return self.internal_proofs.contains_key(&InternalProofIndex {
                layer_idx: effective_final_layer,
                idx: 0,
            });
        }
        false
    }

    /// The final internal proof's state — the completion artifact for a
    /// `proof_type=Stark` proof — if the proof has completed and the slot is
    /// occupied. Shared by [`get_stark_proof`](Self::get_stark_proof) (which
    /// decodes the proof bytes) and persistence, which also needs the
    /// deferral merkle-proof bytes to build the on-disk `VmStarkProof`.
    pub fn final_internal_state(&self) -> Option<&InternalProofState> {
        if !matches!(self.status, ProofStatus::Completed) {
            return None;
        }

        let num_segments = self.num_segments?;
        let num_leaf_proofs = num_segments.div_ceil(self.leaf_arity);
        let num_internal_layers =
            num_internal_layers_for_leaf_count(num_leaf_proofs, self.internal_arity);
        let effective_final_layer = num_internal_layers.max(2) - 1;
        self.internal_proofs.get(&InternalProofIndex {
            layer_idx: effective_final_layer,
            idx: 0,
        })
    }

    /// Get the final STARK proof if completed (decoded from wire bytes).
    pub fn get_stark_proof(&self) -> Option<ProofWithPublicValue<F>> {
        let bytes = self.final_internal_state()?.proof.as_ref()?;
        proof::decode_proof(bytes).ok()
    }

    /// Get the final EVM (halo2-wrapped) proof bytes if this is an Evm proof
    /// that has reached [`ProofStatus::Completed`]. Returns the opaque
    /// bincode-encoded wire bytes — consumers decode via `proof::decode_evm_proof`
    /// (kept undecoded here so the manager doesn't pull halo2 deps).
    pub fn get_evm_proof(&self) -> Option<Vec<u8>> {
        if !matches!(self.status, ProofStatus::Completed) {
            return None;
        }
        self.evm_proof.as_ref()?.proof.clone()
    }

    /// Drop heavy proof payloads once the proof has finished.
    ///
    /// The manager only needs timing/sub-metric summaries for post-completion
    /// queries and report generation. Retaining all proof bytes for every
    /// recently completed proof causes avoidable heap growth during long
    /// sequential benchmark runs.
    pub fn compact_completed_state(&mut self) {
        // Only compact once the proof has truly drained — workers may still
        // be running for a `Failing` proof, so we keep proof bytes around in
        // case any late-result accounting needs them.
        if !matches!(
            self.status,
            ProofStatus::Completed | ProofStatus::Failed(_) | ProofStatus::Canceled
        ) {
            return;
        }

        for app_state in self.app_proofs.values_mut() {
            app_state.proof = None;
        }

        for leaf_state in self.leaf_proofs.values_mut() {
            leaf_state.proof = None;
        }

        for internal_state in self.internal_proofs.values_mut() {
            internal_state.proof = None;
        }

        if let Some(execute_e2_state) = &mut self.execute_e2_state {
            execute_e2_state.segments.clear();
            execute_e2_state.segments.shrink_to_fit();
        }
    }

    /// Create a lightweight version for API responses.
    pub fn to_lightweight_state(&self) -> LightweightProofState {
        let execute_time_ms = self.execute_e2_state.as_ref().map(|s| s.execute_time_ms);
        let total_app_prove_ms = if self.app_proofs.is_empty() {
            None
        } else {
            Some(self.app_proofs.values().map(|p| p.prove_time_ms).sum())
        };
        let total_leaf_prove_ms = if self.leaf_proofs.is_empty() {
            None
        } else {
            Some(self.leaf_proofs.values().map(|p| p.prove_time_ms).sum())
        };
        let (total_internal_prove_ms, compression_time_ms) = if self.internal_proofs.is_empty() {
            (None, None)
        } else {
            (
                Some(self.internal_proofs.values().map(|p| p.prove_time_ms).sum()),
                Some(
                    self.internal_proofs
                        .values()
                        .map(|p| p.compression_time_ms)
                        .sum(),
                ),
            )
        };

        LightweightProofState {
            proof_uuid: self.context.proof_uuid.clone(),
            program: self.context.program.clone(),
            status: self.status.clone(),
            num_segments: self.num_segments,
            num_instructions: self.num_instructions,
            proof_start_time: self.proof_start_time,
            e2e_latency_ms: self.e2e_latency_ms,
            proving_latency_ms: self.proving_latency_ms,
            app_proofs_count: self.app_proofs.len(),
            leaf_proofs_count: self.leaf_proofs.len(),
            internal_proofs_count: self.internal_proofs.len(),
            last_updated: self.last_updated,
            execute_time_ms,
            total_app_prove_ms,
            total_leaf_prove_ms,
            total_internal_prove_ms,
            compression_time_ms,
            // EVM tail: `prove_time_ms` is halo2, `root_prove_time_ms` is root.
            total_root_prove_ms: self.evm_proof.as_ref().map(|e| e.root_prove_time_ms),
            total_halo2_prove_ms: self.evm_proof.as_ref().map(|e| e.prove_time_ms),
            app_prove_started_at: self.app_prove_started_at,
            app_prove_ended_at: self.app_prove_ended_at,
            leaf_prove_started_at: self.leaf_prove_started_at,
            leaf_prove_ended_at: self.leaf_prove_ended_at,
            internal_prove_started_at: self.internal_prove_started_at,
            internal_prove_ended_at: self.internal_prove_ended_at,
            persisted_final_proof_path: self.persisted_final_proof_path.clone(),
            persisted_leaf_failure_app_proofs_path: self
                .persisted_leaf_failure_app_proofs_path
                .clone(),
        }
    }

    /// Create the task timeline for API responses. The view carries no proof
    /// bytes and no EVM proof. It also holds no execute record.
    pub fn to_pipeline(&self) -> ProofPipeline {
        let mut app_proofs: Vec<TaskTiming> = self
            .app_proofs
            .values()
            .map(|s| TaskTiming {
                worker_id: s.worker_id,
                completed_at_ms: s.completed_at_ms,
                segment_start: s.segment_idx,
                segment_end: s.segment_idx,
                queue_wait_ms: s.queue_wait_ms,
                metered_time_ms: s.metered_time_ms,
                prove_time_ms: s.prove_time_ms,
                fastfwd_time_ms: s.fastfwd_time_ms,
                stark_prove_time_ms: s.stark_prove_time_ms,
                sub_metrics: s.sub_metrics.clone(),
                ..Default::default()
            })
            .collect();

        let mut leaf_proofs: Vec<TaskTiming> = self
            .leaf_proofs
            .values()
            .map(|s| TaskTiming {
                worker_id: s.worker_id,
                completed_at_ms: s.completed_at_ms,
                segment_start: s.segment_start,
                segment_end: s.segment_end,
                prove_time_ms: s.prove_time_ms,
                sub_metrics: s.sub_metrics.clone(),
                ..Default::default()
            })
            .collect();

        let mut internal_proofs: Vec<TaskTiming> = self
            .internal_proofs
            .values()
            .map(|s| TaskTiming {
                worker_id: s.worker_id,
                completed_at_ms: s.completed_at_ms,
                segment_start: s.segment_start,
                segment_end: s.segment_end,
                layer_idx: Some(s.layer_idx),
                prove_time_ms: s.prove_time_ms,
                compression_time_ms: s.compression_time_ms,
                sub_metrics: s.sub_metrics.clone(),
                wrap_sub_metrics: s.wrap_sub_metrics.clone(),
                ..Default::default()
            })
            .collect();

        for tasks in [&mut app_proofs, &mut leaf_proofs, &mut internal_proofs] {
            tasks.sort_by_key(|t| (t.segment_start, t.layer_idx));
        }

        ProofPipeline {
            proof_start_time: self.proof_start_time,
            app_proofs,
            leaf_proofs,
            internal_proofs,
        }
    }

    /// Check if the proof should be evicted based on age.
    pub fn should_evict(&self, now: DateTime<Utc>) -> bool {
        let age = now - self.last_updated;
        match &self.status {
            ProofStatus::InProgress => age > chrono::Duration::hours(10),
            // Failing is transient — drain orchestrator transitions it to
            // Failed (or TTL fires). Don't evict directly from here.
            ProofStatus::Failing(_) => false,
            ProofStatus::Completed | ProofStatus::Failed(_) | ProofStatus::Canceled => {
                age > chrono::Duration::minutes(5)
            }
        }
    }
}

/// Lightweight proof state for API responses.
#[derive(Debug, Serialize, Clone)]
pub struct LightweightProofState {
    pub proof_uuid: String,
    pub program: protocol::ProgramRef,
    pub status: ProofStatus,
    pub num_segments: Option<usize>,
    pub num_instructions: Option<u64>,
    pub proof_start_time: DateTime<Utc>,
    pub e2e_latency_ms: Option<u64>,
    /// E2E latency excluding the `/start_proof` submission overhead (input read
    /// + worker upload fan-out). Wall-clock from work dispatch to completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proving_latency_ms: Option<u64>,
    pub app_proofs_count: usize,
    pub leaf_proofs_count: usize,
    pub internal_proofs_count: usize,
    pub last_updated: DateTime<Utc>,
    /// Aggregated timing from worker proofs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execute_time_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_app_prove_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_leaf_prove_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_internal_prove_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression_time_ms: Option<u64>,
    /// EVM-tail timing (`proof_type=evm` only): root prove and halo2 prove, in
    /// ms. Their sum is the time added on top of the STARK recursion to produce
    /// the EVM proof. Both `None` for stark proofs (no EVM tail runs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_root_prove_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_halo2_prove_ms: Option<u64>,
    /// Wallclock timestamps for each proving phase
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_prove_started_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_prove_ended_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leaf_prove_started_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leaf_prove_ended_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub internal_prove_started_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub internal_prove_ended_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persisted_final_proof_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persisted_leaf_failure_app_proofs_path: Option<String>,
}

/// Task timeline of one proof, for API responses.
#[derive(Debug, Serialize, Clone)]
pub struct ProofPipeline {
    pub proof_start_time: DateTime<Utc>,
    pub app_proofs: Vec<TaskTiming>,
    pub leaf_proofs: Vec<TaskTiming>,
    pub internal_proofs: Vec<TaskTiming>,
}

/// One task of the timeline, stamped by the manager on receipt.
#[derive(Debug, Default, Serialize, Clone)]
pub struct TaskTiming {
    pub worker_id: usize,
    pub completed_at_ms: u64,
    pub segment_start: usize,
    pub segment_end: usize,
    /// Internal tasks only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_idx: Option<usize>,
    pub queue_wait_ms: u64,
    pub metered_time_ms: u64,
    pub prove_time_ms: u64,
    pub fastfwd_time_ms: u64,
    pub stark_prove_time_ms: u64,
    pub compression_time_ms: u64,
    pub sub_metrics: HashMap<String, f64>,
    pub wrap_sub_metrics: HashMap<String, f64>,
}

#[cfg(all(test, feature = "mock-provers"))]
mod tests {
    use super::*;
    use protocol::{
        current_timestamp, AppProof, LeafProof, MessageEnvelope, ProofContext, ProofResult,
    };

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
    fn test_new_proof_state() {
        let context = make_context();
        let state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);

        assert_eq!(state.context.proof_uuid, "test-proof");
        assert!(matches!(state.status, ProofStatus::InProgress));
        assert_eq!(state.num_segments, None);
        assert!(!state.is_completed());
    }

    #[test]
    fn test_lightweight_state() {
        let context = make_context();
        let state = ProofState::new(context, 1_000_000, 4, 4, 3, 300);
        let lightweight = state.to_lightweight_state();

        assert_eq!(lightweight.proof_uuid, "test-proof");
        assert_eq!(lightweight.app_proofs_count, 0);
    }

    #[test]
    fn test_is_timed_out_inside_window() {
        let mut state = ProofState::new(make_context(), 1_000_000, 4, 4, 3, 300);
        state.proof_start_time = Utc::now() - chrono::Duration::seconds(10);
        assert!(!state.is_timed_out(Utc::now()));
    }

    #[test]
    fn test_is_timed_out_outside_window() {
        let mut state = ProofState::new(make_context(), 1_000_000, 4, 4, 3, 300);
        state.proof_start_time = Utc::now() - chrono::Duration::seconds(301);
        assert!(state.is_timed_out(Utc::now()));
    }

    #[test]
    fn test_is_timed_out_skips_terminal() {
        let mut state = ProofState::new(make_context(), 1_000_000, 4, 4, 3, 300);
        state.proof_start_time = Utc::now() - chrono::Duration::seconds(9999);
        state.status = ProofStatus::Completed;
        assert!(!state.is_timed_out(Utc::now()));
    }

    #[test]
    fn test_mark_timed_out_transitions_state() {
        let mut state = ProofState::new(make_context(), 1_000_000, 4, 4, 3, 60);
        state.mark_timed_out();
        // Timeout moves to Failing (transient), not Failed. The drain
        // orchestrator transitions Failing → Failed once workers report.
        assert!(matches!(state.status, ProofStatus::Failing(ref m) if m.contains("timed out")));
        assert!(state.last_error_result.is_some());
        assert_eq!(state.last_error_result.as_ref().unwrap().step, "timeout");
    }

    #[test]
    fn test_transition_failing_to_failed() {
        let mut state = ProofState::new(make_context(), 1_000_000, 4, 4, 3, 60);
        state.mark_failing("worker died".to_string());
        assert!(matches!(state.status, ProofStatus::Failing(_)));
        state.transition_failing_to_failed();
        assert!(matches!(state.status, ProofStatus::Failed(ref m) if m == "worker died"));
    }

    #[test]
    fn test_transition_failing_to_failed_noop_on_other_states() {
        let mut state = ProofState::new(make_context(), 1_000_000, 4, 4, 3, 60);
        state.status = ProofStatus::Completed;
        state.transition_failing_to_failed();
        // Should remain Completed.
        assert!(matches!(state.status, ProofStatus::Completed));
    }

    #[test]
    fn test_mark_timed_out_idempotent_when_terminal() {
        let mut state = ProofState::new(make_context(), 1_000_000, 4, 4, 3, 60);
        state.status = ProofStatus::Completed;
        state.mark_timed_out();
        // Should not change status from Completed → Failed.
        assert!(matches!(state.status, ProofStatus::Completed));
        assert!(state.last_error_result.is_none());
    }

    #[test]
    fn test_pipeline_reports_worker_and_segments_per_task() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);
        state.num_segments = Some(4);

        let app_result = |worker_id, segment_idx| {
            let mut result = ProofResult::App(AppProof {
                context: context.clone(),
                state: AppProofState {
                    proof: Some(make_mock_proof()),
                    segment_idx,
                    prove_time_ms: 0,
                    fastfwd_time_ms: 0,
                    stark_prove_time_ms: 0,
                    queue_wait_ms: 12,
                    metered_time_ms: 34,
                    sub_metrics: HashMap::new(),
                    final_merkle_path_bytes: None,
                    deferral_merkle_proofs_bytes: None,
                    worker_id: 0,
                    completed_at_ms: 0,
                },
            });
            result.stamp(worker_id, current_timestamp());
            result
        };
        state
            .handle_proof_result_with_envelope(MessageEnvelope::with_metadata(app_result(3, 2)))
            .unwrap();
        state
            .handle_proof_result_with_envelope(MessageEnvelope::with_metadata(app_result(5, 0)))
            .unwrap();

        let mut leaf_result = ProofResult::Leaf(LeafProof {
            context: context.clone(),
            state: LeafProofState {
                proof: Some(make_mock_proof()),
                segment_start: 0,
                segment_end: 3,
                prove_time_ms: 0,
                sub_metrics: HashMap::new(),
                worker_id: 0,
                completed_at_ms: 0,
            },
        });
        leaf_result.stamp(7, current_timestamp());
        state
            .handle_proof_result_with_envelope(MessageEnvelope::with_metadata(leaf_result))
            .unwrap();

        state.internal_proofs.insert(
            InternalProofIndex {
                layer_idx: 1,
                idx: 0,
            },
            InternalProofState {
                proof: None,
                layer_idx: 1,
                segment_start: 0,
                segment_end: 3,
                prove_time_ms: 400,
                compression_time_ms: 250,
                sub_metrics: HashMap::new(),
                wrap_sub_metrics: HashMap::from([("trace_gen_time_ms".to_string(), 60.0)]),
                deferral_merkle_proofs_bytes: None,
                ready_for_evm: false,
                worker_id: 9,
                completed_at_ms: 77,
            },
        );

        let pipeline = state.to_pipeline();

        let app_segments: Vec<usize> = pipeline
            .app_proofs
            .iter()
            .map(|t| t.segment_start)
            .collect();
        assert_eq!(app_segments, vec![0, 2]);
        assert_eq!(
            (
                pipeline.app_proofs[0].worker_id,
                pipeline.app_proofs[1].worker_id
            ),
            (5, 3)
        );
        assert!(pipeline.app_proofs.iter().all(|t| t.completed_at_ms > 0));
        assert_eq!(
            (
                pipeline.app_proofs[0].queue_wait_ms,
                pipeline.app_proofs[0].metered_time_ms
            ),
            (12, 34)
        );

        let leaf = &pipeline.leaf_proofs[0];
        assert_eq!(
            (leaf.worker_id, leaf.segment_start, leaf.segment_end),
            (7, 0, 3)
        );
        assert!(leaf.completed_at_ms > 0);

        let internal = &pipeline.internal_proofs[0];
        assert_eq!(internal.layer_idx, Some(1));
        assert_eq!((internal.worker_id, internal.compression_time_ms), (9, 250));
        assert_eq!(
            internal.wrap_sub_metrics.get("trace_gen_time_ms"),
            Some(&60.0)
        );
    }
}
