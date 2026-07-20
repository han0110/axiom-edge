//! Request types for the manager ↔ worker protocol.
//!
//! Organized by transport direction:
//!
//! 1. **Client → Manager** — public API surface that external clients use to
//!    drive proof generation. Currently just [`StartProofRequest`].
//! 2. **Worker → Manager** — workers reporting in. Currently just
//!    [`RegisterWorkerRequest`]; result payloads (the other worker → manager
//!    direction) live in `results.rs` alongside the result types they wrap.
//! 3. **Manager → Worker** — work assignments the manager dispatches to
//!    workers. [`ShardedAppProveRequest`] is the proof-start kickoff;
//!    [`GeneralProveRequest`] (and its [`LeafProveRequest`] /
//!    [`InternalProveRequest`] variants) carries follow-up aggregation
//!    requests once app and leaf proofs become available.
//!
//! Proof and segment payloads are carried as opaque `Vec<u8>`
//! ([`ProofBytes`] / [`SegmentBytes`]) — bincode-encoded
//! `proof::ProofWithPublicValue<F>` / `proof::Segment`. Decoders that need
//! to inspect proof internals add the `proof` crate.

use serde::{Deserialize, Serialize};

use super::{ProgramRef, ProofContext, ProofType, Step, WithProofContext};

// =============================================================================
// Shared
// =============================================================================

/// Bincode-encoded proof payload (opaque on the wire).
///
/// Bytes are produced by `proof::encode_proof(&ProofWithPublicValue<F>)`
/// and decoded with `proof::decode_proof(bytes)`. Most consumers can treat
/// them as opaque; only consumers that verify or inspect proofs depend on
/// the `proof` crate.
pub type ProofBytes = Vec<u8>;

/// Bincode-encoded segment payload (opaque on the wire).
///
/// Bytes are produced by `proof::encode_segment(&Segment)` and decoded with
/// `proof::decode_segment(bytes)`. The manager treats segments opaquely;
/// only the worker that produced them needs typed access.
pub type SegmentBytes = Vec<u8>;

/// Request details extracted from any prove request.
pub struct RequestDetails {
    pub context: ProofContext,
    pub step: Step,
}

// =============================================================================
// Client → Manager
// =============================================================================
//
// Public API surface. External clients call these to drive proof generation.
// These are the only types most third-party integrators ever see.

/// Start a new proof.
///
/// The client-facing entry point: posted as JSON to the manager's
/// `/start_proof` endpoint to kick off proving for a given input.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StartProofRequest {
    pub proof_uuid: String,
    /// Target program.
    ///
    /// - When present, must be in the manager's loadout; otherwise the
    ///   request is rejected with 409.
    /// - When omitted, the manager resolves it to the loaded program
    ///   **iff exactly one program is loaded**. Any other loadout size
    ///   makes the field required and the request is rejected with 400.
    ///
    /// This makes single-program dev deployments terser without changing
    /// the wire format for multi-program prod.
    #[serde(default)]
    pub program: Option<ProgramRef>,
    /// Opaque, deployment-defined key/value labels carried with the proof.
    /// The edge never interprets these — they're forwarded in lifecycle
    /// webhook events and emitted as metric attributes for downstream
    /// integrations (a caller might set, for example,
    /// `{"block_number": "24000000"}` or `{"batch_id": "…"}`).
    #[serde(default)]
    pub labels: std::collections::BTreeMap<String, String>,
    /// Requested final proof artifact type. Defaults to STARK for existing clients.
    #[serde(default)]
    pub proof_type: ProofType,
    /// Selects how the proof input reached the workers — the two supported
    /// transports (see `docs/CONCEPTS.md`):
    ///
    /// - `false` (default) — **manager-staged (Flow 2).** The caller first
    ///   `POST`s the bincode `StdIn` bytes to the manager's
    ///   `/upload_input/{proof_uuid}` (and, for a deferral proof, each
    ///   `DeferralState`/`DeferralInput` to the manager), then calls
    ///   `/start_proof`. The manager fans the input out to every worker. This
    ///   is the only transport that supports deferral and multi-element input.
    /// - `true` — **worker pre-uploaded (Flow 1).** The caller pushed the
    ///   input directly to every worker (e.g. via `/upload_input_compact`)
    ///   before calling `/start_proof`; the manager skips fan-out. Deferral is
    ///   NOT supported on this path (stage deferral artifacts on the manager).
    ///
    /// Either way the worker reads its input from the deterministic staged path
    /// `/dev/shm/edge_{proof_uuid}/input.bin`; there is no caller-supplied path.
    ///
    /// Whether a proof is a *deferral* proof — and how many circuits — is not
    /// declared here: it is inferred by the manager from the `DeferralState` /
    /// `DeferralInput` artifacts the caller uploaded to it (one pair per circuit,
    /// at contiguous indices `0..N`), before `/start_proof`. The worker in turn
    /// validates that count against the deployment's loaded deferral keyset.
    #[serde(default)]
    pub input_already_uploaded: bool,
    /// Optional override for OPENVM_MAX_SEGMENT_MEMORY.
    #[serde(default)]
    pub segment_memory: Option<usize>,
    /// Optional override for leaf packing threshold.
    /// Set to a large number (e.g., 1000) to always pack leaf proofs onto busy workers.
    #[serde(default)]
    pub leaf_pack_threshold: Option<usize>,
    /// Optional override for the watchdog timeout (in seconds). Falls
    /// through to manager `[proof] timeout_secs` if unset. Useful for one-
    /// off oversized proofs without raising the global default.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

// Note: no `From<StartProofRequest> for ProofContext` impl — building a
// `ProofContext` requires a concrete `ProgramRef`, but `program` is now
// `Option`. Resolution against the manager's loadout happens server-side
// in `start_proof`, so callers there construct the context explicitly.

// =============================================================================
// Worker → Manager
// =============================================================================
//
// Internal to the deployment: workers reporting in to the manager.
// Result payloads (the other worker → manager direction) live in `results.rs`.

/// Deployment role a worker plays in the prover fleet.
///
/// Reported at registration so the manager learns each worker's role. `Full`
/// (the default) runs STARK proving (app + leaf + internal), plus the EVM
/// step (root + halo2) when the worker is built with EVM support (the
/// `evm-prove` feature); a stark-only build has no EVM step, so `Full` runs the
/// STARK proving only. `StarkOnly` and `EvmDedicated` are the opt-in dedicated-halo2
/// deployment mode: a `StarkOnly` worker runs only STARK proving (app + leaf +
/// internal), and the single `EvmDedicated` worker runs only the EVM step. The
/// serde representation is snake_case
/// (`full` / `stark_only` / `evm_dedicated`), matching the wire form used in both
/// `worker.toml` and this request.
#[derive(Clone, Copy, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerRole {
    /// app + leaf + internal, plus the EVM step (root + halo2) when built with
    /// EVM support (the `evm-prove` feature). Default.
    #[default]
    Full,
    /// app + leaf + internal only; no root/halo2 (dedicated-halo2 mode).
    StarkOnly,
    /// root + halo2 only (the EVM step); no app/leaf/internal (dedicated-halo2 mode).
    EvmDedicated,
}

impl WorkerRole {
    /// Whether a worker with this role runs STARK proving
    /// (app + leaf + internal). `true` for [`WorkerRole::Full`] and
    /// [`WorkerRole::StarkOnly`]; `false` for [`WorkerRole::EvmDedicated`].
    ///
    /// Drives two role-gating decisions: the manager's app-eligible (normal)
    /// worker set for sharding, and the worker's app/leaf/internal
    /// prover-pool + app-execution-context construction. Since the default
    /// role is `Full`, a default deployment has every worker
    /// `runs_stark_proving() == true`, so sharding and pool sizes are unchanged.
    pub fn runs_stark_proving(self) -> bool {
        matches!(self, WorkerRole::Full | WorkerRole::StarkOnly)
    }

    /// Whether a worker with this role runs the EVM step (root + halo2).
    /// `true` for [`WorkerRole::Full`] and [`WorkerRole::EvmDedicated`];
    /// `false` for [`WorkerRole::StarkOnly`].
    ///
    /// Drives the worker's root/halo2 prover-pool construction and gates which
    /// workers the manager may dispatch the `EvmProve` step to. `Full` and
    /// `EvmDedicated` workers build those provers and are eligible; a `StarkOnly`
    /// worker never builds them, so the manager routes the EVM step to the
    /// `EvmDedicated` worker instead.
    pub fn runs_evm_prove(self) -> bool {
        matches!(self, WorkerRole::Full | WorkerRole::EvmDedicated)
    }
}

/// Worker self-registration with the manager.
///
/// Workers post this on startup and periodically thereafter so the manager
/// knows the stack is healthy and which IDs are bound to which URLs.
///
/// The capacity fields (`max_app_provers`, `max_leaf_provers`,
/// `max_internal_provers`) report the worker's actual configured capacity.
/// The manager validates these against its own `[provers]` config and rejects
/// registrations that drift — the templated deploy is the single source of
/// truth, so disagreement signals stale config rather than legitimate
/// variation.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RegisterWorkerRequest {
    /// HTTP URL where this worker can be reached.
    pub worker_url: String,
    /// Exact stable worker ID (usually config `prover_id`).
    /// Manager validates that this URL is bound to that ID.
    pub worker_id: usize,
    /// Number of pre-loaded GPU app prover instances on this worker, also
    /// the maximum app-prove parallelism used per proof.
    pub max_app_provers: usize,
    /// Number of concurrent leaf proofs this worker can run.
    pub max_leaf_provers: usize,
    /// Number of concurrent internal proofs this worker can run.
    pub max_internal_provers: usize,
    /// Programs this worker has loaded vmexes for. Advisory, and the manager
    /// pushes every registered program absent from this list, so a worker
    /// that joins or restarts converges on the current loadout.
    #[serde(default)]
    pub loaded_programs: Vec<ProgramRef>,
    /// The deployment role this worker plays. Defaults to `Full` (today's
    /// behavior) when absent, so an older worker or an existing registration
    /// that omits the field deserializes wire-compatibly. The manager learns
    /// the role here but does **not** gate or enforce on it yet.
    #[serde(default)]
    pub worker_role: WorkerRole,
}

/// Response shape for `GET /loadout`. Returns the manager's current program
/// list, seeded from `EDGE_PROGRAMS` and extended by `/register_program`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LoadoutResponse {
    pub programs: Vec<ProgramRef>,
}

// =============================================================================
// Manager → Worker
// =============================================================================
//
// Internal to the deployment: work assignments dispatched from manager to
// workers. Two flavors:
//
// 1. `ShardedAppProveRequest` — proof-start kickoff, sent at /sharded_app_prove.
//    The manager posts one of these to every worker when a proof begins.
//    Each worker independently runs the executor and proves its modulo-
//    assigned segments.
//
// 2. `GeneralProveRequest` — aggregation step dispatch, sent at /recursion_prove.
//    Carries either a `LeafProveRequest` (aggregate K app proofs) or an
//    `InternalProveRequest` (aggregate K leaf or internal proofs into a
//    higher-layer internal proof).

/// Sharded app prove kickoff for a single worker.
///
/// Sent from manager to every worker at proof start. Each worker
/// independently runs the executor (to discover segment boundaries) and
/// then generates app STARK proofs for the segments where
/// `segment_idx % num_provers == prover_id`. Workers run in parallel; the
/// segment shard is implicit in (`prover_id`, `num_provers`).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ShardedAppProveRequest {
    pub proof_uuid: String,
    pub program: ProgramRef,
    // Note: no input/deferral paths and no labels here. The worker reads its
    // input (and any `DeferralState`s) from the deterministic staged paths it
    // reconstructs from `proof_uuid` + its loaded deferral keyset — the manager
    // fanned those files out to the same deterministic locations. Labels live on
    // the manager's ProofContext (lifecycle events + metrics). Workers only
    // need (proof_uuid, program, shard) to prove.
    /// Worker ID (0-indexed).
    pub prover_id: usize,
    /// Total number of workers.
    pub num_provers: usize,
    /// Optional override for OPENVM_MAX_SEGMENT_MEMORY.
    #[serde(default)]
    pub segment_memory: Option<usize>,
}

/// Leaf prove request — aggregate a batch of app proofs into a leaf proof.
#[derive(Serialize, Deserialize, Clone)]
pub struct LeafProveRequest {
    pub context: ProofContext,
    /// App proofs to be aggregated into a single leaf proof.
    /// Each entry is a bincode-encoded `proof::ProofWithPublicValue<F>`.
    pub app_proofs: Vec<ProofBytes>,
    pub segment_start: usize,
    /// Inclusive end segment index.
    pub segment_end: usize,
}

impl std::fmt::Debug for LeafProveRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeafProveRequest")
            .field("context", &self.context)
            .field("app_proofs_count", &self.app_proofs.len())
            .field("segment_start", &self.segment_start)
            .field("segment_end", &self.segment_end)
            .finish()
    }
}

/// Internal prove request — aggregate a group of leaf or internal proofs
/// into a higher-layer internal proof.
#[derive(Serialize, Deserialize, Clone)]
pub struct InternalProveRequest {
    pub context: ProofContext,
    /// Each entry is a bincode-encoded `proof::ProofWithPublicValue<F>`.
    pub child_proofs: Vec<ProofBytes>,
    pub layer_idx: usize,
    pub segment_start: usize,
    pub segment_end: usize,
    /// Whether this is the final proof in the recursion tree.
    pub is_final_proof: bool,
    /// The manager attaches this on the final
    /// internal prove of a deferral job so the worker that runs `run_evm_prove`
    /// can sequence `prove_def → prove_mixed → wrap` before root. `None` (the
    /// default) selects the non-deferral tail. This request travels as
    /// bincode, where a `None` still encodes a tag byte and `#[serde(default)]`
    /// provides no cross-version tolerance — see the crate-level *Transports
    /// and compatibility* notes.
    #[serde(default)]
    pub deferral_tail: Option<DeferralTailDispatch>,
    /// Encoded COMPLETE depth-0 `DeferralMerkleProofs` for a proof that made
    /// no deferred calls on a deferral deployment (built by the terminal app
    /// worker, buffered on the manager). The manager attaches it to the final
    /// internal prove of an `Evm` proof so `run_evm_prove` can hand the
    /// depth-0 proofs to root prove — a deferral-configured root circuit
    /// rejects a `VmStarkProof` carrying none. Mutually exclusive with a
    /// non-empty `deferral_tail` (a real deferral proof computes its own
    /// merkle proofs on the tail worker). `None` otherwise.
    #[serde(default)]
    pub deferral_merkle_proofs_bytes: Option<Vec<u8>>,
}

/// Manager → tail-worker handoff carrying the inputs the deferral merge
/// needs. Built by the manager on the final InternalProveRequest of a
/// deferral job; consumed by `run_evm_prove` to sequence `prove_def →
/// prove_mixed → wrap` before root.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeferralTailDispatch {
    /// Initial `InternalLayerMetadata`: manager-owned tree-shape state
    /// at the point the final internal proof is produced. The tail worker
    /// threads this locally through `prove_mixed → wrap`, mutating it in
    /// place — only the initial value crosses the wire.
    pub layer_metadata: InternalLayerMetadataWire,
    /// Depth-independent `(DEFERRAL_AS, 0)` authentication path
    /// from the FINAL memory merkle tree, extracted on the terminal app
    /// worker and buffered by the manager. Encoded via
    /// `proof::encode_deferral_auth_path` (length-prefixed digest slice).
    /// The tail worker decodes it, recomputes the INITIAL path locally
    /// from the exe, finalizes both with `depth` (from the merged proof's
    /// `DeferralPvs`), and attaches a `DeferralMerkleProofs` to the
    /// `VmStarkProof` before root. Empty `Vec` means the manager couldn't
    /// capture the path (terminal `AppProof` never reported it) — the
    /// tail worker will fail fast with a clear error.
    #[serde(default)]
    pub final_merkle_path_bytes: Vec<u8>,
}

/// Wire encoding of openvm SDK's `InternalLayerMetadata`. Three small u32s
/// (the third packs `ProofsType` as `ProofsTypeWire`); the tail worker
/// converts to the SDK type before invoking `prove_mixed`/`wrap`.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default)]
pub struct InternalLayerMetadataWire {
    /// Number of `internal_recursive` rounds reached at the point the final
    /// internal proof was produced (openvm starts counting at 1 for the
    /// first internal_recursive round; matches edge's `effective_final_layer`).
    pub internal_recursive_layer: u32,
    /// Highest assigned internal node index across all `internal_*_prover`
    /// invocations in the VM tree (openvm's monotonic counter); equals
    /// `total_internal_proofs - 1` (the SDK uses -1 init then post-increment).
    pub internal_node_idx: u32,
    /// `ProofsType` flag at the point the final internal proof emerged. For
    /// the VM tree this is always `Vm`; `prove_mixed`/`wrap` flip it on the
    /// tail worker.
    pub proofs_type: ProofsTypeWire,
}

/// Mirror of openvm SDK's `ProofsType` for the wire (the SDK enum is not
/// `Serialize`/`Deserialize`).
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProofsTypeWire {
    #[default]
    Vm,
    Deferral,
    Mix,
    Combined,
}

impl std::fmt::Debug for InternalProveRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InternalProveRequest")
            .field("context", &self.context)
            .field("child_proofs_count", &self.child_proofs.len())
            .field("layer_idx", &self.layer_idx)
            .field("segment_start", &self.segment_start)
            .field("segment_end", &self.segment_end)
            .field("is_final_proof", &self.is_final_proof)
            .field("has_deferral_tail", &self.deferral_tail.is_some())
            .finish()
    }
}

/// EVM prove request — the finished internal proof handed off to an eligible
/// worker, which runs root → halo2 and posts the `Evm` result. Dispatched by
/// the manager as a first-class [`Step::EvmProve`] to any `runs_evm_prove()`
/// worker: a `Full` worker in the default deployment, or the `EvmDedicated`
/// worker in dedicated-halo2 mode.
///
/// The handoff boundary is drawn **after** the deferral tail merge (which
/// stays on the final-internal worker), so the EVM-step worker is
/// deferral-agnostic: this request carries only plain bytes/data — **no
/// deferral inputs, no tail-dispatch**. Deferral and non-deferral proofs use
/// the identical handoff.
#[derive(Serialize, Deserialize, Clone)]
pub struct EvmProveRequest {
    pub context: ProofContext,
    /// The finished internal proof: a bincode-encoded
    /// `proof::ProofWithPublicValue<F>`. Root prove builds its `VmStarkProof`
    /// on this. For a deferral proof it is the merged (`prove_def →
    /// prove_mixed → wrap`) proof; for a non-deferral proof it is the raw final
    /// internal (no merge ran).
    pub internal_proof_bytes: ProofBytes,
    /// Serialized deferral merkle proofs (`verify_stark::deferral::
    /// DeferralMerkleProofs::encode`), attached to root's `VmStarkProof` before
    /// tracegen. `Some` from the tail merge (deferral proof) or the depth-0
    /// proofs (no-deferral proof on a deferral deployment); `None` on a
    /// non-deferral deployment.
    #[serde(default)]
    pub deferral_merkle_proofs_bytes: Option<Vec<u8>>,
    /// Whether this proof ran the deferral tail merge. Drives root's
    /// `proofs_type` (`Combined` when set, `Vm` otherwise) — distinct from
    /// `deferral_merkle_proofs_bytes.is_some()`, since a no-deferral proof on a
    /// deferral deployment carries depth-0 merkle proofs yet keeps `Vm`.
    pub proof_has_deferral: bool,
}

impl std::fmt::Debug for EvmProveRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvmProveRequest")
            .field("context", &self.context)
            .field("internal_proof_bytes_len", &self.internal_proof_bytes.len())
            .field(
                "deferral_merkle_proofs_bytes_len",
                &self.deferral_merkle_proofs_bytes.as_ref().map(|b| b.len()),
            )
            .field("proof_has_deferral", &self.proof_has_deferral)
            .finish()
    }
}

/// Aggregation prove requests dispatched per-step on the worker.
///
/// The sharded app prove stage is dispatched separately via
/// [`ShardedAppProveRequest`] (one request per worker at proof start). This
/// enum covers the follow-up aggregation steps the manager assigns as
/// individual app/leaf proofs become available.
#[derive(Serialize, Deserialize, Clone)]
pub enum GeneralProveRequest {
    LeafProve(LeafProveRequest),
    InternalProve(InternalProveRequest),
    /// The EVM step (root → halo2), dispatched in every deployment mode to
    /// any `runs_evm_prove()` worker: a `Full` worker in the default
    /// deployment, or the `EvmDedicated` worker in dedicated-halo2 mode.
    EvmProve(EvmProveRequest),
}

impl std::fmt::Debug for GeneralProveRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LeafProve(req) => f.debug_tuple("LeafProve").field(req).finish(),
            Self::InternalProve(req) => f.debug_tuple("InternalProve").field(req).finish(),
            Self::EvmProve(req) => f.debug_tuple("EvmProve").field(req).finish(),
        }
    }
}

impl From<&GeneralProveRequest> for Step {
    fn from(req: &GeneralProveRequest) -> Self {
        match req {
            GeneralProveRequest::LeafProve(_) => Step::LeafProve,
            GeneralProveRequest::InternalProve(_) => Step::InternalProve,
            GeneralProveRequest::EvmProve(_) => Step::EvmProve,
        }
    }
}

impl GeneralProveRequest {
    pub fn request_details(&self) -> RequestDetails {
        RequestDetails {
            context: self.context().clone(),
            step: Step::from(self),
        }
    }
}

impl WithProofContext for GeneralProveRequest {
    fn context(&self) -> &ProofContext {
        match self {
            GeneralProveRequest::LeafProve(req) => &req.context,
            GeneralProveRequest::InternalProve(req) => &req.context,
            GeneralProveRequest::EvmProve(req) => &req.context,
        }
    }

    fn proof_uuid(&self) -> &str {
        &self.context().proof_uuid
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::{ProofType, RegisterWorkerRequest, StartProofRequest, WorkerRole};

    #[test]
    fn start_proof_request_defaults_input_already_uploaded_to_false() {
        let request: StartProofRequest = serde_json::from_value(serde_json::json!({
            "proof_uuid": "proof-1",
            "program": {"name": "program-1", "version": 1},
            "labels": {"block_number": "1"},
        }))
        .unwrap();

        // Default transport is manager-staged (Flow 2).
        assert!(!request.input_already_uploaded);
        let program = request.program.expect("program field present in fixture");
        assert_eq!(program.name, "program-1");
        assert_eq!(program.version, 1);
        assert_eq!(
            request.labels.get("block_number").map(String::as_str),
            Some("1")
        );
        assert_eq!(request.proof_type, ProofType::Stark);
    }

    #[test]
    fn start_proof_request_program_field_is_optional() {
        // Single-program deployments can omit `program`; manager resolves
        // it server-side iff the loadout has exactly one program.
        let request: StartProofRequest = serde_json::from_value(serde_json::json!({
            "proof_uuid": "proof-2",
        }))
        .unwrap();
        assert!(request.program.is_none());
        assert_eq!(request.proof_type, ProofType::Stark);
    }

    #[test]
    fn start_proof_request_deserializes_evm_proof_type() {
        let request: StartProofRequest = serde_json::from_value(serde_json::json!({
            "proof_uuid": "proof-3",
            "proof_type": "evm"
        }))
        .unwrap();

        assert_eq!(request.proof_type, ProofType::Evm);
    }

    #[test]
    fn start_proof_request_minimal_deserializes() {
        // Minimal request: only proof_uuid. Deferral is not declared on the
        // request — the manager infers it from staged uploads — so a bare
        // request is a valid non-deferral, manager-staged (Flow 2) proof.
        let request: StartProofRequest = serde_json::from_value(serde_json::json!({
            "proof_uuid": "proof-no-def",
        }))
        .unwrap();
        assert!(!request.input_already_uploaded);
        assert_eq!(request.proof_type, ProofType::Stark);
    }

    #[test]
    fn sharded_app_prove_request_carries_no_paths() {
        // The request carries no input/deferral paths — the worker reconstructs
        // the deterministic staged paths from `proof_uuid` (+ its keyset count).
        let json = serde_json::json!({
            "proof_uuid": "proof-x",
            "program": {"name": "p", "version": 1},
            "prover_id": 0,
            "num_provers": 1,
        });
        let req: super::ShardedAppProveRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.proof_uuid, "proof-x");
        assert_eq!(req.prover_id, 0);
        assert_eq!(req.num_provers, 1);
    }

    #[test]
    fn register_request_defaults_loaded_programs_to_empty() {
        let request: RegisterWorkerRequest = serde_json::from_value(serde_json::json!({
            "worker_url": "http://10.0.0.1:8001",
            "worker_id": 7,
            "max_app_provers": 2,
            "max_leaf_provers": 2,
            "max_internal_provers": 1,
        }))
        .unwrap();

        assert_eq!(request.worker_id, 7);
        assert_eq!(request.max_leaf_provers, 2);
        assert!(request.loaded_programs.is_empty());
    }

    #[test]
    fn register_request_defaults_worker_role_to_full() {
        // Back-compat: an older worker (or existing test fixture) that omits
        // `worker_role` deserializes to `Full`, i.e. today's behavior.
        let request: RegisterWorkerRequest = serde_json::from_value(serde_json::json!({
            "worker_url": "http://10.0.0.1:8001",
            "worker_id": 0,
            "max_app_provers": 2,
            "max_leaf_provers": 2,
            "max_internal_provers": 1,
        }))
        .unwrap();

        assert_eq!(request.worker_role, WorkerRole::Full);
    }

    #[test]
    fn register_request_parses_worker_role_variants() {
        for (wire, expected) in [
            ("full", WorkerRole::Full),
            ("stark_only", WorkerRole::StarkOnly),
            ("evm_dedicated", WorkerRole::EvmDedicated),
        ] {
            let request: RegisterWorkerRequest = serde_json::from_value(serde_json::json!({
                "worker_url": "http://10.0.0.1:8001",
                "worker_id": 0,
                "max_app_provers": 2,
                "max_leaf_provers": 2,
                "max_internal_provers": 1,
                "worker_role": wire,
            }))
            .unwrap();
            assert_eq!(request.worker_role, expected);
        }
    }

    #[test]
    fn register_request_round_trips_worker_role() {
        let request = RegisterWorkerRequest {
            worker_url: "http://10.0.0.1:8001".to_string(),
            worker_id: 3,
            max_app_provers: 2,
            max_leaf_provers: 2,
            max_internal_provers: 1,
            loaded_programs: Vec::new(),
            worker_role: WorkerRole::EvmDedicated,
        };

        let encoded = serde_json::to_value(&request).unwrap();
        // Serializes to the stable snake_case wire form.
        assert_eq!(encoded["worker_role"], serde_json::json!("evm_dedicated"));

        let decoded: RegisterWorkerRequest = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.worker_role, WorkerRole::EvmDedicated);
        assert_eq!(decoded.worker_id, 3);
    }

    #[test]
    fn worker_role_predicates() {
        // Full runs both halves — today's behavior.
        assert!(WorkerRole::Full.runs_stark_proving());
        assert!(WorkerRole::Full.runs_evm_prove());
        // StarkOnly runs app/leaf/internal only.
        assert!(WorkerRole::StarkOnly.runs_stark_proving());
        assert!(!WorkerRole::StarkOnly.runs_evm_prove());
        // EvmDedicated runs the EVM step only.
        assert!(!WorkerRole::EvmDedicated.runs_stark_proving());
        assert!(WorkerRole::EvmDedicated.runs_evm_prove());
        // The default role is Full, so a default deployment is stark-eligible.
        assert!(WorkerRole::default().runs_stark_proving());
    }
}
