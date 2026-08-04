//! HTTP endpoint handlers for the Edge manager.

use axum::{
    body::Bytes,
    extract::{Multipart, Path, State},
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::config::ManagerConfig;
use crate::lifecycle::LifecycleReporter;
use crate::proof_state::{ProofResultEnvelopeOutcome, ProofState, ProofStatus};
use crate::scheduler::{AssignedWork, EdgeStateStore};
use crate::worker_registry::{app_eligible_workers, EdgeWorkerRegistry, RegisteredWorker};
use protocol::{
    current_timestamp, GeneralProveRequest, LoadoutResponse, MessageEnvelope, ProgramRef,
    ProofContext, ProofResult, RegisterWorkerRequest, ResultPayload, ShardedAppProveRequest,
    StartProofRequest, Step, WithProofContext,
};
use std::collections::{BTreeMap, HashSet};

/// (manager-staged) input bytes, uploaded to the manager by `proof_uuid`
/// via `/upload_input`, `/upload_deferral_state`, and `/upload_deferral_input`
/// before `/start_proof` is called. Held in memory only briefly:
///
/// - `main` and `deferral_states` are consumed by `start_proof`, which fans
///   them out to every worker, then drops them.
/// - `deferral_inputs` are *retained* past `start_proof` (they are NOT
///   broadcast) and pushed just-in-time to the single worker that produces the
///   final internal proof, right before that dispatch (see the JIT upload in
///   the result path). They are dropped once dispatched.
///
/// Deferral maps are keyed by circuit index so uploads may arrive in any order.
#[derive(Default)]
pub struct StagedInputs {
    /// bincode `StdIn` bytes for the main program input.
    pub main: Option<Bytes>,
    /// `DeferralState` bytes per circuit index.
    pub deferral_states: BTreeMap<usize, Bytes>,
    /// `DeferralInput` bytes per circuit index (retained until JIT dispatch).
    pub deferral_inputs: BTreeMap<usize, Bytes>,
    /// When these bytes were staged (set on the `/upload_input` request). Used
    /// by the watchdog to reclaim *orphaned* uploads — an `/upload_input` that
    /// no `/start_proof` ever followed. `None` for the re-staged retained
    /// deferral inputs (those are protected by their live proof's entry in
    /// `proof_states`, so the sweep never touches them).
    pub staged_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Shared application state.
pub struct AppState {
    pub config: ManagerConfig,
    pub worker_registry: EdgeWorkerRegistry,
    pub state_store: EdgeStateStore,
    /// Admission gate for `/start_proof`. Serializes the "is a proof already
    /// running?" check and the subsequent state insert into a single critical
    /// section so two concurrent requests with distinct UUIDs cannot both
    /// observe an empty store and both initialize against the same physical
    /// workers (the "one active proof at a time" invariant). Held only across
    /// the synchronous check→insert, never across the later input upload.
    pub start_proof_gate: Mutex<()>,
    pub proof_states: DashMap<String, Arc<Mutex<ProofState>>>,
    pub lifecycle_reporter: Option<Arc<LifecycleReporter>>,
    /// HTTP client for general requests (30s timeout)
    pub http_client: reqwest::Client,
    /// HTTP client for large uploads (5 minute timeout)
    pub upload_client: reqwest::Client,
    /// Canonical program loadout, parsed once at startup from
    /// `EDGE_PROGRAMS`. Two representations of the same data:
    ///
    /// - `programs` (Vec) preserves operator-supplied order, used in
    ///   API responses (`/loadout`, 409 body) and log lines where stable
    ///   order matters.
    /// - `programs_set` (HashSet) is used for the two set-ops we do:
    ///   `/start_proof` membership check and `/register_worker`
    ///   loadout-equality check.
    ///
    /// At 3–5 entries either alone would be fine; carrying both keeps
    /// every call site terse without a `.collect()` on the read path.
    pub programs: Vec<ProgramRef>,
    pub programs_set: HashSet<ProgramRef>,
    /// Input bytes uploaded to the manager, keyed by `proof_uuid`.
    /// Populated by the `/upload_input*` endpoints, drained by `start_proof`
    /// (main + deferral states) and the final-internal JIT dispatch
    /// (deferral inputs). See [`StagedInputs`].
    pub staged_inputs: DashMap<String, StagedInputs>,
    /// Root of the artifacts export mounted read-only into the container
    /// (from `server.artifacts_path`, defaulting to `/data/artifacts`).
    /// `GET /vk/{name}` serves per-program verification baselines from it.
    pub artifacts_path: std::path::PathBuf,
}

impl AppState {
    pub fn new(config: ManagerConfig, programs: Vec<ProgramRef>) -> Self {
        let lifecycle_reporter = LifecycleReporter::from_config(&config.lifecycle).map(Arc::new);
        let worker_registry =
            EdgeWorkerRegistry::new(config.server.num_workers, config.provers.clone());
        let state_store = EdgeStateStore::new(config.provers.max_leaf_provers);
        let programs_set: HashSet<ProgramRef> = programs.iter().cloned().collect();
        // Root of the mounted artifacts export, served by `GET /vk/{name}`.
        let artifacts_path = config
            .server
            .artifacts_path
            .clone()
            .unwrap_or_else(|| std::path::PathBuf::from("/data/artifacts"));
        Self {
            config,
            worker_registry,
            state_store,
            start_proof_gate: Mutex::new(()),
            proof_states: DashMap::new(),
            lifecycle_reporter,
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("Failed to create HTTP client"),
            upload_client: reqwest::Client::builder()
                // Fan-out uploads go worker-to-worker on the internal network,
                // so even large inputs finish in seconds; a shorter ceiling
                // fails a hung worker fast instead of stalling the whole
                // fan-out (and, in turn, the sharded_app_prove dispatch).
                .timeout(std::time::Duration::from_secs(60))
                .connect_timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create upload HTTP client"),
            programs,
            programs_set,
            staged_inputs: DashMap::new(),
            artifacts_path,
        }
    }
}

/// Max accepted `proof_uuid` length for manager endpoints that take one.
const MAX_PROOF_UUID_LEN: usize = 128;

/// Validate a caller-supplied `proof_uuid`. Bounds length and restricts the
/// charset (alphanumeric + `_`/`-`) so the uuid is safe both as an in-memory
/// staging key and as a filename component — it later flows into filesystem
/// paths (metrics reports, persisted proofs) and into worker-side staging
/// dirs, so path separators and `..` must never get through.
fn validate_manager_proof_uuid(proof_uuid: &str) -> Result<(), &'static str> {
    if proof_uuid.is_empty() {
        return Err("must not be empty");
    }
    if proof_uuid.len() > MAX_PROOF_UUID_LEN {
        return Err("too long");
    }
    if !proof_uuid
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
    {
        return Err("contains invalid characters");
    }
    Ok(())
}

/// `POST /upload_input/{proof_uuid}` — stage all input
/// for a proof in ONE `multipart/form-data` request, before `/start_proof`.
///
/// Parts (all optional):
/// - `input` — the bincode `StdIn` bytes (the main program input).
/// - `deferral_state_{i}` / `deferral_input_{i}` — one pair per deferral
///   circuit, at contiguous indices `0..N`. Omit entirely for a non-deferral
///   proof.
///
/// The manager holds the parsed bytes in memory keyed by `proof_uuid`;
/// `/start_proof` then fans the input + each `DeferralState` out to the workers
/// and retains each `DeferralInput` for the just-in-time tail-worker push.
/// One request replaces the former per-artifact endpoints, so a caller makes a
/// single upload call regardless of circuit count.
pub async fn upload_input(
    State(state): State<Arc<AppState>>,
    Path(proof_uuid): Path<String>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    if let Err(reason) = validate_manager_proof_uuid(&proof_uuid) {
        return (
            StatusCode::BAD_REQUEST,
            format!("Invalid proof_uuid: {reason}"),
        );
    }

    let mut staged = StagedInputs::default();
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Malformed multipart body: {e}"),
                );
            }
        };
        let name = field.name().unwrap_or_default().to_string();
        let bytes = match field.bytes().await {
            Ok(bytes) => bytes,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to read multipart part `{name}`: {e}"),
                );
            }
        };

        if name == "input" {
            staged.main = Some(bytes);
        } else if let Some(idx) = name.strip_prefix("deferral_state_") {
            match idx.parse::<usize>() {
                Ok(i) => {
                    staged.deferral_states.insert(i, bytes);
                }
                Err(_) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        format!(
                            "Invalid multipart part name `{name}` (expected deferral_state_<idx>)"
                        ),
                    );
                }
            }
        } else if let Some(idx) = name.strip_prefix("deferral_input_") {
            match idx.parse::<usize>() {
                Ok(i) => {
                    staged.deferral_inputs.insert(i, bytes);
                }
                Err(_) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        format!(
                            "Invalid multipart part name `{name}` (expected deferral_input_<idx>)"
                        ),
                    );
                }
            }
        } else {
            return (
                StatusCode::BAD_REQUEST,
                format!("Unexpected multipart part `{name}`"),
            );
        }
    }

    let (has_main, n_states, n_inputs) = (
        staged.main.is_some(),
        staged.deferral_states.len(),
        staged.deferral_inputs.len(),
    );
    // Stamp for the watchdog's orphan sweep (upload with no following start_proof).
    staged.staged_at = Some(chrono::Utc::now());
    state.staged_inputs.insert(proof_uuid.clone(), staged);
    info!(
        "Staged input for proof {} (main={}, deferral_states={}, deferral_inputs={})",
        proof_uuid, has_main, n_states, n_inputs
    );
    (StatusCode::OK, "Input staged".to_string())
}

/// Whether `name` is acceptable as a program name in a filesystem path.
///
/// Names are restricted to non-empty ASCII `[A-Za-z0-9._-]` with no `..`
/// substring. Axum single-segment path params can never contain `/`, but the
/// name becomes a filesystem path component, so it is validated here anyway
/// (belt and braces against traversal).
fn is_acceptable_program_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains("..")
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// The relative path of a program's verification baseline inside the mounted
/// artifacts export.
///
/// `convert_fixtures keygen` writes `baseline.bin` beside the program's vmexe,
/// so the layout matches the one the workers already load from.
fn baseline_rel_path(name: &str, version: u32) -> String {
    format!("programs/{name}/{version}/baseline.bin")
}

/// Download a program's verification baseline, the bitcode encoding of the
/// openvm `VerificationBaseline`.
///
/// The baseline is a pure function of the guest ELF under the deployment's VM
/// config, so a caller identifies a program by name and the loadout supplies
/// the version. The manager serves the bytes verbatim and decodes nothing.
pub async fn download_vk(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if !is_acceptable_program_name(&name) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "invalid program name"})),
        )
            .into_response();
    }
    // The loadout is the manager's only source of versions, so a name carrying
    // two of them cannot be resolved from the path alone.
    let versions: Vec<u32> = state
        .programs
        .iter()
        .filter(|program| program.name == name)
        .map(|program| program.version)
        .collect();
    let version = match versions.as_slice() {
        [version] => *version,
        [] => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("program '{name}' is not in the loadout")
                })),
            )
                .into_response();
        }
        _ => {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": format!(
                        "program '{name}' has {} versions in the loadout, so a baseline cannot be selected by name",
                        versions.len()
                    )
                })),
            )
                .into_response();
        }
    };
    match tokio::fs::read(state.artifacts_path.join(baseline_rel_path(&name, version))).await {
        Ok(bytes) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/octet-stream")],
            bytes,
        )
            .into_response(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no baseline for program '{name}' on this deployment")
            })),
        )
            .into_response(),
        Err(e) => {
            error!("failed to read baseline for program '{name}': {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "failed to read baseline"})),
            )
                .into_response()
        }
    }
}

/// Download an uncompressed final proof from persistent storage.
///
/// Disk is the source of truth, so proofs remain available after in-memory
/// eviction or restart. Persistence-disabled and missing proofs return 404.
pub async fn download_proof(
    State(state): State<Arc<AppState>>,
    Path(proof_uuid): Path<String>,
) -> impl IntoResponse {
    if let Err(reason) = validate_manager_proof_uuid(&proof_uuid) {
        return (
            StatusCode::BAD_REQUEST,
            format!("Invalid proof_uuid: {reason}"),
        )
            .into_response();
    }
    let Some(dir) = state.config.proof.persist_final_proofs_dir.as_ref() else {
        return (
            StatusCode::NOT_FOUND,
            "proof persistence is disabled".to_string(),
        )
            .into_response();
    };
    let stark_path = dir.join(format!("{proof_uuid}.proof.bin"));
    let evm_path = dir.join(format!("{proof_uuid}.evm.bin"));
    let (stark_exists, evm_exists) = match tokio::try_join!(
        tokio::fs::try_exists(&stark_path),
        tokio::fs::try_exists(&evm_path)
    ) {
        Ok(exists) => exists,
        Err(e) => {
            error!("failed to inspect persisted proof paths for {proof_uuid}: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("inspect persisted proof: {e}"),
            )
                .into_response();
        }
    };
    let path = match (stark_exists, evm_exists) {
        (true, false) => stark_path,
        (false, true) => evm_path,
        (false, false) => {
            return (StatusCode::NOT_FOUND, "proof not found".to_string()).into_response();
        }
        (true, true) => {
            error!("both STARK and EVM persisted artifacts exist for {proof_uuid}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "multiple persisted proof artifacts found".to_string(),
            )
                .into_response();
        }
    };

    // Read (and, if the deployment compresses persisted proofs, decompress) off
    // the async executor — proofs can be multi-MB and zstd decode is CPU-bound.
    let compressed = state.config.proof.compress_persisted_final_proofs;
    let read = tokio::task::spawn_blocking(move || -> std::io::Result<Vec<u8>> {
        let bytes = std::fs::read(&path)?;
        if compressed {
            zstd::decode_all(&bytes[..])
        } else {
            Ok(bytes)
        }
    })
    .await;
    match read {
        Ok(Ok(bytes)) => (StatusCode::OK, bytes).into_response(),
        Ok(Err(e)) => {
            error!("failed to read persisted proof for {proof_uuid}: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("read persisted proof: {e}"),
            )
                .into_response()
        }
        Err(e) => {
            error!("persisted-proof read task failed for {proof_uuid}: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("read task failed: {e}"),
            )
                .into_response()
        }
    }
}

/// Response for start_proof endpoint.
#[derive(serde::Serialize)]
pub struct StartProofResponse {
    pub proof_uuid: String,
    pub status: String,
}

/// Response for healthz endpoint.
#[derive(serde::Serialize)]
pub struct HealthzResponse {
    pub status: String,
}

/// Response for manager-controlled worker readiness.
#[derive(serde::Serialize)]
pub struct EdgeReadyResponse {
    pub ready: bool,
    pub num_workers: usize,
    pub expected_num_workers: usize,
    pub workers: Vec<(usize, RegisteredWorker)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Health check endpoint.
pub async fn healthz() -> Json<HealthzResponse> {
    Json(HealthzResponse {
        status: "healthy".to_string(),
    })
}

/// Register an Edge worker.
pub async fn register_worker(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterWorkerRequest>,
) -> impl IntoResponse {
    debug!(
        "Registering Edge worker: worker_url={}, worker_id={}, loaded_programs={}, worker_role={:?}",
        req.worker_url,
        req.worker_id,
        req.loaded_programs.len(),
        req.worker_role,
    );

    // Consistency check: worker's loaded programs must match the manager's
    // canonical loadout (both come from the same EDGE_PROGRAMS env). A
    // mismatch means one container started with the wrong env — fail loud.
    let actual: HashSet<ProgramRef> = req.loaded_programs.iter().cloned().collect();
    if actual != state.programs_set {
        error!(
            "Worker {} program set mismatch. expected={:?}, actual={:?}",
            req.worker_id, state.programs, req.loaded_programs
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "program_set_mismatch",
                "message": "Worker loaded_programs does not match manager EDGE_PROGRAMS",
                "expected": state.programs,
                "actual": req.loaded_programs,
            })),
        );
    }

    let provers_config = crate::config::ProversConfig {
        max_app_provers: req.max_app_provers,
        max_leaf_provers: req.max_leaf_provers,
        max_internal_provers: req.max_internal_provers,
    };

    match state.worker_registry.register(
        &req.worker_url,
        req.worker_id,
        provers_config,
        req.worker_role,
    ) {
        Ok(worker_id) => (
            StatusCode::OK,
            Json(serde_json::json!({"status": "ok", "worker_id": worker_id})),
        ),
        Err(e) => {
            error!("Failed to register worker: {}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": e.to_string()})),
            )
        }
    }
}

/// Return the canonical program loadout. The response shape is stable so an
/// upstream orchestration layer can proxy it directly.
pub async fn get_loadout(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(LoadoutResponse {
        programs: state.programs.clone(),
    })
}

/// Get registered Edge workers.
pub async fn list_workers(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let status = state.worker_registry.get_status();
    Json(status)
}

async fn probe_worker_health(
    state: &AppState,
    workers: &[(usize, RegisteredWorker)],
    max_retries: usize,
    retry_delay: std::time::Duration,
) -> Result<(), String> {
    for (worker_id, worker) in workers {
        let url = format!("{}/readyz", worker.worker_url);
        let mut retries = 0;

        loop {
            match state.http_client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => break,
                Ok(resp) => {
                    retries += 1;
                    if retries >= max_retries {
                        return Err(format!(
                            "Worker {} at {} not ready: {}",
                            worker_id,
                            worker.worker_url,
                            resp.status()
                        ));
                    }
                    if retries == 1 {
                        info!(
                            "Waiting for worker {} to be ready (provers initializing)...",
                            worker_id
                        );
                    }
                    tokio::time::sleep(retry_delay).await;
                }
                Err(e) => {
                    retries += 1;
                    if retries >= max_retries {
                        return Err(format!(
                            "Worker {} not reachable at {}: {}",
                            worker_id, worker.worker_url, e
                        ));
                    }
                    tokio::time::sleep(retry_delay).await;
                }
            }
        }
    }

    Ok(())
}

async fn get_ready_workers(
    state: &AppState,
    max_retries: usize,
    retry_delay: std::time::Duration,
) -> Result<Vec<(usize, RegisteredWorker)>, String> {
    let workers = state
        .worker_registry
        .ready_workers()
        .map_err(|e| e.to_string())?;
    probe_worker_health(state, &workers, max_retries, retry_delay).await?;
    Ok(workers)
}

/// Return the worker list only when the full registered stack is ready.
pub async fn readyz_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let expected_num_workers = state.worker_registry.expected_worker_count();
    match get_ready_workers(&state, 1, std::time::Duration::from_secs(0)).await {
        Ok(workers) => (
            StatusCode::OK,
            Json(EdgeReadyResponse {
                ready: true,
                num_workers: workers.len(),
                expected_num_workers,
                workers,
                message: None,
            }),
        ),
        Err(message) => {
            let status = state.worker_registry.get_status();
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(EdgeReadyResponse {
                    ready: false,
                    num_workers: status.num_workers,
                    expected_num_workers,
                    workers: status.workers,
                    message: Some(message),
                }),
            )
        }
    }
}

/// Start an Edge proof.
pub async fn start_proof(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StartProofRequest>,
) -> impl IntoResponse {
    // The uuid becomes a filename component (metrics report, persisted
    // proofs) and a worker staging-dir name, so reject anything outside the
    // strict allowlist before it reaches a path join.
    if let Err(reason) = validate_manager_proof_uuid(&req.proof_uuid) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("Invalid proof_uuid: {reason}")})),
        );
    }

    // Resolve `program`: required when ≥ 2 programs loaded; optional when
    // exactly 1 (the sole program is used). Validate against the loadout
    // whether explicit or inferred — the 409 body carries a stable,
    // machine-readable `error` code (`program_not_in_loadout`) so an
    // upstream layer can forward it straight to the user.
    let program = match req.program.clone() {
        Some(p) => {
            if !state.programs_set.contains(&p) {
                warn!(
                    "Rejecting proof {}: program {} not in loadout",
                    req.proof_uuid, p
                );
                return (
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "error": "program_not_in_loadout",
                        "message": format!("Program {} is not in the current loadout", p),
                        "current_loadout": state.programs,
                    })),
                );
            }
            p
        }
        None => match state.programs.as_slice() {
            [only] => only.clone(),
            _ => {
                warn!(
                    "Rejecting proof {}: `program` omitted but loadout has {} programs",
                    req.proof_uuid,
                    state.programs.len()
                );
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": "program_required",
                        "message": "Multiple programs loaded; specify `program: {name, version}` in the request",
                        "current_loadout": state.programs,
                    })),
                );
            }
        },
    };

    info!(
        "Starting Edge proof: proof_uuid={}, program={}",
        req.proof_uuid, program
    );

    // Check if workers are registered
    let workers = match get_ready_workers(&state, 60, std::time::Duration::from_secs(2)).await {
        Ok(w) => w,
        Err(e) => {
            error!("Workers not ready: {}", e);
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": e.to_string()})),
            );
        }
    };

    // Hold the admission gate so the single-active-proof check and the state
    // insert below form one atomic critical section. Without this, two
    // concurrent /start_proof calls on different runtime threads can both
    // observe an empty store and both initialize against the same workers.
    // The guarded region is .await-free; the guard is dropped before the
    // (potentially long) input upload so it never serializes that work.
    let admission = state.start_proof_gate.lock().await;

    // Check if another proof is already running
    if state.state_store.has_any_proofs() {
        error!(
            "Cannot start proof {}: another proof is already running",
            req.proof_uuid
        );
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "Another proof is already running"})),
        );
    }

    // Check for duplicate proof_uuid
    if state.proof_states.contains_key(&req.proof_uuid) {
        error!("Proof {} already exists", req.proof_uuid);
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": "Proof already exists"})),
        );
    }

    let proof_uuid = req.proof_uuid.clone();

    // App sharding runs over the **app-eligible (normal) worker set** — every
    // registered worker except `EvmDedicated` ones. This set — not
    // `workers.len()` — drives `num_provers`, the per-segment `prover_id`/shard
    // assignment, the proof-state app-proof count, and scheduler init, so the
    // dedicated worker owns no modulo shard and receives no `sharded_app_prove`.
    // In a default deployment (no `EvmDedicated` worker) this is all N workers,
    // so every value below is bit-identical to today. Input / deferral uploads
    // still fan out to every worker (harmless for the dedicated worker; keeps
    // the diff scoped to sharding).
    let app_workers = app_eligible_workers(&workers);
    let num_provers = app_workers.len();

    // Create proof context with the resolved program (req.program is
    // `Option`; we use the validated `program` from the resolution above).
    //
    // The deferral KEYSET is a deployment toggle (the worker's keyset is
    // either deferral-configured or not), but whether a given proof runs
    // the deferral machinery is PER-PROOF: when the caller staged deferral
    // artifacts the manager attaches `DeferralTailDispatch` to the final
    // InternalProveRequest so the tail worker knows where to load each
    // `DeferralInput` from + which `InternalLayerMetadata` to seed
    // `prove_mixed` with. Workers key their per-proof shape (final-internal
    // wrap-skip, root `ProofsType::Combined`, the merge itself) on the
    // presence of that tail — so a deferral deployment serves non-deferral
    // proofs too (they take the normal wrap and get a depth-0
    // `DeferralMerkleProofs` from the terminal app worker).
    let mut context =
        ProofContext::new(req.proof_uuid.clone(), program.clone(), req.labels.clone());
    context.proof_type = req.proof_type;

    // Resolve per-request timeout override, falling through to manager config.
    let timeout_secs = req.timeout_secs.unwrap_or(state.config.proof.timeout_secs);

    // Initialize proof state
    let proof_state = Arc::new(Mutex::new(ProofState::new(
        context.clone(),
        u64::MAX, // No cost limit in Edge mode
        num_provers,
        state.config.proof.leaf_arity,
        state.config.proof.internal_arity,
        timeout_secs,
    )));
    state.proof_states.insert(proof_uuid.clone(), proof_state);

    // Initialize scheduler state with the FULL worker set (not just the
    // app-eligible set): the `EvmDedicated` worker must be tracked so the
    // manager can dispatch the `EvmProve` tail to it. `init_proof` seeds the
    // dedicated worker idle (no shard) and derives the sharding `num_workers`
    // from the app-eligible subset, so a default (all-`Full`) deployment is
    // unchanged.
    let leaf_pack_threshold = req
        .leaf_pack_threshold
        .unwrap_or(state.config.proof.leaf_pack_threshold);
    state
        .state_store
        .init_proof(&proof_uuid, workers.clone(), leaf_pack_threshold);

    // The proof now owns the single active slot; release the admission gate so
    // it isn't held across the input upload / worker fan-out below.
    drop(admission);

    if let Some(reporter) = &state.lifecycle_reporter {
        reporter.report_queued(&proof_uuid, &req.labels);
    }

    // ---- Resolve the proof input transport (see `StartProofRequest` docs) ----
    //
    // Flow 1 (`input_already_uploaded == true`): the caller pushed the input
    // directly to every worker (e.g. `/upload_input_compact`), so the manager
    // skips fan-out. Deferral is not supported on this path.
    //
    // Flow 2 (default): the caller staged the input on the manager (via
    // `/upload_input/{uuid}`, plus `/upload_deferral_{state,input}` for a
    // deferral proof). The manager fans the main input + each `DeferralState`
    // out to every worker and RETAINS each `DeferralInput` for the just-in-time
    // push to the worker that produces the final internal proof.
    // Drain this proof's manager-staged bytes. `main` + `deferral_states` are
    // consumed here (fanned out below); only the retained `deferral_inputs` are
    // re-staged afterwards for the JIT push at the final-internal dispatch.
    let mut staged = state
        .staged_inputs
        .remove(&proof_uuid)
        .map(|(_, v)| v)
        .unwrap_or_default();

    // Whether this is a deferral proof — and how many circuits — is INFERRED
    // from what the caller staged (one `DeferralState` + one `DeferralInput`
    // per circuit); no separate count rides on the request. Require the staged
    // circuits to form a contiguous `0..N` index set with both artifacts
    // present. (The worker independently validates N against the loaded
    // deferral keyset, which is the ultimate source of truth.)
    let num_deferral_circuits = staged.deferral_states.len();
    let deferral_complete = (0..num_deferral_circuits).all(|i| {
        staged.deferral_states.contains_key(&i) && staged.deferral_inputs.contains_key(&i)
    }) && staged.deferral_inputs.len() == num_deferral_circuits;
    if !deferral_complete {
        return abort_proof_with_failure(
            &state,
            &proof_uuid,
            format!(
                "incomplete deferral upload for proof {proof_uuid}: need one DeferralState and \
                 one DeferralInput per circuit at contiguous indices 0..N (got state idx {:?}, \
                 input idx {:?})",
                staged.deferral_states.keys().collect::<Vec<_>>(),
                staged.deferral_inputs.keys().collect::<Vec<_>>(),
            ),
        )
        .await;
    }

    // Deferral is manager-staged only: it can't ride the worker-pre-uploaded
    // (Flow 1) transport.
    if req.input_already_uploaded && num_deferral_circuits > 0 {
        return abort_proof_with_failure(
            &state,
            &proof_uuid,
            "input_already_uploaded=true is incompatible with deferral: stage the deferral \
             artifacts (and the input) on the manager instead"
                .to_string(),
        )
        .await;
    }

    let input_data: Option<Vec<u8>> = if req.input_already_uploaded {
        info!(
            "Proof {} uses worker-pre-uploaded input (Flow 1); manager skips fan-out",
            proof_uuid
        );
        None
    } else {
        match staged.main.take() {
            Some(bytes) => Some(bytes.to_vec()),
            None => {
                return abort_proof_with_failure(
                    &state,
                    &proof_uuid,
                    format!(
                        "no input staged for proof {proof_uuid}: POST the bincode StdIn to \
                         /upload_input/{proof_uuid} before /start_proof (or set \
                         input_already_uploaded=true to pre-upload directly to workers)"
                    ),
                )
                .await;
            }
        }
    };

    // First, upload input to all workers if the manager holds it.
    if let Some(ref data) = input_data {
        // The fan-out begins and ends on this one clock, so the elapsed time is
        // the input-transfer cost with no cross-host skew in it.
        let fanout_started = Instant::now();
        let fanout_bytes = data.len();
        let mut upload_handles = vec![];
        for (worker_id, worker) in &workers {
            let client = state.upload_client.clone(); // Use upload client with longer timeout
            let worker_url = worker.worker_url.clone();
            let proof_uuid_clone = proof_uuid.clone();
            let data_clone = data.clone();
            let wid = *worker_id;

            let handle = tokio::spawn(async move {
                // Retry loop with exponential backoff
                let max_retries = 5;
                let mut retries = 0;
                let initial_delay = std::time::Duration::from_millis(500);

                loop {
                    // proof_uuid rides in the URL path; body is the raw input.
                    let url = format!("{}/upload_input/{}", worker_url, proof_uuid_clone);
                    let body = data_clone.clone();

                    match client
                        .post(&url)
                        .header("Content-Type", "application/octet-stream")
                        .body(body)
                        .send()
                        .await
                    {
                        Ok(resp) => {
                            if resp.status().is_success() {
                                info!(
                                    "Successfully uploaded input to worker {} for proof {}",
                                    wid, proof_uuid_clone
                                );
                                return Ok(wid);
                            }
                            // Non-success status, retry
                            let status = resp.status();
                            retries += 1;
                            if retries >= max_retries {
                                error!(
                                    "Failed to upload input to {} after {} retries: {}",
                                    worker_url, max_retries, status
                                );
                                return Err(format!("Upload to worker {} failed: {}", wid, status));
                            }
                            warn!(
                                "Upload to {} returned {}, retrying ({}/{})",
                                worker_url, status, retries, max_retries
                            );
                        }
                        Err(e) => {
                            retries += 1;
                            if retries >= max_retries {
                                error!(
                                    "Failed to upload input to {} after {} retries: {}",
                                    worker_url, max_retries, e
                                );
                                return Err(format!("Upload to worker {} failed: {}", wid, e));
                            }
                            warn!(
                                "Upload to {} failed: {}, retrying ({}/{})",
                                worker_url, e, retries, max_retries
                            );
                        }
                    }

                    // Exponential backoff with cap at 8 seconds
                    let delay = initial_delay * (1 << retries.min(4));
                    tokio::time::sleep(delay).await;
                }
            });
            upload_handles.push(handle);
        }

        // Wait for all uploads to complete and check for failures
        let mut upload_failures = Vec::new();
        for handle in upload_handles {
            match handle.await {
                Ok(Ok(_)) => {
                    // Upload succeeded
                }
                Ok(Err(e)) => {
                    // Upload failed after retries
                    upload_failures.push(e);
                }
                Err(e) => {
                    // Task panicked or was cancelled
                    upload_failures.push(format!("Upload task failed: {:?}", e));
                }
            }
        }

        info!(
            "Input fan-out complete for proof {}: workers={}, bytes={}, elapsed={}ms",
            proof_uuid,
            workers.len(),
            fanout_bytes,
            fanout_started.elapsed().as_millis()
        );

        // If any uploads failed, abort the proof
        if !upload_failures.is_empty() {
            let error_msg = format!(
                "Failed to upload input to workers: {}",
                upload_failures.join("; ")
            );
            error!(
                "Upload failed to {} workers, aborting proof {}",
                upload_failures.len(),
                proof_uuid
            );

            // Mark proof as failed (queryable via /proof_state).
            // Clone the Arc out of the DashMap Ref so the shard read lock is
            // released before we await on the per-proof Mutex (otherwise the
            // Ref pins the shard lock to this Tokio worker thread across the
            // await — see the dashmap-ref-across-await wedge hazard).
            let proof_state = state.proof_states.get(&proof_uuid).map(|s| s.clone());
            if let Some(proof_state) = proof_state {
                let mut guard = proof_state.lock().await;
                guard.status = ProofStatus::Failed(error_msg.clone());
            }
            state.state_store.remove_proof(&proof_uuid);

            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": error_msg,
                    "proof_uuid": proof_uuid
                })),
            );
        }
    }

    // Fan out ALL `DeferralState`s to every worker in ONE call each (app workers
    // read them into `StdIn.deferrals`), and RETAIN each `DeferralInput` on the
    // manager for the just-in-time push to the final-internal worker. Presence
    // and contiguity (0..N) were already validated above, so the bundle is
    // simply the states in def-idx order. Failures abort the whole proof — a
    // partial fan-out leaves some workers unable to start. The workers read the
    // states from the deterministic staged paths they reconstruct themselves,
    // so no paths ride on `ShardedAppProveRequest`.
    if num_deferral_circuits > 0 {
        let state_bundle: Vec<Vec<u8>> = (0..num_deferral_circuits)
            .map(|i| {
                staged
                    .deferral_states
                    .remove(&i)
                    .unwrap_or_default()
                    .to_vec()
            })
            .collect();

        if let Err(failures) = fanout_deferral_bundle(
            &state.upload_client,
            &workers,
            &proof_uuid,
            "/upload_deferral_state",
            &state_bundle,
        )
        .await
        {
            return abort_proof_with_failure(
                &state,
                &proof_uuid,
                format!("Failed to fan out deferral states: {}", failures.join("; ")),
            )
            .await;
        }
    }

    if num_deferral_circuits > 0 {
        // Record the circuit count so the result handler attaches a
        // `DeferralTailDispatch` to the final InternalProveRequest. No paths are
        // recorded or sent: the tail worker reconstructs each `DeferralInput`
        // path from the deterministic staging convention + its loaded keyset;
        // the manager just pushes the bytes there just-in-time.
        if let Some(proof_state) = state.proof_states.get(&proof_uuid).map(|s| s.clone()) {
            let mut guard = proof_state.lock().await;
            guard.deferral_circuit_count = num_deferral_circuits;
        }
        // Re-stage ONLY the retained `DeferralInput` bytes for the JIT push;
        // `main` + `deferral_states` are already fanned out and dropped.
        state.staged_inputs.insert(
            proof_uuid.clone(),
            StagedInputs {
                deferral_inputs: std::mem::take(&mut staged.deferral_inputs),
                ..Default::default()
            },
        );
    }

    if let Some(reporter) = &state.lifecycle_reporter {
        reporter.report_proving(&proof_uuid, &req.labels);
    }

    // Anchor the proving-only latency clock now: the input is uploaded to every
    // worker and we are about to dispatch work. This excludes the admission +
    // input read + upload fan-out above (the "submitting proof" overhead) from
    // `proving_latency_ms`. Clone the Arc out of the DashMap Ref so the shard
    // read lock is released before we await the per-proof Mutex.
    let proof_state_for_anchor = state.proof_states.get(&proof_uuid).map(|s| s.clone());
    if let Some(proof_state) = proof_state_for_anchor {
        proof_state.lock().await.proving_started_at = Some(chrono::Utc::now());
    }

    // Send the app-prove kickoff to the app-eligible (normal) worker set only.
    // The `EvmDedicated` worker (if any) is excluded here and owns no shard.
    let mut work_handles = Vec::new();
    for (worker_id, worker) in app_workers {
        // No input/deferral paths: the worker reconstructs them from
        // `proof_uuid` (+ its deferral keyset count) and reads the files the
        // manager staged at those deterministic locations.
        let work_request = ShardedAppProveRequest {
            proof_uuid: proof_uuid.clone(),
            program: program.clone(),
            prover_id: worker_id,
            num_provers,
            segment_memory: req.segment_memory,
        };

        let client = state.http_client.clone();
        let worker_url = worker.worker_url.clone();

        let handle = tokio::spawn(async move {
            let url = format!("{}/sharded_app_prove", worker_url);
            let outcome = match client.post(&url).json(&work_request).send().await {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_default();
                        let msg = format!("Worker {} returned {}: {}", worker_id, status, body);
                        error!("{}", msg);
                        Err(msg)
                    } else {
                        info!(
                            "Work sent to worker {} for proof {}: prover_id={}, num_provers={}",
                            worker_id,
                            work_request.proof_uuid,
                            work_request.prover_id,
                            work_request.num_provers
                        );
                        Ok(())
                    }
                }
                Err(e) => {
                    let msg = format!("Failed to send work to worker {}: {}", worker_id, e);
                    error!("{}", msg);
                    Err(msg)
                }
            };
            (worker_id, outcome)
        });
        work_handles.push(handle);
    }

    // Wait for all work dispatches and check for failures.
    let mut work_failures = Vec::new();
    let mut failed_worker_ids: Vec<usize> = Vec::new();
    let mut work_successes = 0;
    for handle in work_handles {
        match handle.await {
            Ok((_, Ok(()))) => work_successes += 1,
            Ok((wid, Err(e))) => {
                failed_worker_ids.push(wid);
                work_failures.push(e);
            }
            Err(e) => work_failures.push(format!("Work dispatch task failed: {:?}", e)),
        }
    }

    // All-or-nothing initial dispatch: any worker failing to accept the
    // sharded_app_prove kickoff aborts the whole proof. A partial dispatch
    // would leave the scheduler waiting forever for segments owned by the
    // failed worker's modulo slice (`segment_idx % num_provers == prover_id`),
    // since worker shards are pinned to `(prover_id, num_provers)` at
    // dispatch time and the manager has no way to replan ownership without
    // a worker-side cancel channel.
    //
    // Workers that already accepted will keep proving their shards; their
    // results arrive as "late results" and drop silently via the existing
    // is_terminal() early-return in the accumulator. Bounded waste.
    if !work_failures.is_empty() {
        let total = work_successes + work_failures.len();
        let base_error = format!(
            "{} of {} workers failed to accept work: {}",
            work_failures.len(),
            total,
            work_failures.join("; ")
        );
        let error_msg = base_error;
        error!("Proof {} aborted: {}", proof_uuid, error_msg);

        // Mark proof as Failing — workers that already accepted /sharded_app_prove
        // keep running their shards. The drain orchestrator transitions
        // Failing → Failed once all workers report completion (or TTL).
        // For workers whose dispatch failed: release their scheduler-side
        // busy slot now so they're not counted against drain progress.
        let proof_state = state.proof_states.get(&proof_uuid).map(|s| s.clone());
        if let Some(proof_state) = proof_state {
            let mut guard = proof_state.lock().await;
            guard.mark_failing(error_msg.clone());
        }
        for &worker_id in &failed_worker_ids {
            state
                .state_store
                .release_worker(&proof_uuid, worker_id)
                .await;
        }
        // Trigger immediate drain check (no workers running for the all-failed
        // case will transition Failing → Failed immediately).
        try_finalize_failing_proof(&state, &proof_uuid).await;

        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": error_msg,
                "proof_uuid": proof_uuid
            })),
        );
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "proof_uuid": proof_uuid,
            "status": "started",
            "num_workers": num_provers
        })),
    )
}

/// Receive proof result from a worker.
pub async fn proof_result(State(state): State<Arc<AppState>>, body: Bytes) -> impl IntoResponse {
    // Deserialize bincode payload
    let payload: ResultPayload = match bincode::deserialize(&body) {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to deserialize ResultPayload: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid bincode payload"})),
            );
        }
    };

    let proof_uuid = payload.proof_uuid.clone();
    let worker_id = payload.worker_id;
    let result = payload.result.message.clone();

    // Validate that the result's context matches the declared proof_uuid
    // This prevents misrouted worker responses from contaminating wrong proof state
    let result_proof_uuid = result.context().proof_uuid.clone();
    if result_proof_uuid != proof_uuid {
        warn!(
            "Result proof_uuid mismatch: payload={}, result_context={}",
            proof_uuid, result_proof_uuid
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({"error": "proof_uuid mismatch between payload and result context"}),
            ),
        );
    }

    // Get proof state
    let proof_state = match state.proof_states.get(&proof_uuid) {
        Some(s) => s.clone(),
        None => {
            warn!("Received result for unknown proof: {}", proof_uuid);
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "Proof not found"})),
            );
        }
    };

    // Process result and classify whether it was fresh, duplicate, or late.
    let outcome = {
        let mut guard = proof_state.lock().await;
        let outcome = match guard.handle_proof_result_with_envelope_outcome(payload.result) {
            Ok(outcome) => outcome,
            Err(e) => {
                error!("Failed to handle proof result: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                );
            }
        };
        // Persist the final proof before releasing the lock. `/proof_state` and
        // `/proof_events` read the status through this same mutex, so a caller
        // that observes `completed` can always download the artifact it names.
        // `finalize_proof` calls this again later and gets the cached path.
        if matches!(guard.status, ProofStatus::Completed) {
            if let Some(dir) = state.config.proof.persist_final_proofs_dir.as_ref() {
                if let Err(e) = guard.persist_final_proof_to_disk(
                    dir,
                    state.config.proof.compress_persisted_final_proofs,
                ) {
                    error!("Failed to persist final proof {}: {}", proof_uuid, e);
                    guard.status = ProofStatus::Failed(format!("persist final proof: {e}"));
                    guard.notify_completion();
                }
            }
        }
        outcome
    };

    let (follow_up_requests, transitioned_to_terminal) = match outcome {
        ProofResultEnvelopeOutcome::Late {
            should_log_notice,
            status,
        } => {
            if should_log_notice {
                debug!(
                    "Proof {} already terminal ({:?}); dropping late worker results. First late result: worker={}, kind={}",
                    proof_uuid,
                    status,
                    worker_id,
                    result.kind()
                );
            }
            // `Failing` is terminal only for result *accumulation* — its
            // workers are still draining. Account this completion against the
            // worker's scheduler slot (without dispatching new work) so
            // `is_fully_drained` can transition Failing → Failed as soon as
            // the last busy worker reports, instead of stalling the single
            // active-proof slot until the drain TTL fires.
            if matches!(status, ProofStatus::Failing(_)) {
                if let Err(e) = state
                    .state_store
                    .worker_drained(&proof_uuid, worker_id, &result)
                    .await
                {
                    warn!(
                        "Drain accounting failed for proof {} worker {}: {}",
                        proof_uuid, worker_id, e
                    );
                }
                try_finalize_failing_proof(&state, &proof_uuid).await;
            }
            evict_stale_proofs(&state.proof_states);
            return (StatusCode::OK, Json(serde_json::json!({"status": "ok"})));
        }
        ProofResultEnvelopeOutcome::Processed {
            follow_up_requests,
            transitioned_to_terminal,
        } => (follow_up_requests, transitioned_to_terminal),
    };

    info!(
        "Received result from worker {}: proof_uuid={}, kind={}",
        worker_id,
        proof_uuid,
        result.kind()
    );

    // Handle ExecuteE2 result specially to set num_segments before scheduling more work.
    if let ProofResult::ExecuteE2(ref e2_result) = result {
        let num_segments = e2_result.state.num_segments;
        if let Err(e) = state
            .state_store
            .set_num_segments(&proof_uuid, num_segments)
            .await
        {
            error!("Failed to set num_segments: {}", e);
        }
    }

    if !transitioned_to_terminal {
        // Mark worker as completed for this result and get any pending work.
        let pending_work = match state
            .state_store
            .worker_completed(&proof_uuid, worker_id, &result)
            .await
        {
            Ok(work) => work,
            Err(e) => {
                warn!("Failed to mark worker {} as completed: {}", worker_id, e);
                None
            }
        };

        // Dispatch any pending work that was waiting for this worker (non-blocking).
        if let Some(work) = pending_work {
            let state_clone = state.clone();
            tokio::spawn(async move {
                send_work_with_retry(&state_clone, work, 3).await;
            });
        }
    }

    // On the first transition to a *drained* terminal state (Failed/Completed/
    // Canceled — not Failing), emit metrics and clean up. For Failing proofs,
    // this fires later on the drain transition via `try_finalize_failing_proof`.
    if transitioned_to_terminal {
        let is_failing = {
            let ps = state.proof_states.get(&proof_uuid).map(|s| s.clone());
            match ps {
                Some(ps) => matches!(ps.lock().await.status, ProofStatus::Failing(_)),
                None => false,
            }
        };
        if !is_failing {
            finalize_proof(&state, &proof_uuid).await;
        }
    }

    // Also opportunistically check whether a previously-Failing proof has now
    // fully drained. Cheap — typically a single DashMap lookup.
    try_finalize_failing_proof(&state, &proof_uuid).await;

    // Evict stale proofs to prevent memory leaks
    evict_stale_proofs(&state.proof_states);

    // Process follow-up requests: enqueue synchronously, dispatch HTTP in background
    for request in follow_up_requests {
        let step = Step::from(&request);
        let envelope = MessageEnvelope::with_metadata(request);

        if let Ok(Some(assigned)) = state
            .state_store
            .enqueue_or_assign(&proof_uuid, envelope, step)
            .await
        {
            let state_clone = state.clone();
            tokio::spawn(async move {
                send_work_with_retry(&state_clone, assigned, 3).await;
            });
        }
    }

    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

/// Get proof state.
pub async fn proof_state(
    State(state): State<Arc<AppState>>,
    Path(proof_uuid): Path<String>,
) -> impl IntoResponse {
    // Clone the Arc out of the Ref before awaiting the per-proof Mutex.
    let proof_state = state.proof_states.get(&proof_uuid).map(|s| s.clone());
    match proof_state {
        Some(proof_state) => {
            let guard = proof_state.lock().await;
            let lightweight = guard.to_lightweight_state();
            (StatusCode::OK, Json(serde_json::json!(lightweight)))
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Proof not found"})),
        ),
    }
}

/// How often `/proof_events` rechecks a proof's status. It reads the status
/// rather than every writer publishing to it, since it is written from a dozen
/// places across the scheduler and the result handler.
const PROOF_EVENT_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);

/// `GET /proof_events/{proof_uuid}` — the proof's status as server-sent events.
///
/// Emits the current status on subscribe, then one event per change, and ends
/// the stream once the status settles. Each event carries only the status, so
/// a subscriber never has to poll `/proof_state`. Subscribing after a change
/// still yields the current status, which makes a reconnect safe.
pub async fn proof_events(
    State(state): State<Arc<AppState>>,
    Path(proof_uuid): Path<String>,
) -> Response {
    let Some(proof) = state.proof_states.get(&proof_uuid).map(|s| s.clone()) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Proof not found"})),
        )
            .into_response();
    };

    Sse::new(status_events(proof))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Emits `proof`'s status on subscribe and again on every change, ending once
/// it settles.
fn status_events(
    proof: Arc<Mutex<ProofState>>,
) -> impl futures::Stream<Item = Result<Event, axum::Error>> {
    async_stream::try_stream! {
        let mut last: Option<ProofStatus> = None;
        loop {
            let current = {
                let guard = proof.lock().await;
                guard.status.clone()
            };
            if last.as_ref() != Some(&current) {
                yield Event::default().event("status").json_data(&current)?;
                if current.is_settled() {
                    break;
                }
                last = Some(current);
            }
            tokio::time::sleep(PROOF_EVENT_POLL_INTERVAL).await;
        }
    }
}

/// Get scheduler debug state for a proof.
///
/// This endpoint is intended for diagnosing stalls. It exposes per-worker
/// segment progress and active work as tracked by the manager scheduler.
pub async fn proof_debug(
    State(state): State<Arc<AppState>>,
    Path(proof_uuid): Path<String>,
) -> impl IntoResponse {
    match state.state_store.proof_debug_state(&proof_uuid).await {
        Some(debug_state) => (StatusCode::OK, Json(serde_json::json!(debug_state))),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Proof not found"})),
        ),
    }
}

/// Cancel proof request.
#[derive(serde::Deserialize)]
pub struct CancelProofRequest {
    pub proof_uuid: String,
}

/// Cancel a proof.
pub async fn cancel_proof(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CancelProofRequest>,
) -> impl IntoResponse {
    if let Err(reason) = validate_manager_proof_uuid(&req.proof_uuid) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("Invalid proof_uuid: {reason}")})),
        );
    }

    info!("Canceling proof: {}", req.proof_uuid);

    // Clone the Arc out of the Ref before awaiting the per-proof Mutex.
    let proof_state = state.proof_states.get(&req.proof_uuid).map(|s| s.clone());
    if let Some(proof_state) = proof_state {
        let mut guard = proof_state.lock().await;
        if matches!(guard.status, ProofStatus::InProgress) {
            guard.status = ProofStatus::Canceled;
            guard.notify_completion();
        }
    }

    // Remove from scheduler state and drop any staged/retained input bytes for
    // this proof (e.g. a deferral proof's retained DeferralInput).
    state.state_store.remove_proof(&req.proof_uuid);
    state.staged_inputs.remove(&req.proof_uuid);

    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "canceled"})),
    )
}

/// Metadata captured under the proof lock for the `completed` lifecycle event,
/// used after the lock is released.
struct CompletedProofMeta {
    labels: std::collections::BTreeMap<String, String>,
    proving_time_ms: Option<u64>,
    proving_cycles: Option<u64>,
}

/// Finalize a proof that has reached a drained terminal state (Failed,
/// Completed, or Canceled — not Failing): persist outputs, emit completion
/// metrics, write the report, fire the `completed` lifecycle event, and free
/// the scheduler slot. Safe to call once a proof has truly drained; idempotent
/// on missing proofs.
async fn finalize_proof(state: &Arc<AppState>, proof_uuid: &str) {
    // Drop any retained staged bytes for this proof (a deferral proof
    // that terminated before its final-internal JIT dispatch would otherwise
    // leak its retained `DeferralInput`). No-op once the JIT push has freed it.
    state.staged_inputs.remove(proof_uuid);

    // The terminal-state persistence below runs bincode + zstd-19 + several
    // synchronous `fs` writes while holding the per-proof lock. Run that whole
    // locked section on a blocking thread (acquiring the lock via
    // `blocking_lock`) so it never stalls an async runtime worker; tasks
    // awaiting this proof's lock (e.g. `/proof_state`) then wait asynchronously
    // instead of the runtime spinning. See proof_state::persistence. The values
    // needed after the lock drops are returned out of the closure.
    //
    // `completed_meta` is only set on a true Completion (not Failed/Canceled)
    // and drives the `completed` lifecycle event fired below.
    let ps = state.proof_states.get(proof_uuid).map(|s| s.clone());
    let (persisted_proof_path, completed_meta, terminal_status_for_log): (
        Option<std::path::PathBuf>,
        Option<CompletedProofMeta>,
        Option<ProofStatus>,
    ) = if let Some(ps) = ps {
        let state = state.clone();
        let proof_uuid = proof_uuid.to_string();
        let join = tokio::task::spawn_blocking(move || {
            let mut guard = ps.blocking_lock();
            let mut persisted_proof_path: Option<std::path::PathBuf> = None;
            let mut completed_meta: Option<CompletedProofMeta> = None;
            let terminal_status_for_log = Some(guard.status.clone());
            if let Some(dir) = state
                .config
                .proof
                .persist_leaf_failure_app_proofs_dir
                .as_ref()
            {
                match guard.persist_leaf_failure_app_proofs_to_disk(dir) {
                    Ok(Some(path)) => {
                        info!(
                            "Persisted leaf-failure app proofs {} to {}",
                            proof_uuid,
                            path.display()
                        );
                    }
                    Ok(None) => {}
                    Err(e) => {
                        warn!(
                            "Failed to persist leaf-failure app proofs {} to {}: {}",
                            proof_uuid,
                            dir.display(),
                            e
                        );
                    }
                }
            }
            if let Some(dir) = state.config.proof.persist_final_proofs_dir.as_ref() {
                match guard.persist_final_proof_to_disk(
                    dir,
                    state.config.proof.compress_persisted_final_proofs,
                ) {
                    Ok(Some(path)) => {
                        info!("Persisted final proof {} to {}", proof_uuid, path.display());
                        persisted_proof_path = Some(path);
                    }
                    Ok(None) => {
                        warn!(
                            "Proof {} completed without a final proof payload to persist",
                            proof_uuid
                        );
                    }
                    Err(e) => {
                        warn!(
                            "Failed to persist final proof {} to {}: {}",
                            proof_uuid,
                            dir.display(),
                            e
                        );
                    }
                }
            }
            if matches!(guard.status, ProofStatus::Completed) {
                completed_meta = Some(CompletedProofMeta {
                    labels: guard.context.labels.clone(),
                    proving_time_ms: guard.e2e_latency_ms,
                    proving_cycles: guard.num_instructions,
                });
            }
            guard.emit_completion_metrics();

            // Write the per-proof Markdown report to the manager-configured
            // output dir.
            let report = guard.generate_metrics_report();
            let metrics_dir = &state.config.metrics.output_dir;
            if let Err(e) = std::fs::create_dir_all(metrics_dir) {
                warn!(
                    "Failed to create metrics dir {}: {}",
                    metrics_dir.display(),
                    e
                );
            } else {
                let path = metrics_dir.join(format!("{}.md", proof_uuid));
                match std::fs::write(&path, &report) {
                    Ok(_) => info!("Wrote metrics report to {}", path.display()),
                    Err(e) => warn!(
                        "Failed to write metrics report to {}: {}",
                        path.display(),
                        e
                    ),
                }
            }

            guard.compact_completed_state();
            (
                persisted_proof_path,
                completed_meta,
                terminal_status_for_log,
            )
        });
        match join.await {
            Ok(values) => values,
            Err(e) => {
                error!("finalize_proof persistence task panicked: {e}");
                (None, None, None)
            }
        }
    } else {
        (None, None, None)
    };

    // Fire the `completed` lifecycle event (only on a true Completion). The
    // reporter spawns its own delivery task, so this doesn't block.
    if let (Some(reporter), Some(meta)) = (&state.lifecycle_reporter, completed_meta) {
        reporter.report_completed(
            proof_uuid,
            &meta.labels,
            meta.proving_time_ms,
            meta.proving_cycles,
            persisted_proof_path.as_deref(),
        );
    }

    info!(
        "Proof {} reached terminal state {:?}, cleaning up scheduler state",
        proof_uuid,
        terminal_status_for_log.unwrap_or(ProofStatus::Canceled)
    );
    state.state_store.remove_proof(proof_uuid);
}

/// TTL for how long a `Failing` proof can wait for workers to drain before
/// the watchdog force-transitions it to `Failed`. Failsafe for workers that
/// crashed or otherwise will never report completion.
const FAILING_DRAIN_TTL: chrono::Duration = chrono::Duration::seconds(60);

/// If `proof_uuid` is `Failing` AND either the scheduler reports it fully
/// drained OR the drain TTL has expired, transition `Failing` → `Failed` and
/// run `finalize_proof`. No-op otherwise.
async fn try_finalize_failing_proof(state: &Arc<AppState>, proof_uuid: &str) {
    // Quick pre-check without locking: is there a state with status Failing
    // and a TTL we can compute against now?
    let (is_failing, ttl_expired) = {
        let ps = state.proof_states.get(proof_uuid).map(|s| s.clone());
        match ps {
            Some(ps) => {
                let guard = ps.lock().await;
                let is_failing = matches!(guard.status, ProofStatus::Failing(_));
                let ttl_expired =
                    is_failing && (chrono::Utc::now() - guard.last_updated) > FAILING_DRAIN_TTL;
                (is_failing, ttl_expired)
            }
            None => (false, false),
        }
    };
    if !is_failing {
        return;
    }
    let drained = state.state_store.is_fully_drained(proof_uuid).await;
    if !drained && !ttl_expired {
        return;
    }
    if ttl_expired && !drained {
        warn!(
            "Failing proof {} forcing terminal via drain TTL; some workers never reported completion",
            proof_uuid
        );
    }

    // Transition Failing -> Failed.
    let ps = state.proof_states.get(proof_uuid).map(|s| s.clone());
    if let Some(ps) = ps {
        let mut guard = ps.lock().await;
        guard.transition_failing_to_failed();
    }

    finalize_proof(state, proof_uuid).await;
}

/// How often the timeout watchdog wakes to scan for stuck proofs.
pub const PROOF_TIMEOUT_WATCHDOG_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

/// Background task: every [`PROOF_TIMEOUT_WATCHDOG_INTERVAL`], walk every
/// in-progress proof and mark any whose wall-clock age exceeds its
/// `timeout_secs` as `Failed("timed out…")`. Frees the proof's scheduler
/// slot so subsequent proofs can claim those workers.
///
/// In-flight worker computation is not canceled (no manager→worker cancel
/// channel today); workers finish their assigned shards and their results
/// arrive as "late results" to be silently dropped.
pub async fn proof_timeout_watchdog_task(
    state: Arc<AppState>,
    interval: std::time::Duration,
    cancel_token: tokio_util::sync::CancellationToken,
) {
    let mut ticker = tokio::time::interval(interval);
    // Skip the first immediate tick so we don't fire during manager startup.
    ticker.tick().await;
    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                info!("Proof timeout watchdog cancelled");
                break;
            }
            _ = ticker.tick() => {
                scan_and_timeout_proofs(state.clone()).await;
            }
        }
    }
}

/// One pass of the watchdog: snapshot which proofs are currently in-progress,
/// then await on each one's Mutex outside the DashMap iteration to avoid
/// pinning shard locks across awaits.
async fn scan_and_timeout_proofs(state: Arc<AppState>) {
    let now = chrono::Utc::now();

    // Phase 1 (sync): collect (uuid, Arc) pairs. Refs are dropped at the
    // end of this expression, so no shard locks survive into the await
    // below.
    let candidates: Vec<(String, Arc<Mutex<ProofState>>)> = state
        .proof_states
        .iter()
        .map(|entry| (entry.key().clone(), entry.value().clone()))
        .collect();

    // Phase 2 (async): per-proof lock + check + mark.
    for (uuid, proof_state) in candidates {
        // First: if the proof has been Failing for too long, transition it.
        // This is the TTL backstop in case the result-arrival path never sees
        // a final completion event (worker crash, etc.).
        try_finalize_failing_proof(&state, &uuid).await;

        let mut guard = proof_state.lock().await;
        if !guard.is_timed_out(now) {
            continue;
        }
        warn!(
            "Proof {} timed out after {}s; marking Failing and freeing scheduler slot once workers drain",
            uuid, guard.timeout_secs
        );
        guard.mark_timed_out();
        drop(guard);

        // Try immediate drain check (handles the case where all workers were
        // already idle at the moment of the timeout — common during early-
        // phase wedge scenarios).
        try_finalize_failing_proof(&state, &uuid).await;
    }

    // Reclaim orphaned staged inputs: an `/upload_input` that no `/start_proof`
    // ever followed. Bytes for a live proof are protected by its `proof_states`
    // entry (and freed by start_proof / finalize / abort / cancel / the JIT
    // push), so we only sweep entries with no such entry that have aged past
    // the TTL. `staged_at` is `None` only for re-staged retained deferral
    // inputs, which always have a `proof_states` entry — so they're skipped.
    let orphan_cutoff = now - chrono::Duration::seconds(ORPHANED_STAGED_INPUT_TTL_SECS);
    let orphans: Vec<String> = state
        .staged_inputs
        .iter()
        .filter(|e| {
            !state.proof_states.contains_key(e.key())
                && e.value().staged_at.is_some_and(|t| t < orphan_cutoff)
        })
        .map(|e| e.key().clone())
        .collect();
    for uuid in orphans {
        warn!(
            "Reclaiming orphaned staged input for proof {} (uploaded but no start_proof within {}s)",
            uuid, ORPHANED_STAGED_INPUT_TTL_SECS
        );
        state.staged_inputs.remove(&uuid);
    }
}

/// How long a manager-staged input may sit without a `/start_proof` before the
/// watchdog reclaims it as orphaned.
const ORPHANED_STAGED_INPUT_TTL_SECS: i64 = 300;

/// Evict stale proof states to prevent memory leaks.
/// Called periodically during result processing.
fn evict_stale_proofs(proof_states: &DashMap<String, Arc<Mutex<ProofState>>>) {
    let now = chrono::Utc::now();
    let mut to_evict = Vec::new();

    // Collect UUIDs to evict (can't mutate while iterating)
    for entry in proof_states.iter() {
        let uuid = entry.key().clone();
        // Try to lock - if we can't, skip this one
        if let Ok(guard) = entry.value().try_lock() {
            if guard.should_evict(now) {
                to_evict.push(uuid);
            }
        }
    }

    // Actually evict
    for uuid in to_evict {
        info!("Evicting stale proof state: {}", uuid);
        proof_states.remove(&uuid);
    }
}

/// Send work to a worker. Returns true if successful, false if failed.
async fn send_work_to_worker(client: &reqwest::Client, work: &AssignedWork) -> bool {
    let url = format!("{}/recursion_prove", work.worker_url);
    let body = match bincode::serialize(&work.envelope) {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to serialize work: {}", e);
            return false;
        }
    };

    // The envelope timestamp is stamped when the result handler built this
    // request, so the gap to now is how long the task waited on the manager for
    // a free worker. The task descriptor repeats the identity the worker puts
    // on its own span, which is what lets the two logs be joined.
    let queue_wait_ms = current_timestamp().saturating_sub(work.envelope.timestamp);
    let task = match &work.envelope.message {
        GeneralProveRequest::LeafProve(req) => format!(
            "segments=[{}, {}], children={}",
            req.segment_start,
            req.segment_end,
            req.app_proofs.len()
        ),
        GeneralProveRequest::InternalProve(req) => format!(
            "layer={}, segments=[{}, {}], children={}, is_final={}",
            req.layer_idx,
            req.segment_start,
            req.segment_end,
            req.child_proofs.len(),
            req.is_final_proof
        ),
        GeneralProveRequest::EvmProve(req) => {
            format!("has_deferral={}", req.proof_has_deferral)
        }
    };

    info!(
        "Sending {} work to worker {} for proof {}: {}, queue_wait={}ms",
        work.step.as_str(),
        work.worker_id,
        work.proof_uuid,
        task,
        queue_wait_ms
    );

    match client
        .post(&url)
        .header("Content-Type", "application/octet-stream")
        .body(body)
        .send()
        .await
    {
        Ok(resp) => {
            if !resp.status().is_success() {
                error!(
                    "Worker {} returned error for {}: {}",
                    work.worker_id,
                    work.step.as_str(),
                    resp.status()
                );
                false
            } else {
                true
            }
        }
        Err(e) => {
            error!("Failed to send work to worker {}: {}", work.worker_id, e);
            false
        }
    }
}

/// Just-in-time push of retained `DeferralInput` bytes to the worker assigned
/// the final internal prove of a deferral proof.
///
/// The manager holds each `DeferralInput` (rather than broadcasting it to all
/// workers at proof start); pushing it here — right before the final internal
/// dispatch — guarantees the file is present, on exactly the one worker that
/// runs the deferral tail merge, by the time that merge runs. Idempotent
/// (overwrites the worker's staged file), so it is safe to call on every send
/// attempt. A push failure is logged and surfaces later as a tail-merge
/// "input not found" error on the worker.
///
/// No-op for any request that is not the final internal prove of a deferral
/// job (only that request carries a `deferral_tail`). Returns `true` iff `work`
/// is that deferral-tail dispatch, so the caller frees the retained bytes only
/// after *that* send succeeds — not after some earlier leaf/internal send for
/// the same proof.
/// Returns `None` if `work` is not a deferral-tail dispatch (nothing to push);
/// `Some(true)` if the retained `DeferralInput` was pushed to the worker (or
/// there was nothing to push); `Some(false)` if the push failed and the caller
/// should retry WITHOUT dispatching or freeing the retained bytes.
async fn push_deferral_inputs_jit(state: &Arc<AppState>, work: &AssignedWork) -> Option<bool> {
    let is_deferral_tail = matches!(
        &work.envelope.message,
        GeneralProveRequest::InternalProve(req) if req.deferral_tail.is_some()
    );
    if !is_deferral_tail {
        return None;
    }

    // Snapshot the retained bytes into an index-ordered bundle without holding
    // the DashMap ref across an await (Bytes clone is a refcount bump). The
    // BTreeMap iterates in key (circuit-index) order.
    let bundle: Vec<Vec<u8>> = match state.staged_inputs.get(&work.proof_uuid) {
        Some(entry) => entry
            .deferral_inputs
            .values()
            .map(|bytes| bytes.to_vec())
            .collect(),
        None => {
            // Nothing retained — the proof is typically already terminal
            // (finalize/cancel removed it). Nothing to deliver; proceed.
            warn!(
                "Final internal for proof {} has no retained DeferralInput staged; tail merge \
                 on worker {} will fail if it needs one",
                work.proof_uuid, work.worker_id
            );
            return Some(true);
        }
    };

    // One request carries all circuits' DeferralInput; the worker validates the
    // count against its loaded keyset.
    let body = match bincode::serialize(&bundle) {
        Ok(b) => b,
        Err(e) => {
            warn!(
                "Failed to serialize DeferralInput bundle for proof {}: {}",
                work.proof_uuid, e
            );
            return Some(false);
        }
    };
    let url = format!(
        "{}/upload_deferral_input/{}",
        work.worker_url, work.proof_uuid
    );
    match state
        .upload_client
        .post(&url)
        .header("Content-Type", "application/octet-stream")
        .body(body)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            info!(
                "Pushed {} DeferralInput circuit(s) to final-internal worker {} for proof {}",
                bundle.len(),
                work.worker_id,
                work.proof_uuid
            );
            Some(true)
        }
        Ok(resp) => {
            warn!(
                "Push of DeferralInput bundle to worker {} for proof {} returned {} — will retry",
                work.worker_id,
                work.proof_uuid,
                resp.status()
            );
            Some(false)
        }
        Err(e) => {
            warn!(
                "Push of DeferralInput bundle to worker {} for proof {} failed: {} — will retry",
                work.worker_id, work.proof_uuid, e
            );
            Some(false)
        }
    }
}

/// Send work to a worker with retry on failure. If all retries fail, the
/// proof is failed immediately: workers share one private network with the
/// manager, so a worker that stays unreachable past the retry budget is a
/// deployment problem, not a transient to route around. (Same policy as the
/// all-or-nothing `/sharded_app_prove` kickoff.)
async fn send_work_with_retry(state: &Arc<AppState>, work: AssignedWork, max_retries: usize) {
    let client = &state.http_client;
    let mut retries = 0;
    let initial_delay = std::time::Duration::from_millis(100);

    loop {
        // For a deferral job's final internal prove, push the retained
        // DeferralInput to the assigned worker BEFORE the prove request. If that
        // push fails, do NOT dispatch and do NOT free the retained bytes — fall
        // through to the retry so a transient upload failure can't become an
        // unrecoverable tail-merge failure.
        let push = push_deferral_inputs_jit(state, &work).await;
        let push_ok = push != Some(false);
        let is_deferral_tail = matches!(push, Some(true));

        if push_ok && send_work_to_worker(client, &work).await {
            // Both the (optional) DeferralInput push and the dispatch succeeded.
            // Free the retained bytes only now, and only for the deferral-tail
            // dispatch they belong to — earlier leaf/internal sends for the same
            // proof must not drop them.
            if is_deferral_tail {
                state.staged_inputs.remove(&work.proof_uuid);
            }
            return; // Success
        }

        retries += 1;
        if retries >= max_retries {
            // Max retries reached — fail the proof now instead of waiting for
            // the timeout watchdog. Other workers' in-flight results drain as
            // Late results via the existing accounting; the retained
            // DeferralInput (if any) is cleaned up when the proof finalizes.
            let reason = format!(
                "Failed to dispatch {} to worker {} after {} retries",
                work.step.as_str(),
                work.worker_id,
                max_retries
            );
            error!("{reason}; marking proof {} as failing", work.proof_uuid);

            // The assigned worker never received the work, so it isn't doing
            // anything for us: free its scheduler slot so drain progress
            // doesn't wait on a result that will never arrive.
            state
                .state_store
                .release_worker(&work.proof_uuid, work.worker_id)
                .await;

            // Bind the Arc out of the DashMap ref before awaiting the
            // per-proof lock (holding a `Ref` across `.await` is unsound).
            let ps = state.proof_states.get(&work.proof_uuid).map(|s| s.clone());
            if let Some(ps) = ps {
                let mut guard = ps.lock().await;
                if matches!(guard.status, ProofStatus::InProgress) {
                    guard.status = ProofStatus::Failing(reason);
                    guard.notify_completion();
                }
            }
            try_finalize_failing_proof(state, &work.proof_uuid).await;
            return;
        }

        // Exponential backoff with jitter
        let delay = initial_delay * (1 << retries.min(5));
        tokio::time::sleep(delay).await;
    }
}

/// Helper: POST a whole deferral-artifact bundle (all circuits of one kind) to
/// every worker in one request each, mirroring the `/upload_input` retry loop.
/// `bundle[i]` is circuit `i`'s bytes; the body is a bincode `Vec<Vec<u8>>` and
/// the worker validates the count against its loaded keyset.
async fn fanout_deferral_bundle(
    client: &reqwest::Client,
    workers: &[(usize, RegisteredWorker)],
    proof_uuid: &str,
    endpoint: &'static str,
    bundle: &[Vec<u8>],
) -> Result<(), Vec<String>> {
    let body = match bincode::serialize(bundle) {
        Ok(b) => b,
        Err(e) => return Err(vec![format!("failed to serialize deferral bundle: {e}")]),
    };
    let mut handles = Vec::with_capacity(workers.len());
    for (worker_id, worker) in workers {
        let client = client.clone();
        let worker_url = worker.worker_url.clone();
        let proof_uuid = proof_uuid.to_string();
        let body = body.clone();
        let wid = *worker_id;

        handles.push(tokio::spawn(async move {
            let max_retries = 5;
            let mut retries = 0;
            let initial_delay = std::time::Duration::from_millis(500);

            loop {
                // proof_uuid rides in the URL path; body is the bincode bundle.
                let url = format!("{}{}/{}", worker_url, endpoint, proof_uuid);

                match client
                    .post(&url)
                    .header("Content-Type", "application/octet-stream")
                    .body(body.clone())
                    .send()
                    .await
                {
                    Ok(resp) if resp.status().is_success() => {
                        info!(
                            "Uploaded deferral bundle (endpoint {}) to worker {}",
                            endpoint, wid
                        );
                        return Ok(wid);
                    }
                    Ok(resp) => {
                        let status = resp.status();
                        retries += 1;
                        if retries >= max_retries {
                            error!(
                                "Failed deferral upload to {} after {} retries: {}",
                                worker_url, max_retries, status
                            );
                            return Err(format!("Upload to worker {} failed: {}", wid, status));
                        }
                        warn!(
                            "Deferral upload to {} returned {}, retrying ({}/{})",
                            worker_url, status, retries, max_retries
                        );
                    }
                    Err(e) => {
                        retries += 1;
                        if retries >= max_retries {
                            error!(
                                "Failed deferral upload to {} after {} retries: {}",
                                worker_url, max_retries, e
                            );
                            return Err(format!("Upload to worker {} failed: {}", wid, e));
                        }
                        warn!(
                            "Deferral upload to {} failed: {}, retrying ({}/{})",
                            worker_url, e, retries, max_retries
                        );
                    }
                }
                let delay = initial_delay * (1 << retries.min(4));
                tokio::time::sleep(delay).await;
            }
        }));
    }

    let mut failures = Vec::new();
    for handle in handles {
        match handle.await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => failures.push(e),
            Err(e) => failures.push(format!("Upload task failed: {:?}", e)),
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

/// Helper: mark a proof Failed, clean up scheduler state, and return the
/// JSON 500 response. Used when manager-side fan-out (deferral upload, etc.)
/// fails before the sharded_app_prove dispatch.
async fn abort_proof_with_failure(
    state: &Arc<AppState>,
    proof_uuid: &str,
    error_msg: String,
) -> (StatusCode, Json<serde_json::Value>) {
    error!(
        "Upload failed for proof {}, aborting: {}",
        proof_uuid, error_msg
    );

    let proof_state = state.proof_states.get(proof_uuid).map(|s| s.clone());
    if let Some(proof_state) = proof_state {
        let mut guard = proof_state.lock().await;
        guard.status = ProofStatus::Failed(error_msg.clone());
    }
    state.state_store.remove_proof(proof_uuid);
    // Drop any staged bytes for this proof.
    state.staged_inputs.remove(proof_uuid);

    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({
            "error": error_msg,
            "proof_uuid": proof_uuid,
        })),
    )
}
