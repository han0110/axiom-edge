//! HTTP endpoint handlers for the Edge worker.

use axum::{
    body::Bytes,
    extract::{Path as UrlPath, State},
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
};
use dashmap::DashSet;
use sdk_v2::StdIn;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, instrument, warn};

use protocol::{
    GeneralProveRequest, MessageEnvelope, RegisterProgramRequest, ShardedAppProveRequest,
};

use crate::prover_pool::{JobType, ProverPool};
use crate::provers::{InternalProverJob, LeafProverJob, ProverResult, ShardedAppProverJob};
use crate::registration::RegistrationResult;
use crate::result_client::ResultClient;

/// Shared application state.
pub struct AppState {
    pub prover_pool: ProverPool,
    pub result_client: ResultClient,
    pub worker_config: WorkerInfo,
    pub active_uploaded_proofs: DashSet<String>,
}

const UPLOADED_INPUT_ROOT: &str = "/dev/shm";
const MAX_PROOF_UUID_LEN: usize = 256;
/// How long a staged `/dev/shm` input dir may live, un-referenced by an active
/// proof, before the janitor reclaims it. This must exceed the longest a staged
/// file is legitimately needed — which spans the whole proof, not just
/// app-prove: a deferral `DeferralInput` is staged at the *final internal*
/// dispatch and read only after that prove completes. So the TTL is set well
/// above the max proof lifetime; it's a leak backstop for orphaned dirs
/// (crashed/abandoned proofs), NOT a tight reaper — proofs that finish normally
/// delete their own dirs immediately regardless of this value. Too-short a TTL
/// races the fan-out / final-internal windows and deletes inputs still in use.
pub const STALE_UPLOADED_INPUT_TTL: Duration = Duration::from_secs(600);
pub const STALE_UPLOADED_INPUT_JANITOR_INTERVAL: Duration = Duration::from_secs(60);

/// Worker information for work requests.
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub prover_id: usize,
    pub num_provers: usize,
    /// Number of GPU app prover instances (also the per-proof app-prove parallelism).
    pub max_app_provers: usize,
    /// Default VM max memory applied when a prove request does not set `segment_memory`.
    pub default_segment_memory: Option<usize>,
    /// Deployment role this worker plays. Drives role-aware readiness: an
    /// `EvmDedicated` worker additionally requires EVM artifacts to be ready,
    /// while `Full`/`StarkOnly` are never gated on them (today's behavior).
    pub worker_role: protocol::WorkerRole,
}

/// Health check response.
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub app_workers_busy: usize,
    pub leaf_workers_busy: usize,
    pub internal_workers_busy: usize,
    /// Total app jobs dispatched since process start.
    pub app_jobs_total: u64,
    /// Total GPU prover swaps since process start. Used to decide whether
    /// program-affinity dispatch is worth implementing — high swap rate
    /// means a sticky dispatcher would eliminate avoidable GPU churn.
    pub app_swaps_total: u64,
}

/// Readiness check response.
#[derive(Serialize)]
pub struct ReadyResponse {
    pub ready: bool,
    pub message: String,
    /// Programs this worker can prove right now. The manager checks the one it
    /// is about to dispatch against this, since a worker can be ready in
    /// general while missing a program whose push never reached it.
    pub programs: Vec<protocol::ProgramRef>,
}

/// Health check endpoint.
pub async fn healthz(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        app_workers_busy: state.prover_pool.busy_count(JobType::ShardedApp),
        leaf_workers_busy: state.prover_pool.busy_count(JobType::Leaf),
        internal_workers_busy: state.prover_pool.busy_count(JobType::Internal),
        app_jobs_total: state.prover_pool.app_jobs_total(),
        app_swaps_total: state.prover_pool.app_swaps_total(),
    })
}

/// What a readiness evaluation resolved to. Kept separate from the HTTP
/// response so [`evaluate_readiness`] stays a pure function unit tests can
/// drive without an artifact store or prover pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Readiness {
    Ready,
    ArtifactsNotLoaded,
    /// EVM artifacts (halo2 key) not loaded — only reachable for `EvmDedicated`.
    EvmArtifactsNotLoaded,
    ProversInitializing,
}

/// Decide readiness from the worker's role and component state.
///
/// `Full` (default) and `StarkOnly` reproduce today's gate exactly: ready once
/// the stark artifacts and provers are up, and **never** gated on EVM
/// artifacts (a stark-only deployment stays ready). `EvmDedicated` — whose
/// entire job is the EVM step (root → halo2) — additionally requires the EVM
/// artifacts (halo2 key) to be loaded. The artifacts→(evm)→provers ordering
/// preserves the original not-ready message precedence for Full/StarkOnly.
fn evaluate_readiness(
    role: protocol::WorkerRole,
    artifacts_ready: bool,
    provers_ready: bool,
    evm_artifacts_ready: bool,
) -> Readiness {
    if !artifacts_ready {
        return Readiness::ArtifactsNotLoaded;
    }
    if role == protocol::WorkerRole::EvmDedicated && !evm_artifacts_ready {
        return Readiness::EvmArtifactsNotLoaded;
    }
    if !provers_ready {
        return Readiness::ProversInitializing;
    }
    Readiness::Ready
}

/// Readiness check endpoint.
///
/// `Full`/`StarkOnly` are ready once artifacts are loaded AND all provers are
/// initialized (unchanged). An `EvmDedicated` worker additionally requires the
/// EVM artifacts (halo2 key) to be loaded, since it serves only the EVM step
/// (root → halo2).
pub async fn readyz(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let artifacts_ready = crate::artifacts::ArtifactStore::global()
        .map(|s| s.is_ready())
        .unwrap_or(false);

    let provers_ready = state.prover_pool.all_provers_initialized();

    // EVM-artifact presence is only a concept in an evm-prove, non-mock build.
    // Elsewhere the halo2 key doesn't exist to check, so treat it as "ready"
    // and let `EvmDedicated` fall back to today's artifacts+provers gate — a
    // dedicated worker only exists in an evm-prove deployment anyway. `Full`/
    // `StarkOnly` never consult this value.
    #[cfg(all(feature = "evm-prove", not(feature = "mock-provers")))]
    let evm_artifacts_ready = crate::artifacts::ArtifactStore::global()
        .and_then(|s| s.get_edge_artifacts().map(|a| a.evm.is_some()))
        .unwrap_or(false);
    #[cfg(not(all(feature = "evm-prove", not(feature = "mock-provers"))))]
    let evm_artifacts_ready = true;

    let programs = crate::artifacts::ArtifactStore::global()
        .map(|s| s.configured_programs())
        .unwrap_or_default();

    let (status, ready, message) = match evaluate_readiness(
        state.worker_config.worker_role,
        artifacts_ready,
        provers_ready,
        evm_artifacts_ready,
    ) {
        Readiness::Ready => (StatusCode::OK, true, "Worker is ready".to_string()),
        Readiness::ArtifactsNotLoaded => (
            StatusCode::SERVICE_UNAVAILABLE,
            false,
            "Artifacts not loaded".to_string(),
        ),
        Readiness::EvmArtifactsNotLoaded => (
            StatusCode::SERVICE_UNAVAILABLE,
            false,
            "EVM artifacts (halo2 key) not loaded".to_string(),
        ),
        Readiness::ProversInitializing => (
            StatusCode::SERVICE_UNAVAILABLE,
            false,
            format!(
                "Provers initializing: app={}/{}, leaf={}/{}, internal={}/{}",
                state.prover_pool.initialized_count(JobType::ShardedApp),
                state.prover_pool.configured_count(JobType::ShardedApp),
                state.prover_pool.initialized_count(JobType::Leaf),
                state.prover_pool.configured_count(JobType::Leaf),
                state.prover_pool.initialized_count(JobType::Internal),
                state.prover_pool.configured_count(JobType::Internal),
            ),
        ),
    };

    (
        status,
        Json(ReadyResponse {
            ready,
            message,
            programs,
        }),
    )
}

/// Handle `POST /cancel_proof/{proof_uuid}`.
///
/// Records the cancellation so the proving loops stop at their next segment.
/// Jobs for the proof that have not started are unaffected, since the manager
/// stops dispatching them as soon as it cancels.
pub async fn handle_cancel_proof(UrlPath(proof_uuid): UrlPath<String>) -> impl IntoResponse {
    if let Err(reason) = validate_uploaded_proof_uuid(&proof_uuid) {
        error!("Invalid proof_uuid: {}", reason);
        return (
            StatusCode::BAD_REQUEST,
            format!("Invalid proof_uuid: {}", reason),
        );
    }

    info!("Canceling {}", proof_uuid);
    crate::cancellation::cancel(proof_uuid);
    (StatusCode::OK, "Canceled".to_string())
}

/// Handle `POST /register_program`, whose body is a bincode
/// [`RegisterProgramRequest`].
///
/// An accepting response body is the program's bincode verification baseline,
/// or empty when this build derived none. The AOT compile and GPU prover
/// preload continue after the response, so `/readyz` reports when the program
/// is actually servable.
pub async fn handle_register_program(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let request: RegisterProgramRequest = match bincode::deserialize(&body) {
        Ok(request) => request,
        Err(e) => {
            error!("Failed to deserialize register_program request: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                format!("Failed to deserialize register_program request: {}", e),
            )
                .into_response();
        }
    };

    let program = request.program.clone();
    info!(
        "Received register_program for {} ({} ELF bytes)",
        program,
        request.elf.len()
    );

    let baseline = match crate::registration::register_program(state, request).await {
        RegistrationResult::Accepted(baseline)
        | RegistrationResult::AlreadyRegistered(baseline) => baseline,
        RegistrationResult::Conflict(reason) => {
            warn!("Rejected registration of {}: {}", program, reason);
            return (StatusCode::CONFLICT, reason).into_response();
        }
        RegistrationResult::Invalid(reason) => {
            error!("Invalid registration of {}: {}", program, reason);
            return (StatusCode::BAD_REQUEST, reason).into_response();
        }
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        baseline.unwrap_or_default(),
    )
        .into_response()
}

/// Handle `POST /upload_input/{proof_uuid}` — body is the raw bincode `StdIn`.
pub async fn handle_upload_input(
    State(state): State<Arc<AppState>>,
    UrlPath(proof_uuid): UrlPath<String>,
    body: Bytes,
) -> impl IntoResponse {
    handle_upload_input_impl(state, proof_uuid, body, UploadedInputFormat::BincodeStdin).await
}

/// Handle `POST /upload_input_compact/{proof_uuid}` — body is the raw compact
/// guest bytes (the worker wraps them into a `StdIn`).
pub async fn handle_upload_input_compact(
    State(state): State<Arc<AppState>>,
    UrlPath(proof_uuid): UrlPath<String>,
    body: Bytes,
) -> impl IntoResponse {
    handle_upload_input_impl(state, proof_uuid, body, UploadedInputFormat::CompactBytes).await
}

/// Handle `POST /upload_deferral_state/{proof_uuid}` — ALL circuits' caller-
/// derived `DeferralState` in one call (opaque to edge; staged for app-worker
/// execution). Body is bincode `Vec<Vec<u8>>`, index = circuit. The worker
/// validates the count against its loaded deferral keyset and writes each to
/// `deferral_state_{i}.bin`.
pub async fn handle_upload_deferral_state(
    State(state): State<Arc<AppState>>,
    UrlPath(proof_uuid): UrlPath<String>,
    body: Bytes,
) -> impl IntoResponse {
    handle_upload_deferral_bundle(state, proof_uuid, body, DeferralArtifactKind::State).await
}

/// Handle `POST /upload_deferral_input/{proof_uuid}` — ALL circuits' caller-
/// derived `DeferralInput` in one call (opaque to edge; staged for the tail
/// worker). Body is bincode `Vec<Vec<u8>>`, index = circuit. The worker
/// validates the count against its loaded deferral keyset and writes each to
/// `deferral_input_{i}.bin`.
pub async fn handle_upload_deferral_input(
    State(state): State<Arc<AppState>>,
    UrlPath(proof_uuid): UrlPath<String>,
    body: Bytes,
) -> impl IntoResponse {
    handle_upload_deferral_bundle(state, proof_uuid, body, DeferralArtifactKind::Input).await
}

#[derive(Debug, Clone, Copy)]
enum UploadedInputFormat {
    BincodeStdin,
    CompactBytes,
}

impl UploadedInputFormat {
    fn label(self) -> &'static str {
        match self {
            Self::BincodeStdin => "upload_input",
            Self::CompactBytes => "upload_input_compact",
        }
    }
}

fn validate_uploaded_proof_uuid(proof_uuid: &str) -> Result<(), &'static str> {
    if proof_uuid.is_empty() {
        return Err("must not be empty");
    }

    if proof_uuid.len() > MAX_PROOF_UUID_LEN {
        return Err("too long");
    }

    if !proof_uuid
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err("contains invalid characters");
    }

    Ok(())
}

async fn handle_upload_input_impl(
    state: Arc<AppState>,
    proof_uuid: String,
    body: Bytes,
    format: UploadedInputFormat,
) -> impl IntoResponse {
    // `proof_uuid` is a URL path segment; the body is the raw input bytes.
    if let Err(reason) = validate_uploaded_proof_uuid(&proof_uuid) {
        error!("Invalid proof_uuid: {}", reason);
        return (
            StatusCode::BAD_REQUEST,
            format!("Invalid proof_uuid: {}", reason),
        );
    }

    let input_data = body.as_ref();
    let serialized_input = match prepare_uploaded_input_bytes(format, input_data) {
        Ok(bytes) => bytes,
        Err(e) => {
            error!(
                "Failed to prepare {} payload for proof {}: {}",
                format.label(),
                proof_uuid,
                e
            );
            return (StatusCode::BAD_REQUEST, format!("Invalid input: {}", e));
        }
    };

    info!(
        "Received {} for proof {}: raw={} bytes serialized={} bytes",
        format.label(),
        proof_uuid,
        input_data.len(),
        serialized_input.len()
    );

    let input_root = uploaded_input_root();
    let input_path = match write_uploaded_input_file(input_root, &proof_uuid, &serialized_input)
        .await
    {
        Ok(input_path) => input_path,
        Err(e) if is_no_space_error(&e) => {
            warn!(
                "Worker /dev/shm is full while uploading proof {}; cleaning stale inputs before retry",
                proof_uuid
            );
            let removed = cleanup_stale_uploaded_inputs(
                input_root,
                &state.active_uploaded_proofs,
                STALE_UPLOADED_INPUT_TTL,
            )
            .await;
            if removed > 0 {
                info!(
                    "Removed {} stale uploaded input directories before retrying proof {}",
                    removed, proof_uuid
                );
            }
            match write_uploaded_input_file(input_root, &proof_uuid, &serialized_input).await {
                Ok(input_path) => input_path,
                Err(retry_err) => {
                    error!(
                        "Failed to write input file for proof {} after stale cleanup retry: {}",
                        proof_uuid, retry_err
                    );
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to write input: {}", retry_err),
                    );
                }
            }
        }
        Err(e) => {
            error!("Failed to write input file for proof {}: {}", proof_uuid, e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to write input: {}", e),
            );
        }
    };

    info!("Input file written to {:?}", input_path);
    (StatusCode::OK, "Input file received".to_string())
}

fn prepare_uploaded_input_bytes(
    format: UploadedInputFormat,
    input_data: &[u8],
) -> eyre::Result<Vec<u8>> {
    match format {
        UploadedInputFormat::BincodeStdin => Ok(input_data.to_vec()),
        UploadedInputFormat::CompactBytes => {
            let mut stdin: StdIn<sdk_v2::F> = StdIn::default();
            stdin.write_bytes(input_data);
            Ok(bincode::serialize(&stdin)?)
        }
    }
}

fn uploaded_input_root() -> &'static Path {
    Path::new(UPLOADED_INPUT_ROOT)
}

fn uploaded_input_work_dir(root: &Path, proof_uuid: &str) -> PathBuf {
    root.join(format!("edge_{}", proof_uuid))
}

/// Deferral artifact filename inside the per-proof work dir.
///
/// State and input artifacts are staged side-by-side under
/// `/dev/shm/edge_{proof_uuid}/` so the existing cleanup of that
/// directory (after the app-prove phase) also reclaims them.
pub fn deferral_state_filename(circuit_idx: usize) -> String {
    format!("deferral_state_{}.bin", circuit_idx)
}

pub fn deferral_input_filename(circuit_idx: usize) -> String {
    format!("deferral_input_{}.bin", circuit_idx)
}

/// Resolve the absolute, worker-visible path for a proof's staged main input.
/// Deterministic from `proof_uuid` — the worker reads from here whether the
/// manager fanned the input out (Flow 2) or a producer pushed it directly
/// (Flow 1); no path is carried on `ShardedAppProveRequest`.
pub fn staged_input_path(proof_uuid: &str) -> PathBuf {
    uploaded_input_work_dir(uploaded_input_root(), proof_uuid).join("input.bin")
}

/// Resolve the absolute, worker-visible path for a staged deferral state.
pub fn staged_deferral_state_path(proof_uuid: &str, circuit_idx: usize) -> PathBuf {
    uploaded_input_work_dir(uploaded_input_root(), proof_uuid)
        .join(deferral_state_filename(circuit_idx))
}

/// Resolve the absolute, worker-visible path for a staged deferral input.
/// The tail worker resolves this in `run_deferral_tail_merge` when consuming
/// the JIT-pushed `DeferralInput`s.
pub fn staged_deferral_input_path(proof_uuid: &str, circuit_idx: usize) -> PathBuf {
    uploaded_input_work_dir(uploaded_input_root(), proof_uuid)
        .join(deferral_input_filename(circuit_idx))
}

#[derive(Debug, Clone, Copy)]
enum DeferralArtifactKind {
    State,
    Input,
}

impl DeferralArtifactKind {
    fn label(self) -> &'static str {
        match self {
            Self::State => "upload_deferral_state",
            Self::Input => "upload_deferral_input",
        }
    }

    fn filename(self, circuit_idx: usize) -> String {
        match self {
            Self::State => deferral_state_filename(circuit_idx),
            Self::Input => deferral_input_filename(circuit_idx),
        }
    }
}

/// Number of deferral circuits this worker's loaded keyset expects; `0` means
/// this isn't a deferral deployment (no keyset loaded) — equivalently, zero
/// circuits. This is the authoritative circuit count — the same source
/// `run_deferral_tail_merge` uses — so the upload handler validates the caller's
/// bundle length against it.
#[cfg(not(feature = "mock-provers"))]
fn loaded_deferral_circuit_count() -> usize {
    let Some(store) = crate::artifacts::ArtifactStore::global() else {
        return 0;
    };
    let Some(edge) = store.get_edge_artifacts() else {
        return 0;
    };
    let Some(deferral) = edge.deferral.as_ref() else {
        return 0;
    };
    deferral
        .cached_pk
        .app_pk
        .app_vm_pk
        .vm_config
        .deferral
        .as_ref()
        .map(|d| d.circuits.len())
        .unwrap_or(0)
}

/// Mock builds don't load a real deferral keyset (the mock e2e suite never
/// exercises deferral), so there are zero circuits.
#[cfg(feature = "mock-provers")]
fn loaded_deferral_circuit_count() -> usize {
    0
}

/// Stage ALL circuits of one deferral artifact kind for a proof in a single
/// call. Body is bincode `Vec<Vec<u8>>` (index = circuit). Validates the bundle
/// length against the loaded keyset's circuit count before writing anything, so
/// a wrong count fails fast at the authoritative source rather than surfacing
/// later at execution / tail-merge.
async fn handle_upload_deferral_bundle(
    state: Arc<AppState>,
    proof_uuid: String,
    body: Bytes,
    kind: DeferralArtifactKind,
) -> (StatusCode, String) {
    if let Err(reason) = validate_uploaded_proof_uuid(&proof_uuid) {
        error!("Invalid proof_uuid: {}", reason);
        return (
            StatusCode::BAD_REQUEST,
            format!("Invalid proof_uuid: {}", reason),
        );
    }

    // 0 = not a deferral deployment (or a keyset with zero circuits); any
    // deferral upload is then rejected by the count check below.
    let expected = loaded_deferral_circuit_count();

    let artifacts: Vec<Vec<u8>> = match bincode::deserialize(body.as_ref()) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!(
                    "{}: malformed bundle (expected bincode Vec<Vec<u8>>): {e}",
                    kind.label()
                ),
            );
        }
    };

    if artifacts.len() != expected {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "{}: got {} artifact(s) but the loaded deferral keyset expects {}",
                kind.label(),
                artifacts.len(),
                expected
            ),
        );
    }

    info!(
        "Received {} for proof {}: {} circuit(s)",
        kind.label(),
        proof_uuid,
        artifacts.len()
    );

    let input_root = uploaded_input_root();
    let work_dir = uploaded_input_work_dir(input_root, &proof_uuid);
    for (circuit_idx, data) in artifacts.iter().enumerate() {
        let filename = kind.filename(circuit_idx);
        if let Err(e) =
            write_deferral_with_no_space_retry(&state, input_root, &work_dir, &filename, data).await
        {
            error!(
                "Failed to write {} circuit {} for proof {}: {}",
                kind.label(),
                circuit_idx,
                proof_uuid,
                e
            );
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to write deferral artifact: {}", e),
            );
        }
    }

    (StatusCode::OK, "Deferral artifacts received".to_string())
}

/// Write one deferral artifact file, retrying once after reclaiming stale
/// uploaded inputs if `/dev/shm` is full.
async fn write_deferral_with_no_space_retry(
    state: &AppState,
    input_root: &Path,
    work_dir: &Path,
    filename: &str,
    data: &[u8],
) -> Result<(), std::io::Error> {
    match write_deferral_artifact_file(work_dir, filename, data).await {
        Ok(_) => Ok(()),
        Err(e) if is_no_space_error(&e) => {
            warn!(
                "Worker /dev/shm full writing {}; cleaning stale inputs before retry",
                filename
            );
            cleanup_stale_uploaded_inputs(
                input_root,
                &state.active_uploaded_proofs,
                STALE_UPLOADED_INPUT_TTL,
            )
            .await;
            write_deferral_artifact_file(work_dir, filename, data)
                .await
                .map(|_| ())
        }
        Err(e) => Err(e),
    }
}

async fn write_deferral_artifact_file(
    work_dir: &Path,
    filename: &str,
    data: &[u8],
) -> Result<PathBuf, std::io::Error> {
    fs::create_dir_all(work_dir).await?;
    let final_path = work_dir.join(filename);
    let temp_path = work_dir.join(format!("{}.tmp", filename));
    fs::write(&temp_path, data).await?;
    fs::rename(&temp_path, &final_path).await?;
    Ok(final_path)
}

fn uploaded_input_proof_uuid_from_path(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    name.strip_prefix("edge_").map(ToOwned::to_owned)
}

fn is_no_space_error(error: &std::io::Error) -> bool {
    error.raw_os_error() == Some(28)
}

async fn write_uploaded_input_file(
    input_root: &Path,
    proof_uuid: &str,
    input_data: &[u8],
) -> Result<PathBuf, std::io::Error> {
    let work_dir = uploaded_input_work_dir(input_root, proof_uuid);
    fs::create_dir_all(&work_dir).await?;

    let input_path = work_dir.join("input.bin");
    let temp_path = work_dir.join("input.bin.tmp");
    fs::write(&temp_path, input_data).await?;
    fs::rename(&temp_path, &input_path).await?;
    Ok(input_path)
}

async fn cleanup_uploaded_input_dir(
    input_root: &Path,
    active_uploaded_proofs: &DashSet<String>,
    proof_uuid: &str,
) {
    active_uploaded_proofs.remove(proof_uuid);
    // The whole per-proof dir is reclaimed here once the app-prove phase
    // drains. A deferral job's `DeferralInput` is NOT in this dir at this point:
    // the manager pushes it just-in-time to the final-internal worker (a later
    // phase), where it lands in a freshly-created dir that `run_deferral_tail_merge`
    // removes after consuming it. So there is nothing deferral-specific to
    // preserve here.
    let work_dir = uploaded_input_work_dir(input_root, proof_uuid);
    match fs::remove_dir_all(&work_dir).await {
        Ok(_) => info!("Cleaned uploaded input directory {:?}", work_dir),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => warn!(
            "Failed to clean uploaded input directory {:?}: {}",
            work_dir, e
        ),
    }
}

async fn cleanup_stale_uploaded_inputs(
    input_root: &Path,
    active_uploaded_proofs: &DashSet<String>,
    stale_after: Duration,
) -> usize {
    let mut removed = 0usize;
    let mut entries = match fs::read_dir(input_root).await {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return 0,
        Err(e) => {
            warn!(
                "Failed to scan uploaded input root {}: {}",
                input_root.display(),
                e
            );
            return 0;
        }
    };

    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        let Some(proof_uuid) = uploaded_input_proof_uuid_from_path(&path) else {
            continue;
        };
        if active_uploaded_proofs.contains(&proof_uuid) {
            continue;
        }

        let metadata = match entry.metadata().await {
            Ok(metadata) if metadata.is_dir() => metadata,
            Ok(_) => continue,
            Err(e) => {
                warn!("Failed to stat uploaded input dir {:?}: {}", path, e);
                continue;
            }
        };

        let modified_at = match metadata.modified() {
            Ok(modified_at) => modified_at,
            Err(e) => {
                warn!("Failed to read modified time for {:?}: {}", path, e);
                continue;
            }
        };

        let age = SystemTime::now()
            .duration_since(modified_at)
            .unwrap_or(Duration::ZERO);
        if age < stale_after {
            continue;
        }

        // Re-check right before deleting: several awaits (dir scan, stat) have
        // elapsed since the first check, during which a `/sharded_app_prove`
        // (or a JIT deferral push + its prove) could have marked this proof
        // active. Don't delete an input that just became in-use.
        if active_uploaded_proofs.contains(&proof_uuid) {
            continue;
        }

        match fs::remove_dir_all(&path).await {
            Ok(_) => {
                removed += 1;
                info!(
                    "Removed stale uploaded input directory {:?} (age={}s)",
                    path,
                    age.as_secs()
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => warn!(
                "Failed to remove stale uploaded input dir {:?}: {}",
                path, e
            ),
        }
    }

    removed
}

pub async fn uploaded_input_janitor_task(
    state: Arc<AppState>,
    interval: Duration,
    stale_after: Duration,
    cancel_token: CancellationToken,
) {
    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                info!("Uploaded input janitor stopped");
                return;
            }
            _ = tokio::time::sleep(interval) => {
                let removed = cleanup_stale_uploaded_inputs(
                    uploaded_input_root(),
                    &state.active_uploaded_proofs,
                    stale_after,
                ).await;
                if removed > 0 {
                    info!("Uploaded input janitor removed {} stale directories", removed);
                }
            }
        }
    }
}

/// Handle the sharded app prove kickoff request.
/// Request body is JSON.
#[instrument(skip(state, req), fields(proof_uuid = %req.proof_uuid))]
pub async fn handle_sharded_app_prove(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ShardedAppProveRequest>,
) -> impl IntoResponse {
    let proof_uuid = req.proof_uuid.clone();

    if let Err(reason) = validate_uploaded_proof_uuid(&proof_uuid) {
        error!("Invalid proof_uuid: {}", reason);
        return (
            StatusCode::BAD_REQUEST,
            format!("Invalid proof_uuid: {}", reason),
        );
    }

    state.active_uploaded_proofs.insert(proof_uuid.clone());

    // The request carries no paths. Reconstruct the deterministic staged
    // locations from `proof_uuid`: the main input at `input.bin`, and one
    // `DeferralState` per circuit — the count is this worker's loaded deferral
    // keyset size (the manager fanned exactly that many `DeferralState`s here;
    // `0` on a non-deferral deployment).
    let input_path: String = staged_input_path(&proof_uuid)
        .to_string_lossy()
        .into_owned();
    let deferral_state_paths: Vec<String> = (0..loaded_deferral_circuit_count())
        .map(|i| {
            staged_deferral_state_path(&proof_uuid, i)
                .to_string_lossy()
                .into_owned()
        })
        .collect();

    info!(
        "Received sharded_app_prove for proof {}: prover_id={}, num_provers={}, deferral_circuits={}",
        proof_uuid,
        req.prover_id,
        req.num_provers,
        deferral_state_paths.len()
    );

    // Wait for input file(s) to exist (with 30-second timeout). For deferral
    // jobs this also includes one DeferralState file per circuit; all must be
    // present before execution can populate `StdIn.deferrals`.
    //
    // Skipped under `mock-provers`: the mock app prover fabricates proofs
    // without reading the input (see `prove_sharded_app_impl`), so there is no
    // file to wait for — the mock e2e tests exercise manager orchestration only
    // and never stage a real input.
    #[cfg(not(feature = "mock-provers"))]
    {
        let mut required_paths: Vec<PathBuf> = vec![PathBuf::from(&input_path)];
        required_paths.extend(deferral_state_paths.iter().map(PathBuf::from));
        let deadline = std::time::Instant::now() + Duration::from_secs(30);

        for path in &required_paths {
            loop {
                match fs::metadata(path).await {
                    Ok(_) => break, // File exists
                    Err(_) => {
                        if std::time::Instant::now() > deadline {
                            error!("Timeout waiting for input/deferral file: {:?}", path);
                            cleanup_uploaded_input_dir(
                                uploaded_input_root(),
                                &state.active_uploaded_proofs,
                                &proof_uuid,
                            )
                            .await;
                            return (
                                StatusCode::REQUEST_TIMEOUT,
                                format!("Timeout waiting for input file: {}", path.display()),
                            );
                        }
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        }
    }

    // Check if any app worker is free. Workers are program-agnostic in
    // the swap design — whichever worker picks up the job will lazily
    // load the right GPU prover for `req.program` (paying ~1 s if the
    // worker was previously loaded for a different program).
    if !state.prover_pool.has_available_worker(JobType::ShardedApp) {
        warn!(
            "No available app workers for proof {} (program {})",
            proof_uuid, req.program
        );
        cleanup_uploaded_input_dir(
            uploaded_input_root(),
            &state.active_uploaded_proofs,
            &proof_uuid,
        )
        .await;
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "No available app workers".to_string(),
        );
    }

    // Build ProofContext from request
    let context = protocol::ProofContext::new(
        req.proof_uuid.clone(),
        req.program.clone(),
        Default::default(),
    );

    // Spawn the proving task (non-blocking)
    let state_clone = state.clone();
    let proof_uuid_clone = proof_uuid.clone();
    tokio::spawn(async move {
        // Create streaming channel: prover threads send results here as each segment completes
        let (result_tx, result_rx) = crossbeam::channel::bounded::<protocol::ProofResult>(4);

        let job = ShardedAppProverJob {
            context: context.clone(),
            num_provers: req.num_provers,
            prover_id: req.prover_id,
            input_path,
            segment_memory: req
                .segment_memory
                .or(state.worker_config.default_segment_memory),
            max_app_provers: state.worker_config.max_app_provers,
            result_tx: Some(result_tx),
            deferral_state_paths,
        };

        // Bridge: drain crossbeam channel on a blocking thread, POST each result via HTTP
        let sender_state = state_clone.clone();
        let sender_uuid = proof_uuid_clone.clone();
        let sender_task = tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            info!("Sender task started for proof {}", sender_uuid);
            let mut count = 0u32;
            for proof_result in result_rx.iter() {
                count += 1;
                info!(
                    "Streaming result {count} for proof {} (worker_id={})",
                    sender_uuid,
                    sender_state.result_client.worker_id()
                );
                if let Err(e) = rt.block_on(
                    sender_state
                        .result_client
                        .submit_single_result(&sender_uuid, proof_result),
                ) {
                    error!("Failed to stream result: {}", e);
                }
            }
            info!(
                "Sender task finished for proof {}: streamed {} results",
                sender_uuid, count
            );
        });

        // Submit the proving job (blocks until all proving is done; results already streamed)
        match state_clone.prover_pool.submit_sharded_app_job(job).await {
            Ok(ProverResult::Success(_)) => {
                // Results already streamed via result_tx
            }
            Ok(ProverResult::Error(e)) => {
                error!("App proving failed: {}", e);
                if let Err(submit_err) = state_clone
                    .result_client
                    .submit_error(&proof_uuid_clone, &e)
                    .await
                {
                    error!("Failed to submit error: {}", submit_err);
                }
            }
            // The manager canceled the proof, so it wants no result.
            Ok(ProverResult::Canceled) => {
                info!("App proving canceled for {}", proof_uuid_clone);
            }
            Err(e) => {
                error!("Failed to submit app job: {}", e);
                if let Err(submit_err) = state_clone
                    .result_client
                    .submit_error(&proof_uuid_clone, &e.to_string())
                    .await
                {
                    error!("Failed to submit error: {}", submit_err);
                }
            }
        }

        // Wait for all streamed results to be sent
        let _ = sender_task.await;

        // Uploaded inputs are only needed for the app-proving phase on this
        // worker. Clean them after the streaming pipeline is fully drained so
        // long benchmark runs do not leak /dev/shm per proof UUID. A deferral
        // job's `DeferralInput` is not staged here — the manager pushes it
        // just-in-time to the final-internal worker later — so there is nothing
        // to preserve.
        cleanup_uploaded_input_dir(
            uploaded_input_root(),
            &state_clone.active_uploaded_proofs,
            &proof_uuid_clone,
        )
        .await;
    });

    (StatusCode::OK, "Edge work accepted".to_string())
}

/// Handle edge_prove_work request (LeafProve / InternalProve).
/// Request body is bincode-serialized `MessageEnvelope<GeneralProveRequest>`.
#[instrument(skip(state, body))]
pub async fn handle_recursion_prove(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> impl IntoResponse {
    // Deserialize bincode payload
    let envelope: MessageEnvelope<GeneralProveRequest> = match bincode::deserialize(&body) {
        Ok(e) => e,
        Err(e) => {
            error!("Failed to deserialize work envelope: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                format!("Failed to deserialize work envelope: {}", e),
            );
        }
    };

    let proof_uuid = match &envelope.message {
        GeneralProveRequest::LeafProve(req) => req.context.proof_uuid.clone(),
        GeneralProveRequest::InternalProve(req) => req.context.proof_uuid.clone(),
        GeneralProveRequest::EvmProve(req) => req.context.proof_uuid.clone(),
    };

    info!("Received edge_prove_work for proof {}", proof_uuid);

    match envelope.message {
        GeneralProveRequest::LeafProve(req) => {
            let segment_start = req.segment_start;
            let segment_end = req.segment_end;

            info!(
                "Processing LeafProve for proof {}: segments [{}-{}]",
                proof_uuid, segment_start, segment_end
            );

            if !state.prover_pool.has_available_worker(JobType::Leaf) {
                warn!("No available leaf workers for proof {}", proof_uuid);
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "No available leaf workers".to_string(),
                );
            }

            let app_proofs = match req
                .app_proofs
                .iter()
                .map(|bytes| proof::decode_proof(bytes))
                .collect::<eyre::Result<Vec<_>>>()
            {
                Ok(proofs) => proofs,
                Err(e) => {
                    error!(
                        "Failed to decode app proofs for proof {}: {}",
                        proof_uuid, e
                    );
                    return (
                        StatusCode::BAD_REQUEST,
                        format!("Invalid app proofs: {}", e),
                    );
                }
            };

            let state_clone = state.clone();
            let proof_uuid_clone = proof_uuid.clone();
            tokio::spawn(async move {
                let job = LeafProverJob {
                    context: req.context.clone(),
                    app_proofs,
                    segment_start,
                    segment_end,
                };

                match state_clone.prover_pool.submit_leaf_job(job).await {
                    Ok(ProverResult::Success(results)) => {
                        if let Err(e) = state_clone
                            .result_client
                            .submit_result(&proof_uuid_clone, results)
                            .await
                        {
                            error!("Failed to submit leaf proof results: {}", e);
                        }
                    }
                    Ok(ProverResult::Error(e)) => {
                        error!("Leaf proving failed: {}", e);
                        if let Err(submit_err) = state_clone
                            .result_client
                            .submit_error(&proof_uuid_clone, &e)
                            .await
                        {
                            error!("Failed to submit error: {}", submit_err);
                        }
                    }
                    // The manager canceled the proof, so it wants no result.
                    Ok(ProverResult::Canceled) => {
                        info!("Leaf proving canceled for {}", proof_uuid_clone);
                    }
                    Err(e) => {
                        error!("Failed to submit leaf job: {}", e);
                        if let Err(submit_err) = state_clone
                            .result_client
                            .submit_error(&proof_uuid_clone, &e.to_string())
                            .await
                        {
                            error!("Failed to submit error: {}", submit_err);
                        }
                    }
                }
            });
        }
        GeneralProveRequest::InternalProve(req) => {
            let layer_idx = req.layer_idx;
            let segment_start = req.segment_start;
            let segment_end = req.segment_end;
            let is_final = req.is_final_proof;
            // Only the final internal proof of a deferral job carries
            // the tail-merge dispatch (the manager attaches it from its
            // tree-shape state). Cloned out before the request is moved
            // into the job so we can forward it to the tail-merge prep
            // (`drive_evm_prep_and_post`) / stark deferral merge.
            let deferral_tail = req.deferral_tail.clone();
            // Per-proof deferral flag: does THIS proof run the tail merge?
            // Drives the final-internal wrap-skip (and, in the evm path, the
            // root wrap-retry `proofs_type`). On a deferral deployment a
            // no-deferral proof has no tail, so it takes the normal wrap.
            let proof_has_deferral = deferral_tail.is_some();
            // Depth-0 `DeferralMerkleProofs` for a no-deferral proof on a
            // deferral deployment (built by the terminal app worker). The evm
            // path hands these to root prove; `None` for real deferral proofs
            // (they build their own on the tail) and non-deferral deployments.
            // Only consumed on the evm/mock path; gate the capture so a
            // stark-only default build doesn't flag it unused.
            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
            let depth0_merkle_bytes = req.deferral_merkle_proofs_bytes.clone();

            info!(
                "Processing InternalProve for proof {}: layer={}, segments [{}-{}], final={}, deferral_tail={}",
                proof_uuid, layer_idx, segment_start, segment_end, is_final,
                deferral_tail.is_some()
            );

            if !state.prover_pool.has_available_worker(JobType::Internal) {
                warn!("No available internal workers for proof {}", proof_uuid);
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "No available internal workers".to_string(),
                );
            }

            let child_proofs = match req
                .child_proofs
                .iter()
                .map(|bytes| proof::decode_proof(bytes))
                .collect::<eyre::Result<Vec<_>>>()
            {
                Ok(proofs) => proofs,
                Err(e) => {
                    error!(
                        "Failed to decode child proofs for proof {}: {}",
                        proof_uuid, e
                    );
                    return (
                        StatusCode::BAD_REQUEST,
                        format!("Invalid child proofs: {}", e),
                    );
                }
            };

            let state_clone = state.clone();
            let proof_uuid_clone = proof_uuid.clone();
            tokio::spawn(async move {
                let ctx = req.context.clone();
                let job = InternalProverJob {
                    context: ctx.clone(),
                    child_proofs,
                    layer_idx,
                    segment_start,
                    segment_end,
                    is_final_proof: is_final,
                    proof_has_deferral,
                };

                match state_clone.prover_pool.submit_internal_job(job).await {
                    Ok(ProverResult::Success(results)) => {
                        // A `proof_type=stark` deferral job's completion
                        // artifact is the MERGED final internal proof
                        // (`prove_def → prove_mixed → wrap`) carrying its
                        // deferral merkle proofs — not the raw internal
                        // proof. So for that case we run the merge and submit
                        // the merged proof in place of the raw one. (For
                        // `proof_type=evm` the same merge runs inside
                        // `drive_evm_prep_and_post`, whose ready-for-evm
                        // message carries the merkle proofs to the dispatched
                        // `EvmProve` step for root prove to consume.)
                        #[cfg(not(feature = "mock-provers"))]
                        let stark_deferral_final = is_final
                            && ctx.proof_type == protocol::ProofType::Stark
                            && deferral_tail.is_some();
                        #[cfg(feature = "mock-provers")]
                        let stark_deferral_final = false;

                        // Unified EVM step: EVERY final-internal worker of an
                        // `Evm` proof runs the tail-merge / merkle-prep half and
                        // submits ONLY the POST-merge ready-for-evm message (NOT the
                        // raw internal). The manager then dispatches the
                        // `EvmProve` step (root → halo2) to any eligible
                        // `runs_evm_prove()` worker — `Full` or `EvmDedicated`. A
                        // `Full` worker that produced the final internal is
                        // itself eligible, so the EVM step can dispatch back to it;
                        // there is no longer a separate in-process path.
                        let evm_handoff = is_final && ctx.proof_type == protocol::ProofType::Evm;

                        if stark_deferral_final {
                            #[cfg(not(feature = "mock-provers"))]
                            drive_stark_deferral_merge_and_post(
                                &state_clone,
                                &proof_uuid_clone,
                                &ctx,
                                results,
                                deferral_tail.clone(),
                            )
                            .await;
                        } else if evm_handoff {
                            // Run the tail-merge/merkle-prep half and submit the
                            // POST-merge proof + merkle bytes as the ready-for-evm
                            // message (NOT the raw internal). Do NOT continue to
                            // root in-process — root → halo2 is a dispatched
                            // `EvmProve` step, regardless of this worker's role.
                            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
                            drive_evm_prep_and_post(
                                &state_clone,
                                &proof_uuid_clone,
                                &ctx,
                                results,
                                deferral_tail.clone(),
                                depth0_merkle_bytes.clone(),
                            )
                            .await;
                        } else if let Err(e) = state_clone
                            // Non-Evm (or non-final) internal: submit the raw
                            // Internal result(s) to the manager as today.
                            .result_client
                            .submit_result(&proof_uuid_clone, results)
                            .await
                        {
                            error!("Failed to submit internal proof results: {}", e);
                        }
                    }
                    Ok(ProverResult::Error(e)) => {
                        error!("Internal proving failed: {}", e);
                        if let Err(submit_err) = state_clone
                            .result_client
                            .submit_error(&proof_uuid_clone, &e)
                            .await
                        {
                            error!("Failed to submit error: {}", submit_err);
                        }
                    }
                    // The manager canceled the proof, so it wants no result.
                    Ok(ProverResult::Canceled) => {
                        info!("Internal proving canceled for {}", proof_uuid_clone);
                    }
                    Err(e) => {
                        error!("Failed to submit internal job: {}", e);
                        if let Err(submit_err) = state_clone
                            .result_client
                            .submit_error(&proof_uuid_clone, &e.to_string())
                            .await
                        {
                            error!("Failed to submit error: {}", submit_err);
                        }
                    }
                }
            });
        }
        GeneralProveRequest::EvmProve(req) => {
            // Dedicated-halo2 mode: the `EvmDedicated` worker runs the EVM step
            // (root → halo2) on the finished (post-tail-merge) internal proof +
            // merkle bytes the StarkOnly worker handed off, then posts the `Evm`
            // result. The tail merge already ran on the StarkOnly worker, so this
            // path is deferral-agnostic (deferral and non-deferral proofs are
            // handled identically). This request is never dispatched in the
            // default `Full` deployment.
            info!(
                "Processing EvmProve for proof {}: dedicated EVM step (root → halo2, has_deferral={})",
                proof_uuid, req.proof_has_deferral
            );
            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
            {
                let state_clone = state.clone();
                tokio::spawn(async move {
                    drive_evm_prove_from_request(&state_clone, req).await;
                });
            }
            #[cfg(not(any(feature = "evm-prove", feature = "mock-provers")))]
            {
                let msg = format!(
                    "EvmProve dispatched to proof {} on a worker built without EVM support",
                    req.context.proof_uuid
                );
                error!("{msg}");
                if let Err(e) = state.result_client.submit_error(&proof_uuid, &msg).await {
                    error!("Failed to submit EvmProve error: {}", e);
                }
            }
        }
    }

    (StatusCode::OK, "Work completed".to_string())
}

/// Run the EVM step (root → halo2) on a FINISHED (post-tail-merge) internal
/// proof, returning the final `Evm` proof. This is the second half of the EVM
/// prove: the final-internal worker runs the tail-merge / merkle-prep half
/// ([`drive_evm_prep_and_post`]) and ships a ready-for-evm message, then the
/// manager dispatches the `EvmProve` step to an eligible `runs_evm_prove()`
/// worker (`Full` or `EvmDedicated`) whose handler
/// ([`drive_evm_prove_from_request`]) calls this on the handed-off proof. The
/// tail merge already happened on the final-internal worker, so this half is
/// deferral-agnostic — it takes the finished proof, the serialized merkle
/// proofs (already decoded by the caller), and the `proof_has_deferral` flag.
///
/// The root proof is a worker-internal intermediate consumed by the halo2
/// stage — it is not returned or reported; its timing is folded into the single
/// `Evm` result.
///
/// Fail-fast: if the worker is built with `evm-prove` but `EdgeArtifacts.evm`
/// is `None` (no halo2_pk_path configured), returns `Err` with a clear message
/// instead of panicking or silently dropping.
#[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
async fn run_evm_prove(
    pool: &crate::prover_pool::ProverPool,
    proof_uuid: &str,
    context: &protocol::ProofContext,
    final_internal_proof: proof::ProofWithPublicValue<proof::F>,
    #[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
    deferral_merkle_proofs: Option<verify_stark::deferral::DeferralMerkleProofs<proof::F>>,
    #[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))] proof_has_deferral: bool,
) -> std::result::Result<protocol::ProofResult, String> {
    use crate::provers::{Halo2ProverJob, RootProverJob};

    // Fail-fast: a stark-only worker (no EVM artifacts loaded) cannot run the
    // EVM step. Surface a clear error instead of waiting for root prove to
    // fail with the same diagnostic later in the pipeline. Skipped in mock
    // mode (no real artifacts loaded; the mock provers don't read them).
    #[cfg(not(feature = "mock-provers"))]
    {
        let evm_ready = crate::artifacts::ArtifactStore::global()
            .and_then(|s| s.get_edge_artifacts().map(|a| a.evm.is_some()))
            .unwrap_or(false);
        if !evm_ready {
            return Err(format!(
                "EVM prove requested but worker has no EVM artifacts \
                 (evm-prove build without halo2_pk_path); aborting proof {}",
                proof_uuid
            ));
        }
    }

    // --- Root prove (root prover thread) ---
    // The root proof is a worker-internal intermediate: it is what the halo2
    // stage verifies, but it is NOT reported to the manager — only the final
    // EVM proof is.
    let root_job = RootProverJob {
        context: context.clone(),
        final_internal_proof,
        #[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
        deferral_merkle_proofs,
        #[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
        proof_has_deferral,
    };
    let root_state = pool
        .submit_root_job(root_job)
        .await
        .map_err(|e| format!("Root prove failed for {}: {}", proof_uuid, e))?;
    let root_proof_bytes = root_state.proof.as_ref().ok_or_else(|| {
        format!(
            "EVM prove: root result for {} has no proof bytes",
            proof_uuid
        )
    })?;
    let root_proof = proof::decode_root_proof(root_proof_bytes).map_err(|e| {
        format!(
            "EVM prove: failed to decode root proof for {}: {}",
            proof_uuid, e
        )
    })?;

    // --- Halo2 prove (halo2 prover thread) → the final EVM proof ---
    let halo2_job = Halo2ProverJob {
        context: context.clone(),
        root_proof,
    };
    let mut evm = match pool.submit_halo2_job(halo2_job).await {
        Ok(ProverResult::Success(r)) => r
            .into_iter()
            .find_map(|res| match res {
                protocol::ProofResult::Evm(e) => Some(e),
                _ => None,
            })
            .ok_or_else(|| {
                format!(
                    "EVM prove: halo2 stage for {} produced no Evm proof",
                    proof_uuid
                )
            })?,
        Ok(ProverResult::Error(e)) => {
            return Err(format!("Halo2 prove failed for {}: {}", proof_uuid, e));
        }
        // The halo2 prover runs no cancellation check, so this is unreachable
        // in practice and reported rather than asserted.
        Ok(ProverResult::Canceled) => {
            return Err(format!("Halo2 prove for {} was canceled", proof_uuid));
        }
        Err(e) => {
            return Err(format!(
                "Failed to submit halo2 job for {}: {}",
                proof_uuid, e
            ));
        }
    };

    // The root proof itself is a worker-internal intermediate (not reported),
    // but its timing IS worth keeping. Fold the root prove_time_ms + sub-metrics
    // into the (only) reported Evm result. Sub-metrics are prefixed `root_` /
    // `halo2_` so the two stages stay distinguishable on the manager side.
    evm.state.root_prove_time_ms = root_state.prove_time_ms;
    let halo2_sub = std::mem::take(&mut evm.state.sub_metrics);
    evm.state.sub_metrics = halo2_sub
        .into_iter()
        .map(|(k, v)| (format!("halo2_{k}"), v))
        .chain(
            root_state
                .sub_metrics
                .into_iter()
                .map(|(k, v)| (format!("root_{k}"), v)),
        )
        .collect();

    info!(
        "EVM prove complete for {}: root_prove_time_ms={}, halo2_prove_time_ms={}",
        proof_uuid, evm.state.root_prove_time_ms, evm.state.prove_time_ms
    );

    // Only the final EVM proof is returned (and POSTed); it now carries both the
    // halo2 timing (`prove_time_ms`) and the folded-in root timing.
    Ok(protocol::ProofResult::Evm(evm))
}

/// Dedicated-worker `EvmProve` driver: decode the handed-off (post-tail-merge)
/// internal proof + serialized merkle bytes, run the EVM step (root → halo2)
/// via [`run_evm_prove`], and POST the resulting `Evm` proof (or the error) to
/// the manager. The deferral tail merge already ran on the StarkOnly worker, so
/// this path never sees deferral inputs.
#[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
async fn drive_evm_prove_from_request(state: &Arc<AppState>, req: protocol::EvmProveRequest) {
    let proof_uuid = req.context.proof_uuid.clone();

    // Decode the finished internal proof this EVM step builds on.
    let final_internal_proof = match proof::decode_proof(&req.internal_proof_bytes) {
        Ok(p) => p,
        Err(e) => {
            let msg = format!(
                "EvmProve: failed to decode handed-off internal proof for {}: {}",
                proof_uuid, e
            );
            error!("{msg}");
            if let Err(e) = state.result_client.submit_error(&proof_uuid, &msg).await {
                error!("Failed to submit EvmProve error: {}", e);
            }
            return;
        }
    };

    // Decode the serialized deferral merkle proofs (real builds only). `None`
    // on a non-deferral deployment; `Some` from the tail merge or depth-0.
    #[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
    let deferral_merkle_proofs = match req.deferral_merkle_proofs_bytes.as_ref() {
        Some(bytes) => {
            let mut reader = std::io::Cursor::new(bytes.as_slice());
            match verify_stark::deferral::DeferralMerkleProofs::decode(&mut reader) {
                Ok(mp) => Some(mp),
                Err(e) => {
                    let msg = format!(
                        "EvmProve: failed to decode deferral merkle proofs for {}: {}",
                        proof_uuid, e
                    );
                    error!("{msg}");
                    if let Err(e) = state.result_client.submit_error(&proof_uuid, &msg).await {
                        error!("Failed to submit EvmProve error: {}", e);
                    }
                    return;
                }
            }
        }
        None => None,
    };

    let result = run_evm_prove(
        &state.prover_pool,
        &proof_uuid,
        &req.context,
        final_internal_proof,
        #[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
        deferral_merkle_proofs,
        #[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
        req.proof_has_deferral,
    )
    .await;

    match result {
        Ok(evm_result) => {
            if let Err(e) = state
                .result_client
                .submit_result(&proof_uuid, vec![evm_result])
                .await
            {
                error!("Failed to submit EvmProve result: {}", e);
            }
        }
        Err(msg) => {
            error!("EvmProve error for {}: {}", proof_uuid, msg);
            if let Err(e) = state.result_client.submit_error(&proof_uuid, &msg).await {
                error!("Failed to submit EvmProve error: {}", e);
            }
        }
    }
}

/// The final-internal worker's EVM prep step: run the tail-merge /
/// merkle-prep half of the EVM prove on the final internal proof, then submit
/// the **post-tail-merge** internal proof + serialized merkle bytes as a
/// *ready-for-evm* message (`InternalProofState::ready_for_evm = true`) — NOT the
/// raw internal. The manager routes only on this ready-for-evm message, emitting
/// the `EvmProve` step (root → halo2) to an eligible `runs_evm_prove()` worker;
/// it never dispatches on the raw unmerged proof.
///
/// EVERY final-internal worker of an `Evm` proof runs this, regardless of role:
/// a `Full` worker that produced the final internal is itself an eligible
/// dispatch target for the follow-up `EvmProve`, so root → halo2 may run back
/// on it — there is no separate in-process path. The merge itself is identical
/// (`prove_def → prove_mixed → wrap`); deferral and non-deferral proofs use the
/// same handoff (non-deferral just carries the raw proof + optional depth-0
/// merkle bytes).
#[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
async fn drive_evm_prep_and_post(
    state: &Arc<AppState>,
    proof_uuid: &str,
    context: &protocol::ProofContext,
    results: Vec<protocol::ProofResult>,
    deferral_tail: Option<protocol::DeferralTailDispatch>,
    depth0_merkle_bytes: Option<Vec<u8>>,
) {
    // The internal prove yields exactly one Internal result.
    let final_internal = results.into_iter().find_map(|r| match r {
        protocol::ProofResult::Internal(ip) => Some(ip),
        _ => None,
    });
    let Some(mut final_internal) = final_internal else {
        let msg = format!("evm prep: expected an Internal result for {proof_uuid}, found none");
        error!("{msg}");
        let _ = state.result_client.submit_error(proof_uuid, &msg).await;
        return;
    };

    // Compute the finished (post-merge) proof bytes + serialized merkle bytes.
    // Real deferral proof: run the merge and re-encode. Everything else (a
    // no-deferral proof, or mock mode) ships the raw internal bytes and carries
    // the depth-0 merkle bytes forward if present.
    #[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
    let prepped: std::result::Result<(Vec<u8>, Option<Vec<u8>>), String> = {
        // The merge is blocking (fs read + decode + GPU proving + encode). Run
        // it on a blocking thread so it never stalls a tokio runtime worker
        // (which also serves /healthz, /readyz, and the registration heartbeat).
        // SDK proof types stay inside the closure; only Vec<u8> crosses back.
        let internal_bytes = final_internal.state.proof.clone();
        let uuid = proof_uuid.to_string();
        let context = context.clone();
        tokio::task::spawn_blocking(move || {
            let internal_bytes = internal_bytes.ok_or_else(|| {
                format!("evm prep: final internal proof for {uuid} has no proof bytes")
            })?;
            if let Some(tail) = deferral_tail.as_ref() {
                let final_internal_proof = proof::decode_proof(&internal_bytes).map_err(|e| {
                    format!("evm prep: failed to decode final internal proof for {uuid}: {e}")
                })?;
                let (merged, merkle_proofs) =
                    run_deferral_tail_merge(&uuid, &context, final_internal_proof, tail).map_err(
                        |e| format!("evm prep: deferral tail merge failed for {uuid}: {e}"),
                    )?;
                let merged_bytes = proof::encode_proof(&merged).map_err(|e| {
                    format!("evm prep: failed to encode merged proof for {uuid}: {e}")
                })?;
                let mut merkle_bytes = Vec::new();
                merkle_proofs.encode(&mut merkle_bytes).map_err(|e| {
                    format!("evm prep: failed to encode merkle proofs for {uuid}: {e}")
                })?;
                Ok((merged_bytes, Some(merkle_bytes)))
            } else {
                // No tail merge (this proof made no deferred calls): finished
                // proof is the raw internal; carry depth-0 merkle bytes if any.
                Ok((internal_bytes, depth0_merkle_bytes))
            }
        })
        .await
        .unwrap_or_else(|e| {
            Err(format!(
                "evm prep: tail-merge task failed for {proof_uuid}: {e}"
            ))
        })
    };

    #[cfg(feature = "mock-provers")]
    let prepped: std::result::Result<(Vec<u8>, Option<Vec<u8>>), String> = {
        // Mock mode: no real merge — ship the raw internal bytes, carrying the
        // depth-0 merkle bytes forward if present.
        let _ = (context, &deferral_tail);
        match final_internal.state.proof.clone() {
            Some(bytes) => Ok((bytes, depth0_merkle_bytes)),
            None => Err(format!(
                "evm prep: final internal proof for {proof_uuid} has no proof bytes"
            )),
        }
    };

    match prepped {
        Ok((finished_proof_bytes, merkle_bytes)) => {
            final_internal.state.proof = Some(finished_proof_bytes);
            final_internal.state.deferral_merkle_proofs_bytes = merkle_bytes;
            final_internal.state.ready_for_evm = true;
            info!(
                "ready-for-evm for {proof_uuid} (dedicated mode); submitting post-merge internal \
                 proof + merkle to the manager for EvmProve dispatch"
            );
            if let Err(e) = state
                .result_client
                .submit_result(
                    proof_uuid,
                    vec![protocol::ProofResult::Internal(final_internal)],
                )
                .await
            {
                error!("Failed to submit evm ready-for-evm proof for {proof_uuid}: {e}");
            }
        }
        Err(msg) => {
            error!("{msg}");
            let _ = state.result_client.submit_error(proof_uuid, &msg).await;
        }
    }
}

/// Stark-mode deferral completion: run the deferral tail merge
/// (`prove_def → prove_mixed → wrap`) on the final internal proof and submit
/// the MERGED proof — carrying its `DeferralMerkleProofs` — as the single
/// Internal completion result, in place of the raw internal proof.
///
/// This is the `proof_type=stark` analogue of the merge
/// [`drive_evm_prep_and_post`] runs before shipping its ready-for-evm
/// message. There is no root/halo2 wrap here, so the merged
/// stark proof itself is the deliverable and its merkle proofs must travel to
/// the manager (persisted, then read back by `load_final_proof`) to be
/// verifiable. Only invoked by [`handle_recursion_prove`] for the final
/// internal proof of a `Stark` proof that carries a deferral tail.
#[cfg(not(feature = "mock-provers"))]
async fn drive_stark_deferral_merge_and_post(
    state: &Arc<AppState>,
    proof_uuid: &str,
    context: &protocol::ProofContext,
    results: Vec<protocol::ProofResult>,
    deferral_tail: Option<protocol::DeferralTailDispatch>,
) {
    // The internal prove yields exactly one Internal result.
    let final_internal = results.into_iter().find_map(|r| match r {
        protocol::ProofResult::Internal(ip) => Some(ip),
        _ => None,
    });
    let Some(mut final_internal) = final_internal else {
        let msg = format!(
            "stark deferral merge: expected an Internal result for {proof_uuid}, found none"
        );
        error!("{msg}");
        let _ = state.result_client.submit_error(proof_uuid, &msg).await;
        return;
    };
    // `handle_recursion_prove` only reaches here with `deferral_tail.is_some()`.
    let Some(tail) = deferral_tail else {
        let msg = format!("stark deferral merge: missing deferral tail dispatch for {proof_uuid}");
        error!("{msg}");
        let _ = state.result_client.submit_error(proof_uuid, &msg).await;
        return;
    };

    // Decode the raw final internal proof, run the shared tail merge
    // (`prove_def → prove_mixed → wrap`), and re-encode the merged proof and
    // its merkle proofs to wire bytes. The EVM path drives the same
    // `run_deferral_tail_merge` but keeps the in-memory structs (root prove
    // consumes them); here we ship the encoded bytes to the manager instead.
    // Blocking (fs read + decode + GPU proving + encode): run on a blocking
    // thread so it never stalls a tokio runtime worker. SDK proof types stay
    // inside the closure; only Vec<u8> crosses back.
    let merged: eyre::Result<(Vec<u8>, Vec<u8>)> = {
        let internal_bytes = final_internal.state.proof.clone();
        let uuid = proof_uuid.to_string();
        let context = context.clone();
        tokio::task::spawn_blocking(move || -> eyre::Result<(Vec<u8>, Vec<u8>)> {
            let internal_bytes = internal_bytes.ok_or_else(|| {
                eyre::eyre!(
                    "stark deferral merge: final internal proof for {uuid} has no proof bytes"
                )
            })?;
            let final_internal_proof = proof::decode_proof(&internal_bytes).map_err(|e| {
                eyre::eyre!(
                    "stark deferral merge: failed to decode final internal proof for {uuid}: {e}"
                )
            })?;

            let (merged, merkle_proofs) =
                run_deferral_tail_merge(&uuid, &context, final_internal_proof, &tail)?;

            let merged_bytes = proof::encode_proof(&merged).map_err(|e| {
                eyre::eyre!("stark deferral merge: failed to encode merged proof for {uuid}: {e}")
            })?;
            let mut merkle_bytes = Vec::new();
            merkle_proofs.encode(&mut merkle_bytes).map_err(|e| {
                eyre::eyre!("stark deferral merge: failed to encode merkle proofs for {uuid}: {e}")
            })?;
            Ok((merged_bytes, merkle_bytes))
        })
        .await
        .unwrap_or_else(|e| {
            Err(eyre::eyre!(
                "stark deferral merge: tail-merge task failed for {proof_uuid}: {e}"
            ))
        })
    };

    match merged {
        Ok((merged_proof_bytes, merkle_bytes)) => {
            final_internal.state.proof = Some(merged_proof_bytes);
            final_internal.state.deferral_merkle_proofs_bytes = Some(merkle_bytes);
            info!(
                "Stark deferral merge complete for {proof_uuid}; submitting merged \
                 final internal proof with deferral merkle proofs as completion"
            );
            if let Err(e) = state
                .result_client
                .submit_result(
                    proof_uuid,
                    vec![protocol::ProofResult::Internal(final_internal)],
                )
                .await
            {
                error!("Failed to submit merged stark deferral proof for {proof_uuid}: {e}");
            }
        }
        Err(e) => {
            let msg = format!("stark deferral merge failed for {proof_uuid}: {e}");
            error!("{msg}");
            let _ = state.result_client.submit_error(proof_uuid, &msg).await;
        }
    }
}

/// Run the tail merge — `prove_def → prove_mixed → wrap` — on the final
/// internal stark proof, returning an updated
/// `ProofWithPublicValue` whose `.proof` is the merged inner stark proof
/// that downstream root prove will consume.
///
/// The deferral SDK is reconstructed thread-locally per call via
/// `Sdk::from_deferral_cached_proving_key` (the SDK is `!Send + !Sync`,
/// so it can't be cached on the shared artifact store). The
/// SDK is then used through `Sdk::prover(exe)` to obtain a `StarkProver`
/// whose public `agg_prover` + `def_prover` we drive directly — we
/// deliberately bypass `StarkProver::prove` because we already have the
/// VM-tree's final internal proof in hand and only need the deferral
/// merge stages.
///
/// Inputs the manager owns:
/// - the per-circuit `DeferralInput` bytes — pushed just-in-time to this
///   worker's deterministic staged path (`staged_deferral_input_path`) right
///   before the final internal dispatch; the count comes from the loaded
///   keyset, so no paths ride on `DeferralTailDispatch`;
/// - `tail.layer_metadata` — initial `InternalLayerMetadata` (V1: built
///   from manager tree-shape state). Mutated locally on this worker by
///   `prove_mixed`/`wrap_proof`; only the initial value crosses the wire.
///
/// The merged proof's `deferral_merkle_proofs` is now computed
/// here: the manager-forwarded final-side path (`tail.final_merkle_path_bytes`)
/// is decoded, the initial-side path is rebuilt locally from the exe
/// (`crate::deferral_merkle::build_initial_memory_tree`), both are
/// finalized with `depth` from `DeferralPvs[DEF_PVS_AIR_ID]` of the
/// merged inner proof, and the `DeferralMerkleProofs` returned alongside
/// the merged `ProofWithPublicValue` so root prove can attach them to
/// its `VmStarkProof`. The feature-gated deferral integration test covers the
/// full end-to-end verification with real keygen and a real proof round-trip.
#[cfg(not(feature = "mock-provers"))]
fn run_deferral_tail_merge(
    proof_uuid: &str,
    context: &protocol::ProofContext,
    final_internal_proof: proof::ProofWithPublicValue<proof::F>,
    tail: &protocol::DeferralTailDispatch,
) -> eyre::Result<(
    proof::ProofWithPublicValue<proof::F>,
    verify_stark::deferral::DeferralMerkleProofs<proof::F>,
)> {
    use continuations_v2::circuit::inner::ProofsType;
    use sdk_v2::prover::InternalLayerMetadata as SdkInternalLayerMetadata;
    use sdk_v2::{DeferralInput, Sdk};
    use std::borrow::Borrow;
    use verify_stark::{
        deferral::DeferralMerkleProofs,
        pvs::{DeferralPvs, DEF_PVS_AIR_ID},
        VmStarkProof,
    };

    let artifact_store = crate::artifacts::ArtifactStore::global()
        .ok_or_else(|| eyre::eyre!("Artifact store not initialized"))?;
    let edge_artifacts = artifact_store
        .get_edge_artifacts()
        .ok_or_else(|| eyre::eyre!("Edge artifacts not loaded"))?;
    let deferral_artifacts = edge_artifacts.deferral.as_ref().ok_or_else(|| {
        eyre::eyre!(
            "Tail deferral merge requested but this worker is not a deferral \
             deployment (enable_deferral not set)."
        )
    })?;

    // The number of deferral circuits comes from THIS worker's loaded keyset —
    // the authoritative source. The manager doesn't pass paths or a count: the
    // `DeferralInput` for each circuit was pushed just-in-time to the
    // deterministic staged path `/dev/shm/edge_{proof_uuid}/deferral_input_{idx}.bin`
    // (see `staged_deferral_input_path`), which we reconstruct here.
    let num_deferral_circuits = deferral_artifacts
        .cached_pk
        .app_pk
        .app_vm_pk
        .vm_config
        .deferral
        .as_ref()
        .map(|d| d.circuits.len())
        .unwrap_or(0);

    // Load + decode each `DeferralInput`. The wire format is bincode
    // (`Serialize`/`Deserialize` on the struct — mirrors what
    // `DeferralState` uses); the inner `byte_vec: Vec<Vec<u8>>` is opaque
    // and uses the stark-backend codec per-element when the caller
    // produced it via `DeferralInput::from_inputs`.
    let mut def_inputs: Vec<DeferralInput> = Vec::with_capacity(num_deferral_circuits);
    for idx in 0..num_deferral_circuits {
        let input_path = staged_deferral_input_path(proof_uuid, idx);
        let bytes = std::fs::read(&input_path).map_err(|e| {
            eyre::eyre!(
                "Failed to read deferral input for circuit {} at {}: {} (the manager pushes \
                 it just-in-time before the final internal prove)",
                idx,
                input_path.display(),
                e
            )
        })?;
        let input: DeferralInput = bincode::deserialize(&bytes).map_err(|e| {
            eyre::eyre!(
                "Failed to deserialize DeferralInput for circuit {} at {}: {}",
                idx,
                input_path.display(),
                e
            )
        })?;
        def_inputs.push(input);
    }

    // The `DeferralInput` bytes are now in memory. Reclaim the per-proof dir
    // the manager's just-in-time push created on this worker (`/dev/shm/edge_
    // {proof_uuid}/`). The app-prove phase already cleaned its own dir; this
    // dir holds only the JIT-pushed deferral inputs, so removing it here is the
    // deterministic cleanup for the deferral tail (no reliance on stale-cleanup).
    let tail_work_dir = uploaded_input_work_dir(uploaded_input_root(), proof_uuid);
    match std::fs::remove_dir_all(&tail_work_dir) {
        Ok(_) => info!("Cleaned tail deferral-input dir {:?}", tail_work_dir),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => warn!(
            "Failed to clean tail deferral-input dir {:?}: {}",
            tail_work_dir, e
        ),
    }

    // Look up the exe — needed for `Sdk::prover(exe)` even though we
    // bypass `StarkProver::prove`. The SDK API path requires building an
    // AppProver internally (~1 s of GPU init per tail merge; a future
    // optimization is direct AggProver construction).
    let exe = artifact_store
        .vmexe(&context.program)
        .ok_or_else(|| eyre::eyre!("vmexe for {} not loaded on this worker", context.program))?;

    // Reconstruct the deferral-enabled SDK thread-locally from the cached pk.
    info!(
        "EVM prove[{}]: reconstructing thread-local deferral SDK from cached_pk",
        proof_uuid
    );
    let sdk = Sdk::from_deferral_cached_proving_key((*deferral_artifacts.cached_pk).clone())
        .map_err(|e| eyre::eyre!("Sdk::from_deferral_cached_proving_key failed: {}", e))?;

    // `sdk.prover(exe)` constructs a `StarkProver { app_prover, agg_prover,
    // deferral_setup }`. We only use `agg_prover` + the deferral prover; the
    // AppProver inside is unused (sunk cost). (rc.3: the deferral prover moved
    // from a `def_prover: Option<..>` field to `deferral_setup.prover()`.)
    let stark_prover = sdk
        .prover((*exe).clone())
        .map_err(|e| eyre::eyre!("Sdk::prover failed: {}", e))?;
    let agg_prover = &stark_prover.agg_prover;
    let def_prover = stark_prover
        .deferral_setup
        .prover()
        .ok_or_else(|| eyre::eyre!("Reconstructed deferral SDK has no deferral prover"))?;

    // Reconstruct the VM-tree's final stark proof from the worker's
    // bincode-encoded `ProofWithPublicValue`. `deferral_merkle_proofs`
    // is finalized after `wrap_proof` (we need `depth` from
    // `DeferralPvs` of the merged inner proof to slice / zero-pad the
    // paths).
    let user_pvs_proof = final_internal_proof
        .user_public_values
        .clone()
        .ok_or_else(|| {
            eyre::eyre!(
                "EVM prove: final internal proof for {} has no user_public_values; \
                 cannot reconstruct VmStarkProof for deferral merge",
                proof_uuid
            )
        })?;
    let mut stark_proof = VmStarkProof {
        inner: final_internal_proof.proof.clone(),
        user_pvs_proof,
        deferral_merkle_proofs: None,
    };

    // Decode the manager-forwarded final-side depth-independent
    // path now (early validation: empty/malformed bytes should fail before
    // burning prove-time work). The path stays "depth-0" until
    // `prove_mixed` produces a `depth` to finalize against.
    if tail.final_merkle_path_bytes.is_empty() {
        eyre::bail!(
            "EVM prove[{}]: deferral tail dispatch is missing final_merkle_path_bytes; \
             the terminal AppProof never carried the (DEFERRAL_AS,0) path.",
            proof_uuid
        );
    }
    let final_depth_indep_path = proof::decode_deferral_auth_path(&tail.final_merkle_path_bytes)
        .map_err(|e| {
            eyre::eyre!(
                "EVM prove[{}]: failed to decode final-side deferral path bytes \
                 ({} bytes) from DeferralTailDispatch: {}",
                proof_uuid,
                tail.final_merkle_path_bytes.len(),
                e,
            )
        })?;

    // Recompute the INITIAL-side depth-independent path locally
    // from the exe (the initial memory tree is free and deterministic from
    // the exe). The initial memory image is fully
    // determined by `exe.init_memory`, so no executor state is needed.
    // `vm_config.as_ref()` returns `&SystemConfig` (the SDK's
    // `VmConfig: AsRef<SystemConfig>` impl), matching what
    // `RootProverInstance::new` uses to read `memory_dimensions`.
    let system_config = deferral_artifacts
        .cached_pk
        .app_pk
        .app_vm_pk
        .vm_config
        .as_ref();
    let memory_dimensions = system_config.memory_config.memory_dimensions();
    let initial_tree =
        crate::deferral_merkle::build_initial_memory_tree(exe.as_ref(), system_config);
    let initial_depth_indep_path =
        crate::deferral_merkle::extract_deferral_auth_path(&memory_dimensions, &initial_tree);

    // Translate the manager-supplied wire metadata into the SDK type.
    let mut metadata = SdkInternalLayerMetadata {
        internal_recursive_layer: tail.layer_metadata.internal_recursive_layer,
        internal_node_idx: tail.layer_metadata.internal_node_idx,
        proofs_type: match tail.layer_metadata.proofs_type {
            protocol::ProofsTypeWire::Vm => ProofsType::Vm,
            protocol::ProofsTypeWire::Deferral => ProofsType::Deferral,
            protocol::ProofsTypeWire::Mix => ProofsType::Mix,
            protocol::ProofsTypeWire::Combined => ProofsType::Combined,
        },
    };

    // 1. prove_def — turn the per-circuit deferral inputs into aggregated
    //    deferral proofs, then collapse to one `DeferralProof`.
    info!(
        "EVM prove[{}]: prove_def over {} deferral input(s)",
        proof_uuid,
        def_inputs.len()
    );
    let def_hook_proofs = def_prover
        .multi_deferral_circuit_prover
        .prove(&def_inputs)
        .map_err(|e| eyre::eyre!("multi_deferral_circuit_prover.prove failed: {}", e))?;
    let (def_proof, def_internal_recursive_layer) = def_prover
        .agg_prover
        .prove_def(def_hook_proofs)
        .map_err(|e| eyre::eyre!("agg_prover.prove_def failed: {}", e))?;

    // 2. prove_mixed — fold the deferral aggregation into the VM-tree's
    //    final internal proof, balancing recursion depths.
    info!(
        "EVM prove[{}]: prove_mixed (def_irl={})",
        proof_uuid, def_internal_recursive_layer
    );
    stark_proof = agg_prover
        .prove_mixed(
            stark_proof,
            def_proof,
            &mut metadata,
            def_internal_recursive_layer,
        )
        .map_err(|e| eyre::eyre!("agg_prover.prove_mixed failed: {}", e))?;

    // 3. wrap_proof — one additional internal_recursive layer to keep
    //    proof size down (canonical sequence in
    //    openvm/crates/sdk/src/prover/stark.rs:120-126).
    info!(
        "EVM prove[{}]: wrap_proof (one internal_recursive layer)",
        proof_uuid
    );
    stark_proof = agg_prover
        .wrap_proof(stark_proof, &mut metadata)
        .map_err(|e| eyre::eyre!("agg_prover.wrap_proof failed: {}", e))?;

    // Read `depth` from `DeferralPvs[DEF_PVS_AIR_ID]` of the merged
    // proof, then finalize both depth-independent paths (zero-pad the
    // first `depth` siblings). Mirrors openvm `stark.rs:142-152`.
    let def_pvs_slice = stark_proof
        .inner
        .public_values
        .get(DEF_PVS_AIR_ID)
        .ok_or_else(|| {
            eyre::eyre!(
                "EVM prove[{}]: merged proof has only {} public-value AIRs but DeferralPvs \
                 lives at DEF_PVS_AIR_ID={}; deferral keyset mismatch?",
                proof_uuid,
                stark_proof.inner.public_values.len(),
                DEF_PVS_AIR_ID,
            )
        })?
        .as_slice();
    let def_pvs: &DeferralPvs<proof::F> = def_pvs_slice.borrow();
    let depth =
        openvm_stark_backend::p3_field::PrimeField32::as_canonical_u32(&def_pvs.depth) as usize;

    let final_merkle_proof =
        crate::deferral_merkle::finalize_deferral_path(&final_depth_indep_path, depth);
    let initial_merkle_proof =
        crate::deferral_merkle::finalize_deferral_path(&initial_depth_indep_path, depth);
    let deferral_merkle_proofs = DeferralMerkleProofs {
        initial_merkle_proof,
        final_merkle_proof,
    };

    let proofs_type_label = match metadata.proofs_type {
        ProofsType::Vm => "Vm",
        ProofsType::Deferral => "Deferral",
        ProofsType::Mix => "Mix",
        ProofsType::Combined => "Combined",
    };
    info!(
        "EVM prove[{}]: deferral merge complete (final metadata: irl={} idx={} type={}); \
         attached deferral_merkle_proofs (depth={}, overall_height={}).",
        proof_uuid,
        metadata.internal_recursive_layer,
        metadata.internal_node_idx,
        proofs_type_label,
        depth,
        memory_dimensions.overall_height(),
    );

    // Re-encode the merged inner proof as a `ProofWithPublicValue` so the
    // root prove step below picks it up via its existing API
    // (RootProverJob{final_internal_proof}). Merkle proofs travel
    // alongside via `RootProverJob.deferral_merkle_proofs`.
    Ok((
        proof::ProofWithPublicValue {
            proof: stark_proof.inner,
            user_public_values: Some(stark_proof.user_pvs_proof),
        },
        deferral_merkle_proofs,
    ))
}

#[cfg(all(test, feature = "mock-provers"))]
mod evm_prove_tests {
    use super::*;
    use crate::config::ProversConfig;
    use crate::prover_pool::ProverPool;
    use crate::provers::InternalProverJob;
    use proof::{ProofWithPublicValue, F};
    use protocol::{ProgramRef, ProofContext, ProofType};

    fn make_pool() -> ProverPool {
        let config = ProversConfig::default();
        // ProverPool::new in mock mode needs the provers config + role.
        // `Full` builds every prover class (today's behavior) so the EVM
        // prove tests still have root/halo2 workers.
        ProverPool::new(&config, protocol::WorkerRole::Full).expect("pool")
    }

    fn make_ctx(proof_uuid: &str, proof_type: ProofType) -> ProofContext {
        let mut ctx = ProofContext::new(
            proof_uuid.to_string(),
            ProgramRef::new("test", 1),
            Default::default(),
        );
        ctx.proof_type = proof_type;
        ctx
    }

    fn make_final_internal_job(ctx: &ProofContext) -> InternalProverJob {
        // The mock internal prover just sleeps and emits a byte-vec proof;
        // child_proofs only needs to be non-empty and pass the segment-range check.
        let child_proofs = vec![ProofWithPublicValue::<F> {
            proof: vec![0u8; 64],
            public_values: vec![F::default(); 4],
        }];
        InternalProverJob {
            context: ctx.clone(),
            child_proofs,
            layer_idx: 2,
            segment_start: 0,
            segment_end: 4,
            is_final_proof: true,
            proof_has_deferral: false,
        }
    }

    /// The final-internal worker runs the tail-merge / merkle-prep half and
    /// ships a ready-for-evm `Internal` result (`ready_for_evm == true`) carrying
    /// the post-merge proof bytes — never the raw internal, and never the `Evm`
    /// proof (root → halo2 is the dispatched `EvmProve` step). This is the
    /// unified path for EVERY final-internal worker of an `Evm` proof.
    #[tokio::test]
    async fn evm_prep_emits_ready_for_evm_internal() {
        let pool = make_pool();
        let ctx = make_ctx("proof-evm", ProofType::Evm);
        let job = make_final_internal_job(&ctx);

        let internal_result = pool.submit_internal_job(job).await.expect("internal");
        let internal_results = match internal_result {
            crate::provers::ProverResult::Success(r) => r,
            crate::provers::ProverResult::Error(e) => panic!("internal prove failed: {}", e),
            crate::provers::ProverResult::Canceled => panic!("unexpected cancellation"),
        };

        // `drive_evm_prep_and_post` POSTs to the manager; the pure prep
        // logic is otherwise inline there. Assert the mock internal prover
        // produced a decodable Internal proof that the prep step would ship as
        // ready-for-evm — i.e. the final-internal stage yields exactly the
        // Internal result the ready-for-evm handoff is built from.
        let final_internal = match &internal_results[0] {
            protocol::ProofResult::Internal(ip) => ip,
            other => panic!(
                "internal stage emitted {:?}, expected Internal",
                other.kind()
            ),
        };
        assert!(
            final_internal.state.proof.is_some(),
            "final internal proof carries proof bytes for the ready-for-evm handoff"
        );
        // The raw internal is NOT ready-for-evm until the prep step sets the flag;
        // this pins that the fresh internal result starts un-flagged.
        assert!(
            !final_internal.state.ready_for_evm,
            "raw internal result is not ready-for-evm before prep"
        );
    }

    /// Dedicated `EvmProve` step: `run_evm_prove` runs root → halo2 on the
    /// handed-off (finished) internal proof and returns the Evm proof. This is
    /// what the `EvmProve` handler drives on an eligible `runs_evm_prove()`
    /// worker (`Full` or `EvmDedicated`) after the manager dispatches the EVM step.
    /// (Under mock the deferral params are cfg'd out; deferral vs non-deferral
    /// are identical here because the merge already ran on the final-internal
    /// worker.)
    #[tokio::test]
    async fn evm_prove_emits_evm_result() {
        let pool = make_pool();
        let ctx = make_ctx("proof-evm-prove", ProofType::Evm);

        // The finished (post-tail-merge) internal proof handed off to the
        // dedicated worker — decoded from `EvmProveRequest::internal_proof_bytes`
        // in the real handler; constructed directly here.
        let finished_internal = ProofWithPublicValue::<F> {
            proof: vec![0u8; 64],
            public_values: vec![F::default(); 4],
        };

        let evm_result = run_evm_prove(&pool, &ctx.proof_uuid, &ctx, finished_internal)
            .await
            .expect("evm prove");

        let evm = match evm_result {
            protocol::ProofResult::Evm(e) => e,
            other => panic!("run_evm_prove returned {:?}, expected Evm", other.kind()),
        };
        assert!(
            evm.state.root_prove_time_ms > 0,
            "root prove time should be folded into the Evm result, got {}",
            evm.state.root_prove_time_ms
        );
        assert!(
            evm.state.proof.is_some(),
            "dedicated EVM step produces the Evm proof"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdk_v2::StdIn;

    #[test]
    fn proof_uuid_validation_rejects_empty_and_invalid_chars() {
        for proof_uuid in [
            "",
            "with/slash",
            "with\\slash",
            "with.dot",
            "with space",
            "colon:bad",
        ] {
            assert!(
                validate_uploaded_proof_uuid(proof_uuid).is_err(),
                "{proof_uuid:?} should be rejected"
            );
        }
    }

    #[test]
    fn proof_uuid_validation_accepts_allowlist_and_enforces_length_cap() {
        assert!(validate_uploaded_proof_uuid("proof-ABC_123").is_ok());
        assert!(validate_uploaded_proof_uuid(&"a".repeat(MAX_PROOF_UUID_LEN)).is_ok());
        assert!(validate_uploaded_proof_uuid(&"a".repeat(MAX_PROOF_UUID_LEN + 1)).is_err());
    }

    #[test]
    fn readiness_full_and_normal_ignore_evm_artifacts() {
        // Full/StarkOnly are ready on stark artifacts + provers alone — never
        // gated on EVM artifacts (byte-for-byte today's behavior). This is the
        // "StarkOnly ready WITHOUT EVM artifacts" acceptance case, and confirms
        // the default (Full) path is unchanged.
        for role in [protocol::WorkerRole::Full, protocol::WorkerRole::StarkOnly] {
            assert_eq!(
                evaluate_readiness(role, true, true, false),
                Readiness::Ready,
                "{role:?} should be ready without EVM artifacts",
            );
            assert_eq!(
                evaluate_readiness(role, false, true, true),
                Readiness::ArtifactsNotLoaded,
            );
            assert_eq!(
                evaluate_readiness(role, true, false, true),
                Readiness::ProversInitializing,
            );
        }
    }

    #[test]
    fn readiness_evm_dedicated_requires_evm_artifacts() {
        // EvmDedicated additionally requires EVM artifacts: ready only when the
        // halo2 key is loaded, not-ready otherwise even with stark artifacts +
        // provers up.
        assert_eq!(
            evaluate_readiness(protocol::WorkerRole::EvmDedicated, true, true, true),
            Readiness::Ready,
        );
        assert_eq!(
            evaluate_readiness(protocol::WorkerRole::EvmDedicated, true, true, false),
            Readiness::EvmArtifactsNotLoaded,
        );
        // Artifacts are still the first gate, ahead of the EVM check.
        assert_eq!(
            evaluate_readiness(protocol::WorkerRole::EvmDedicated, false, true, false),
            Readiness::ArtifactsNotLoaded,
        );
    }

    #[tokio::test]
    async fn cleanup_stale_uploaded_inputs_only_removes_old_inactive_dirs() {
        let temp_dir = tempfile::tempdir().unwrap();
        let active_uploaded_proofs = DashSet::new();

        let stale_path = uploaded_input_work_dir(temp_dir.path(), "stale-proof");
        let active_path = uploaded_input_work_dir(temp_dir.path(), "active-proof");
        fs::create_dir_all(&stale_path).await.unwrap();
        fs::create_dir_all(&active_path).await.unwrap();
        active_uploaded_proofs.insert("active-proof".to_string());

        tokio::time::sleep(Duration::from_millis(25)).await;

        let fresh_path = uploaded_input_work_dir(temp_dir.path(), "fresh-proof");
        fs::create_dir_all(&fresh_path).await.unwrap();

        let removed = cleanup_stale_uploaded_inputs(
            temp_dir.path(),
            &active_uploaded_proofs,
            Duration::from_millis(10),
        )
        .await;

        assert_eq!(removed, 1);
        assert!(!stale_path.exists());
        assert!(active_path.exists());
        assert!(fresh_path.exists());
    }

    #[test]
    fn staged_deferral_paths_share_uploaded_input_root() {
        // Worker-resolvable paths used by ShardedAppProveRequest::deferral_state_paths
        // when the manager fans out (vs. caller-supplied paths when input_already_uploaded=true).
        let state = super::staged_deferral_state_path("proof-abc", 0);
        let input = super::staged_deferral_input_path("proof-abc", 2);
        assert!(state.starts_with(super::uploaded_input_root()));
        assert!(input.starts_with(super::uploaded_input_root()));
        assert!(state
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.contains("deferral_state_0"))
            .unwrap_or(false));
        assert!(input
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.contains("deferral_input_2"))
            .unwrap_or(false));
    }

    #[test]
    fn compact_upload_serializes_to_same_stdin_as_bincode_mode() {
        let raw_input = vec![1u8, 2, 3, 4, 5, 6, 7];

        let compact =
            prepare_uploaded_input_bytes(UploadedInputFormat::CompactBytes, &raw_input).unwrap();
        let direct = prepare_uploaded_input_bytes(UploadedInputFormat::BincodeStdin, &{
            let mut stdin: StdIn<sdk_v2::F> = StdIn::default();
            stdin.write_bytes(&raw_input);
            bincode::serialize(&stdin).unwrap()
        })
        .unwrap();

        assert_eq!(compact, direct);
    }
}
