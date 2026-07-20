//! HTTP server for the Edge worker.

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
use eyre::{eyre, Result};
use protocol::parse_programs_env;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::info;

use crate::artifacts::ArtifactStore;
use crate::config::WorkerConfig;
use crate::handlers::{
    handle_cancel_proof, handle_recursion_prove, handle_register_program, handle_sharded_app_prove,
    handle_upload_deferral_input, handle_upload_deferral_state, handle_upload_input,
    handle_upload_input_compact, healthz, readyz, uploaded_input_janitor_task, AppState,
    WorkerInfo, STALE_UPLOADED_INPUT_JANITOR_INTERVAL, STALE_UPLOADED_INPUT_TTL,
};
use crate::prover_pool::ProverPool;
use crate::result_client::{registration_task, ResultClient};

/// Run the Edge worker HTTP server.
pub async fn run_server(config: WorkerConfig) -> Result<()> {
    let cancel_token = CancellationToken::new();

    // Seed the loadout from `EDGE_PROGRAMS`, for a worker whose artifacts are
    // already staged on disk. A registration-driven worker leaves it unset and
    // starts empty, taking its programs from `/register_program`.
    let programs = parse_programs_env().map_err(|e| eyre!("Failed to parse EDGE_PROGRAMS: {e}"))?;
    info!(
        "Worker loadout: {} program(s) — {}",
        programs.len(),
        programs
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Register with manager first.
    // Worker IDs are deterministic from config; registration validates that the
    // manager agrees with this exact URL -> worker_id mapping.
    let worker_url = config.effective_worker_url();
    let registration_client =
        ResultClient::new(&config.worker.manager_url, config.worker.prover_id)?;
    let worker_id = registration_client
        .register_worker(
            &worker_url,
            config.worker.prover_id,
            config.provers.max_app_provers,
            config.provers.max_leaf_provers,
            config.provers.max_internal_provers,
            programs.clone(),
            config.worker.worker_role,
        )
        .await?;
    if worker_id != config.worker.prover_id {
        return Err(eyre!(
            "manager returned worker_id {} but config prover_id is {}",
            worker_id,
            config.worker.prover_id
        ));
    }

    // Initialize artifact store (loads shared keys + per-program vmexes
    // for real provers).
    ArtifactStore::init(&config.artifacts, programs.clone())?;
    if let Some(store) = ArtifactStore::global() {
        info!(
            "Artifact store initialized: programs={}, ready={}",
            store.configured_programs().len(),
            store.is_ready()
        );
    }

    // Build the per-program `AppExecutionInstances` (CPU AOT compile,
    // ~115 s each) IN PARALLEL across all configured programs. The
    // result is held forever on this worker; only the GPU prover is
    // swapped on program change at runtime.
    //
    // Role-gated startup: an `EvmDedicated` worker runs only
    // the EVM step, so it skips this ~115 s AOT compile and the
    // app/leaf/internal prover instances entirely — but it STILL loads the
    // STARK-side proving keys root prove needs (`agg_stark_pk`, `app_pk`,
    // plus root/halo2/deferral-cached keys), which `ArtifactStore::init`
    // above already loaded. `Full`/`StarkOnly` build the context as before.
    //
    // An empty loadout has nothing to build here either, since the context
    // arrives with the first `/register_program`, which the prover threads
    // wait for.
    #[cfg(not(feature = "mock-provers"))]
    let app_ctx = if programs.is_empty() {
        info!("Empty loadout; app execution context arrives via /register_program");
        None
    } else if config.worker.worker_role.runs_stark_proving() {
        Some(build_app_worker_context(&programs)?)
    } else {
        info!(
            "Worker role {:?} skips app-execution-context build (no app/leaf/internal work); \
             STARK-side keys remain loaded for the EVM step",
            config.worker.worker_role
        );
        None
    };

    // Create prover pool. App workers are program-agnostic in the swap
    // design — they lazily load + swap a `ProverType` on first job per
    // program. Leaf and internal pools build their prover instances
    // eagerly at thread startup, as before. The role gates which prover
    // classes are built (see `ProverPool::new`).
    let prover_pool = ProverPool::new(
        &config.provers,
        config.worker.worker_role,
        #[cfg(not(feature = "mock-provers"))]
        app_ctx,
    )?;

    // Create shared state using the deterministic configured worker_id.
    let state = Arc::new(AppState {
        prover_pool,
        result_client: ResultClient::new(&config.worker.manager_url, config.worker.prover_id)?,
        worker_config: WorkerInfo {
            prover_id: config.worker.prover_id,
            num_provers: config.worker.num_provers,
            max_app_provers: config.provers.max_app_provers,
            default_segment_memory: config.provers.default_segment_memory,
            worker_role: config.worker.worker_role,
        },
        active_uploaded_proofs: Default::default(),
    });

    // Build router.
    //
    // Disable body limit for routes that receive large payloads in the
    // trusted internal setup:
    // - upload_input: raw input files (can be 30MB+)
    // - edge_prove_work: leaf/internal prove requests with serialized proofs (can be large)
    // - register_program: a guest ELF plus its VM config
    let large_payload_routes = Router::new()
        .route("/register_program", post(handle_register_program))
        .route("/upload_input/{proof_uuid}", post(handle_upload_input))
        .route(
            "/upload_input_compact/{proof_uuid}",
            post(handle_upload_input_compact),
        )
        .route(
            "/upload_deferral_state/{proof_uuid}",
            post(handle_upload_deferral_state),
        )
        .route(
            "/upload_deferral_input/{proof_uuid}",
            post(handle_upload_deferral_input),
        )
        .route("/recursion_prove", post(handle_recursion_prove))
        .layer(DefaultBodyLimit::disable()); // No limit for large input files

    let app = Router::new()
        // Health endpoints
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        // Edge endpoints (per API spec)
        .merge(large_payload_routes) // Routes with larger body limit
        .route("/sharded_app_prove", post(handle_sharded_app_prove))
        .route("/cancel_proof/{proof_uuid}", post(handle_cancel_proof))
        // Add state
        .with_state(state.clone())
        // Add middleware
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    // Start registration background task (re-validates the deterministic worker_id mapping).
    let bg_worker_url = config.effective_worker_url();
    let bg_registration_client =
        ResultClient::new(&config.worker.manager_url, config.worker.prover_id)?;
    let registration_cancel = cancel_token.clone();
    let bg_worker_role = config.worker.worker_role;
    tokio::spawn(async move {
        registration_task(
            bg_registration_client,
            bg_worker_url,
            config.worker.prover_id,
            config.provers.max_app_provers,
            config.provers.max_leaf_provers,
            config.provers.max_internal_provers,
            bg_worker_role,
            Duration::from_secs(30),
            registration_cancel,
        )
        .await;
    });

    let janitor_state = state.clone();
    let janitor_cancel = cancel_token.clone();
    tokio::spawn(async move {
        uploaded_input_janitor_task(
            janitor_state,
            STALE_UPLOADED_INPUT_JANITOR_INTERVAL,
            STALE_UPLOADED_INPUT_TTL,
            janitor_cancel,
        )
        .await;
    });

    // Start server
    let listener = TcpListener::bind(&config.server.listen_addr).await?;
    info!("Edge Worker listening on {}", config.server.listen_addr);

    // Setup graceful shutdown
    let shutdown_cancel = cancel_token.clone();
    let shutdown_signal = async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
        info!("Shutdown signal received");
        shutdown_cancel.cancel();
    };

    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;

    info!("Server shutdown complete");
    Ok(())
}

/// Build the per-program `AppExecutionInstances` map in parallel via
/// rayon. Each entry is the result of an ~115 s AOT compile (gcc) that
/// briefly allocates ~1.66 GB on the GPU while building the
/// interpreters, then frees it. Parallelizing across the loadout lets
/// total boot time stay roughly constant in N programs (bounded by one
/// program's build time + parallelism overhead).
#[cfg(not(feature = "mock-provers"))]
fn build_app_worker_context(
    programs: &[protocol::ProgramRef],
) -> Result<std::sync::Arc<crate::prover_pool::AppWorkerContext>> {
    use rayon::prelude::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    let store = ArtifactStore::global()
        .ok_or_else(|| eyre!("ArtifactStore not initialized before app worker context build"))?;
    let edge_artifacts = store
        .get_edge_artifacts()
        .ok_or_else(|| eyre!("Edge artifacts not loaded"))?;
    let app_pk = edge_artifacts.app_pk.clone();

    let total_start = std::time::Instant::now();
    info!(
        "Building AppExecutionInstances for {} program(s) in parallel...",
        programs.len()
    );

    let entries: Vec<
        Result<(
            protocol::ProgramRef,
            Arc<crate::provers::AppExecutionInstances>,
        )>,
    > = programs
        .par_iter()
        .map(|p| {
            let exe = store
                .vmexe(p)
                .ok_or_else(|| eyre!("vmexe for {p} missing from artifact store"))?;
            let instances = crate::provers::AppExecutionInstances::new(p, &app_pk, exe)?;
            Ok((p.clone(), Arc::new(instances)))
        })
        .collect();

    let execution_instances: HashMap<_, _> = entries.into_iter().collect::<Result<_>>()?;

    info!(
        "All AppExecutionInstances ready in {} ms (programs={})",
        total_start.elapsed().as_millis(),
        programs.len()
    );

    // Publish the context so a later `/register_program` extends it rather
    // than replacing it, and so app workers see programs registered after
    // boot.
    let app_ctx = Arc::new(crate::prover_pool::AppWorkerContext {
        app_pk,
        execution_instances,
    });
    store.publish_app_worker_context(app_ctx.clone());

    Ok(app_ctx)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[tokio::test]
    async fn test_health_endpoint() {
        // This test would require mocking the prover pool
        // For now, just verify the route is set up correctly
    }
}
