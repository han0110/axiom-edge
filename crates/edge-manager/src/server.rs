//! HTTP server for the Edge manager.

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
use eyre::{eyre, Result};
use protocol::parse_programs_env;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;

use crate::config::ManagerConfig;
use crate::handlers::{
    cancel_proof, download_proof, download_vk, get_loadout, healthz, list_workers, proof_debug,
    proof_events, proof_result, proof_state, proof_timeout_watchdog_task, readyz_handler,
    register_worker, start_proof, upload_input, AppState, PROOF_TIMEOUT_WATCHDOG_INTERVAL,
};

/// Run the HTTP server.
pub async fn run_server(config: ManagerConfig) -> Result<()> {
    // Parse the deployment's program loadout once at startup. The same
    // EDGE_PROGRAMS value is also injected onto every worker container;
    // /register_worker rejects workers whose loaded_programs differs.
    let programs = parse_programs_env().map_err(|e| eyre!("Failed to parse EDGE_PROGRAMS: {e}"))?;
    tracing::info!(
        "Manager loadout: {} program(s) — {}",
        programs.len(),
        programs
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let state = Arc::new(AppState::new(config.clone(), programs));
    let cancel_token = CancellationToken::new();

    // Spawn the proof-timeout watchdog. It periodically scans
    // `state.proof_states` and marks any proof whose wall-clock age exceeds
    // its `timeout_secs` as `Failed("timed out...")`, freeing the proof's
    // scheduler slot so workers become available for new dispatches.
    let watchdog_state = state.clone();
    let watchdog_cancel = cancel_token.clone();
    tokio::spawn(async move {
        proof_timeout_watchdog_task(
            watchdog_state,
            PROOF_TIMEOUT_WATCHDOG_INTERVAL,
            watchdog_cancel,
        )
        .await;
    });

    // Large-body routes: axum's default 2 MB body limit is disabled for them.
    // - `/upload_input`: ONE multipart request carries all of a proof's input
    //   (main `StdIn` + any `DeferralState`/`DeferralInput` parts), so the
    //   caller makes a single upload call before `/start_proof` regardless of
    //   deferral circuit count.
    // - `/proof_result`: workers POST bincode `ResultPayload`s embedding full
    //   proof bytes. Typical app/leaf/internal results are well under 2 MB
    //   today, but a larger program or segment config can cross it — and a
    //   413 here is a silent proof stall (the worker's streaming loop drops
    //   the result without retry).
    let large_body_routes = Router::new()
        .route("/upload_input/{proof_uuid}", post(upload_input))
        .route("/proof_result", post(proof_result))
        .layer(DefaultBodyLimit::disable());

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/start_proof", post(start_proof))
        .route("/register_worker", post(register_worker))
        .route("/workers", get(list_workers))
        .route("/readyz", get(readyz_handler))
        .route("/loadout", get(get_loadout))
        .route("/proof_state/{proof_uuid}", get(proof_state))
        // Server-sent status stream, so a caller follows a proof without
        // polling /proof_state.
        .route("/proof_events/{proof_uuid}", get(proof_events))
        .route("/proof_debug/{proof_uuid}", get(proof_debug))
        .route("/cancel_proof", post(cancel_proof))
        // Caller-facing downloads: per-program verification baselines from
        // the mounted export, and the completed proof.
        .route("/vk/{name}", get(download_vk))
        .route("/proof/{proof_uuid}", get(download_proof))
        .merge(large_body_routes)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = config.server.listen_addr.parse()?;
    let listener = TcpListener::bind(addr).await?;

    tracing::info!("Edge Manager listening on {}", addr);

    axum::serve(listener, app).await?;

    cancel_token.cancel();
    Ok(())
}
