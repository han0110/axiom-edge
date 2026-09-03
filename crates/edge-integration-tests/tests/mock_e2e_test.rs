//! Self-contained mock E2E test.
//!
//! Boots one manager and one worker in-process on kernel-assigned (dynamic)
//! ports, registers the worker, kicks off a proof, and waits for it to
//! complete. The mock prover (`mock-provers` feature) returns synthetic
//! results — no GPU, no artifacts, no input file required.
//!
//! Run with:
//!     cargo test --features mock-provers -p edge-integration-tests \
//!                --test mock_e2e_test
//!
//! Goals:
//! - One `cargo test` invocation, no separate build step.
//! - No subprocesses, no hardcoded ports, no global filesystem state.
//! - Exercises the full manager <-> worker control loop: register, dispatch,
//!   sharded app prove, leaf, internal, terminal `Completed`.

#![cfg(feature = "mock-provers")]

use std::net::TcpListener;
use std::sync::Once;
use std::time::{Duration, Instant};

use edge_manager::config::{
    ManagerConfig, MetricsConfig, ProofConfig, ProversConfig as ManagerProversConfig,
    ServerConfig as ManagerServerConfig, TelemetryConfig as ManagerTelemetryConfig,
};
use edge_worker::config::{
    ArtifactsConfig, ProversConfig as WorkerProversConfig, ServerConfig as WorkerServerConfig,
    TelemetryConfig as WorkerTelemetryConfig, WorkerConfig, WorkerSettings,
};
use eyre::Result;
use protocol::{LoadoutResponse, ProgramRef, StartProofRequest};
use reqwest::Client;
use serde_json::Value;

/// One canonical loadout for all tests in this file. Manager + worker
/// both parse `EDGE_PROGRAMS` from the process env at boot; cargo runs
/// tests in the same process, so setting it once is sufficient.
static INIT_ENV: Once = Once::new();

fn ensure_env() {
    INIT_ENV.call_once(|| {
        // Two programs — single-program tests just target one of them;
        // the multi-program test exercises both.
        std::env::set_var(
            "EDGE_PROGRAMS",
            r#"[{"name":"mock-program","version":1},{"name":"keccak","version":1}]"#,
        );
    });
}

/// Bind a TCP socket to 127.0.0.1:0, grab the kernel-assigned port, drop the
/// socket, return the port. There's a small TOCTOU window before the server
/// binds, but it's good enough for tests.
fn pick_free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

/// Serializes the E2E tests. Each boots a manager + workers on freshly
/// `pick_free_port()`'d ephemeral ports; that pick-then-bind window means
/// running the tests concurrently (cargo's default) lets ports collide.
/// On busy CI runners that surfaces as "Address already in use" worker
/// crashes or cross-test 404s (one test's manager dispatching to another
/// test's process). Holding this lock for each test's duration removes the
/// contention — only one test's servers are alive at a time.
static TEST_SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn build_manager_config(manager_port: u16) -> ManagerConfig {
    ManagerConfig {
        server: ManagerServerConfig {
            listen_addr: format!("127.0.0.1:{}", manager_port),
            num_workers: 1,
            artifacts_path: None,
        },
        proof: ProofConfig::default(),
        provers: ManagerProversConfig {
            max_app_provers: 1,
            max_leaf_provers: 1,
            max_internal_provers: 1,
        },
        lifecycle: Default::default(),
        telemetry: ManagerTelemetryConfig {
            log_level: "warn".to_string(),
            otlp_endpoint: None,
        },
        metrics: MetricsConfig::default(),
    }
}

fn build_worker_config(worker_port: u16, manager_port: u16) -> WorkerConfig {
    WorkerConfig {
        server: WorkerServerConfig {
            listen_addr: format!("127.0.0.1:{}", worker_port),
        },
        worker: WorkerSettings {
            prover_id: 0,
            num_provers: 1,
            worker_url: Some(format!("http://127.0.0.1:{}", worker_port)),
            manager_url: format!("http://127.0.0.1:{}", manager_port),
            worker_role: Default::default(),
        },
        artifacts: ArtifactsConfig {
            // Mock provers don't actually load artifacts from disk.
            artifacts_path: None,
            halo2_pk_path: None,
            enable_deferral: false,
        },
        provers: WorkerProversConfig {
            max_app_provers: 1,
            max_leaf_provers: 1,
            max_internal_provers: 1,
            max_root_provers: 1,
            max_halo2_provers: 1,
            default_segment_memory: None,
        },
        telemetry: WorkerTelemetryConfig {
            log_level: "warn".to_string(),
            otlp_endpoint: None,
        },
    }
}

/// Poll an HTTP endpoint until it returns 200 OK or the deadline is hit.
async fn wait_for_ok(client: &Client, url: &str, deadline: Instant) -> Result<()> {
    loop {
        if Instant::now() >= deadline {
            return Err(eyre::eyre!("timed out waiting for {} to return 200", url));
        }
        match client.get(url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
}

/// Poll `/proof_state/{uuid}` until the status is terminal (Completed,
/// Failed, Canceled) or the deadline is hit. Returns the final status
/// string.
async fn wait_for_terminal_status(
    client: &Client,
    manager_url: &str,
    proof_uuid: &str,
    deadline: Instant,
) -> Result<String> {
    loop {
        if Instant::now() >= deadline {
            return Err(eyre::eyre!(
                "timed out waiting for proof {} to reach terminal status",
                proof_uuid
            ));
        }
        let url = format!("{}/proof_state/{}", manager_url, proof_uuid);
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                let body: Value = resp.json().await?;
                let status = body
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if matches!(status.as_str(), "completed" | "failed" | "canceled") {
                    return Ok(status);
                }
            }
            _ => {}
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn boot_manager_and_worker(
    manager_port: u16,
    worker_port: u16,
    bootstrap_deadline: Instant,
    client: &Client,
) -> Result<(String, String)> {
    let manager_url = format!("http://127.0.0.1:{}", manager_port);
    let worker_url = format!("http://127.0.0.1:{}", worker_port);

    let manager_config = build_manager_config(manager_port);
    let _manager_handle = tokio::spawn(async move {
        if let Err(e) = edge_manager::server::run_server(manager_config).await {
            eprintln!("manager exited: {e:?}");
        }
    });

    wait_for_ok(
        client,
        &format!("{}/healthz", manager_url),
        bootstrap_deadline,
    )
    .await?;

    let worker_config = build_worker_config(worker_port, manager_port);
    let _worker_handle = tokio::spawn(async move {
        if let Err(e) = edge_worker::server::run_server(worker_config).await {
            eprintln!("worker exited: {e:?}");
        }
    });

    wait_for_ok(
        client,
        &format!("{}/healthz", worker_url),
        bootstrap_deadline,
    )
    .await?;
    wait_for_ok(
        client,
        &format!("{}/readyz", manager_url),
        bootstrap_deadline,
    )
    .await?;

    Ok((manager_url, worker_url))
}

fn make_proof_request(program: &ProgramRef) -> StartProofRequest {
    StartProofRequest {
        proof_uuid: uuid::Uuid::new_v4().to_string(),
        program: Some(program.clone()),
        labels: Default::default(),
        proof_type: protocol::ProofType::Stark,
        // Flow 1: the mock harness marks the input as already on the workers,
        // so the manager skips fan-out (no real input file needed).
        input_already_uploaded: true,
        segment_memory: None,
        leaf_pack_threshold: None,
        timeout_secs: Some(30),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mock_e2e_single_worker_completes_a_proof() -> Result<()> {
    let _serial = TEST_SERIAL.lock().await;
    ensure_env();
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
    let bootstrap_deadline = Instant::now() + Duration::from_secs(15);
    let (manager_url, _worker_url) = boot_manager_and_worker(
        pick_free_port(),
        pick_free_port(),
        bootstrap_deadline,
        &client,
    )
    .await?;

    let req = make_proof_request(&ProgramRef::new("mock-program", 1));
    let proof_uuid = req.proof_uuid.clone();
    let resp = client
        .post(format!("{}/start_proof", manager_url))
        .json(&req)
        .send()
        .await?;
    assert!(
        resp.status().is_success(),
        "/start_proof failed: status={}, body={:?}",
        resp.status(),
        resp.text().await
    );

    let terminal_deadline = Instant::now() + Duration::from_secs(30);
    let status =
        wait_for_terminal_status(&client, &manager_url, &proof_uuid, terminal_deadline).await?;
    assert_eq!(
        status, "completed",
        "expected proof to reach Completed; got {status}"
    );

    Ok(())
}

/// Multi-program path: loadout has two programs, the manager exposes both
/// via `/loadout`, and each completes when targeted by `/start_proof`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mock_e2e_multi_program() -> Result<()> {
    let _serial = TEST_SERIAL.lock().await;
    ensure_env();
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
    let bootstrap_deadline = Instant::now() + Duration::from_secs(15);
    let (manager_url, _worker_url) = boot_manager_and_worker(
        pick_free_port(),
        pick_free_port(),
        bootstrap_deadline,
        &client,
    )
    .await?;

    // /loadout returns both programs from EDGE_PROGRAMS.
    let loadout: LoadoutResponse = client
        .get(format!("{}/loadout", manager_url))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(loadout.programs.len(), 2);
    let names: std::collections::HashSet<_> = loadout
        .programs
        .iter()
        .map(|p| (p.name.clone(), p.version))
        .collect();
    assert!(names.contains(&("mock-program".to_string(), 1)));
    assert!(names.contains(&("keccak".to_string(), 1)));

    // Run a proof against each program; they run sequentially because the
    // manager enforces single-proof mode.
    for program in [
        ProgramRef::new("mock-program", 1),
        ProgramRef::new("keccak", 1),
    ] {
        let req = make_proof_request(&program);
        let proof_uuid = req.proof_uuid.clone();
        let resp = client
            .post(format!("{}/start_proof", manager_url))
            .json(&req)
            .send()
            .await?;
        assert!(
            resp.status().is_success(),
            "/start_proof for {program} failed: status={}, body={:?}",
            resp.status(),
            resp.text().await
        );
        let terminal_deadline = Instant::now() + Duration::from_secs(30);
        let status =
            wait_for_terminal_status(&client, &manager_url, &proof_uuid, terminal_deadline).await?;
        assert_eq!(
            status, "completed",
            "expected proof for {program} to reach Completed; got {status}"
        );
    }
    Ok(())
}

/// /start_proof rejects programs not in the loadout with a 409 + the
/// canonical loadout in the body — a stable, machine-readable rejection an
/// upstream orchestration layer can forward as-is.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mock_e2e_start_proof_rejects_unknown_program() -> Result<()> {
    let _serial = TEST_SERIAL.lock().await;
    ensure_env();
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
    let bootstrap_deadline = Instant::now() + Duration::from_secs(15);
    let (manager_url, _worker_url) = boot_manager_and_worker(
        pick_free_port(),
        pick_free_port(),
        bootstrap_deadline,
        &client,
    )
    .await?;

    let req = make_proof_request(&ProgramRef::new("not-loaded", 7));
    let resp = client
        .post(format!("{}/start_proof", manager_url))
        .json(&req)
        .send()
        .await?;
    assert_eq!(resp.status(), reqwest::StatusCode::CONFLICT);
    let body: Value = resp.json().await?;
    assert_eq!(body["error"], "program_not_in_loadout");
    assert!(body["current_loadout"].is_array());

    Ok(())
}

/// Single-active-proof gate (the "one active proof at a time" invariant): while
/// one proof is active, a second `/start_proof` with a distinct uuid is
/// rejected with 409 + "Another proof is already running". The second request
/// is fired immediately after the first is admitted — the manager initializes
/// the first proof's scheduler state *before* `/start_proof` returns, so the
/// gate is reliably engaged regardless of mock proving speed (no race).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mock_e2e_rejects_second_active_proof() -> Result<()> {
    let _serial = TEST_SERIAL.lock().await;
    ensure_env();
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
    let bootstrap_deadline = Instant::now() + Duration::from_secs(15);
    let (manager_url, _worker_url) = boot_manager_and_worker(
        pick_free_port(),
        pick_free_port(),
        bootstrap_deadline,
        &client,
    )
    .await?;

    let program = ProgramRef::new("mock-program", 1);

    // Admit the first proof (leave it running).
    let first = make_proof_request(&program);
    let resp = client
        .post(format!("{}/start_proof", manager_url))
        .json(&first)
        .send()
        .await?;
    assert!(
        resp.status().is_success(),
        "first /start_proof should be admitted: status={}, body={:?}",
        resp.status(),
        resp.text().await
    );

    // Immediately submit a second, distinct proof while the first is active.
    let second = make_proof_request(&program);
    assert_ne!(first.proof_uuid, second.proof_uuid);
    let resp = client
        .post(format!("{}/start_proof", manager_url))
        .json(&second)
        .send()
        .await?;
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::CONFLICT,
        "a second proof while one is active must be rejected"
    );
    let body: Value = resp.json().await?;
    assert_eq!(body["error"], "Another proof is already running");

    Ok(())
}

/// Duplicate-uuid dedup: after a proof completes and frees the active slot,
/// resubmitting the SAME proof_uuid is rejected with 409 + "Proof already
/// exists" (the proof-state entry persists past completion until TTL eviction,
/// so the duplicate is caught even though the active-proof gate has cleared).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mock_e2e_rejects_duplicate_proof_uuid() -> Result<()> {
    let _serial = TEST_SERIAL.lock().await;
    ensure_env();
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
    let bootstrap_deadline = Instant::now() + Duration::from_secs(15);
    let (manager_url, _worker_url) = boot_manager_and_worker(
        pick_free_port(),
        pick_free_port(),
        bootstrap_deadline,
        &client,
    )
    .await?;

    let req = make_proof_request(&ProgramRef::new("mock-program", 1));
    let proof_uuid = req.proof_uuid.clone();

    // First submission runs to completion.
    let resp = client
        .post(format!("{}/start_proof", manager_url))
        .json(&req)
        .send()
        .await?;
    assert!(
        resp.status().is_success(),
        "first submission should be admitted: status={}, body={:?}",
        resp.status(),
        resp.text().await
    );
    let terminal_deadline = Instant::now() + Duration::from_secs(30);
    let status =
        wait_for_terminal_status(&client, &manager_url, &proof_uuid, terminal_deadline).await?;
    assert_eq!(status, "completed");

    // Resubmitting the same uuid (active slot now free) is a duplicate -> 409.
    let resp = client
        .post(format!("{}/start_proof", manager_url))
        .json(&req)
        .send()
        .await?;
    assert_eq!(resp.status(), reqwest::StatusCode::CONFLICT);
    let body: Value = resp.json().await?;
    assert_eq!(body["error"], "Proof already exists");

    Ok(())
}

/// Mechanics: the manager is configured with `persist_final_proofs_dir`
/// pointing at a tempdir; once the proof completes, the manager writes
/// `{uuid}.evm.bin` there (per-proof_type branch in
/// `persist_final_proof_to_disk`). The test reads it back to confirm the
/// evm artifact survived the whole pipeline non-empty.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mock_e2e_evm_proof_type_completes_with_evm_artifact() -> Result<()> {
    let _serial = TEST_SERIAL.lock().await;
    ensure_env();
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
    let bootstrap_deadline = Instant::now() + Duration::from_secs(15);

    let persist_dir = tempfile::tempdir()?;
    let manager_port = pick_free_port();
    let worker_port = pick_free_port();
    let manager_url = format!("http://127.0.0.1:{}", manager_port);
    let worker_url = format!("http://127.0.0.1:{}", worker_port);

    // Manager config with persistence enabled so the final evm artifact
    // lands on disk, where the test can inspect it.
    let mut manager_config = build_manager_config(manager_port);
    manager_config.proof.persist_final_proofs_dir = Some(persist_dir.path().to_path_buf());

    let _manager_handle = tokio::spawn(async move {
        if let Err(e) = edge_manager::server::run_server(manager_config).await {
            eprintln!("manager exited: {e:?}");
        }
    });
    wait_for_ok(
        &client,
        &format!("{}/healthz", manager_url),
        bootstrap_deadline,
    )
    .await?;

    let worker_config = build_worker_config(worker_port, manager_port);
    let _worker_handle = tokio::spawn(async move {
        if let Err(e) = edge_worker::server::run_server(worker_config).await {
            eprintln!("worker exited: {e:?}");
        }
    });
    wait_for_ok(
        &client,
        &format!("{}/healthz", worker_url),
        bootstrap_deadline,
    )
    .await?;
    wait_for_ok(
        &client,
        &format!("{}/readyz", manager_url),
        bootstrap_deadline,
    )
    .await?;

    // proof_type=Evm. The worker (built with mock-provers) will run the
    // in-process EVM prove after the final internal proof.
    let req = StartProofRequest {
        proof_uuid: uuid::Uuid::new_v4().to_string(),
        program: Some(ProgramRef::new("mock-program", 1)),
        labels: Default::default(),
        proof_type: protocol::ProofType::Evm,
        input_already_uploaded: true,
        segment_memory: None,
        leaf_pack_threshold: None,
        timeout_secs: Some(30),
    };
    let proof_uuid = req.proof_uuid.clone();
    let resp = client
        .post(format!("{}/start_proof", manager_url))
        .json(&req)
        .send()
        .await?;
    assert!(
        resp.status().is_success(),
        "/start_proof failed: status={}, body={:?}",
        resp.status(),
        resp.text().await
    );

    let terminal_deadline = Instant::now() + Duration::from_secs(30);
    let status =
        wait_for_terminal_status(&client, &manager_url, &proof_uuid, terminal_deadline).await?;
    assert_eq!(
        status, "completed",
        "expected Evm proof to reach Completed; got {status}"
    );

    // Verify the evm artifact landed on disk under `{uuid}.evm.bin`. The
    // file is written by the finalize-proof persistence path inside the
    // manager when proof_type=Evm; presence + non-empty payload is the
    // E2E completion signal that the full root→halo2 EVM prove flowed
    // through (worker → manager → persistence).
    //
    // Persistence runs on a spawn_blocking task after the proof state
    // flips to Completed and notifies waiters. wait_for_terminal_status
    // returns as soon as `/proof_state` reports `completed`, so we briefly
    // poll for the file to appear instead of asserting on a single read.
    let evm_path = persist_dir.path().join(format!("{}.evm.bin", proof_uuid));
    let stark_path = persist_dir.path().join(format!("{}.proof.bin", proof_uuid));
    let evm_artifact_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if evm_path.is_file() {
            break;
        }
        if Instant::now() >= evm_artifact_deadline {
            return Err(eyre::eyre!(
                "expected evm artifact at {} within timeout",
                evm_path.display()
            ));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let bytes = std::fs::read(&evm_path)?;
    assert!(
        !bytes.is_empty(),
        "persisted evm artifact at {} is empty",
        evm_path.display()
    );

    // Sanity check: a Stark proof would have been written to {uuid}.proof.bin
    // instead; for an Evm proof that file should NOT exist (the persistence
    // path picks one based on proof_type).
    assert!(
        !stark_path.is_file(),
        "Evm proof should not have written {}.proof.bin",
        proof_uuid
    );

    Ok(())
}

/// Cancel path: `/cancel_proof` returns 200 + {"status":"canceled"} and drives
/// the proof to a terminal state (it does not wedge). Whether the proof lands
/// in Canceled vs Completed depends on whether the cancel wins the race with
/// the fast mock prover; both are terminal. We deliberately do NOT assert that
/// a new proof can start immediately afterward: cancel frees the *manager's*
/// active slot, but there is no manager->worker cancel channel, so the worker
/// may still be draining the cancelled proof's in-flight work (see
/// ARCHITECTURE.md Failure Model). A new proof can therefore transiently get
/// "503 No available app workers" until that drains.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mock_e2e_cancel_proof_terminalizes() -> Result<()> {
    let _serial = TEST_SERIAL.lock().await;
    ensure_env();
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
    let bootstrap_deadline = Instant::now() + Duration::from_secs(15);
    let (manager_url, _worker_url) = boot_manager_and_worker(
        pick_free_port(),
        pick_free_port(),
        bootstrap_deadline,
        &client,
    )
    .await?;

    let program = ProgramRef::new("mock-program", 1);
    let first = make_proof_request(&program);
    let first_uuid = first.proof_uuid.clone();
    let resp = client
        .post(format!("{}/start_proof", manager_url))
        .json(&first)
        .send()
        .await?;
    assert!(
        resp.status().is_success(),
        "first /start_proof should be admitted: status={}, body={:?}",
        resp.status(),
        resp.text().await
    );

    // Cancel it. The endpoint is idempotent and always 200 {"status":"canceled"}.
    let resp = client
        .post(format!("{}/cancel_proof", manager_url))
        .json(&serde_json::json!({ "proof_uuid": first_uuid }))
        .send()
        .await?;
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: Value = resp.json().await?;
    assert_eq!(body["status"], "canceled");

    // The cancelled proof reaches a terminal state and the slot is freed.
    let terminal_deadline = Instant::now() + Duration::from_secs(30);
    let status =
        wait_for_terminal_status(&client, &manager_url, &first_uuid, terminal_deadline).await?;
    assert!(
        matches!(status.as_str(), "canceled" | "completed"),
        "cancelled proof should be terminal; got {status}"
    );

    Ok(())
}

/// Regression test: `/proof_result` must not sit behind axum's default 2 MB
/// body limit — workers POST bincode `ResultPayload`s embedding full proof
/// bytes, and a 413 there is a silent proof stall (the worker's streaming
/// loop drops the result without retry). A >2 MB payload for an unknown
/// proof must reach the handler (404 "Proof not found"), not be rejected at
/// the extractor (413).
#[tokio::test]
async fn test_proof_result_accepts_bodies_over_2mb() -> Result<()> {
    ensure_env();
    let _serial = TEST_SERIAL.lock().await;

    let client = Client::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    let manager_port = pick_free_port();
    let manager_url = format!("http://127.0.0.1:{}", manager_port);

    let manager_config = build_manager_config(manager_port);
    tokio::spawn(async move {
        if let Err(e) = edge_manager::server::run_server(manager_config).await {
            eprintln!("manager exited: {e:?}");
        }
    });
    wait_for_ok(&client, &format!("{}/healthz", manager_url), deadline).await?;

    let proof_uuid = "body-limit-regression";
    let result = protocol::ProofResult::App(protocol::AppProof {
        context: protocol::ProofContext::new(
            proof_uuid.to_string(),
            ProgramRef::new("mock-program", 1),
            Default::default(),
        ),
        state: protocol::AppProofState {
            proof: Some(vec![0u8; 3 * 1024 * 1024]),
            segment_idx: 0,
            prove_time_ms: 0,
            fastfwd_time_ms: 0,
            stark_prove_time_ms: 0,
            queue_wait_ms: 0,
            metered_time_ms: 0,
            sub_metrics: Default::default(),
            final_merkle_path_bytes: None,
            deferral_merkle_proofs_bytes: None,
            worker_id: 0,
            completed_at_ms: 0,
        },
    });
    let payload = protocol::ResultPayload {
        worker_id: 0,
        proof_uuid: proof_uuid.to_string(),
        result: protocol::MessageEnvelope::with_metadata(result),
    };
    let body = bincode::serialize(&payload)?;
    assert!(
        body.len() > 2 * 1024 * 1024,
        "payload must exceed the default limit"
    );

    let resp = client
        .post(format!("{}/proof_result", manager_url))
        .header("Content-Type", "application/octet-stream")
        .body(body)
        .send()
        .await?;
    assert_ne!(
        resp.status(),
        reqwest::StatusCode::PAYLOAD_TOO_LARGE,
        "/proof_result must accept bodies larger than axum's 2 MB default"
    );
    // Unknown proof: the payload got past the extractor and into the handler.
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);

    Ok(())
}
