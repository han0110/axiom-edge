//! Tests that a settled proof frees the single proof slot before a client can observe it.

use crate::config::{MetricsConfig, ProofConfig, ProversConfig, ServerConfig, TelemetryConfig};
use crate::handlers::*;
use futures::poll;
use protocol::{EvmProof, EvmProofState, ProofType, WorkerRole};
use std::future::Future;

const PROOF_UUID: &str = "p-slot";

fn test_state(dir: &std::path::Path) -> Arc<AppState> {
    let config = ManagerConfig {
        server: ServerConfig {
            listen_addr: "127.0.0.1:0".to_string(),
            num_workers: 1,
            artifacts_path: dir.to_path_buf(),
        },
        proof: ProofConfig {
            persist_final_proofs_dir: Some(dir.to_path_buf()),
            ..Default::default()
        },
        provers: ProversConfig {
            max_app_provers: 1,
            max_leaf_provers: 1,
            max_internal_provers: 1,
        },
        lifecycle: Default::default(),
        telemetry: TelemetryConfig::default(),
        metrics: MetricsConfig {
            output_dir: dir.to_path_buf(),
            ..Default::default()
        },
    };
    Arc::new(AppState::new(
        config,
        vec![ProgramRef::new("test-program", 1)],
    ))
}

fn evm_context() -> ProofContext {
    let mut context = ProofContext::new(
        PROOF_UUID.to_string(),
        ProgramRef::new("test-program", 1),
        Default::default(),
    );
    context.proof_type = ProofType::Evm;
    context
}

/// Registers an in-progress proof that takes the single proof slot.
fn admit_proof(state: &AppState) -> Arc<Mutex<ProofState>> {
    let proof = Arc::new(Mutex::new(ProofState::new(
        evm_context(),
        u64::MAX,
        1,
        4,
        3,
        300,
    )));
    state
        .proof_states
        .insert(PROOF_UUID.to_string(), proof.clone());
    let worker = RegisteredWorker {
        worker_url: "http://worker-0:8001".to_string(),
        last_seen: chrono::Utc::now(),
        worker_role: WorkerRole::Full,
    };
    state
        .state_store
        .init_proof(PROOF_UUID, vec![(0, worker)], 48);
    assert!(state.state_store.has_any_proofs());
    proof
}

/// The worker result that completes the proof.
fn final_evm_result() -> Bytes {
    let payload = ResultPayload {
        worker_id: 0,
        proof_uuid: PROOF_UUID.to_string(),
        result: MessageEnvelope::with_metadata(ProofResult::Evm(EvmProof {
            context: evm_context(),
            state: EvmProofState {
                proof: Some(vec![1u8; 128]),
                prove_time_ms: 0,
                root_prove_time_ms: 0,
                sub_metrics: Default::default(),
            },
        })),
    };
    Bytes::from(bincode::serialize(&payload).expect("result payload encodes"))
}

/// Runs `settle` up to its `turn`-th acquisition of the proof lock and
/// queues a client read right behind that acquisition. Returns the status
/// that the client reads and whether the single proof slot is still taken
/// at that moment.
async fn read_after_lock_turn(
    state: &AppState,
    proof: &Mutex<ProofState>,
    settle: impl Future,
    turn: usize,
) -> (ProofStatus, bool) {
    tokio::pin!(settle);
    let mut gate = proof.lock().await;
    assert!(poll!(&mut settle).is_pending());
    for _ in 1..turn {
        let next_gate = proof.lock();
        tokio::pin!(next_gate);
        assert!(poll!(&mut next_gate).is_pending());
        drop(gate);
        assert!(poll!(&mut settle).is_pending());
        gate = next_gate.await;
    }
    let read = async {
        let status = proof.lock().await.status.clone();
        (status, state.state_store.has_any_proofs())
    };
    tokio::pin!(read);
    assert!(poll!(&mut read).is_pending());
    drop(gate);
    assert!(poll!(&mut settle).is_pending());
    let observed = read.await;
    settle.await;
    observed
}

/// A client that reads `Completed` can start the next proof at once.
#[tokio::test]
async fn completion_frees_the_slot_before_it_is_observable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = test_state(dir.path());
    let proof = admit_proof(&state);

    let settle = proof_result(State(state.clone()), final_evm_result());
    let (status, slot_taken) = read_after_lock_turn(&state, &proof, settle, 1).await;

    assert_eq!(status, ProofStatus::Completed);
    assert!(!slot_taken, "the next /start_proof gets 409");
}

/// A client that reads the drained `Failed` can start the next proof at once.
#[tokio::test]
async fn drained_failure_frees_the_slot_before_it_is_observable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = test_state(dir.path());
    let proof = admit_proof(&state);
    proof.lock().await.mark_failing("worker lost".to_string());
    state.state_store.release_worker(PROOF_UUID, 0).await;

    // The first lock turn is the `Failing` check and the second one settles.
    let settle = try_finalize_failing_proof(&state, PROOF_UUID);
    let (status, slot_taken) = read_after_lock_turn(&state, &proof, settle, 2).await;

    assert_eq!(status, ProofStatus::Failed("worker lost".to_string()));
    assert!(!slot_taken, "the next /start_proof gets 409");
}

/// A `Failing` proof with busy workers keeps the slot until they drain.
#[tokio::test]
async fn undrained_failure_keeps_the_slot() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = test_state(dir.path());
    let proof = admit_proof(&state);
    proof.lock().await.mark_failing("worker lost".to_string());

    try_finalize_failing_proof(&state, PROOF_UUID).await;

    assert_eq!(
        proof.lock().await.status,
        ProofStatus::Failing("worker lost".to_string())
    );
    assert!(state.state_store.has_any_proofs());
}
