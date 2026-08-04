//! Sharded App Prover implementation.
//!
//! This prover handles the initial proof generation for individual segments
//! of the Edge computation. It combines E2 execution with immediate app proof generation.
//!
//! # Architecture
//!
//! The `AppProverInstance` struct holds the reusable prover state (VM, interpreter, etc.)
//! that is expensive to create. Worker threads create one instance at startup and
//! reuse it across all jobs via `prove_sharded_app_with_prover`.
//!
//! The `prove_sharded_app` function is used in mock mode and creates instances internally.

use eyre::Result;
#[cfg(feature = "mock-provers")]
use std::time::Duration;
#[cfg(feature = "mock-provers")]
use std::time::Instant;
use tracing::{info, instrument};

#[cfg(feature = "mock-provers")]
use proof::{ProofWithPublicValue, F};
use protocol::ProofResult;
#[cfg(feature = "mock-provers")]
use protocol::{AppProof, AppProofState, ExecuteE2Result, ExecuteE2State};

use super::{ProverResult, ShardedAppProverJob};

// --- Executor snapshot schedule ---
//
// The executor thread keeps a small deque of VM snapshots so each assigned
// segment can be proved from a nearby start-state instead of replaying from
// instruction 0. The discipline: at the suspend that closes segment `c`,
// save a snapshot iff segment `c + 2` is ours (`saves_snapshot_at`); pop the
// oldest snapshot when proving assigned segment `c`. The `+2` lookahead
// exists because the suspend closing segment `c` lands *after* segment `c`'s
// boundary — i.e. already inside segment `c + 1` — so the freshest snapshot
// usable as a start-state for segment `c + 2` is the suspend-`c` one
// (~1 segment of pure-interpreter fast-forward).

/// Whether this prover seeds the initial (instruction 0) snapshot before the
/// executor loop.
///
/// Provers 0 and 1 hit their first pop (proving segment `prover_id`) before
/// any suspend snapshot targeting it exists, so they need the seed. For
/// `prover_id >= 2` the first save (at `c = prover_id - 2`) precedes the
/// first pop — seeding would leave a permanently unconsumed extra element in
/// the deque, aging every later pop by a full round-robin round and
/// inflating fast-forward from ~1 to ~`num_provers` segments per assigned
/// segment.
#[cfg(any(test, not(feature = "mock-provers")))]
fn seeds_initial_snapshot(prover_id: usize) -> bool {
    prover_id < 2
}

/// Whether the suspend closing segment `curr_idx` should save a snapshot:
/// it is the designated start-state donor for assigned segment
/// `curr_idx + 2`. The single-prover case saves at every suspend (every
/// segment is ours).
#[cfg(any(test, not(feature = "mock-provers")))]
fn saves_snapshot_at(curr_idx: usize, num_provers: usize, prover_id: usize) -> bool {
    num_provers == 1 || (curr_idx + 2) % num_provers == prover_id
}

/// Simulation tests for the snapshot schedule — mirror the executor loop's
/// pop/save order exactly (pop for an assigned segment happens *before* the
/// end-of-iteration save; the final iteration saves nothing).
#[cfg(test)]
mod snapshot_schedule_tests {
    use super::{saves_snapshot_at, seeds_initial_snapshot};
    use std::collections::VecDeque;

    /// Simulate one prover's executor loop and return, per assigned segment,
    /// how many segments of pure-interpreter fast-forward the popped snapshot
    /// implies. Deque entries hold the segment index from whose start the
    /// snapshot is usable: the initial seed is usable from segment 0; a
    /// snapshot saved at the suspend closing segment `k` lands inside segment
    /// `k + 1`, so proving segment `c` from it replays ~`c - (k + 1)`
    /// segments.
    fn simulate(num_provers: usize, prover_id: usize, total_segments: usize) -> Vec<usize> {
        let mut snapshots: VecDeque<usize> = VecDeque::new();
        if seeds_initial_snapshot(prover_id) {
            snapshots.push_back(0);
        }
        let mut replay_distances = Vec::new();
        for curr_idx in 0..total_segments {
            let is_final = curr_idx == total_segments - 1;
            if curr_idx % num_provers == prover_id {
                let usable_from = if num_provers == 1 && curr_idx == 0 {
                    *snapshots.front().expect("seed present for single prover")
                } else {
                    snapshots
                        .pop_front()
                        .expect("snapshot missing for assigned segment")
                };
                assert!(
                    usable_from <= curr_idx,
                    "n={num_provers} p={prover_id}: snapshot for segment {curr_idx} \
                     starts inside segment {usable_from} — past the boundary"
                );
                replay_distances.push(curr_idx - usable_from);
            }
            if is_final {
                break;
            }
            if saves_snapshot_at(curr_idx, num_provers, prover_id) {
                snapshots.push_back(curr_idx + 1);
            }
        }
        replay_distances
    }

    /// Every assigned segment must fast-forward at most ~1 segment from its
    /// popped snapshot, for every (num_provers, prover_id, total_segments)
    /// shape. Before the seeding fix, provers with id >= 2 permanently popped
    /// stale snapshots (~num_provers segments of replay each).
    #[test]
    fn snapshot_pops_are_at_most_one_segment_stale() {
        for num_provers in 1..=6 {
            for prover_id in 0..num_provers {
                for total_segments in 1..=24 {
                    let distances = simulate(num_provers, prover_id, total_segments);
                    for (i, d) in distances.iter().enumerate() {
                        assert!(
                            *d <= 1,
                            "n={num_provers} p={prover_id} total={total_segments}: \
                             assigned segment #{i} replays {d} segments"
                        );
                    }
                }
            }
        }
    }
}

/// Execute sharded app proving for assigned segments.
#[instrument(skip_all, fields(
    proof_id = %job.context.proof_uuid,
    prover_id = job.prover_id,
    num_provers = job.num_provers,
    input_path = %job.input_path
))]
pub fn prove_sharded_app(job: ShardedAppProverJob) -> ProverResult {
    info!(
        "Starting sharded app prove: prover_id={}, num_provers={}",
        job.prover_id, job.num_provers
    );

    match prove_sharded_app_impl(job) {
        Ok(results) => ProverResult::Success(results),
        Err(e) => ProverResult::Error(format!("Sharded app prove failed: {}", e)),
    }
}

#[cfg(feature = "mock-provers")]
fn prove_sharded_app_impl(job: ShardedAppProverJob) -> Result<Vec<ProofResult>> {
    use proof::Segment;

    // Time the execution phase
    let exec_start = Instant::now();

    // Mock implementation - simulate execution time
    std::thread::sleep(Duration::from_millis(100));

    // Generate mock proofs for assigned segments
    // In a real implementation, this would read from input_path and generate actual proofs
    let num_segments = 16; // Example: assume 16 total segments
    let execute_time_ms = exec_start.elapsed().as_millis() as u64;

    let mut results = Vec::new();

    // First worker (prover_id == 0) emits the ExecuteE2Result with segment count
    // This is required for the manager to know the total number of segments
    if job.prover_id == 0 {
        let mock_segments: Vec<Segment> = (0..num_segments).map(|_| Vec::new()).collect();
        let segment_bytes: Vec<Vec<u8>> = mock_segments
            .iter()
            .map(proof::encode_segment)
            .collect::<Result<_>>()?;

        let e2_result = ExecuteE2Result {
            context: job.context.clone(),
            state: ExecuteE2State {
                num_segments,
                segments: segment_bytes,
                cost: 1000, // Mock cost
                execute_time_ms,
            },
        };
        if let Some(ref tx) = job.result_tx {
            let _ = tx.send(ProofResult::ExecuteE2(e2_result));
        } else {
            results.push(ProofResult::ExecuteE2(e2_result));
        }
        info!(
            "Emitted ExecuteE2Result with num_segments={}, execute_time_ms={}",
            num_segments, execute_time_ms
        );
    }

    // Use round-robin segment assignment to match the scheduler's expectation:
    // Worker N handles segments where segment_idx % num_provers == prover_id
    // This matches the real E2AppProver behavior.
    for segment_idx in 0..num_segments {
        // Only process segments assigned to this worker (round-robin)
        if segment_idx % job.num_provers != job.prover_id {
            continue;
        }

        // Simulate per-segment proving time
        let segment_start = Instant::now();
        std::thread::sleep(Duration::from_millis(20));
        let prove_time_ms = segment_start.elapsed().as_millis() as u64;

        let mock_proof = ProofWithPublicValue::<F> {
            proof: vec![0u8; 256],
            public_values: vec![F::default(); 4],
        };
        let proof = ProofResult::App(AppProof {
            context: job.context.clone(),
            state: AppProofState {
                proof: Some(proof::encode_proof(&mock_proof)?),
                segment_idx,
                prove_time_ms,
                fastfwd_time_ms: 0,
                stark_prove_time_ms: prove_time_ms,
                sub_metrics: std::collections::HashMap::new(),
                final_merkle_path_bytes: None,
                deferral_merkle_proofs_bytes: None,
            },
        });

        info!(
            "Generated mock app proof for segment {} ({}ms)",
            segment_idx, prove_time_ms
        );

        if let Some(ref tx) = job.result_tx {
            let _ = tx.send(proof);
        } else {
            results.push(proof);
        }
    }

    Ok(results)
}

// ============================================================================
// Real prover implementation (not mock-provers)
// ============================================================================

#[cfg(not(feature = "mock-provers"))]
mod real_impl {
    use super::super::real_prover_types::{RecursionEngine, SdkVmBuilder};
    use super::*;
    use crate::artifacts::ArtifactStore;
    use crossbeam::channel::bounded;
    use openvm_sdk_config::{SdkVmConfig, SegmentProver};
    use proof::ProofWithPublicValue;
    use protocol::{AppProof, AppProofState, ExecuteE2Result, ExecuteE2State};
    use sdk_v2::openvm_circuit::arch::{
        deferral::DeferralState,
        execution_mode::{ExecutionCtx, MeteredCtx, Segment},
        hasher::poseidon2::vm_poseidon2_hasher,
        SystemConfig, VmExecutor, VmInstance, VmState,
    };
    // `VmExecState` drives the interpreter metered loop; under rvr the metered
    // loop threads `(VmState, MeteredCtx)` by value instead, so it's unused.
    #[cfg(not(feature = "rvr"))]
    use sdk_v2::openvm_circuit::arch::VmExecState;
    use sdk_v2::openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProof;
    use sdk_v2::openvm_circuit::system::memory::{
        dimensions::MemoryDimensions, online::GuestMemory, AddressMap,
    };
    use sdk_v2::prover::vm::new_local_prover;
    use sdk_v2::StdIn;
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::time::Instant;
    use tracing::{error, info_span, Span};

    /// On a deferral job, extract the depth-independent
    /// `(DEFERRAL_AS, 0)` authentication path from the final memory
    /// image and encode it for the wire (`proof::encode_deferral_auth_path`).
    /// Called from the terminal-segment branch of every prove path
    /// (single-prover, parallel coordinator-consumer, parallel consumer).
    /// Reuses the same `&memory` borrow as `UserPublicValuesProof::compute`
    /// so we don't traverse memory twice.
    ///
    /// Mirrors openvm-sdk's pre-/post-`prove` tree build in
    /// `stark.rs:82-152`, except depth is left unset here: only the
    /// depth-independent path is shipped, and the tail worker finalizes
    /// with `depth` from `DeferralPvs` (see
    /// `crate::deferral_merkle::finalize_deferral_path`).
    fn terminal_deferral_path_bytes(
        memory: &AddressMap,
        memory_dimensions: &MemoryDimensions,
        hasher: &sdk_v2::openvm_circuit::arch::hasher::poseidon2::Poseidon2Hasher<proof::F>,
    ) -> Result<Vec<u8>> {
        let tree = crate::deferral_merkle::build_memory_tree(memory, memory_dimensions, hasher);
        let path = crate::deferral_merkle::extract_deferral_auth_path(memory_dimensions, &tree);
        proof::encode_deferral_auth_path(&path)
            .map_err(|e| eyre::eyre!("encode_deferral_auth_path failed: {}", e))
    }

    /// On a deferral deployment, a proof that made no deferred calls
    /// (`is_deferral_job == false`) still needs a depth-0
    /// `DeferralMerkleProofs` attached before it can verify — the deferral
    /// verifying key rejects a proof that carries none
    /// (`MissingDeferralMerkleProofs`). A real deferral proof gets its merkle
    /// proofs finalized on the tail worker; a no-deferral proof has no tail
    /// worker, so the terminal app worker (the only stage holding both the
    /// exe and the final memory) builds the COMPLETE depth-0 proofs here and
    /// ships the encoded bytes downstream (stark: attached at persist; evm:
    /// forwarded to root prove).
    ///
    /// Returns `Ok(None)` when this is not a deferral deployment — today's
    /// non-deferral path, byte-for-byte unchanged.
    fn terminal_depth0_deferral_proofs_bytes(
        program: &protocol::ProgramRef,
        system_config: &SystemConfig,
        final_memory: &AddressMap,
        hasher: &sdk_v2::openvm_circuit::arch::hasher::poseidon2::Poseidon2Hasher<proof::F>,
    ) -> Result<Option<Vec<u8>>> {
        let store = crate::artifacts::ArtifactStore::global()
            .ok_or_else(|| eyre::eyre!("Artifact store not initialized"))?;
        let edge_artifacts = store
            .get_edge_artifacts()
            .ok_or_else(|| eyre::eyre!("Edge artifacts not loaded"))?;
        if !edge_artifacts.is_deferral_deployment() {
            return Ok(None);
        }
        let exe = store
            .vmexe(program)
            .ok_or_else(|| eyre::eyre!("vmexe for {} not loaded on this worker", program))?;
        let proofs = crate::deferral_merkle::depth0_deferral_merkle_proofs(
            exe.as_ref(),
            system_config,
            final_memory,
            hasher,
        );
        let mut bytes = Vec::new();
        proofs
            .encode(&mut bytes)
            .map_err(|e| eyre::eyre!("encode depth-0 DeferralMerkleProofs failed: {}", e))?;
        Ok(Some(bytes))
    }

    /// Validate deferral request shape and load each `DeferralState` from its
    /// staged path. This is plumbing-only: edge knows `num_deferral_circuits`
    /// from the loaded artifact and rejects mismatched / unconfigured deferral
    /// jobs cleanly, so the downstream tail merge sees a well-defined channel.
    ///
    /// Non-deferral jobs (`deferral_state_paths.is_empty()`) return `Ok(vec![])`
    /// regardless of the artifact state, preserving today's path byte-for-byte.
    fn load_and_validate_deferral_states(
        deferral_state_paths: &[String],
    ) -> Result<Vec<DeferralState>> {
        if deferral_state_paths.is_empty() {
            return Ok(Vec::new());
        }

        // Edge derives `num_deferral_circuits` from the loaded deferral artifact
        // (compiled into every real build; toggled at runtime by
        // `enable_deferral`). Reject a mismatched or unconfigured deferral job
        // with a clear error rather than dropping bytes on the floor.
        {
            let artifact_store = ArtifactStore::global()
                .ok_or_else(|| eyre::eyre!("Artifact store not initialized"))?;
            let edge_artifacts = artifact_store
                .get_edge_artifacts()
                .ok_or_else(|| eyre::eyre!("Edge artifacts not loaded"))?;
            let deferral = edge_artifacts.deferral.as_ref().ok_or_else(|| {
                eyre::eyre!(
                    "Deferral job received ({} circuits) but this worker is not a deferral \
                     deployment (enable_deferral not set). Refusing to start.",
                    deferral_state_paths.len()
                )
            })?;
            let num_deferral_circuits = deferral
                .cached_pk
                .app_pk
                .app_vm_pk
                .vm_config
                .deferral
                .as_ref()
                .map(|d| d.circuits.len())
                .unwrap_or(0);
            if deferral_state_paths.len() != num_deferral_circuits {
                eyre::bail!(
                    "Deferral job has {} circuits in request but the loaded keyset expects {}; \
                     the caller's def_inputs.len() must match num_deferral_circuits.",
                    deferral_state_paths.len(),
                    num_deferral_circuits
                );
            }

            // Read + decode each `DeferralState`. Bytes are opaque to edge —
            // we just round-trip bincode (`Serialize`/`Deserialize` round-tripping
            // is the bytes' only contract).
            let mut states = Vec::with_capacity(deferral_state_paths.len());
            for (idx, path) in deferral_state_paths.iter().enumerate() {
                let bytes = std::fs::read(path).map_err(|e| {
                    eyre::eyre!(
                        "Failed to read deferral state file for circuit {} at {}: {}",
                        idx,
                        path,
                        e
                    )
                })?;
                let state: DeferralState = bincode::deserialize(&bytes).map_err(|e| {
                    eyre::eyre!(
                        "Failed to deserialize DeferralState for circuit {} at {}: {}",
                        idx,
                        path,
                        e
                    )
                })?;
                states.push(state);
            }
            Ok(states)
        }
    }

    /// Build the `StdIn` consumed by execution. Reads the caller's serialized
    /// `StdIn` (today's path — `buffer` only) and grafts `deferrals` onto it
    /// from the per-circuit staged files. Non-deferral jobs round-trip
    /// byte-identically.
    fn build_execution_stdin(input_bytes: &[u8], deferral_state_paths: &[String]) -> Result<StdIn> {
        let mut stdin: StdIn = bincode::deserialize(input_bytes)
            .map_err(|e| eyre::eyre!("Failed to deserialize input: {}", e))?;
        let deferrals = load_and_validate_deferral_states(deferral_state_paths)?;
        if !deferrals.is_empty() {
            // Graft the per-circuit `DeferralState`s onto the StdIn. On a
            // deferral deployment (B-model: single keyset selected by
            // `enable_deferral`) the app VM runs the deferral extension, so the
            // CALL/OUTPUT opcodes consume these during execution. On a
            // non-deferral deployment `deferrals` is empty and this is a no-op.
            stdin.deferrals = deferrals;
        }
        Ok(stdin)
    }

    /// Encode a slice of `Segment`s into wire-form opaque byte strings.
    fn encode_segments(segments: &[Segment]) -> Result<Vec<Vec<u8>>> {
        segments.iter().map(proof::encode_segment).collect()
    }

    /// Type alias for the VmExe type used in this module
    type VmExeType = sdk_v2::openvm_circuit::arch::instructions::exe::VmExe<
        openvm_stark_backend::Val<sdk_v2::SC>,
    >;

    /// Fixed-program prover: the GPU `VmInstance` plus a `SegmentProver`
    /// prepared once at construction. Under cuda+rvr `SegmentProver` resolves to
    /// `openvm_sdk_config::preflight_driver::SegmentProver`, which uploads the
    /// guest program to the device once and runs the native rvr preflight per
    /// segment (no per-segment upload, no interpreter). Held in
    /// `Option<ProverType>` on each app worker thread, swapped on program change.
    pub struct ProverType {
        segment_prover: SegmentProver,
        instance: VmInstance<RecursionEngine, SdkVmBuilder>,
    }

    impl ProverType {
        fn new(instance: VmInstance<RecursionEngine, SdkVmBuilder>) -> Result<Self> {
            let segment_prover = SegmentProver::new(&instance)?;
            Ok(Self {
                segment_prover,
                instance,
            })
        }

        /// Prove one segment from its (fast-forwarded) start state via the
        /// standalone `SegmentProver`. Returns the segment proof and, on
        /// successful termination, the final memory.
        fn prove_segment(
            &mut self,
            state: VmState<GuestMemory>,
            segment: &Segment,
        ) -> std::result::Result<
            (
                openvm_stark_backend::proof::Proof<sdk_v2::SC>,
                Option<GuestMemory>,
            ),
            sdk_v2::openvm_circuit::arch::VirtualMachineError,
        > {
            self.segment_prover
                .prove(&mut self.instance, state, segment)
        }
    }

    // Execution instance types for segment discovery (metered) and
    // fast-forwarding to a segment start (pure).
    //
    // Default backend: openvm's `InterpretedInstance` (v2.1.0 dropped its field
    // type parameter — it is now generic only over the execution `Ctx`).
    //
    // Pure (fast-forward) instance: ALWAYS the interpreter, both backends.
    // The fast-forward must land EXACTLY at a segment's `instret_start` before
    // proving. The interpreter is instruction-exact; rvr's pure
    // `execute_from_state_for` only stops at a basic-block boundary (it may
    // over/undershoot `num_insns`), which lands the segment at the wrong start
    // state and makes its trace fail the LogUp argument. So we keep the exact
    // interpreter for fast-forward (a cheap ~1-segment replay) and use the
    // native rvr backend only for the expensive metered segment discovery.
    type PureInstanceType =
        sdk_v2::openvm_circuit::arch::InterpretedInstance<'static, ExecutionCtx>;

    // Metered (segment discovery) instance: interpreter by default; the native
    // rvr segment-boundary instance under `rvr` (the successor to the removed
    // `aot` backend). The `MeteredDriver` below abstracts the two APIs so the
    // pipelined executor loop is written once.
    #[cfg(not(feature = "rvr"))]
    type MeteredInstanceType =
        sdk_v2::openvm_circuit::arch::InterpretedInstance<'static, MeteredCtx>;
    #[cfg(feature = "rvr")]
    type MeteredInstanceType =
        sdk_v2::openvm_circuit::arch::rvr::RvrMeteredSegmentInstance<'static>;

    /// Fast-forward `vm_state` by exactly `num_ins` instructions on the pure
    /// (interpreter) instance, returning the resulting VM state. Instruction-
    /// exact: it lands precisely at the segment's `instret_start`.
    fn fast_forward(
        pure: &PureInstanceType,
        vm_state: VmState<GuestMemory>,
        num_ins: u64,
    ) -> std::result::Result<VmState<GuestMemory>, sdk_v2::openvm_circuit::arch::ExecutionError>
    {
        Ok(pure.execute_from_state_for(vm_state, num_ins)?.into_inner())
    }

    /// Backend-abstracted driver for the pipelined metered segmentation loop.
    ///
    /// Both backends discover the same `Segment`s and expose the same
    /// per-boundary data (segments so far, `instret`, and the VM state at the
    /// boundary). They differ only in how one "run until the next segment
    /// boundary" step is driven: the interpreter threads a single `VmExecState`;
    /// the rvr backend threads `(VmState, MeteredCtx)` by value and returns a
    /// `SegmentationState`. Writing the loop against this driver keeps the
    /// snapshot / `+2`-lookahead / dispatch logic identical across both.
    struct MeteredDriver<'a> {
        instance: &'a MeteredInstanceType,
        #[cfg(not(feature = "rvr"))]
        exec_state: Option<VmExecState<GuestMemory, MeteredCtx>>,
        #[cfg(feature = "rvr")]
        vm_state: Option<VmState<GuestMemory>>,
        #[cfg(feature = "rvr")]
        ctx: Option<MeteredCtx>,
    }

    impl<'a> MeteredDriver<'a> {
        fn new(
            instance: &'a MeteredInstanceType,
            vm_state: VmState<GuestMemory>,
            metered_ctx: MeteredCtx,
        ) -> Self {
            #[cfg(not(feature = "rvr"))]
            {
                Self {
                    instance,
                    exec_state: Some(VmExecState::new(vm_state, metered_ctx)),
                }
            }
            #[cfg(feature = "rvr")]
            {
                Self {
                    instance,
                    vm_state: Some(vm_state),
                    ctx: Some(metered_ctx),
                }
            }
        }

        /// Advance to the next segment boundary. Returns `true` iff the program
        /// terminated (i.e. the just-closed segment is the final one). A
        /// non-zero guest exit is an error on both backends.
        fn step(&mut self) -> Result<bool> {
            #[cfg(not(feature = "rvr"))]
            {
                let es = self
                    .exec_state
                    .take()
                    .expect("exec_state present between steps");
                let mut es = self.instance.execute_metered_until_suspend(es)?;
                let mut exit_code = Ok(None);
                std::mem::swap(&mut exit_code, &mut es.exit_code);
                let exit_code = exit_code?;
                let terminated = match exit_code {
                    Some(code) if code != 0 => {
                        return Err(eyre::eyre!("VM exited with non-zero exit code: {}", code));
                    }
                    Some(_) => true,
                    None => false,
                };
                self.exec_state = Some(es);
                Ok(terminated)
            }
            #[cfg(feature = "rvr")]
            {
                let vm_state = self
                    .vm_state
                    .take()
                    .expect("vm_state present between steps");
                let ctx = self.ctx.take().expect("ctx present between steps");
                // A non-zero guest exit surfaces here as an `Err` (rvr
                // `GuestExit`), matching the interpreter's explicit non-zero
                // check above.
                let (outcome, vm_state) = self
                    .instance
                    .execute_metered_from_state_until_segment_boundary(vm_state, ctx)?;
                let (seg_state, terminated) = match outcome {
                    sdk_v2::openvm_circuit::arch::ExecutionOutcome::Terminated(s) => (s, true),
                    sdk_v2::openvm_circuit::arch::ExecutionOutcome::Suspended(s) => (s, false),
                };
                self.vm_state = Some(vm_state);
                self.ctx = Some(seg_state.ctx);
                Ok(terminated)
            }
        }

        /// Segments discovered so far (the last entry is the just-closed one).
        fn segments(&self) -> &[Segment] {
            #[cfg(not(feature = "rvr"))]
            {
                &self
                    .exec_state
                    .as_ref()
                    .expect("exec_state present")
                    .ctx
                    .segmentation_ctx
                    .segments
            }
            #[cfg(feature = "rvr")]
            {
                &self
                    .ctx
                    .as_ref()
                    .expect("ctx present")
                    .segmentation_ctx
                    .segments
            }
        }

        /// Instruction count retired at the current boundary.
        fn instret(&self) -> u64 {
            #[cfg(not(feature = "rvr"))]
            {
                self.exec_state
                    .as_ref()
                    .expect("exec_state present")
                    .ctx
                    .segmentation_ctx
                    .instret
            }
            #[cfg(feature = "rvr")]
            {
                self.ctx
                    .as_ref()
                    .expect("ctx present")
                    .segmentation_ctx
                    .instret
            }
        }

        /// Clone the VM state at the current boundary (a snapshot start-state).
        fn clone_boundary_state(&self) -> VmState<GuestMemory> {
            #[cfg(not(feature = "rvr"))]
            {
                self.exec_state
                    .as_ref()
                    .expect("exec_state present")
                    .vm_state
                    .clone()
            }
            #[cfg(feature = "rvr")]
            {
                self.vm_state.as_ref().expect("vm_state present").clone()
            }
        }

        /// Consume the driver, returning all discovered segments.
        fn into_segments(self) -> Vec<Segment> {
            #[cfg(not(feature = "rvr"))]
            {
                self.exec_state
                    .expect("exec_state present")
                    .ctx
                    .segmentation_ctx
                    .segments
            }
            #[cfg(feature = "rvr")]
            {
                self.ctx.expect("ctx present").segmentation_ctx.segments
            }
        }
    }

    fn build_metered_ctx(
        app_prover: &ProverType,
        exe: &VmExeType,
        segment_memory: Option<usize>,
    ) -> MeteredCtx {
        let mut metered_ctx = app_prover
            .instance
            .vm
            .build_metered_ctx(exe)
            .with_suspend_on_segment(true);

        // Segmentation trace height is no longer configurable per job: openvm derives it
        // from the prover's stacked height (openvm #2888). Only segment memory is tunable.
        if let Some(m) = segment_memory {
            info!("Overriding max_memory to {}", m);
            metered_ctx = metered_ctx.with_max_memory(m);
        }

        metered_ctx
    }

    /// Cached execution instances reused across proofs.
    pub(crate) struct ExecutionInstances {
        pub(crate) pure: PureInstanceType,
        pub(crate) metered: MeteredInstanceType,
    }

    /// Snapshot of VM state at a point during execution.
    /// Used to avoid re-executing from instruction 0 for each segment.
    #[derive(Clone)]
    struct VmSnapshot {
        vm_state: VmState<GuestMemory>,
        instret: u64,
    }

    /// Data sent from executor thread to prover thread(s) for each assigned segment.
    pub(crate) struct ProveData {
        snapshot: VmSnapshot,
        segment: Segment,
        segment_idx: usize,
        is_final: bool,
        /// When the executor handed this segment to the bounded channel. The
        /// gap to the moment a prover picks it up separates a prover-bound run
        /// from an executor-bound one.
        sent_at: Instant,
    }

    /// Result returned by the executor thread after execution completes.
    struct ExecutorResult {
        execute_time_ms: u64,
        segments: Vec<Segment>,
        num_segments: usize,
    }

    /// Coordinator closure for parallel proving — runs the executor, owns consumer-0,
    /// and returns its result. The pool merges this with every consumer result.
    pub type ParallelCoordinatorFn =
        Box<dyn FnOnce(&AppExecutionInstances, &mut ProverType) -> ProverResult + Send>;

    /// Segment-consumer closure for parallel proving — pulls segments from the shared
    /// channel and proves them. Streaming jobs send proofs out-of-band and return an
    /// empty success; non-streaming jobs return their proofs to the pool for merging.
    pub type SegmentConsumerFn =
        Box<dyn FnOnce(&AppExecutionInstances, &mut ProverType) -> ProverResult + Send>;

    /// CPU-side per-program state that survives the worker's lifetime.
    ///
    /// Holds the AOT-compiled pure + metered interpreters (the expensive
    /// ~115s gcc artifacts), the vmexe, and a cached `vm_config`. Built
    /// once per program at worker boot via `AppExecutionInstances::new`,
    /// then shared across all app worker threads via `Arc<>`.
    ///
    /// **No GPU state lives here** — the heavy GPU `ProverType` is held
    /// separately by each worker thread inside an `Option<ProverType>`,
    /// swapped on program change.
    pub struct AppExecutionInstances {
        /// Cached VM config for computing user public values.
        pub vm_config: SdkVmConfig,
        /// Cached executable for this program version.
        pub exe: Arc<VmExeType>,
        /// The program this instance proves against.
        pub program: protocol::ProgramRef,
        /// Cached AOT instances (pure + metered) to avoid per-proof AOT generation.
        ///
        /// **Drop order**: declared before `_executor_keepalive` so the AOT
        /// `.so`s (whose machine code holds raw pointers into the executor's
        /// `Arc<ExecutorInventory>`) are unloaded before the executor's Arc
        /// handle is dropped. Rust drops fields top-to-bottom.
        pub(crate) execution_instances: Arc<ExecutionInstances>,
        /// CPU-only keepalive for the AOT `.so`s' baked `&FieldExpr` pointers.
        ///
        /// `VmExecutor` holds `Arc<ExecutorInventory>` internally (the
        /// `FieldExpr` configs live inside it). The AOT codegen step bakes
        /// raw pointers to those `FieldExpr` instances into the compiled
        /// `.so`. Dropping the GPU `ProverType` would normally release the
        /// last Arc handle and free the inventory → use-after-free. By
        /// cloning the executor here (a cheap `Arc::clone` + a `Copy`
        /// config copy, **no GPU state**) we keep the inventory alive for
        /// the lifetime of this struct.
        _executor_keepalive: &'static VmExecutor<proof::F, SdkVmConfig>,
    }

    impl AppExecutionInstances {
        /// Build the CPU-side AOT interpreters for `program`.
        ///
        /// Internally creates a *temporary* GPU prover (~1s, ~1.66 GB GPU)
        /// to drive the `interpreter()` / `metered_interpreter()` AOT
        /// compile (~115s wall). Before dropping the prover, we clone its
        /// `VmExecutor` (a cheap `Arc` refcount bump on the executor
        /// inventory — no GPU state copied) and store it as
        /// `_executor_keepalive` so the `&FieldExpr` pointers baked into
        /// the AOT `.so`s stay valid forever. The GPU prover is then
        /// dropped, freeing its engine / proving key / chip complex.
        pub fn new(
            program: &protocol::ProgramRef,
            app_pk: &Arc<sdk_v2::keygen::AppProvingKey<SdkVmConfig>>,
            exe: Arc<VmExeType>,
        ) -> Result<Self> {
            let total_start = Instant::now();
            let vm_config = app_pk.app_vm_pk.vm_config.clone();

            info!("Creating AppExecutionInstances for {program}...");

            // Temporary GPU prover, only needed to build the AOT interpreters.
            let t_gpu_start = Instant::now();
            let temp_prover = new_local_prover::<RecursionEngine, _>(
                SdkVmBuilder {},
                &app_pk.app_vm_pk,
                exe.clone(),
            )?;
            let t_gpu_ms = t_gpu_start.elapsed().as_millis();
            info!(
                "AppExecutionInstances[{program}] (a) new_local_prover (GPU, temporary) took {}ms",
                t_gpu_ms
            );

            // Clone the executor BEFORE dropping the prover. VmExecutor owns
            // `inventory: Arc<ExecutorInventory<...>>`, so this is just an
            // Arc::clone — no GPU state, no chip complex, no proving key.
            // rc.2 exposes interpreter lifetimes, so the cloned executor is
            // kept for the process lifetime and the cached instances borrow it.
            let executor_keepalive = Box::leak(Box::new(temp_prover.vm.executor().clone()));
            let executor_idx_to_air_idx = temp_prover.vm.executor_idx_to_air_idx();

            // (b) Pure execution instance for fast-forwarding (by an exact
            //     instruction count) to a segment start. ALWAYS the interpreter
            //     — even under `rvr` — because it must be instruction-exact
            //     (see `PureInstanceType`). Under rvr, `.instance()` would yield
            //     the native (basic-block-boundary, imprecise) pure backend, so
            //     we call `.interpreter_instance()` for the exact interpreter.
            let t_pure_start = Instant::now();
            #[cfg(not(feature = "rvr"))]
            let pure = executor_keepalive.instance(exe.as_ref())?;
            #[cfg(feature = "rvr")]
            let pure = executor_keepalive.interpreter_instance(exe.as_ref())?;
            let t_pure_ms = t_pure_start.elapsed().as_millis();
            info!(
                "AppExecutionInstances[{program}] (b) pure execution instance took {}ms",
                t_pure_ms
            );

            // (c) Metered execution instance for incremental segment discovery
            //     (suspends at each segment boundary). Interpreter by default;
            //     the native rvr segment-boundary instance under `rvr`.
            let t_metered_start = Instant::now();
            #[cfg(not(feature = "rvr"))]
            let metered =
                executor_keepalive.metered_instance(exe.as_ref(), &executor_idx_to_air_idx)?;
            #[cfg(feature = "rvr")]
            let metered = executor_keepalive.metered_segment_instance(
                exe.as_ref(),
                &executor_idx_to_air_idx,
                temp_prover.vm.num_airs(),
                None,
            )?;
            let t_metered_ms = t_metered_start.elapsed().as_millis();
            info!(
                "AppExecutionInstances[{program}] (c) metered interpreter (AOT) took {}ms",
                t_metered_ms
            );

            // Drop the temporary prover — frees ~1.66 GB of GPU memory
            // (engine + DeviceMultiStarkProvingKey + VmChipComplex). The
            // executor's Arc<ExecutorInventory> still has refcount >= 1 via
            // executor_keepalive, so the baked pointers remain valid.
            drop(temp_prover);

            let execution_instances = Arc::new(ExecutionInstances { pure, metered });

            info!(
                "AppExecutionInstances[{program}] ready (CPU-only): total={}ms (gpu_temp={}ms pure={}ms metered={}ms)",
                total_start.elapsed().as_millis(),
                t_gpu_ms,
                t_pure_ms,
                t_metered_ms,
            );

            Ok(Self {
                vm_config,
                exe,
                program: program.clone(),
                execution_instances,
                _executor_keepalive: executor_keepalive,
            })
        }
    }

    /// Build a GPU `ProverType` for `program`. Cheap (~1s, ~1.66 GB GPU
    /// transfer). Held in `Option<ProverType>` on each app worker thread
    /// and swapped out on program change.
    pub fn build_gpu_prover(
        program: &protocol::ProgramRef,
        app_pk: &Arc<sdk_v2::keygen::AppProvingKey<SdkVmConfig>>,
        exe: Arc<VmExeType>,
    ) -> Result<ProverType> {
        let start = Instant::now();
        let instance =
            new_local_prover::<RecursionEngine, _>(SdkVmBuilder {}, &app_pk.app_vm_pk, exe)?;
        let prover = ProverType::new(instance)?;
        info!(
            "build_gpu_prover[{program}] took {}ms",
            start.elapsed().as_millis()
        );
        Ok(prover)
    }

    /// Execute sharded app proving using the provided per-program
    /// execution instances and a GPU prover loaded for the same program.
    #[instrument(skip_all, fields(
        proof_id = %job.context.proof_uuid,
        prover_id = job.prover_id,
        num_provers = job.num_provers,
        input_path = %job.input_path
    ))]
    pub fn prove_sharded_app_with_prover(
        job: ShardedAppProverJob,
        instances: &AppExecutionInstances,
        app_prover: &mut ProverType,
    ) -> ProverResult {
        info!(
            "Starting sharded app prove (with prover): prover_id={}, num_provers={}",
            job.prover_id, job.num_provers
        );

        match prove_sharded_app_impl_with_prover(job, instances, app_prover) {
            Ok(results) => ProverResult::Success(results),
            Err(e) => ProverResult::Error(format!("Sharded app prove failed: {}", e)),
        }
    }

    fn prove_sharded_app_impl_with_prover(
        job: ShardedAppProverJob,
        instances: &AppExecutionInstances,
        app_prover: &mut ProverType,
    ) -> Result<Vec<ProofResult>> {
        // Validate job parameters
        if job.num_provers == 0 {
            return Err(eyre::eyre!("num_provers cannot be 0"));
        }
        if job.prover_id >= job.num_provers {
            return Err(eyre::eyre!(
                "prover_id {} must be less than num_provers {}",
                job.prover_id,
                job.num_provers
            ));
        }

        // Read input from disk
        info!("Reading input from {}", job.input_path);
        let input_bytes = std::fs::read(&job.input_path)
            .map_err(|e| eyre::eyre!("Failed to read input file {}: {}", job.input_path, e))?;
        let stdin = build_execution_stdin(&input_bytes, &job.deferral_state_paths)?;

        info!(
            "Sharded app prover: prover_id={}, num_provers={}, input_size={}, deferral_circuits={}",
            job.prover_id,
            job.num_provers,
            input_bytes.len(),
            job.deferral_state_paths.len()
        );

        let exe = instances.exe.as_ref();
        let vm_config = &instances.vm_config;
        let execution_instances = instances.execution_instances.clone();
        let metered_ctx = build_metered_ctx(app_prover, exe, job.segment_memory);

        // Create initial VM state
        let vm_state = execution_instances.metered.create_initial_vm_state(stdin);

        // Bounded channel: executor -> prover. Backpressure when channel is full
        // prevents unbounded snapshot accumulation (matching E2AppProver pattern).
        let (prove_tx, prove_rx) = bounded::<ProveData>(2);

        let num_provers = job.num_provers;
        let prover_id = job.prover_id;
        let executor_result_tx = job.result_tx.clone();
        let executor_context = if prover_id == 0 {
            Some(job.context.clone())
        } else {
            None
        };
        let execution_instances_for_executor = execution_instances.clone();
        // A spawned thread does not inherit the ambient span, so carry it over
        // explicitly and every line the executor emits keeps its proof_id.
        let executor_span = Span::current();

        // --- Executor thread ---
        // Runs metered execution, discovers segments, sends snapshots + metadata
        // to the prover via bounded channel. Continues executing while prover works.
        let executor_handle = std::thread::spawn(move || -> Result<ExecutorResult> {
            let _executor_span = executor_span.enter();
            let metered_interpreter = &execution_instances_for_executor.metered;
            let mut snapshots: VecDeque<VmSnapshot> = VecDeque::with_capacity(2);
            if seeds_initial_snapshot(prover_id) {
                snapshots.push_back(VmSnapshot {
                    vm_state: vm_state.clone(),
                    instret: 0,
                });
            }

            let mut driver = MeteredDriver::new(metered_interpreter, vm_state, metered_ctx);
            let exec_start = std::time::Instant::now();

            info!("Executor thread: starting metered execution");

            loop {
                let should_break = driver.step()?;

                let curr_idx = driver.segments().len() - 1;

                // If this segment is assigned to us, send it to the prover
                if curr_idx % num_provers == prover_id {
                    let snapshot = if num_provers == 1 && curr_idx == 0 {
                        snapshots[0].clone()
                    } else {
                        snapshots
                            .pop_front()
                            .expect("Snapshot missing for assigned segment")
                    };

                    let segment = driver.segments()[curr_idx].clone();

                    info!("Executor: sending segment {} for proving", curr_idx);
                    prove_tx
                        .send(ProveData {
                            snapshot,
                            segment,
                            segment_idx: curr_idx,
                            is_final: should_break,
                            sent_at: Instant::now(),
                        })
                        .map_err(|_| eyre::eyre!("Prover disconnected"))?;
                }

                if should_break {
                    break;
                }

                // Save snapshot for future assigned segment (+2 lookahead) —
                // see the snapshot-schedule notes on `seeds_initial_snapshot`
                // / `saves_snapshot_at` at the top of this file.
                if saves_snapshot_at(curr_idx, num_provers, prover_id) {
                    let instret = driver.instret();
                    snapshots.push_back(VmSnapshot {
                        vm_state: driver.clone_boundary_state(),
                        instret,
                    });
                }
            }

            let execute_time_ms = exec_start.elapsed().as_millis() as u64;
            let segments = driver.into_segments();
            let num_segments = segments.len();

            info!(
                "Executor thread: finished, {} segments discovered in {}ms",
                num_segments, execute_time_ms
            );

            // Send ExecuteE2 result immediately so manager knows num_segments ASAP
            if let (Some(ref tx), Some(ref ctx)) = (&executor_result_tx, &executor_context) {
                let e2_result = ProofResult::ExecuteE2(ExecuteE2Result {
                    context: ctx.clone(),
                    state: ExecuteE2State {
                        num_segments,
                        segments: encode_segments(&segments)?,
                        cost: 0,
                        execute_time_ms,
                    },
                });
                let _ = tx.send(e2_result);
                info!(
                    "Executor: streamed ExecuteE2Result with num_segments={}",
                    num_segments
                );
            }

            Ok(ExecutorResult {
                execute_time_ms,
                segments,
                num_segments,
            })
        });

        // --- Prover (main thread) ---
        // Receives ProveData from executor, fast-forwards from snapshot, generates STARK proof.
        // Runs on the main thread which owns the AppProverInstance (GPU resources).
        let pure_interpreter = &execution_instances.pure;
        let mut app_results: Vec<ProofResult> = Vec::new();
        let mut prover_err: Option<eyre::Report> = None;
        for prove_data in prove_rx.iter() {
            info!("Prover: proving segment {}", prove_data.segment_idx);
            // How long the segment sat in the bounded channel. Near zero means
            // the provers keep up and the executor sets the pace.
            let queue_wait_ms = prove_data.sent_at.elapsed().as_millis() as u64;
            let segment_start = std::time::Instant::now();

            // Fast-forward from snapshot to segment start
            let fastfwd_start = std::time::Instant::now();
            let mut vm_state = prove_data.snapshot.vm_state;
            let num_ins = prove_data.segment.instret_start - prove_data.snapshot.instret;
            if num_ins > 0 {
                match fast_forward(pure_interpreter, vm_state, num_ins) {
                    Ok(state) => vm_state = state,
                    Err(e) => {
                        prover_err = Some(e.into());
                        break;
                    }
                }
            }
            let fastfwd_time_ms = fastfwd_start.elapsed().as_millis() as u64;

            // Generate STARK proof for this segment
            let _ = telemetry::span_timing::drain_span_timings(); // clear stale timings
            let stark_start = std::time::Instant::now();
            // openvm v2.1 ships only touched-marked pages to the device (unmarked
            // → zero). Our snapshot + fast-forward reconstructs correct memory
            // *values* but not the touched-page metadata a sequential preflight
            // would carry, so rebuild it from the image before proving.
            vm_state.memory.memory.recompute_touched_pages();
            let prove_result = app_prover.prove_segment(vm_state, &prove_data.segment);
            let (proof, final_memory) = match prove_result {
                Ok(r) => r,
                Err(e) => {
                    prover_err = Some(e.into());
                    break;
                }
            };
            let stark_prove_time_ms = stark_start.elapsed().as_millis() as u64;
            let sub_metrics = telemetry::span_timing::drain_span_timings();

            let prove_time_ms = segment_start.elapsed().as_millis() as u64;

            // Compute user public values for the final segment, and
            // (on a deferral job) extract the depth-independent
            // `(DEFERRAL_AS, 0)` authentication path from the same
            // final memory image — sharing the `memory` borrow so we
            // don't build the tree machinery twice.
            let is_deferral_job = !job.deferral_state_paths.is_empty();
            let mut user_public_values = None;
            let mut final_merkle_path_bytes = None;
            let mut deferral_merkle_proofs_bytes: Option<Vec<u8>> = None;
            if prove_data.is_final {
                let memory = match final_memory.ok_or_else(|| {
                    eyre::eyre!(
                        "Terminal segment {} did not return final_memory",
                        prove_data.segment_idx
                    )
                }) {
                    Ok(m) => m,
                    Err(e) => {
                        prover_err = Some(e);
                        break;
                    }
                };
                let top_tree =
                    match app_prover.instance.vm.memory_top_tree().ok_or_else(|| {
                        eyre::eyre!("Memory top tree should exist for terminal segment")
                    }) {
                        Ok(t) => t,
                        Err(e) => {
                            prover_err = Some(e);
                            break;
                        }
                    };
                let memory_dimensions = vm_config.system.config.memory_config.memory_dimensions();
                let hasher = vm_poseidon2_hasher();
                user_public_values = Some(UserPublicValuesProof::compute(
                    &vm_config.system.config,
                    &hasher,
                    &memory.memory,
                    top_tree,
                ));
                if is_deferral_job {
                    match terminal_deferral_path_bytes(&memory.memory, &memory_dimensions, &hasher)
                    {
                        Ok(bytes) => final_merkle_path_bytes = Some(bytes),
                        Err(e) => {
                            prover_err = Some(e);
                            break;
                        }
                    }
                } else {
                    // Deferral deployment + no deferral input for this proof →
                    // build the depth-0 DeferralMerkleProofs the deferral VK
                    // still requires. `None` (no-op) on a non-deferral deployment.
                    match terminal_depth0_deferral_proofs_bytes(
                        &job.context.program,
                        &vm_config.system.config,
                        &memory.memory,
                        &hasher,
                    ) {
                        Ok(bytes) => deferral_merkle_proofs_bytes = bytes,
                        Err(e) => {
                            prover_err = Some(e);
                            break;
                        }
                    }
                }
            }

            let spans = telemetry::span_timing::format_span_timings(&sub_metrics);
            let app_proof = ProofResult::App(AppProof {
                context: job.context.clone(),
                state: AppProofState {
                    proof: Some(proof::encode_proof(&ProofWithPublicValue {
                        proof,
                        user_public_values,
                    })?),
                    segment_idx: prove_data.segment_idx,
                    prove_time_ms,
                    fastfwd_time_ms,
                    stark_prove_time_ms,
                    sub_metrics,
                    final_merkle_path_bytes,
                    deferral_merkle_proofs_bytes,
                },
            });

            info!(
                "Prover: generated app proof for segment {}: queue_wait={}ms, fastfwd={}ms, stark={}ms, prove={}ms, spans={}",
                prove_data.segment_idx,
                queue_wait_ms,
                fastfwd_time_ms,
                stark_prove_time_ms,
                prove_time_ms,
                spans
            );

            if let Some(ref tx) = job.result_tx {
                let _ = tx.send(app_proof);
            } else {
                app_results.push(app_proof);
            }
        }

        // Drop receiver to unblock executor if it's still sending
        drop(prove_rx);

        // Wait for executor thread to finish
        let executor_join = executor_handle
            .join()
            .map_err(|e| eyre::eyre!("Executor thread panicked: {:?}", e))?;

        // Prioritize prover error over executor error
        if let Some(e) = prover_err {
            return Err(e);
        }

        let executor_result = executor_join?;

        if executor_result.num_segments == 0 {
            return Err(eyre::eyre!("No segments produced from execution"));
        }

        info!(
            "Completed {} segments, proved {} app proofs with pipelining",
            executor_result.num_segments,
            app_results.len()
        );

        // E2 result: already streamed from executor thread when result_tx is set.
        // Only add to batch in non-streaming mode.
        if job.prover_id == 0 && job.result_tx.is_none() {
            app_results.insert(
                0,
                ProofResult::ExecuteE2(ExecuteE2Result {
                    context: job.context.clone(),
                    state: ExecuteE2State {
                        num_segments: executor_result.num_segments,
                        segments: encode_segments(&executor_result.segments)?,
                        cost: 0,
                        execute_time_ms: executor_result.execute_time_ms,
                    },
                }),
            );
        }

        Ok(app_results)
    }

    fn consumer_failure(error_message: String) -> ProverResult {
        error!("{}", error_message);
        ProverResult::Error(error_message)
    }

    /// Consume segments from a shared channel and prove them.
    ///
    /// This is the worker loop for parallel proving consumers.
    /// Each consumer borrows the per-program execution instances (shared
    /// `Arc<AppExecutionInstances>`) and its own GPU `ProverType` slot.
    ///
    /// `is_deferral_job` mirrors `!job.deferral_state_paths.is_empty()`
    /// from the coordinator's `ShardedAppProverJob`: when the terminal
    /// segment lands on this consumer, the deferral merkle auth path is
    /// extracted and attached to its `AppProof` so the manager can
    /// forward it in `DeferralTailDispatch`.
    pub fn consume_segments_loop(
        instances: &AppExecutionInstances,
        app_prover: &mut ProverType,
        prove_rx: crossbeam::channel::Receiver<ProveData>,
        streaming_tx: Option<crossbeam::channel::Sender<ProofResult>>,
        context: protocol::ProofContext,
        is_deferral_job: bool,
    ) -> ProverResult {
        let vm_config = &instances.vm_config;
        let pure_interpreter = &instances.execution_instances.pure;
        let mut results = Vec::new();

        for prove_data in prove_rx.iter() {
            info!("Consumer: proving segment {}", prove_data.segment_idx);
            // How long the segment sat in the bounded channel. Near zero means
            // the provers keep up and the executor sets the pace.
            let queue_wait_ms = prove_data.sent_at.elapsed().as_millis() as u64;
            let segment_start = std::time::Instant::now();

            // Fast-forward from snapshot to segment start
            let fastfwd_start = std::time::Instant::now();
            let mut vm_state = prove_data.snapshot.vm_state;
            let num_ins = prove_data.segment.instret_start - prove_data.snapshot.instret;
            if num_ins > 0 {
                match fast_forward(pure_interpreter, vm_state, num_ins) {
                    Ok(state) => vm_state = state,
                    Err(e) => {
                        return consumer_failure(format!(
                            "Consumer fast-forward failed for segment {}: {}",
                            prove_data.segment_idx, e
                        ));
                    }
                }
            }
            let fastfwd_time_ms = fastfwd_start.elapsed().as_millis() as u64;

            // Generate STARK proof for this segment
            let _ = telemetry::span_timing::drain_span_timings();
            let stark_start = std::time::Instant::now();
            // openvm v2.1 ships only touched-marked pages to the device (unmarked
            // → zero). Our snapshot + fast-forward reconstructs correct memory
            // *values* but not the touched-page metadata a sequential preflight
            // would carry, so rebuild it from the image before proving.
            vm_state.memory.memory.recompute_touched_pages();
            let prove_result = app_prover.prove_segment(vm_state, &prove_data.segment);
            let (proof, final_memory) = match prove_result {
                Ok(r) => r,
                Err(e) => {
                    return consumer_failure(format!(
                        "Consumer STARK prove failed for segment {}: {}",
                        prove_data.segment_idx, e
                    ));
                }
            };
            let stark_prove_time_ms = stark_start.elapsed().as_millis() as u64;
            let sub_metrics = telemetry::span_timing::drain_span_timings();
            let prove_time_ms = segment_start.elapsed().as_millis() as u64;

            // Compute user public values for the final segment, plus
            // the deferral merkle auth path when this is a deferral job
            // (shares the `memory` borrow with `UserPublicValuesProof::compute`).
            let mut user_public_values = None;
            let mut final_merkle_path_bytes: Option<Vec<u8>> = None;
            let mut deferral_merkle_proofs_bytes: Option<Vec<u8>> = None;
            if prove_data.is_final {
                let memory = match final_memory {
                    Some(m) => m,
                    None => {
                        return consumer_failure(format!(
                            "Terminal segment {} did not return final_memory",
                            prove_data.segment_idx
                        ));
                    }
                };
                let top_tree = match app_prover.instance.vm.memory_top_tree() {
                    Some(t) => t,
                    None => {
                        return consumer_failure(format!(
                            "Memory top tree missing for terminal segment {}",
                            prove_data.segment_idx
                        ));
                    }
                };
                let memory_dimensions = vm_config.system.config.memory_config.memory_dimensions();
                let hasher = vm_poseidon2_hasher();
                user_public_values = Some(UserPublicValuesProof::compute(
                    &vm_config.system.config,
                    &hasher,
                    &memory.memory,
                    top_tree,
                ));
                if is_deferral_job {
                    match terminal_deferral_path_bytes(&memory.memory, &memory_dimensions, &hasher)
                    {
                        Ok(bytes) => final_merkle_path_bytes = Some(bytes),
                        Err(e) => {
                            return consumer_failure(format!(
                                "Consumer deferral path extraction failed for segment {}: {}",
                                prove_data.segment_idx, e
                            ));
                        }
                    }
                } else {
                    // Deferral deployment + no deferral input for this proof →
                    // build the depth-0 DeferralMerkleProofs the deferral VK
                    // still requires. `None` (no-op) on a non-deferral deployment.
                    match terminal_depth0_deferral_proofs_bytes(
                        &context.program,
                        &vm_config.system.config,
                        &memory.memory,
                        &hasher,
                    ) {
                        Ok(bytes) => deferral_merkle_proofs_bytes = bytes,
                        Err(e) => {
                            return consumer_failure(format!(
                                "Consumer depth-0 deferral merkle build failed for segment {}: {}",
                                prove_data.segment_idx, e
                            ));
                        }
                    }
                }
            }

            info!(
                "Consumer: proved segment {}: queue_wait={}ms, fastfwd={}ms, stark={}ms, prove={}ms, spans={}",
                prove_data.segment_idx,
                queue_wait_ms,
                fastfwd_time_ms,
                stark_prove_time_ms,
                prove_time_ms,
                telemetry::span_timing::format_span_timings(&sub_metrics)
            );

            let encoded_proof = match proof::encode_proof(&ProofWithPublicValue {
                proof,
                user_public_values,
            }) {
                Ok(bytes) => bytes,
                Err(e) => {
                    return consumer_failure(format!(
                        "Consumer failed to encode proof for segment {}: {}",
                        prove_data.segment_idx, e
                    ));
                }
            };
            let app_proof = ProofResult::App(AppProof {
                context: context.clone(),
                state: AppProofState {
                    proof: Some(encoded_proof),
                    segment_idx: prove_data.segment_idx,
                    prove_time_ms,
                    fastfwd_time_ms,
                    stark_prove_time_ms,
                    sub_metrics,
                    final_merkle_path_bytes,
                    deferral_merkle_proofs_bytes,
                },
            });
            if let Some(ref tx) = streaming_tx {
                if tx.send(app_proof).is_err() {
                    return consumer_failure(format!(
                        "Consumer failed to stream proof for segment {}: result channel closed",
                        prove_data.segment_idx
                    ));
                }
            } else {
                results.push(app_proof);
            }
        }

        ProverResult::Success(results)
    }

    /// Create coordinator and consumer closures for parallel proving.
    ///
    /// Returns:
    /// - A coordinator closure that runs the executor + acts as consumer-0 + collects all results
    /// - N-1 consumer closures that consume segments from the shared channel
    ///
    /// The coordinator and consumer closures each return a `ProverResult`; the
    /// pool waits for and merges all of them before completing the parallel job.
    pub fn create_parallel_prove_jobs(
        job: ShardedAppProverJob,
    ) -> Result<(ParallelCoordinatorFn, Vec<SegmentConsumerFn>)> {
        let max_app_provers = job.max_app_provers;
        let streaming_tx = job.result_tx.clone();

        // Create shared MPMC channel for segment distribution
        let (prove_tx, prove_rx) = crossbeam::channel::bounded::<ProveData>(max_app_provers * 2);

        // Create N-1 consumer closures
        let is_deferral_job = !job.deferral_state_paths.is_empty();
        let mut consumers: Vec<SegmentConsumerFn> = Vec::new();
        for consumer_idx in 1..max_app_provers {
            let consumer_prove_rx = prove_rx.clone();
            let consumer_streaming_tx = streaming_tx.clone();
            let context = job.context.clone();
            // Consumers run on long-lived pool threads that carry no ambient
            // span, so each gets its own and every segment line it emits is
            // attributable to a proof and a prover.
            let consumer_span = info_span!(
                "app_consumer",
                proof_id = %job.context.proof_uuid,
                prover_id = job.prover_id,
                consumer_idx
            );

            consumers.push(Box::new(
                move |instances: &AppExecutionInstances, prover: &mut ProverType| {
                    let _consumer_span = consumer_span.enter();
                    consume_segments_loop(
                        instances,
                        prover,
                        consumer_prove_rx,
                        consumer_streaming_tx,
                        context,
                        is_deferral_job,
                    )
                },
            ));
        }

        // Create coordinator closure
        let coordinator: ParallelCoordinatorFn = Box::new(
            move |instances: &AppExecutionInstances, app_prover: &mut ProverType| {
                match coordinate_parallel_prove(job, instances, app_prover, prove_tx, prove_rx) {
                    Ok(results) => ProverResult::Success(results),
                    Err(e) => ProverResult::Error(format!("Parallel app prove failed: {}", e)),
                }
            },
        );

        Ok((coordinator, consumers))
    }

    /// Coordinator logic for parallel proving.
    ///
    /// The coordinator:
    /// 1. Reads input and creates the executor thread (feeds segments via prove_tx)
    /// 2. Acts as consumer-0 (proves segments from prove_rx)
    /// 3. Streams results via job.result_tx as each segment completes
    #[instrument(skip_all, fields(
        proof_id = %job.context.proof_uuid,
        prover_id = job.prover_id,
        num_provers = job.num_provers
    ))]
    fn coordinate_parallel_prove(
        job: ShardedAppProverJob,
        instances: &AppExecutionInstances,
        app_prover: &mut ProverType,
        prove_tx: crossbeam::channel::Sender<ProveData>,
        prove_rx: crossbeam::channel::Receiver<ProveData>,
    ) -> Result<Vec<ProofResult>> {
        // Validate job parameters
        if job.num_provers == 0 {
            return Err(eyre::eyre!("num_provers cannot be 0"));
        }
        if job.prover_id >= job.num_provers {
            return Err(eyre::eyre!(
                "prover_id {} must be less than num_provers {}",
                job.prover_id,
                job.num_provers
            ));
        }

        info!(
            "Parallel coordinator: prover_id={}, num_provers={}, max_app_provers={}",
            job.prover_id, job.num_provers, job.max_app_provers
        );
        let setup_start = std::time::Instant::now();

        // Read input from disk
        let input_read_start = std::time::Instant::now();
        let input_bytes = std::fs::read(&job.input_path)
            .map_err(|e| eyre::eyre!("Failed to read input file {}: {}", job.input_path, e))?;
        let input_read_ms = input_read_start.elapsed().as_millis();
        let stdin_start = std::time::Instant::now();
        let stdin = build_execution_stdin(&input_bytes, &job.deferral_state_paths)?;
        let stdin_ms = stdin_start.elapsed().as_millis();
        let is_deferral_job = !job.deferral_state_paths.is_empty();

        let exe = instances.exe.as_ref();
        let vm_config = &instances.vm_config;
        let execution_instances = instances.execution_instances.clone();

        let metered_ctx_start = std::time::Instant::now();
        let metered_ctx = build_metered_ctx(app_prover, exe, job.segment_memory);
        let metered_ctx_ms = metered_ctx_start.elapsed().as_millis();

        let vm_state_start = std::time::Instant::now();
        let vm_state = execution_instances.metered.create_initial_vm_state(stdin);
        let vm_state_ms = vm_state_start.elapsed().as_millis();

        let num_provers = job.num_provers;
        let prover_id = job.prover_id;
        let executor_result_tx = job.result_tx.clone();
        let executor_context = if prover_id == 0 {
            Some(job.context.clone())
        } else {
            None
        };
        let execution_instances_for_executor = execution_instances.clone();
        // A spawned thread does not inherit the ambient span, so carry it over
        // explicitly and every line the executor emits keeps its proof_id.
        let executor_span = Span::current();

        // --- Executor thread ---
        // Same as single-threaded version: discovers segments, sends via shared channel
        let executor_handle = std::thread::spawn(move || -> Result<ExecutorResult> {
            let _executor_span = executor_span.enter();
            let metered_interpreter = &execution_instances_for_executor.metered;
            let mut snapshots: VecDeque<VmSnapshot> = VecDeque::with_capacity(2);
            let snapshot_clone_start = std::time::Instant::now();
            if seeds_initial_snapshot(prover_id) {
                snapshots.push_back(VmSnapshot {
                    vm_state: vm_state.clone(),
                    instret: 0,
                });
            }
            let snapshot_clone_ms = snapshot_clone_start.elapsed().as_millis();

            let mut driver = MeteredDriver::new(metered_interpreter, vm_state, metered_ctx);
            let exec_start = std::time::Instant::now();

            info!(
                "VM setup: input_read={}ms, stdin={}ms, metered_ctx={}ms, vm_state={}ms, snapshot_clone={}ms, total={}ms",
                input_read_ms,
                stdin_ms,
                metered_ctx_ms,
                vm_state_ms,
                snapshot_clone_ms,
                setup_start.elapsed().as_millis()
            );
            info!("Executor thread (parallel): starting metered execution");

            loop {
                let should_break = driver.step()?;

                let curr_idx = driver.segments().len() - 1;

                if curr_idx % num_provers == prover_id {
                    let snapshot = if num_provers == 1 && curr_idx == 0 {
                        snapshots[0].clone()
                    } else {
                        snapshots
                            .pop_front()
                            .expect("Snapshot missing for assigned segment")
                    };

                    let segment = driver.segments()[curr_idx].clone();

                    info!(
                        "Executor (parallel): sending segment {} for proving",
                        curr_idx
                    );
                    prove_tx
                        .send(ProveData {
                            snapshot,
                            segment,
                            segment_idx: curr_idx,
                            is_final: should_break,
                            sent_at: Instant::now(),
                        })
                        .map_err(|_| eyre::eyre!("All prover consumers disconnected"))?;
                }

                if should_break {
                    break;
                }

                // Save snapshot for future assigned segment (+2 lookahead) —
                // see the snapshot-schedule notes on `seeds_initial_snapshot`
                // / `saves_snapshot_at` at the top of this file.
                if saves_snapshot_at(curr_idx, num_provers, prover_id) {
                    let instret = driver.instret();
                    snapshots.push_back(VmSnapshot {
                        vm_state: driver.clone_boundary_state(),
                        instret,
                    });
                }
            }

            let execute_time_ms = exec_start.elapsed().as_millis() as u64;
            let segments = driver.into_segments();
            let num_segments = segments.len();

            // Send ExecuteE2 result immediately so manager knows num_segments ASAP
            if let (Some(ref tx), Some(ref ctx)) = (&executor_result_tx, &executor_context) {
                let e2_result = ProofResult::ExecuteE2(ExecuteE2Result {
                    context: ctx.clone(),
                    state: ExecuteE2State {
                        num_segments,
                        segments: encode_segments(&segments)?,
                        cost: 0,
                        execute_time_ms,
                    },
                });
                let _ = tx.send(e2_result);
                info!(
                    "Executor (parallel): streamed ExecuteE2Result with num_segments={}",
                    num_segments
                );
            }

            info!(
                "Executor thread (parallel): finished, {} segments in {}ms",
                num_segments, execute_time_ms
            );

            Ok(ExecutorResult {
                execute_time_ms,
                segments,
                num_segments,
            })
        });

        // --- Coordinator acts as consumer-0 ---
        let pure_interpreter = &execution_instances.pure;
        let mut my_results: Vec<ProofResult> = Vec::new();
        let mut prover_err: Option<eyre::Report> = None;

        for prove_data in prove_rx.iter() {
            info!(
                "Coordinator-consumer: proving segment {}",
                prove_data.segment_idx
            );
            // How long the segment sat in the bounded channel. Near zero means
            // the provers keep up and the executor sets the pace.
            let queue_wait_ms = prove_data.sent_at.elapsed().as_millis() as u64;
            let segment_start = std::time::Instant::now();

            let fastfwd_start = std::time::Instant::now();
            let mut vm_state = prove_data.snapshot.vm_state;
            let num_ins = prove_data.segment.instret_start - prove_data.snapshot.instret;
            if num_ins > 0 {
                match fast_forward(pure_interpreter, vm_state, num_ins) {
                    Ok(state) => vm_state = state,
                    Err(e) => {
                        prover_err = Some(e.into());
                        break;
                    }
                }
            }
            let fastfwd_time_ms = fastfwd_start.elapsed().as_millis() as u64;

            let _ = telemetry::span_timing::drain_span_timings();
            let stark_start = std::time::Instant::now();
            // openvm v2.1 ships only touched-marked pages to the device (unmarked
            // → zero). Our snapshot + fast-forward reconstructs correct memory
            // *values* but not the touched-page metadata a sequential preflight
            // would carry, so rebuild it from the image before proving.
            vm_state.memory.memory.recompute_touched_pages();
            let prove_result = app_prover.prove_segment(vm_state, &prove_data.segment);
            let (proof, final_memory) = match prove_result {
                Ok(r) => r,
                Err(e) => {
                    prover_err = Some(e.into());
                    break;
                }
            };
            let stark_prove_time_ms = stark_start.elapsed().as_millis() as u64;
            let sub_metrics = telemetry::span_timing::drain_span_timings();
            let prove_time_ms = segment_start.elapsed().as_millis() as u64;

            // Terminal segment: compute user public values, and (on a
            // deferral job) extract the depth-independent
            // `(DEFERRAL_AS, 0)` authentication path from the same
            // memory image.
            let mut user_public_values = None;
            let mut final_merkle_path_bytes: Option<Vec<u8>> = None;
            let mut deferral_merkle_proofs_bytes: Option<Vec<u8>> = None;
            if prove_data.is_final {
                let memory = match final_memory.ok_or_else(|| {
                    eyre::eyre!(
                        "Terminal segment {} did not return final_memory",
                        prove_data.segment_idx
                    )
                }) {
                    Ok(m) => m,
                    Err(e) => {
                        prover_err = Some(e);
                        break;
                    }
                };
                let top_tree =
                    match app_prover.instance.vm.memory_top_tree().ok_or_else(|| {
                        eyre::eyre!("Memory top tree should exist for terminal segment")
                    }) {
                        Ok(t) => t,
                        Err(e) => {
                            prover_err = Some(e);
                            break;
                        }
                    };
                let memory_dimensions = vm_config.system.config.memory_config.memory_dimensions();
                let hasher = vm_poseidon2_hasher();
                user_public_values = Some(UserPublicValuesProof::compute(
                    &vm_config.system.config,
                    &hasher,
                    &memory.memory,
                    top_tree,
                ));
                if is_deferral_job {
                    match terminal_deferral_path_bytes(&memory.memory, &memory_dimensions, &hasher)
                    {
                        Ok(bytes) => final_merkle_path_bytes = Some(bytes),
                        Err(e) => {
                            prover_err = Some(e);
                            break;
                        }
                    }
                } else {
                    // Deferral deployment + no deferral input for this proof →
                    // build the depth-0 DeferralMerkleProofs the deferral VK
                    // still requires. `None` (no-op) on a non-deferral deployment.
                    match terminal_depth0_deferral_proofs_bytes(
                        &job.context.program,
                        &vm_config.system.config,
                        &memory.memory,
                        &hasher,
                    ) {
                        Ok(bytes) => deferral_merkle_proofs_bytes = bytes,
                        Err(e) => {
                            prover_err = Some(e);
                            break;
                        }
                    }
                }
            }

            let spans = telemetry::span_timing::format_span_timings(&sub_metrics);
            let app_proof = ProofResult::App(AppProof {
                context: job.context.clone(),
                state: AppProofState {
                    proof: Some(proof::encode_proof(&ProofWithPublicValue {
                        proof,
                        user_public_values,
                    })?),
                    segment_idx: prove_data.segment_idx,
                    prove_time_ms,
                    fastfwd_time_ms,
                    stark_prove_time_ms,
                    sub_metrics,
                    final_merkle_path_bytes,
                    deferral_merkle_proofs_bytes,
                },
            });

            info!(
                "Coordinator-consumer: proved segment {}: queue_wait={}ms, fastfwd={}ms, stark={}ms, prove={}ms, spans={}",
                prove_data.segment_idx,
                queue_wait_ms,
                fastfwd_time_ms,
                stark_prove_time_ms,
                prove_time_ms,
                spans
            );

            if let Some(ref tx) = job.result_tx {
                let _ = tx.send(app_proof);
            } else {
                my_results.push(app_proof);
            }
        }

        // Drop prove_rx to unblock executor if still sending
        drop(prove_rx);

        // Wait for executor
        let executor_join = executor_handle
            .join()
            .map_err(|e| eyre::eyre!("Executor thread panicked: {:?}", e))?;

        if let Some(e) = prover_err {
            return Err(e);
        }

        let executor_result = executor_join?;

        if executor_result.num_segments == 0 {
            return Err(eyre::eyre!("No segments produced from execution"));
        }

        // E2 result: already streamed from executor thread when result_tx is set.
        // Only add to batch in non-streaming mode.
        if job.prover_id == 0 && job.result_tx.is_none() {
            my_results.insert(
                0,
                ProofResult::ExecuteE2(ExecuteE2Result {
                    context: job.context.clone(),
                    state: ExecuteE2State {
                        num_segments: executor_result.num_segments,
                        segments: encode_segments(&executor_result.segments)?,
                        cost: 0,
                        execute_time_ms: executor_result.execute_time_ms,
                    },
                }),
            );
        }

        info!(
            "Parallel prove complete: {} segments (max_app_provers={})",
            executor_result.num_segments, job.max_app_provers
        );

        Ok(my_results)
    }

    /// Legacy convenience implementation that builds everything inline.
    ///
    /// Mainly for one-shot test tools. Production paths use
    /// `prove_sharded_app_with_prover` with pre-built `AppExecutionInstances`
    /// + an externally-managed `ProverType`.
    pub fn prove_sharded_app_impl(job: ShardedAppProverJob) -> Result<Vec<ProofResult>> {
        let artifact_store =
            ArtifactStore::global().ok_or_else(|| eyre::eyre!("Artifact store not initialized"))?;
        let edge_artifacts = artifact_store
            .get_edge_artifacts()
            .ok_or_else(|| eyre::eyre!("Edge artifacts not loaded"))?;
        let exe = artifact_store.vmexe(&job.context.program).ok_or_else(|| {
            eyre::eyre!(
                "vmexe for program {} not loaded on this worker",
                job.context.program
            )
        })?;
        let instances =
            AppExecutionInstances::new(&job.context.program, &edge_artifacts.app_pk, exe.clone())?;
        let mut prover = build_gpu_prover(&job.context.program, &edge_artifacts.app_pk, exe)?;
        prove_sharded_app_impl_with_prover(job, &instances, &mut prover)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn consumer_failure_returns_prover_error() {
            let result = consumer_failure("consumer failed on segment 7".to_string());
            let ProverResult::Error(error) = result else {
                panic!("expected ProverResult::Error");
            };
            assert_eq!(error, "consumer failed on segment 7");
        }
    }
}

#[cfg(not(feature = "mock-provers"))]
pub use real_impl::{
    build_gpu_prover, create_parallel_prove_jobs, prove_sharded_app_with_prover,
    AppExecutionInstances, ParallelCoordinatorFn, ProverType, SegmentConsumerFn,
};

#[cfg(not(feature = "mock-provers"))]
fn prove_sharded_app_impl(job: ShardedAppProverJob) -> Result<Vec<ProofResult>> {
    real_impl::prove_sharded_app_impl(job)
}

#[cfg(all(test, feature = "mock-provers"))]
mod tests {
    use super::*;
    use protocol::ProofContext;

    #[test]
    fn test_sharded_app_prove_mock() {
        let job = ShardedAppProverJob {
            context: ProofContext::new(
                "test-proof-id".to_string(),
                protocol::ProgramRef::new("test-program", 1),
                Default::default(),
            ),
            num_provers: 4,
            prover_id: 0,
            input_path: "/tmp/test-input".to_string(),
            segment_memory: None,
            max_app_provers: 1,
            result_tx: None,
            deferral_state_paths: vec![],
        };

        let result = prove_sharded_app(job);

        match result {
            ProverResult::Success(proofs) => {
                assert!(!proofs.is_empty());

                // First result should be ExecuteE2Result for prover_id=0
                let mut found_e2_result = false;
                let mut app_proof_count = 0;

                for proof in proofs {
                    match proof {
                        ProofResult::ExecuteE2(e2_result) => {
                            assert_eq!(e2_result.state.num_segments, 16);
                            found_e2_result = true;
                        }
                        ProofResult::App(app_proof) => {
                            assert!(app_proof.state.segment_idx < 16);
                            app_proof_count += 1;
                        }
                        _ => panic!("Unexpected proof type"),
                    }
                }

                // Worker 0 should emit ExecuteE2Result
                assert!(found_e2_result, "Expected ExecuteE2Result from worker 0");
                // Worker 0 with 4 workers handles segments 0, 4, 8, 12 (round-robin)
                assert_eq!(app_proof_count, 4, "Expected 4 app proofs");
            }
            ProverResult::Error(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_sharded_app_prove_non_zero_worker() {
        let job = ShardedAppProverJob {
            context: ProofContext::new(
                "test-proof-id".to_string(),
                protocol::ProgramRef::new("test-program", 1),
                Default::default(),
            ),
            num_provers: 4,
            prover_id: 1, // Non-zero worker
            input_path: "/tmp/test-input".to_string(),
            segment_memory: None,
            max_app_provers: 1,
            result_tx: None,
            deferral_state_paths: vec![],
        };

        let result = prove_sharded_app(job);

        match result {
            ProverResult::Success(proofs) => {
                // Non-zero worker should NOT emit ExecuteE2Result
                for proof in &proofs {
                    if matches!(proof, ProofResult::ExecuteE2(_)) {
                        panic!("Worker 1 should not emit ExecuteE2Result");
                    }
                }

                // Worker 1 with 4 workers handles segments 1, 5, 9, 13 (round-robin)
                assert_eq!(proofs.len(), 4, "Expected 4 app proofs");

                // Verify segment indices are correct (round-robin)
                let segment_indices: Vec<usize> = proofs
                    .iter()
                    .filter_map(|p| match p {
                        ProofResult::App(app) => Some(app.state.segment_idx),
                        _ => None,
                    })
                    .collect();
                assert_eq!(segment_indices, vec![1, 5, 9, 13]);
            }
            ProverResult::Error(e) => panic!("Unexpected error: {}", e),
        }
    }
}
