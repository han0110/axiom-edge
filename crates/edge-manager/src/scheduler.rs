//! Edge work scheduler and state store.
//!
//! Manages per-proof state and work assignment to workers.

use dashmap::DashMap;
use eyre::Result;
use serde::Serialize;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

use crate::priority_queue::{EdgeWorkItem, PriorityWorkQueue};
use crate::worker_registry::RegisteredWorker;
use protocol::{GeneralProveRequest, MessageEnvelope, ProofResult, Step, WorkerRole};

/// Assigned work item ready to be sent to a worker.
#[derive(Clone)]
pub struct AssignedWork {
    pub proof_uuid: String,
    pub worker_id: usize,
    pub worker_url: String,
    pub envelope: MessageEnvelope<GeneralProveRequest>,
    pub step: Step,
}

/// Per-worker debug state for an in-flight proof.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerDebugState {
    pub worker_id: usize,
    pub worker_url: String,
    pub active_proof_count: usize,
    pub active_steps: Vec<String>,
    /// Number of app segment results received from this worker (raw count).
    pub completed_segments_received: usize,
    /// Number of received segments matching scheduler ownership rule (`seg % num_workers == worker_id`).
    pub completed_segments_mod_match: Option<usize>,
    /// Expected number of segments by scheduler ownership rule.
    pub expected_segments_mod_match: Option<usize>,
    /// Remaining segments by scheduler ownership rule (`expected - matched`, min 0).
    pub remaining_segments_mod_match: Option<usize>,
}

/// Debug snapshot for a proof UUID.
#[derive(Debug, Clone, Serialize)]
pub struct ProofDebugState {
    pub proof_uuid: String,
    pub num_workers: usize,
    pub num_segments: Option<usize>,
    pub pending_work_empty: bool,
    pub workers: Vec<WorkerDebugState>,
}

/// Track what proof types a worker is currently running.
#[derive(Clone, Debug)]
struct ActiveProof {
    step: Step,
    layer_idx: Option<usize>,
}

/// Per-worker state within a proof.
#[derive(Clone)]
struct EdgeWorkerState {
    id: usize,
    worker_url: String,
    /// Deployment role. Gates which steps this worker can accept: an
    /// `EvmDedicated` worker takes only the `EvmProve` step (single slot) and
    /// no app/leaf/internal work; `Full`/`StarkOnly` run STARK proving but not
    /// a dispatched EVM step. In a default deployment every worker is `Full`,
    /// so gating is a no-op and scheduling is unchanged.
    worker_role: WorkerRole,
    /// Number of active proofs (0-2 for leaf/internal, ShardedAppProve counts as 1)
    active_proof_count: usize,
    /// Track what proof types are currently running
    active_proofs: Vec<ActiveProof>,
    /// For sharded_app_prove: completed segments
    completed_segments: HashSet<usize>,
}

/// Per-proof Edge state tracking workers and pending work.
struct EdgeProofState {
    workers: Vec<EdgeWorkerState>,
    pending: PriorityWorkQueue,
    /// Total number of segments, learned from ExecuteE2Result
    num_segments: Option<usize>,
    /// Number of workers for this proof
    num_workers: usize,
    /// Total number of leaf proofs assigned so far
    assigned_leaf_proofs: usize,
    /// Threshold for leaf packing optimization
    leaf_pack_threshold: usize,
    /// Per-worker concurrent leaf proof capacity (deployment-wide).
    max_leaf_provers: usize,
}

fn active_step_label(active: &ActiveProof) -> String {
    if active.step == Step::InternalProve {
        let layer = active.layer_idx.unwrap_or(usize::MAX);
        return format!("{}(layer={})", active.step.as_str(), layer);
    }
    active.step.as_str().to_string()
}

/// Create ActiveProof from request.
fn active_proof_from_request(request: &GeneralProveRequest, step: Step) -> ActiveProof {
    let layer_idx = match request {
        GeneralProveRequest::InternalProve(req) => Some(req.layer_idx),
        _ => None,
    };
    ActiveProof { step, layer_idx }
}

/// Determine if a worker can accept a specific type of work.
fn can_accept_work(worker: &EdgeWorkerState, step: Step, max_leaf_provers: usize) -> bool {
    // The dispatched EVM step (`EvmProve` = root → halo2) goes to any worker
    // whose role `runs_evm_prove()` (`Full` or `EvmDedicated`). It runs in
    // ISOLATION: accept only when the worker is completely idle, so halo2 never
    // overlaps app/leaf/internal or a second EVM step. Same rule for both roles
    // (the EVM step only ever fires after the recursion tree drains, so the
    // eligible workers are idle by then); the reverse guard below keeps other
    // work off a worker that is already running one.
    if step == Step::EvmProve {
        if !worker.worker_role.runs_evm_prove() {
            // `StarkOnly` (dedicated-halo2 mode) has no root/halo2 provers.
            return false;
        }
        return worker.active_proof_count == 0;
    }

    // The `EvmDedicated` worker runs ONLY the EVM step (handled above): it owns
    // no shard and takes no app/leaf/internal work. This also keeps a dedicated
    // worker out of `find_best_worker_for_leaf`'s "free worker" pick, since it
    // never carries a ShardedAppProve marker.
    if worker.worker_role == WorkerRole::EvmDedicated {
        return false;
    }

    // An in-flight EVM step runs in isolation (see the `EvmProve` branch): a
    // worker already running one accepts no other work until it completes.
    if worker
        .active_proofs
        .iter()
        .any(|p| p.step == Step::EvmProve)
    {
        return false;
    }

    // Workers doing the sharded app prove kickoff cannot accept any other work.
    if worker
        .active_proofs
        .iter()
        .any(|p| p.step == Step::ShardedAppProve)
    {
        return false;
    }

    match step {
        Step::LeafProve => worker.active_proof_count < max_leaf_provers,
        Step::InternalProve => worker.active_proof_count == 0,
        // ShardedAppProve is dispatched separately (round-robin pre-assigned
        // at proof start), so the manager never schedules it via this path.
        Step::ShardedAppProve => false,
        // Handled at the top of the function for every role.
        Step::EvmProve => false,
        // RootProve / Halo2Prove run in-process on the worker that produced
        // the final internal proof — no manager dispatch.
        Step::RootProve | Step::Halo2Prove => false,
    }
}

/// Find best worker for leaf proof assignment.
fn find_best_worker_for_leaf(
    workers: &[EdgeWorkerState],
    assigned_leaf_proofs: usize,
    leaf_pack_threshold: usize,
    max_leaf_provers: usize,
) -> Option<usize> {
    // First, try to find a completely free worker. Exclude the `EvmDedicated`
    // worker: it is idle (no ShardedAppProve marker, active_proof_count 0) but
    // runs only the EVM step, so it must never be picked for a leaf proof.
    if let Some(idx) = workers
        .iter()
        .enumerate()
        .find(|(_, w)| {
            w.worker_role.runs_stark_proving()
                && w.active_proof_count == 0
                && !w
                    .active_proofs
                    .iter()
                    .any(|p| p.step == Step::ShardedAppProve)
        })
        .map(|(idx, _)| idx)
    {
        return Some(idx);
    }

    // No free workers. Check if we should pack or queue.
    if assigned_leaf_proofs < leaf_pack_threshold {
        workers
            .iter()
            .enumerate()
            .filter(|(_, w)| {
                w.worker_role.runs_stark_proving()
                    && w.active_proof_count < max_leaf_provers
                    && !w
                        .active_proofs
                        .iter()
                        .any(|p| p.step == Step::ShardedAppProve)
            })
            .min_by_key(|(_, w)| {
                let active_bias = if let Some(active) = w.active_proofs.first() {
                    match active.step {
                        Step::InternalProve => {
                            1000usize.saturating_sub(active.layer_idx.unwrap_or(0) * 10)
                        }
                        Step::LeafProve => 2000,
                        _ => 1500,
                    }
                } else {
                    1500
                };
                w.active_proof_count.saturating_mul(10_000) + active_bias
            })
            .map(|(idx, _)| idx)
    } else {
        None
    }
}

impl EdgeProofState {
    fn new(
        workers: Vec<EdgeWorkerState>,
        num_workers: usize,
        leaf_pack_threshold: usize,
        max_leaf_provers: usize,
    ) -> Self {
        Self {
            workers,
            pending: PriorityWorkQueue::new(),
            num_segments: None,
            num_workers,
            assigned_leaf_proofs: 0,
            leaf_pack_threshold,
            max_leaf_provers,
        }
    }

    /// Check if worker has completed ALL their assigned segments.
    /// Uses O(1) arithmetic to calculate expected count instead of O(n) iteration.
    fn worker_completed_all_segments(
        worker_id: usize,
        completed_segments: &HashSet<usize>,
        num_workers: usize,
        total_segments: Option<usize>,
    ) -> bool {
        let Some(total) = total_segments else {
            return false;
        };

        // Calculate expected segment count using arithmetic:
        // Worker N handles segments where segment_idx % num_workers == worker_id
        // That's ceil((total - worker_id) / num_workers) when worker_id < total
        let expected_count = if worker_id < total {
            (total - worker_id).div_ceil(num_workers)
        } else {
            0
        };

        // Count segments in completed_segments that belong to this worker
        let actual_count = completed_segments
            .iter()
            .filter(|&&seg| seg % num_workers == worker_id)
            .count();

        let all_complete = actual_count >= expected_count;

        if all_complete && expected_count > 0 {
            info!(
                "Worker {} completed all segments: expected={}, actual={}, total_segments={}",
                worker_id, expected_count, actual_count, total
            );
        }

        all_complete
    }

    /// Assign work to an available worker or queue it.
    fn assign_or_queue(
        &mut self,
        proof_uuid: &str,
        envelope: MessageEnvelope<GeneralProveRequest>,
        step: Step,
    ) -> Option<AssignedWork> {
        let worker_idx = if step == Step::LeafProve {
            find_best_worker_for_leaf(
                &self.workers,
                self.assigned_leaf_proofs,
                self.leaf_pack_threshold,
                self.max_leaf_provers,
            )
        } else {
            self.workers
                .iter()
                .position(|w| can_accept_work(w, step, self.max_leaf_provers))
        };

        if let Some(idx) = worker_idx {
            let worker = &mut self.workers[idx];

            if step == Step::LeafProve {
                info!(
                    "Assigning LeafProve to worker {}: active_count={}",
                    worker.id, worker.active_proof_count
                );
            }

            worker.active_proof_count += 1;
            worker
                .active_proofs
                .push(active_proof_from_request(&envelope.message, step));

            if step == Step::LeafProve {
                self.assigned_leaf_proofs += 1;
            }

            return Some(AssignedWork {
                proof_uuid: proof_uuid.to_string(),
                worker_id: worker.id,
                worker_url: worker.worker_url.clone(),
                envelope,
                step,
            });
        }

        self.pending.push(EdgeWorkItem { envelope, step });
        None
    }

    /// Apply a worker's completion accounting for `result`, freeing the
    /// worker's busy slot where appropriate. Returns `true` when a slot was
    /// freed (i.e. pending work may now be dispatchable). Shared by the live
    /// path ([`complete_work`](Self::complete_work), which then dequeues) and
    /// the drain path (`EdgeStateStore::worker_drained`, which must NOT
    /// dispatch new work for a failing proof).
    fn apply_completion(
        &mut self,
        proof_uuid: &str,
        worker_id: usize,
        result: &ProofResult,
    ) -> Result<bool> {
        let worker = self
            .workers
            .iter_mut()
            .find(|w| w.id == worker_id)
            .ok_or_else(|| eyre::eyre!("Worker {worker_id} not registered for {proof_uuid}"))?;

        // Handle app_prove completion specially
        if let ProofResult::App(app_proof) = result {
            let segment_idx = app_proof.state.segment_idx;
            worker.completed_segments.insert(segment_idx);

            let all_complete = Self::worker_completed_all_segments(
                worker_id,
                &worker.completed_segments,
                self.num_workers,
                self.num_segments,
            );

            if !all_complete {
                return Ok(false);
            }
            worker.active_proof_count = worker.active_proof_count.saturating_sub(1);
            worker
                .active_proofs
                .retain(|p| p.step != Step::ShardedAppProve);
            info!(
                "Worker {} completed all {} segments for proof {}",
                worker_id,
                worker.completed_segments.len(),
                proof_uuid
            );
            return Ok(true);
        }

        if let ProofResult::ExecuteE2(_) = result {
            // ExecuteE2 is an informational result that tells us the segment count.
            // It does NOT indicate work completion - the worker is still busy with
            // ShardedAppProve until all App results are received.
            // Do not modify worker state here.
            return Ok(false);
        }

        // Leaf / Internal / EVM proof completion
        let completed_step = match result {
            ProofResult::Leaf(_) => Step::LeafProve,
            ProofResult::Internal(_) => Step::InternalProve,
            // Completing the dispatched EVM step frees the dedicated
            // worker's single slot so a queued EvmProve can dequeue. (In the
            // live flow an Evm result also completes the proof, so the
            // manager takes the terminal path and skips `worker_completed`;
            // this arm keeps the single-slot lifecycle correct when the
            // scheduler is driven directly.)
            ProofResult::Evm(_) => Step::EvmProve,
            ProofResult::Error(_) => {
                // Error result - the worker should be freed
                if !worker.active_proofs.is_empty() {
                    worker.active_proof_count = worker.active_proof_count.saturating_sub(1);
                    worker.active_proofs.remove(0);
                }
                return Ok(true);
            }
            _ => {
                // Unknown result type - ignore
                return Ok(false);
            }
        };

        worker.active_proof_count = worker.active_proof_count.saturating_sub(1);

        if let Some(pos) = worker
            .active_proofs
            .iter()
            .position(|p| p.step == completed_step)
        {
            worker.active_proofs.remove(pos);
        }
        Ok(true)
    }

    /// Process worker completion and attempt to dequeue pending work.
    fn complete_work(
        &mut self,
        proof_uuid: &str,
        worker_id: usize,
        result: &ProofResult,
    ) -> Result<Option<AssignedWork>> {
        if self.apply_completion(proof_uuid, worker_id, result)? {
            self.try_dequeue_work(proof_uuid)
        } else {
            Ok(None)
        }
    }

    /// Try to dequeue pending work and assign it to an available worker.
    fn try_dequeue_work(&mut self, proof_uuid: &str) -> Result<Option<AssignedWork>> {
        if let Some(item) = self.pending.pop() {
            let step = item.step;

            let worker_idx = if step == Step::LeafProve {
                find_best_worker_for_leaf(
                    &self.workers,
                    self.assigned_leaf_proofs,
                    self.leaf_pack_threshold,
                    self.max_leaf_provers,
                )
            } else {
                self.workers
                    .iter()
                    .position(|w| can_accept_work(w, step, self.max_leaf_provers))
            };

            if let Some(idx) = worker_idx {
                let available_worker = &mut self.workers[idx];
                available_worker.active_proof_count += 1;
                available_worker
                    .active_proofs
                    .push(active_proof_from_request(&item.envelope.message, step));

                if step == Step::LeafProve {
                    self.assigned_leaf_proofs += 1;
                }

                return Ok(Some(AssignedWork {
                    proof_uuid: proof_uuid.to_string(),
                    worker_id: available_worker.id,
                    worker_url: available_worker.worker_url.clone(),
                    envelope: item.envelope,
                    step: item.step,
                }));
            } else {
                self.pending.requeue(item);
            }
        }
        Ok(None)
    }
}

/// Store for per-proof Edge state.
///
/// Holds deployment-wide capacity values (`max_leaf_provers`) read from
/// manager config and used uniformly across all workers in this deployment.
pub struct EdgeStateStore {
    proofs: DashMap<String, Arc<Mutex<EdgeProofState>>>,
    max_leaf_provers: usize,
}

impl EdgeStateStore {
    pub fn new(max_leaf_provers: usize) -> Self {
        Self {
            proofs: DashMap::new(),
            max_leaf_provers,
        }
    }

    /// Initialize proof state with workers.
    ///
    /// `workers` is the **full** ready set (including any `EvmDedicated`
    /// worker, so the `EvmProve` step can be dispatched to it). The
    /// sharding-relevant worker count (`num_workers`, which drives the
    /// `segment % num_workers == prover_id` ownership math) is the
    /// **app-eligible (normal) set size** — every worker whose role runs
    /// STARK proving — NOT `workers.len()`. The `EvmDedicated` worker is seeded
    /// idle (no ShardedAppProve, `active_proof_count = 0`) so it owns no shard
    /// and `set_num_segments` skips it; it only ever accepts `EvmProve`.
    ///
    /// In a default deployment every worker is `Full`, so `num_workers ==
    /// workers.len()` and every worker gets a ShardedAppProve slot exactly as
    /// today — behavior is unchanged.
    pub fn init_proof(
        &self,
        proof_uuid: &str,
        workers: Vec<(usize, RegisteredWorker)>,
        leaf_pack_threshold: usize,
    ) {
        // num_workers = app-eligible (normal) set size, not the full count.
        let num_workers = workers
            .iter()
            .filter(|(_, w)| w.worker_role.runs_stark_proving())
            .count();
        info!(
            "init_proof {}: {} workers registered ({} app-eligible / normal-set)",
            proof_uuid,
            workers.len(),
            num_workers
        );

        let worker_states = workers
            .into_iter()
            .map(|(id, worker)| {
                // The `EvmDedicated` worker runs only the EVM step: seed it idle
                // with no ShardedAppProve so it owns no shard and is never
                // treated as a busy app worker. STARK-proving workers start the
                // sharded app prove exactly as today.
                let runs_stark = worker.worker_role.runs_stark_proving();
                EdgeWorkerState {
                    id,
                    worker_url: worker.worker_url,
                    worker_role: worker.worker_role,
                    active_proof_count: if runs_stark { 1 } else { 0 },
                    active_proofs: if runs_stark {
                        vec![ActiveProof {
                            step: Step::ShardedAppProve,
                            layer_idx: None,
                        }]
                    } else {
                        Vec::new()
                    },
                    completed_segments: HashSet::new(),
                }
            })
            .collect();

        let state = Arc::new(Mutex::new(EdgeProofState::new(
            worker_states,
            num_workers,
            leaf_pack_threshold,
            self.max_leaf_provers,
        )));
        self.proofs.insert(proof_uuid.to_string(), state);
    }

    /// Set the number of segments for a proof.
    pub async fn set_num_segments(&self, proof_uuid: &str, num_segments: usize) -> Result<()> {
        let state = {
            let Some(entry) = self.proofs.get(proof_uuid) else {
                return Ok(());
            };
            entry.value().clone()
        };
        let mut guard = state.lock().await;
        guard.num_segments = Some(num_segments);

        info!(
            "set_num_segments for {}: num_segments={}, num_workers={}",
            proof_uuid, num_segments, guard.num_workers
        );

        // Check if any workers can now have their ShardedAppProve marked as complete
        // This is important for workers that have no assigned segments (num_segments < num_workers)
        // or workers that have already completed all their segments
        let num_workers = guard.num_workers;
        for worker in &mut guard.workers {
            if worker.active_proof_count > 0 {
                // Check if this worker has any segments assigned
                // Using the same calculation as worker_completed_all_segments
                let expected_count = if worker.id < num_segments {
                    (num_segments - worker.id).div_ceil(num_workers)
                } else {
                    0
                };

                // If no segments assigned, or all segments completed, mark as done
                let all_complete = expected_count == 0
                    || EdgeProofState::worker_completed_all_segments(
                        worker.id,
                        &worker.completed_segments,
                        num_workers,
                        Some(num_segments),
                    );
                if all_complete {
                    info!(
                        "set_num_segments: Worker {} completed (expected={}, completed={})",
                        worker.id,
                        expected_count,
                        worker.completed_segments.len()
                    );
                    worker.active_proof_count = worker.active_proof_count.saturating_sub(1);
                    worker
                        .active_proofs
                        .retain(|p| p.step != Step::ShardedAppProve);
                }
            }
        }
        Ok(())
    }

    /// Remove proof state.
    pub fn remove_proof(&self, proof_uuid: &str) {
        self.proofs.remove(proof_uuid);
    }

    /// Enqueue or assign work.
    pub async fn enqueue_or_assign(
        &self,
        proof_uuid: &str,
        envelope: MessageEnvelope<GeneralProveRequest>,
        step: Step,
    ) -> Result<Option<AssignedWork>> {
        let state = {
            let Some(entry) = self.proofs.get(proof_uuid) else {
                return Ok(None);
            };
            entry.value().clone()
        };
        let mut guard = state.lock().await;
        Ok(guard.assign_or_queue(proof_uuid, envelope, step))
    }

    /// Process worker completion.
    pub async fn worker_completed(
        &self,
        proof_uuid: &str,
        worker_id: usize,
        result: &ProofResult,
    ) -> Result<Option<AssignedWork>> {
        let state = {
            let Some(entry) = self.proofs.get(proof_uuid) else {
                return Ok(None);
            };
            entry.value().clone()
        };
        let mut guard = state.lock().await;
        guard.complete_work(proof_uuid, worker_id, result)
    }

    /// Drain-mode completion accounting for a proof already declared
    /// `Failing`: free the worker's busy slot for this late result WITHOUT
    /// dispatching pending work (no new work may start for a failing proof).
    /// Keeps `is_fully_drained` progressing so the drain orchestrator can
    /// transition `Failing` → `Failed` as soon as the last worker reports,
    /// instead of stalling until the drain-TTL backstop fires.
    pub async fn worker_drained(
        &self,
        proof_uuid: &str,
        worker_id: usize,
        result: &ProofResult,
    ) -> Result<()> {
        let state = {
            let Some(entry) = self.proofs.get(proof_uuid) else {
                return Ok(());
            };
            entry.value().clone()
        };
        let mut guard = state.lock().await;
        guard.apply_completion(proof_uuid, worker_id, result)?;
        Ok(())
    }

    /// Check if a proof exists.
    pub fn has_proof(&self, proof_uuid: &str) -> bool {
        self.proofs.contains_key(proof_uuid)
    }

    /// Check if any proofs exist.
    pub fn has_any_proofs(&self) -> bool {
        !self.proofs.is_empty()
    }

    /// Release a specific worker's scheduler-side busy slot for a proof.
    /// Used when the partial-dispatch path detects a worker never received
    /// `/sharded_app_prove` — that worker isn't actually doing anything for
    /// us, so it shouldn't count against drain progress.
    pub async fn release_worker(&self, proof_uuid: &str, worker_id: usize) {
        let state = {
            let Some(entry) = self.proofs.get(proof_uuid) else {
                return;
            };
            entry.value().clone()
        };
        let mut guard = state.lock().await;
        if let Some(worker) = guard.workers.iter_mut().find(|w| w.id == worker_id) {
            worker.active_proof_count = 0;
            worker.active_proofs.clear();
        }
    }

    /// Whether every worker for this proof has drained (active_proof_count == 0).
    /// Used by the drain orchestrator to know when a `Failing` proof can
    /// transition to `Failed`. Returns true if the proof is unknown (i.e.,
    /// already removed).
    pub async fn is_fully_drained(&self, proof_uuid: &str) -> bool {
        let state = {
            let Some(entry) = self.proofs.get(proof_uuid) else {
                return true;
            };
            entry.value().clone()
        };
        let guard = state.lock().await;
        guard.workers.iter().all(|w| w.active_proof_count == 0)
    }

    /// Return a scheduler debug snapshot for a proof.
    pub async fn proof_debug_state(&self, proof_uuid: &str) -> Option<ProofDebugState> {
        let state = {
            let entry = self.proofs.get(proof_uuid)?;
            entry.value().clone()
        };
        let guard = state.lock().await;
        let num_workers = guard.num_workers;
        let num_segments = guard.num_segments;

        let workers = guard
            .workers
            .iter()
            .map(|worker| {
                let (matched, expected, remaining) = if let Some(total) = num_segments {
                    let expected_segments = if worker.id < total {
                        (total - worker.id).div_ceil(num_workers)
                    } else {
                        0
                    };
                    let matched_segments = worker
                        .completed_segments
                        .iter()
                        .filter(|&&seg| seg % num_workers == worker.id)
                        .count();
                    let remaining_segments = expected_segments.saturating_sub(matched_segments);
                    (
                        Some(matched_segments),
                        Some(expected_segments),
                        Some(remaining_segments),
                    )
                } else {
                    (None, None, None)
                };

                WorkerDebugState {
                    worker_id: worker.id,
                    worker_url: worker.worker_url.clone(),
                    active_proof_count: worker.active_proof_count,
                    active_steps: worker.active_proofs.iter().map(active_step_label).collect(),
                    completed_segments_received: worker.completed_segments.len(),
                    completed_segments_mod_match: matched,
                    expected_segments_mod_match: expected,
                    remaining_segments_mod_match: remaining,
                }
            })
            .collect();

        Some(ProofDebugState {
            proof_uuid: proof_uuid.to_string(),
            num_workers,
            num_segments,
            pending_work_empty: guard.pending.is_empty(),
            workers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{AppProof, AppProofState, LeafProveRequest, ProofContext};
    use std::collections::HashMap;

    fn make_leaf_envelope(
        proof_uuid: &str,
        segment_start: usize,
        segment_end: usize,
    ) -> MessageEnvelope<GeneralProveRequest> {
        MessageEnvelope::new(
            GeneralProveRequest::LeafProve(LeafProveRequest {
                context: ProofContext::new(
                    proof_uuid.to_string(),
                    protocol::ProgramRef::new("test-program", 1),
                    Default::default(),
                ),
                app_proofs: Vec::new(),
                segment_start,
                segment_end,
            }),
            &format!("leaf-{segment_start}-{segment_end}"),
        )
    }

    fn make_test_workers(n: usize) -> Vec<(usize, RegisteredWorker)> {
        (0..n)
            .map(|i| {
                (
                    i,
                    RegisteredWorker {
                        worker_url: format!("http://worker-{}:8001", i),
                        last_seen: chrono::Utc::now(),
                        worker_role: protocol::WorkerRole::Full,
                    },
                )
            })
            .collect()
    }

    fn make_app_result(proof_uuid: &str, segment_idx: usize) -> ProofResult {
        ProofResult::App(AppProof {
            context: ProofContext::new(
                proof_uuid.to_string(),
                protocol::ProgramRef::new("test-program", 1),
                Default::default(),
            ),
            state: AppProofState {
                proof: Some(vec![0u8; 16]),
                segment_idx,
                prove_time_ms: 0,
                fastfwd_time_ms: 0,
                stark_prove_time_ms: 0,
                queue_wait_ms: 0,
                metered_time_ms: 0,
                sub_metrics: HashMap::new(),
                final_merkle_path_bytes: None,
                deferral_merkle_proofs_bytes: None,
                worker_id: 0,
                completed_at_ms: 0,
            },
        })
    }

    #[test]
    fn test_state_store_init() {
        let store = EdgeStateStore::new(2);
        assert!(!store.has_any_proofs());

        let workers = vec![
            (
                0,
                RegisteredWorker {
                    worker_url: "http://worker-0:8001".to_string(),
                    last_seen: chrono::Utc::now(),
                    worker_role: protocol::WorkerRole::Full,
                },
            ),
            (
                1,
                RegisteredWorker {
                    worker_url: "http://worker-1:8001".to_string(),
                    last_seen: chrono::Utc::now(),
                    worker_role: protocol::WorkerRole::Full,
                },
            ),
        ];

        store.init_proof("test-proof", workers, 48);
        assert!(store.has_proof("test-proof"));
        assert!(store.has_any_proofs());

        store.remove_proof("test-proof");
        assert!(!store.has_proof("test-proof"));
    }

    #[tokio::test]
    async fn test_worker_completed_with_fewer_segments_than_workers() {
        // When num_segments < num_workers, some workers have 0 assigned segments
        // They should be immediately released when set_num_segments is called

        let store = EdgeStateStore::new(2);
        let workers = vec![
            (
                0,
                RegisteredWorker {
                    worker_url: "http://worker-0:8001".to_string(),
                    last_seen: chrono::Utc::now(),
                    worker_role: protocol::WorkerRole::Full,
                },
            ),
            (
                1,
                RegisteredWorker {
                    worker_url: "http://worker-1:8001".to_string(),
                    last_seen: chrono::Utc::now(),
                    worker_role: protocol::WorkerRole::Full,
                },
            ),
            (
                2,
                RegisteredWorker {
                    worker_url: "http://worker-2:8001".to_string(),
                    last_seen: chrono::Utc::now(),
                    worker_role: protocol::WorkerRole::Full,
                },
            ),
            (
                3,
                RegisteredWorker {
                    worker_url: "http://worker-3:8001".to_string(),
                    last_seen: chrono::Utc::now(),
                    worker_role: protocol::WorkerRole::Full,
                },
            ),
        ];

        store.init_proof("test-proof", workers, 48);

        // All 4 workers start with ShardedAppProve (active_proof_count = 1)
        {
            let state = store.proofs.get("test-proof").unwrap().clone();
            let guard = state.lock().await;
            for worker in &guard.workers {
                assert_eq!(worker.active_proof_count, 1);
                assert!(worker
                    .active_proofs
                    .iter()
                    .any(|p| p.step == Step::ShardedAppProve));
            }
        }

        // Set num_segments = 2 (fewer than 4 workers)
        // Workers 0 and 1 should handle segments 0 and 1 respectively
        // Workers 2 and 3 should be immediately released (no segments assigned)
        store.set_num_segments("test-proof", 2).await.unwrap();

        {
            let state = store.proofs.get("test-proof").unwrap().clone();
            let guard = state.lock().await;

            // Workers 0 and 1 should still be busy (have segments to prove)
            assert_eq!(guard.workers[0].active_proof_count, 1);
            assert_eq!(guard.workers[1].active_proof_count, 1);

            // Workers 2 and 3 should be released (no segments assigned)
            assert_eq!(guard.workers[2].active_proof_count, 0);
            assert_eq!(guard.workers[3].active_proof_count, 0);

            // Workers 2 and 3 should not have ShardedAppProve in active_proofs
            assert!(!guard.workers[2]
                .active_proofs
                .iter()
                .any(|p| p.step == Step::ShardedAppProve));
            assert!(!guard.workers[3]
                .active_proofs
                .iter()
                .any(|p| p.step == Step::ShardedAppProve));
        }
    }

    #[test]
    fn test_round_robin_segment_calculation() {
        // Verify the round-robin segment count calculation is correct
        // Worker N handles segments where segment_idx % num_workers == N

        // 16 segments, 4 workers
        // Worker 0: 0, 4, 8, 12 = 4 segments
        // Worker 1: 1, 5, 9, 13 = 4 segments
        // Worker 2: 2, 6, 10, 14 = 4 segments
        // Worker 3: 3, 7, 11, 15 = 4 segments
        let completed: HashSet<usize> = [0, 4, 8, 12].into_iter().collect();
        assert!(EdgeProofState::worker_completed_all_segments(
            0,
            &completed,
            4,
            Some(16)
        ));

        let partial: HashSet<usize> = [0, 4, 8].into_iter().collect();
        assert!(!EdgeProofState::worker_completed_all_segments(
            0,
            &partial,
            4,
            Some(16)
        ));

        // 2 segments, 4 workers
        // Worker 0: 0 = 1 segment
        // Worker 1: 1 = 1 segment
        // Worker 2: none = 0 segments
        // Worker 3: none = 0 segments
        let empty: HashSet<usize> = HashSet::new();
        assert!(EdgeProofState::worker_completed_all_segments(
            2,
            &empty,
            4,
            Some(2)
        ));
        assert!(EdgeProofState::worker_completed_all_segments(
            3,
            &empty,
            4,
            Some(2)
        ));
    }

    #[tokio::test]
    async fn test_proof_debug_state_exposes_worker_progress() {
        let store = EdgeStateStore::new(2);
        let workers = vec![
            (
                0,
                RegisteredWorker {
                    worker_url: "http://worker-0:8001".to_string(),
                    last_seen: chrono::Utc::now(),
                    worker_role: protocol::WorkerRole::Full,
                },
            ),
            (
                1,
                RegisteredWorker {
                    worker_url: "http://worker-1:8001".to_string(),
                    last_seen: chrono::Utc::now(),
                    worker_role: protocol::WorkerRole::Full,
                },
            ),
        ];

        store.init_proof("debug-proof", workers, 48);

        {
            let state = store.proofs.get("debug-proof").unwrap().clone();
            let mut guard = state.lock().await;
            guard.num_segments = Some(5);

            // Worker 0 gets segments 0,2,4 by modulo rule.
            guard.workers[0].completed_segments.insert(0);
            guard.workers[0].completed_segments.insert(2);

            // Worker 1 gets segments 1,3 by modulo rule.
            // Insert one mismatched segment to ensure mod-match counter differs.
            guard.workers[1].completed_segments.insert(2);
        }

        let debug = store.proof_debug_state("debug-proof").await.unwrap();
        assert_eq!(debug.num_workers, 2);
        assert_eq!(debug.num_segments, Some(5));
        assert_eq!(debug.workers.len(), 2);

        let w0 = debug.workers.iter().find(|w| w.worker_id == 0).unwrap();
        assert_eq!(w0.completed_segments_received, 2);
        assert_eq!(w0.completed_segments_mod_match, Some(2));
        assert_eq!(w0.expected_segments_mod_match, Some(3));
        assert_eq!(w0.remaining_segments_mod_match, Some(1));

        let w1 = debug.workers.iter().find(|w| w.worker_id == 1).unwrap();
        assert_eq!(w1.completed_segments_received, 1);
        assert_eq!(w1.completed_segments_mod_match, Some(0));
        assert_eq!(w1.expected_segments_mod_match, Some(2));
        assert_eq!(w1.remaining_segments_mod_match, Some(2));
    }

    #[tokio::test]
    async fn test_leaf_assignment_respects_registered_leaf_capacity() {
        let store = EdgeStateStore::new(1);
        let workers = vec![(
            0,
            RegisteredWorker {
                worker_url: "http://worker-0:8001".to_string(),
                last_seen: chrono::Utc::now(),
                worker_role: protocol::WorkerRole::Full,
            },
        )];

        store.init_proof("leaf-capacity-proof", workers, 48);
        store
            .set_num_segments("leaf-capacity-proof", 0)
            .await
            .unwrap();

        let first = store
            .enqueue_or_assign(
                "leaf-capacity-proof",
                make_leaf_envelope("leaf-capacity-proof", 0, 3),
                Step::LeafProve,
            )
            .await
            .unwrap();
        assert!(first.is_some());

        let second = store
            .enqueue_or_assign(
                "leaf-capacity-proof",
                make_leaf_envelope("leaf-capacity-proof", 4, 7),
                Step::LeafProve,
            )
            .await
            .unwrap();
        assert!(second.is_none());

        let debug = store
            .proof_debug_state("leaf-capacity-proof")
            .await
            .unwrap();
        assert_eq!(debug.workers.len(), 1);
        assert_eq!(debug.workers[0].active_proof_count, 1);
        assert_eq!(
            debug.workers[0].active_steps,
            vec![Step::LeafProve.as_str().to_string()]
        );
        assert!(!debug.pending_work_empty);
    }

    // ------------------------------------------------------------------
    // Drain / failure-mode tests.
    // ------------------------------------------------------------------

    /// Partial-dispatch case: a worker's HTTP failed at start_proof, so the
    /// manager calls `release_worker` to free its scheduler slot. After all
    /// workers are released, `is_fully_drained` returns true so the proof
    /// can transition Failing → Failed and a new proof can start.
    #[tokio::test]
    async fn test_release_worker_marks_drained_for_partial_dispatch() {
        let store = EdgeStateStore::new(2);
        store.init_proof("p1", make_test_workers(3), 48);

        // Fresh init: every worker active_proof_count=1, nothing drained.
        assert!(!store.is_fully_drained("p1").await);

        // Worker 0's dispatch failed → release it. Others still busy.
        store.release_worker("p1", 0).await;
        assert!(!store.is_fully_drained("p1").await);

        // Workers 1 and 2 also failed (all-fail case) → release.
        store.release_worker("p1", 1).await;
        store.release_worker("p1", 2).await;
        assert!(store.is_fully_drained("p1").await);
    }

    /// Worker disappears mid-proof. Of N workers, one never reports
    /// completion. `is_fully_drained` stays false until either the
    /// remaining worker is explicitly released or the drain TTL fires
    /// (TTL is checked in handlers.rs, not here — this test verifies the
    /// stuck-without-release case).
    #[tokio::test]
    async fn test_is_fully_drained_false_when_worker_disappears() {
        let store = EdgeStateStore::new(2);
        store.init_proof("p1", make_test_workers(2), 48);
        store.set_num_segments("p1", 4).await.unwrap();

        // Worker 0 completes its segments (0, 2 by modulo).
        store
            .worker_completed("p1", 0, &make_app_result("p1", 0))
            .await
            .unwrap();
        store
            .worker_completed("p1", 0, &make_app_result("p1", 2))
            .await
            .unwrap();

        // Worker 1 never reports. Drain check still false.
        assert!(!store.is_fully_drained("p1").await);

        // Simulate operator-side release (or TTL backstop in handlers.rs)
        // by manually releasing worker 1.
        store.release_worker("p1", 1).await;
        assert!(store.is_fully_drained("p1").await);
    }

    /// Drain-by-completion: a late result for a `Failing` proof frees the
    /// worker's busy slot via `worker_drained` so `is_fully_drained` can
    /// progress — but pending work must NOT be dispatched (nothing new may
    /// start for a failing proof).
    #[tokio::test]
    async fn test_worker_drained_frees_slot_without_dispatching_pending() {
        // max_leaf_provers = 1: the single worker holds one leaf, a second
        // leaf queues.
        let store = EdgeStateStore::new(1);
        store.init_proof("p1", make_test_workers(1), 48);
        store.set_num_segments("p1", 0).await.unwrap();

        let first = store
            .enqueue_or_assign("p1", make_leaf_envelope("p1", 0, 3), Step::LeafProve)
            .await
            .unwrap();
        assert!(first.is_some(), "first leaf dispatches");
        let second = store
            .enqueue_or_assign("p1", make_leaf_envelope("p1", 4, 7), Step::LeafProve)
            .await
            .unwrap();
        assert!(second.is_none(), "second leaf queues");
        assert!(!store.is_fully_drained("p1").await);

        // The proof is declared Failing; the in-flight leaf's result arrives
        // late and is routed through drain accounting.
        let leaf_result = ProofResult::Leaf(protocol::LeafProof {
            context: ProofContext::new(
                "p1".to_string(),
                protocol::ProgramRef::new("test-program", 1),
                Default::default(),
            ),
            state: protocol::LeafProofState {
                proof: Some(vec![0u8; 16]),
                segment_start: 0,
                segment_end: 3,
                prove_time_ms: 0,
                sub_metrics: HashMap::new(),
                worker_id: 0,
                completed_at_ms: 0,
            },
        });
        store.worker_drained("p1", 0, &leaf_result).await.unwrap();

        // Slot freed → fully drained; the queued leaf stays queued.
        assert!(store.is_fully_drained("p1").await);
        let debug = store.proof_debug_state("p1").await.unwrap();
        assert_eq!(debug.workers[0].active_proof_count, 0);
        assert!(
            !debug.pending_work_empty,
            "pending work must not dispatch during drain"
        );
    }

    /// Timeout + cleanup: `remove_proof` (called by `finalize_proof` after
    /// the drain orchestrator transitions Failing → Failed) clears the
    /// proof from the scheduler, freeing workers globally. After cleanup,
    /// `has_any_proofs` is false so `/start_proof` can accept a new proof.
    #[tokio::test]
    async fn test_remove_proof_frees_scheduler_after_drain() {
        let store = EdgeStateStore::new(2);
        store.init_proof("p1", make_test_workers(2), 48);
        store.set_num_segments("p1", 4).await.unwrap();
        assert!(store.has_any_proofs());

        // Workers complete their shards (simulating successful drain).
        for (worker_id, segments) in [(0, [0usize, 2].as_slice()), (1, [1, 3].as_slice())] {
            for &seg in segments {
                store
                    .worker_completed("p1", worker_id, &make_app_result("p1", seg))
                    .await
                    .unwrap();
            }
        }
        assert!(store.is_fully_drained("p1").await);

        // remove_proof clears the scheduler state.
        store.remove_proof("p1");
        assert!(!store.has_any_proofs());
        assert!(!store.has_proof("p1"));
    }

    // ------------------------------------------------------------------
    // EvmProve step lifecycle: the EVM step's dispatch / queue / retry
    // lifecycle. The step dispatches to any `runs_evm_prove()` worker (`Full`
    // or `EvmDedicated`); these tests cover both the dedicated-split layout
    // and the all-`Full` fleet, plus the per-role accept rule.
    // ------------------------------------------------------------------

    /// `n_normal` StarkOnly workers at ids `0..n_normal-1` plus one `EvmDedicated`
    /// worker at the top id `n_normal` — the supported dedicated-halo2 layout.
    fn make_workers_with_dedicated(n_normal: usize) -> Vec<(usize, RegisteredWorker)> {
        let mut workers = (0..n_normal)
            .map(|i| {
                (
                    i,
                    RegisteredWorker {
                        worker_url: format!("http://worker-{i}:8001"),
                        last_seen: chrono::Utc::now(),
                        worker_role: WorkerRole::StarkOnly,
                    },
                )
            })
            .collect::<Vec<_>>();
        workers.push((
            n_normal,
            RegisteredWorker {
                worker_url: format!("http://worker-{n_normal}:8001"),
                last_seen: chrono::Utc::now(),
                worker_role: WorkerRole::EvmDedicated,
            },
        ));
        workers
    }

    fn make_evm_prove_envelope(proof_uuid: &str) -> MessageEnvelope<GeneralProveRequest> {
        MessageEnvelope::new(
            GeneralProveRequest::EvmProve(protocol::EvmProveRequest {
                context: ProofContext::new(
                    proof_uuid.to_string(),
                    protocol::ProgramRef::new("test-program", 1),
                    Default::default(),
                ),
                internal_proof_bytes: vec![0u8; 32],
                deferral_merkle_proofs_bytes: None,
                proof_has_deferral: false,
            }),
            "evm-prove",
        )
    }

    fn make_evm_result(proof_uuid: &str) -> ProofResult {
        ProofResult::Evm(protocol::EvmProof {
            context: ProofContext::new(
                proof_uuid.to_string(),
                protocol::ProgramRef::new("test-program", 1),
                Default::default(),
            ),
            state: protocol::EvmProofState {
                proof: Some(vec![0u8; 64]),
                prove_time_ms: 0,
                root_prove_time_ms: 0,
                sub_metrics: std::collections::HashMap::new(),
            },
        })
    }

    /// EvmProve routes only to the `EvmDedicated` worker: with 2 StarkOnly workers
    /// (ids 0,1) and a dedicated worker (id 2), the dispatch lands on id 2.
    #[tokio::test]
    async fn test_evm_prove_dispatched_to_dedicated_worker() {
        let store = EdgeStateStore::new(2);
        store.init_proof("p-evm", make_workers_with_dedicated(2), 48);
        // Clear the StarkOnly workers' ShardedAppProve slots (0 segments) so they
        // are idle and would otherwise be picked for any generic work.
        store.set_num_segments("p-evm", 0).await.unwrap();

        let assigned = store
            .enqueue_or_assign("p-evm", make_evm_prove_envelope("p-evm"), Step::EvmProve)
            .await
            .unwrap()
            .expect("EvmProve should dispatch to the dedicated worker");
        assert_eq!(
            assigned.worker_id, 2,
            "EvmProve must go to the EvmDedicated worker (top id), not a StarkOnly worker"
        );
        assert_eq!(assigned.step, Step::EvmProve);
    }

    /// Single slot (queue-of-1): a second EvmProve queues while the dedicated
    /// worker is busy, then dispatches on completion of the first — never
    /// erroring. Also exercises completion → dequeue via an `Evm` result.
    #[tokio::test]
    async fn test_evm_prove_single_slot_queue_and_completion() {
        let store = EdgeStateStore::new(2);
        store.init_proof("p-evm", make_workers_with_dedicated(1), 48);
        store.set_num_segments("p-evm", 0).await.unwrap();
        let dedicated_id = 1;

        // First EvmProve occupies the single slot on the dedicated worker.
        let first = store
            .enqueue_or_assign("p-evm", make_evm_prove_envelope("p-evm"), Step::EvmProve)
            .await
            .unwrap()
            .expect("first EvmProve dispatches");
        assert_eq!(first.worker_id, dedicated_id);

        // Second EvmProve queues (single slot busy) rather than erroring.
        let second = store
            .enqueue_or_assign("p-evm", make_evm_prove_envelope("p-evm"), Step::EvmProve)
            .await
            .unwrap();
        assert!(
            second.is_none(),
            "concurrent EvmProve must queue while the dedicated slot is busy"
        );
        assert!(
            !store
                .proof_debug_state("p-evm")
                .await
                .unwrap()
                .pending_work_empty,
            "the queued EvmProve should be pending, not dropped"
        );

        // Completing the first EVM step (Evm result) frees the slot and
        // dequeues the queued EvmProve onto the same dedicated worker.
        let dequeued = store
            .worker_completed("p-evm", dedicated_id, &make_evm_result("p-evm"))
            .await
            .unwrap()
            .expect("queued EvmProve dispatches on completion");
        assert_eq!(dequeued.worker_id, dedicated_id);
        assert_eq!(dequeued.step, Step::EvmProve);
    }

    /// The dedicated worker takes NO leaf/internal work: with the sole StarkOnly
    /// worker busy, a LeafProve queues instead of spilling onto the idle
    /// dedicated worker.
    #[tokio::test]
    async fn test_dedicated_worker_rejects_leaf_work() {
        // max_leaf_provers = 1 so the single StarkOnly worker can hold only one
        // leaf; the second must queue (never land on the dedicated worker).
        let store = EdgeStateStore::new(1);
        store.init_proof("p-evm", make_workers_with_dedicated(1), 48);
        store.set_num_segments("p-evm", 0).await.unwrap();

        let first = store
            .enqueue_or_assign("p-evm", make_leaf_envelope("p-evm", 0, 3), Step::LeafProve)
            .await
            .unwrap()
            .expect("first leaf goes to the StarkOnly worker");
        assert_eq!(first.worker_id, 0, "leaf work goes to the StarkOnly worker");

        let second = store
            .enqueue_or_assign("p-evm", make_leaf_envelope("p-evm", 4, 7), Step::LeafProve)
            .await
            .unwrap();
        assert!(
            second.is_none(),
            "second leaf must queue, not spill onto the idle EvmDedicated worker"
        );

        // The dedicated worker (id 1) stays idle — it owns no app/leaf/internal.
        let debug = store.proof_debug_state("p-evm").await.unwrap();
        let dedicated = debug.workers.iter().find(|w| w.worker_id == 1).unwrap();
        assert_eq!(dedicated.active_proof_count, 0);
        assert!(dedicated.active_steps.is_empty());
    }

    /// Default (all-`Full`) invariance: `init_proof` over the full set with no
    /// dedicated worker seeds every worker with a ShardedAppProve slot and sets
    /// `num_workers == workers.len()`, exactly as before the EvmProve step.
    #[tokio::test]
    async fn test_init_proof_all_full_is_unchanged() {
        let store = EdgeStateStore::new(2);
        store.init_proof("p-full", make_test_workers(3), 48);

        let debug = store.proof_debug_state("p-full").await.unwrap();
        assert_eq!(
            debug.num_workers, 3,
            "num_workers == full count when no dedicated"
        );
        assert_eq!(debug.workers.len(), 3);
        for w in &debug.workers {
            assert_eq!(
                w.active_proof_count, 1,
                "every Full worker starts ShardedAppProve"
            );
            assert_eq!(
                w.active_steps,
                vec![Step::ShardedAppProve.as_str().to_string()]
            );
        }
    }

    /// Unified path on an all-`Full` fleet: a ready-for-evm `EvmProve` DOES
    /// dispatch (to a `Full` worker), rather than being rejected as it was when
    /// only the dedicated worker could take it. The ShardedAppProve slots are
    /// cleared first (0 segments) so the workers are idle, mirroring the live
    /// flow where the EVM step fires after the recursion tree drains.
    #[tokio::test]
    async fn test_evm_prove_dispatched_to_full_worker() {
        let store = EdgeStateStore::new(2);
        store.init_proof("p-full-evm", make_test_workers(2), 48);
        store.set_num_segments("p-full-evm", 0).await.unwrap();

        let assigned = store
            .enqueue_or_assign(
                "p-full-evm",
                make_evm_prove_envelope("p-full-evm"),
                Step::EvmProve,
            )
            .await
            .unwrap()
            .expect("EvmProve should dispatch to a Full worker on an all-Full fleet");
        assert_eq!(assigned.step, Step::EvmProve);
        // Any Full worker is eligible; the first idle one is picked.
        assert!(
            assigned.worker_id < 2,
            "EvmProve lands on one of the Full workers"
        );
    }

    /// Per-role accept rule for the EVM step: a `Full` worker accepts `EvmProve`
    /// only when idle, a `StarkOnly` worker never does (no root/halo2 provers),
    /// and an `EvmDedicated` worker accepts it only when idle. Also pins the
    /// isolation invariant: a `Full` worker rejects `EvmProve` while any other
    /// work is active, and rejects other work while an `EvmProve` is active.
    #[test]
    fn can_accept_work_evm_prove_by_role() {
        let mk = |role: WorkerRole, active: Vec<ActiveProof>| EdgeWorkerState {
            id: 0,
            worker_url: "http://w:8001".to_string(),
            worker_role: role,
            active_proof_count: active.len(),
            active_proofs: active,
            completed_segments: HashSet::new(),
        };

        // StarkOnly never accepts EvmProve.
        assert!(!can_accept_work(
            &mk(WorkerRole::StarkOnly, vec![]),
            Step::EvmProve,
            2
        ));

        // Full accepts EvmProve when idle.
        assert!(can_accept_work(
            &mk(WorkerRole::Full, vec![]),
            Step::EvmProve,
            2
        ));

        // Isolation: a `Full` worker REJECTS EvmProve while a leaf/internal is
        // active — halo2 never overlaps other proving on the same worker.
        assert!(!can_accept_work(
            &mk(
                WorkerRole::Full,
                vec![ActiveProof {
                    step: Step::InternalProve,
                    layer_idx: Some(1),
                }],
            ),
            Step::EvmProve,
            2
        ));

        // Isolation (reverse): a `Full` worker running an EvmProve REJECTS other
        // work — a leaf would otherwise fit its `< max_leaf_provers` slot.
        assert!(!can_accept_work(
            &mk(
                WorkerRole::Full,
                vec![ActiveProof {
                    step: Step::EvmProve,
                    layer_idx: None,
                }],
            ),
            Step::LeafProve,
            2
        ));

        // ...and rejects a second EvmProve (only one at a time).
        assert!(!can_accept_work(
            &mk(
                WorkerRole::Full,
                vec![ActiveProof {
                    step: Step::EvmProve,
                    layer_idx: None,
                }],
            ),
            Step::EvmProve,
            2
        ));

        // EvmDedicated accepts EvmProve only when idle (single slot).
        assert!(can_accept_work(
            &mk(WorkerRole::EvmDedicated, vec![]),
            Step::EvmProve,
            2
        ));
        assert!(!can_accept_work(
            &mk(
                WorkerRole::EvmDedicated,
                vec![ActiveProof {
                    step: Step::EvmProve,
                    layer_idx: None,
                }],
            ),
            Step::EvmProve,
            2
        ));
    }
}
