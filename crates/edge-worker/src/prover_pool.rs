//! Thread-pinned prover pool for GPU proving.
//!
//! Each prover is created on its own OS thread and lives for the worker lifetime.
//! Provers are !Send + !Sync because they hold GPU state.
//!
//! # Architecture
//!
//! The pool creates typed worker threads that each own a prover instance:
//! - App workers own an `AppProverInstance` (VM + interpreter)
//! - Leaf workers own a `LeafProver`
//! - Internal workers own an `InternalProverInstance` (multiple proving keys)
//!
//! Job functions receive references to these prover instances rather than
//! creating their own, ensuring provers are reused across jobs.

use crossbeam::channel::{bounded, Receiver, Sender};
use eyre::Result;
#[cfg(not(feature = "mock-provers"))]
use once_cell::sync::Lazy;
use protocol::{ProgramRef, WorkerRole};
#[cfg(not(feature = "mock-provers"))]
use std::collections::BTreeMap;
#[cfg(not(feature = "mock-provers"))]
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
#[cfg(not(feature = "mock-provers"))]
use std::sync::Mutex;
use std::time::Duration;
use thread_priority::{set_current_thread_priority, ThreadPriority, ThreadPriorityValue};
use tokio::sync::{oneshot, Notify};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::config::ProversConfig;
#[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
use crate::provers::{Halo2ProverJob, RootProverJob};
use crate::provers::{InternalProverJob, LeafProverJob, ProverResult, ShardedAppProverJob};

// Re-export prover instance types for use by worker threads
#[cfg(not(feature = "mock-provers"))]
pub use crate::provers::{
    AppExecutionInstances, InternalProverInstance, LeafProverInstance, ProverType,
};
#[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
pub use crate::provers::{Halo2ProverInstance, RootProverInstance};

/// Shared context that every app worker reads from to lazily load a
/// program's GPU prover on first use.
///
/// Built from the deployment's loadout and held via `Arc<>` so all N app
/// worker threads can look up `execution_instances[&ctx.program]` without
/// contention.
#[cfg(not(feature = "mock-provers"))]
pub struct AppWorkerContext {
    pub app_pk: Arc<sdk_v2::keygen::AppProvingKey<openvm_sdk_config::SdkVmConfig>>,
    /// One entry per loaded program, built via `AppExecutionInstances::new`
    /// (the expensive ~115 s AOT compile) at boot or at registration.
    pub execution_instances: HashMap<ProgramRef, Arc<AppExecutionInstances>>,
}

/// Job types for the prover pool — app workers only handle app jobs.
///
/// Every variant carries the `target_program` separately from any inner
/// `ShardedAppProverJob` so the worker thread can lazily ensure its GPU
/// `ProverType` is loaded for the right program before invoking the
/// inner closure.
pub enum AppProverJob {
    ShardedApp {
        target_program: ProgramRef,
        job: Box<ShardedAppProverJob>,
    },
    /// Coordinator for parallel proving — runs executor + acts as consumer-0 + collects results.
    #[cfg(not(feature = "mock-provers"))]
    ParallelCoordinator {
        target_program: ProgramRef,
        f: crate::provers::ParallelCoordinatorFn,
    },
    /// Segment consumer for parallel proving — proves segments from shared channel.
    #[cfg(not(feature = "mock-provers"))]
    SegmentConsumer {
        target_program: ProgramRef,
        f: crate::provers::SegmentConsumerFn,
    },
    /// Builds the worker's GPU prover for a just registered program ahead of
    /// its first job. Dispatched to every idle worker once the program's
    /// artifacts are installed.
    #[cfg(not(feature = "mock-provers"))]
    Preload { target_program: ProgramRef },
}

impl AppProverJob {
    #[cfg(not(feature = "mock-provers"))]
    pub fn target_program(&self) -> &ProgramRef {
        match self {
            AppProverJob::ShardedApp { target_program, .. }
            | AppProverJob::ParallelCoordinator { target_program, .. }
            | AppProverJob::SegmentConsumer { target_program, .. }
            | AppProverJob::Preload { target_program } => target_program,
        }
    }
}

/// Job types for leaf workers.
pub enum LeafProverJobWrapper {
    Leaf(LeafProverJob),
}

/// Job types for internal workers.
pub enum InternalProverJobWrapper {
    Internal(InternalProverJob),
}

/// Job types for root workers (in-process EVM prove; not network-dispatched).
#[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
pub enum RootProverJobWrapper {
    Root(RootProverJob),
}

/// Job types for halo2 workers (in-process EVM prove; not network-dispatched).
#[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
pub enum Halo2ProverJobWrapper {
    Halo2(Halo2ProverJob),
}

/// Generic worker handle: a job channel plus the worker's liveness flags.
///
/// One type for all worker kinds. `J` is the job the worker accepts and `O`
/// is the result it sends back over the per-job oneshot. App workers use
/// `WorkerHandle<AppProverJob>`, root uses
/// `WorkerHandle<RootProverJobWrapper, Result<protocol::RootProofState>>`,
/// and the others use `WorkerHandle<XJobWrapper>` (defaulting `O` to
/// [`ProverResult`]).
struct WorkerHandle<J, O = ProverResult> {
    job_sender: Sender<(J, oneshot::Sender<O>)>,
    cancel_sender: Sender<()>,
    is_busy: Arc<AtomicBool>,
    /// Set to true once the prover is fully initialized and ready to accept jobs.
    is_initialized: Arc<AtomicBool>,
    #[allow(dead_code)]
    join_handle: std::thread::JoinHandle<()>,
}

impl<J, O> WorkerHandle<J, O> {
    fn cancel(&self) {
        let _ = self.cancel_sender.try_send(());
    }
}

/// Describes a "uniform" worker kind (leaf, internal, root, halo2): one that
/// builds a single prover instance at boot and then runs jobs against it in
/// the shared [`ProverPool::worker_loop`]. App is deliberately *not* a
/// `WorkerKind` — its loop body does lazy per-program GPU swapping and stays
/// bespoke.
///
/// `init` and `run` execute *inside* the worker thread, so the `!Send`
/// `Instance` never crosses a thread boundary.
trait WorkerKind {
    type Job: Send + 'static;
    type Output: Send + 'static;
    type Instance;
    const NAME_PREFIX: &'static str;

    /// Build the prover instance. Runs once, in the worker thread.
    fn init() -> eyre::Result<Self::Instance>;

    /// Run a single job against the worker's instance.
    fn run(name: &str, inst: &Self::Instance, job: Self::Job) -> Self::Output;

    /// Convert a caught panic into this kind's output type.
    fn on_panic(name: &str, info: Box<dyn std::any::Any + Send>) -> Self::Output;
}

/// Marker type for the leaf worker kind.
struct LeafKind;
/// Marker type for the internal worker kind.
struct InternalKind;
/// Marker type for the root worker kind (in-process EVM prove).
#[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
struct RootKind;
/// Marker type for the halo2 worker kind (in-process EVM prove).
#[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
struct Halo2Kind;

#[cfg(feature = "mock-provers")]
impl WorkerKind for LeafKind {
    type Job = LeafProverJobWrapper;
    type Output = ProverResult;
    type Instance = ();
    const NAME_PREFIX: &'static str = "leaf-prover";

    fn init() -> eyre::Result<()> {
        Ok(())
    }

    fn run(_name: &str, _inst: &(), job: LeafProverJobWrapper) -> ProverResult {
        match job {
            LeafProverJobWrapper::Leaf(j) => crate::provers::prove_leaf(j),
        }
    }

    fn on_panic(name: &str, info: Box<dyn std::any::Any + Send>) -> ProverResult {
        ProverPool::panic_to_prover_result(name, info)
    }
}

#[cfg(not(feature = "mock-provers"))]
impl WorkerKind for LeafKind {
    type Job = LeafProverJobWrapper;
    type Output = ProverResult;
    type Instance = LeafProverInstance;
    const NAME_PREFIX: &'static str = "leaf-prover";

    fn init() -> eyre::Result<LeafProverInstance> {
        LeafProverInstance::new()
    }

    fn run(name: &str, inst: &LeafProverInstance, job: LeafProverJobWrapper) -> ProverResult {
        match job {
            LeafProverJobWrapper::Leaf(j) => {
                let proof_uuid = j.context.proof_uuid.clone();
                let segment_start = j.segment_start;
                let segment_end = j.segment_end;
                let active_scope =
                    ActiveLeafJobScope::new(name, &proof_uuid, segment_start, segment_end);

                // Catch the panic here, *inside* the live `active_scope`, so the
                // snapshot below still includes this job. The generic
                // `worker_loop` also catches, but by the time it runs
                // `active_scope` is already dropped (this job removed from the
                // registry) and the snapshot would miss the job that panicked.
                let raw = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::provers::prove_leaf_with_prover(j, inst)
                }));
                let mut result = ProverPool::handle_panic_result(name, raw);
                if let ProverResult::Error(err) = &mut result {
                    let active_leaf_jobs = active_scope.snapshot_summary();
                    error!(
                        worker = name,
                        proof_uuid = %proof_uuid,
                        segment_start,
                        segment_end,
                        active_leaf_jobs = %active_leaf_jobs,
                        "Leaf worker failure with active leaf snapshot"
                    );
                    err.push_str(&format!(" [active_leaf_jobs={}]", active_leaf_jobs));
                }
                result
            }
        }
    }

    fn on_panic(name: &str, info: Box<dyn std::any::Any + Send>) -> ProverResult {
        ProverPool::panic_to_prover_result(name, info)
    }
}

#[cfg(feature = "mock-provers")]
impl WorkerKind for InternalKind {
    type Job = InternalProverJobWrapper;
    type Output = ProverResult;
    type Instance = ();
    const NAME_PREFIX: &'static str = "internal-prover";

    fn init() -> eyre::Result<()> {
        Ok(())
    }

    fn run(_name: &str, _inst: &(), job: InternalProverJobWrapper) -> ProverResult {
        match job {
            InternalProverJobWrapper::Internal(j) => crate::provers::prove_internal(j),
        }
    }

    fn on_panic(name: &str, info: Box<dyn std::any::Any + Send>) -> ProverResult {
        ProverPool::panic_to_prover_result(name, info)
    }
}

#[cfg(not(feature = "mock-provers"))]
impl WorkerKind for InternalKind {
    type Job = InternalProverJobWrapper;
    type Output = ProverResult;
    type Instance = InternalProverInstance;
    const NAME_PREFIX: &'static str = "internal-prover";

    fn init() -> eyre::Result<InternalProverInstance> {
        InternalProverInstance::new()
    }

    fn run(
        _name: &str,
        inst: &InternalProverInstance,
        job: InternalProverJobWrapper,
    ) -> ProverResult {
        match job {
            InternalProverJobWrapper::Internal(j) => {
                crate::provers::prove_internal_with_prover(j, inst)
            }
        }
    }

    fn on_panic(name: &str, info: Box<dyn std::any::Any + Send>) -> ProverResult {
        ProverPool::panic_to_prover_result(name, info)
    }
}

#[cfg(feature = "mock-provers")]
impl WorkerKind for RootKind {
    type Job = RootProverJobWrapper;
    type Output = Result<protocol::RootProofState>;
    type Instance = ();
    const NAME_PREFIX: &'static str = "root-prover";

    fn init() -> eyre::Result<()> {
        Ok(())
    }

    fn run(_name: &str, _inst: &(), job: RootProverJobWrapper) -> Result<protocol::RootProofState> {
        match job {
            RootProverJobWrapper::Root(j) => crate::provers::prove_root(j),
        }
    }

    fn on_panic(
        name: &str,
        info: Box<dyn std::any::Any + Send>,
    ) -> Result<protocol::RootProofState> {
        ProverPool::panic_to_root_result(name, info)
    }
}

#[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
impl WorkerKind for RootKind {
    type Job = RootProverJobWrapper;
    type Output = Result<protocol::RootProofState>;
    type Instance = RootProverInstance;
    const NAME_PREFIX: &'static str = "root-prover";

    fn init() -> eyre::Result<RootProverInstance> {
        RootProverInstance::new()
    }

    fn run(
        _name: &str,
        inst: &RootProverInstance,
        job: RootProverJobWrapper,
    ) -> Result<protocol::RootProofState> {
        match job {
            RootProverJobWrapper::Root(j) => crate::provers::prove_root_with_prover(j, inst),
        }
    }

    fn on_panic(
        name: &str,
        info: Box<dyn std::any::Any + Send>,
    ) -> Result<protocol::RootProofState> {
        ProverPool::panic_to_root_result(name, info)
    }
}

#[cfg(feature = "mock-provers")]
impl WorkerKind for Halo2Kind {
    type Job = Halo2ProverJobWrapper;
    type Output = ProverResult;
    type Instance = ();
    const NAME_PREFIX: &'static str = "halo2-prover";

    fn init() -> eyre::Result<()> {
        Ok(())
    }

    fn run(_name: &str, _inst: &(), job: Halo2ProverJobWrapper) -> ProverResult {
        match job {
            Halo2ProverJobWrapper::Halo2(j) => crate::provers::prove_halo2(j),
        }
    }

    fn on_panic(name: &str, info: Box<dyn std::any::Any + Send>) -> ProverResult {
        ProverPool::panic_to_prover_result(name, info)
    }
}

#[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
impl WorkerKind for Halo2Kind {
    type Job = Halo2ProverJobWrapper;
    type Output = ProverResult;
    type Instance = Halo2ProverInstance;
    const NAME_PREFIX: &'static str = "halo2-prover";

    fn init() -> eyre::Result<Halo2ProverInstance> {
        Halo2ProverInstance::new()
    }

    fn run(_name: &str, inst: &Halo2ProverInstance, job: Halo2ProverJobWrapper) -> ProverResult {
        match job {
            Halo2ProverJobWrapper::Halo2(j) => crate::provers::prove_halo2_with_prover(j, inst),
        }
    }

    fn on_panic(name: &str, info: Box<dyn std::any::Any + Send>) -> ProverResult {
        ProverPool::panic_to_prover_result(name, info)
    }
}

#[cfg(not(feature = "mock-provers"))]
#[derive(Clone, Debug)]
struct ActiveLeafJobDebug {
    proof_uuid: String,
    segment_start: usize,
    segment_end: usize,
}

#[cfg(not(feature = "mock-provers"))]
static ACTIVE_LEAF_JOBS: Lazy<Mutex<BTreeMap<String, ActiveLeafJobDebug>>> =
    Lazy::new(|| Mutex::new(BTreeMap::new()));

/// Process-wide app job and GPU program-swap counters exposed via `/healthz`.
/// Their ratio estimates the maximum benefit of program-affinity dispatch.
static APP_JOBS_TOTAL: AtomicU64 = AtomicU64::new(0);
static APP_SWAPS_TOTAL: AtomicU64 = AtomicU64::new(0);

#[cfg(not(feature = "mock-provers"))]
struct ActiveLeafJobScope {
    worker_name: String,
}

#[cfg(not(feature = "mock-provers"))]
impl ActiveLeafJobScope {
    fn new(worker_name: &str, proof_uuid: &str, segment_start: usize, segment_end: usize) -> Self {
        let mut jobs = ACTIVE_LEAF_JOBS
            .lock()
            .expect("active leaf debug registry poisoned");
        jobs.insert(
            worker_name.to_string(),
            ActiveLeafJobDebug {
                proof_uuid: proof_uuid.to_string(),
                segment_start,
                segment_end,
            },
        );
        Self {
            worker_name: worker_name.to_string(),
        }
    }

    fn snapshot_summary(&self) -> String {
        let jobs = ACTIVE_LEAF_JOBS
            .lock()
            .expect("active leaf debug registry poisoned");
        let mut entries = Vec::with_capacity(jobs.len());
        for (worker_name, job) in jobs.iter() {
            entries.push(format!(
                "{}:{}[{}-{}]",
                worker_name, job.proof_uuid, job.segment_start, job.segment_end
            ));
        }
        if entries.is_empty() {
            "none".to_string()
        } else {
            entries.join(", ")
        }
    }
}

#[cfg(not(feature = "mock-provers"))]
impl Drop for ActiveLeafJobScope {
    fn drop(&mut self) {
        let mut jobs = ACTIVE_LEAF_JOBS
            .lock()
            .expect("active leaf debug registry poisoned");
        jobs.remove(&self.worker_name);
    }
}

/// Thread-pinned prover pool.
///
/// Manages worker threads that each own exactly one prover instance.
/// Jobs are dispatched to available workers.
///
/// **Multi-ELF (swap design):** the app pool is a flat `Vec<AppWorkerHandle>`
/// of size `max_app_provers`. Each app worker holds an
/// `Option<(ProgramRef, ProverType)>` and lazily (re)builds the GPU
/// prover when a job for a different program arrives. At idle, all
/// workers' option is `None` → zero GPU memory. Leaf and internal pools
/// are program-agnostic (they verify STARK proofs by VK) so they stay
/// as flat vectors with eagerly-built provers.
pub struct ProverPool {
    /// Flat pool of `max_app_provers` workers. Workers self-load the
    /// right `ProverType` on every job; load-on-first-use, drop+rebuild
    /// only on program switch.
    app_workers: Vec<WorkerHandle<AppProverJob>>,
    leaf_workers: Vec<WorkerHandle<LeafProverJobWrapper>>,
    internal_workers: Vec<WorkerHandle<InternalProverJobWrapper>>,
    /// Root prover workers. Drive the in-process EVM prove after a
    /// final internal proof of an Evm-typed proof. Empty in stark-only
    /// builds (no `evm-prove`/`mock-provers`).
    #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
    root_workers: Vec<WorkerHandle<RootProverJobWrapper, Result<protocol::RootProofState>>>,
    /// Halo2 prover workers. Same EVM prove flow as root_workers.
    #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
    halo2_workers: Vec<WorkerHandle<Halo2ProverJobWrapper>>,
    /// Signalled after any worker becomes available for another job.
    worker_free: Arc<Notify>,
    cancel_token: CancellationToken,
}

#[cfg(not(feature = "mock-provers"))]
fn merge_parallel_app_results(
    coordinator_result: ProverResult,
    consumer_proofs: Vec<protocol::ProofResult>,
    consumer_errors: Vec<String>,
    consumer_canceled: bool,
) -> ProverResult {
    match coordinator_result {
        // Any canceled part leaves the segment set incomplete, so the job has
        // nothing to report whatever the others did.
        ProverResult::Canceled => ProverResult::Canceled,
        _ if consumer_canceled => ProverResult::Canceled,
        ProverResult::Success(mut results) if consumer_errors.is_empty() => {
            results.extend(consumer_proofs);
            ProverResult::Success(results)
        }
        ProverResult::Success(_) => ProverResult::Error(format!(
            "Parallel app consumer failure(s): {}",
            consumer_errors.join("; ")
        )),
        ProverResult::Error(coordinator_error) if consumer_errors.is_empty() => {
            ProverResult::Error(coordinator_error)
        }
        ProverResult::Error(coordinator_error) => ProverResult::Error(format!(
            "Parallel app coordinator failed: {}; consumer failure(s): {}",
            coordinator_error,
            consumer_errors.join("; ")
        )),
    }
}

impl ProverPool {
    /// Create a new prover pool.
    ///
    /// Leaf and internal worker threads eagerly construct their (program-
    /// agnostic) `*ProverInstance` at boot. App worker threads start
    /// unloaded (`Option<(ProgramRef, ProverType)> = None`) and lazily
    /// load on first job that targets a particular program — they look
    /// up `AppExecutionInstances` from `app_ctx.execution_instances`,
    /// build a `ProverType` via `app_pk + exe`, and hold both until the
    /// next program switch.
    ///
    /// # Failure handling
    ///
    /// Prover construction happens *inside* each spawned worker thread, so a
    /// construction failure does **not** make this constructor return `Err` or
    /// panic — the affected thread logs the error and exits, leaving its slot
    /// permanently un-initialized (`is_initialized` stays `false`). The node
    /// then runs at reduced capacity, or, if every worker in a class dies, with
    /// that class unable to serve.
    ///
    /// This is an availability concern only — a missing worker never produces a
    /// wrong proof, it produces no result. Recovery is external, not in-process:
    /// the manager's per-proof timeout (`proof.timeout_secs`) fails any proof
    /// that wedges on a dead worker, and `/readyz` reports the node not-ready
    /// while any worker is un-initialized. Operators MUST alert on proof-failure
    /// rate and worker readiness, and restart a node whose prover threads die;
    /// nothing here restarts them automatically.
    /// `role` gates which prover classes are built (dedicated-halo2 mode):
    /// - `Full` (default): every class, at the configured sizes — today's
    ///   behavior, byte-for-byte (all `runs_*` predicates are true).
    /// - `StarkOnly`: app/leaf/internal only; **no** root/halo2 (skips the
    ///   ~10 GB halo2 key load), so those pools are empty even in an
    ///   `evm-prove`/`mock-provers` build.
    /// - `EvmDedicated`: root/halo2 only; **no** app/leaf/internal pools (its
    ///   GPU pool stays small). For this role `app_ctx` is `None` — the caller
    ///   skips the app-execution-context build entirely.
    pub fn new(
        config: &ProversConfig,
        role: WorkerRole,
        #[cfg(not(feature = "mock-provers"))] app_ctx: Option<Arc<AppWorkerContext>>,
    ) -> Result<Self> {
        let cancel_token = CancellationToken::new();
        let worker_free = Arc::new(Notify::new());

        // Role-gated pool sizes. A default `Full` worker sets every count to
        // the configured value, so the pool is unchanged from today.
        let runs_stark = role.runs_stark_proving();
        // Only consulted by the root/halo2 pools, which exist solely in an
        // `evm-prove`/`mock-provers` build; unused in a stark-only build.
        #[cfg_attr(
            not(any(feature = "evm-prove", feature = "mock-provers")),
            allow(unused_variables)
        )]
        let runs_evm = role.runs_evm_prove();
        let app_count = if runs_stark {
            config.max_app_provers
        } else {
            0
        };
        let leaf_count = if runs_stark {
            config.max_leaf_provers
        } else {
            0
        };
        let internal_count = if runs_stark {
            config.max_internal_provers
        } else {
            0
        };

        info!(
            "Creating prover pool (role={:?}): app={}, leaf={}, internal={}",
            role, app_count, leaf_count, internal_count
        );

        // App workers need the app-execution context. A disk-seeded worker
        // gets it from the caller (`Some`); a registration-driven one gets
        // `None` and its workers wait for the store to publish it.
        // `EvmDedicated` also passes `None`, but its `app_count` is 0 so no
        // worker is spawned to wait.
        #[cfg(not(feature = "mock-provers"))]
        let app_workers = Self::spawn_app_workers(app_count, app_ctx, worker_free.clone())?;
        #[cfg(feature = "mock-provers")]
        let app_workers = Self::spawn_app_workers(app_count, worker_free.clone())?;

        let leaf_workers = Self::spawn_workers::<LeafKind>(leaf_count, worker_free.clone())?;

        let internal_workers =
            Self::spawn_workers::<InternalKind>(internal_count, worker_free.clone())?;

        #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
        let root_workers = Self::spawn_workers::<RootKind>(
            if runs_evm { config.max_root_provers } else { 0 },
            worker_free.clone(),
        )?;

        #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
        let halo2_workers = Self::spawn_workers::<Halo2Kind>(
            if runs_evm {
                config.max_halo2_provers
            } else {
                0
            },
            worker_free.clone(),
        )?;

        Ok(Self {
            app_workers,
            leaf_workers,
            internal_workers,
            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
            root_workers,
            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
            halo2_workers,
            worker_free,
            cancel_token,
        })
    }

    /// Spawn `count` app prover worker threads. Each worker is
    /// program-agnostic at startup; it lazily loads + swaps a
    /// `ProverType` as jobs arrive (see `app_worker_loop`).
    fn spawn_app_workers(
        count: usize,
        #[cfg(not(feature = "mock-provers"))] app_ctx: Option<Arc<AppWorkerContext>>,
        worker_free: Arc<Notify>,
    ) -> Result<Vec<WorkerHandle<AppProverJob>>> {
        let mut workers = Vec::with_capacity(count);

        for i in 0..count {
            let (job_sender, job_receiver) =
                bounded::<(AppProverJob, oneshot::Sender<ProverResult>)>(1);
            let (cancel_sender, cancel_receiver) = bounded(1);
            let is_busy = Arc::new(AtomicBool::new(false));
            let is_busy_clone = is_busy.clone();
            let is_initialized = Arc::new(AtomicBool::new(false));
            let is_initialized_clone = is_initialized.clone();
            let worker_free_clone = worker_free.clone();
            let worker_name = format!("app-prover-{}", i);
            #[cfg(not(feature = "mock-provers"))]
            let ctx_for_worker = app_ctx.clone();

            let join_handle = std::thread::Builder::new()
                .name(worker_name.clone())
                .spawn(move || {
                    Self::app_worker_loop(
                        &worker_name,
                        #[cfg(not(feature = "mock-provers"))]
                        ctx_for_worker,
                        job_receiver,
                        is_busy_clone,
                        is_initialized_clone,
                        cancel_receiver,
                        worker_free_clone,
                    );
                })?;

            workers.push(WorkerHandle {
                job_sender,
                cancel_sender,
                is_busy,
                is_initialized,
                join_handle,
            });
        }

        Ok(workers)
    }

    /// Spawn `count` uniform worker threads for kind `W`.
    ///
    /// Each thread builds its own `W::Instance` via `W::init()` (so the
    /// `!Send` prover never crosses a thread boundary) and then runs the
    /// shared [`Self::worker_loop`]. Replaces the per-kind `spawn_*_workers`
    /// functions for leaf/internal/root/halo2.
    fn spawn_workers<W: WorkerKind>(
        count: usize,
        worker_free: Arc<Notify>,
    ) -> Result<Vec<WorkerHandle<W::Job, W::Output>>> {
        let mut workers = Vec::with_capacity(count);

        for i in 0..count {
            let (job_sender, job_receiver) = bounded::<(W::Job, oneshot::Sender<W::Output>)>(1);
            let (cancel_sender, cancel_receiver) = bounded(1);
            let is_busy = Arc::new(AtomicBool::new(false));
            let is_busy_clone = is_busy.clone();
            let is_initialized = Arc::new(AtomicBool::new(false));
            let is_initialized_clone = is_initialized.clone();
            let worker_free_clone = worker_free.clone();
            let worker_name = format!("{}-{}", W::NAME_PREFIX, i);

            let join_handle = std::thread::Builder::new()
                .name(worker_name.clone())
                .spawn(move || {
                    Self::worker_loop::<W>(
                        &worker_name,
                        job_receiver,
                        is_busy_clone,
                        is_initialized_clone,
                        cancel_receiver,
                        worker_free_clone,
                    );
                })?;

            workers.push(WorkerHandle {
                job_sender,
                cancel_sender,
                is_busy,
                is_initialized,
                join_handle,
            });
        }

        Ok(workers)
    }

    /// App worker loop (real mode) — program-agnostic, lazy load + swap.
    ///
    /// Holds `Option<(ProgramRef, ProverType)>` for whatever program is
    /// currently loaded (`None` at boot, i.e. 0 GPU memory). On each
    /// job: if the target program differs from what's loaded, drop the
    /// existing prover (frees ~1.66 GB GPU) and build a new one
    /// (~1 s); then look up the program's pre-built
    /// `AppExecutionInstances` from `app_ctx` and invoke the inner
    /// closure.
    #[cfg(not(feature = "mock-provers"))]
    fn app_worker_loop(
        name: &str,
        app_ctx: Option<Arc<AppWorkerContext>>,
        job_receiver: Receiver<(AppProverJob, oneshot::Sender<ProverResult>)>,
        is_busy: Arc<AtomicBool>,
        is_initialized: Arc<AtomicBool>,
        cancel_receiver: Receiver<()>,
        worker_free: Arc<Notify>,
    ) {
        Self::set_thread_priority(name);
        info!("Worker {} starting (unloaded — lazy program load)", name);

        // A registration-driven worker boots without a deployment, so park
        // here until one is published. `is_initialized` stays false until
        // then, which keeps /readyz false while the worker cannot prove.
        let mut app_ctx = match app_ctx {
            Some(ctx) => ctx,
            None => crate::artifacts::ArtifactStore::global()
                .expect("artifact store initialized before the prover pool")
                .wait_for_app_worker_context(),
        };

        // Worker is immediately "initialized" in the swap design — it
        // simply has no program loaded yet. /readyz gates on the
        // execution_instances being built, not on a per-worker prover.
        is_initialized.store(true, Ordering::SeqCst);
        worker_free.notify_waiters();

        let mut loaded: Option<(ProgramRef, ProverType)> = None;

        loop {
            crossbeam::channel::select! {
                recv(cancel_receiver) -> _ => {
                    info!("Worker {name} shutting down");
                    break;
                }
                recv(job_receiver) -> received => match received {
                    Ok((job, result_sender)) => {
                        is_busy.store(true, Ordering::SeqCst);

                        // Pick up a deployment extended after this thread
                        // started, so a program registered later is visible.
                        if !app_ctx.execution_instances.contains_key(job.target_program()) {
                            if let Some(extended) = crate::artifacts::ArtifactStore::global()
                                .and_then(|store| store.app_worker_context())
                            {
                                app_ctx = extended;
                            }
                        }

                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            Self::run_app_job(name, &app_ctx, &mut loaded, job)
                        }));

                        is_busy.store(false, Ordering::SeqCst);
                        worker_free.notify_waiters();

                        let result = Self::handle_panic_result(name, result);
                        if result_sender.send(result).is_err() {
                            warn!("Worker {name}: result receiver dropped");
                        }
                    }
                    Err(_) => {
                        info!("Worker {name} channel disconnected");
                        break;
                    }
                },
            }
        }
    }

    /// App worker loop — mock mode (no prover instance needed).
    #[cfg(feature = "mock-provers")]
    fn app_worker_loop(
        name: &str,
        job_receiver: Receiver<(AppProverJob, oneshot::Sender<ProverResult>)>,
        is_busy: Arc<AtomicBool>,
        is_initialized: Arc<AtomicBool>,
        cancel_receiver: Receiver<()>,
        worker_free: Arc<Notify>,
    ) {
        Self::set_thread_priority(name);
        info!("Worker {} started (mock mode)", name);

        // Mock mode is immediately initialized
        is_initialized.store(true, Ordering::SeqCst);
        worker_free.notify_waiters();

        loop {
            crossbeam::channel::select! {
                recv(cancel_receiver) -> _ => {
                    info!("Worker {name} shutting down");
                    break;
                }
                recv(job_receiver) -> received => match received {
                    Ok((job, result_sender)) => {
                        is_busy.store(true, Ordering::SeqCst);

                        let result =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match job {
                                AppProverJob::ShardedApp { job, .. } => {
                                    crate::provers::prove_sharded_app(*job)
                                }
                            }));

                        is_busy.store(false, Ordering::SeqCst);
                        worker_free.notify_waiters();

                        let result = Self::handle_panic_result(name, result);
                        if result_sender.send(result).is_err() {
                            warn!("Worker {name}: result receiver dropped");
                        }
                    }
                    Err(_) => {
                        info!("Worker {name} channel disconnected");
                        break;
                    }
                },
            }
        }
    }

    /// Handle a single app job: ensure the worker is loaded for the
    /// job's target program (swap if needed), then invoke the variant's
    /// inner closure with `(instances, prover)`.
    #[cfg(not(feature = "mock-provers"))]
    fn run_app_job(
        name: &str,
        app_ctx: &AppWorkerContext,
        loaded: &mut Option<(ProgramRef, ProverType)>,
        job: AppProverJob,
    ) -> ProverResult {
        let target = job.target_program().clone();
        APP_JOBS_TOTAL.fetch_add(1, Ordering::Relaxed);

        // Find the program's pre-built CPU artifacts.
        let instances = match app_ctx.execution_instances.get(&target).cloned() {
            Some(i) => i,
            None => {
                return ProverResult::Error(format!(
                    "Worker {name}: program {target} not in execution_instances; \
                     manager validation should have rejected this dispatch"
                ));
            }
        };

        // Swap GPU prover if loaded program differs from target.
        let previous_program = loaded.as_ref().map(|(program, _)| program.clone());
        let needs_swap = previous_program
            .as_ref()
            .map(|program| program != &target)
            .unwrap_or(true);
        if needs_swap {
            if previous_program.is_some() {
                APP_SWAPS_TOTAL.fetch_add(1, Ordering::Relaxed);
            }
            // Drop existing first so its 1.66 GB frees before we
            // allocate the next one.
            *loaded = None;
            match previous_program {
                Some(from) => info!(
                    worker = name,
                    from = %from,
                    to = %target,
                    "swapping GPU prover"
                ),
                None => info!(worker = name, to = %target, "loading GPU prover"),
            }
            let prover = match crate::provers::build_gpu_prover(
                &target,
                &app_ctx.app_pk,
                instances.exe.clone(),
            ) {
                Ok(p) => p,
                Err(e) => {
                    return ProverResult::Error(format!(
                        "Worker {name}: failed to build GPU prover for {target}: {e}"
                    ));
                }
            };
            *loaded = Some((target.clone(), prover));
        }

        let prover = &mut loaded.as_mut().expect("just-loaded above").1;

        match job {
            AppProverJob::ShardedApp { job, .. } => {
                crate::provers::prove_sharded_app_with_prover(*job, &instances, prover)
            }
            AppProverJob::ParallelCoordinator { f, .. } => f(&instances, prover),
            AppProverJob::SegmentConsumer { f, .. } => f(&instances, prover),
            // The swap above is the whole job.
            AppProverJob::Preload { .. } => ProverResult::Success(vec![]),
        }
    }

    /// The single job loop for all uniform worker kinds.
    ///
    /// Builds the kind's prover instance via `W::init()` (in this thread),
    /// then services jobs until cancelled or the channel disconnects. On
    /// init failure it logs and returns — the thread exits and
    /// `is_initialized` stays `false`. Replaces the per-kind
    /// `*_worker_loop` / `run_*_job_loop` functions for
    /// leaf/internal/root/halo2.
    fn worker_loop<W: WorkerKind>(
        name: &str,
        job_receiver: Receiver<(W::Job, oneshot::Sender<W::Output>)>,
        is_busy: Arc<AtomicBool>,
        is_initialized: Arc<AtomicBool>,
        cancel_receiver: Receiver<()>,
        worker_free: Arc<Notify>,
    ) {
        Self::set_thread_priority(name);
        info!("Worker {} starting, initializing prover...", name);

        // Build the prover instance once at worker startup. On failure the
        // thread exits with `is_initialized` left false.
        let instance = match W::init() {
            Ok(inst) => {
                info!("Worker {} prover initialized successfully", name);
                inst
            }
            Err(e) => {
                error!("Worker {} failed to create prover: {}", name, e);
                return;
            }
        };

        // Mark as initialized before entering the job loop.
        is_initialized.store(true, Ordering::SeqCst);
        worker_free.notify_waiters();

        loop {
            crossbeam::channel::select! {
                recv(cancel_receiver) -> _ => {
                    info!("Worker {name} shutting down");
                    break;
                }
                recv(job_receiver) -> received => match received {
                    Ok((job, result_sender)) => {
                        is_busy.store(true, Ordering::SeqCst);

                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            W::run(name, &instance, job)
                        }));

                        is_busy.store(false, Ordering::SeqCst);
                        worker_free.notify_waiters();

                        let result = match result {
                            Ok(r) => r,
                            Err(info) => W::on_panic(name, info),
                        };
                        if result_sender.send(result).is_err() {
                            warn!("Worker {name}: result receiver dropped");
                        }
                    }
                    Err(_) => {
                        info!("Worker {name} channel disconnected");
                        break;
                    }
                },
            }
        }
    }

    /// Set thread priority for a worker.
    fn set_thread_priority(name: &str) {
        if let Err(e) = set_current_thread_priority(ThreadPriority::Crossplatform(
            ThreadPriorityValue::try_from(80).unwrap_or(ThreadPriorityValue::MAX),
        )) {
            warn!("Failed to set thread priority for {}: {}", name, e);
        }
    }

    /// Handle panic result from catch_unwind.
    fn panic_message(panic_info: Box<dyn std::any::Any + Send>) -> String {
        if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        }
    }

    /// Handle panic result from catch_unwind for the app loop, whose
    /// `catch_unwind` produces `Result<ProverResult, _>` directly.
    fn handle_panic_result(
        name: &str,
        result: std::result::Result<ProverResult, Box<dyn std::any::Any + Send>>,
    ) -> ProverResult {
        match result {
            Ok(r) => r,
            Err(panic_info) => Self::panic_to_prover_result(name, panic_info),
        }
    }

    /// Convert a caught panic into a [`ProverResult::Error`]. Used by the
    /// generic [`WorkerKind::on_panic`] for the `ProverResult`-typed kinds.
    fn panic_to_prover_result(
        name: &str,
        panic_info: Box<dyn std::any::Any + Send>,
    ) -> ProverResult {
        let panic_msg = Self::panic_message(panic_info);
        error!("Worker {} panicked: {}", name, panic_msg);
        ProverResult::Error(format!("Prover panicked: {}", panic_msg))
    }

    /// Convert a caught panic into an `Err`, for the root channel which
    /// carries `Result<RootProofState>` rather than [`ProverResult`]. Used
    /// by [`RootKind`]'s [`WorkerKind::on_panic`].
    #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
    fn panic_to_root_result(
        name: &str,
        panic_info: Box<dyn std::any::Any + Send>,
    ) -> Result<protocol::RootProofState> {
        let panic_msg = Self::panic_message(panic_info);
        error!("Worker {} panicked: {}", name, panic_msg);
        Err(eyre::eyre!("Prover panicked: {}", panic_msg))
    }

    /// Submit an sharded app proving job.
    ///
    /// Workers are program-agnostic in the swap design — any free
    /// worker can take the job and will self-load the right
    /// `ProverType` if it doesn't already match `job.context.program`.
    ///
    /// If `job.max_app_provers > 1` and we're in real mode, dispatches a coordinator
    /// to one app worker and N-1 segment consumers to other app workers. All workers
    /// independently swap-and-load to the target program on first job; subsequent
    /// jobs for the same program are zero-cost.
    pub async fn submit_sharded_app_job(&self, job: ShardedAppProverJob) -> Result<ProverResult> {
        #[cfg(not(feature = "mock-provers"))]
        if job.max_app_provers > 1 {
            return self.submit_parallel_app_job(job).await;
        }

        #[cfg(feature = "mock-provers")]
        if job.max_app_provers > 1 {
            warn!(
                "max_app_provers={} ignored in mock mode, using 1",
                job.max_app_provers
            );
        }

        let target_program = job.context.program.clone();
        self.submit_to_workers(
            "app",
            &self.app_workers,
            AppProverJob::ShardedApp {
                target_program,
                job: Box::new(job),
            },
        )
        .await
    }

    /// Submit a parallel sharded app proving job (real mode only).
    #[cfg(not(feature = "mock-provers"))]
    async fn submit_parallel_app_job(&self, job: ShardedAppProverJob) -> Result<ProverResult> {
        let max_app_provers = job.max_app_provers;
        let consumer_count = max_app_provers.saturating_sub(1);
        let program = job.context.program.clone();

        info!(
            "Submitting parallel app job for {}: max_app_provers={}",
            program, max_app_provers
        );

        // Check we have enough free workers in the flat pool. (Workers
        // self-load on dispatch — no per-program slot lookup needed.)
        let available = self
            .app_workers
            .iter()
            .filter(|w| {
                !w.is_busy.load(Ordering::SeqCst) && w.is_initialized.load(Ordering::SeqCst)
            })
            .count();

        if available < max_app_provers {
            return Err(eyre::eyre!(
                "Not enough available app workers for parallel proving of {}: need {}, have {}",
                program,
                max_app_provers,
                available
            ));
        }

        // Create coordinator and consumer closures
        let (coordinator_fn, consumer_fns) = crate::provers::create_parallel_prove_jobs(job)?;

        // Dispatch N-1 consumers first (they'll block on prove_rx until executor starts).
        // Retain every result receiver so the parallel job cannot report success until
        // all consumers have completed successfully.
        let mut consumer_receivers = Vec::with_capacity(consumer_count);
        for consumer_fn in consumer_fns {
            consumer_receivers.push(
                self.dispatch_app_job(AppProverJob::SegmentConsumer {
                    target_program: program.clone(),
                    f: consumer_fn,
                })
                .await?,
            );
        }

        // Ensure consumer workers are actually running before coordinator dispatch.
        // Without this barrier, coordinator can race onto the same worker queue as a
        // consumer and never execute, stalling the proof.
        self.wait_for_busy_app_workers(consumer_count, Duration::from_secs(2))
            .await?;

        // The coordinator must run before awaiting consumers: consumers block on
        // segments produced by the coordinator's executor. Once it returns, await
        // every consumer and merge non-streamed proofs into the overall result.
        let coordinator_result = self
            .submit_to_workers(
                "app",
                &self.app_workers,
                AppProverJob::ParallelCoordinator {
                    target_program: program,
                    f: coordinator_fn,
                },
            )
            .await?;

        let mut consumer_proofs = Vec::new();
        let mut consumer_errors = Vec::new();
        let mut consumer_canceled = false;
        for receiver in consumer_receivers {
            match receiver.await {
                Ok(ProverResult::Success(results)) => consumer_proofs.extend(results),
                Ok(ProverResult::Error(error)) => consumer_errors.push(error),
                Ok(ProverResult::Canceled) => consumer_canceled = true,
                Err(_) => consumer_errors.push("Consumer dropped result channel".to_string()),
            }
        }

        Ok(merge_parallel_app_results(
            coordinator_result,
            consumer_proofs,
            consumer_errors,
            consumer_canceled,
        ))
    }

    #[cfg(not(feature = "mock-provers"))]
    async fn wait_for_busy_app_workers(&self, min_busy: usize, timeout: Duration) -> Result<()> {
        if min_busy == 0 {
            return Ok(());
        }

        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let busy = self.busy_count(JobType::ShardedApp);
            if busy >= min_busy {
                return Ok(());
            }

            if tokio::time::Instant::now() >= deadline {
                return Err(eyre::eyre!(
                    "Timed out waiting for app consumers to start: need busy >= {}, got {}",
                    min_busy,
                    busy
                ));
            }

            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Dispatch an app job and return its result receiver without awaiting it.
    #[cfg(not(feature = "mock-provers"))]
    async fn dispatch_app_job(&self, job: AppProverJob) -> Result<oneshot::Receiver<ProverResult>> {
        let mut job_opt = Some(job);

        loop {
            // Register before scanning so a worker becoming free during the
            // scan cannot leave this dispatcher asleep until the backstop.
            let worker_free = self.worker_free.notified();
            tokio::pin!(worker_free);
            worker_free.as_mut().enable();

            for (idx, worker) in self.app_workers.iter().enumerate() {
                if !worker.is_busy.load(Ordering::SeqCst)
                    && worker.is_initialized.load(Ordering::SeqCst)
                {
                    if let Some(job) = job_opt.take() {
                        let (result_sender, result_receiver) = oneshot::channel();
                        match worker.job_sender.try_send((job, result_sender)) {
                            Ok(()) => {
                                info!("Dispatched app job to slot={}", idx);
                                return Ok(result_receiver);
                            }
                            Err(crossbeam::channel::TrySendError::Full(returned)) => {
                                job_opt = Some(returned.0);
                            }
                            Err(crossbeam::channel::TrySendError::Disconnected(_)) => {
                                return Err(eyre::eyre!("Worker channel closed"));
                            }
                        }
                    }
                }
            }

            if self.cancel_token.is_cancelled() {
                return Err(eyre::eyre!("Prover pool cancelled"));
            }

            tokio::select! {
                _ = &mut worker_free => {}
                _ = self.cancel_token.cancelled() => {}
                _ = tokio::time::sleep(Duration::from_secs(1)) => {}
            }
        }
    }

    /// Build the GPU prover for `program` on every idle app worker, so the
    /// first job dispatched after its publish starts proving immediately.
    ///
    /// A busy worker is skipped and swaps lazily when its next job arrives.
    /// A worker still parked waiting for its first deployment queues the job
    /// and runs it as soon as the install wakes it, so eligibility ignores
    /// `is_initialized`. A build failure is logged and left for the lazy
    /// path to retry at job time.
    #[cfg(not(feature = "mock-provers"))]
    pub async fn preload_app_provers(&self, program: &ProgramRef) {
        let mut result_receivers = Vec::with_capacity(self.app_workers.len());
        for worker in &self.app_workers {
            if worker.is_busy.load(Ordering::SeqCst) {
                continue;
            }
            let (result_sender, result_receiver) = oneshot::channel();
            let job = AppProverJob::Preload {
                target_program: program.clone(),
            };
            if worker.job_sender.try_send((job, result_sender)).is_ok() {
                result_receivers.push(result_receiver);
            }
        }
        for result_receiver in result_receivers {
            match result_receiver.await {
                Ok(ProverResult::Error(error)) => {
                    warn!("Failed to preload a GPU prover for {program}: {error}")
                }
                Ok(_) => {}
                Err(_) => warn!("Preload of {program} dropped its result channel"),
            }
        }
    }

    /// Submit a leaf proving job.
    pub async fn submit_leaf_job(&self, job: LeafProverJob) -> Result<ProverResult> {
        self.submit_to_workers("leaf", &self.leaf_workers, LeafProverJobWrapper::Leaf(job))
            .await
    }

    /// Submit an internal proving job.
    pub async fn submit_internal_job(&self, job: InternalProverJob) -> Result<ProverResult> {
        self.submit_to_workers(
            "internal",
            &self.internal_workers,
            InternalProverJobWrapper::Internal(job),
        )
        .await
    }

    /// Submit a root proving job (in-process EVM prove).
    ///
    /// Called locally by the handler after the worker reports the final
    /// `Internal` result of an Evm-typed proof. Not exposed over the wire.
    #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
    pub async fn submit_root_job(&self, job: RootProverJob) -> Result<protocol::RootProofState> {
        // `submit_to_workers` yields `Result<Self::Output>` where the root
        // kind's `Output` is itself `Result<RootProofState>`; the `?`
        // flattens the dispatch-error layer, matching the old behavior.
        self.submit_to_workers("root", &self.root_workers, RootProverJobWrapper::Root(job))
            .await?
    }

    /// Submit a halo2 proving job (in-process EVM prove).
    ///
    /// Called locally by the handler after a successful root prove. Not
    /// exposed over the wire.
    #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
    pub async fn submit_halo2_job(&self, job: Halo2ProverJob) -> Result<ProverResult> {
        self.submit_to_workers(
            "halo2",
            &self.halo2_workers,
            Halo2ProverJobWrapper::Halo2(job),
        )
        .await
    }

    /// Dispatch `job` to the first free, initialized worker in `workers`,
    /// awaiting its result over a per-job oneshot.
    ///
    /// Shared by every kind (app/leaf/internal/root/halo2). If all workers are
    /// busy, waits until a worker becomes free or the pool is cancelled, with
    /// a one-second backstop. Returns the worker's `O` on success. For root,
    /// `O` is itself a `Result<RootProofState>`, so the caller flattens.
    async fn submit_to_workers<J, O>(
        &self,
        kind: &str,
        workers: &[WorkerHandle<J, O>],
        job: J,
    ) -> Result<O> {
        if workers.is_empty() {
            return Err(eyre::eyre!(
                "No {kind} workers available - check prover pool configuration"
            ));
        }

        let mut job_opt = Some(job);

        loop {
            // `Notify::notify_waiters` does not retain a permit. Register the
            // waiter before scanning to avoid losing a notification that races
            // with the availability checks below.
            let worker_free = self.worker_free.notified();
            tokio::pin!(worker_free);
            worker_free.as_mut().enable();

            for worker in workers {
                if !worker.is_busy.load(Ordering::SeqCst)
                    && worker.is_initialized.load(Ordering::SeqCst)
                {
                    if let Some(job) = job_opt.take() {
                        let (result_sender, result_receiver) = oneshot::channel();

                        match worker.job_sender.try_send((job, result_sender)) {
                            Ok(()) => {
                                return result_receiver
                                    .await
                                    .map_err(|_| eyre::eyre!("Worker dropped result channel"));
                            }
                            Err(crossbeam::channel::TrySendError::Full(returned)) => {
                                job_opt = Some(returned.0);
                            }
                            Err(crossbeam::channel::TrySendError::Disconnected(_)) => {
                                return Err(eyre::eyre!("Worker channel closed"));
                            }
                        }
                    }
                }
            }

            if self.cancel_token.is_cancelled() {
                return Err(eyre::eyre!("Prover pool cancelled"));
            }

            tokio::select! {
                _ = &mut worker_free => {}
                _ = self.cancel_token.cancelled() => {}
                _ = tokio::time::sleep(Duration::from_secs(1)) => {}
            }
        }
    }

    /// Check if any worker of the given type is available. App workers
    /// in the swap design are program-agnostic — any free worker can
    /// take any job and will swap-load if needed.
    pub fn has_available_worker(&self, job_type: JobType) -> bool {
        match job_type {
            JobType::ShardedApp => self.app_workers.iter().any(|w| {
                !w.is_busy.load(Ordering::SeqCst) && w.is_initialized.load(Ordering::SeqCst)
            }),
            JobType::Leaf => self.leaf_workers.iter().any(|w| {
                !w.is_busy.load(Ordering::SeqCst) && w.is_initialized.load(Ordering::SeqCst)
            }),
            JobType::Internal => self.internal_workers.iter().any(|w| {
                !w.is_busy.load(Ordering::SeqCst) && w.is_initialized.load(Ordering::SeqCst)
            }),
            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
            JobType::Root => self.root_workers.iter().any(|w| {
                !w.is_busy.load(Ordering::SeqCst) && w.is_initialized.load(Ordering::SeqCst)
            }),
            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
            JobType::Halo2 => self.halo2_workers.iter().any(|w| {
                !w.is_busy.load(Ordering::SeqCst) && w.is_initialized.load(Ordering::SeqCst)
            }),
        }
    }

    /// Get count of busy workers by type.
    pub fn busy_count(&self, job_type: JobType) -> usize {
        match job_type {
            JobType::ShardedApp => self
                .app_workers
                .iter()
                .filter(|w| w.is_busy.load(Ordering::SeqCst))
                .count(),
            JobType::Leaf => self
                .leaf_workers
                .iter()
                .filter(|w| w.is_busy.load(Ordering::SeqCst))
                .count(),
            JobType::Internal => self
                .internal_workers
                .iter()
                .filter(|w| w.is_busy.load(Ordering::SeqCst))
                .count(),
            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
            JobType::Root => self
                .root_workers
                .iter()
                .filter(|w| w.is_busy.load(Ordering::SeqCst))
                .count(),
            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
            JobType::Halo2 => self
                .halo2_workers
                .iter()
                .filter(|w| w.is_busy.load(Ordering::SeqCst))
                .count(),
        }
    }

    /// Check if all provers are initialized and ready to accept work.
    pub fn all_provers_initialized(&self) -> bool {
        let app_ready = self.app_workers.is_empty()
            || self
                .app_workers
                .iter()
                .all(|w| w.is_initialized.load(Ordering::SeqCst));
        let leaf_ready = self.leaf_workers.is_empty()
            || self
                .leaf_workers
                .iter()
                .all(|w| w.is_initialized.load(Ordering::SeqCst));
        let internal_ready = self.internal_workers.is_empty()
            || self
                .internal_workers
                .iter()
                .all(|w| w.is_initialized.load(Ordering::SeqCst));
        #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
        let root_ready = self.root_workers.is_empty()
            || self
                .root_workers
                .iter()
                .all(|w| w.is_initialized.load(Ordering::SeqCst));
        #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
        let halo2_ready = self.halo2_workers.is_empty()
            || self
                .halo2_workers
                .iter()
                .all(|w| w.is_initialized.load(Ordering::SeqCst));
        #[cfg(not(any(feature = "evm-prove", feature = "mock-provers")))]
        let (root_ready, halo2_ready) = (true, true);

        app_ready && leaf_ready && internal_ready && root_ready && halo2_ready
    }

    /// Get count of initialized workers by type.
    pub fn initialized_count(&self, job_type: JobType) -> usize {
        match job_type {
            JobType::ShardedApp => self
                .app_workers
                .iter()
                .filter(|w| w.is_initialized.load(Ordering::SeqCst))
                .count(),
            JobType::Leaf => self
                .leaf_workers
                .iter()
                .filter(|w| w.is_initialized.load(Ordering::SeqCst))
                .count(),
            JobType::Internal => self
                .internal_workers
                .iter()
                .filter(|w| w.is_initialized.load(Ordering::SeqCst))
                .count(),
            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
            JobType::Root => self
                .root_workers
                .iter()
                .filter(|w| w.is_initialized.load(Ordering::SeqCst))
                .count(),
            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
            JobType::Halo2 => self
                .halo2_workers
                .iter()
                .filter(|w| w.is_initialized.load(Ordering::SeqCst))
                .count(),
        }
    }

    /// Get the configured pool size for a given worker type. Used by
    /// `/readyz` to report initialization progress as "X / configured".
    pub fn configured_count(&self, job_type: JobType) -> usize {
        match job_type {
            JobType::ShardedApp => self.app_workers.len(),
            JobType::Leaf => self.leaf_workers.len(),
            JobType::Internal => self.internal_workers.len(),
            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
            JobType::Root => self.root_workers.len(),
            #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
            JobType::Halo2 => self.halo2_workers.len(),
        }
    }

    /// Total app jobs executed since process start.
    pub fn app_jobs_total(&self) -> u64 {
        APP_JOBS_TOTAL.load(Ordering::Relaxed)
    }

    /// Total app jobs that switched from one loaded GPU program to another.
    pub fn app_swaps_total(&self) -> u64 {
        APP_SWAPS_TOTAL.load(Ordering::Relaxed)
    }

    /// Shutdown the prover pool.
    pub fn shutdown(&self) {
        info!("Shutting down prover pool");
        self.cancel_token.cancel();
        for worker in &self.app_workers {
            worker.cancel();
        }
        for worker in &self.leaf_workers {
            worker.cancel();
        }
        for worker in &self.internal_workers {
            worker.cancel();
        }
        #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
        for worker in &self.root_workers {
            worker.cancel();
        }
        #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
        for worker in &self.halo2_workers {
            worker.cancel();
        }
    }
}

impl Drop for ProverPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Job type for worker availability checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobType {
    ShardedApp,
    Leaf,
    Internal,
    /// In-process EVM prove step; not network-dispatched.
    #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
    Root,
    /// In-process EVM prove step; not network-dispatched.
    #[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
    Halo2,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prover_config_defaults() {
        let config = ProversConfig::default();
        assert_eq!(config.max_app_provers, 2);
        assert_eq!(config.max_leaf_provers, 2);
        assert_eq!(config.max_internal_provers, 1);
        assert_eq!(config.max_root_provers, 1);
        assert_eq!(config.max_halo2_provers, 1);
    }

    #[cfg(not(feature = "mock-provers"))]
    #[test]
    fn parallel_app_result_fails_when_any_consumer_fails() {
        let result = merge_parallel_app_results(
            ProverResult::Success(vec![]),
            vec![],
            vec!["segment 7 failed".to_string()],
            false,
        );

        let ProverResult::Error(error) = result else {
            panic!("consumer failure must fail the parallel app job");
        };
        assert!(error.contains("segment 7 failed"));
    }

    #[cfg(not(feature = "mock-provers"))]
    #[test]
    fn parallel_app_result_is_canceled_when_any_part_is() {
        assert!(matches!(
            merge_parallel_app_results(ProverResult::Canceled, vec![], vec![], false),
            ProverResult::Canceled
        ));
        // A consumer that stopped leaves the segment set incomplete, so the
        // coordinator's own success does not make the job a success.
        assert!(matches!(
            merge_parallel_app_results(ProverResult::Success(vec![]), vec![], vec![], true),
            ProverResult::Canceled
        ));
    }

    // Role-gated pool construction. Under `mock-provers` each prover instance
    // builds instantly (no GPU), so we can construct a real pool and assert
    // which classes it built via `configured_count` (the pool vector lengths).
    #[cfg(feature = "mock-provers")]
    mod role_gating {
        use super::*;

        fn config() -> ProversConfig {
            ProversConfig {
                max_app_provers: 2,
                max_leaf_provers: 2,
                max_internal_provers: 1,
                max_root_provers: 1,
                max_halo2_provers: 1,
                default_segment_memory: None,
            }
        }

        /// `Full` builds every prover class at the configured sizes — today's
        /// behavior, unchanged.
        #[test]
        fn full_builds_all_classes() {
            let pool = ProverPool::new(&config(), WorkerRole::Full).expect("pool");
            assert_eq!(pool.configured_count(JobType::ShardedApp), 2);
            assert_eq!(pool.configured_count(JobType::Leaf), 2);
            assert_eq!(pool.configured_count(JobType::Internal), 1);
            assert_eq!(pool.configured_count(JobType::Root), 1);
            assert_eq!(pool.configured_count(JobType::Halo2), 1);
        }

        /// `StarkOnly` builds app/leaf/internal but NOT root/halo2 (no ~10 GB
        /// halo2 key load), even in a build where those kinds exist.
        #[test]
        fn normal_skips_root_and_halo2() {
            let pool = ProverPool::new(&config(), WorkerRole::StarkOnly).expect("pool");
            assert_eq!(pool.configured_count(JobType::ShardedApp), 2);
            assert_eq!(pool.configured_count(JobType::Leaf), 2);
            assert_eq!(pool.configured_count(JobType::Internal), 1);
            assert_eq!(pool.configured_count(JobType::Root), 0);
            assert_eq!(pool.configured_count(JobType::Halo2), 0);
        }

        /// `EvmDedicated` builds only root/halo2 — no app/leaf/internal pools.
        #[test]
        fn evm_dedicated_skips_app_leaf_internal() {
            let pool = ProverPool::new(&config(), WorkerRole::EvmDedicated).expect("pool");
            assert_eq!(pool.configured_count(JobType::ShardedApp), 0);
            assert_eq!(pool.configured_count(JobType::Leaf), 0);
            assert_eq!(pool.configured_count(JobType::Internal), 0);
            assert_eq!(pool.configured_count(JobType::Root), 1);
            assert_eq!(pool.configured_count(JobType::Halo2), 1);
        }
    }
}
