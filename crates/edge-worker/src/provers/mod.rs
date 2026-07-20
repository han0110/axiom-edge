//! Prover implementations for Edge mode.
//!
//! This module provides prover implementations for the recursion tree:
//! - Sharded App Proving: Initial proof generation for program segments
//! - Leaf Proving: Aggregation of app proofs into leaf proofs
//! - Internal Proving: Recursive aggregation of leaf/internal proofs
//!
//! Under `evm-prove`, two more provers run **locally** on the producing worker,
//! after the final internal proof of an Evm-typed proof:
//! - Root Proving: verifies the final stark proof in the root verifier circuit.
//!   Follows the `cuda` feature — GPU on a cuda build, CPU otherwise.
//! - Halo2 Proving: wraps the root proof into an EVM proof. CPU-only in rc.2
//!   (no GPU halo2 variant upstream yet).
//!
//! Root + halo2 are NOT network-dispatched. The worker that produces the
//! final internal proof drives them in-process via the prover pool's
//! dedicated root/halo2 worker threads.
//!
//! # Architecture
//!
//! Each prover type has:
//! - A job struct (e.g., `ShardedAppProverJob`) containing job parameters
//! - A prover instance struct (e.g., `AppProverInstance`) that holds reusable prover state
//! - A `prove_*` function for mock mode (creates instances internally)
//! - A `prove_*_with_prover` function for real mode (uses provided instance)
//!
//! The ProverPool creates prover instances at worker startup and passes them
//! to job functions, ensuring efficient reuse of expensive GPU/VM state.

#[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
mod halo2_prover;
mod internal_prover;
mod leaf_prover;
#[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
mod root_prover;
mod sharded_app_prover;

#[cfg(feature = "mock-provers")]
pub use halo2_prover::prove_halo2;
pub use internal_prover::prove_internal;
pub use leaf_prover::prove_leaf;
#[cfg(feature = "mock-provers")]
pub use root_prover::prove_root;
pub use sharded_app_prover::prove_sharded_app;

// Export prover instance types and _with_prover functions for real mode
#[cfg(not(feature = "mock-provers"))]
pub use internal_prover::{prove_internal_with_prover, InternalProverInstance};
#[cfg(not(feature = "mock-provers"))]
pub use leaf_prover::{prove_leaf_with_prover, LeafProverInstance};
#[cfg(not(feature = "mock-provers"))]
pub use sharded_app_prover::{
    build_gpu_prover, create_parallel_prove_jobs, prove_sharded_app_with_prover,
    AppExecutionInstances, ParallelCoordinatorFn, ProverType, SegmentConsumerFn,
};

#[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
pub use halo2_prover::{prove_halo2_with_prover, Halo2ProverInstance};
#[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
pub use root_prover::{prove_root_with_prover, RootProverInstance};

use protocol::{ProofContext, ProofResult};

/// Result from a prover.
pub enum ProverResult {
    /// Successful proof results (may be multiple for sharded app prove)
    Success(Vec<ProofResult>),
    /// Error during proving
    Error(String),
    /// The proof was canceled, so the job stopped at its next segment and has
    /// nothing to report. The manager has already dropped the proof, so this
    /// is not sent on as a failure.
    Canceled,
}

/// Job for the sharded app prover.
pub struct ShardedAppProverJob {
    pub context: ProofContext,
    pub num_provers: usize,
    pub prover_id: usize,
    pub input_path: String,
    pub segment_memory: Option<usize>,
    /// Number of app prover threads to use concurrently (1 = sequential, >1 = parallel MPMC)
    pub max_app_provers: usize,
    /// Optional channel for streaming results as each segment proof completes.
    /// When Some, results are sent immediately instead of being collected into a Vec.
    pub result_tx: Option<crossbeam::channel::Sender<ProofResult>>,
    /// Per-circuit `DeferralState` paths. One entry per
    /// deferral circuit, in def-idx order. Empty selects today's non-deferral
    /// execution path. The prover reads each file, decodes a `DeferralState`,
    /// and populates `StdIn.deferrals[idx]` before VM execution.
    pub deferral_state_paths: Vec<String>,
}

/// Job for the leaf prover.
pub struct LeafProverJob {
    pub context: ProofContext,
    pub app_proofs: Vec<proof::ProofWithPublicValue<proof::F>>,
    pub segment_start: usize,
    pub segment_end: usize,
}

/// Job for the internal prover.
pub struct InternalProverJob {
    pub context: ProofContext,
    pub child_proofs: Vec<proof::ProofWithPublicValue<proof::F>>,
    pub layer_idx: usize,
    pub segment_start: usize,
    pub segment_end: usize,
    pub is_final_proof: bool,
    /// True iff THIS proof carries a deferral tail (the manager attached a
    /// `DeferralTailDispatch`). Distinct from the deployment-wide
    /// `is_deferral_deployment`: on a deferral deployment a proof whose
    /// program made no deferred calls arrives with no tail, so it must take
    /// the normal final-internal wrap path (matching the SDK's
    /// `StarkProver::prove`, which gates `prove_def`/`prove_mixed` on
    /// `!def_inputs.is_empty()` and always runs the final wrap). Only the
    /// final internal proof consults this; non-final jobs set it `false`.
    pub proof_has_deferral: bool,
}

/// Job for the root prover (in-process EVM prove).
///
/// Triggered locally by the worker that produced the final internal proof of
/// an Evm-typed proof. Carries the decoded `ProofWithPublicValue<F>` (the
/// wire-side payload of the just-emitted `Internal` result) so root prove
/// can reconstruct a `VmStarkProof` without a re-decode hop.
///
/// On a deferral job the tail merge (`run_deferral_tail_merge`) also
/// produces the `DeferralMerkleProofs` for `(DEFERRAL_AS, 0)` and threads
/// them here so root prove can attach them to its `VmStarkProof` before
/// running tracegen — the merged proof verifies only with these attached.
/// `None` on non-deferral jobs.
#[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
pub struct RootProverJob {
    pub context: ProofContext,
    pub final_internal_proof: proof::ProofWithPublicValue<proof::F>,
    #[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
    pub deferral_merkle_proofs: Option<verify_stark::deferral::DeferralMerkleProofs<proof::F>>,
    /// True iff THIS proof ran the deferral tail merge (`prove_mixed` set
    /// `proofs_type = Combined`). Per-proof, NOT deployment-wide: on a
    /// deferral deployment a no-deferral proof keeps `proofs_type = Vm`, so
    /// the root wrap-retry must use `Vm` for it and `Combined` only for a
    /// real deferral proof. Note this is distinct from
    /// `deferral_merkle_proofs.is_some()` — a no-deferral proof on a deferral
    /// deployment still carries a depth-0 `DeferralMerkleProofs` yet has
    /// `proofs_type = Vm`.
    #[cfg(all(not(feature = "mock-provers"), feature = "evm-prove"))]
    pub proof_has_deferral: bool,
}

/// Job for the halo2 prover (in-process EVM prove).
///
/// Triggered locally after a successful root prove. Carries the typed root
/// proof so we don't re-encode/decode between the root + halo2 stages
/// inside the same worker process.
#[cfg(any(feature = "evm-prove", feature = "mock-provers"))]
pub struct Halo2ProverJob {
    pub context: ProofContext,
    pub root_proof: proof::RootProof,
}

// ============================================================================
// Real prover type aliases (default, disabled when mock-provers is enabled)
// ============================================================================

#[cfg(not(feature = "mock-provers"))]
pub mod real_prover_types {
    use sdk_v2::config::{MAX_NUM_CHILDREN_INTERNAL, MAX_NUM_CHILDREN_LEAF};

    // Type alias - sdk_v2::DefaultStarkEngine resolves to GPU or CPU engine
    // based on whether the cuda feature is enabled on sdk-v2
    pub type RecursionEngine = sdk_v2::DefaultStarkEngine;

    // VM Builder type alias - uses GPU builder when cuda is enabled, CPU otherwise
    #[cfg(feature = "cuda")]
    pub type SdkVmBuilder = openvm_sdk_config::SdkVmGpuBuilder;

    #[cfg(not(feature = "cuda"))]
    pub type SdkVmBuilder = openvm_sdk_config::SdkVmCpuBuilder;

    // Prover type aliases based on GPU feature
    #[cfg(not(feature = "cuda"))]
    pub type LeafProver = continuations_v2::prover::InnerCpuProver<MAX_NUM_CHILDREN_LEAF>;

    #[cfg(feature = "cuda")]
    pub type LeafProver = continuations_v2::prover::InnerGpuProver<MAX_NUM_CHILDREN_LEAF>;

    #[cfg(not(feature = "cuda"))]
    pub type InternalProver = continuations_v2::prover::InnerCpuProver<MAX_NUM_CHILDREN_INTERNAL>;

    #[cfg(feature = "cuda")]
    pub type InternalProver = continuations_v2::prover::InnerGpuProver<MAX_NUM_CHILDREN_INTERNAL>;

    #[cfg(feature = "evm-prove")]
    pub type RootProver = sdk_v2::prover::RootProver;

    /// Engine flavor used by the root prover (Bn254-flavored, distinct from
    /// the BabyBear `RecursionEngine` used in the leaf/internal layers). The
    /// concrete type is private to sdk-v2; the alias re-derives it the same
    /// way sdk-v2 does so we can cache instances by value.
    #[cfg(all(feature = "evm-prove", not(feature = "cuda")))]
    pub type RootEngine =
        openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;

    #[cfg(all(feature = "evm-prove", feature = "cuda"))]
    pub type RootEngine = openvm_cuda_backend::BabyBearBn254Poseidon2GpuEngine;

    #[cfg(feature = "evm-prove")]
    pub type Halo2Prover = sdk_v2::prover::Halo2Prover;

    // Re-export prover types
    pub use continuations_v2::prover::ChildVkKind;
}

#[cfg(not(feature = "mock-provers"))]
pub use real_prover_types::*;
