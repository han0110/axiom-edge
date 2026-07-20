//! Runtime program registration.
//!
//! A client posts a guest ELF plus the `SdkVmConfig` to build it under and the
//! worker derives the rest itself, namely app and aggregation keygen, the
//! transpile to a `VmExe`, the AOT execution instances, and the program's
//! verification baseline. The result is installed into
//! [`crate::artifacts::ArtifactStore`], which wakes the prover threads parked
//! on it, and published once every idle app prover has preloaded the
//! program's GPU prover.
//!
//! Preparation runs detached and the request waits for the part of it that
//! yields the verification baseline, so the response carries the key while the
//! AOT compile is still running. The manager compares those baselines across
//! workers once, at registration, and releases the key on `/program_vk` only
//! when every worker's `/readyz` also serves the program, so a completed
//! registration means the provers are initialized and preloaded and the first
//! proof starts unhindered.
//!
//! The first registration to be reserved pins the worker's VM config. Its
//! keyset is what every prover thread is built against, and those threads
//! cannot swap keysets, so a config change is a restart rather than a second
//! registration. The pin lifts only if that reservation is released before
//! anything is published.

use protocol::RegisterProgramRequest;

use crate::handlers::AppState;

/// How a `/register_program` request was classified. The HTTP layer answers
/// 200 for the two accepting variants and 409 or 400 for the rest.
///
/// The accepting variants carry the program's bincode verification baseline,
/// absent only on a mock build, which derives no keys.
pub enum RegistrationResult {
    /// The verifying key is derived and the AOT compile is running.
    Accepted(Option<Vec<u8>>),
    /// Nothing to do, since the same program, ELF and config are already here.
    AlreadyRegistered(Option<Vec<u8>>),
    /// Incompatible with what this worker already serves.
    Conflict(String),
    /// The request could not be parsed.
    Invalid(String),
}

#[cfg(not(feature = "mock-provers"))]
mod real_registration {
    use super::*;

    use eyre::{eyre, Result};
    use openvm_sdk_config::SdkVmConfig;
    use sdk_v2::{config::AppConfig, types::ExecutableFormat, Sdk};
    use std::sync::Arc;
    use std::time::Instant;
    use tracing::{error, info};
    use verify_stark::vk::VerificationBaseline;

    use crate::artifacts::{
        canonical_vm_config, ArtifactStore, DerivedProgram, PreparedProgram, RegistrationOutcome,
    };
    use crate::openvm_config::edge_app_and_agg_params;
    use crate::provers::AppExecutionInstances;

    /// Classify `request` and, when it is new, start preparing it. Returns once
    /// the verifying key is derived, leaving the AOT compile running.
    ///
    /// Preparation is detached from this request so a caller that gives up
    /// partway cannot strand a reservation with neither a key nor a failure
    /// recorded against it, which nothing later would be able to clear.
    pub async fn register_program(
        state: Arc<AppState>,
        request: RegisterProgramRequest,
    ) -> RegistrationResult {
        let vm_config: SdkVmConfig = match serde_json::from_str(&request.vm_config) {
            Ok(config) => config,
            Err(e) => {
                return RegistrationResult::Invalid(format!(
                    "vm_config is not a valid SdkVmConfig: {e}"
                ))
            }
        };
        let vm_config_json = canonical_vm_config(&vm_config);

        // The store is initialized before the HTTP server binds, so it is
        // always present for the lifetime of a request.
        let store = ArtifactStore::global().expect("artifact store initialized before serving");
        let program = request.program.clone();
        let fresh = match store.begin_registration(&program, &request.elf, &vm_config_json) {
            RegistrationOutcome::Conflict(reason) => return RegistrationResult::Conflict(reason),
            RegistrationOutcome::AlreadyRegistered => false,
            RegistrationOutcome::Accepted => {
                tokio::spawn(prepare_program(state, request, vm_config, vm_config_json));
                true
            }
        };

        // Both paths wait, a fresh registration for its own derivation and a
        // duplicate for the one already running, so the answer always carries
        // the key. The wait ends either way, since the detached preparation
        // always records a baseline or releases the reservation.
        let waited = program.clone();
        let baseline = tokio::task::spawn_blocking(move || store.wait_for_baseline(&waited))
            .await
            .ok()
            .flatten()
            .map(encode_baseline);

        match (baseline, fresh) {
            (Some(baseline), true) => RegistrationResult::Accepted(Some(baseline)),
            (Some(baseline), false) => RegistrationResult::AlreadyRegistered(Some(baseline)),
            (None, _) => RegistrationResult::Invalid(format!(
                "Failed to derive a verifying key for {program}"
            )),
        }
    }

    /// Derive and compile `request`'s program, then install it, preload the
    /// idle app provers with it, and publish it, so the first dispatched
    /// proof starts on an already built GPU prover.
    async fn prepare_program(
        state: Arc<AppState>,
        request: RegisterProgramRequest,
        vm_config: SdkVmConfig,
        vm_config_json: String,
    ) {
        let store = ArtifactStore::global().expect("artifact store initialized before serving");
        let program = request.program.clone();

        // The SDK panics rather than erroring on some malformed inputs (an
        // undecodable ELF, for one), so a panic releases the reservation too or
        // the program could never be retried.
        let derived =
            tokio::task::spawn_blocking(move || derive_program(request, vm_config, vm_config_json))
                .await;
        let derived = match derived {
            Ok(Ok(derived)) => derived,
            Ok(Err(e)) => {
                error!("Failed to derive {program}: {e:#}");
                return store.release_registration(&program);
            }
            Err(e) => {
                error!("Derivation of {program} panicked: {e}");
                return store.release_registration(&program);
            }
        };
        store.record_baseline(&program, derived.baseline.clone());

        match tokio::task::spawn_blocking(move || compile_program(derived)).await {
            Ok(Ok(prepared)) => {
                store.install_registration(prepared);
                state.prover_pool.preload_app_provers(&program).await;
                store.publish_registration(&program);
            }
            Ok(Err(e)) => {
                error!("Failed to compile {program}: {e:#}");
                store.fail_registration(&program);
            }
            Err(e) => {
                error!("Compilation of {program} panicked: {e}");
                store.fail_registration(&program);
            }
        }
    }

    /// Derive the keys, vmexe and verification baseline for `request`'s
    /// program. Takes seconds, dominated by keygen.
    fn derive_program(
        request: RegisterProgramRequest,
        vm_config: SdkVmConfig,
        vm_config_json: String,
    ) -> Result<DerivedProgram> {
        let RegisterProgramRequest { program, elf, .. } = request;
        let start = Instant::now();
        info!("Deriving {program} from {} ELF bytes", elf.len());

        // Only the VM config comes from the request. The app and aggregation
        // params are the ones every edge SDK is built with.
        let (app_params, agg_params) = edge_app_and_agg_params();
        let sdk = Sdk::new(AppConfig::new(vm_config, app_params), agg_params)
            .map_err(|e| eyre!("Failed to build SDK for {program}: {e}"))?;

        let exe = sdk
            .convert_to_exe(ExecutableFormat::from(&elf[..]))
            .map_err(|e| eyre!("Failed to transpile ELF for {program}: {e}"))?;

        // Keygen is deterministic in the VM config, so a registration that
        // extends a pinned deployment derives the keys that deployment
        // already holds. The store keeps the installed ones either way.
        let app_pk = Arc::new(sdk.app_pk().clone());
        let agg_stark_pk = Arc::new(sdk.agg_pk());

        // Generated from the exe this worker will prove with, so the baseline
        // a client verifies against is the one its proofs commit to.
        let baseline = sdk
            .prover(exe.clone())
            .map_err(|e| eyre!("Failed to build prover for {program}: {e}"))?
            .generate_baseline();

        info!("Derived {program} in {} ms", start.elapsed().as_millis());

        Ok(DerivedProgram {
            program,
            vm_config: vm_config_json,
            app_pk,
            agg_stark_pk,
            exe,
            baseline,
        })
    }

    /// AOT-compile a derived program's execution instances. Takes minutes.
    fn compile_program(derived: DerivedProgram) -> Result<PreparedProgram> {
        let start = Instant::now();
        info!("Compiling {}", derived.program);

        let execution_instances = Arc::new(AppExecutionInstances::new(
            &derived.program,
            &derived.app_pk,
            derived.exe.clone(),
        )?);

        info!(
            "Compiled {} in {} ms",
            derived.program,
            start.elapsed().as_millis()
        );

        Ok(PreparedProgram {
            derived,
            execution_instances,
        })
    }

    fn encode_baseline(baseline: VerificationBaseline) -> Vec<u8> {
        bincode::serialize(&baseline).expect("VerificationBaseline is bincode-serializable")
    }
}

#[cfg(not(feature = "mock-provers"))]
pub use real_registration::register_program;

#[cfg(feature = "mock-provers")]
mod mock_registration {
    use super::*;

    use std::sync::Arc;
    use tracing::info;

    use crate::artifacts::ArtifactStore;

    /// Mock builds derive no keys, so registration only records the program in
    /// the advertised loadout and reports no verifying key. The mock provers
    /// serve it without artifacts, and there is no GPU prover to preload.
    pub async fn register_program(
        _state: Arc<AppState>,
        request: RegisterProgramRequest,
    ) -> RegistrationResult {
        info!("Registering {} (mock build: no keygen)", request.program);
        ArtifactStore::global()
            .expect("artifact store initialized before serving")
            .record_registration(request.program);
        RegistrationResult::Accepted(None)
    }
}

#[cfg(feature = "mock-provers")]
pub use mock_registration::register_program;
