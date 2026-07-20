//! Artifact management for the worker's proving artifacts.
//!
//! Disk layout (multi-ELF):
//! ```text
//! {artifacts_path}/
//!     app_pk                                     # shared, deployment-level
//!     agg_stark_pk                               # shared, deployment-level
//!     root_pk                                    # shared, only under `evm-prove`
//!     programs/{name}/{version}/program.vmexe   # per-ELF
//!
//! {halo2_pk_path}/                               # only under `evm-prove`
//!     halo2_pk                                   # ~10GB+
//!     kzg_bn254_<verifier_k>.srs
//!     kzg_bn254_<wrapper_k>.srs
//! ```
//!
//! `app_pk` and `agg_stark_pk` are produced by `sdk.app_keygen()` and
//! `sdk.agg_pk()` — both depend only on the SDK config (`edge_vm_config`),
//! not on any ELF. They are loaded once per worker. `program.vmexe` is the
//! compiled ELF; the worker loads one per `(name, version)` entry in
//! `EDGE_PROGRAMS`.
//!
//! Under `evm-prove`, `root_pk` lives alongside the stark keys (small, cheap,
//! produced by the same `keygen` binary). `halo2_pk` and its SRS files live in
//! a separate directory (`halo2_pk_path`) because the pk is huge and the SRS
//! is trusted-setup material that's typically shared read-only across many
//! deployments. Missing/unconfigured `halo2_pk_path` is tolerated — a worker
//! built with `evm-prove` but missing halo2 inputs reports not-ready instead
//! of panicking, so a stark-only deployment can still boot the evm-prove
//! binary.
//!
//! The disk layout above is optional. With `EDGE_PROGRAMS` unset the worker
//! boots with an empty deployment and derives everything from the first
//! `/register_program` instead (see [`crate::registration`]), so the store
//! publishes its artifacts behind a mutex rather than owning them outright.

use eyre::Result;
use once_cell::sync::OnceCell;
use protocol::ProgramRef;
#[cfg(not(feature = "mock-provers"))]
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::config::ArtifactsConfig;

/// Global artifact store instance.
static ARTIFACT_STORE: OnceCell<Arc<ArtifactStore>> = OnceCell::new();

// ============================================================================
// Real prover types (default, disabled when mock-provers is enabled)
// ============================================================================

#[cfg(not(feature = "mock-provers"))]
mod real_artifacts {
    use super::*;
    use openvm_sdk_config::SdkVmConfig;
    use openvm_stark_backend::Val;
    // Deferral artifacts (unconditional — deferral proving is STARK-level and
    // does not need `root-prover`/`evm-prove`; only halo2/root wrapping does).
    use openvm_stark_sdk::config::baby_bear_poseidon2::Digest;
    use sdk_v2::fs::read_object_from_file;
    use sdk_v2::keygen::{AggProvingKey, AppProvingKey, SdkCachedProvingKey};
    use sdk_v2::openvm_circuit::arch::instructions::exe::VmExe;
    use sdk_v2::Sdk;
    use sdk_v2::SC;
    use std::path::Path;
    use std::sync::{Condvar, Mutex, MutexGuard};
    use tracing::warn;
    use verify_stark::vk::VerificationBaseline;

    use crate::prover_pool::AppWorkerContext;
    use crate::provers::AppExecutionInstances;
    // Halo2/root artifacts — only under `evm-prove`.
    #[cfg(feature = "evm-prove")]
    use {
        sdk_v2::fs::read_halo2_pk_from_file,
        sdk_v2::halo2_params::CacheHalo2ParamsReader,
        sdk_v2::keygen::{Halo2ProvingKey, RootProvingKey},
    };

    pub type Exe = VmExe<Val<SC>>;

    /// EVM (halo2) inputs needed at worker boot to construct the halo2 prover.
    ///
    /// The pk + params reader pair is what `Halo2Prover::new(reader, halo2_pk)`
    /// consumes — the reader resolves `kzg_bn254_<k>.srs` files lazily using
    /// the `k` values baked into the pk's verifier/wrapper pinnings.
    ///
    /// `Clone` is an `Arc` bump per field; a registration that extends an
    /// already-published deployment carries these keys over untouched.
    #[cfg(feature = "evm-prove")]
    #[derive(Clone)]
    pub struct EvmArtifacts {
        pub root_pk: Arc<RootProvingKey>,
        pub halo2_pk: Arc<Halo2ProvingKey>,
        pub halo2_params_reader: Arc<CacheHalo2ParamsReader>,
    }

    /// Deferral-aware artifacts (verify-stark deferral path).
    ///
    /// Holds only the serialized `SdkCachedProvingKey` produced by
    /// `keygen --with-deferral`. The full `Sdk` cannot live in the shared
    /// `ArtifactStore` — its `Transpiler<F>` carries `Rc<dyn ...>` which is
    /// `!Send + !Sync` — so each consumer thread reconstructs its own SDK via
    /// [`Sdk::from_deferral_cached_proving_key`] when it needs one. The
    /// reconstruction round-trip is verified once at load time (see
    /// `try_load_deferral`) so a broken artifact surfaces at boot rather than
    /// at first-prove. The per-worker reconstruction happens at the tail
    /// flow's entry point (`run_deferral_tail_merge`).
    #[derive(Clone)]
    pub struct DeferralArtifacts {
        pub cached_pk: Arc<SdkCachedProvingKey<SdkVmConfig>>,
        /// `def_hook_cached_commit` extracted at boot reconstruction. The
        /// VM-tree leaf/internal/internal_recursive provers must be built
        /// with this as `Some(...)` to bake the deferral hook AIR into the
        /// circuit. Stored as a plain `Digest` (Send+Sync) so it can be
        /// reused across worker threads without holding the SDK itself
        /// (`Sdk` is `!Send + !Sync`).
        pub def_hook_cached_commit: Digest,
        /// `def_hook_commit` extracted at boot reconstruction. Used by the
        /// deferral-aware root prover (whose root circuit binds to the
        /// deferral-path hook commit, not the cached one).
        pub def_hook_commit: Digest,
    }

    /// Per-deployment shared keys + per-program vmexes.
    #[derive(Clone)]
    pub struct EdgeArtifacts {
        pub app_pk: Arc<AppProvingKey<SdkVmConfig>>,
        pub agg_stark_pk: Arc<AggProvingKey>,
        /// vmexe keyed by (name, version).
        pub programs: HashMap<ProgramRef, Arc<Exe>>,
        /// Root + halo2 keys + KZG params reader. The field itself is
        /// `#[cfg(feature = "evm-prove")]`-gated — stark-only builds don't
        /// compile it in. When the feature is on, `Some(...)` means the
        /// worker can serve Evm-typed proofs; `None` means the build is
        /// evm-capable but the operator didn't supply `halo2_pk_path` or
        /// the load failed (stark-only deployment using the evm-prove
        /// binary). The EVM prove path fails fast in that case rather
        /// than panicking.
        #[cfg(feature = "evm-prove")]
        pub evm: Option<EvmArtifacts>,
        /// Deferral-aware proving artifacts. Under `evm-prove`:
        /// `Some(...)` when `enable_deferral` is set (the cached_pk loads from
        /// `<artifacts_path>/deferral/cached_pk`, or boot hard-fails); `None`
        /// when deferral is disabled. The non-deferral path keeps working
        /// regardless. Not feature-gated — deferral is compiled into every real
        /// build and toggled at runtime by `enable_deferral`.
        pub deferral: Option<DeferralArtifacts>,
    }

    impl EdgeArtifacts {
        /// Load shared keys plus a vmexe for each entry in `programs`.
        ///
        /// **Deferral mode is a deployment toggle.** When
        /// `enable_deferral` is set and the cached pk loads
        /// cleanly, the deferral keyset becomes the worker's *single*
        /// keyset: `app_pk`, `agg_stark_pk`, and `evm.root_pk` all come
        /// from `cached_pk.{app_pk, agg_pk, root_pk}` (the on-disk
        /// `<base>/app_pk` / `<base>/agg_stark_pk` / `<base>/root_pk`
        /// files are NOT required in that mode — the cached pk carries
        /// deferral-aware versions of all three). Switching modes is a
        /// reconfigure + redeploy; there is no per-job dual-keyset
        /// routing.
        ///
        /// In the default (non-deferral) case the loader keeps today's
        /// behavior byte-identical: stark files load from disk, `evm`
        /// loads best-effort, `deferral` stays `None`.
        pub fn load_from_disk(
            base_path: &Path,
            programs: &[ProgramRef],
            #[cfg(feature = "evm-prove")] halo2_pk_path: Option<&Path>,
            enable_deferral: bool,
        ) -> Result<Self> {
            // Load the deferral cached pk first (if enabled). In
            // deferral mode this becomes the single source of
            // `app_pk`/`agg_stark_pk`/`root_pk` for the rest of the boot
            // path. Derived from `base_path` (like the other shared keys);
            // a hard error if enabled but missing.
            let deferral = Self::try_load_deferral(base_path, enable_deferral)?;
            let in_deferral_mode = deferral.is_some();

            let (app_pk, agg_stark_pk) = if in_deferral_mode {
                let d = deferral.as_ref().expect("checked above");
                info!(
                    "Deferral mode: sourcing app_pk + agg_stark_pk from cached_pk; \
                     disk app_pk/agg_stark_pk not consulted"
                );
                (
                    Arc::new(d.cached_pk.app_pk.clone()),
                    Arc::new(d.cached_pk.agg_pk.clone()),
                )
            } else {
                let app_pk_path = base_path.join("app_pk");
                let agg_stark_pk_path = base_path.join("agg_stark_pk");

                info!("Loading app_pk from {}", app_pk_path.display());
                let app_pk: AppProvingKey<SdkVmConfig> = read_object_from_file(&app_pk_path)?;

                info!("Loading agg_stark_pk from {}", agg_stark_pk_path.display());
                let agg_stark_pk: AggProvingKey = read_object_from_file(&agg_stark_pk_path)?;
                (Arc::new(app_pk), Arc::new(agg_stark_pk))
            };

            let mut program_map = HashMap::with_capacity(programs.len());
            for program in programs {
                let vmexe_path = base_path
                    .join("programs")
                    .join(&program.name)
                    .join(program.version.to_string())
                    .join("program.vmexe");
                info!(
                    "Loading vmexe for {} from {}",
                    program,
                    vmexe_path.display()
                );
                let exe: Exe = read_object_from_file(&vmexe_path)?;
                program_map.insert(program.clone(), Arc::new(exe));
            }

            #[cfg(feature = "evm-prove")]
            let evm = Self::try_load_evm(base_path, halo2_pk_path, deferral.as_ref());

            Ok(Self {
                app_pk,
                agg_stark_pk,
                programs: program_map,
                #[cfg(feature = "evm-prove")]
                evm,
                deferral,
            })
        }

        /// The same shared keys carrying one more program's vmexe. A
        /// registration that extends an already-published deployment reuses
        /// the installed keys rather than the ones it just derived, so the
        /// prover threads keep proving against the keyset they were built
        /// with.
        fn with_program(&self, program: ProgramRef, exe: Arc<Exe>) -> Self {
            let mut extended = self.clone();
            extended.programs.insert(program, exe);
            extended
        }

        /// Convenience: `Some(def_hook_cached_commit)` when this worker
        /// is deferral-configured, `None` otherwise. The leaf/internal
        /// prover construction passes this directly into
        /// `*Prover::from_pk(.., def_hook_cached_commit)` — so the
        /// deferral hook AIR ships with the circuit when the deployment
        /// is in deferral mode, and is absent (today's path) when it
        /// isn't.
        pub fn def_hook_cached_commit(&self) -> Option<Digest> {
            self.deferral.as_ref().map(|d| d.def_hook_cached_commit)
        }

        /// Convenience: `Some(def_hook_commit)` when this worker is
        /// deferral-configured, `None` otherwise. The root prover
        /// construction passes this directly so root binds to the
        /// deferral-path hook commit.
        pub fn def_hook_commit(&self) -> Option<Digest> {
            self.deferral.as_ref().map(|d| d.def_hook_commit)
        }

        /// Whether this worker loaded the deferral keyset (single-keyset model):
        /// leaf/internal/root VKs carry the deferral hook AIR and the
        /// verifier expects `DeferralMerkleProofs` on every proof.
        ///
        /// This is a KEYSET-level fact, not a per-job one. A deferral
        /// deployment serves BOTH deferral and non-deferral proofs (mixed
        /// programs on one keyset). The prover *shape* toggles —
        /// `prove_def → prove_mixed → wrap`, the final-internal wrap-skip, and
        /// root's `ProofsType::Combined` — are keyed PER-PROOF on whether that
        /// proof carries a deferral tail (`DeferralTailDispatch`), mirroring
        /// the SDK's `StarkProver::prove` (which gates the merge on
        /// `!def_inputs.is_empty()` and always runs the final wrap). A
        /// no-deferral proof here takes the normal wrap path and gets a
        /// depth-0 `DeferralMerkleProofs` built by the terminal app worker.
        pub fn is_deferral_deployment(&self) -> bool {
            self.deferral.is_some()
        }

        /// Load the deferral cached pk when `enable_deferral` is set. The
        /// cached_pk lives at the conventional `{base_path}/deferral/cached_pk`
        /// (alongside the other shared keys), so the location is derived, not
        /// configured — see `ArtifactsConfig::enable_deferral`.
        ///
        /// `enable_deferral == false` → `Ok(None)` (today's non-deferral path,
        /// byte-identical). `enable_deferral == true` → the cached_pk MUST load
        /// and reconstruct cleanly; any failure is a hard error so a deferral
        /// deployment can never silently boot as non-deferral.
        fn try_load_deferral(
            base_path: &Path,
            enable_deferral: bool,
        ) -> Result<Option<DeferralArtifacts>> {
            if !enable_deferral {
                info!("Deferral disabled; worker will serve non-deferral proofs only");
                return Ok(None);
            }

            let path = base_path.join("deferral").join("cached_pk");
            info!(
                "Loading deferral SdkCachedProvingKey from {}",
                path.display()
            );
            let cached_pk: SdkCachedProvingKey<SdkVmConfig> = read_object_from_file(&path)
                .map_err(|e| {
                    eyre::eyre!(
                        "enable_deferral is set but the deferral cached_pk at {} failed to load \
                         (did `keygen --with-deferral` run?): {e}",
                        path.display()
                    )
                })?;

            // Smoke-test reconstruction once at load time so a broken cached_pk
            // surfaces at boot, not at first-prove. The reconstructed SDK is
            // dropped — it can't be cached in `ArtifactStore` (its transpiler
            // is `!Send`), so per-worker-thread reconstruction happens later.
            info!("Verifying deferral SDK reconstruction round-trip");
            let rebuilt =
                Sdk::from_deferral_cached_proving_key(cached_pk.clone()).map_err(|e| {
                    eyre::eyre!(
                        "Failed to reconstruct deferral SDK from {}: {e}",
                        path.display()
                    )
                })?;

            // Extract the two deferral-path commits while we have the SDK in
            // hand — they're cheap to copy out and are needed at
            // leaf/internal/root prover construction time without
            // re-reconstructing the SDK (which is `!Send + !Sync`).
            let def_hook_cached_commit = rebuilt.def_hook_cached_commit().ok_or_else(|| {
                eyre::eyre!(
                    "Reconstructed deferral SDK from {} but def_hook_cached_commit() returned None",
                    path.display()
                )
            })?;
            let def_hook_commit = rebuilt.def_hook_commit().ok_or_else(|| {
                eyre::eyre!(
                    "Reconstructed deferral SDK from {} but def_hook_commit() returned None",
                    path.display()
                )
            })?;
            drop(rebuilt);

            Ok(Some(DeferralArtifacts {
                cached_pk: Arc::new(cached_pk),
                def_hook_cached_commit,
                def_hook_commit,
            }))
        }

        #[cfg(feature = "evm-prove")]
        fn try_load_evm(
            base_path: &Path,
            halo2_pk_path: Option<&Path>,
            deferral: Option<&DeferralArtifacts>,
        ) -> Option<EvmArtifacts> {
            let Some(halo2_dir) = halo2_pk_path else {
                info!(
                    "halo2_pk_path is not configured; evm-prove build will serve stark proofs only"
                );
                return None;
            };

            // Deferral mode: `root_pk` is part of the deferral
            // keyset (`cached_pk.root_pk`). If it wasn't materialised at
            // keygen time, the worker can't serve evm-typed proofs — we
            // warn and return `None` so the worker boots as
            // not-evm-ready (`/readyz` reports the right thing).
            //
            // Non-deferral mode: today's behavior — load `<base>/root_pk`.
            let root_pk: RootProvingKey = if let Some(d) = deferral {
                match d.cached_pk.root_pk.as_ref() {
                    Some(pk) => {
                        info!(
                            "Deferral mode: sourcing root_pk from cached_pk; \
                             disk root_pk not consulted"
                        );
                        pk.clone()
                    }
                    None => {
                        warn!(
                            "Deferral mode: cached_pk.root_pk is None (run \
                             `keygen --with-deferral` end-to-end to populate it). \
                             Worker will not be evm-ready."
                        );
                        return None;
                    }
                }
            } else {
                let root_pk_path = base_path.join("root_pk");
                info!("Loading root_pk from {}", root_pk_path.display());
                match read_object_from_file::<RootProvingKey, _>(&root_pk_path) {
                    Ok(pk) => pk,
                    Err(e) => {
                        warn!(
                            "Failed to load root_pk from {}: {}. Worker will not be evm-ready.",
                            root_pk_path.display(),
                            e
                        );
                        return None;
                    }
                }
            };

            let halo2_pk_path = halo2_dir.join("halo2_pk");
            info!("Loading halo2_pk from {}", halo2_pk_path.display());
            let halo2_pk: Halo2ProvingKey = match read_halo2_pk_from_file(&halo2_pk_path) {
                Ok(pk) => pk,
                Err(e) => {
                    warn!(
                        "Failed to load halo2_pk from {}: {}. Worker will not be evm-ready.",
                        halo2_pk_path.display(),
                        e
                    );
                    return None;
                }
            };

            info!(
                "Building Halo2 params reader over {} (verifier_k={}, wrapper_k={})",
                halo2_dir.display(),
                halo2_pk.verifier.pinning.metadata.config_params.k,
                halo2_pk.wrapper.pinning.metadata.config_params.k,
            );
            let halo2_params_reader = CacheHalo2ParamsReader::new(halo2_dir);

            Some(EvmArtifacts {
                root_pk: Arc::new(root_pk),
                halo2_pk: Arc::new(halo2_pk),
                halo2_params_reader: Arc::new(halo2_params_reader),
            })
        }
    }

    /// Everything the worker can prove with right now.
    ///
    /// Published either at boot from disk (`EDGE_PROGRAMS`) or by
    /// `/register_program`, which is why it lives behind a mutex instead of
    /// being owned by the store.
    #[derive(Default)]
    struct Deployment {
        /// Loadout this worker advertises, meaning the programs it booted
        /// with plus every program registered since.
        programs: Vec<ProgramRef>,
        /// Shared keys and vmexes. `None` until the first publish.
        artifacts: Option<Arc<EdgeArtifacts>>,
        /// App-worker context covering the same programs, rebuilt on publish.
        app_ctx: Option<Arc<AppWorkerContext>>,
        /// Canonical JSON of the VM config this worker's keyset was built
        /// under, pinned by the disk load or by the first registration to be
        /// reserved. A registration under a different config needs a different
        /// keyset, which the running prover threads cannot swap.
        ///
        /// Pinned at reservation rather than at publish because the AOT compile
        /// between the two is long enough for a second registration to slip in
        /// under a config this worker will never build keys for.
        vm_config: Option<String>,
        /// Per-program registration records, keyed by the registered program.
        /// Disk-seeded programs have no entry and expose no baseline.
        registered: HashMap<ProgramRef, RegisteredProgram>,
    }

    /// A program accepted by `/register_program`. `baseline` is filled in as
    /// soon as the derivation finishes, which is well ahead of the AOT compile
    /// that makes the program servable.
    struct RegisteredProgram {
        /// Kept verbatim so a repeat registration is classified exactly, with
        /// identical bytes idempotent and different bytes a conflict.
        elf: Vec<u8>,
        baseline: Option<VerificationBaseline>,
        /// Set when preparation failed, so the program stays visible to
        /// `is_ready` and holds the worker out of the ready set instead of
        /// vanishing and leaving it advertising a program it cannot serve.
        failed: bool,
    }

    /// What a registration derives before its AOT compile, which is everything
    /// needed to answer for the program's verifying key.
    pub struct DerivedProgram {
        pub program: ProgramRef,
        /// Canonical JSON of the config the program was built under.
        pub vm_config: String,
        pub app_pk: Arc<AppProvingKey<SdkVmConfig>>,
        pub agg_stark_pk: Arc<AggProvingKey>,
        pub exe: Arc<Exe>,
        pub baseline: VerificationBaseline,
    }

    /// A derived program whose AOT compile has finished, so it can be published.
    pub struct PreparedProgram {
        pub derived: DerivedProgram,
        pub execution_instances: Arc<AppExecutionInstances>,
    }

    /// How a `/register_program` request compares against the published
    /// deployment.
    pub enum RegistrationOutcome {
        /// The program is reserved; preparation should start.
        Accepted,
        /// The same program, ELF and config are already here.
        AlreadyRegistered,
        /// Incompatible with what this worker already serves.
        Conflict(String),
    }

    /// Canonical JSON for a VM config, used to compare a registration against
    /// the config this worker is pinned to. Serde emits struct fields in
    /// declaration order and the config is a plain struct tree, so equal
    /// configs always produce equal strings and serialization cannot fail.
    pub fn canonical_vm_config(config: &SdkVmConfig) -> String {
        serde_json::to_string(config).expect("SdkVmConfig is JSON-serializable")
    }

    /// Store for proving artifacts.
    pub struct ArtifactStore {
        #[allow(dead_code)]
        artifacts_path: PathBuf,
        deployment: Mutex<Deployment>,
        /// Signalled after every publish. Prover threads are OS threads, not
        /// tasks, so they block here rather than awaiting.
        published: Condvar,
    }

    impl ArtifactStore {
        /// Initialize the global artifact store and load all configured
        /// artifacts. On load failure, the store still initializes but
        /// `is_ready()` returns false and `/readyz` reports not ready.
        ///
        /// An empty loadout means the deployment is registration-driven, so
        /// there is nothing on disk to load and the shared keys arrive with
        /// the first `/register_program`. A deferral deployment always sets
        /// `EDGE_PROGRAMS`, so its keyset still loads here.
        pub fn init(config: &ArtifactsConfig, programs: Vec<ProgramRef>) -> Result<()> {
            let artifacts_path = config
                .artifacts_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("/data/artifacts"));

            info!(
                "Initializing artifact store: path={}, programs={}",
                artifacts_path.display(),
                programs.len()
            );

            #[cfg(feature = "evm-prove")]
            let halo2_pk_path = config.halo2_pk_path.clone();
            let enable_deferral = config.enable_deferral;

            let edge_artifacts = if programs.is_empty() {
                info!("No programs configured; deployment arrives via /register_program");
                None
            } else {
                let load_result = EdgeArtifacts::load_from_disk(
                    &artifacts_path,
                    &programs,
                    #[cfg(feature = "evm-prove")]
                    halo2_pk_path.as_deref(),
                    enable_deferral,
                );

                match load_result {
                    Ok(artifacts) => {
                        info!(
                            "Successfully loaded Edge artifacts ({} vmexes)",
                            artifacts.programs.len()
                        );
                        Some(Arc::new(artifacts))
                    }
                    Err(e) => {
                        warn!(
                            "Failed to load Edge artifacts: {}. Worker will report not ready.",
                            e
                        );
                        None
                    }
                }
            };

            // Pin the config the disk keyset was built under so a later
            // registration is checked against it instead of being served by a
            // keyset it does not match.
            let vm_config = edge_artifacts
                .as_ref()
                .map(|a| canonical_vm_config(&a.app_pk.app_vm_pk.vm_config));

            let store = Self {
                artifacts_path,
                deployment: Mutex::new(Deployment {
                    programs,
                    artifacts: edge_artifacts,
                    vm_config,
                    ..Default::default()
                }),
                published: Condvar::new(),
            };

            ARTIFACT_STORE
                .set(Arc::new(store))
                .map_err(|_| eyre::eyre!("Artifact store already initialized"))?;
            Ok(())
        }

        pub fn global() -> Option<Arc<ArtifactStore>> {
            ARTIFACT_STORE.get().cloned()
        }

        fn lock(&self) -> MutexGuard<'_, Deployment> {
            self.deployment.lock().expect("artifact store poisoned")
        }

        /// Shared deployment-level proving keys and the per-program vmexe
        /// map. `None` until a deployment is published.
        pub fn get_edge_artifacts(&self) -> Option<Arc<EdgeArtifacts>> {
            self.lock().artifacts.clone()
        }

        /// Block until a deployment is published, then return its shared keys.
        ///
        /// Prover threads call this at startup. On a registration-driven
        /// worker they park here and stay un-initialized, which keeps
        /// `/readyz` false until the worker can actually prove.
        pub fn wait_for_edge_artifacts(&self) -> Arc<EdgeArtifacts> {
            let mut deployment = self.lock();
            if deployment.artifacts.is_none() {
                info!("Waiting for a deployment before building provers");
            }
            while deployment.artifacts.is_none() {
                deployment = self
                    .published
                    .wait(deployment)
                    .expect("artifact store poisoned");
            }
            deployment.artifacts.clone().expect("published above")
        }

        /// Look up the vmexe for a specific program. Returns `None` if the
        /// program is not part of the published deployment.
        pub fn vmexe(&self, program: &ProgramRef) -> Option<Arc<Exe>> {
            let deployment = self.lock();
            deployment
                .artifacts
                .as_ref()?
                .programs
                .get(program)
                .cloned()
        }

        /// The configured loadout, even if loading is not yet complete.
        /// Used by `/register_worker` to advertise what this worker is
        /// supposed to serve.
        pub fn configured_programs(&self) -> Vec<ProgramRef> {
            self.lock().programs.clone()
        }

        /// True iff every shared key plus every per-program vmexe loaded and
        /// no registration is still preparing. This does **not** require EVM
        /// artifacts to be present — a worker that only serves stark proofs is
        /// "ready" even with `evm: None`.
        ///
        /// Every registered program must also be published, which is what
        /// holds `/start_proof` back while an AOT compile runs and after one
        /// has failed, since a program's verifying key goes out before its
        /// compile ends.
        pub fn is_ready(&self) -> bool {
            let deployment = self.lock();
            deployment
                .registered
                .keys()
                .all(|program| deployment.programs.contains(program))
                && deployment
                    .artifacts
                    .as_ref()
                    .is_some_and(|a| a.programs.len() == deployment.programs.len())
        }

        /// The app-worker context for the published deployment.
        pub fn app_worker_context(&self) -> Option<Arc<AppWorkerContext>> {
            self.lock().app_ctx.clone()
        }

        /// Block until an app-worker context is published. App worker threads
        /// on a registration-driven worker park here at startup.
        pub fn wait_for_app_worker_context(&self) -> Arc<AppWorkerContext> {
            let mut deployment = self.lock();
            if deployment.app_ctx.is_none() {
                info!("Waiting for a deployment before serving app jobs");
            }
            while deployment.app_ctx.is_none() {
                deployment = self
                    .published
                    .wait(deployment)
                    .expect("artifact store poisoned");
            }
            deployment.app_ctx.clone().expect("published above")
        }

        /// Publish the context built at boot from the disk-seeded loadout, so
        /// a later registration extends it instead of replacing it.
        pub fn publish_app_worker_context(&self, app_ctx: Arc<AppWorkerContext>) {
            self.lock().app_ctx = Some(app_ctx);
            self.published.notify_all();
        }

        /// Classify a registration against the published deployment and, when
        /// it is new, reserve the program so a concurrent duplicate does not
        /// start a second preparation.
        ///
        /// `vm_config` is the canonical JSON of the parsed config, so the
        /// pinned-config comparison is a byte comparison.
        pub fn begin_registration(
            &self,
            program: &ProgramRef,
            elf: &[u8],
            vm_config: &str,
        ) -> RegistrationOutcome {
            let mut deployment = self.lock();

            if deployment
                .vm_config
                .as_deref()
                .is_some_and(|pinned| pinned != vm_config)
            {
                return RegistrationOutcome::Conflict(format!(
                    "worker is pinned to a different vm_config; restart it to serve {program}"
                ));
            }
            // A deferral keyset carries hook AIRs that cannot be re-derived
            // from `vm_config` alone, so registration cannot produce keys or a
            // baseline that match what this worker proves with.
            if deployment
                .artifacts
                .as_ref()
                .is_some_and(|a| a.is_deferral_deployment())
            {
                return RegistrationOutcome::Conflict(
                    "worker booted a deferral keyset, which registration cannot derive; \
                     restart it with enable_deferral off to accept registrations"
                        .to_string(),
                );
            }

            let outcome = match deployment.registered.get(program) {
                Some(existing) if existing.elf != elf => RegistrationOutcome::Conflict(format!(
                    "{program} is registered with different ELF bytes; use a new version"
                )),
                // A record left behind by a failed preparation is retried
                // rather than reported as already registered.
                Some(existing) if !existing.failed => RegistrationOutcome::AlreadyRegistered,
                _ if deployment.programs.contains(program) => RegistrationOutcome::Conflict(
                    format!("{program} is already served from disk artifacts"),
                ),
                _ => {
                    deployment.registered.insert(
                        program.clone(),
                        RegisteredProgram {
                            elf: elf.to_vec(),
                            baseline: None,
                            failed: false,
                        },
                    );
                    RegistrationOutcome::Accepted
                }
            };

            // Pin the config the moment a program is reserved, so the long AOT
            // compile that follows cannot be overlapped by a registration this
            // worker's keyset will never match.
            if matches!(outcome, RegistrationOutcome::Accepted) {
                deployment
                    .vm_config
                    .get_or_insert_with(|| vm_config.to_string());
            }
            outcome
        }

        /// Drop a reservation that never reported anything, so the worker is
        /// left exactly as it was before the registration arrived.
        ///
        /// This is the derivation's failure path. The response carries the
        /// failure, so the manager rolls the program out of its loadout and
        /// never replays it, and a record left behind would hold the worker out
        /// of the ready set with nothing able to clear it.
        pub fn release_registration(&self, program: &ProgramRef) {
            let mut deployment = self.lock();
            deployment.registered.remove(program);
            // The pin exists to keep a second config away from a keyset this
            // worker is committed to. Nothing is committed once the last
            // reservation is gone and nothing has been published.
            if deployment.artifacts.is_none() && deployment.registered.is_empty() {
                deployment.vm_config = None;
            }
            drop(deployment);
            self.published.notify_all();
        }

        /// Mark a reservation whose AOT compile failed. The record stays so
        /// `is_ready` keeps reporting the worker unable to serve the program,
        /// and the manager's replay of the program it still holds retries it.
        pub fn fail_registration(&self, program: &ProgramRef) {
            if let Some(entry) = self.lock().registered.get_mut(program) {
                entry.failed = true;
            }
            self.published.notify_all();
        }

        /// Record a program's verification baseline as soon as it is derived,
        /// ahead of the AOT compile, so `/register_program` can answer with it.
        pub fn record_baseline(&self, program: &ProgramRef, baseline: VerificationBaseline) {
            self.lock()
                .registered
                .get_mut(program)
                .expect("program was reserved by begin_registration")
                .baseline = Some(baseline);
            self.published.notify_all();
        }

        /// Block until `program`'s reservation has derived a baseline, then
        /// return it. `None` once the preparation failed, and for a program
        /// this worker holds no reservation for.
        ///
        /// A registration that arrives while an identical one is still
        /// deriving waits here rather than answering with no key, so the
        /// manager never has to treat a worker as having declined to report.
        pub fn wait_for_baseline(&self, program: &ProgramRef) -> Option<VerificationBaseline> {
            let mut deployment = self.lock();
            loop {
                let entry = deployment.registered.get(program)?;
                if entry.failed {
                    return None;
                }
                if let Some(baseline) = &entry.baseline {
                    return Some(baseline.clone());
                }
                deployment = self
                    .published
                    .wait(deployment)
                    .expect("artifact store poisoned");
            }
        }

        /// Install a prepared program's artifacts. Installs the keyset if
        /// this is the first deployment, adds the program's vmexe and
        /// execution instances, and wakes every parked prover thread. The
        /// program is not servable until [`Self::publish_registration`], so
        /// `is_ready` keeps holding `/start_proof` back while the app
        /// provers preload it.
        pub fn install_registration(&self, prepared: PreparedProgram) {
            let PreparedProgram {
                derived:
                    DerivedProgram {
                        program,
                        vm_config,
                        app_pk,
                        agg_stark_pk,
                        exe,
                        // Already recorded by `record_baseline`.
                        baseline: _,
                    },
                execution_instances,
            } = prepared;

            let mut deployment = self.lock();

            let artifacts = match &deployment.artifacts {
                Some(existing) => existing.with_program(program.clone(), exe),
                None => EdgeArtifacts {
                    app_pk,
                    agg_stark_pk,
                    programs: HashMap::from([(program.clone(), exe)]),
                    #[cfg(feature = "evm-prove")]
                    evm: None,
                    deferral: None,
                },
            };

            let mut instances = deployment
                .app_ctx
                .as_ref()
                .map(|ctx| ctx.execution_instances.clone())
                .unwrap_or_default();
            instances.insert(program.clone(), execution_instances);

            deployment.app_ctx = Some(Arc::new(AppWorkerContext {
                app_pk: artifacts.app_pk.clone(),
                execution_instances: instances,
            }));
            deployment.artifacts = Some(Arc::new(artifacts));
            deployment.vm_config = Some(vm_config);

            info!("Installed artifacts for {program}");
            self.published.notify_all();
        }

        /// Mark an installed program servable, completing its registration.
        /// From here `is_ready` counts it and `/readyz` lists it, which is
        /// what lets the manager release its key on `/program_vk`.
        pub fn publish_registration(&self, program: &ProgramRef) {
            let mut deployment = self.lock();
            deployment.programs.push(program.clone());

            info!(
                "Published {program}; worker now serves {} program(s)",
                deployment.programs.len()
            );
            self.published.notify_all();
        }
    }
}

#[cfg(not(feature = "mock-provers"))]
pub use real_artifacts::*;

// ============================================================================
// Mock prover types (only when mock-provers feature is enabled)
// ============================================================================

#[cfg(feature = "mock-provers")]
mod mock_artifacts {
    use super::*;

    use std::sync::Mutex;

    /// Store for proving artifacts (mock mode).
    pub struct ArtifactStore {
        /// Mock builds derive no keys, so a registration only records the
        /// program here. Behind a mutex because it grows at runtime.
        configured_programs: Mutex<Vec<ProgramRef>>,
        #[allow(dead_code)]
        artifacts_path: PathBuf,
    }

    impl ArtifactStore {
        /// Initialize the global artifact store. Idempotent in mock mode:
        /// tests routinely spawn multiple workers in the same process, and
        /// a second `init` should be a no-op (first config wins) rather
        /// than an error.
        pub fn init(config: &ArtifactsConfig, programs: Vec<ProgramRef>) -> Result<()> {
            if ARTIFACT_STORE.get().is_some() {
                return Ok(());
            }
            let store = Self::new(config, programs)?;
            let _ = ARTIFACT_STORE.set(Arc::new(store));
            Ok(())
        }

        pub fn global() -> Option<Arc<ArtifactStore>> {
            ARTIFACT_STORE.get().cloned()
        }

        fn new(config: &ArtifactsConfig, programs: Vec<ProgramRef>) -> Result<Self> {
            let artifacts_path = config
                .artifacts_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("/data/artifacts"));

            info!(
                "Initializing artifact store (mock mode): path={}, programs={}",
                artifacts_path.display(),
                programs.len()
            );

            Ok(Self {
                configured_programs: Mutex::new(programs),
                artifacts_path,
            })
        }

        pub fn configured_programs(&self) -> Vec<ProgramRef> {
            self.lock().clone()
        }

        /// Add a registered program to the advertised loadout. Idempotent, so
        /// a repeat registration is a no-op.
        pub fn record_registration(&self, program: ProgramRef) {
            let mut programs = self.lock();
            if !programs.contains(&program) {
                programs.push(program);
            }
        }

        fn lock(&self) -> std::sync::MutexGuard<'_, Vec<ProgramRef>> {
            self.configured_programs
                .lock()
                .expect("artifact store poisoned")
        }

        /// Always ready in mock mode.
        pub fn is_ready(&self) -> bool {
            true
        }
    }
}

#[cfg(feature = "mock-provers")]
pub use mock_artifacts::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_path() {
        let config = ArtifactsConfig {
            artifacts_path: Some(PathBuf::from("/data/artifacts")),
            halo2_pk_path: None,
            enable_deferral: false,
        };
        // Just verify config is valid - ArtifactStore::init requires static OnceCell
        assert_eq!(
            config.artifacts_path,
            Some(PathBuf::from("/data/artifacts"))
        );
    }

    #[test]
    #[cfg(feature = "mock-provers")]
    fn test_mock_artifacts_ready() {
        // In mock mode, artifact store readiness is exercised via the `global()`
        // accessor in integration tests; this asserts the module compiles under
        // the mock feature.
        assert!(ArtifactStore::global().is_none());
    }
}
