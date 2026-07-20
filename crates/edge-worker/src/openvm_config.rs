//! OpenVM SDK config — the single source of truth for the edge's VM config.
//!
//! This config is consumed only by the offline / generation binaries
//! (`keygen`, `convert_fixtures`, `generate_edge_vm_vk`, `verify_edge_final_proof`):
//! `keygen` bakes it into `app_pk`/`agg_stark_pk`, `convert_fixtures` drives the
//! ELF→vmexe transpiler from it. The running manager/worker never re-derive it —
//! they load the config out of `app_pk`. So every binary that builds the config
//! MUST agree, or the on-disk keys won't match what later consumers expect.
//!
//! By default the config is the OpenVM "standard" extension set
//! (<https://github.com/openvm-org/openvm/blob/main/crates/sdk/src/config/openvm_standard.toml>).
//! Setting `EDGE_OPENVM_CONFIG` to a path overrides the *extension set* with a
//! custom `openvm.toml` (`[app_vm_config.*]` sections). The edge's own system
//! invariants (`max_constraint_degree`, `num_public_values`) are re-applied on
//! top regardless of source, since the aggregation/recursion layers depend on
//! them.
//!
//! `start-provers.py --openvm-config-file <toml>` sets `EDGE_OPENVM_CONFIG` on
//! the generation subprocesses; `scripts/ops/generate-vm-vk.sh` must pass the
//! same file so any externally-generated verifying key matches.

use color_eyre::eyre::{Result, WrapErr};
use openvm_sdk_config::SdkVmConfig;
use openvm_stark_sdk::config::{
    app_params_with_100_bits_security, internal_params_with_100_bits_security,
    leaf_params_with_100_bits_security, MAX_APP_LOG_STACKED_HEIGHT,
};
use sdk_v2::{
    config::{AggregationSystemParams, AppConfig, DEFAULT_APP_L_SKIP},
    Sdk,
};
use {
    openvm_sdk_config::deferral::SupportedDeferral,
    openvm_stark_sdk::config::hook_params_with_100_bits_security,
    sdk_v2::openvm_circuit::arch::{instructions::DEFERRAL_AS, DEFAULT_DEFERRAL_ADDR_SPACE_CELLS},
    sdk_v2::prover::DeferralAggProver,
};

/// Env var pointing at a custom OpenVM `app_vm_config` TOML. Unset → standard.
pub const EDGE_OPENVM_CONFIG_ENV: &str = "EDGE_OPENVM_CONFIG";

const VM_MAX_CONSTRAINT_DEGREE: usize = 3;
const VM_NUM_PUBLIC_VALUES: usize = 32;

/// Build the edge's `SdkVmConfig`.
///
/// Base extension set comes from `EDGE_OPENVM_CONFIG` (a custom `openvm.toml`)
/// when set, otherwise [`SdkVmConfig::standard`]. The edge system invariants
/// are then applied on top so they hold regardless of the chosen extensions.
pub fn edge_vm_config() -> SdkVmConfig {
    let mut config = match std::env::var(EDGE_OPENVM_CONFIG_ENV) {
        Ok(path) if !path.trim().is_empty() => {
            let toml = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!("Failed to read {EDGE_OPENVM_CONFIG_ENV} ({path}): {e}")
            });
            SdkVmConfig::from_toml(&toml).unwrap_or_else(|e| {
                panic!("Failed to parse {EDGE_OPENVM_CONFIG_ENV} ({path}) as openvm.toml: {e}")
            })
        }
        _ => SdkVmConfig::standard(),
    };
    // Edge system invariants — applied regardless of the extension source so
    // the toml only customizes extensions, not these.
    config.system.config = config
        .system
        .config
        .with_max_constraint_degree(VM_MAX_CONSTRAINT_DEGREE)
        .with_public_values(VM_NUM_PUBLIC_VALUES);
    config
}

/// App + aggregation params used for every edge SDK build.
pub(crate) fn edge_app_and_agg_params(
) -> (openvm_stark_backend::SystemParams, AggregationSystemParams) {
    let mut app_params = app_params_with_100_bits_security(MAX_APP_LOG_STACKED_HEIGHT);
    app_params.l_skip = DEFAULT_APP_L_SKIP;
    app_params.n_stack = MAX_APP_LOG_STACKED_HEIGHT - DEFAULT_APP_L_SKIP;

    let agg_params = AggregationSystemParams {
        leaf: leaf_params_with_100_bits_security(),
        internal: internal_params_with_100_bits_security(),
    };
    (app_params, agg_params)
}

/// Build the edge `Sdk` (app + aggregation params) from [`edge_vm_config`].
pub fn create_edge_sdk() -> Result<Sdk> {
    let (app_params, agg_params) = edge_app_and_agg_params();
    let app_config = AppConfig::new(edge_vm_config(), app_params);
    Sdk::new(app_config, agg_params).wrap_err("Failed to create axiom-edge SDK config")
}

/// Build the edge `Sdk` configured with a Halo2 KZG params reader rooted at
/// `kzg_params_dir`.
///
/// The reader is consulted lazily by `sdk.halo2_pk()` / `sdk.halo2_prover()` to
/// read `kzg_bn254_<k>.srs` files; the rest of the SDK is identical to
/// [`create_edge_sdk`]. The directory itself is not validated here — missing
/// files surface as a panic from inside the SDK when `halo2_pk()` is invoked.
#[cfg(feature = "evm-prove")]
pub fn create_edge_sdk_for_halo2(kzg_params_dir: &std::path::Path) -> Result<Sdk> {
    let (app_params, agg_params) = edge_app_and_agg_params();
    let app_config = AppConfig::new(edge_vm_config(), app_params);
    Sdk::builder()
        .app_config(app_config)
        .agg_params(agg_params)
        .halo2_params_dir(kzg_params_dir)
        .build()
        .wrap_err("Failed to create axiom-edge SDK (evm-prove) with halo2 params reader")
}

/// Build a **deferral-enabled** edge `Sdk` (verify-stark deferral path).
///
/// Mirrors the openvm `examples/verify-stark/host/src/lib.rs::keygen()` flow:
///   1. construct a `DeferralAggProver::verify_stark(...)` (derives the deferral
///      path's fixed-point `def_hook_commit` from a dummy circuit, so callers
///      don't need a `child_agg_vk`);
///   2. derive the deferral config via
///      `deferral_agg_prover.multi_deferral_circuit_prover.make_config(...)` —
///      its `commit` field must come from the keys, never hand-authored;
///   3. set `vm_config.deferral = Some(deferral_config)` (and re-`.optimize()`
///      to revalidate the `DEFERRAL_AS` address space slot);
///   4. build the SDK via the builder's `.deferral_agg_prover(...)` injection.
pub fn create_edge_sdk_with_deferral() -> Result<Sdk> {
    create_edge_sdk_with_deferral_impl(None)
}

/// Deferral-enabled edge `Sdk` with a Halo2 KZG params reader rooted at
/// `kzg_params_dir` — the deferral counterpart of [`create_edge_sdk_for_halo2`].
///
/// Required for `halo2-keygen --with-deferral`: the halo2 wrapper circuit's
/// fixed `log_heights_per_air` is derived from the ROOT verifier's AIR set,
/// which a deferral deployment extends (the deferral PVs AIR + shifted
/// heights). A halo2_pk built from the non-deferral SDK rejects a deferral
/// root proof at prove time ("per-AIR log heights from proof must match this
/// circuit's fixed log_heights_per_air"). Building over the deferral SDK makes
/// the wrapper circuit match.
#[cfg(feature = "evm-prove")]
pub fn create_edge_sdk_with_deferral_for_halo2(kzg_params_dir: &std::path::Path) -> Result<Sdk> {
    create_edge_sdk_with_deferral_impl(Some(kzg_params_dir))
}

fn create_edge_sdk_with_deferral_impl(kzg_params_dir: Option<&std::path::Path>) -> Result<Sdk> {
    let (app_params, agg_params) = edge_app_and_agg_params();
    let mut vm_config = edge_vm_config();

    let system_config = &vm_config.system.config;
    let memory_dimensions = system_config.memory_config.memory_dimensions();
    let num_user_pvs = system_config.num_public_values;

    let deferral_agg_prover = DeferralAggProver::verify_stark(
        &agg_params,
        hook_params_with_100_bits_security(),
        memory_dimensions,
        num_user_pvs,
    );

    let deferral_config = deferral_agg_prover
        .multi_deferral_circuit_prover
        .make_config(vec![SupportedDeferral::VerifyStark]);
    vm_config.deferral = Some(deferral_config);
    // Restore the DEFERRAL_AS cell count. `edge_vm_config()` derives from
    // `SdkVmConfig::standard()`, which ends in `.optimize()` while `deferral`
    // is still `None` — that pass *zeroes* `DEFERRAL_AS.num_cells` (sdk-config
    // `apply_optimizations`). Re-enabling deferral here does NOT restore it
    // (`apply_optimizations` only asserts the slot exists when deferral is
    // `Some`), so without this the deferral address space has 0 cells and the
    // prover panics with "memory_size=0" the first time it touches DEFERRAL_AS.
    // The canonical openvm builder avoids this by setting `.deferral(..)`
    // before its single `.optimize()`; we patch the count back to the default
    // (matches `MemoryConfig::default()` / the canonical verify-stark path).
    vm_config.system.config.memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells =
        DEFAULT_DEFERRAL_ADDR_SPACE_CELLS;
    vm_config.apply_optimizations();

    let app_config = AppConfig::new(vm_config, app_params);
    // `mut` + the kzg dir are only used under `evm-prove` (the halo2 params
    // reader). A stark-only deferral build never sets a halo2 dir.
    #[cfg_attr(not(feature = "evm-prove"), allow(unused_mut))]
    let mut builder = Sdk::builder()
        .app_config(app_config)
        .agg_params(agg_params)
        .deferral_agg_prover(deferral_agg_prover);
    #[cfg(feature = "evm-prove")]
    if let Some(dir) = kzg_params_dir {
        builder = builder.halo2_params_dir(dir);
    }
    #[cfg(not(feature = "evm-prove"))]
    let _ = kzg_params_dir;
    builder
        .build()
        .wrap_err("Failed to create deferral-enabled axiom-edge SDK")
}

#[cfg(all(test, feature = "evm-prove"))]
mod tests {
    use super::*;
    use sdk_v2::Sdk;

    /// End-to-end round-trip: build a deferral-enabled edge SDK, materialize
    /// app + agg proving keys, persist via `cached_proving_key()`, and
    /// reconstruct via `Sdk::from_deferral_cached_proving_key`.
    ///
    /// **Ignored by default**: `DeferralAggProver::verify_stark` + `app_keygen`
    /// + `agg_pk` are full keygen on the edge standard VM config; CI would
    ///   time out. Run explicitly with `cargo test -p edge-worker
    ///   --features evm-prove deferral_round_trip -- --ignored --nocapture`.
    ///
    /// `root_pk` is intentionally NOT materialized — round-trip correctness
    /// doesn't require it, and skipping it cuts the runtime materially. This
    /// test only verifies that the cached pk can rebuild a deferral-enabled
    /// SDK; root keygen lives in the `keygen` binary's actual run.
    #[test]
    #[ignore]
    fn deferral_round_trip() {
        let sdk = create_edge_sdk_with_deferral().expect("build deferral SDK");
        let initial_commit = sdk
            .def_hook_commit()
            .expect("deferral SDK must expose a def_hook_commit");

        // Force the lazy keygen so the cached_pk actually carries keys.
        let _ = sdk.app_keygen();
        let _ = sdk.agg_pk();

        let cached_pk = sdk
            .cached_proving_key()
            .expect("cached_proving_key after app+agg keygen");
        assert!(
            cached_pk.deferral_pk.is_some(),
            "deferral_pk must be populated on a deferral-enabled SDK",
        );
        assert!(
            cached_pk.deferral_agg_pk.is_some(),
            "deferral_agg_pk must be populated on a deferral-enabled SDK",
        );
        assert!(
            cached_pk.app_pk.app_vm_pk.vm_config.deferral.is_some(),
            "app_pk's vm_config must carry the deferral config",
        );

        let rebuilt = Sdk::from_deferral_cached_proving_key(cached_pk)
            .expect("reconstruct deferral SDK from cached pk");
        let rebuilt_commit = rebuilt
            .def_hook_commit()
            .expect("reconstructed SDK must also expose def_hook_commit");
        assert_eq!(
            initial_commit, rebuilt_commit,
            "def_hook_commit must round-trip through cached_pk",
        );
    }
}
