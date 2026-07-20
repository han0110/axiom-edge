//! Worker configuration.

use eyre::{bail, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Worker configuration loaded from TOML.
///
/// Note: CUDA runtime tuning (VPMM_PAGE_SIZE, VPMM_PAGES) is read by the
/// upstream `openvm-cuda-common` dependency from environment variables. Those
/// values live in `config/defaults.toml` under `[cuda]`; `start-provers`
/// renders them as container env vars (via the compose template) at deploy
/// time - the worker binary does not read or set them.
#[derive(Debug, Deserialize, Clone)]
pub struct WorkerConfig {
    pub server: ServerConfig,
    pub worker: WorkerSettings,
    pub artifacts: ArtifactsConfig,
    pub provers: ProversConfig,
    pub telemetry: TelemetryConfig,
}

/// HTTP server configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    /// HTTP listen address (e.g., "0.0.0.0:8001")
    pub listen_addr: String,
}

/// Worker configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct WorkerSettings {
    /// Worker ID (0-indexed, must be unique per worker)
    pub prover_id: usize,
    /// Total number of workers in the pool
    pub num_provers: usize,
    /// This worker's URL for manager registration
    pub worker_url: Option<String>,
    /// Manager URL for registration and result submission
    pub manager_url: String,
    /// Deployment role this worker plays, reported to the manager at
    /// registration. Defaults to [`WorkerRole::Full`] (today's behavior) when
    /// absent from `worker.toml`. Currently **inert** — nothing branches on it
    /// yet; it exists for the opt-in dedicated-halo2 deployment mode.
    #[serde(default)]
    pub worker_role: protocol::WorkerRole,
}

/// Artifacts configuration.
///
/// Only the base path lives here. The `EDGE_PROGRAMS` env var, parsed by
/// `protocol::parse_programs_env`, optionally seeds the deployment's program
/// list, and `start-provers.py` renders it onto both manager and worker
/// containers.
#[derive(Debug, Deserialize, Clone)]
pub struct ArtifactsConfig {
    /// Base path for artifact files. Layout:
    /// - `{artifacts_path}/app_pk` (shared, deployment-level)
    /// - `{artifacts_path}/agg_stark_pk` (shared, deployment-level)
    /// - `{artifacts_path}/root_pk` (shared, deployment-level; only with `evm-prove`)
    /// - `{artifacts_path}/programs/{name}/{version}/program.vmexe`
    pub artifacts_path: Option<PathBuf>,

    /// Directory holding the halo2 proving key (`halo2_pk`) plus its two SRS
    /// files (`kzg_bn254_<verifier_k>.srs`, `kzg_bn254_<wrapper_k>.srs`).
    /// Produced by the offline `halo2-keygen` binary. Separate from
    /// `artifacts_path` so the >10GB pk can live on its own read-only mount.
    /// Only meaningful when built with the `evm-prove` feature; absent in that
    /// build leaves the worker stark-only-ready.
    #[serde(default)]
    pub halo2_pk_path: Option<PathBuf>,

    /// Enable deferral mode (verify-stark / proof-of-proof). When `true`, the
    /// worker loads the deferral-aware `SdkCachedProvingKey` from the
    /// conventional location `{artifacts_path}/deferral/cached_pk`
    #[serde(default)]
    pub enable_deferral: bool,
}

/// Prover thread pool configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct ProversConfig {
    /// Number of GPU app prover instances pre-loaded at startup. Also the
    /// maximum app-prove parallelism a single proof uses on this worker.
    #[serde(default = "default_app_provers")]
    pub max_app_provers: usize,
    /// Number of LeafProver threads
    #[serde(default = "default_leaf_provers")]
    pub max_leaf_provers: usize,
    /// Number of InternalProver threads
    #[serde(default = "default_internal_provers")]
    pub max_internal_provers: usize,
    /// Number of RootProver threads (in-process EVM prove; runs after
    /// the final internal proof of an Evm-typed proof). Only meaningful when
    /// the worker is built with `evm-prove` or `mock-provers`. One root
    /// thread is plenty since root prove is 1→1 with the final stark proof.
    #[serde(default = "default_root_provers")]
    pub max_root_provers: usize,
    /// Number of Halo2Prover threads (in-process EVM prove; runs after
    /// root prove). Same scope notes as `max_root_provers`.
    #[serde(default = "default_halo2_provers")]
    pub max_halo2_provers: usize,
    /// Default VM max memory applied when a prove request does not set `segment_memory`.
    #[serde(default)]
    pub default_segment_memory: Option<usize>,
}

impl Default for ProversConfig {
    fn default() -> Self {
        Self {
            max_app_provers: default_app_provers(),
            max_leaf_provers: default_leaf_provers(),
            max_internal_provers: default_internal_provers(),
            max_root_provers: default_root_provers(),
            max_halo2_provers: default_halo2_provers(),
            default_segment_memory: None,
        }
    }
}

fn default_app_provers() -> usize {
    2
}

fn default_leaf_provers() -> usize {
    2
}

fn default_internal_provers() -> usize {
    1
}

fn default_root_provers() -> usize {
    1
}

fn default_halo2_provers() -> usize {
    1
}

/// Telemetry configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct TelemetryConfig {
    /// Log level (trace, debug, info, warn, error)
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// OpenTelemetry collector endpoint (optional)
    pub otlp_endpoint: Option<String>,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            otlp_endpoint: None,
        }
    }
}

fn default_log_level() -> String {
    "info".to_string()
}

impl WorkerConfig {
    /// Load configuration from a TOML file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration.
    fn validate(&self) -> Result<()> {
        if self.worker.prover_id >= self.worker.num_provers {
            bail!(
                "worker.prover_id ({}) must be < worker.num_provers ({})",
                self.worker.prover_id,
                self.worker.num_provers
            );
        }
        // Prover-capacity requirements are role-aware. `Full` (the default) and
        // `StarkOnly` both run app/leaf/internal, so they require app-prover
        // capacity exactly as before (this branch is byte-for-byte the old
        // check). `EvmDedicated` runs only the EVM step (root → halo2) and is
        // assigned no app/leaf/internal work, so it is permitted zero app
        // provers but must instead carry root + halo2 capacity.
        match self.worker.worker_role {
            protocol::WorkerRole::Full | protocol::WorkerRole::StarkOnly => {
                if self.provers.max_app_provers == 0 {
                    bail!("provers.max_app_provers must be > 0");
                }
            }
            protocol::WorkerRole::EvmDedicated => {
                if self.provers.max_root_provers == 0 || self.provers.max_halo2_provers == 0 {
                    bail!(
                        "EvmDedicated worker requires root + halo2 capacity: \
                         provers.max_root_provers ({}) and provers.max_halo2_provers ({}) \
                         must both be > 0",
                        self.provers.max_root_provers,
                        self.provers.max_halo2_provers,
                    );
                }
                // This role runs ONLY the EVM step (root + halo2) and is
                // assigned no app/leaf/internal work — the prover pool builds
                // none — so it must report zero app/leaf/internal capacity.
                // Reject a stray non-zero value rather than silently rewriting
                // it, so `worker.toml` and what the worker actually builds stay
                // in lockstep.
                if self.provers.max_app_provers != 0
                    || self.provers.max_leaf_provers != 0
                    || self.provers.max_internal_provers != 0
                {
                    bail!(
                        "EvmDedicated worker must set app/leaf/internal capacity to 0 \
                         (got app={}, leaf={}, internal={}); this role runs only root + halo2",
                        self.provers.max_app_provers,
                        self.provers.max_leaf_provers,
                        self.provers.max_internal_provers,
                    );
                }
            }
        }
        Ok(())
    }

    /// Derive worker_url from listen_addr if not explicitly set.
    /// Converts 0.0.0.0 bind address to 127.0.0.1 for reachability.
    pub fn effective_worker_url(&self) -> String {
        self.worker.worker_url.clone().unwrap_or_else(|| {
            // If listen_addr uses 0.0.0.0 (bind all), convert to 127.0.0.1 for the worker URL
            // since 0.0.0.0 is not routable from other hosts
            let addr = self.server.listen_addr.replace("0.0.0.0", "127.0.0.1");
            format!("http://{}", addr)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_config() {
        let config_content = r#"
[server]
listen_addr = "0.0.0.0:8001"

[worker]
prover_id = 0
num_provers = 4
manager_url = "http://localhost:3000"

[artifacts]

[provers]
max_app_provers = 4
max_leaf_provers = 2
max_internal_provers = 2

[telemetry]
log_level = "info"
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let config = WorkerConfig::load(file.path()).unwrap();
        assert_eq!(config.server.listen_addr, "0.0.0.0:8001");
        assert_eq!(config.worker.prover_id, 0);
        assert_eq!(config.worker.num_provers, 4);
        assert_eq!(config.provers.max_app_provers, 4);
        // No `worker_role` in the TOML above ⇒ defaults to Full (today's behavior).
        assert_eq!(config.worker.worker_role, protocol::WorkerRole::Full);
    }

    #[test]
    fn test_worker_role_parses_from_toml() {
        let config_content = r#"
[server]
listen_addr = "0.0.0.0:8001"

[worker]
prover_id = 0
num_provers = 4
manager_url = "http://localhost:3000"
worker_role = "evm_dedicated"

[artifacts]

[provers]
max_app_provers = 0
max_leaf_provers = 0
max_internal_provers = 0

[telemetry]
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let config = WorkerConfig::load(file.path()).unwrap();
        assert_eq!(
            config.worker.worker_role,
            protocol::WorkerRole::EvmDedicated
        );
    }

    #[test]
    fn test_evm_dedicated_permits_zero_app_provers() {
        // An honest EvmDedicated worker runs only root + halo2 and is assigned
        // no app/leaf/internal work, so app/leaf/internal = 0 must validate
        // (the `> 0` reject that fires for Full/StarkOnly must not fire here).
        // Root/halo2 capacity default to 1, satisfying the role's requirement.
        let config_content = r#"
[server]
listen_addr = "0.0.0.0:8001"

[worker]
prover_id = 0
num_provers = 4
manager_url = "http://localhost:3000"
worker_role = "evm_dedicated"

[artifacts]

[provers]
max_app_provers = 0
max_leaf_provers = 0
max_internal_provers = 0

[telemetry]
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let config = WorkerConfig::load(file.path()).unwrap();
        assert_eq!(config.provers.max_app_provers, 0);
        assert_eq!(
            config.worker.worker_role,
            protocol::WorkerRole::EvmDedicated
        );
    }

    #[test]
    fn test_evm_dedicated_rejects_nonzero_app_leaf_internal() {
        // A worker.toml that leaves app/leaf/internal non-zero for an
        // EvmDedicated worker is rejected at load — this role runs only
        // root+halo2, so the config must report 0/0/0. We fail loudly rather
        // than silently rewriting the values, keeping `worker.toml` honest
        // about what the worker builds.
        let config_content = r#"
[server]
listen_addr = "0.0.0.0:8001"

[worker]
prover_id = 3
num_provers = 4
manager_url = "http://localhost:3000"
worker_role = "evm_dedicated"

[artifacts]

[provers]
max_app_provers = 4
max_leaf_provers = 2
max_internal_provers = 1

[telemetry]
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let err = WorkerConfig::load(file.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("EvmDedicated worker must set app/leaf/internal capacity to 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_evm_dedicated_requires_root_and_halo2_capacity() {
        // With zero app provers *and* zero root capacity, an EvmDedicated
        // worker has no way to run its EVM step — reject it.
        let config_content = r#"
[server]
listen_addr = "0.0.0.0:8001"

[worker]
prover_id = 0
num_provers = 4
manager_url = "http://localhost:3000"
worker_role = "evm_dedicated"

[artifacts]

[provers]
max_app_provers = 0
max_root_provers = 0

[telemetry]
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let err = WorkerConfig::load(file.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("EvmDedicated worker requires root + halo2 capacity"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_stark_only_validates_without_halo2_and_requires_app_provers() {
        // A StarkOnly worker runs app/leaf/internal only and never loads the halo2
        // key, so a config with no `halo2_pk_path` validates. It still requires
        // app-prover capacity (same as Full).
        let config_content = r#"
[server]
listen_addr = "0.0.0.0:8001"

[worker]
prover_id = 0
num_provers = 4
manager_url = "http://localhost:3000"
worker_role = "stark_only"

[artifacts]

[provers]
max_app_provers = 2

[telemetry]
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let config = WorkerConfig::load(file.path()).unwrap();
        assert_eq!(config.worker.worker_role, protocol::WorkerRole::StarkOnly);
        assert!(config.artifacts.halo2_pk_path.is_none());
        assert_eq!(config.provers.max_app_provers, 2);
    }

    #[test]
    fn test_stark_only_rejects_zero_app_provers() {
        // StarkOnly keeps Full's app-capacity requirement.
        let config_content = r#"
[server]
listen_addr = "0.0.0.0:8001"

[worker]
prover_id = 0
num_provers = 4
manager_url = "http://localhost:3000"
worker_role = "stark_only"

[artifacts]

[provers]
max_app_provers = 0

[telemetry]
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let err = WorkerConfig::load(file.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("provers.max_app_provers must be > 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_full_rejects_zero_app_provers_unchanged() {
        // Full (the default role) validates exactly as before: zero app
        // provers is rejected with the original message.
        let config_content = r#"
[server]
listen_addr = "0.0.0.0:8001"

[worker]
prover_id = 0
num_provers = 4
manager_url = "http://localhost:3000"

[artifacts]

[provers]
max_app_provers = 0

[telemetry]
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let err = WorkerConfig::load(file.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("provers.max_app_provers must be > 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_invalid_prover_id() {
        let config_content = r#"
[server]
listen_addr = "0.0.0.0:8001"

[worker]
prover_id = 5
num_provers = 4
manager_url = "http://localhost:3000"

[artifacts]

[provers]

[telemetry]
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        assert!(WorkerConfig::load(file.path()).is_err());
    }
}
