//! Manager configuration.

use eyre::{bail, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Manager configuration loaded from TOML.
#[derive(Debug, Deserialize, Clone)]
pub struct ManagerConfig {
    pub server: ServerConfig,
    pub proof: ProofConfig,
    pub provers: ProversConfig,
    #[serde(default)]
    pub lifecycle: LifecycleConfig,
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub metrics: MetricsConfig,
}

/// HTTP server configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    /// HTTP listen address (e.g., "0.0.0.0:3000")
    pub listen_addr: String,
    /// Total number of workers expected in the proving stack.
    /// Manager gates `/readyz` and `start_proof` on full registration.
    pub num_workers: usize,
    /// Directory of the pre-provisioned artifacts export mounted read-only into
    /// the container. `GET /vk/{name}` serves per-program verifying-key blobs
    /// verbatim from its `vk/` subdir. When unset, defaults to the standard
    /// `--from-artifacts` container mount `/data/artifacts`; a deployment
    /// without such files simply answers `404` there.
    #[serde(default)]
    pub artifacts_path: Option<PathBuf>,
}

/// Per-worker prover capacity expected by every worker in the stack.
///
/// This is the manager's view of capacity. Each worker reports its own
/// configured capacity at registration; the manager rejects mismatches so
/// drift between manager and worker config surfaces immediately. The
/// templated deploy (`scripts/dev/start-provers.py`) renders both
/// `manager.toml` and `worker.toml` from the same args, so values are
/// expected to match.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
pub struct ProversConfig {
    /// Number of GPU app prover instances per worker. Also the maximum
    /// app-prove parallelism a single proof uses on each worker.
    pub max_app_provers: usize,
    /// Number of concurrent leaf proofs each worker can run.
    /// Used by the scheduler when assigning leaf work.
    pub max_leaf_provers: usize,
    /// Number of concurrent internal proofs each worker can run.
    pub max_internal_provers: usize,
}

/// Proof management configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct ProofConfig {
    /// Proof timeout in seconds
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// Max leaf proofs before disabling packing optimization
    #[serde(default = "default_leaf_pack_threshold")]
    pub leaf_pack_threshold: usize,

    /// Optional directory for final proof persistence.
    #[serde(default)]
    pub persist_final_proofs_dir: Option<PathBuf>,

    /// When true, zstd-compress persisted final proof payloads before writing.
    #[serde(default)]
    pub compress_persisted_final_proofs: bool,

    /// Optional directory where app proofs are snapshotted when a leaf prover
    /// fails with the known logup nonzero-root-sum error.
    #[serde(default)]
    pub persist_leaf_failure_app_proofs_dir: Option<PathBuf>,

    /// Leaf-circuit fan-in: number of app proofs aggregated into one leaf
    /// proof. Bounded above by the SDK's `MAX_NUM_CHILDREN_LEAF`.
    #[serde(default = "default_leaf_arity")]
    pub leaf_arity: usize,

    /// Internal-circuit fan-in: number of child proofs aggregated into one
    /// internal proof. Bounded above by the SDK's `MAX_NUM_CHILDREN_INTERNAL`.
    #[serde(default = "default_internal_arity")]
    pub internal_arity: usize,
}

/// Metrics export configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct MetricsConfig {
    /// Directory where per-proof metrics reports (.md) are written.
    #[serde(default = "default_metrics_output_dir")]
    pub output_dir: PathBuf,

    /// OTLP HTTP endpoint for metrics export. When None, metrics are silently dropped.
    #[serde(default)]
    pub endpoint: Option<String>,

    /// API key sent as X-API-Key header on metrics uploads.
    #[serde(default)]
    pub api_key: Option<String>,
}

/// Proof lifecycle webhook configuration.
///
/// Generic, destination-agnostic: when `webhook_url` is set, the manager
/// POSTs a small JSON event on each proof transition (queued / proving /
/// completed). External integrations (e.g. a reporter sidecar)
/// consume these and translate to their own APIs. The manager itself knows
/// nothing about any specific downstream.
#[derive(Debug, Deserialize, Clone)]
pub struct LifecycleConfig {
    /// Optional webhook URL. When unset, no lifecycle events are emitted.
    #[serde(default)]
    pub webhook_url: Option<String>,

    /// Request timeout for webhook POSTs, in milliseconds.
    #[serde(default = "default_lifecycle_timeout_ms")]
    pub timeout_ms: u64,
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

fn default_timeout_secs() -> u64 {
    300
}

fn default_leaf_pack_threshold() -> usize {
    48
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_lifecycle_timeout_ms() -> u64 {
    6000
}

fn default_leaf_arity() -> usize {
    4
}

fn default_internal_arity() -> usize {
    3
}

fn default_metrics_output_dir() -> PathBuf {
    PathBuf::from("/data/metrics")
}

impl Default for ProofConfig {
    fn default() -> Self {
        Self {
            timeout_secs: default_timeout_secs(),
            leaf_pack_threshold: default_leaf_pack_threshold(),
            persist_final_proofs_dir: None,
            compress_persisted_final_proofs: false,
            persist_leaf_failure_app_proofs_dir: None,
            leaf_arity: default_leaf_arity(),
            internal_arity: default_internal_arity(),
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            output_dir: default_metrics_output_dir(),
            endpoint: None,
            api_key: None,
        }
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            otlp_endpoint: None,
        }
    }
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            webhook_url: None,
            timeout_ms: default_lifecycle_timeout_ms(),
        }
    }
}

impl ManagerConfig {
    /// Load configuration from a TOML file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration.
    fn validate(&self) -> Result<()> {
        if self.server.num_workers == 0 {
            bail!("server.num_workers must be > 0");
        }
        if self.provers.max_app_provers == 0 {
            bail!("provers.max_app_provers must be > 0");
        }
        if self.provers.max_leaf_provers == 0 {
            bail!("provers.max_leaf_provers must be > 0");
        }
        if self.provers.max_internal_provers == 0 {
            bail!("provers.max_internal_provers must be > 0");
        }
        if self.proof.timeout_secs == 0 {
            bail!("proof.timeout_secs must be > 0");
        }
        if self.proof.leaf_pack_threshold == 0 {
            bail!("proof.leaf_pack_threshold must be > 0");
        }
        if self.proof.leaf_arity == 0 {
            bail!("proof.leaf_arity must be > 0");
        }
        if self.proof.internal_arity == 0 {
            bail!("proof.internal_arity must be > 0");
        }
        if let Some(dir) = &self.proof.persist_final_proofs_dir {
            if dir.as_os_str().is_empty() {
                bail!("proof.persist_final_proofs_dir must not be empty");
            }
        }
        if self.proof.compress_persisted_final_proofs
            && self.proof.persist_final_proofs_dir.is_none()
        {
            bail!("proof.compress_persisted_final_proofs requires proof.persist_final_proofs_dir");
        }
        if let Some(dir) = &self.proof.persist_leaf_failure_app_proofs_dir {
            if dir.as_os_str().is_empty() {
                bail!("proof.persist_leaf_failure_app_proofs_dir must not be empty");
            }
        }
        if let Some(url) = self.lifecycle.webhook_url.as_ref() {
            if url.trim().is_empty() {
                bail!("lifecycle.webhook_url must not be empty when set");
            }
            if self.lifecycle.timeout_ms == 0 {
                bail!("lifecycle.timeout_ms must be > 0");
            }
        }
        Ok(())
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
listen_addr = "0.0.0.0:3000"
num_workers = 4

[provers]
max_app_provers = 2
max_leaf_provers = 2
max_internal_provers = 1

[proof]
timeout_secs = 3600
leaf_pack_threshold = 48
persist_final_proofs_dir = "/tmp/edge-final-proofs"
compress_persisted_final_proofs = true
persist_leaf_failure_app_proofs_dir = "/tmp/edge-leaf-failure-app-proofs"

[lifecycle]
webhook_url = "http://127.0.0.1:9100/events"
timeout_ms = 9000

[telemetry]
log_level = "info"
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let config = ManagerConfig::load(file.path()).unwrap();
        assert_eq!(config.server.listen_addr, "0.0.0.0:3000");
        assert_eq!(config.proof.timeout_secs, 3600);
        assert_eq!(config.proof.leaf_pack_threshold, 48);
        assert_eq!(
            config.proof.persist_final_proofs_dir,
            Some(PathBuf::from("/tmp/edge-final-proofs"))
        );
        assert!(config.proof.compress_persisted_final_proofs);
        assert_eq!(
            config.proof.persist_leaf_failure_app_proofs_dir,
            Some(PathBuf::from("/tmp/edge-leaf-failure-app-proofs"))
        );
        assert_eq!(
            config.lifecycle.webhook_url.as_deref(),
            Some("http://127.0.0.1:9100/events")
        );
        assert_eq!(config.lifecycle.timeout_ms, 9000);
        assert_eq!(config.telemetry.log_level, "info");
    }

    /// The manager configs shipped under `config/testing/` are referenced by
    /// the mock compose stack (`docker/docker-compose.mock.yml`) and local
    /// runs (the `--config` default). Guard that they actually parse — a
    /// required field added to `ManagerConfig` without updating them breaks
    /// the no-GPU quickstart at container startup, which CI otherwise never
    /// exercises.
    #[test]
    fn test_shipped_testing_manager_configs_parse() {
        let config_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../config/testing");
        for name in ["manager.toml", "docker-manager.toml"] {
            let path = config_dir.join(name);
            let config = ManagerConfig::load(&path)
                .unwrap_or_else(|e| panic!("config/testing/{name} failed to load: {e}"));
            // The mock compose starts 4 workers; both shipped configs assume
            // that stack shape.
            assert_eq!(config.server.num_workers, 4, "{name}");
        }
    }

    #[test]
    fn test_config_defaults() {
        let config_content = r#"
[server]
listen_addr = "0.0.0.0:3000"
num_workers = 4

[provers]
max_app_provers = 2
max_leaf_provers = 2
max_internal_provers = 1

[proof]

[telemetry]
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let config = ManagerConfig::load(file.path()).unwrap();
        assert_eq!(config.proof.timeout_secs, 300);
        assert_eq!(config.proof.leaf_pack_threshold, 48);
        assert_eq!(config.proof.persist_final_proofs_dir, None);
        assert!(!config.proof.compress_persisted_final_proofs);
        assert_eq!(config.proof.persist_leaf_failure_app_proofs_dir, None);
        assert_eq!(config.lifecycle.webhook_url, None);
        assert_eq!(config.lifecycle.timeout_ms, 6000);
        assert_eq!(config.telemetry.log_level, "info");
    }

    #[test]
    fn test_config_rejects_compressed_final_proofs_without_output_dir() {
        let config_content = r#"
[server]
listen_addr = "0.0.0.0:3000"
num_workers = 4

[provers]
max_app_provers = 2
max_leaf_provers = 2
max_internal_provers = 1

[proof]
compress_persisted_final_proofs = true

[telemetry]
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let err = ManagerConfig::load(file.path()).unwrap_err();
        assert!(err.to_string().contains(
            "proof.compress_persisted_final_proofs requires proof.persist_final_proofs_dir"
        ));
    }

    #[test]
    fn test_config_rejects_empty_lifecycle_webhook_url() {
        let config_content = r#"
[server]
listen_addr = "0.0.0.0:3000"
num_workers = 4

[provers]
max_app_provers = 2
max_leaf_provers = 2
max_internal_provers = 1

[proof]

[lifecycle]
webhook_url = "   "

[telemetry]
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(config_content.as_bytes()).unwrap();

        let err = ManagerConfig::load(file.path()).unwrap_err();
        assert!(err
            .to_string()
            .contains("lifecycle.webhook_url must not be empty when set"));
    }
}
