//! HTTP client for sending proof results back to the manager.

use eyre::Result;
use reqwest::Client;
use std::time::Duration;
use tracing::{error, info, instrument, warn};

use protocol::{MessageEnvelope, ProgramRef, ProofResult, ResultPayload};

/// Client for submitting proof results to the manager.
pub struct ResultClient {
    client: Client,
    manager_url: String,
    worker_id: usize,
}

impl ResultClient {
    /// Get the worker ID.
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }

    /// Create a new result client.
    pub fn new(manager_url: &str, worker_id: usize) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()?;

        Ok(Self {
            client,
            manager_url: manager_url.trim_end_matches('/').to_string(),
            worker_id,
        })
    }

    /// Submit a single proof result to the manager.
    pub async fn submit_single_result(&self, proof_uuid: &str, result: ProofResult) -> Result<()> {
        let url = format!("{}/proof_result", self.manager_url);

        let payload = ResultPayload {
            worker_id: self.worker_id,
            proof_uuid: proof_uuid.to_string(),
            result: MessageEnvelope::with_metadata(result),
        };

        let body = bincode::serialize(&payload)?;

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/octet-stream")
            .body(body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            error!("Failed to submit result: status={}, body={}", status, body);
            return Err(eyre::eyre!(
                "Failed to submit result: status={}, body={}",
                status,
                body
            ));
        }
        Ok(())
    }

    /// Submit proof results to the manager.
    #[instrument(skip(self, results), fields(proof_uuid = %proof_uuid, num_results = results.len()))]
    pub async fn submit_result(&self, proof_uuid: &str, results: Vec<ProofResult>) -> Result<()> {
        for result in results {
            self.submit_single_result(proof_uuid, result).await?;
        }
        info!("Successfully submitted results for proof {}", proof_uuid);
        Ok(())
    }

    /// Submit a proof error to the manager.
    #[instrument(skip(self), fields(proof_uuid = %proof_uuid))]
    pub async fn submit_error(&self, proof_uuid: &str, error_msg: &str) -> Result<()> {
        let url = format!("{}/proof_result", self.manager_url);

        let error_result = ProofResult::Error(protocol::ErrorResult {
            // Error path doesn't have a real ProgramRef to hand in; use a
            // placeholder. The result_handler in the manager only reads the
            // proof_uuid out of context for routing purposes.
            context: protocol::ProofContext::new(
                proof_uuid.to_string(),
                ProgramRef::new(String::new(), 0),
                Default::default(),
            ),
            step: "unknown".to_string(),
            error: error_msg.to_string(),
        });

        let payload = ResultPayload {
            worker_id: self.worker_id,
            proof_uuid: proof_uuid.to_string(),
            result: MessageEnvelope::with_metadata(error_result),
        };

        warn!("Submitting error for proof {}: {}", proof_uuid, error_msg);

        // Serialize with bincode as per API spec
        let body = bincode::serialize(&payload)?;

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/octet-stream")
            .body(body)
            .send()
            .await?;

        if response.status().is_success() {
            info!("Successfully submitted error for proof {}", proof_uuid);
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            error!("Failed to submit error: status={}, body={}", status, body);
            Err(eyre::eyre!(
                "Failed to submit error: status={}, body={}",
                status,
                body
            ))
        }
    }

    /// Register this worker with the manager.
    ///
    /// Returns the confirmed worker ID. `loaded_programs` advertises what the
    /// worker holds, and the manager pushes back whatever is missing from it.
    // Arguments mirror the fields of `RegisterWorkerRequest` one-to-one; this
    // just builds that request and posts it, so the arg count tracks the wire
    // type rather than indicating a function that does too much.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip(self, loaded_programs), fields(worker_url = %worker_url))]
    pub async fn register_worker(
        &self,
        worker_url: &str,
        worker_id: usize,
        max_app_provers: usize,
        max_leaf_provers: usize,
        max_internal_provers: usize,
        loaded_programs: Vec<ProgramRef>,
        worker_role: protocol::WorkerRole,
    ) -> Result<usize> {
        let url = format!("{}/register_worker", self.manager_url);

        info!("Registering worker at {} with manager", worker_url);

        let payload = protocol::RegisterWorkerRequest {
            worker_url: worker_url.to_string(),
            worker_id,
            max_app_provers,
            max_leaf_provers,
            max_internal_provers,
            loaded_programs,
            worker_role,
        };

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        if response.status().is_success() {
            let body: serde_json::Value = response.json().await?;
            let worker_id = body["worker_id"]
                .as_u64()
                .ok_or_else(|| eyre::eyre!("Missing worker_id in registration response"))?
                as usize;
            info!(
                "Successfully registered with manager, confirmed worker_id={}",
                worker_id
            );
            Ok(worker_id)
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            warn!(
                "Failed to register with manager: status={}, body={}",
                status, body
            );
            Err(eyre::eyre!(
                "Failed to register: status={}, body={}",
                status,
                body
            ))
        }
    }

    /// Check if the manager is healthy.
    pub async fn check_manager_health(&self) -> bool {
        let url = format!("{}/healthz", self.manager_url);

        match self.client.get(&url).send().await {
            Ok(response) => response.status().is_success(),
            Err(e) => {
                warn!("Manager health check failed: {}", e);
                false
            }
        }
    }
}

/// Background task to periodically register with the manager.
#[allow(clippy::too_many_arguments)]
pub async fn registration_task(
    client: ResultClient,
    worker_url: String,
    worker_id: usize,
    max_app_provers: usize,
    max_leaf_provers: usize,
    max_internal_provers: usize,
    worker_role: protocol::WorkerRole,
    interval: Duration,
    cancel_token: tokio_util::sync::CancellationToken,
) {
    let mut interval_timer = tokio::time::interval(interval);

    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                info!("Registration task cancelled");
                break;
            }
            _ = interval_timer.tick() => {
                if let Err(e) = client
                    .register_worker(
                        &worker_url,
                        worker_id,
                        max_app_provers,
                        max_leaf_provers,
                        max_internal_provers,
                        // Report what this worker holds right now, not what it
                        // booted with. A stale snapshot would make the manager
                        // re-push every registered program on every tick.
                        loaded_programs(),
                        worker_role,
                    )
                    .await
                {
                    warn!("Failed to register with manager: {}", e);
                }
            }
        }
    }
}

/// The programs this worker currently has artifacts for.
fn loaded_programs() -> Vec<ProgramRef> {
    crate::artifacts::ArtifactStore::global()
        .map(|store| store.configured_programs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_client_creation() {
        let client = ResultClient::new("http://localhost:3000", 0).unwrap();
        assert_eq!(client.manager_url, "http://localhost:3000");
    }

    #[test]
    fn test_result_client_strips_trailing_slash() {
        let client = ResultClient::new("http://localhost:3000/", 0).unwrap();
        assert_eq!(client.manager_url, "http://localhost:3000");
    }
}
