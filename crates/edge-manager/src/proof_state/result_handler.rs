//! Result-handling and dispatch logic.
//!
//! This is the manager's per-proof scheduler: results arrive (via
//! [`ProofState::handle_proof_result_with_envelope_outcome`] / its
//! lower-level siblings), the recursion tree gets updated, and follow-up
//! [`GeneralProveRequest`]s are emitted for the next layer of work.
//!
//! The math that decides "what's the tree shape" lives in
//! [`super::recursion`]; this file is the state machine that drives the
//! tree forward as proofs come in.

use chrono::Utc;
use eyre::{bail, Result};
use tracing::{error, info, warn};

use protocol::{
    current_timestamp, AppProof, DeferralTailDispatch, ErrorResult, EvmProof, EvmProveRequest,
    ExecuteE2Result, GeneralProveRequest, InternalLayerMetadataWire, InternalProof,
    InternalProofState, InternalProveRequest, LeafProof, LeafProveRequest, MessageEnvelope,
    ProofResult, ProofType, ProofsTypeWire,
};

use super::recursion::{
    num_internal_layers_for_leaf_count, num_proofs_at_internal_layer_for_leaf_count,
    segment_start_of_internal_proof_for_leaf_count, segment_start_to_internal_idx_with_batch,
};
use super::state::{InternalProofIndex, ProofState, ProofStatus};

/// Outcome returned by [`ProofState::handle_proof_result_with_envelope_outcome`].
///
/// Lets callers distinguish late results (arrived after the proof reached a
/// terminal state) from real progress, so they can emit the right metrics and
/// avoid spamming logs.
pub enum ProofResultEnvelopeOutcome {
    Late {
        should_log_notice: bool,
        status: ProofStatus,
    },
    Processed {
        follow_up_requests: Vec<GeneralProveRequest>,
        transitioned_to_terminal: bool,
    },
}

impl ProofState {
    /// Handle a proof result and classify how the manager should react.
    pub fn handle_proof_result_with_envelope_outcome(
        &mut self,
        envelope: MessageEnvelope<ProofResult>,
    ) -> Result<ProofResultEnvelopeOutcome> {
        if self.is_terminal() {
            let should_log_notice = if self.late_result_notice_emitted {
                false
            } else {
                self.late_result_notice_emitted = true;
                true
            };
            return Ok(ProofResultEnvelopeOutcome::Late {
                should_log_notice,
                status: self.status.clone(),
            });
        }

        self.last_updated = Utc::now();
        let requests = self.handle_proof_result(envelope.message)?;
        Ok(ProofResultEnvelopeOutcome::Processed {
            follow_up_requests: requests,
            transitioned_to_terminal: self.is_terminal(),
        })
    }

    /// Handle a proof result with idempotency protection.
    pub fn handle_proof_result_with_envelope(
        &mut self,
        envelope: MessageEnvelope<ProofResult>,
    ) -> Result<Vec<GeneralProveRequest>> {
        match self.handle_proof_result_with_envelope_outcome(envelope)? {
            ProofResultEnvelopeOutcome::Processed {
                follow_up_requests, ..
            } => Ok(follow_up_requests),
            ProofResultEnvelopeOutcome::Late { .. } => Ok(vec![]),
        }
    }

    /// Handle a proof result.
    pub fn handle_proof_result(&mut self, result: ProofResult) -> Result<Vec<GeneralProveRequest>> {
        if !matches!(self.status, ProofStatus::InProgress) {
            warn!(
                "Proof {} is not in progress, ignoring result of kind: {}",
                self.context.proof_uuid,
                result.kind()
            );
            return Ok(vec![]);
        }

        let requests = match result {
            ProofResult::Error(result) => {
                self.handle_error_result(result);
                vec![]
            }
            ProofResult::ExecuteE2(result) => {
                info!("Execute e2 done, cost: {}", result.state.cost);
                if result.state.cost > self.cost_limit {
                    let error_msg = format!(
                        "ExecuteE2 cost {} exceeds cost limit {}",
                        result.state.cost, self.cost_limit
                    );
                    warn!("{}", error_msg);
                    self.status = ProofStatus::Failing(error_msg);
                    self.notify_completion();
                    return Ok(vec![]);
                }
                self.handle_execute_e2_result(result)?
            }
            ProofResult::App(result) => self.handle_app_result(result)?,
            ProofResult::Leaf(result) => self.handle_leaf_result(result)?,
            ProofResult::Internal(result) => self.handle_internal_result(result)?,
            ProofResult::Evm(result) => self.handle_halo2_result(result)?,
        };

        if self.is_completed() {
            info!("Proof {} is completed", self.context.proof_uuid);
            self.status = ProofStatus::Completed;
            self.notify_completion();

            let completion_time = Utc::now();
            let e2e_latency_ms = (completion_time - self.proof_start_time).num_milliseconds();
            self.e2e_latency_ms = if e2e_latency_ms >= 0 {
                Some(e2e_latency_ms as u64)
            } else {
                warn!(
                    "Negative e2e latency detected: {}ms, setting to 0",
                    e2e_latency_ms
                );
                Some(0)
            };

            // Proving-only latency: from work dispatch (after input upload) to
            // completion, excluding the `/start_proof` submission overhead.
            // `proving_started_at` is set by the start_proof handler once all
            // workers have the input; if it was never set (e.g. an aborted
            // dispatch) we leave this as None rather than guess.
            if let Some(proving_started_at) = self.proving_started_at {
                let proving_latency_ms = (completion_time - proving_started_at).num_milliseconds();
                self.proving_latency_ms = Some(proving_latency_ms.max(0) as u64);
            }

            metrics::gauge!("proof_ended_at", self.proof_metric_labels())
                .set(current_timestamp() as f64);
            // The Markdown report is written by the HTTP handler that owns
            // the manager-side `metrics.output_dir` config — keeps deploy-
            // wide config out of per-proof state.
        }

        Ok(requests)
    }

    fn handle_error_result(&mut self, result: ErrorResult) {
        self.last_error_result = Some(result.clone());
        let reason = format!("step '{}' failed: {}", result.step, result.error);
        error!(
            "Proof {} failed at step {}: {}",
            self.context.proof_uuid, result.step, result.error
        );
        self.status = ProofStatus::Failing(reason);
        self.notify_completion();
    }

    fn handle_execute_e2_result(
        &mut self,
        result: ExecuteE2Result,
    ) -> Result<Vec<GeneralProveRequest>> {
        if self.execute_e2_state.is_some() {
            info!(
                "Proof {} had already received ExecuteE2 state before this result",
                self.context.proof_uuid
            );
            return Ok(vec![]);
        }

        let num_segments = result.state.num_segments;

        // Handle num_segments = 0 as a failure case
        if num_segments == 0 {
            error!(
                "Proof {} has 0 segments - this is invalid",
                self.context.proof_uuid
            );
            self.status = ProofStatus::Failing("ExecuteE2 returned 0 segments".to_string());
            self.notify_completion();
            return Ok(vec![]);
        }

        self.num_segments = Some(num_segments);

        // Calculate total instructions (only available with real Segment type)
        #[cfg(not(feature = "mock-provers"))]
        {
            let total_instructions: u64 = result
                .state
                .segments
                .iter()
                .map(|bytes| {
                    proof::decode_segment(bytes)
                        .map(|s| s.num_insns)
                        .unwrap_or_default()
                })
                .sum();
            self.num_instructions = Some(total_instructions);
        }
        #[cfg(feature = "mock-provers")]
        {
            // Mock mode: estimate instructions as segment count * average
            self.num_instructions = Some(num_segments as u64 * 1_000_000);
        }
        self.execute_e2_state = Some(result.state);

        // Now that we know num_segments, check for any pending partial batches
        // that arrived before ExecuteE2. This handles edge cases where app proofs
        // arrive before ExecuteE2Result.
        let mut requests = self.flush_pending_leaf_batches();

        // Also trigger tail proofs (internal proofs for incomplete recursion groups)
        requests.extend(self.trigger_tail_proofs());

        Ok(requests)
    }

    /// Flush any complete partial batches after num_segments is known.
    /// This handles the case where app proofs arrive before ExecuteE2Result.
    fn flush_pending_leaf_batches(&mut self) -> Vec<GeneralProveRequest> {
        let Some(num_segments) = self.num_segments else {
            return vec![];
        };

        let batch_size = self.leaf_arity;
        let num_full_batches = num_segments / batch_size;
        let tail_batch_size = num_segments % batch_size;

        let mut requests = vec![];

        // Check the tail batch (last partial batch) if it exists
        if tail_batch_size > 0 {
            let batch_start = num_full_batches * batch_size;
            let batch_end = num_segments - 1;

            // Skip if we already have this leaf proof
            if self.leaf_proofs.contains_key(&batch_start) {
                return requests;
            }

            // Check if all app proofs for this batch are ready
            let mut app_proofs = Vec::with_capacity(tail_batch_size);
            let mut all_ready = true;

            for i in batch_start..=batch_end {
                if let Some(app_state) = self.app_proofs.get(&i) {
                    if let Some(proof) = &app_state.proof {
                        app_proofs.push(proof.clone());
                    } else {
                        all_ready = false;
                        break;
                    }
                } else {
                    all_ready = false;
                    break;
                }
            }

            if all_ready && !app_proofs.is_empty() {
                info!(
                    "flush_pending_leaf_batches: Creating tail leaf proof for segments [{}-{}]",
                    batch_start, batch_end
                );
                requests.push(GeneralProveRequest::LeafProve(LeafProveRequest {
                    context: self.context.clone(),
                    app_proofs,
                    segment_start: batch_start,
                    segment_end: batch_end,
                }));
            }
        }

        requests
    }

    fn handle_app_result(&mut self, result: AppProof) -> Result<Vec<GeneralProveRequest>> {
        let now = Utc::now();
        if self.app_prove_started_at.is_none() {
            self.app_prove_started_at = Some(now);
        }
        self.app_prove_ended_at = Some(now);

        if result.state.proof.is_none() {
            error!(
                "Received an AppProof result without proof for segment {}. Marking proof as failed.",
                result.state.segment_idx
            );
            self.status = ProofStatus::Failing(format!(
                "AppProof for segment {} missing proof data",
                result.state.segment_idx
            ));
            self.notify_completion();
            return Ok(vec![]);
        }

        let segment_idx = result.state.segment_idx;

        // Validate segment_idx is in bounds if we know num_segments
        if let Some(num_segments) = self.num_segments {
            if segment_idx >= num_segments {
                error!(
                    "Received AppProof with out-of-bounds segment_idx={} (num_segments={})",
                    segment_idx, num_segments
                );
                // Don't fail the proof - just ignore this invalid result
                return Ok(vec![]);
            }
        }

        if self
            .app_proofs
            .insert(segment_idx, result.state.clone())
            .is_some()
        {
            warn!(
                "Received duplicated AppProof result for segment {}",
                segment_idx
            );
            return Ok(vec![]);
        }

        // The terminal app worker attaches the depth-independent
        // `(DEFERRAL_AS, 0)` authentication path on a deferral job; buffer
        // it by `proof_uuid` so `build_deferral_tail_dispatch` can forward
        // it to the tail worker. Non-deferral jobs leave it `None`.
        if let Some(bytes) = result.state.final_merkle_path_bytes.as_ref() {
            if self.deferral_final_merkle_path_bytes.is_some() {
                warn!(
                    "Received deferral final-merkle-path twice for {} (segment {}); \
                     keeping the first.",
                    self.context.proof_uuid, segment_idx
                );
            } else {
                self.deferral_final_merkle_path_bytes = Some(bytes.clone());
            }
        }

        // A no-deferral proof on a deferral deployment carries the COMPLETE
        // depth-0 `DeferralMerkleProofs` instead (the terminal app worker
        // builds it whole — there is no tail worker to finalize a path).
        // Buffer it for attachment to the terminal proof (stark: at persist;
        // evm: forwarded to the final-internal worker's tail-merge prep, then
        // on to the dispatched `EvmProve` step). Mutually exclusive with
        // `final_merkle_path_bytes` above.
        if let Some(bytes) = result.state.deferral_merkle_proofs_bytes.as_ref() {
            if self.deferral_depth0_merkle_proofs_bytes.is_some() {
                warn!(
                    "Received depth-0 deferral merkle proofs twice for {} (segment {}); \
                     keeping the first.",
                    self.context.proof_uuid, segment_idx
                );
            } else {
                self.deferral_depth0_merkle_proofs_bytes = Some(bytes.clone());
            }
        }

        let batch_size = self.leaf_arity;
        let batch_start = (segment_idx / batch_size) * batch_size;
        let batch_end = batch_start + batch_size - 1;

        let actual_batch_end = if let Some(num_segments) = self.num_segments {
            if num_segments == 0 {
                return Ok(vec![]);
            }
            batch_end.min(num_segments - 1)
        } else {
            // Don't know num_segments yet - wait for full batch
            batch_end
        };

        // Sanity check: ensure batch_start <= actual_batch_end
        if batch_start > actual_batch_end {
            error!(
                "Invalid batch range: start={} > end={} (segment_idx={})",
                batch_start, actual_batch_end, segment_idx
            );
            return Ok(vec![]);
        }

        let mut app_proofs = Vec::with_capacity(actual_batch_end - batch_start + 1);
        for i in batch_start..=actual_batch_end {
            if let Some(app_state) = self.app_proofs.get(&i) {
                if let Some(proof) = &app_state.proof {
                    app_proofs.push(proof.clone());
                } else {
                    return Ok(vec![]);
                }
            } else {
                info!(
                    "Batch [{}-{}] not yet complete, missing segment {}",
                    batch_start, actual_batch_end, i
                );
                return Ok(vec![]);
            }
        }

        info!(
            "Batch [{}-{}] complete! Creating leaf proof request with {} app proofs",
            batch_start,
            actual_batch_end,
            app_proofs.len()
        );

        let leaf_request = LeafProveRequest {
            context: self.context.clone(),
            app_proofs,
            segment_start: batch_start,
            segment_end: actual_batch_end,
        };

        Ok(vec![GeneralProveRequest::LeafProve(leaf_request)])
    }

    fn trigger_tail_proofs(&self) -> Vec<GeneralProveRequest> {
        let mut requests = vec![];
        let num_segments = self.num_segments.unwrap();
        let leaf_arity = self.leaf_arity;
        let internal_arity = self.internal_arity;
        let num_leaf_proofs = num_segments.div_ceil(leaf_arity);
        let num_internal_layers =
            num_internal_layers_for_leaf_count(num_leaf_proofs, internal_arity);
        let effective_final_layer = num_internal_layers.max(2) - 1;

        // The only record of the recursion tree's shape. Both arities are here
        // because a parser needs them to place every task in the tree.
        info!(
            "trigger_tail_proofs for proof {}: num_segments={}, leaf_arity={}, \
             internal_arity={}, num_leaf_proofs={}, num_internal_layers={}, \
             effective_final_layer={}",
            self.context.proof_uuid,
            num_segments,
            leaf_arity,
            internal_arity,
            num_leaf_proofs,
            num_internal_layers,
            effective_final_layer
        );

        for layer_idx in 0..num_internal_layers {
            let num_proofs = if layer_idx == 0 {
                num_leaf_proofs
            } else {
                num_proofs_at_internal_layer_for_leaf_count(
                    num_leaf_proofs,
                    layer_idx - 1,
                    internal_arity,
                )
            };
            let num_tail_proofs = num_proofs % internal_arity;
            if num_tail_proofs > 0 {
                let idx_start = num_proofs - num_tail_proofs;

                // Check if we've already created or received an internal proof for this tail
                // The tail proof index is always num_proofs / internal_arity (the last group)
                let tail_idx = num_proofs / internal_arity;
                if self.internal_proofs.contains_key(&InternalProofIndex {
                    layer_idx,
                    idx: tail_idx,
                }) {
                    // Already have this internal proof, skip to avoid duplicate
                    continue;
                }

                let mut proofs = Vec::with_capacity(num_tail_proofs);
                let segment_start = if layer_idx == 0 {
                    idx_start * leaf_arity
                } else {
                    segment_start_of_internal_proof_for_leaf_count(
                        leaf_arity,
                        internal_arity,
                        layer_idx - 1,
                        idx_start,
                    )
                };
                let mut actual_segment_end = segment_start;
                for i in 0..num_tail_proofs {
                    let idx = idx_start + i;
                    let mut proof_exists = false;
                    if layer_idx == 0 {
                        let leaf_segment_start = idx * leaf_arity;
                        if let Some(leaf_proof) = self.leaf_proofs.get(&leaf_segment_start) {
                            proofs.push(leaf_proof.proof.clone().unwrap());
                            actual_segment_end = leaf_proof.segment_end;
                            proof_exists = true;
                        }
                    } else {
                        let child_layer = layer_idx - 1;
                        if let Some(proof) = self.internal_proofs.get(&InternalProofIndex {
                            layer_idx: child_layer,
                            idx,
                        }) {
                            proofs.push(proof.proof.clone().unwrap());
                            actual_segment_end = proof.segment_end;
                            proof_exists = true;
                        }
                    }
                    if !proof_exists {
                        break;
                    }
                }
                if proofs.len() == num_tail_proofs {
                    let is_final_proof = layer_idx == effective_final_layer;
                    info!(
                        "trigger_tail_proofs: Creating tail internal proof for layer {} \
                         with {} child proofs, segments {}-{}, is_final_proof={}",
                        layer_idx,
                        proofs.len(),
                        segment_start,
                        actual_segment_end,
                        is_final_proof
                    );
                    let deferral_tail = if is_final_proof {
                        self.build_deferral_tail_dispatch(layer_idx)
                    } else {
                        None
                    };
                    requests.push(GeneralProveRequest::InternalProve(InternalProveRequest {
                        context: self.context.clone(),
                        child_proofs: proofs,
                        layer_idx,
                        segment_start,
                        segment_end: actual_segment_end,
                        is_final_proof,
                        deferral_tail,
                        // depth-0 merkle proofs (no-deferral proof on a
                        // deferral deployment) are consumed only by the final
                        // proof's tail-merge prep (then the dispatched
                        // `EvmProve`); `None` for real deferral proofs (they
                        // build their own on the tail).
                        deferral_merkle_proofs_bytes: if is_final_proof {
                            self.deferral_depth0_merkle_proofs_bytes.clone()
                        } else {
                            None
                        },
                    }));
                }
            }
        }
        requests
    }

    fn handle_leaf_result(&mut self, result: LeafProof) -> Result<Vec<GeneralProveRequest>> {
        let now = Utc::now();
        if self.leaf_prove_started_at.is_none() {
            self.leaf_prove_started_at = Some(now);
        }
        self.leaf_prove_ended_at = Some(now);

        if result.state.proof.is_none() {
            error!(
                "Received a LeafProof result without proof for segments [{}-{}]. Marking proof as failed.",
                result.state.segment_start, result.state.segment_end
            );
            self.status = ProofStatus::Failing(format!(
                "LeafProof for segments [{}-{}] missing proof data",
                result.state.segment_start, result.state.segment_end
            ));
            self.notify_completion();
            return Ok(vec![]);
        }

        let leaf_segment_start = result.state.segment_start;
        let leaf_segment_end = result.state.segment_end;

        // Validate segment bounds if we know num_segments
        if let Some(num_segments) = self.num_segments {
            if leaf_segment_start >= num_segments || leaf_segment_end >= num_segments {
                error!(
                    "Received LeafProof with out-of-bounds segments [{}-{}] (num_segments={})",
                    leaf_segment_start, leaf_segment_end, num_segments
                );
                return Ok(vec![]);
            }
        }

        info!(
            "Received leaf proof for segments [{}-{}]",
            leaf_segment_start, leaf_segment_end
        );

        if self
            .leaf_proofs
            .insert(leaf_segment_start, result.state.clone())
            .is_some()
        {
            warn!(
                "Received duplicated LeafProof result for segments [{}-{}]",
                leaf_segment_start, leaf_segment_end
            );
            return Ok(vec![]);
        }

        let leaf_arity = self.leaf_arity;
        let internal_arity = self.internal_arity;
        let leaf_idx = leaf_segment_start / leaf_arity;
        let group_start_leaf_idx = (leaf_idx / internal_arity) * internal_arity;
        let mut group_end_leaf_idx = group_start_leaf_idx + internal_arity - 1;
        let group_segment_start = group_start_leaf_idx * leaf_arity;

        if let Some(num_segments) = self.num_segments {
            if num_segments == 0 {
                error!("Received leaf proof but num_segments is 0. This indicates a bug.");
                return Ok(vec![]);
            }
            let num_leaf_proofs = num_segments.div_ceil(leaf_arity);
            group_end_leaf_idx = group_end_leaf_idx.min(num_leaf_proofs - 1);
        }

        let mut leaf_proofs = Vec::with_capacity(group_end_leaf_idx - group_start_leaf_idx + 1);
        let mut actual_segment_end = group_segment_start;
        for i in group_start_leaf_idx..=group_end_leaf_idx {
            let expected_segment_start = i * leaf_arity;
            if let Some(leaf_state) = self.leaf_proofs.get(&expected_segment_start) {
                leaf_proofs.push(leaf_state.proof.clone().unwrap());
                actual_segment_end = leaf_state.segment_end;
            } else {
                info!(
                    "Leaf group [{}-{}] not yet complete, missing leaf {}",
                    group_start_leaf_idx, group_end_leaf_idx, i
                );
                return Ok(vec![]);
            }
        }

        let is_final_proof = false;
        info!(
            "Leaf group [{}-{}] complete! Creating internal proof for layer 0 \
             with {} child proofs, segments [{}-{}]",
            group_start_leaf_idx,
            group_end_leaf_idx,
            leaf_proofs.len(),
            group_segment_start,
            actual_segment_end
        );

        Ok(vec![GeneralProveRequest::InternalProve(
            InternalProveRequest {
                context: self.context.clone(),
                child_proofs: leaf_proofs,
                layer_idx: 0,
                segment_start: group_segment_start,
                segment_end: actual_segment_end,
                is_final_proof,
                deferral_tail: None,
                deferral_merkle_proofs_bytes: None,
            },
        )])
    }

    fn handle_internal_result(
        &mut self,
        result: InternalProof,
    ) -> Result<Vec<GeneralProveRequest>> {
        let now = Utc::now();
        if self.internal_prove_started_at.is_none() {
            self.internal_prove_started_at = Some(now);
        }
        self.internal_prove_ended_at = Some(now);

        let leaf_arity = self.leaf_arity;
        let internal_arity = self.internal_arity;
        info!(
            "Received internal proof: layer_idx={}, segment_start={}, segment_end={}",
            result.state.layer_idx, result.state.segment_start, result.state.segment_end
        );

        if result.state.proof.is_none() {
            error!(
                "Received an InternalProof result without proof for layer_idx={}. Marking proof as failed.",
                result.state.layer_idx
            );
            self.status = ProofStatus::Failing(format!(
                "InternalProof for layer {} missing proof data",
                result.state.layer_idx
            ));
            self.notify_completion();
            return Ok(vec![]);
        }

        // Validate segment bounds if we know num_segments
        if let Some(num_segments) = self.num_segments {
            if result.state.segment_start >= num_segments
                || result.state.segment_end >= num_segments
            {
                error!(
                    "Received InternalProof with out-of-bounds segments [{}-{}] (num_segments={})",
                    result.state.segment_start, result.state.segment_end, num_segments
                );
                return Ok(vec![]);
            }
        }

        let layer_idx = result.state.layer_idx;
        let idx = segment_start_to_internal_idx_with_batch(
            result.state.segment_start,
            result.state.layer_idx,
            leaf_arity,
            internal_arity,
        );

        if self
            .internal_proofs
            .insert(InternalProofIndex { layer_idx, idx }, result.state.clone())
            .is_some()
        {
            warn!(
                "Received duplicated InternalProof result for segment {} at layer {}",
                result.state.segment_start, result.state.layer_idx
            );
            return Ok(vec![]);
        }

        let idx_start = idx / internal_arity * internal_arity;
        let mut idx_end = idx_start + internal_arity - 1;
        let mut is_next_layer_final = false;

        if let Some(num_segments) = self.num_segments {
            let num_leaf_proofs = num_segments.div_ceil(leaf_arity);
            let num_internal_layers =
                num_internal_layers_for_leaf_count(num_leaf_proofs, internal_arity);
            let effective_final_layer = num_internal_layers.max(2) - 1;

            // Final internal proof.
            //
            // `proof_type=Stark`: this is the completion artifact — the
            // surrounding `is_completed()` check picks it up via the
            // recursion-tree path and flips status to `Completed`.
            //
            // `proof_type=Evm`: this is an INTERMEDIATE step. The final-internal
            // worker always runs the tail merge / merkle-prep and submits the
            // POST-tail-merge proof + merkle bytes as a ready-for-evm message
            // (`ready_for_evm == true`); we promote that to the dispatched
            // `EvmProve` step (root → halo2), routed by the scheduler to any
            // eligible `runs_evm_prove()` worker (`Full` or `EvmDedicated`).
            // `ready_for_evm` is therefore an invariant here, not a branch: a
            // false value means an Evm proof reached the final layer without the
            // worker's ready-for-evm hand-off, so we fail loud rather than silently
            // skip dispatch (which would hang the proof with no EVM step).
            if layer_idx == effective_final_layer {
                if self.context.proof_type == ProofType::Evm {
                    if !result.state.ready_for_evm {
                        bail!(
                            "Evm proof {} reached the final internal layer without a \
                             ready-for-evm (post-merge) proof; cannot dispatch the EVM step",
                            self.context.proof_uuid
                        );
                    }
                    info!(
                        "Final internal proof for Evm proof {} is ready-for-evm; \
                         dispatching EvmProve to an eligible EVM worker",
                        self.context.proof_uuid
                    );
                    return Ok(vec![self.build_evm_prove_request(&result.state)]);
                }
                return Ok(vec![]);
            }

            // Check if we need an extra recursive layer
            if num_internal_layers == layer_idx + 1 && num_internal_layers == 1 {
                info!(
                    "Natural top of tree at layer {}, triggering additional internal_recursive proof",
                    layer_idx
                );
                let deferral_tail = self.build_deferral_tail_dispatch(layer_idx + 1);
                let internal_prove_request = InternalProveRequest {
                    context: self.context.clone(),
                    child_proofs: vec![result.state.proof.unwrap()],
                    layer_idx: layer_idx + 1,
                    segment_start: result.state.segment_start,
                    segment_end: result.state.segment_end,
                    is_final_proof: true,
                    deferral_tail,
                    deferral_merkle_proofs_bytes: self.deferral_depth0_merkle_proofs_bytes.clone(),
                };
                return Ok(vec![GeneralProveRequest::InternalProve(
                    internal_prove_request,
                )]);
            }

            is_next_layer_final = layer_idx + 1 == effective_final_layer;
            let num_proofs_at_layer = num_proofs_at_internal_layer_for_leaf_count(
                num_leaf_proofs,
                layer_idx,
                internal_arity,
            );
            if num_proofs_at_layer == 0 {
                error!(
                    "Received internal proof at layer {} idx {} but num_proofs_at_layer is 0",
                    layer_idx, idx
                );
                return Ok(vec![]);
            }
            idx_end = idx_end.min(num_proofs_at_layer - 1);
        }

        let mut proofs = Vec::with_capacity(idx_end - idx_start + 1);
        let mut segment_start = usize::MAX;
        let mut segment_end = usize::MIN;
        for i in idx_start..=idx_end {
            if let Some(proof) = self
                .internal_proofs
                .get(&InternalProofIndex { layer_idx, idx: i })
            {
                proofs.push(proof.proof.clone().unwrap());
                segment_start = segment_start.min(proof.segment_start);
                segment_end = segment_end.max(proof.segment_end);
            } else {
                return Ok(vec![]);
            }
        }

        info!(
            "Creating internal proof for layer {} (from {} child proofs at layer {}), \
             segments {}-{}, is_final_proof={}",
            layer_idx + 1,
            proofs.len(),
            layer_idx,
            segment_start,
            segment_end,
            is_next_layer_final
        );

        let deferral_tail = if is_next_layer_final {
            self.build_deferral_tail_dispatch(layer_idx + 1)
        } else {
            None
        };
        let internal_prove_request = InternalProveRequest {
            context: self.context.clone(),
            child_proofs: proofs,
            layer_idx: layer_idx + 1,
            segment_start,
            segment_end,
            is_final_proof: is_next_layer_final,
            deferral_tail,
            deferral_merkle_proofs_bytes: if is_next_layer_final {
                self.deferral_depth0_merkle_proofs_bytes.clone()
            } else {
                None
            },
        };

        Ok(vec![GeneralProveRequest::InternalProve(
            internal_prove_request,
        )])
    }

    fn handle_halo2_result(&mut self, result: EvmProof) -> Result<Vec<GeneralProveRequest>> {
        let proof_uuid = self.context.proof_uuid.clone();

        if self.context.proof_type != ProofType::Evm {
            let reason = format!(
                "received Evm result for {} but proof_type is {:?}",
                proof_uuid, self.context.proof_type
            );
            error!("{}", reason);
            self.status = ProofStatus::Failing(reason);
            self.notify_completion();
            return Ok(vec![]);
        }

        if result.state.proof.is_none() {
            let reason = format!("Evm result for {} missing proof bytes", proof_uuid);
            error!("{}", reason);
            self.status = ProofStatus::Failing(reason);
            self.notify_completion();
            return Ok(vec![]);
        }

        if self.evm_proof.is_some() {
            warn!("Received duplicated Evm result for {}", proof_uuid);
            return Ok(vec![]);
        }

        info!(
            "Evm proof received for {}, prove_time_ms={}",
            proof_uuid, result.state.prove_time_ms
        );

        self.evm_proof = Some(result.state);
        // The shared completion path in handle_proof_result picks up that
        // is_completed() now returns true and flips status accordingly.
        Ok(vec![])
    }

    /// Build the tail-merge dispatch for the final `InternalProveRequest`.
    ///
    /// Returns `Some(...)` only when this proof carries deferral inputs;
    /// otherwise returns `None` and the tail worker takes today's
    /// non-deferral path (byte-identical). `final_layer_idx` is the edge
    /// `layer_idx` at which the final internal proof is being produced; the
    /// SDK's `InternalLayerMetadata::internal_recursive_layer` equals it
    /// because edge layer 0 is `internal_for_leaf` (before the SDK starts
    /// counting `internal_recursive` rounds), and each subsequent edge
    /// layer maps 1:1 to an `internal_recursive` round.
    fn build_deferral_tail_dispatch(&self, final_layer_idx: usize) -> Option<DeferralTailDispatch> {
        if self.deferral_circuit_count == 0 {
            return None;
        }

        // Without segmentation info we can't reason about the tree shape;
        // skip the dispatch and surface the issue when the worker fails
        // fast on the missing tail data. This branch is not reachable in
        // practice (E2 always precedes the final InternalProve dispatch).
        let num_segments = match self.num_segments {
            Some(n) => n,
            None => {
                warn!(
                    "build_deferral_tail_dispatch called for {} before num_segments is known; \
                     emitting tail dispatch without InternalLayerMetadata.",
                    self.context.proof_uuid
                );
                return None;
            }
        };

        let num_leaf_proofs = num_segments.div_ceil(self.leaf_arity);
        let num_internal_layers =
            num_internal_layers_for_leaf_count(num_leaf_proofs, self.internal_arity);
        // Effective tree depth: at least 2 layers so the manager always
        // emits a wrap-style "final" recursive layer. Matches the
        // `effective_final_layer` used elsewhere in the result handler.
        let effective_final_layer = num_internal_layers.max(2) - 1;
        // `internal_node_idx` is openvm's monotonic post-increment counter
        // across *all* internal_for_leaf + internal_recursive proofs
        // (initial -1, then ++ per proof; returned as the LAST assigned
        // value). Total proofs = sum over edge layers 0..=effective_final
        // of the OUTPUT count at that layer. `num_proofs_at_internal_layer_for_leaf_count(..., layer, ...)`
        // returns the OUTPUT count at edge layer `layer`; `.max(1)` covers
        // the natural-top-of-tree wrap case where the tree would otherwise
        // produce zero proofs at the wrap layer.
        let total_internal_proofs: usize = (0..=effective_final_layer)
            .map(|layer| {
                num_proofs_at_internal_layer_for_leaf_count(
                    num_leaf_proofs,
                    layer,
                    self.internal_arity,
                )
                .max(1)
            })
            .sum();
        let internal_node_idx = (total_internal_proofs.saturating_sub(1)) as u32;
        // openvm starts `internal_recursive_layer` at 1 for the first
        // internal_recursive round (edge layer 1), and each additional
        // round bumps it. So edge final_layer_idx maps directly.
        let internal_recursive_layer = final_layer_idx.max(1) as u32;

        // Surface (once per build, at warn level) when the terminal app
        // worker never reported the deferral path — the tail worker will
        // bail with a clear error, but logging on the manager side helps
        // pinpoint which side dropped it.
        let final_merkle_path_bytes = match self.deferral_final_merkle_path_bytes.as_ref() {
            Some(bytes) => bytes.clone(),
            None => {
                warn!(
                    "build_deferral_tail_dispatch for {}: terminal AppProof never carried \
                     final_merkle_path_bytes; tail will fail fast.",
                    self.context.proof_uuid
                );
                Vec::new()
            }
        };

        Some(DeferralTailDispatch {
            layer_metadata: InternalLayerMetadataWire {
                internal_recursive_layer,
                internal_node_idx,
                // VM tree is still pure-VM at the final-internal point;
                // `prove_mixed` flips it to `Mix`/`Combined` on the tail
                // worker.
                proofs_type: ProofsTypeWire::Vm,
            },
            final_merkle_path_bytes,
        })
    }

    /// Build the [`EvmProve`](GeneralProveRequest::EvmProve) request from a
    /// **ready-for-evm** final internal proof.
    ///
    /// Carries the finished (post-tail-merge) proof bytes + serialized merkle
    /// bytes the final-internal worker shipped on the ready-for-evm message, plus
    /// the manager's own `proof_has_deferral` — whether this proof carries
    /// deferral inputs, which drives root's `proofs_type` (`Combined` vs `Vm`)
    /// and is authoritative here (the same source `build_deferral_tail_dispatch`
    /// gates on). The scheduler dispatches the result to an eligible
    /// `runs_evm_prove()` worker (`Full` or `EvmDedicated`); this is the only
    /// place the manager routes on the post-merge proof, never the raw internal.
    fn build_evm_prove_request(&self, state: &InternalProofState) -> GeneralProveRequest {
        GeneralProveRequest::EvmProve(EvmProveRequest {
            context: self.context.clone(),
            // Proof-bytes presence is validated by the `state.proof.is_none()`
            // guard at the top of `handle_internal_result`, so a ready-for-evm
            // message always carries the finished proof bytes here.
            internal_proof_bytes: state.proof.clone().unwrap_or_default(),
            deferral_merkle_proofs_bytes: state.deferral_merkle_proofs_bytes.clone(),
            proof_has_deferral: self.deferral_circuit_count > 0,
        })
    }
}

#[cfg(test)]
mod deferral_tail_dispatch_tests {
    use super::*;
    use protocol::{ProgramRef, ProofContext};

    fn make_state(num_segments: usize, leaf_arity: usize, internal_arity: usize) -> ProofState {
        let ctx = ProofContext::new(
            "p-def".to_string(),
            ProgramRef::new("test-program", 1),
            Default::default(),
        );
        let mut state = ProofState::new(ctx, 1_000_000, 1, leaf_arity, internal_arity, 300);
        state.num_segments = Some(num_segments);
        state
    }

    /// Non-deferral proofs (`deferral_circuit_count == 0`) must not
    /// emit a tail dispatch — the worker takes today's byte-identical
    /// non-deferral root → halo2 path.
    #[test]
    fn no_tail_dispatch_when_proof_is_not_deferral() {
        let state = make_state(8, 4, 4);
        assert!(state.build_deferral_tail_dispatch(2).is_none());
    }

    /// Deferral proof — circuits are emitted in def-idx order and the
    /// `internal_recursive_layer` mirrors the edge `layer_idx` at the
    /// final-internal dispatch (openvm starts counting at 1 for the first
    /// `internal_recursive` round, which is edge layer 1).
    #[test]
    fn deferral_tail_dispatch_emits_circuits_and_metadata() {
        let mut state = make_state(8, 4, 4);
        state.deferral_circuit_count = 1;

        // 8 segments, leaf_arity=4 → num_leaf_proofs = 2.
        // num_internal_layers = 1 (2 leaves fold in one internal round).
        // effective_final_layer = max(2, 1) - 1 = 1.
        // total_internal_proofs:
        //   layer 0 = ceil(2/4) = 1
        //   layer 1 = max(1, num_proofs_at_internal_layer_for_leaf_count(2, 0, 4)) = max(1, 1) = 1
        //   = 2 total, so internal_node_idx = 1.
        // internal_recursive_layer = final_layer_idx (passed below) = 1.
        let dispatch = state
            .build_deferral_tail_dispatch(1)
            .expect("deferral tail dispatch should be emitted");

        assert_eq!(dispatch.layer_metadata.internal_recursive_layer, 1);
        assert_eq!(dispatch.layer_metadata.internal_node_idx, 1);
        assert_eq!(
            dispatch.layer_metadata.proofs_type,
            protocol::ProofsTypeWire::Vm
        );
    }

    /// Larger tree exercises both the per-layer fan-in math and the
    /// monotonic `internal_node_idx` counter (sum-of-counts - 1).
    #[test]
    fn deferral_tail_dispatch_internal_node_idx_grows_with_layers() {
        let mut state = make_state(64, 4, 4);
        state.deferral_circuit_count = 1;

        // 64 segments, leaf_arity=4 → num_leaf_proofs = 16.
        // Layer math (internal_arity=4):
        //   layer 0 (internal_for_leaf): ceil(16/4) = 4
        //   layer 1 (internal_recursive 0): ceil(4/4) = 1
        // num_internal_layers = 2, effective_final_layer = max(2, 2) - 1 = 1.
        // total_internal_proofs = 4 + 1 = 5, internal_node_idx = 4.
        let dispatch = state
            .build_deferral_tail_dispatch(1)
            .expect("deferral tail dispatch should be emitted");
        assert_eq!(dispatch.layer_metadata.internal_recursive_layer, 1);
        assert_eq!(dispatch.layer_metadata.internal_node_idx, 4);
    }

    /// Without `num_segments` we can't reason about the tree shape, so no
    /// tail dispatch is emitted (`None`); a deferral job then fails fast
    /// rather than shipping metadata-less tail data. Not reachable in
    /// practice (the E2 execute phase always sets `num_segments` before the
    /// final InternalProve dispatch) — this pins the guard.
    #[test]
    fn deferral_tail_dispatch_none_when_num_segments_missing() {
        let mut state = make_state(8, 4, 4);
        state.num_segments = None;
        state.deferral_circuit_count = 1;

        assert!(state.build_deferral_tail_dispatch(1).is_none());
    }
}

#[cfg(test)]
mod envelope_outcome_tests {
    use super::*;
    use protocol::{ProgramRef, ProofContext};

    fn make_context() -> ProofContext {
        ProofContext::new(
            "test-proof".to_string(),
            ProgramRef::new("test-program", 1),
            Default::default(),
        )
    }

    fn make_error_result(context: ProofContext) -> ProofResult {
        ProofResult::Error(ErrorResult {
            context,
            step: "LeafProve".to_string(),
            error: "Test error".to_string(),
        })
    }

    #[test]
    fn terminal_proof_only_requests_one_late_result_notice() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);
        state.status = ProofStatus::Completed;

        let outcome1 = state
            .handle_proof_result_with_envelope_outcome(MessageEnvelope::with_metadata(
                make_error_result(context.clone()),
            ))
            .unwrap();
        assert!(matches!(
            outcome1,
            ProofResultEnvelopeOutcome::Late {
                should_log_notice: true,
                ..
            }
        ));

        let outcome2 = state
            .handle_proof_result_with_envelope_outcome(MessageEnvelope::with_metadata(
                make_error_result(context),
            ))
            .unwrap();
        assert!(matches!(
            outcome2,
            ProofResultEnvelopeOutcome::Late {
                should_log_notice: false,
                ..
            }
        ));
    }

    #[test]
    fn error_result_reports_terminal_transition_once() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);

        let outcome = state
            .handle_proof_result_with_envelope_outcome(MessageEnvelope::with_metadata(
                make_error_result(context),
            ))
            .unwrap();

        assert!(matches!(
            outcome,
            ProofResultEnvelopeOutcome::Processed {
                transitioned_to_terminal: true,
                ..
            }
        ));
        assert!(matches!(state.status, ProofStatus::Failing(_)));
    }
}

#[cfg(all(test, feature = "mock-provers"))]
mod tests {
    use super::super::state::InternalProofIndex;
    use super::*;
    use proof::{ProofWithPublicValue, F};
    use protocol::{
        AppProofState, ExecuteE2State, InternalProofState, LeafProofState, ProofContext,
    };
    use std::collections::HashMap;

    fn make_context() -> ProofContext {
        ProofContext::new(
            "test-proof".to_string(),
            protocol::ProgramRef::new("test-program", 1),
            Default::default(),
        )
    }

    fn make_mock_proof() -> Vec<u8> {
        proof::encode_proof(&ProofWithPublicValue::<F> {
            proof: vec![0u8; 256],
            public_values: vec![F::default(); 4],
        })
        .expect("mock proof encodes")
    }

    #[test]
    fn test_execute_e2_result_handling() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);

        // Create ExecuteE2Result with 8 segments
        let segments: Vec<proof::Segment> = (0..8).map(|_| Vec::new()).collect();
        let e2_result = ExecuteE2Result {
            context: context.clone(),
            state: ExecuteE2State {
                num_segments: 8,
                segments,
                cost: 500,
                execute_time_ms: 0,
            },
        };

        let requests = state
            .handle_proof_result(ProofResult::ExecuteE2(e2_result))
            .unwrap();

        // num_segments should be set
        assert_eq!(state.num_segments, Some(8));

        // ExecuteE2 only sets up the state; tail proofs require existing proofs
        // trigger_tail_proofs checks for existing leaf/internal proofs to form batches
        // Since no proofs exist yet, requests will be empty
        assert!(requests.is_empty());

        // But the state should be correctly initialized
        assert!(state.execute_e2_state.is_some());
        assert_eq!(state.execute_e2_state.as_ref().unwrap().num_segments, 8);
    }

    #[test]
    fn test_app_proof_batching() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);

        // Set num_segments first
        state.num_segments = Some(8);

        // Create 4 app proofs for segments 0-3 (one batch)
        for i in 0..4 {
            let app_proof = AppProof {
                context: context.clone(),
                state: AppProofState {
                    proof: Some(make_mock_proof()),
                    segment_idx: i,
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
            };

            let requests = state
                .handle_proof_result(ProofResult::App(app_proof))
                .unwrap();

            if i < 3 {
                // Not yet complete batch
                assert!(
                    requests.is_empty(),
                    "Should not trigger leaf for segment {}",
                    i
                );
            } else {
                // Batch complete (0-3)
                assert_eq!(requests.len(), 1, "Should trigger 1 leaf proof request");
                if let GeneralProveRequest::LeafProve(req) = &requests[0] {
                    assert_eq!(req.segment_start, 0);
                    assert_eq!(req.segment_end, 3);
                    assert_eq!(req.app_proofs.len(), 4);
                } else {
                    panic!("Expected LeafProve request");
                }
            }
        }
    }

    #[test]
    fn test_leaf_proof_aggregation() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);
        state.num_segments = Some(12);

        // With 12 segments and batch size 4, we have 3 leaf proofs (0-3, 4-7, 8-11)
        // These 3 leaves should trigger 1 internal proof at layer 0

        for leaf_idx in 0..3 {
            let segment_start = leaf_idx * 4;
            let segment_end = segment_start + 3;

            let leaf_proof = LeafProof {
                context: context.clone(),
                state: LeafProofState {
                    proof: Some(make_mock_proof()),
                    segment_start,
                    segment_end,
                    prove_time_ms: 0,
                    sub_metrics: HashMap::new(),
                    worker_id: 0,
                    completed_at_ms: 0,
                },
            };

            let requests = state
                .handle_proof_result(ProofResult::Leaf(leaf_proof))
                .unwrap();

            if leaf_idx < 2 {
                // Not yet 3 leaves
                assert!(
                    requests.is_empty(),
                    "Should not trigger internal for leaf {}",
                    leaf_idx
                );
            } else {
                // 3 leaves complete
                assert_eq!(requests.len(), 1, "Should trigger 1 internal proof request");
                if let GeneralProveRequest::InternalProve(req) = &requests[0] {
                    assert_eq!(req.layer_idx, 0);
                    assert_eq!(req.child_proofs.len(), 3);
                } else {
                    panic!("Expected InternalProve request");
                }
            }
        }
    }

    #[test]
    fn test_full_proof_completion() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);

        // 4 segments with batch size 4 = 1 leaf proof
        // 1 leaf proof needs 1 internal at layer 0, then 1 internal at layer 1
        state.num_segments = Some(4);

        // Add leaf proof for segments 0-3
        let leaf_proof = LeafProof {
            context: context.clone(),
            state: LeafProofState {
                proof: Some(make_mock_proof()),
                segment_start: 0,
                segment_end: 3,
                prove_time_ms: 0,
                sub_metrics: HashMap::new(),
                worker_id: 0,
                completed_at_ms: 0,
            },
        };
        state.leaf_proofs.insert(0, leaf_proof.state.clone());

        // Add internal proof at layer 0
        let internal_0 = InternalProof {
            context: context.clone(),
            state: InternalProofState {
                proof: Some(make_mock_proof()),
                layer_idx: 0,
                segment_start: 0,
                segment_end: 3,
                prove_time_ms: 0,
                compression_time_ms: 0,
                sub_metrics: HashMap::new(),
                wrap_sub_metrics: HashMap::new(),
                deferral_merkle_proofs_bytes: None,
                ready_for_evm: false,
                worker_id: 0,
                completed_at_ms: 0,
            },
        };
        state.internal_proofs.insert(
            InternalProofIndex {
                layer_idx: 0,
                idx: 0,
            },
            internal_0.state.clone(),
        );

        assert!(!state.is_completed(), "Should not be complete yet");

        // Add internal proof at layer 1 (the final layer)
        let internal_1 = InternalProof {
            context: context.clone(),
            state: InternalProofState {
                proof: Some(make_mock_proof()),
                layer_idx: 1,
                segment_start: 0,
                segment_end: 3,
                prove_time_ms: 0,
                compression_time_ms: 0,
                sub_metrics: HashMap::new(),
                wrap_sub_metrics: HashMap::new(),
                deferral_merkle_proofs_bytes: None,
                ready_for_evm: false,
                worker_id: 0,
                completed_at_ms: 0,
            },
        };
        state.internal_proofs.insert(
            InternalProofIndex {
                layer_idx: 1,
                idx: 0,
            },
            internal_1.state.clone(),
        );

        assert!(state.is_completed(), "Should be complete now");
        assert!(
            state.get_stark_proof().is_none(),
            "get_stark_proof needs Completed status"
        );

        // Set status to completed
        state.status = ProofStatus::Completed;
        let stark_proof = state.get_stark_proof();
        assert!(stark_proof.is_some(), "Should have stark proof");
    }

    #[test]
    fn test_error_result_handling() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);

        let error = ErrorResult {
            context: context.clone(),
            step: "LeafProve".to_string(),
            error: "Test error".to_string(),
        };

        let requests = state
            .handle_proof_result(ProofResult::Error(error))
            .unwrap();

        assert!(requests.is_empty());
        assert!(matches!(state.status, ProofStatus::Failing(_)));
    }

    #[test]
    fn test_cost_limit_exceeded() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 100, 4, 4, 3, 300); // Low cost limit

        let segments: Vec<proof::Segment> = (0..8).map(|_| Vec::new()).collect();
        let e2_result = ExecuteE2Result {
            context: context.clone(),
            state: ExecuteE2State {
                num_segments: 8,
                segments,
                cost: 500, // Exceeds limit
                execute_time_ms: 0,
            },
        };

        let requests = state
            .handle_proof_result(ProofResult::ExecuteE2(e2_result))
            .unwrap();

        assert!(requests.is_empty());
        assert!(matches!(state.status, ProofStatus::Failing(_)));
    }

    #[test]
    fn test_zero_segments_fails() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);

        let e2_result = ExecuteE2Result {
            context: context.clone(),
            state: ExecuteE2State {
                num_segments: 0,
                segments: vec![],
                cost: 100,
                execute_time_ms: 0,
            },
        };

        let requests = state
            .handle_proof_result(ProofResult::ExecuteE2(e2_result))
            .unwrap();

        assert!(requests.is_empty());
        assert!(matches!(state.status, ProofStatus::Failing(_)));
        if let ProofStatus::Failing(msg) = &state.status {
            assert!(msg.contains("0 segments"));
        }
    }

    #[test]
    fn test_single_segment_proof() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);

        // With 1 segment and batch size 4, we have 1 tail batch
        let segments: Vec<proof::Segment> = (0..1).map(|_| Vec::new()).collect();
        let e2_result = ExecuteE2Result {
            context: context.clone(),
            state: ExecuteE2State {
                num_segments: 1,
                segments,
                cost: 100,
                execute_time_ms: 0,
            },
        };

        // First receive ExecuteE2
        let _ = state
            .handle_proof_result(ProofResult::ExecuteE2(e2_result))
            .unwrap();
        assert_eq!(state.num_segments, Some(1));

        // Then receive the single app proof
        let app_proof = AppProof {
            context: context.clone(),
            state: AppProofState {
                proof: Some(make_mock_proof()),
                segment_idx: 0,
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
        };

        let requests = state
            .handle_proof_result(ProofResult::App(app_proof))
            .unwrap();

        // Should trigger a leaf proof for the single segment
        assert_eq!(requests.len(), 1);
        if let GeneralProveRequest::LeafProve(req) = &requests[0] {
            assert_eq!(req.segment_start, 0);
            assert_eq!(req.segment_end, 0);
            assert_eq!(req.app_proofs.len(), 1);
        } else {
            panic!("Expected LeafProve request");
        }
    }

    #[test]
    fn test_app_proofs_before_execute_e2() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);

        // Receive app proof before ExecuteE2 (edge case)
        let app_proof = AppProof {
            context: context.clone(),
            state: AppProofState {
                proof: Some(make_mock_proof()),
                segment_idx: 0,
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
        };

        let requests = state
            .handle_proof_result(ProofResult::App(app_proof))
            .unwrap();

        // Without knowing num_segments, we wait for a full batch
        assert!(requests.is_empty());
        assert_eq!(state.app_proofs.len(), 1);

        // Now receive ExecuteE2 with num_segments = 1 (a tail batch)
        let segments: Vec<proof::Segment> = (0..1).map(|_| Vec::new()).collect();
        let e2_result = ExecuteE2Result {
            context: context.clone(),
            state: ExecuteE2State {
                num_segments: 1,
                segments,
                cost: 100,
                execute_time_ms: 0,
            },
        };

        let requests = state
            .handle_proof_result(ProofResult::ExecuteE2(e2_result))
            .unwrap();

        // flush_pending_leaf_batches should now create the tail leaf proof
        assert_eq!(requests.len(), 1);
        if let GeneralProveRequest::LeafProve(req) = &requests[0] {
            assert_eq!(req.segment_start, 0);
            assert_eq!(req.segment_end, 0);
        } else {
            panic!("Expected LeafProve request");
        }
    }

    #[test]
    fn test_out_of_bounds_segment_ignored() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);
        state.num_segments = Some(4); // Only 4 segments valid (0-3)

        // Receive an app proof with out-of-bounds segment_idx
        let app_proof = AppProof {
            context: context.clone(),
            state: AppProofState {
                proof: Some(make_mock_proof()),
                segment_idx: 10, // Out of bounds
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
        };

        let requests = state
            .handle_proof_result(ProofResult::App(app_proof))
            .unwrap();

        // Should be ignored, no requests generated
        assert!(requests.is_empty());
        assert!(!state.app_proofs.contains_key(&10));
    }

    #[test]
    fn test_late_results_only_request_one_notice() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);
        state.status = ProofStatus::Completed;

        let app_proof = AppProof {
            context,
            state: AppProofState {
                proof: Some(make_mock_proof()),
                segment_idx: 0,
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
        };

        let outcome1 = state
            .handle_proof_result_with_envelope_outcome(MessageEnvelope::with_metadata(
                ProofResult::App(app_proof.clone()),
            ))
            .unwrap();
        assert!(matches!(
            outcome1,
            ProofResultEnvelopeOutcome::Late {
                should_log_notice: true,
                ..
            }
        ));

        let outcome2 = state
            .handle_proof_result_with_envelope_outcome(MessageEnvelope::with_metadata(
                ProofResult::App(app_proof),
            ))
            .unwrap();
        assert!(matches!(
            outcome2,
            ProofResultEnvelopeOutcome::Late {
                should_log_notice: false,
                ..
            }
        ));
    }

    // -- Evm completion path -----------------------------------------------
    fn make_evm_context() -> ProofContext {
        let mut ctx = ProofContext::new(
            "test-proof-evm".to_string(),
            protocol::ProgramRef::new("test-program", 1),
            Default::default(),
        );
        ctx.proof_type = protocol::ProofType::Evm;
        ctx
    }

    /// Set up a single-leaf-batch state (4 segments, leaf_arity=4) and
    /// install the final internal proof at layer 1 idx 0. Mirrors the
    /// terminal shape of `test_full_proof_completion`.
    fn state_with_final_internal(context: ProofContext) -> ProofState {
        let mut state = ProofState::new(context, 1_000_000, 4, 4, 3, 300);
        state.num_segments = Some(4);
        state.internal_proofs.insert(
            InternalProofIndex {
                layer_idx: 1,
                idx: 0,
            },
            InternalProofState {
                proof: Some(make_mock_proof()),
                layer_idx: 1,
                segment_start: 0,
                segment_end: 3,
                prove_time_ms: 0,
                compression_time_ms: 0,
                sub_metrics: HashMap::new(),
                wrap_sub_metrics: HashMap::new(),
                deferral_merkle_proofs_bytes: None,
                ready_for_evm: false,
                worker_id: 0,
                completed_at_ms: 0,
            },
        );
        state
    }

    #[test]
    fn evm_proof_completes_on_evm_result_not_final_internal() {
        let context = make_evm_context();
        let mut state = state_with_final_internal(context.clone());

        // For an Evm proof, the final-internal proof being present is NOT
        // sufficient for is_completed() (only the Evm artifact is). The root
        // proof is a worker-internal intermediate and is never reported here.
        assert!(
            !state.is_completed(),
            "Evm proof should not complete on the final internal alone"
        );
        assert!(matches!(state.status, ProofStatus::InProgress));

        // The Evm result arrives (the only result the worker posts for the EVM
        // step): flip to Completed via the shared completion path.
        state
            .handle_proof_result(ProofResult::Evm(protocol::EvmProof {
                context: context.clone(),
                state: protocol::EvmProofState {
                    proof: Some(vec![1u8; 128]),
                    prove_time_ms: 20,
                    root_prove_time_ms: 7,
                    sub_metrics: HashMap::new(),
                },
            }))
            .unwrap();
        assert!(matches!(state.status, ProofStatus::Completed));
        assert!(state.is_completed());
        assert!(
            state.e2e_latency_ms.is_some(),
            "Completion path should capture e2e latency"
        );

        let evm_bytes = state.get_evm_proof().expect("evm proof bytes available");
        assert_eq!(evm_bytes, vec![1u8; 128]);
    }

    #[test]
    fn stark_proof_still_completes_on_final_internal() {
        // Stark path must remain unchanged: final-internal arrival flips
        // is_completed() and the shared path marks Completed.
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);
        state.num_segments = Some(4);
        // Push a non-final and final internal through the handler so the
        // completion code path actually runs.
        let _ = state
            .handle_proof_result(ProofResult::Internal(InternalProof {
                context: context.clone(),
                state: InternalProofState {
                    proof: Some(make_mock_proof()),
                    layer_idx: 0,
                    segment_start: 0,
                    segment_end: 3,
                    prove_time_ms: 0,
                    compression_time_ms: 0,
                    sub_metrics: HashMap::new(),
                    wrap_sub_metrics: HashMap::new(),
                    deferral_merkle_proofs_bytes: None,
                    ready_for_evm: false,
                    worker_id: 0,
                    completed_at_ms: 0,
                },
            }))
            .unwrap();
        // num_internal_layers for 4 segments / arity 4 = 1 leaf proof → max 2 layers.
        let _ = state
            .handle_proof_result(ProofResult::Internal(InternalProof {
                context: context.clone(),
                state: InternalProofState {
                    proof: Some(make_mock_proof()),
                    layer_idx: 1,
                    segment_start: 0,
                    segment_end: 3,
                    prove_time_ms: 0,
                    compression_time_ms: 0,
                    sub_metrics: HashMap::new(),
                    wrap_sub_metrics: HashMap::new(),
                    deferral_merkle_proofs_bytes: None,
                    ready_for_evm: false,
                    worker_id: 0,
                    completed_at_ms: 0,
                },
            }))
            .unwrap();

        assert!(matches!(state.status, ProofStatus::Completed));
        assert!(state.get_stark_proof().is_some());
        assert!(
            state.get_evm_proof().is_none(),
            "Stark proof has no evm artifact"
        );
    }

    #[test]
    fn evm_result_against_stark_proof_is_rejected() {
        let context = make_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);

        let requests = state
            .handle_proof_result(ProofResult::Evm(protocol::EvmProof {
                context: context.clone(),
                state: protocol::EvmProofState {
                    proof: Some(vec![0u8; 64]),
                    prove_time_ms: 0,
                    root_prove_time_ms: 0,
                    sub_metrics: HashMap::new(),
                },
            }))
            .unwrap();
        assert!(requests.is_empty());
        assert!(
            matches!(state.status, ProofStatus::Failing(ref m) if m.contains("Evm result")
                && m.contains("Stark")),
            "Evm for Stark proof must transition to Failing, got {:?}",
            state.status
        );
        assert!(
            state.evm_proof.is_none(),
            "Evm artifact must not be stored on a Stark proof"
        );
    }

    /// Manager-level result-sequence test: feed a ready-for-evm final Internal
    /// then the Evm result to a fresh proof_type=Evm state and verify the
    /// manager reports `Completed` with a non-empty evm artifact. (The manager
    /// promotes the ready-for-evm internal to a dispatched `EvmProve`; the worker
    /// runs root→halo2 and posts only the final Evm proof, so the manager never
    /// sees a Root result.) This is the documented fallback for the mock E2E
    /// acceptance criterion when driving real worker EVM prove through HTTP isn't
    /// worth the scaffolding.
    #[test]
    fn evm_proof_completes_via_internal_then_evm_sequence() {
        let context = make_evm_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);
        state.num_segments = Some(4);

        // Layer 0 internal (intermediate).
        let _ = state
            .handle_proof_result(ProofResult::Internal(InternalProof {
                context: context.clone(),
                state: InternalProofState {
                    proof: Some(make_mock_proof()),
                    layer_idx: 0,
                    segment_start: 0,
                    segment_end: 3,
                    prove_time_ms: 5,
                    compression_time_ms: 0,
                    sub_metrics: HashMap::new(),
                    wrap_sub_metrics: HashMap::new(),
                    deferral_merkle_proofs_bytes: None,
                    ready_for_evm: false,
                    worker_id: 0,
                    completed_at_ms: 0,
                },
            }))
            .unwrap();
        assert!(matches!(state.status, ProofStatus::InProgress));

        // Layer 1 internal = final internal proof, submitted as a ready-for-evm
        // message. For Evm this is an intermediate step — the manager promotes it
        // to an `EvmProve` dispatch and waits for the Evm result to follow.
        let _ = state
            .handle_proof_result(ProofResult::Internal(InternalProof {
                context: context.clone(),
                state: InternalProofState {
                    proof: Some(make_mock_proof()),
                    layer_idx: 1,
                    segment_start: 0,
                    segment_end: 3,
                    prove_time_ms: 7,
                    compression_time_ms: 1,
                    sub_metrics: HashMap::new(),
                    wrap_sub_metrics: HashMap::new(),
                    deferral_merkle_proofs_bytes: None,
                    ready_for_evm: true,
                    worker_id: 0,
                    completed_at_ms: 0,
                },
            }))
            .unwrap();
        assert!(
            matches!(state.status, ProofStatus::InProgress),
            "Evm proof must NOT complete on final internal; got {:?}",
            state.status
        );

        // Evm → completion (the worker ran root→halo2 in-process).
        let _ = state
            .handle_proof_result(ProofResult::Evm(protocol::EvmProof {
                context: context.clone(),
                state: protocol::EvmProofState {
                    proof: Some(vec![0xAB; 4096]),
                    prove_time_ms: 50,
                    root_prove_time_ms: 30,
                    sub_metrics: HashMap::new(),
                },
            }))
            .unwrap();

        assert!(matches!(state.status, ProofStatus::Completed));
        let evm = state.get_evm_proof().expect("evm proof present");
        assert_eq!(evm.len(), 4096);
        assert!(state.e2e_latency_ms.is_some());

        // The EVM-tail timing is surfaced on `/proof_state` (root=30, halo2=50
        // from the Evm result above) so benchmarks can read the root+halo2 time
        // added on top of the STARK recursion, per proof.
        let lw = state.to_lightweight_state();
        assert_eq!(lw.total_root_prove_ms, Some(30));
        assert_eq!(lw.total_halo2_prove_ms, Some(50));
    }

    // -- Dedicated-halo2 EvmProve dispatch --------------------------------

    /// Unified path: the final-internal worker submits the POST-tail-merge
    /// final internal proof as a ready-for-evm message (`ready_for_evm = true`).
    /// The manager promotes it to a dispatched `EvmProve`, routing on the
    /// finished proof bytes (which the scheduler sends to any eligible
    /// `runs_evm_prove()` worker). Non-deferral proof → no merkle,
    /// `proof_has_deferral` false.
    #[test]
    fn evm_proof_dispatches_evm_prove_on_ready_for_evm() {
        let context = make_evm_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);
        state.num_segments = Some(4);

        let finished_proof = vec![0x11u8; 96];
        let requests = state
            .handle_proof_result(ProofResult::Internal(InternalProof {
                context: context.clone(),
                state: InternalProofState {
                    proof: Some(finished_proof.clone()),
                    layer_idx: 1,
                    segment_start: 0,
                    segment_end: 3,
                    prove_time_ms: 7,
                    compression_time_ms: 1,
                    sub_metrics: HashMap::new(),
                    wrap_sub_metrics: HashMap::new(),
                    deferral_merkle_proofs_bytes: None,
                    ready_for_evm: true,
                    worker_id: 0,
                    completed_at_ms: 0,
                },
            }))
            .unwrap();

        assert_eq!(
            requests.len(),
            1,
            "ready-for-evm final internal must emit exactly one EvmProve"
        );
        match &requests[0] {
            GeneralProveRequest::EvmProve(req) => {
                assert_eq!(
                    req.internal_proof_bytes, finished_proof,
                    "manager routes on the post-merge proof bytes"
                );
                assert!(req.deferral_merkle_proofs_bytes.is_none());
                assert!(!req.proof_has_deferral, "non-deferral proof");
                assert_eq!(req.context.proof_uuid, context.proof_uuid);
            }
            other => panic!("expected EvmProve, got {other:?}"),
        }
        // The EVM step hasn't run yet — wait for the dedicated worker's Evm.
        assert!(matches!(state.status, ProofStatus::InProgress));
        assert!(!state.is_completed());
    }

    /// Fail-loud invariant: in the unified design every final-internal worker
    /// ships the final internal as a ready-for-evm message, so a raw
    /// (`ready_for_evm = false`) final internal for an Evm proof is a bug. The
    /// manager rejects it with an error rather than silently skipping dispatch
    /// (which would hang the proof with no EVM step) or routing root → halo2 on
    /// the unmerged proof.
    #[test]
    fn evm_proof_errors_on_raw_final_internal() {
        let context = make_evm_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);
        state.num_segments = Some(4);

        let err = state
            .handle_proof_result(ProofResult::Internal(InternalProof {
                context: context.clone(),
                state: InternalProofState {
                    proof: Some(make_mock_proof()),
                    layer_idx: 1,
                    segment_start: 0,
                    segment_end: 3,
                    prove_time_ms: 7,
                    compression_time_ms: 1,
                    sub_metrics: HashMap::new(),
                    wrap_sub_metrics: HashMap::new(),
                    deferral_merkle_proofs_bytes: None,
                    ready_for_evm: false,
                    worker_id: 0,
                    completed_at_ms: 0,
                },
            }))
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("without a ready-for-evm (post-merge) proof"),
            "expected a fail-loud error on the raw final internal, got: {err}"
        );
    }

    /// Deferral and non-deferral proofs use the IDENTICAL handoff: a deferral
    /// proof's ready-for-evm message carries the merged proof + serialized merkle
    /// bytes and drives `proof_has_deferral = true` (root's `proofs_type =
    /// Combined`). The manager derives the flag from its own
    /// `deferral_circuit_count`, not from the merkle-bytes presence.
    #[test]
    fn evm_prove_dispatch_carries_deferral_merkle_and_flag() {
        let context = make_evm_context();
        let mut state = ProofState::new(context.clone(), 1_000_000, 4, 4, 3, 300);
        state.num_segments = Some(4);
        state.deferral_circuit_count = 1;

        let merged_proof = vec![0x22u8; 128];
        let merkle = vec![0x33u8; 64];
        let requests = state
            .handle_proof_result(ProofResult::Internal(InternalProof {
                context: context.clone(),
                state: InternalProofState {
                    proof: Some(merged_proof.clone()),
                    layer_idx: 1,
                    segment_start: 0,
                    segment_end: 3,
                    prove_time_ms: 7,
                    compression_time_ms: 1,
                    sub_metrics: HashMap::new(),
                    wrap_sub_metrics: HashMap::new(),
                    deferral_merkle_proofs_bytes: Some(merkle.clone()),
                    ready_for_evm: true,
                    worker_id: 0,
                    completed_at_ms: 0,
                },
            }))
            .unwrap();

        assert_eq!(requests.len(), 1);
        match &requests[0] {
            GeneralProveRequest::EvmProve(req) => {
                assert_eq!(req.internal_proof_bytes, merged_proof);
                assert_eq!(
                    req.deferral_merkle_proofs_bytes.as_deref(),
                    Some(merkle.as_slice())
                );
                assert!(
                    req.proof_has_deferral,
                    "deferral proof drives root proofs_type = Combined"
                );
            }
            other => panic!("expected EvmProve, got {other:?}"),
        }
    }
}
