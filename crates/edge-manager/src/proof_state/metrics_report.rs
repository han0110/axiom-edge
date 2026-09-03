//! Per-proof completion reporting: human-readable Markdown report and the
//! per-segment OTel metric emission.
//!
//! Both are read-only consumers of [`ProofState`] — they walk the recursion
//! tree to summarize timings and sub-step breakdowns. The two sinks are
//! independent: `generate_metrics_report` produces a Markdown string callers
//! write to disk; `emit_completion_metrics` pushes gauges through the
//! `metrics` crate.

use std::collections::HashMap;
use tracing::info;

use super::state::{ProofState, ProofStatus};

/// Format a number with comma separators (e.g. 17848 -> "17,848").
fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Sanitize a metric key for OTEL: only allow [a-zA-Z0-9_.\-/].
fn sanitize_metric_key(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-' || c == '/' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

impl ProofState {
    /// Generate a metrics.md report from the proof state.
    ///
    /// Format matches the standard openvm-prof output format.
    pub fn generate_metrics_report(&self) -> String {
        let mut report = String::new();

        // Collect per-group stats
        struct GroupStats {
            name: String,
            total_proof_time_ms: Vec<u64>,
            fastfwd_time_ms: Vec<u64>,
            stark_prove_time_ms: Vec<u64>,
            /// Aggregated STARK sub-step timings: metric_name -> per-proof values
            sub_metrics: HashMap<String, Vec<f64>>,
        }

        impl GroupStats {
            fn sum(&self) -> u64 {
                self.total_proof_time_ms.iter().sum()
            }
            fn max(&self) -> u64 {
                self.total_proof_time_ms.iter().copied().max().unwrap_or(0)
            }
        }

        let mut groups: Vec<GroupStats> = Vec::new();

        // App prove group
        if !self.app_proofs.is_empty() {
            let mut total_times = Vec::new();
            let mut fastfwd_times = Vec::new();
            let mut stark_times = Vec::new();
            let mut sub_metrics_map: HashMap<String, Vec<f64>> = HashMap::new();
            for state in self.app_proofs.values() {
                total_times.push(state.prove_time_ms);
                fastfwd_times.push(state.fastfwd_time_ms);
                stark_times.push(state.stark_prove_time_ms);
                for (key, &val) in &state.sub_metrics {
                    sub_metrics_map.entry(key.clone()).or_default().push(val);
                }
            }
            groups.push(GroupStats {
                name: "app_prove".to_string(),
                total_proof_time_ms: total_times,
                fastfwd_time_ms: fastfwd_times,
                stark_prove_time_ms: stark_times,
                sub_metrics: sub_metrics_map,
            });
        }

        // Leaf prove group
        if !self.leaf_proofs.is_empty() {
            let times: Vec<u64> = self.leaf_proofs.values().map(|s| s.prove_time_ms).collect();
            let mut sub_metrics_map: HashMap<String, Vec<f64>> = HashMap::new();
            for state in self.leaf_proofs.values() {
                for (key, &val) in &state.sub_metrics {
                    sub_metrics_map.entry(key.clone()).or_default().push(val);
                }
            }
            groups.push(GroupStats {
                name: "leaf_prove".to_string(),
                total_proof_time_ms: times,
                fastfwd_time_ms: vec![],
                stark_prove_time_ms: vec![],
                sub_metrics: sub_metrics_map,
            });
        }

        // Internal prove groups (one per layer)
        if !self.internal_proofs.is_empty() {
            let mut layers: HashMap<usize, Vec<u64>> = HashMap::new();
            let mut layer_sub_metrics: HashMap<usize, HashMap<String, Vec<f64>>> = HashMap::new();
            for (idx, state) in &self.internal_proofs {
                layers
                    .entry(idx.layer_idx)
                    .or_default()
                    .push(state.prove_time_ms);
                let sm = layer_sub_metrics.entry(idx.layer_idx).or_default();
                for (key, &val) in &state.sub_metrics {
                    sm.entry(key.clone()).or_default().push(val);
                }
            }
            let mut layer_keys: Vec<usize> = layers.keys().copied().collect();
            layer_keys.sort();
            for layer_idx in layer_keys {
                let times = layers.remove(&layer_idx).unwrap();
                let sub_metrics = layer_sub_metrics.remove(&layer_idx).unwrap_or_default();
                groups.push(GroupStats {
                    name: format!("internal.{}", layer_idx),
                    total_proof_time_ms: times,
                    fastfwd_time_ms: vec![],
                    stark_prove_time_ms: vec![],
                    sub_metrics,
                });
            }
        }

        // Compression (from final internal proof)
        let compression_ms: u64 = self
            .internal_proofs
            .values()
            .map(|s| s.compression_time_ms)
            .sum();

        // --- Summary table ---
        // "Total Time" = sum of all per-segment times (total CPU work)
        // "Parallel Proof Time" = max per group (1 worker bottleneck)
        // "Realized Time (N GPU)" = sum / N for all groups (work distributed across N GPUs)
        let n = self.num_workers.max(1);
        let total_sum: u64 = groups.iter().map(|g| g.sum()).sum::<u64>() + compression_ms;
        let total_parallel: u64 = groups.iter().map(|g| g.max()).sum::<u64>() + compression_ms;

        // Realized time: all proving phases are distributed across N GPUs
        let realized_time = |group: &GroupStats| -> f64 { group.sum() as f64 / n as f64 };
        let total_realized: f64 =
            groups.iter().map(realized_time).sum::<f64>() + compression_ms as f64;

        report.push_str(&format!(
            "| Summary | Total Time (s) | Parallel Proof Time (s) | Realized Time ({} GPU) (s) |\n",
            n
        ));
        report.push_str("|:---|---:|---:|---:|\n");
        report.push_str(&format!(
            "| Total | {:.2} | {:.2} | {:.2} |\n",
            total_sum as f64 / 1000.0,
            total_parallel as f64 / 1000.0,
            total_realized / 1000.0
        ));
        for group in &groups {
            report.push_str(&format!(
                "| {} | {:.2} | {:.2} | {:.2} |\n",
                group.name,
                group.sum() as f64 / 1000.0,
                group.max() as f64 / 1000.0,
                realized_time(group) / 1000.0
            ));
        }
        if compression_ms > 0 {
            report.push_str(&format!(
                "| compression | {:.2} | {:.2} | {:.2} |\n",
                compression_ms as f64 / 1000.0,
                compression_ms as f64 / 1000.0,
                compression_ms as f64 / 1000.0
            ));
        }
        report.push('\n');

        // --- Per-group detail tables ---
        for group in &groups {
            let times = &group.total_proof_time_ms;
            let count = times.len() as f64;
            let sum: u64 = times.iter().sum();
            let max: u64 = times.iter().copied().max().unwrap_or(0);
            let min: u64 = times.iter().copied().min().unwrap_or(0);
            let avg = if count > 0.0 { sum as f64 / count } else { 0.0 };

            report.push_str(&format!("| {} |||||\n", group.name));
            report.push_str("|:---|---:|---:|---:|---:|\n");
            report.push_str("|metric|avg|sum|max|min|\n");
            report.push_str(&format!(
                "| `total_proof_time_ms ` | {:.0} | {:>7} | {:>5} | {:>5} |\n",
                avg,
                format_with_commas(sum),
                max,
                min
            ));

            // Fast-forward and STARK sub-step times (app_prove only)
            if !group.fastfwd_time_ms.is_empty() {
                let ff = &group.fastfwd_time_ms;
                let ff_sum: u64 = ff.iter().sum();
                let ff_max: u64 = ff.iter().copied().max().unwrap_or(0);
                let ff_min: u64 = ff.iter().copied().min().unwrap_or(0);
                let ff_avg = if count > 0.0 {
                    ff_sum as f64 / count
                } else {
                    0.0
                };
                report.push_str(&format!(
                    "| `fastfwd_time_ms     ` | {:.0} | {:>7} | {:>5} | {:>5} |\n",
                    ff_avg,
                    format_with_commas(ff_sum),
                    ff_max,
                    ff_min
                ));
            }

            if !group.stark_prove_time_ms.is_empty() {
                let sp = &group.stark_prove_time_ms;
                let sp_sum: u64 = sp.iter().sum();
                let sp_max: u64 = sp.iter().copied().max().unwrap_or(0);
                let sp_min: u64 = sp.iter().copied().min().unwrap_or(0);
                let sp_avg = if count > 0.0 {
                    sp_sum as f64 / count
                } else {
                    0.0
                };
                report.push_str(&format!(
                    "| `stark_prove_time_ms ` | {:.0} | {:>7} | {:>5} | {:>5} |\n",
                    sp_avg,
                    format_with_commas(sp_sum),
                    sp_max,
                    sp_min
                ));
            }

            // STARK sub-step timings from tracing spans (sorted by sum descending)
            if !group.sub_metrics.is_empty() {
                let mut sorted_metrics: Vec<_> = group.sub_metrics.iter().collect();
                sorted_metrics.sort_by(|a, b| {
                    let sum_a: f64 = a.1.iter().sum();
                    let sum_b: f64 = b.1.iter().sum();
                    sum_b
                        .partial_cmp(&sum_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for (metric_name, values) in sorted_metrics {
                    let sm_count = values.len() as f64;
                    let sm_sum: f64 = values.iter().sum();
                    let sm_max: f64 = values.iter().copied().fold(0.0_f64, f64::max);
                    let sm_min: f64 = values.iter().copied().fold(f64::MAX, f64::min);
                    let sm_avg = if sm_count > 0.0 {
                        sm_sum / sm_count
                    } else {
                        0.0
                    };
                    report.push_str(&format!(
                        "| `{:<25}` | {:.0} | {:>7} | {:>5} | {:>5} |\n",
                        metric_name,
                        sm_avg,
                        format_with_commas(sm_sum as u64),
                        sm_max as u64,
                        sm_min as u64
                    ));
                }
            }

            report.push('\n');
        }

        report
    }

    /// Proof-level metric labels: core identifiers (program name/version,
    /// proof_uuid) plus every deployment-defined label carried on the context
    /// (e.g. an ethereum deployment's `block_number`). The edge emits whatever
    /// labels the caller attached without interpreting them. Returned as a
    /// `Vec<Label>` for the `metrics` macro's dynamic-labels form; clone per
    /// emission.
    pub(crate) fn proof_metric_labels(&self) -> Vec<metrics::Label> {
        let mut v = vec![
            metrics::Label::new("program_name", self.context.program.name.clone()),
            metrics::Label::new("program_version", self.context.program.version.to_string()),
            metrics::Label::new("proof_uuid", self.context.proof_uuid.clone()),
        ];
        for (k, val) in &self.context.labels {
            v.push(metrics::Label::new(k.clone(), val.clone()));
        }
        v
    }

    /// Emit metrics for a completed proof via the `metrics` crate.
    ///
    /// Called once when a proof reaches Completed or Failed status.
    /// All gauges are tagged with program_name + program_version + proof_uuid
    /// for Grafana filtering.
    pub fn emit_completion_metrics(&self) {
        let program_name = self.context.program.name.clone();
        let program_version = self.context.program.version.to_string();
        let proof = self.context.proof_uuid.clone();

        // Proof-level label set (core identifiers + deployment labels).
        let proof_labels = self.proof_metric_labels();

        // E2E latency
        if let Some(e2e_ms) = self.e2e_latency_ms {
            metrics::gauge!("edge_e2e_latency_ms", proof_labels.clone()).set(e2e_ms as f64);
        }

        // Proving-only latency (excludes /start_proof submission + upload fan-out)
        if let Some(proving_ms) = self.proving_latency_ms {
            metrics::gauge!("edge_proving_latency_ms", proof_labels.clone()).set(proving_ms as f64);
        }

        // Execution time
        if let Some(ref e2_state) = self.execute_e2_state {
            metrics::gauge!("edge_execute_time_ms", proof_labels.clone())
                .set(e2_state.execute_time_ms as f64);

            metrics::gauge!("edge_num_segments", proof_labels.clone())
                .set(e2_state.num_segments as f64);

            metrics::gauge!("edge_cost", proof_labels.clone()).set(e2_state.cost as f64);
        }

        // Per-segment app prove times
        let mut total_app_prove_ms: u64 = 0;
        for (idx, app_state) in &self.app_proofs {
            metrics::gauge!("edge_app_prove_time_ms",
                "program_name" => program_name.clone(),
                "program_version" => program_version.clone(),
                "proof_uuid" => proof.clone(),
                "segment_idx" => idx.to_string()
            )
            .set(app_state.prove_time_ms as f64);
            metrics::gauge!("edge_segment_fastfwd_time_ms",
                "program_name" => program_name.clone(),
                "program_version" => program_version.clone(),
                "proof_uuid" => proof.clone(),
                "segment_idx" => idx.to_string()
            )
            .set(app_state.fastfwd_time_ms as f64);
            metrics::gauge!("edge_segment_stark_prove_time_ms",
                "program_name" => program_name.clone(),
                "program_version" => program_version.clone(),
                "proof_uuid" => proof.clone(),
                "segment_idx" => idx.to_string()
            )
            .set(app_state.stark_prove_time_ms as f64);
            for (key, &val) in &app_state.sub_metrics {
                let sanitized = sanitize_metric_key(key);
                metrics::gauge!(format!("edge_app_{sanitized}"),
                    "program_name" => program_name.clone(),
                "program_version" => program_version.clone(),
                    "proof_uuid" => proof.clone(),
                    "segment_idx" => idx.to_string()
                )
                .set(val);
            }
            total_app_prove_ms += app_state.prove_time_ms;
        }
        metrics::gauge!("edge_total_app_prove_ms", proof_labels.clone())
            .set(total_app_prove_ms as f64);

        // Leaf prove times
        let mut total_leaf_ms: u64 = 0;
        for (idx, leaf_state) in &self.leaf_proofs {
            for (key, &val) in &leaf_state.sub_metrics {
                let sanitized = sanitize_metric_key(key);
                metrics::gauge!(format!("edge_leaf_{sanitized}"),
                    "program_name" => program_name.clone(),
                "program_version" => program_version.clone(),
                    "proof_uuid" => proof.clone(),
                    "batch_idx" => idx.to_string()
                )
                .set(val);
            }
            total_leaf_ms += leaf_state.prove_time_ms;
        }
        metrics::gauge!("edge_total_leaf_prove_ms", proof_labels.clone()).set(total_leaf_ms as f64);

        // Internal prove times and compression times
        let mut total_internal_ms: u64 = 0;
        let mut total_compression_ms: u64 = 0;
        for (idx, internal_state) in &self.internal_proofs {
            for (key, &val) in &internal_state.sub_metrics {
                let sanitized = sanitize_metric_key(key);
                metrics::gauge!(format!("edge_internal_{sanitized}"),
                    "program_name" => program_name.clone(),
                "program_version" => program_version.clone(),
                    "proof_uuid" => proof.clone(),
                    "layer_idx" => idx.layer_idx.to_string(),
                    "node_idx" => idx.idx.to_string()
                )
                .set(val);
            }
            for (key, &val) in &internal_state.wrap_sub_metrics {
                let sanitized = sanitize_metric_key(key);
                metrics::gauge!(format!("edge_internal_wrap_{sanitized}"),
                    "program_name" => program_name.clone(),
                "program_version" => program_version.clone(),
                    "proof_uuid" => proof.clone(),
                    "layer_idx" => idx.layer_idx.to_string(),
                    "node_idx" => idx.idx.to_string()
                )
                .set(val);
            }
            total_internal_ms += internal_state.prove_time_ms;
            total_compression_ms += internal_state.compression_time_ms;
        }
        metrics::gauge!("edge_total_internal_prove_ms", proof_labels.clone())
            .set(total_internal_ms as f64);

        metrics::gauge!("edge_compression_time_ms", proof_labels.clone())
            .set(total_compression_ms as f64);

        // EVM step (root + halo2): only present for proof_type=Evm proofs. The
        // worker folds the root timing into the single Evm result, so both
        // stages are reported here. sub_metrics keys are prefixed root_/halo2_.
        if let Some(ref evm_state) = self.evm_proof {
            metrics::gauge!("edge_halo2_prove_time_ms", proof_labels.clone())
                .set(evm_state.prove_time_ms as f64);
            metrics::gauge!("edge_root_prove_time_ms", proof_labels.clone())
                .set(evm_state.root_prove_time_ms as f64);
            metrics::gauge!("edge_evm_total_prove_ms", proof_labels.clone())
                .set((evm_state.prove_time_ms + evm_state.root_prove_time_ms) as f64);
            for (key, &val) in &evm_state.sub_metrics {
                let sanitized = sanitize_metric_key(key);
                metrics::gauge!(format!("edge_evm_{sanitized}"),
                    "program_name" => program_name.clone(),
                    "program_version" => program_version.clone(),
                    "proof_uuid" => proof.clone()
                )
                .set(val);
            }
        }

        // Wall-clock phase durations
        let app_prove_wallclock_ms = match (self.app_prove_started_at, self.app_prove_ended_at) {
            (Some(s), Some(e)) => (e - s).num_milliseconds().max(0) as u64,
            _ => 0,
        };
        metrics::gauge!("edge_app_prove_wallclock_ms", proof_labels.clone())
            .set(app_prove_wallclock_ms as f64);

        let leaf_prove_wallclock_ms = match (self.leaf_prove_started_at, self.leaf_prove_ended_at) {
            (Some(s), Some(e)) => (e - s).num_milliseconds().max(0) as u64,
            _ => 0,
        };
        metrics::gauge!("edge_leaf_prove_wallclock_ms", proof_labels.clone())
            .set(leaf_prove_wallclock_ms as f64);

        let internal_prove_wallclock_ms =
            match (self.internal_prove_started_at, self.internal_prove_ended_at) {
                (Some(s), Some(e)) => (e - s).num_milliseconds().max(0) as u64,
                _ => 0,
            };
        metrics::gauge!("edge_internal_prove_wallclock_ms", proof_labels.clone())
            .set(internal_prove_wallclock_ms as f64);

        // Status
        let status_val: f64 = match &self.status {
            ProofStatus::Completed => 1.0,
            ProofStatus::Failed(_) => -1.0,
            _ => 0.0,
        };
        metrics::gauge!("edge_proof_status", proof_labels.clone()).set(status_val);

        info!(
            "Emitted metrics for proof {}: e2e={}ms, proving={}ms, app={}ms, leaf={}ms, internal={}ms, compress={}ms",
            proof,
            self.e2e_latency_ms.unwrap_or(0),
            self.proving_latency_ms.unwrap_or(0),
            total_app_prove_ms,
            total_leaf_ms,
            total_internal_ms,
            total_compression_ms,
        );
    }
}
