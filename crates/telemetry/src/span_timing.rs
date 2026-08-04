//! Thread-local span timing capture for STARK sub-step breakdowns.
//!
//! Provides a tracing [`Layer`] that captures INFO-level span durations into
//! thread-local storage. After a prover call (e.g., `vm.prove()`, `agg_prove()`),
//! call [`drain_span_timings()`] to retrieve and clear the accumulated sub-step
//! timings from the current thread.

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{Id, Subscriber};
use tracing_subscriber::{registry::LookupSpan, Layer};

thread_local! {
    /// Active span timings on this thread, keyed by span numeric ID.
    static ACTIVE_SPANS: RefCell<HashMap<u64, SpanTiming>> = RefCell::new(HashMap::new());
    /// Accumulated span timings since last drain. Key = "{span_name}_time_ms", value = total ms.
    static CAPTURED_TIMINGS: RefCell<HashMap<String, f64>> = RefCell::new(HashMap::new());
}

struct SpanTiming {
    name: String,
    start_time: Instant,
}

/// A tracing layer that captures INFO-level span durations into thread-local storage.
///
/// This replaces the need for `TimingMetricsLayer` + metrics recorder by storing
/// timings directly in thread-local `HashMap`s. Each thread accumulates timings
/// independently, which is exactly what we need since each GPU worker thread
/// runs prover calls sequentially.
///
/// Metric keys follow the `{span_name}_time_ms` convention (matching `TimingMetricsLayer`).
/// If multiple spans with the same name close between drains, their durations are summed.
pub struct SpanTimingLayer;

impl<S> Layer<S> for SpanTimingLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        _attrs: &tracing::span::Attributes<'_>,
        id: &Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if let Some(span) = ctx.span(id) {
            let metadata = span.metadata();
            // Only track INFO-level spans (matches TimingMetricsLayer behavior)
            if metadata.level() <= &tracing::Level::INFO {
                let span_id = id.into_u64();
                ACTIVE_SPANS.with(|spans| {
                    spans.borrow_mut().insert(
                        span_id,
                        SpanTiming {
                            name: metadata.name().to_string(),
                            start_time: Instant::now(),
                        },
                    );
                });
            }
        }
    }

    fn on_close(&self, id: Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        let span_id = id.into_u64();
        ACTIVE_SPANS.with(|spans| {
            if let Some(timing) = spans.borrow_mut().remove(&span_id) {
                let duration_ms = timing.start_time.elapsed().as_secs_f64() * 1000.0;
                let metric_name = format!("{}_time_ms", timing.name);
                CAPTURED_TIMINGS.with(|timings| {
                    *timings.borrow_mut().entry(metric_name).or_insert(0.0) += duration_ms;
                });
            }
        });
    }
}

/// Drain all captured span timings from the current thread.
///
/// Returns a `HashMap` of `"{span_name}_time_ms"` -> total duration in milliseconds,
/// and clears the buffer. Call this after each prover operation to get the sub-step breakdown.
pub fn drain_span_timings() -> HashMap<String, f64> {
    CAPTURED_TIMINGS.with(|timings| std::mem::take(&mut *timings.borrow_mut()))
}

/// Render a drained span-timing map as a compact `{name=ms,...}` string for a
/// single log line. Keys sort lexicographically and values round to whole
/// milliseconds. An empty map renders as `{}`.
pub fn format_span_timings(timings: &HashMap<String, f64>) -> String {
    let mut entries: Vec<(&String, &f64)> = timings.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let inner = entries
        .into_iter()
        .map(|(name, ms)| format!("{}={:.0}", name, ms))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{}}}", inner)
}
