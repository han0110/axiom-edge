//! Proof state management for Edge mode.
//!
//! Tracks the recursion tree of proofs and handles result aggregation.
//! The single central data type is [`ProofState`] (in [`state`]); behavior
//! is split across sibling modules:
//!
//! - [`state`] — `ProofState`, `ProofStatus`, `InternalProofIndex`,
//!   `LightweightProofState`, constructor, and read-only projections.
//! - [`recursion`] — pure tree math (layers, indices, segment ranges).
//! - [`result_handler`] — result handling and follow-up request dispatch.
//! - [`metrics_report`] — completion-time Markdown report and OTel emission.
//! - [`persistence`] — final-proof and failure-snapshot disk persistence.

mod metrics_report;
mod persistence;
mod recursion;
mod result_handler;
mod state;

pub use persistence::{CompletedLeafBatch, PersistedLeafFailureAppProofs};
pub use result_handler::ProofResultEnvelopeOutcome;
pub use state::{
    InternalProofIndex, LightweightProofState, ProofPipeline, ProofState, ProofStatus, TaskTiming,
};
