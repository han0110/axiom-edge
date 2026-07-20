//! Edge Manager library.
//!
//! This crate provides the proof orchestration logic for Edge mode,
//! including worker registration, work assignment, and proof state management.

pub mod config;
pub mod handlers;
pub mod lifecycle;
pub mod loadout;
pub mod otel_metrics;
pub mod priority_queue;
pub mod proof_state;
pub mod scheduler;
pub mod server;
pub mod worker_registry;

// Re-exports for convenience
pub use config::ManagerConfig;
pub use proof_state::ProofState;
pub use scheduler::EdgeStateStore;
pub use worker_registry::EdgeWorkerRegistry;
