//! Edge Worker library.
//!
//! This crate provides the proof generation worker for Edge mode,
//! including the prover thread pool and HTTP handlers.

pub mod artifacts;
pub mod cancellation;
pub mod config;
#[cfg(not(feature = "mock-provers"))]
pub mod deferral_merkle;
pub mod handlers;
#[cfg(not(feature = "mock-provers"))]
pub mod openvm_config;
pub mod prover_pool;
pub mod provers;
pub mod registration;
pub mod result_client;
pub mod server;
#[cfg(not(feature = "mock-provers"))]
pub mod stark_verify;

// Re-exports
pub use config::WorkerConfig;
pub use prover_pool::ProverPool;
pub use server::run_server;
