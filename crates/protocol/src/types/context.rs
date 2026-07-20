//! Proof context — identifying tuple shared by every operation in a proof.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::ProgramRef;

/// Context that all proof operations share.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ProofContext {
    pub proof_uuid: String,
    /// Program targeted by this proof — `(name, version)` from the
    /// deployment's loadout.
    pub program: ProgramRef,

    /// Opaque, deployment-defined key/value labels carried with the proof.
    /// The edge treats these as pass-through metadata — it never interprets
    /// them. They're forwarded in lifecycle webhook events and emitted as
    /// metric attributes, so downstream integrations key off whatever they
    /// need (e.g. an ethereum deployment sets `"block_number"`). BTreeMap for
    /// deterministic ordering in JSON, metric labels, and tests.
    #[serde(default)]
    pub labels: BTreeMap<String, String>,

    /// Final proof artifact requested by the client.
    #[serde(default)]
    pub proof_type: ProofType,
}

/// Final proof artifact type requested for a proof.
#[derive(
    Clone, Copy, Serialize, Deserialize, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum ProofType {
    #[default]
    Stark,
    Evm,
}

impl ProofContext {
    pub fn new(proof_uuid: String, program: ProgramRef, labels: BTreeMap<String, String>) -> Self {
        Self {
            proof_uuid,
            program,
            labels,
            proof_type: ProofType::Stark,
        }
    }
}

/// Trait for types that contain a `ProofContext`.
pub trait WithProofContext {
    fn context(&self) -> &ProofContext;
    fn proof_uuid(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::{ProofContext, ProofType};

    #[test]
    fn proof_context_defaults_proof_type_to_stark_when_absent() {
        let context: ProofContext = serde_json::from_value(serde_json::json!({
            "proof_uuid": "proof-1",
            "program": {"name": "program-1", "version": 1},
            "labels": {"block_number": "1"}
        }))
        .unwrap();

        assert_eq!(context.proof_type, ProofType::Stark);
    }

    #[test]
    fn proof_context_deserializes_evm_proof_type() {
        let context: ProofContext = serde_json::from_value(serde_json::json!({
            "proof_uuid": "proof-1",
            "program": {"name": "program-1", "version": 1},
            "labels": {},
            "proof_type": "evm"
        }))
        .unwrap();

        assert_eq!(context.proof_type, ProofType::Evm);
    }
}
