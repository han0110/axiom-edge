//! Program identifier used to select which ELF to prove against.
//!
//! Programs are identified by `(name, version)` end-to-end. `name` is a
//! user-friendly string chosen at registration time; `version` bumps each
//! time a new ELF is uploaded under the same name.
//!
//! A deployment's loadout is populated at runtime: a client calls
//! `/register_program` with a guest ELF and the VM config to build it under,
//! and the manager fans that registration out to every worker. See
//! [`RegisterProgramRequest`].
//!
//! The `EDGE_PROGRAMS` environment variable (JSON array) optionally seeds the
//! loadout with programs whose artifacts are already on disk. It is unset in a
//! registration-driven deployment, in which case the loadout starts empty. See
//! [`parse_programs_env`].

use serde::{Deserialize, Serialize};

/// Environment variable name carrying the deployment's program loadout
/// as a JSON array of `{name, version}` objects.
pub const ENV_PROGRAMS: &str = "EDGE_PROGRAMS";

/// Identifies a single program version in the deployment loadout.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ProgramRef {
    /// User-friendly program name (e.g. `"sha256"`).
    pub name: String,
    /// Monotonically increasing version, assigned per `name`.
    pub version: u32,
}

impl ProgramRef {
    pub fn new(name: impl Into<String>, version: u32) -> Self {
        Self {
            name: name.into(),
            version,
        }
    }
}

impl std::fmt::Display for ProgramRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@v{}", self.name, self.version)
    }
}

/// Error from parsing the `EDGE_PROGRAMS` env var.
#[derive(Debug)]
pub enum ParseProgramsError {
    /// The env variable is not set or is empty.
    Missing,
    /// The env variable did not parse as a JSON array of `ProgramRef`.
    InvalidJson(String),
    /// The program list is empty.
    Empty,
    /// Two entries share the same `(name, version)`.
    Duplicate(ProgramRef),
}

impl std::fmt::Display for ParseProgramsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing => write!(
                f,
                "{ENV_PROGRAMS} is not set; expected a JSON array of {{name, version}} objects"
            ),
            Self::InvalidJson(e) => write!(f, "{ENV_PROGRAMS} is not valid JSON: {e}"),
            Self::Empty => write!(f, "{ENV_PROGRAMS} must contain at least one program"),
            Self::Duplicate(p) => write!(f, "{ENV_PROGRAMS} contains duplicate program {p}"),
        }
    }
}

impl std::error::Error for ParseProgramsError {}

/// Parse `EDGE_PROGRAMS` from a raw JSON string.
///
/// Validates that the list is non-empty and that no `(name, version)`
/// pair is duplicated. Returns the programs in declaration order.
pub fn parse_programs_str(raw: &str) -> Result<Vec<ProgramRef>, ParseProgramsError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(ParseProgramsError::Missing);
    }
    let programs: Vec<ProgramRef> = serde_json::from_str(trimmed)
        .map_err(|e| ParseProgramsError::InvalidJson(e.to_string()))?;
    if programs.is_empty() {
        return Err(ParseProgramsError::Empty);
    }
    let mut seen = std::collections::HashSet::new();
    for p in &programs {
        if !seen.insert(p.clone()) {
            return Err(ParseProgramsError::Duplicate(p.clone()));
        }
    }
    Ok(programs)
}

/// Parse the loadout `EDGE_PROGRAMS` seeds the deployment with.
///
/// An unset or empty variable yields an empty loadout, which is the normal
/// case for a registration-driven deployment. A variable that is set but
/// malformed is still an error, so a typo fails loud instead of silently
/// starting with no programs.
pub fn parse_programs_env() -> Result<Vec<ProgramRef>, ParseProgramsError> {
    match std::env::var(ENV_PROGRAMS) {
        Ok(s) if !s.trim().is_empty() => parse_programs_str(&s),
        _ => Ok(Vec::new()),
    }
}

/// `POST /register_program` — a guest program and the VM config to build it
/// under, sent by a client to the manager and fanned out to every worker.
///
/// The worker derives everything else from these two fields: it transpiles
/// `elf` into a `VmExe`, runs app and aggregation keygen against `vm_config`,
/// and builds the program's execution instances. Nothing is staged on disk
/// beforehand.
///
/// `vm_config` stays a JSON string rather than a typed value so this crate
/// keeps its promise not to depend on OpenVM. Only the worker parses it, as
/// an `SdkVmConfig`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisterProgramRequest {
    /// Name and version this program is registered under.
    pub program: ProgramRef,
    /// Guest ELF bytes.
    pub elf: Vec<u8>,
    /// Serialized `SdkVmConfig`, opaque to the manager.
    pub vm_config: String,
}

impl RegisterProgramRequest {
    /// Multipart field carrying the JSON-encoded [`ProgramRef`].
    pub const PART_PROGRAM: &'static str = "program";
    /// Multipart field carrying the raw ELF bytes.
    pub const PART_ELF: &'static str = "elf";
    /// Multipart field carrying the serialized `SdkVmConfig`.
    pub const PART_VM_CONFIG: &'static str = "vm_config";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_basic_list() {
        let raw = r#"[{"name":"sha256","version":1},{"name":"keccak","version":2}]"#;
        let parsed = parse_programs_str(raw).unwrap();
        assert_eq!(
            parsed,
            vec![ProgramRef::new("sha256", 1), ProgramRef::new("keccak", 2),]
        );
    }

    #[test]
    fn ignores_surrounding_whitespace() {
        let raw = "  [ {\"name\":\"a\",\"version\":1} ]  ";
        let parsed = parse_programs_str(raw).unwrap();
        assert_eq!(parsed, vec![ProgramRef::new("a", 1)]);
    }

    #[test]
    fn rejects_empty_string() {
        assert!(matches!(
            parse_programs_str(""),
            Err(ParseProgramsError::Missing)
        ));
        assert!(matches!(
            parse_programs_str("   "),
            Err(ParseProgramsError::Missing)
        ));
    }

    #[test]
    fn rejects_empty_list() {
        assert!(matches!(
            parse_programs_str("[]"),
            Err(ParseProgramsError::Empty)
        ));
    }

    #[test]
    fn rejects_duplicate() {
        let raw = r#"[{"name":"a","version":1},{"name":"a","version":1}]"#;
        match parse_programs_str(raw) {
            Err(ParseProgramsError::Duplicate(p)) => {
                assert_eq!(p, ProgramRef::new("a", 1));
            }
            other => panic!("expected Duplicate, got {other:?}"),
        }
    }

    #[test]
    fn allows_same_name_different_version() {
        let raw = r#"[{"name":"a","version":1},{"name":"a","version":2}]"#;
        let parsed = parse_programs_str(raw).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn rejects_invalid_json() {
        assert!(matches!(
            parse_programs_str("not json"),
            Err(ParseProgramsError::InvalidJson(_))
        ));
    }
}
