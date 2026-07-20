//! The set of programs a deployment currently serves.
//!
//! The loadout is populated at runtime by `/register_program` rather than
//! fixed at boot. The manager retains each registration payload so a worker
//! that registers later can be brought up to the same loadout without
//! operator action.
//!
//! `EDGE_PROGRAMS` may still seed the loadout with programs whose artifacts
//! are already staged on the workers' disks. Those entries carry no payload,
//! because there is nothing for the manager to push.

use std::collections::HashMap;
use std::sync::Arc;

use protocol::ProgramRef;

/// A registration payload, retained for replay to late-joining workers.
#[derive(Clone)]
pub struct RegisteredProgram {
    pub elf: Arc<Vec<u8>>,
    pub vm_config: Arc<String>,
    pub baseline: Baseline,
}

/// The verifying key the workers report for a program, folded together as each
/// worker answers `/register_program`.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub enum Baseline {
    /// No worker has reported one yet, or this build derives none.
    #[default]
    Unknown,
    /// Every worker that reported one reported these bincode bytes.
    Agreed(Arc<Vec<u8>>),
    /// Workers disagree, so the deployment cannot serve this program.
    Mismatch,
}

/// Outcome of inserting a program into the loadout.
pub enum InsertOutcome {
    /// Newly added; the caller must push it to every registered worker.
    Inserted,
    /// Already present with identical bytes, so the call is a no-op.
    Unchanged,
    /// Already present under the same name and version with different bytes.
    Conflict,
}

/// Ordered program list plus the payload needed to replay each registration.
///
/// Order is preserved because it is user-visible in `/loadout` and in the 409
/// bodies that name the current loadout.
#[derive(Default)]
pub struct Loadout {
    ordered: Vec<ProgramRef>,
    programs: HashMap<ProgramRef, Option<RegisteredProgram>>,
}

impl Loadout {
    /// Seed from `EDGE_PROGRAMS`. These entries have no payload.
    pub fn seeded(programs: Vec<ProgramRef>) -> Self {
        let map = programs.iter().map(|p| (p.clone(), None)).collect();
        Self {
            ordered: programs,
            programs: map,
        }
    }

    pub fn programs(&self) -> Vec<ProgramRef> {
        self.ordered.clone()
    }

    pub fn contains(&self, program: &ProgramRef) -> bool {
        self.programs.contains_key(program)
    }

    pub fn len(&self) -> usize {
        self.ordered.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ordered.is_empty()
    }

    /// The sole program, when exactly one is loaded. `/start_proof` allows
    /// omitting `program` in that case.
    pub fn sole(&self) -> Option<ProgramRef> {
        match self.ordered.as_slice() {
            [only] => Some(only.clone()),
            _ => None,
        }
    }

    /// Every registration payload, in registration order, for replay to a
    /// worker that has just registered.
    pub fn replayable(&self) -> Vec<(ProgramRef, RegisteredProgram)> {
        self.ordered
            .iter()
            .filter_map(|p| {
                self.programs
                    .get(p)
                    .and_then(|entry| entry.clone())
                    .map(|payload| (p.clone(), payload))
            })
            .collect()
    }

    pub fn insert(
        &mut self,
        program: ProgramRef,
        elf: Arc<Vec<u8>>,
        vm_config: Arc<String>,
    ) -> InsertOutcome {
        match self.programs.get(&program) {
            // A seeded entry names a program staged on the workers' disks,
            // which the workers refuse to re-register. Reject it here for the
            // same reason, rather than adopting a payload they will not take.
            Some(None) => InsertOutcome::Conflict,
            Some(Some(existing)) => {
                if *existing.elf == *elf && *existing.vm_config == *vm_config {
                    InsertOutcome::Unchanged
                } else {
                    InsertOutcome::Conflict
                }
            }
            None => {
                self.ordered.push(program.clone());
                self.programs.insert(
                    program,
                    Some(RegisteredProgram {
                        elf,
                        vm_config,
                        baseline: Baseline::Unknown,
                    }),
                );
                InsertOutcome::Inserted
            }
        }
    }

    /// Discard whatever the workers previously reported, so a re-registration
    /// folds their answers from scratch. This is the only way out of
    /// [`Baseline::Mismatch`].
    pub fn reset_baseline(&mut self, program: &ProgramRef) {
        if let Some(Some(entry)) = self.programs.get_mut(program) {
            entry.baseline = Baseline::Unknown;
        }
    }

    /// The verifying key the workers agree on for `program`.
    pub fn baseline(&self, program: &ProgramRef) -> Baseline {
        self.programs
            .get(program)
            .and_then(|entry| entry.as_ref())
            .map(|entry| entry.baseline.clone())
            .unwrap_or_default()
    }

    /// Fold one worker's reported baseline into the program's cached one.
    ///
    /// Returns false once the workers disagree, which poisons the program so
    /// `/program_vk` reports the deployment as inconsistent rather than handing
    /// out a key only some workers prove against. The poison clears only by
    /// re-registering, which drops the entry and starts the fold over.
    pub fn record_baseline(&mut self, program: &ProgramRef, reported: Vec<u8>) -> bool {
        // Absent when the program was concurrently removed, and `None` for a
        // seeded entry served from disk, which derives no baseline. Neither is
        // a disagreement.
        let Some(Some(entry)) = self.programs.get_mut(program) else {
            return true;
        };
        match &entry.baseline {
            Baseline::Unknown => {
                entry.baseline = Baseline::Agreed(Arc::new(reported));
                true
            }
            Baseline::Agreed(agreed) if **agreed == reported => true,
            Baseline::Agreed(_) => {
                entry.baseline = Baseline::Mismatch;
                false
            }
            Baseline::Mismatch => false,
        }
    }

    /// Undo an [`InsertOutcome::Inserted`] whose fan-out failed, so a retry
    /// starts from a clean state rather than seeing a program the workers
    /// never received.
    pub fn remove(&mut self, program: &ProgramRef) {
        self.ordered.retain(|p| p != program);
        self.programs.remove(program);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(elf: &[u8], vm_config: &str) -> (Arc<Vec<u8>>, Arc<String>) {
        (Arc::new(elf.to_vec()), Arc::new(vm_config.to_string()))
    }

    #[test]
    fn insert_is_idempotent_for_identical_bytes() {
        let mut loadout = Loadout::default();
        let (elf, cfg) = payload(b"elf", "config");
        assert!(matches!(
            loadout.insert(ProgramRef::new("a", 0), elf.clone(), cfg.clone()),
            InsertOutcome::Inserted
        ));
        assert!(matches!(
            loadout.insert(ProgramRef::new("a", 0), elf, cfg),
            InsertOutcome::Unchanged
        ));
        assert_eq!(loadout.len(), 1);
    }

    #[test]
    fn insert_conflicts_when_bytes_differ() {
        let mut loadout = Loadout::default();
        let (elf, cfg) = payload(b"elf", "config");
        loadout.insert(ProgramRef::new("a", 0), elf, cfg.clone());
        let (other_elf, _) = payload(b"different", "config");
        assert!(matches!(
            loadout.insert(ProgramRef::new("a", 0), other_elf, cfg),
            InsertOutcome::Conflict
        ));
    }

    #[test]
    fn differing_vm_config_conflicts_under_the_same_program() {
        let mut loadout = Loadout::default();
        let (elf, cfg) = payload(b"elf", "config");
        loadout.insert(ProgramRef::new("a", 0), elf.clone(), cfg);
        let (_, other_cfg) = payload(b"elf", "other config");
        assert!(matches!(
            loadout.insert(ProgramRef::new("a", 0), elf, other_cfg),
            InsertOutcome::Conflict
        ));
    }

    #[test]
    fn seeded_entries_reject_registration_and_are_never_replayed() {
        let mut loadout = Loadout::seeded(vec![ProgramRef::new("a", 0)]);
        assert_eq!(loadout.len(), 1);
        assert!(loadout.replayable().is_empty());

        let (elf, cfg) = payload(b"elf", "config");
        assert!(matches!(
            loadout.insert(ProgramRef::new("a", 0), elf, cfg),
            InsertOutcome::Conflict
        ));
        assert!(loadout.replayable().is_empty());
        assert_eq!(loadout.len(), 1, "a rejected insert must not disturb it");
    }

    #[test]
    fn sole_resolves_only_for_a_single_program() {
        let mut loadout = Loadout::default();
        assert!(loadout.sole().is_none());
        let (elf, cfg) = payload(b"elf", "config");
        loadout.insert(ProgramRef::new("a", 0), elf.clone(), cfg.clone());
        assert_eq!(loadout.sole(), Some(ProgramRef::new("a", 0)));
        loadout.insert(ProgramRef::new("b", 0), elf, cfg);
        assert!(loadout.sole().is_none());
    }

    #[test]
    fn agreeing_workers_cache_one_baseline() {
        let mut loadout = Loadout::default();
        let (elf, cfg) = payload(b"elf", "config");
        loadout.insert(ProgramRef::new("a", 0), elf, cfg);
        assert_eq!(
            loadout.baseline(&ProgramRef::new("a", 0)),
            Baseline::Unknown
        );

        assert!(loadout.record_baseline(&ProgramRef::new("a", 0), b"vk".to_vec()));
        assert!(loadout.record_baseline(&ProgramRef::new("a", 0), b"vk".to_vec()));
        assert_eq!(
            loadout.baseline(&ProgramRef::new("a", 0)),
            Baseline::Agreed(Arc::new(b"vk".to_vec()))
        );
    }

    #[test]
    fn a_disagreeing_worker_poisons_the_baseline() {
        let mut loadout = Loadout::default();
        let (elf, cfg) = payload(b"elf", "config");
        loadout.insert(ProgramRef::new("a", 0), elf, cfg);
        loadout.record_baseline(&ProgramRef::new("a", 0), b"vk".to_vec());

        assert!(!loadout.record_baseline(&ProgramRef::new("a", 0), b"other".to_vec()));
        assert_eq!(
            loadout.baseline(&ProgramRef::new("a", 0)),
            Baseline::Mismatch
        );
        // The original bytes must not win a later race back to agreement.
        assert!(!loadout.record_baseline(&ProgramRef::new("a", 0), b"vk".to_vec()));
    }

    #[test]
    fn resetting_clears_a_poisoned_baseline() {
        let mut loadout = Loadout::default();
        let (elf, cfg) = payload(b"elf", "config");
        loadout.insert(ProgramRef::new("a", 0), elf, cfg);
        loadout.record_baseline(&ProgramRef::new("a", 0), b"vk".to_vec());
        loadout.record_baseline(&ProgramRef::new("a", 0), b"other".to_vec());

        loadout.reset_baseline(&ProgramRef::new("a", 0));
        assert_eq!(
            loadout.baseline(&ProgramRef::new("a", 0)),
            Baseline::Unknown
        );
        assert!(loadout.record_baseline(&ProgramRef::new("a", 0), b"vk".to_vec()));
    }

    #[test]
    fn remove_undoes_an_insert() {
        let mut loadout = Loadout::default();
        let (elf, cfg) = payload(b"elf", "config");
        loadout.insert(ProgramRef::new("a", 0), elf, cfg);
        loadout.remove(&ProgramRef::new("a", 0));
        assert!(loadout.is_empty());
        assert!(!loadout.contains(&ProgramRef::new("a", 0)));
    }
}
