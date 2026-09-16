//! One `frankenctl verify` plan or run over an inventory's sources.
//!
//! A run takes every source's writer lock (see [`crate::downloads`]) before it
//! examines any artifact path and holds them all until its final report is
//! published. It checks the selected files in inventory order, confirms when the
//! window closes that every checked path still names the file it judged, and
//! publishes the report durably when `--report` is given: as `running` before
//! the first artifact is examined, then in its final state. `complete` means
//! every selected check ran, whatever it found. It is not readiness, admission
//! or qualification, which nothing here grants.
//!
//! A SIGINT, SIGTERM or SIGHUP stops the run within one chunk. The report says
//! `interrupted`, marks what was not checked, and the process exits 128 plus the
//! signal number. Locks are released when the process exits, however it exits.

use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::downloads::{LockError, QueueLock};
use crate::inventory::{Consumer, Inventory, Origin, Source, UncoveredClass, INVENTORY_SCHEMA};
use crate::manifest::{Artifact, ArtifactFile};
use crate::report::{self, ReportError};
use crate::verify::{self, Outcome, Summary, HASH_BUFFER_BYTES};

pub const PLAN_SCHEMA: &str = "frankenctl-artifact-verification-plan/2";
pub const RUN_SCHEMA: &str = "frankenctl-artifact-verification-run/2";

/// What a run does not do, stated in every run report.
pub const NOT_PERFORMED: &[&str] = &[
    "download or transfer of any byte",
    "repair, deletion, rename or promotion of any artifact",
    "model admission, readiness or functional qualification",
    "completion-stamp or recorded-status lookup: every verdict comes from bytes read in this run",
    "examination of loaded model files declared unverifiable or unaccounted: they are listed, not read",
];

/// How the run coordinates with writers, stated in every run report.
pub const LOCK_SEMANTICS: &str = "flock(2) LOCK_EX|LOCK_NB on every inventoried source's writer lock, \
taken before the first artifact path is examined and held until the final report is published; \
a refusal releases every lock already taken before anything is examined";

// --- selection -------------------------------------------------------------

/// A `--source` or `--artifact` selection that does not name the inventory exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    UnknownSource(String),
    UnknownArtifact(String),
    SelectedTwice(String),
    OutsideSelectedSources { artifact: String, source: String },
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSource(id) => write!(f, "no source with id {id:?} in the inventory"),
            Self::UnknownArtifact(key) => write!(f, "no artifact with key {key:?}"),
            Self::SelectedTwice(name) => write!(f, "{name:?} was selected more than once"),
            Self::OutsideSelectedSources { artifact, source } => write!(
                f,
                "artifact {artifact:?} belongs to source {source:?}, which is not selected"
            ),
        }
    }
}

impl std::error::Error for SelectionError {}

/// The artifacts a plan or run covers, in inventory order.
#[derive(Debug)]
pub struct Selection<'a> {
    pub items: Vec<(&'a Source, &'a Artifact)>,
    pub source_ids: Vec<String>,
    pub all_artifacts: bool,
}

/// Every artifact of the named sources (all when none), narrowed to the named
/// artifact keys (all when none). A name that matches nothing, or is given
/// twice, is an error rather than a silently different selection.
pub fn select<'a>(
    inventory: &'a Inventory,
    sources: &[String],
    artifacts: &[String],
) -> Result<Selection<'a>, SelectionError> {
    let mut wanted_sources = BTreeSet::new();
    for id in sources {
        if !inventory.sources.iter().any(|source| &source.id == id) {
            return Err(SelectionError::UnknownSource(id.clone()));
        }
        if !wanted_sources.insert(id.as_str()) {
            return Err(SelectionError::SelectedTwice(id.clone()));
        }
    }
    let mut wanted_artifacts = BTreeSet::new();
    for key in artifacts {
        let Some(owner) = inventory
            .sources
            .iter()
            .find(|source| source.manifest.artifacts.iter().any(|a| &a.key == key))
        else {
            return Err(SelectionError::UnknownArtifact(key.clone()));
        };
        if !wanted_sources.is_empty() && !wanted_sources.contains(owner.id.as_str()) {
            return Err(SelectionError::OutsideSelectedSources {
                artifact: key.clone(),
                source: owner.id.clone(),
            });
        }
        if !wanted_artifacts.insert(key.as_str()) {
            return Err(SelectionError::SelectedTwice(key.clone()));
        }
    }
    let items: Vec<(&Source, &Artifact)> = inventory
        .sources
        .iter()
        .filter(|source| wanted_sources.is_empty() || wanted_sources.contains(source.id.as_str()))
        .flat_map(|source| source.manifest.artifacts.iter().map(move |a| (source, a)))
        .filter(|(_, artifact)| {
            wanted_artifacts.is_empty() || wanted_artifacts.contains(artifact.key.as_str())
        })
        .collect();
    let total: usize = inventory
        .sources
        .iter()
        .map(|source| source.manifest.artifacts.len())
        .sum();
    Ok(Selection {
        all_artifacts: items.len() == total,
        source_ids: sources.to_vec(),
        items,
    })
}

// --- identities and summaries ------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct InventoryIdentity {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub schema: &'static str,
}

impl InventoryIdentity {
    fn of(origin: &Origin) -> Option<Self> {
        match origin {
            Origin::File {
                path,
                bytes,
                sha256,
            } => Some(Self {
                path: display(path),
                bytes: *bytes,
                sha256: sha256.as_str().to_string(),
                schema: INVENTORY_SCHEMA,
            }),
            Origin::ManifestArgument => None,
        }
    }
}

/// One source's manifest identity and lock.
#[derive(Debug, Clone, Serialize)]
pub struct SourceSummary {
    pub id: String,
    pub format: &'static str,
    pub manifest: String,
    pub manifest_bytes: u64,
    pub manifest_sha256: String,
    pub built_at: Option<String>,
    pub total_bytes: u64,
    pub artifacts: usize,
    pub files: usize,
    pub writer_lock: String,
    pub writers: Vec<String>,
}

impl SourceSummary {
    fn of(source: &Source) -> Self {
        let manifest = &source.manifest;
        Self {
            id: source.id.clone(),
            format: manifest.format,
            manifest: display(&source.manifest_path),
            manifest_bytes: manifest.bytes,
            manifest_sha256: manifest.sha256.as_str().to_string(),
            built_at: manifest.built_at.clone(),
            total_bytes: manifest.total_bytes,
            artifacts: manifest.artifacts.len(),
            files: manifest.file_count(),
            writer_lock: display(&source.writer_lock),
            writers: source.writers.clone(),
        }
    }
}

/// What the selection covers.
#[derive(Debug, Clone, Serialize)]
pub struct SelectionSummary {
    pub all_artifacts: bool,
    pub source_ids: Vec<String>,
    pub artifact_keys: Vec<String>,
    pub files: usize,
    pub bytes: u64,
    pub publisher_sha256_files: usize,
    pub size_only_files: usize,
}

impl SelectionSummary {
    fn of(selection: &Selection) -> Self {
        let files = || {
            selection
                .items
                .iter()
                .flat_map(|(_, artifact)| artifact.files.iter())
        };
        let total = files().count();
        let hashed = files().filter(|file| file.sha256.is_some()).count();
        Self {
            all_artifacts: selection.all_artifacts,
            source_ids: selection.source_ids.clone(),
            artifact_keys: selection
                .items
                .iter()
                .map(|(_, artifact)| artifact.key.clone())
                .collect(),
            files: total,
            bytes: files().map(|file| file.size).sum(),
            publisher_sha256_files: hashed,
            size_only_files: total - hashed,
        }
    }
}

// --- loaded model files --------------------------------------------------------

/// How the inventory accounts for one loaded model file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "coverage", rename_all = "kebab-case")]
pub enum Coverage {
    /// An inventoried source records its identity; a run checks it when that
    /// artifact is selected.
    InventorySource { source: String, artifact: String },
    /// Declared impossible to verify, with the reason. Never read.
    DeclaredUnverifiable { class: UncoveredClass, reason: String },
    /// Neither inventoried nor declared. Never read.
    Unaccounted,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConsumerRow {
    pub path: String,
    pub consumed_by: Vec<String>,
    #[serde(flatten)]
    pub coverage: Coverage,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ConsumerSummary {
    pub paths: usize,
    pub in_inventory_sources: usize,
    pub declared_unverifiable: usize,
    pub unaccounted: usize,
    pub stale_declarations: usize,
}

/// Every loaded model file and how the inventory accounts for it.
#[derive(Debug, Clone, Serialize)]
pub struct ConsumerCoverage {
    pub scope: &'static str,
    pub rows: Vec<ConsumerRow>,
    /// Declared uncovered, but nothing loads it any more.
    pub stale_declarations: Vec<String>,
    pub summary: ConsumerSummary,
}

/// Account for every loaded model file against the inventory.
pub fn consumer_coverage(inventory: &Inventory, consumers: Vec<Consumer>) -> ConsumerCoverage {
    let mut summary = ConsumerSummary {
        paths: consumers.len(),
        ..ConsumerSummary::default()
    };
    let mut rows = Vec::with_capacity(consumers.len());
    for consumer in consumers {
        let path = Path::new(&consumer.path);
        let owner = inventory.sources.iter().find_map(|source| {
            source.manifest.artifacts.iter().find_map(|artifact| {
                artifact
                    .files
                    .iter()
                    .any(|file| file.destination == path)
                    .then(|| (source.id.clone(), artifact.key.clone()))
            })
        });
        let coverage = if let Some((source, artifact)) = owner {
            summary.in_inventory_sources += 1;
            Coverage::InventorySource { source, artifact }
        } else if let Some(entry) = inventory.uncovered.iter().find(|e| e.path == path) {
            summary.declared_unverifiable += 1;
            Coverage::DeclaredUnverifiable {
                class: entry.class,
                reason: entry.reason.clone(),
            }
        } else {
            summary.unaccounted += 1;
            Coverage::Unaccounted
        };
        rows.push(ConsumerRow {
            path: consumer.path,
            consumed_by: consumer.consumed_by,
            coverage,
        });
    }
    let stale_declarations: Vec<String> = inventory
        .uncovered
        .iter()
        .filter(|entry| !rows.iter().any(|row| Path::new(&row.path) == entry.path))
        .map(|entry| display(&entry.path))
        .collect();
    summary.stale_declarations = stale_declarations.len();
    ConsumerCoverage {
        scope: "every llama-models.ini preset's model and mmproj, and every services/sidecar-*.env SIDECAR_MODEL and --mmproj",
        rows,
        stale_declarations,
        summary,
    }
}

// --- plan ------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct PlannedFile {
    pub repo_path: String,
    pub destination: String,
    pub size: u64,
    pub sha256: Option<String>,
    /// `sha256-and-size` or `size-only-no-published-hash`.
    pub check: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlannedArtifact {
    pub source: String,
    pub key: String,
    pub capability: String,
    pub repository: String,
    pub revision: String,
    pub files: Vec<PlannedFile>,
}

/// A metadata-only plan. Building it reads the inventory, its manifests and the
/// preset and sidecar files, and no artifact path.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub schema: &'static str,
    pub mode: &'static str,
    pub artifact_access: &'static str,
    pub inventory: Option<InventoryIdentity>,
    pub sources: Vec<SourceSummary>,
    pub writer_locks_run_takes: Vec<String>,
    pub selection: SelectionSummary,
    pub artifacts: Vec<PlannedArtifact>,
    pub consumers: Option<ConsumerCoverage>,
}

impl Plan {
    /// 1 when a loaded model file is unaccounted for, else 0.
    pub fn exit_code(&self) -> u8 {
        match &self.consumers {
            Some(coverage) if coverage.summary.unaccounted > 0 => verify::EXIT_INTEGRITY,
            _ => verify::EXIT_OK,
        }
    }
}

/// Describe what a run over `selection` would check and lock.
pub fn plan(
    inventory: &Inventory,
    selection: &Selection,
    consumers: Option<ConsumerCoverage>,
) -> Plan {
    Plan {
        schema: PLAN_SCHEMA,
        mode: "plan",
        artifact_access: "none: no artifact path was examined and no lock was taken",
        inventory: InventoryIdentity::of(&inventory.origin),
        sources: inventory.sources.iter().map(SourceSummary::of).collect(),
        writer_locks_run_takes: inventory
            .sources
            .iter()
            .map(|source| display(&source.writer_lock))
            .collect(),
        selection: SelectionSummary::of(selection),
        artifacts: selection
            .items
            .iter()
            .map(|(source, artifact)| PlannedArtifact {
                source: source.id.clone(),
                key: artifact.key.clone(),
                capability: artifact.capability.clone(),
                repository: artifact.repository.clone(),
                revision: artifact.revision.clone(),
                files: artifact
                    .files
                    .iter()
                    .map(|file| PlannedFile {
                        repo_path: file.repo_path.clone(),
                        destination: display(&file.destination),
                        size: file.size,
                        sha256: file.sha256.as_ref().map(|d| d.as_str().to_string()),
                        check: if file.sha256.is_some() {
                            "sha256-and-size"
                        } else {
                            "size-only-no-published-hash"
                        },
                    })
                    .collect(),
            })
            .collect(),
        consumers,
    }
}

// --- run ---------------------------------------------------------------------------

/// Where a run is in its lifecycle. Only `complete` means every selected check ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunState {
    /// Locks are held and checks are under way; no verdict yet.
    Running,
    /// Every selected check ran. The outcomes say what it found.
    Complete,
    /// A signal stopped the run; files it did not finish are `not-checked`.
    Interrupted,
    /// Another process holds a writer lock; nothing was examined.
    RefusedLocked,
    /// A writer lock could not be opened or taken; nothing was examined.
    LockFailed,
    /// The running report could not be published; nothing was examined.
    Aborted,
}

#[derive(Debug, Clone, Serialize)]
pub struct LockRefusal {
    pub path: String,
    pub error: String,
    pub errno: Option<i32>,
    pub errno_name: Option<&'static str>,
}

impl LockRefusal {
    fn of(error: &LockError) -> Self {
        let errno = error.os_error().and_then(std::io::Error::raw_os_error);
        Self {
            path: display(error.path()),
            error: error.to_string(),
            errno,
            errno_name: errno.and_then(verify::errno_name),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LockRecord {
    pub semantics: &'static str,
    pub required: Vec<String>,
    pub taken: Vec<String>,
    pub refused: Option<LockRefusal>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReportFailure {
    pub operation: &'static str,
    pub path: String,
    pub errno: Option<i32>,
    pub errno_name: Option<&'static str>,
    pub error: String,
}

impl ReportFailure {
    fn of(error: &ReportError) -> Self {
        let errno = error.error.raw_os_error();
        Self {
            operation: error.operation,
            path: display(&error.path),
            errno,
            errno_name: errno.and_then(verify::errno_name),
            error: error.error.to_string(),
        }
    }
}

/// One file's recorded identity and what the check found.
#[derive(Debug, Clone, Serialize)]
pub struct FileResult {
    pub source: String,
    pub key: String,
    pub capability: String,
    pub repository: String,
    pub revision: String,
    pub repo_path: String,
    pub destination: String,
    pub expected_size: u64,
    pub expected_sha256: Option<String>,
    #[serde(flatten)]
    pub outcome: Outcome,
}

/// The report of one run, in whichever state it was published.
#[derive(Debug, Clone, Serialize)]
pub struct RunReport {
    pub schema: &'static str,
    pub mode: &'static str,
    pub state: RunState,
    pub inventory: Option<InventoryIdentity>,
    pub sources: Vec<SourceSummary>,
    pub selection: SelectionSummary,
    pub writer_locks: LockRecord,
    pub boot_id: Option<String>,
    /// systemd's `INVOCATION_ID`, when the run is a unit activation.
    pub invocation_id: Option<String>,
    pub pid: u32,
    pub started_at_unix: u64,
    pub finished_at_unix: Option<u64>,
    pub hash_buffer_bytes: usize,
    pub not_performed: &'static [&'static str],
    pub results: Vec<FileResult>,
    pub consumers: Option<ConsumerCoverage>,
    pub summary: Summary,
    pub interrupted_by_signal: Option<i32>,
    pub report_failure: Option<ReportFailure>,
    /// The status the checks themselves earned.
    pub verdict_exit_code: Option<u8>,
    /// The process exit status: the verdict, or 73 when publication failed.
    pub exit_code: Option<u8>,
}

/// Test and caller seams: progress after each hashed chunk, and cancellation.
pub struct Hooks<'h> {
    pub after_chunk: &'h mut dyn FnMut(&ArtifactFile, u64),
    pub cancelled: &'h dyn Fn() -> Option<i32>,
}

/// Lock, check, confirm and publish. The returned report is final.
pub fn run(
    inventory: &Inventory,
    selection: &Selection,
    consumers: Option<ConsumerCoverage>,
    report_path: Option<&Path>,
    hooks: Hooks,
) -> RunReport {
    let Hooks {
        after_chunk,
        cancelled,
    } = hooks;
    let mut report = RunReport {
        schema: RUN_SCHEMA,
        mode: "run",
        state: RunState::Running,
        inventory: InventoryIdentity::of(&inventory.origin),
        sources: inventory.sources.iter().map(SourceSummary::of).collect(),
        selection: SelectionSummary::of(selection),
        writer_locks: LockRecord {
            semantics: LOCK_SEMANTICS,
            required: inventory
                .sources
                .iter()
                .map(|source| display(&source.writer_lock))
                .collect(),
            taken: Vec::new(),
            refused: None,
        },
        boot_id: std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .ok()
            .map(|id| id.trim().to_string()),
        invocation_id: std::env::var("INVOCATION_ID").ok(),
        pid: std::process::id(),
        started_at_unix: unix_now(),
        finished_at_unix: None,
        hash_buffer_bytes: HASH_BUFFER_BYTES,
        not_performed: NOT_PERFORMED,
        results: Vec::new(),
        consumers,
        summary: Summary::default(),
        interrupted_by_signal: None,
        report_failure: None,
        verdict_exit_code: None,
        exit_code: None,
    };
    if let Some(signal) = cancelled() {
        report.state = RunState::Interrupted;
        report.interrupted_by_signal = Some(signal);
        return conclude(report, signal_exit(signal), report_path);
    }

    let mut held: Vec<QueueLock> = Vec::with_capacity(inventory.sources.len());
    for source in &inventory.sources {
        match QueueLock::acquire(&source.writer_lock) {
            Ok(lock) => {
                report.writer_locks.taken.push(display(lock.path()));
                held.push(lock);
            }
            Err(error) => {
                drop(held);
                let code = error.exit_code();
                report.state = if code == verify::EXIT_LOCKED {
                    RunState::RefusedLocked
                } else {
                    RunState::LockFailed
                };
                report.writer_locks.refused = Some(LockRefusal::of(&error));
                return conclude(report, code, report_path);
            }
        }
    }

    if let Some(path) = report_path {
        if let Err(error) = publish(&report, path) {
            drop(held);
            report.state = RunState::Aborted;
            report.report_failure = Some(ReportFailure::of(&error));
            report.finished_at_unix = Some(unix_now());
            report.exit_code = Some(verify::EXIT_CANT_CREATE);
            return report;
        }
    }

    let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
    let mut checked: Vec<&ArtifactFile> = Vec::new();
    let mut signal = None;
    for &(source, artifact) in &selection.items {
        for file in &artifact.files {
            signal = signal.or_else(cancelled);
            let outcome = match signal {
                Some(number) => Outcome::NotChecked {
                    reason: format!("not started: the run was interrupted by signal {number}"),
                },
                None => verify::verify_file_with(
                    file,
                    &mut buffer,
                    &mut |bytes| after_chunk(file, bytes),
                    cancelled,
                ),
            };
            report.results.push(FileResult {
                source: source.id.clone(),
                key: artifact.key.clone(),
                capability: artifact.capability.clone(),
                repository: artifact.repository.clone(),
                revision: artifact.revision.clone(),
                repo_path: file.repo_path.clone(),
                destination: display(&file.destination),
                expected_size: file.size,
                expected_sha256: file.sha256.as_ref().map(|d| d.as_str().to_string()),
                outcome,
            });
            checked.push(file);
        }
    }
    signal = signal.or_else(cancelled);
    if signal.is_none() {
        for (result, file) in report.results.iter_mut().zip(&checked) {
            let outcome = std::mem::replace(&mut result.outcome, Outcome::Missing);
            result.outcome = verify::confirm_at_window_close(file, outcome);
        }
    }

    let mut summary = Summary::default();
    for result in &report.results {
        summary.record(&result.outcome);
    }
    let unaccounted = report
        .consumers
        .as_ref()
        .map_or(0, |coverage| coverage.summary.unaccounted);
    let code = match signal {
        Some(number) => {
            report.state = RunState::Interrupted;
            report.interrupted_by_signal = Some(number);
            signal_exit(number)
        }
        None => {
            report.state = RunState::Complete;
            verdict(&summary, unaccounted)
        }
    };
    report.summary = summary;
    let report = conclude(report, code, report_path);
    drop(held);
    report
}

/// 1 when the files passed but a loaded model file is unaccounted for.
fn verdict(summary: &Summary, unaccounted: usize) -> u8 {
    match summary.exit_code() {
        verify::EXIT_OK if unaccounted > 0 => verify::EXIT_INTEGRITY,
        code => code,
    }
}

fn signal_exit(signal: i32) -> u8 {
    u8::try_from(i32::from(verify::EXIT_SIGNAL_BASE) + signal).unwrap_or(u8::MAX)
}

/// Stamp the final status and publish; a failed publication turns the exit into 73.
fn conclude(mut report: RunReport, code: u8, report_path: Option<&Path>) -> RunReport {
    report.finished_at_unix = Some(unix_now());
    report.verdict_exit_code = Some(code);
    report.exit_code = Some(code);
    if let Some(path) = report_path {
        if let Err(error) = publish(&report, path) {
            report.report_failure = Some(ReportFailure::of(&error));
            report.exit_code = Some(verify::EXIT_CANT_CREATE);
        }
    }
    report
}

fn publish(report: &RunReport, path: &Path) -> Result<(), ReportError> {
    let mut bytes = serde_json::to_vec_pretty(report).map_err(|error| ReportError {
        operation: "encode",
        path: path.to_path_buf(),
        error: std::io::Error::other(error),
    })?;
    bytes.push(b'\n');
    report::write_atomic(path, &bytes)
}

// --- cancellation ------------------------------------------------------------------

static CANCELLED_BY: AtomicI32 = AtomicI32::new(0);

extern "C" fn record_cancellation(signal: libc::c_int) {
    // An atomic store is all this handler does: it is async-signal-safe.
    CANCELLED_BY.store(signal, Ordering::SeqCst);
}

/// Make SIGINT, SIGTERM and SIGHUP cancel the run instead of killing the process.
///
/// No `SA_RESTART`: a read blocked when the signal arrives returns `EINTR`, and
/// the hash loop asks [`cancellation`] before retrying it.
pub fn install_cancellation_handlers() -> std::io::Result<()> {
    for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        // SAFETY: an all-zero sigaction is a valid value to fill in; the handler
        // only stores into an atomic, and sigemptyset/sigaction are given
        // pointers to that live local and a null old-action pointer.
        let installed = unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = record_cancellation as extern "C" fn(libc::c_int) as libc::sighandler_t;
            action.sa_flags = 0;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(signal, &action, std::ptr::null_mut())
        };
        if installed != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

/// The signal that cancelled the run, if one has arrived.
pub fn cancellation() -> Option<i32> {
    match CANCELLED_BY.load(Ordering::SeqCst) {
        0 => None,
        signal => Some(signal),
    }
}

fn display(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::Uncovered;
    use crate::manifest::Sha256Hex;
    use serde_json::{json, Value};
    use std::cell::Cell;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicUsize;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().canonicalize().expect("temp dir");
            let dir = root.join(format!(
                "frankenctl-verification-{tag}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&dir).expect("create scratch dir");
            Self(dir)
        }

        fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, bytes).unwrap();
            path
        }

        /// A manifest of two artifacts, `first` and `second`, with one file each.
        fn inventory(&self) -> Inventory {
            let first = self.write("models/first.bin", b"first bytes");
            let second = self.write("models/second.bin", b"second");
            let file = |path: &Path, bytes: &[u8]| {
                json!({"repo_path": path.file_name().unwrap().to_str().unwrap(),
                       "destination": path.to_str().unwrap(), "size": bytes.len(),
                       "sha256": Sha256Hex::of(bytes).as_str()})
            };
            let manifest = json!({
                "schema": "hermes-hf-artifact-queue/1",
                "total_bytes": 17,
                "artifacts": [
                    {"key": "first", "capability": "t", "repository": "o/first", "revision": "r",
                     "files": [file(&first, b"first bytes")]},
                    {"key": "second", "capability": "t", "repository": "o/second", "revision": "r",
                     "files": [file(&second, b"second")]},
                ],
            });
            let path = self.write("download-queue.json", &serde_json::to_vec(&manifest).unwrap());
            Inventory::single_manifest(&path).unwrap()
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn quiet() -> Option<i32> {
        None
    }

    fn run_with(
        inventory: &Inventory,
        report: Option<&Path>,
        after_chunk: &mut dyn FnMut(&ArtifactFile, u64),
        cancelled: &dyn Fn() -> Option<i32>,
    ) -> RunReport {
        let selection = select(inventory, &[], &[]).unwrap();
        run(
            inventory,
            &selection,
            None,
            report,
            Hooks {
                after_chunk,
                cancelled,
            },
        )
    }

    fn outcomes(report: &RunReport) -> Vec<Value> {
        report
            .results
            .iter()
            .map(|result| serde_json::to_value(&result.outcome).unwrap()["outcome"].clone())
            .collect()
    }

    #[test]
    fn selection_is_exact_across_sources_and_artifacts() {
        let scratch = Scratch::new("select");
        let inventory = scratch.inventory();
        assert!(select(&inventory, &[], &[]).unwrap().all_artifacts);
        let one = select(&inventory, &[], &["second".into()]).unwrap();
        assert_eq!(one.items.len(), 1);
        assert!(!one.all_artifacts);
        assert_eq!(
            select(&inventory, &["nope".into()], &[]).unwrap_err(),
            SelectionError::UnknownSource("nope".into())
        );
        assert_eq!(
            select(&inventory, &["manifest".into(), "manifest".into()], &[]).unwrap_err(),
            SelectionError::SelectedTwice("manifest".into())
        );
        assert_eq!(
            select(&inventory, &[], &["absent".into()]).unwrap_err(),
            SelectionError::UnknownArtifact("absent".into())
        );
        assert_eq!(
            select(&inventory, &[], &["first".into(), "first".into()]).unwrap_err(),
            SelectionError::SelectedTwice("first".into())
        );
    }

    #[test]
    fn a_run_holds_the_writer_lock_publishes_running_first_and_then_complete() {
        let scratch = Scratch::new("run");
        let inventory = scratch.inventory();
        let lock = scratch.0.join("download-queue.lock");
        let report_path = scratch.0.join("proofs/latest.json");
        let seen = Cell::new(0);
        let report = run_with(
            &inventory,
            Some(&report_path),
            &mut |_, _| {
                if seen.get() == 0 {
                    // Mid-run: the lock is held, and the published report says
                    // running with nothing judged yet.
                    assert!(matches!(
                        QueueLock::acquire(&lock),
                        Err(LockError::Held { .. })
                    ));
                    let published: Value =
                        serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
                    assert_eq!(published["state"], "running");
                    assert_eq!(published["results"], json!([]));
                    assert!(published["exit_code"].is_null());
                }
                seen.set(seen.get() + 1);
            },
            &quiet,
        );
        assert_eq!(report.state, RunState::Complete);
        assert_eq!(report.exit_code, Some(0));
        assert_eq!(outcomes(&report), [json!("verified-sha256"), json!("verified-sha256")]);
        assert_eq!(report.writer_locks.taken, [lock.display().to_string()]);
        let published: Value =
            serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
        assert_eq!(published["state"], "complete");
        assert_eq!(published["exit_code"], 0);
        assert_eq!(published["summary"]["verified_sha256"], 2);
        QueueLock::acquire(&lock).expect("the lock is released when the run returns");
    }

    #[test]
    fn a_held_writer_lock_refuses_the_run_before_any_artifact_is_examined() {
        let scratch = Scratch::new("held");
        let inventory = scratch.inventory();
        // Unopenable: examining it would have produced an unreadable outcome.
        std::fs::set_permissions(
            scratch.0.join("models/first.bin"),
            std::fs::Permissions::from_mode(0o000),
        )
        .unwrap();
        let _writer = QueueLock::acquire(&scratch.0.join("download-queue.lock")).unwrap();
        let report_path = scratch.0.join("latest.json");
        let report = run_with(
            &inventory,
            Some(&report_path),
            &mut |_, _| panic!("no artifact may be read"),
            &quiet,
        );
        assert_eq!(report.state, RunState::RefusedLocked);
        assert_eq!(report.exit_code, Some(verify::EXIT_LOCKED));
        assert!(report.results.is_empty());
        let refused = report.writer_locks.refused.as_ref().unwrap();
        assert!(refused.path.ends_with("download-queue.lock"));
        let published: Value =
            serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
        assert_eq!(published["state"], "refused-locked");
        assert_eq!(published["exit_code"], 75);
    }

    #[test]
    fn a_signal_mid_read_leaves_no_verdict_and_skips_the_rest() {
        let scratch = Scratch::new("signal");
        let inventory = scratch.inventory();
        let signal = Cell::new(None);
        let report = run_with(
            &inventory,
            None,
            &mut |_, _| signal.set(Some(libc::SIGINT)),
            &|| signal.get(),
        );
        assert_eq!(report.state, RunState::Interrupted);
        assert_eq!(report.interrupted_by_signal, Some(libc::SIGINT));
        assert_eq!(report.exit_code, Some(130));
        assert_eq!(outcomes(&report), [json!("not-checked"), json!("not-checked")]);
        assert_eq!(report.summary.not_checked, 2);
    }

    #[test]
    fn a_signal_before_the_run_takes_no_lock() {
        let scratch = Scratch::new("early");
        let inventory = scratch.inventory();
        let report = run_with(&inventory, None, &mut |_, _| {}, &|| Some(libc::SIGTERM));
        assert_eq!(report.state, RunState::Interrupted);
        assert_eq!(report.exit_code, Some(143));
        assert!(report.writer_locks.taken.is_empty());
        assert!(!scratch.0.join("download-queue.lock").exists());
    }

    #[test]
    fn a_file_replaced_after_its_verdict_is_caught_when_the_window_closes() {
        let scratch = Scratch::new("window");
        let inventory = scratch.inventory();
        let first = scratch.0.join("models/first.bin");
        let report = run_with(
            &inventory,
            None,
            &mut |file, _| {
                if file.repo_path == "second.bin" {
                    // The first file was already verified; replace it with a
                    // same-sized copy of the same bytes.
                    let copy = first.with_extension("copy");
                    std::fs::write(&copy, b"first bytes").unwrap();
                    std::fs::rename(&copy, &first).unwrap();
                }
            },
            &quiet,
        );
        assert_eq!(report.state, RunState::Complete);
        assert_eq!(outcomes(&report), [json!("changed-during-read"), json!("verified-sha256")]);
        assert_eq!(report.exit_code, Some(verify::EXIT_INTEGRITY));
    }

    #[test]
    fn an_unpublishable_report_aborts_before_any_artifact_is_examined() {
        let scratch = Scratch::new("abort");
        let inventory = scratch.inventory();
        let blocker = scratch.write("not-a-directory", b"");
        let report = run_with(
            &inventory,
            Some(&blocker.join("latest.json")),
            &mut |_, _| panic!("no artifact may be read"),
            &quiet,
        );
        assert_eq!(report.state, RunState::Aborted);
        assert_eq!(report.exit_code, Some(verify::EXIT_CANT_CREATE));
        assert!(report.results.is_empty());
        assert_eq!(
            report.report_failure.as_ref().unwrap().operation,
            "create the directory"
        );
        QueueLock::acquire(&scratch.0.join("download-queue.lock")).expect("released");
    }

    #[test]
    fn an_unaccounted_loaded_model_fails_a_run_whose_files_all_passed() {
        let scratch = Scratch::new("consumers");
        let mut inventory = scratch.inventory();
        let destination = inventory.sources[0].manifest.artifacts[0].files[0]
            .destination
            .display()
            .to_string();
        inventory.uncovered.push(Uncovered {
            path: PathBuf::from("/m/converted.gguf"),
            class: UncoveredClass::LocalConversion,
            reason: "converted here".into(),
        });
        inventory.uncovered.push(Uncovered {
            path: PathBuf::from("/m/stale.gguf"),
            class: UncoveredClass::NoRecordedIdentity,
            reason: "no longer loaded".into(),
        });
        let coverage = consumer_coverage(
            &inventory,
            vec![
                Consumer {
                    path: destination,
                    consumed_by: vec!["preset".into()],
                },
                Consumer {
                    path: "/m/converted.gguf".into(),
                    consumed_by: vec!["sidecar".into()],
                },
                Consumer {
                    path: "/m/unknown.gguf".into(),
                    consumed_by: vec!["preset".into()],
                },
            ],
        );
        assert_eq!(
            coverage.summary,
            ConsumerSummary {
                paths: 3,
                in_inventory_sources: 1,
                declared_unverifiable: 1,
                unaccounted: 1,
                stale_declarations: 1
            }
        );
        assert_eq!(coverage.stale_declarations, ["/m/stale.gguf"]);
        let selection = select(&inventory, &[], &[]).unwrap();
        let planned = plan(&inventory, &selection, Some(coverage.clone()));
        assert_eq!(planned.exit_code(), verify::EXIT_INTEGRITY);
        let report = run(
            &inventory,
            &selection,
            Some(coverage),
            None,
            Hooks {
                after_chunk: &mut |_, _| {},
                cancelled: &quiet,
            },
        );
        assert_eq!(report.summary.verified_sha256, 2);
        assert_eq!(report.exit_code, Some(verify::EXIT_INTEGRITY));
    }
}
