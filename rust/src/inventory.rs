//! The maintained inventory of every artifact set `frankenctl verify` checks.
//!
//! `config/artifact-inventory.json` names each manifest that records exact
//! artifact identity and the writer lock that manifest's writers hold. Three
//! manifest formats exist in the repository, and each is read here as strictly
//! as [`crate::manifest`] reads a download queue:
//!
//! * `hermes-hf-artifact-queue/1`: the download queues.
//! * `sha256-size-name-tsv/1`: one `sha256<TAB>size<TAB>name` row per file, as
//!   the GLM shard manifest records them. The TSV names no repository, revision
//!   or directory, so the inventory supplies them, and the row count it expects.
//! * `hf-single-file-manifest/1`: one pinned file, as the projector manifest
//!   records it. Its `status` and `verified_at` fields are ignored: they are a
//!   past claim, and every run judges the bytes it reads.
//!
//! The inventory also accounts for every model file the router presets and the
//! sidecar environment files load. A loaded path is either a destination of an
//! inventoried source or declared under `uncovered` with the reason it cannot be
//! verified: a local conversion, a digest recorded without a size, or no recorded
//! identity at all. Anything else is reported unaccounted, so a preset added
//! without inventorying its weights cannot pass as verified. Research candidates
//! have no place here: the inventory has no format for them, and no executable
//! manifest may name one.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Map, Value};

use crate::config;
use crate::downloads::lock_path_for;
use crate::error::ConfigError;
use crate::manifest::{
    assemble, check_destination, check_repo_path, check_repository, check_revision, read_bounded,
    Artifact, ArtifactFile, ManifestError, Parse, QueueManifest, Sha256Hex, MAX_MANIFEST_BYTES,
    QUEUE_SCHEMA,
};

/// The only inventory schema this module accepts.
pub const INVENTORY_SCHEMA: &str = "frankenctl-artifact-inventory/1";
/// Where `--inventory` looks when given no path, relative to the repository root.
pub const DEFAULT_INVENTORY: &str = "config/artifact-inventory.json";
/// `sha256<TAB>size<TAB>name` rows.
pub const TSV_FORMAT: &str = "sha256-size-name-tsv/1";
/// One pinned Hugging Face file.
pub const SINGLE_FILE_FORMAT: &str = "hf-single-file-manifest/1";
/// Inventories are read with this bound; the tracked one is a few KiB.
pub const MAX_INVENTORY_BYTES: u64 = 1024 * 1024;
/// Most sources one inventory may name.
pub const MAX_SOURCES: usize = 64;
/// Most uncovered declarations one inventory may carry.
pub const MAX_UNCOVERED: usize = 1_000;
/// The router presets, relative to the repository root.
pub const PRESETS: &str = "llama-models.ini";
/// The directory of `sidecar-*.env` files, relative to the repository root.
pub const SIDECAR_ENV_DIRECTORY: &str = "services";

const QUEUE_SOURCE_KEYS: &[&str] = &["format", "id", "manifest", "writer_lock", "writers"];
const DESCRIBED_SOURCE_KEYS: &[&str] = &["artifact", "format", "id", "manifest", "writer_lock", "writers"];

/// Why a model file that is loaded cannot be verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UncoveredClass {
    /// Bytes produced on this host from verified inputs; no publisher digest
    /// exists for them.
    LocalConversion,
    /// A digest is recorded, but no exact size, so there is no size to check first.
    DigestWithoutRecordedSize,
    /// No tracked record states the file's digest or size.
    NoRecordedIdentity,
}

impl UncoveredClass {
    const NAMES: [(&'static str, Self); 3] = [
        ("local-conversion", Self::LocalConversion),
        ("digest-without-recorded-size", Self::DigestWithoutRecordedSize),
        ("no-recorded-identity", Self::NoRecordedIdentity),
    ];

    fn parse(text: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .find(|(name, _)| *name == text)
            .map(|(_, class)| *class)
    }
}

/// A loaded model file declared impossible to verify, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uncovered {
    pub path: PathBuf,
    pub class: UncoveredClass,
    pub reason: String,
}

/// One inventoried manifest, validated, with its writer lock.
#[derive(Debug, Clone)]
pub struct Source {
    pub id: String,
    pub manifest_path: PathBuf,
    pub writer_lock: PathBuf,
    /// The programs that write this manifest's destinations, for the report.
    pub writers: Vec<String>,
    pub manifest: QueueManifest,
}

/// Where the sources came from.
#[derive(Debug, Clone)]
pub enum Origin {
    /// An inventory file, with the identity of the exact bytes read.
    File {
        path: PathBuf,
        bytes: u64,
        sha256: Sha256Hex,
    },
    /// One manifest named with `--manifest`; there is no inventory file.
    ManifestArgument,
}

/// A validated inventory.
#[derive(Debug, Clone)]
pub struct Inventory {
    pub origin: Origin,
    pub sources: Vec<Source>,
    pub uncovered: Vec<Uncovered>,
}

impl Inventory {
    /// Read and validate the inventory at `path` and every manifest it names,
    /// resolving the inventory's relative paths against `root`.
    ///
    /// One bad entry, or one manifest that fails its format, rejects the whole
    /// inventory rather than leaving a quietly smaller set to verify.
    pub fn load(root: &Path, path: &Path) -> Result<Self, ManifestError> {
        let bytes = read_bounded(path, MAX_INVENTORY_BYTES)?;
        let p = Parse::new(path);
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|e| p.error("$", format!("not valid JSON: {e}")))?;
        let top = p.object(&value, "$")?;
        p.keys(top, "$", &["integrity", "schema", "sources", "uncovered"])?;
        let schema = p.text(top, "schema", "$")?;
        if schema != INVENTORY_SCHEMA {
            return Err(p.error(
                "$.schema",
                format!("expected {INVENTORY_SCHEMA:?}, got {schema:?}"),
            ));
        }
        p.text(top, "integrity", "$")?;
        let rows = p.array(top, "sources", "$", MAX_SOURCES)?;
        let mut sources: Vec<Source> = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let at = format!("$.sources[{index}]");
            let source = load_source(root, &p, row, &at)?;
            if sources.iter().any(|seen| seen.id == source.id) {
                return Err(p.error(
                    &format!("{at}.id"),
                    format!("source id {:?} appears more than once", source.id),
                ));
            }
            sources.push(source);
        }
        let owners = cross_check(&p, &sources)?;
        let uncovered = load_uncovered(&p, top, &owners)?;
        Ok(Self {
            origin: Origin::File {
                path: path.to_path_buf(),
                bytes: bytes.len() as u64,
                sha256: Sha256Hex::of(&bytes),
            },
            sources,
            uncovered,
        })
    }

    /// One download-queue manifest as a single source, locked by its paired lock.
    pub fn single_manifest(path: &Path) -> Result<Self, ManifestError> {
        let manifest = QueueManifest::load(path)?;
        Ok(Self {
            origin: Origin::ManifestArgument,
            sources: vec![Source {
                id: "manifest".to_string(),
                manifest_path: path.to_path_buf(),
                writer_lock: lock_path_for(path),
                writers: Vec::new(),
                manifest,
            }],
            uncovered: Vec::new(),
        })
    }
}

fn load_source(root: &Path, p: &Parse, row: &Value, at: &str) -> Result<Source, ManifestError> {
    let fields = p.object(row, at)?;
    let named = p.text(fields, "format", at)?;
    let (format, keys) = match named.as_str() {
        QUEUE_SCHEMA => (QUEUE_SCHEMA, QUEUE_SOURCE_KEYS),
        TSV_FORMAT => (TSV_FORMAT, DESCRIBED_SOURCE_KEYS),
        SINGLE_FILE_FORMAT => (SINGLE_FILE_FORMAT, DESCRIBED_SOURCE_KEYS),
        other => {
            return Err(p.error(
                &format!("{at}.format"),
                format!(
                    "expected {QUEUE_SCHEMA:?}, {TSV_FORMAT:?} or {SINGLE_FILE_FORMAT:?}, got {other:?}"
                ),
            ))
        }
    };
    p.keys(fields, at, keys)?;
    let id = p.text(fields, "id", at)?;
    if id.starts_with('-')
        || !id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err(p.error(
            &format!("{at}.id"),
            format!("expected lowercase letters, digits and '-', got {id:?}"),
        ));
    }
    let manifest_path = repo_relative(root, p, fields, "manifest", at)?;
    let writer_lock = repo_relative(root, p, fields, "writer_lock", at)?;
    let paired = lock_path_for(&manifest_path);
    if writer_lock != paired {
        return Err(p.error(
            &format!("{at}.writer_lock"),
            format!(
                "must be {}: the manifest's path with the extension lock, which is the lock its writers take",
                paired.display()
            ),
        ));
    }
    let writers = writers(p, fields, at)?;
    let manifest = match format {
        QUEUE_SCHEMA => QueueManifest::load(&manifest_path)?,
        TSV_FORMAT => load_tsv(&manifest_path, &TsvArtifact::parse(p, fields, at)?)?,
        _ => load_single_file(&manifest_path, &NamedArtifact::parse(p, fields, at)?)?,
    };
    Ok(Source {
        id,
        manifest_path,
        writer_lock,
        writers,
        manifest,
    })
}

/// A repository-relative path in normal form, resolved against `root`.
fn repo_relative(
    root: &Path,
    p: &Parse,
    fields: &Map<String, Value>,
    key: &str,
    at: &str,
) -> Result<PathBuf, ManifestError> {
    let text = p.text(fields, key, at)?;
    check_repo_path(&text).map_err(|m| p.error(&format!("{at}.{key}"), m))?;
    Ok(root.join(text))
}

fn writers(p: &Parse, fields: &Map<String, Value>, at: &str) -> Result<Vec<String>, ManifestError> {
    let location = format!("{at}.writers");
    let items = match fields.get("writers") {
        Some(Value::Array(items)) if !items.is_empty() => items,
        Some(Value::Array(_)) => return Err(p.error(&location, "must list at least one writer")),
        Some(other) => return Err(p.error(&location, format!("expected an array, got {other}"))),
        None => return Err(p.error(&location, "required field is missing")),
    };
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let at = format!("{location}[{index}]");
            match item {
                Value::String(text) => check_repo_path(text)
                    .map(|()| text.clone())
                    .map_err(|m| p.error(&at, m)),
                other => Err(p.error(&at, format!("expected a string, got {other}"))),
            }
        })
        .collect()
}

/// Artifact keys unique across sources, and one owner per destination.
fn cross_check(p: &Parse, sources: &[Source]) -> Result<BTreeMap<PathBuf, String>, ManifestError> {
    let mut keys: BTreeMap<&str, &str> = BTreeMap::new();
    let mut owners: BTreeMap<PathBuf, String> = BTreeMap::new();
    for (index, source) in sources.iter().enumerate() {
        let at = format!("$.sources[{index}]");
        for artifact in &source.manifest.artifacts {
            if let Some(previous) = keys.insert(artifact.key.as_str(), source.id.as_str()) {
                return Err(p.error(
                    &at,
                    format!(
                        "artifact key {:?} is declared by both {previous} and {}",
                        artifact.key, source.id
                    ),
                ));
            }
            for file in &artifact.files {
                let owner = format!("{}/{}/{}", source.id, artifact.key, file.repo_path);
                if let Some(previous) = owners.insert(file.destination.clone(), owner.clone()) {
                    return Err(p.error(
                        &at,
                        format!(
                            "{} is claimed by both {previous} and {owner}",
                            file.destination.display()
                        ),
                    ));
                }
            }
        }
    }
    Ok(owners)
}

fn load_uncovered(
    p: &Parse,
    top: &Map<String, Value>,
    owners: &BTreeMap<PathBuf, String>,
) -> Result<Vec<Uncovered>, ManifestError> {
    let rows = match top.get("uncovered") {
        Some(Value::Array(rows)) => rows,
        Some(other) => {
            return Err(p.error("$.uncovered", format!("expected an array, got {other}")))
        }
        None => return Err(p.error("$.uncovered", "required field is missing")),
    };
    if rows.len() > MAX_UNCOVERED {
        return Err(p.error(
            "$.uncovered",
            format!("lists {} entries; the limit is {MAX_UNCOVERED}", rows.len()),
        ));
    }
    let mut uncovered: Vec<Uncovered> = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let at = format!("$.uncovered[{index}]");
        let fields = p.object(row, &at)?;
        p.keys(fields, &at, &["class", "path", "reason"])?;
        let text = p.text(fields, "path", &at)?;
        check_destination(&text).map_err(|m| p.error(&format!("{at}.path"), m))?;
        let path = PathBuf::from(text);
        let name = p.text(fields, "class", &at)?;
        let class = UncoveredClass::parse(&name).ok_or_else(|| {
            p.error(
                &format!("{at}.class"),
                format!(
                    "expected local-conversion, digest-without-recorded-size or no-recorded-identity, got {name:?}"
                ),
            )
        })?;
        let reason = p.text(fields, "reason", &at)?;
        if let Some(owner) = owners.get(&path) {
            return Err(p.error(
                &format!("{at}.path"),
                format!(
                    "{} is declared uncovered, but {owner} records its identity",
                    path.display()
                ),
            ));
        }
        if uncovered.iter().any(|entry| entry.path == path) {
            return Err(p.error(
                &format!("{at}.path"),
                format!("{} is declared more than once", path.display()),
            ));
        }
        uncovered.push(Uncovered {
            path,
            class,
            reason,
        });
    }
    Ok(uncovered)
}

/// What the inventory supplies for a TSV, which records only digest, size and name.
struct TsvArtifact {
    key: String,
    capability: String,
    repository: String,
    revision: String,
    destination_directory: PathBuf,
    expected_rows: usize,
}

impl TsvArtifact {
    fn parse(p: &Parse, fields: &Map<String, Value>, at: &str) -> Result<Self, ManifestError> {
        let at = format!("{at}.artifact");
        let object = match fields.get("artifact") {
            Some(value) => p.object(value, &at)?,
            None => return Err(p.error(&at, "required field is missing")),
        };
        p.keys(
            object,
            &at,
            &[
                "capability",
                "destination_directory",
                "expected_rows",
                "key",
                "repository",
                "revision",
            ],
        )?;
        let repository = p.text(object, "repository", &at)?;
        check_repository(&repository).map_err(|m| p.error(&format!("{at}.repository"), m))?;
        let revision = p.text(object, "revision", &at)?;
        check_revision(&revision).map_err(|m| p.error(&format!("{at}.revision"), m))?;
        let directory = p.text(object, "destination_directory", &at)?;
        check_destination(&directory)
            .map_err(|m| p.error(&format!("{at}.destination_directory"), m))?;
        let rows = p.count(object, "expected_rows", &at)?;
        let expected_rows = usize::try_from(rows)
            .ok()
            .filter(|rows| (1..=crate::manifest::MAX_FILES).contains(rows))
            .ok_or_else(|| {
                p.error(
                    &format!("{at}.expected_rows"),
                    format!("expected 1 to {}, got {rows}", crate::manifest::MAX_FILES),
                )
            })?;
        Ok(Self {
            key: p.text(object, "key", &at)?,
            capability: p.text(object, "capability", &at)?,
            repository,
            revision,
            destination_directory: PathBuf::from(directory),
            expected_rows,
        })
    }
}

/// What the inventory supplies for a single-file manifest: the artifact's name.
struct NamedArtifact {
    key: String,
    capability: String,
}

impl NamedArtifact {
    fn parse(p: &Parse, fields: &Map<String, Value>, at: &str) -> Result<Self, ManifestError> {
        let at = format!("{at}.artifact");
        let object = match fields.get("artifact") {
            Some(value) => p.object(value, &at)?,
            None => return Err(p.error(&at, "required field is missing")),
        };
        p.keys(object, &at, &["capability", "key"])?;
        Ok(Self {
            key: p.text(object, "key", &at)?,
            capability: p.text(object, "capability", &at)?,
        })
    }
}

/// Read a `sha256<TAB>size<TAB>name` manifest.
fn load_tsv(path: &Path, spec: &TsvArtifact) -> Result<QueueManifest, ManifestError> {
    let bytes = read_bounded(path, MAX_MANIFEST_BYTES)?;
    let p = Parse::new(path);
    let text =
        std::str::from_utf8(&bytes).map_err(|e| p.error("$", format!("not UTF-8: {e}")))?;
    if text.contains('\r') {
        return Err(p.error("$", "contains a carriage return; rows end with LF only"));
    }
    let mut files = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let at = format!("line {}", index + 1);
        let fields: Vec<&str> = line.split('\t').collect();
        let [digest, size, name] = fields.as_slice() else {
            return Err(p.error(
                &at,
                format!(
                    "expected sha256, size and file name separated by tabs, got {} field(s)",
                    fields.len()
                ),
            ));
        };
        let sha256 = Sha256Hex::parse(digest).ok_or_else(|| {
            p.error(
                &at,
                format!("expected 64 lowercase hexadecimal digits, got {digest:?}"),
            )
        })?;
        let size = parse_size(size)
            .ok_or_else(|| p.error(&at, format!("expected a decimal byte count, got {size:?}")))?;
        check_file_name(name).map_err(|m| p.error(&at, m))?;
        files.push(ArtifactFile {
            repo_path: (*name).to_string(),
            destination: spec.destination_directory.join(name),
            size,
            sha256: Some(sha256),
        });
    }
    if files.len() != spec.expected_rows {
        return Err(p.error(
            "$",
            format!(
                "has {} rows; the inventory expects {}",
                files.len(),
                spec.expected_rows
            ),
        ));
    }
    assemble(
        path,
        &bytes,
        TSV_FORMAT,
        None,
        vec![Artifact {
            key: spec.key.clone(),
            capability: spec.capability.clone(),
            repository: spec.repository.clone(),
            revision: spec.revision.clone(),
            files,
        }],
    )
}

/// Read a one-file manifest: repository, revision, filename, size, digest and
/// local path. Other fields, including any recorded verification status, are
/// informational and ignored.
fn load_single_file(path: &Path, spec: &NamedArtifact) -> Result<QueueManifest, ManifestError> {
    let bytes = read_bounded(path, MAX_MANIFEST_BYTES)?;
    let p = Parse::new(path);
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|e| p.error("$", format!("not valid JSON: {e}")))?;
    let fields = p.object(&value, "$")?;
    let repository = p.text(fields, "repository", "$")?;
    check_repository(&repository).map_err(|m| p.error("$.repository", m))?;
    let revision = p.text(fields, "repository_revision", "$")?;
    check_revision(&revision).map_err(|m| p.error("$.repository_revision", m))?;
    let filename = p.text(fields, "filename", "$")?;
    check_repo_path(&filename).map_err(|m| p.error("$.filename", m))?;
    let size = p.count(fields, "size", "$")?;
    let digest = p.text(fields, "sha256", "$")?;
    let sha256 = Sha256Hex::parse(&digest).ok_or_else(|| {
        p.error(
            "$.sha256",
            format!("expected 64 lowercase hexadecimal digits, got {digest:?}"),
        )
    })?;
    let local_path = p.text(fields, "local_path", "$")?;
    check_destination(&local_path).map_err(|m| p.error("$.local_path", m))?;
    assemble(
        path,
        &bytes,
        SINGLE_FILE_FORMAT,
        None,
        vec![Artifact {
            key: spec.key.clone(),
            capability: spec.capability.clone(),
            repository,
            revision,
            files: vec![ArtifactFile {
                repo_path: filename,
                destination: PathBuf::from(local_path),
                size,
                sha256: Some(sha256),
            }],
        }],
    )
}

/// Decimal digits only, no sign and no leading zero.
fn parse_size(text: &str) -> Option<u64> {
    if text.is_empty()
        || !text.bytes().all(|b| b.is_ascii_digit())
        || (text.len() > 1 && text.starts_with('0'))
    {
        return None;
    }
    text.parse().ok()
}

/// One path component: no `/`, NUL, surrounding whitespace, `.` or `..`.
fn check_file_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\0')
        || name.trim() != name
    {
        return Err(format!(
            "expected one file name with no '/', NUL, surrounding whitespace, '.' or '..', got {name:?}"
        ));
    }
    Ok(())
}

/// A model file some preset or sidecar loads, with everything that loads it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Consumer {
    pub path: String,
    pub consumed_by: Vec<String>,
}

/// Every model file the presets and sidecar environment files under `root` load.
///
/// From `llama-models.ini`: each preset's `model` and `mmproj`. From every
/// `services/sidecar-*.env`: `SIDECAR_MODEL`, and the value after `--mmproj` in
/// `SIDECAR_EXTRA_ARGS`. Paths are compared exactly as written.
pub fn consumers(root: &Path) -> Result<Vec<Consumer>, ManifestError> {
    let mut paths: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let presets_path = root.join(PRESETS);
    let presets =
        config::load_presets(&presets_path).map_err(|error| preset_error(&presets_path, error))?;
    for (alias, preset) in &presets {
        for key in ["model", "mmproj"] {
            if let Some(value) = preset.get(key) {
                paths
                    .entry(value.to_string())
                    .or_default()
                    .push(format!("{PRESETS} [{alias}] {key}"));
            }
        }
    }

    let directory = root.join(SIDECAR_ENV_DIRECTORY);
    let mut environments = Vec::new();
    match std::fs::read_dir(&directory) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry.map_err(|error| ManifestError::Read {
                    path: directory.clone(),
                    error,
                })?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with("sidecar-") && name.ends_with(".env") {
                    environments.push((name, entry.path()));
                }
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(ManifestError::Read {
                path: directory,
                error,
            })
        }
    }
    environments.sort();
    for (name, path) in environments {
        let bytes = read_bounded(&path, MAX_MANIFEST_BYTES)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| Parse::new(&path).error("$", format!("not UTF-8: {e}")))?;
        let origin = format!("{SIDECAR_ENV_DIRECTORY}/{name}");
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let value = unquote(value.trim());
            match key.trim() {
                "SIDECAR_MODEL" => paths
                    .entry(value.to_string())
                    .or_default()
                    .push(format!("{origin} SIDECAR_MODEL")),
                "SIDECAR_EXTRA_ARGS" => {
                    let words: Vec<&str> = value.split_whitespace().collect();
                    for pair in words.windows(2) {
                        if pair[0] == "--mmproj" {
                            let path = unquote(pair[1]);
                            paths
                                .entry(path.to_string())
                                .or_default()
                                .push(format!("{origin} SIDECAR_EXTRA_ARGS --mmproj"));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(paths
        .into_iter()
        .map(|(path, consumed_by)| Consumer { path, consumed_by })
        .collect())
}

fn unquote(value: &str) -> &str {
    for quote in ['"', '\''] {
        if value.len() >= 2 && value.starts_with(quote) && value.ends_with(quote) {
            return &value[1..value.len() - 1];
        }
    }
    value
}

fn preset_error(path: &Path, error: ConfigError) -> ManifestError {
    match error {
        ConfigError::Io(_, message) => ManifestError::Read {
            path: path.to_path_buf(),
            error: match std::fs::symlink_metadata(path) {
                Err(error) => error,
                Ok(_) => io::Error::other(message),
            },
        },
        other => ManifestError::Malformed {
            path: path.to_path_buf(),
            location: "$".to_string(),
            message: other.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const DIGEST: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const OTHER: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().canonicalize().expect("temp dir");
            let dir = root.join(format!(
                "frankenctl-inventory-{tag}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&dir).expect("create scratch dir");
            Self(dir)
        }

        fn write(&self, name: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
            let path = self.0.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, bytes).unwrap();
            path
        }

        fn models(&self) -> String {
            self.0.join("models").display().to_string()
        }

        /// A queue, a TSV and a single-file manifest, and an inventory naming them.
        fn fixture(&self) -> Value {
            let models = self.models();
            self.write(
                "queues/download-queue.json",
                serde_json::to_vec(&json!({
                    "schema": QUEUE_SCHEMA,
                    "total_bytes": 3,
                    "artifacts": [{
                        "key": "queued", "capability": "test", "repository": "owner/queued",
                        "revision": "0123456789abcdef0123456789abcdef01234567",
                        "files": [{"repo_path": "q.bin", "destination": format!("{models}/q.bin"),
                                   "size": 3, "sha256": DIGEST}],
                    }],
                }))
                .unwrap(),
            );
            self.write(
                "shards/manifest.tsv",
                format!("{DIGEST}\t3\tshard-1.gguf\n{OTHER}\t0\tshard-2.gguf\n"),
            );
            self.write(
                "projector-manifest.json",
                serde_json::to_vec(&json!({
                    "repository": "owner/projector", "repository_revision": "abc123",
                    "filename": "mmproj.gguf", "size": 3, "sha256": DIGEST,
                    "status": "verified_local", "verified_at": "2026-09-01T00:00:00-0400",
                    "local_path": format!("{models}/mmproj.gguf"),
                }))
                .unwrap(),
            );
            json!({
                "schema": INVENTORY_SCHEMA,
                "integrity": "fixture",
                "sources": [
                    {"id": "queue", "format": QUEUE_SCHEMA,
                     "manifest": "queues/download-queue.json",
                     "writer_lock": "queues/download-queue.lock",
                     "writers": ["queues/download_queue.py"]},
                    {"id": "shards", "format": TSV_FORMAT, "manifest": "shards/manifest.tsv",
                     "writer_lock": "shards/manifest.lock", "writers": ["shards/writer.py"],
                     "artifact": {"key": "shards", "capability": "test",
                                  "repository": "owner/shards", "revision": "main",
                                  "destination_directory": format!("{models}/shards"),
                                  "expected_rows": 2}},
                    {"id": "projector", "format": SINGLE_FILE_FORMAT,
                     "manifest": "projector-manifest.json",
                     "writer_lock": "projector-manifest.lock", "writers": ["writer.py"],
                     "artifact": {"key": "projector", "capability": "vision"}},
                ],
                "uncovered": [
                    {"path": format!("{models}/converted.gguf"), "class": "local-conversion",
                     "reason": "converted here"},
                ],
            })
        }

        fn load(&self, inventory: &Value) -> Result<Inventory, ManifestError> {
            let path = self.write("inventory.json", serde_json::to_vec(inventory).unwrap());
            Inventory::load(&self.0, &path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn malformed(result: Result<Inventory, ManifestError>) -> (String, String) {
        match result {
            Err(ManifestError::Malformed {
                location, message, ..
            }) => (location, message),
            other => panic!("expected a malformed inventory, got {other:?}"),
        }
    }

    #[test]
    fn all_three_formats_load_into_one_shape_with_their_locks() {
        let scratch = Scratch::new("formats");
        let inventory = scratch.load(&scratch.fixture()).unwrap();
        let models = scratch.models();
        let ids: Vec<(&str, &str)> = inventory
            .sources
            .iter()
            .map(|s| (s.id.as_str(), s.manifest.format))
            .collect();
        assert_eq!(
            ids,
            [
                ("queue", QUEUE_SCHEMA),
                ("shards", TSV_FORMAT),
                ("projector", SINGLE_FILE_FORMAT)
            ]
        );
        assert_eq!(
            inventory.sources[1].writer_lock,
            scratch.0.join("shards/manifest.lock")
        );
        let shards = &inventory.sources[1].manifest.artifacts[0];
        assert_eq!((shards.repository.as_str(), shards.revision.as_str()), ("owner/shards", "main"));
        assert_eq!(shards.files.len(), 2);
        assert_eq!(
            shards.files[1].destination,
            PathBuf::from(format!("{models}/shards/shard-2.gguf"))
        );
        assert_eq!(shards.files[1].size, 0);
        assert_eq!(inventory.sources[1].manifest.total_bytes, 3);
        let projector = &inventory.sources[2].manifest.artifacts[0];
        assert_eq!(projector.revision, "abc123");
        assert_eq!(projector.files[0].repo_path, "mmproj.gguf");
        assert_eq!(
            inventory.uncovered,
            [Uncovered {
                path: PathBuf::from(format!("{models}/converted.gguf")),
                class: UncoveredClass::LocalConversion,
                reason: "converted here".to_string(),
            }]
        );
        assert!(matches!(inventory.origin, Origin::File { .. }));
    }

    #[test]
    fn a_writer_lock_that_is_not_the_manifests_pair_is_rejected() {
        let scratch = Scratch::new("lock");
        let mut value = scratch.fixture();
        value["sources"][1]["writer_lock"] = json!("shards/other.lock");
        let (location, message) = malformed(scratch.load(&value));
        assert_eq!(location, "$.sources[1].writer_lock");
        assert!(message.contains("manifest.lock"), "{message}");
    }

    #[test]
    fn structural_problems_reject_the_whole_inventory() {
        let scratch = Scratch::new("structure");
        let cases: Vec<(&str, Box<dyn Fn(&mut Value)>)> = vec![
            ("$.schema", Box::new(|v| v["schema"] = json!("frankenctl-artifact-inventory/2"))),
            ("$", Box::new(|v| v["extra"] = json!(1))),
            ("$.sources[0].format", Box::new(|v| v["sources"][0]["format"] = json!("research-inventory/1"))),
            ("$.sources[0].id", Box::new(|v| v["sources"][0]["id"] = json!("Queue"))),
            ("$.sources[1].id", Box::new(|v| v["sources"][1]["id"] = json!("queue"))),
            ("$.sources[0].manifest", Box::new(|v| v["sources"][0]["manifest"] = json!("../outside.json"))),
            ("$.sources[0].writers", Box::new(|v| v["sources"][0]["writers"] = json!([]))),
            ("$.sources[1].artifact", Box::new(|v| {
                v["sources"][1]["artifact"].as_object_mut().unwrap().remove("expected_rows");
            })),
            ("$.sources[1].artifact.destination_directory", Box::new(|v| {
                v["sources"][1]["artifact"]["destination_directory"] = json!("models/shards")
            })),
            ("$.sources[1].artifact.repository", Box::new(|v| {
                v["sources"][1]["artifact"]["repository"] = json!("no-owner")
            })),
            ("$.uncovered[0].class", Box::new(|v| v["uncovered"][0]["class"] = json!("probably-fine"))),
            ("$.uncovered[0].path", Box::new(|v| v["uncovered"][0]["path"] = json!("models/relative.gguf"))),
            ("$.uncovered", Box::new(|v| {
                v.as_object_mut().unwrap().remove("uncovered");
            })),
        ];
        for (expected, mutate) in cases {
            let mut value = scratch.fixture();
            mutate(&mut value);
            let (location, message) = malformed(scratch.load(&value));
            assert!(location == expected || location == "$", "{expected}: {location} {message}");
            if expected != "$" && expected != "$.uncovered" {
                assert_eq!(location, expected, "{message}");
            }
        }
    }

    #[test]
    fn a_path_declared_uncovered_that_a_source_records_is_rejected() {
        let scratch = Scratch::new("overlap");
        let mut value = scratch.fixture();
        value["uncovered"][0]["path"] = json!(format!("{}/q.bin", scratch.models()));
        let (location, message) = malformed(scratch.load(&value));
        assert_eq!(location, "$.uncovered[0].path");
        assert!(message.contains("queue/queued/q.bin"), "{message}");
    }

    #[test]
    fn destinations_and_keys_are_unique_across_sources() {
        let scratch = Scratch::new("cross");
        let value = scratch.fixture();
        // The projector now writes where the queue does.
        scratch.write(
            "projector-manifest.json",
            serde_json::to_vec(&json!({
                "repository": "owner/projector", "repository_revision": "abc123",
                "filename": "mmproj.gguf", "size": 3, "sha256": DIGEST,
                "local_path": format!("{}/q.bin", scratch.models()),
            }))
            .unwrap(),
        );
        let (location, message) = malformed(scratch.load(&value));
        assert_eq!(location, "$.sources[2]");
        assert!(message.contains("claimed by both"), "{message}");

        let mut value = scratch.fixture();
        value["sources"][2]["artifact"]["key"] = json!("queued");
        let (_, message) = malformed(scratch.load(&value));
        assert!(message.contains("declared by both"), "{message}");
    }

    #[test]
    fn malformed_tsv_rows_are_rejected_with_their_line() {
        let scratch = Scratch::new("tsv");
        let value = scratch.fixture();
        for (rows, needle) in [
            (format!("{DIGEST}\t3\tshard-1.gguf\n"), "has 1 rows"),
            (format!("{DIGEST}\t3\tshard-1.gguf\n\n{OTHER}\t0\tshard-2.gguf\n"), "line 2"),
            (format!("{DIGEST}\t3\tshard-1.gguf\n{}\t0\tshard-2.gguf\n", DIGEST.to_uppercase()), "lowercase"),
            (format!("{DIGEST}\t-3\tshard-1.gguf\n{OTHER}\t0\tshard-2.gguf\n"), "byte count"),
            (format!("{DIGEST}\t03\tshard-1.gguf\n{OTHER}\t0\tshard-2.gguf\n"), "byte count"),
            (format!("{DIGEST}\t3\t../escape.gguf\n{OTHER}\t0\tshard-2.gguf\n"), "file name"),
            (format!("{DIGEST}\t3\tshard-1.gguf\r\n{OTHER}\t0\tshard-2.gguf\r\n"), "carriage return"),
            (format!("{DIGEST} 3 shard-1.gguf\n{OTHER}\t0\tshard-2.gguf\n"), "separated by tabs"),
            (format!("{DIGEST}\t3\tshard-1.gguf\n{OTHER}\t0\tshard-1.gguf\n"), "claimed by both"),
        ] {
            scratch.write("shards/manifest.tsv", &rows);
            let (location, message) = malformed(scratch.load(&value));
            assert!(
                message.contains(needle) || location.contains(needle),
                "{needle}: location={location} message={message}"
            );
        }
    }

    #[test]
    fn a_single_file_manifest_must_carry_its_identity() {
        let scratch = Scratch::new("single");
        let value = scratch.fixture();
        for field in ["repository_revision", "size", "sha256", "local_path", "filename"] {
            let mut manifest = json!({
                "repository": "owner/projector", "repository_revision": "abc123",
                "filename": "mmproj.gguf", "size": 3, "sha256": DIGEST,
                "local_path": format!("{}/mmproj.gguf", scratch.models()),
            });
            manifest.as_object_mut().unwrap().remove(field);
            scratch.write("projector-manifest.json", serde_json::to_vec(&manifest).unwrap());
            let (location, _) = malformed(scratch.load(&value));
            assert_eq!(location, format!("$.{field}"));
        }
    }

    #[test]
    fn an_absent_source_manifest_is_absent_input() {
        let scratch = Scratch::new("absent");
        let value = scratch.fixture();
        std::fs::remove_file(scratch.0.join("shards/manifest.tsv")).unwrap();
        match scratch.load(&value) {
            Err(ManifestError::Read { path, error }) => {
                assert_eq!(path, scratch.0.join("shards/manifest.tsv"));
                assert_eq!(error.kind(), io::ErrorKind::NotFound);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn consumers_come_from_presets_and_sidecar_environment_files() {
        let scratch = Scratch::new("consumers");
        scratch.write(
            PRESETS,
            "[*]\nctx-size = 4096\n\n[chat]\nmodel = /m/chat.gguf\n\n[vision]\nmodel = /m/chat.gguf\nmmproj = /m/mmproj.gguf\n",
        );
        scratch.write(
            "services/sidecar-ocr.env",
            "# comment\nSIDECAR_MODEL=/m/ocr.gguf\nSIDECAR_EXTRA_ARGS=--mmproj \"/m/ocr-mmproj.gguf\" --temp 0\n",
        );
        scratch.write("services/sidecar-fim.env", "SIDECAR_MODEL='/m/chat.gguf'\n");
        scratch.write("services/other.env", "SIDECAR_MODEL=/m/ignored.gguf\n");
        let found = consumers(&scratch.0).unwrap();
        let paths: Vec<(String, usize)> = found
            .iter()
            .map(|c| (c.path.to_string(), c.consumed_by.len()))
            .collect();
        assert_eq!(
            paths,
            [
                ("/m/chat.gguf".to_string(), 3),
                ("/m/mmproj.gguf".to_string(), 1),
                ("/m/ocr-mmproj.gguf".to_string(), 1),
                ("/m/ocr.gguf".to_string(), 1)
            ]
        );
        assert_eq!(
            found[0].consumed_by,
            [
                "llama-models.ini [chat] model",
                "llama-models.ini [vision] model",
                "services/sidecar-fim.env SIDECAR_MODEL"
            ]
        );
    }

    #[test]
    fn missing_presets_are_absent_input() {
        let scratch = Scratch::new("no-presets");
        match consumers(&scratch.0) {
            Err(ManifestError::Read { error, .. }) => {
                assert_eq!(error.kind(), io::ErrorKind::NotFound)
            }
            other => panic!("{other:?}"),
        }
    }
}
