//! Typed ingestion of `hermes-hf-artifact-queue/1` download-queue manifests.
//!
//! The Python downloader reads a queue file as loose dictionaries. The verifier
//! cannot: a digest read here is what a local file is judged against, and a
//! destination read here is a path that gets opened. So the whole manifest is
//! validated before any artifact path is touched, and one bad entry rejects the
//! manifest rather than leaving a quietly smaller set to verify.
//!
//! The accepted shape is the one the tracked queues use. The top level carries
//! `schema`, `total_bytes` and `artifacts[]`. Each artifact carries `key`,
//! `capability`, `repository`, `revision` and `files[]`, plus an optional
//! `total_bytes` that must agree with its files; other artifact fields are
//! informational and ignored. Each file carries exactly `repo_path`,
//! `destination`, `size` and `sha256`, where `sha256` is `null` when the queue
//! records no publisher digest for that file.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// The only queue schema this module accepts.
pub const QUEUE_SCHEMA: &str = "hermes-hf-artifact-queue/1";
/// Manifests are read with this bound. The tracked queues are under 20 KiB.
pub const MAX_MANIFEST_BYTES: u64 = 8 * 1024 * 1024;
/// Most artifacts one manifest may declare.
pub const MAX_ARTIFACTS: usize = 1_000;
/// Most files one manifest may declare, across all of its artifacts.
pub const MAX_FILES: usize = 10_000;

/// The exact key set of one `files[]` entry, sorted.
const FILE_KEYS: [&str; 4] = ["destination", "repo_path", "sha256", "size"];

/// A SHA-256 digest as 64 lowercase hexadecimal digits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sha256Hex(String);

impl Sha256Hex {
    /// Accept exactly 64 lowercase hexadecimal digits. Uppercase is rejected
    /// rather than folded, so a digest is compared in the one form it was
    /// published in.
    pub fn parse(text: &str) -> Option<Self> {
        let valid =
            text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
        valid.then(|| Self(text.to_string()))
    }

    /// The digest of `bytes`.
    pub fn of(bytes: &[u8]) -> Self {
        Self(to_hex(&Sha256::digest(bytes)))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Lowercase hexadecimal encoding.
pub fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 0x0f)] as char);
    }
    out
}

/// One validated download-queue manifest.
#[derive(Debug, Clone)]
pub struct QueueManifest {
    /// The path the manifest was read from, as given.
    pub path: PathBuf,
    /// Size of the manifest in bytes.
    pub bytes: u64,
    /// SHA-256 of the exact manifest bytes that were parsed.
    pub sha256: Sha256Hex,
    pub built_at: Option<String>,
    /// The declared queue total, already checked against the files.
    pub total_bytes: u64,
    pub artifacts: Vec<Artifact>,
}

/// A repository at a pinned revision and the files taken from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    pub key: String,
    pub capability: String,
    pub repository: String,
    pub revision: String,
    pub files: Vec<ArtifactFile>,
}

/// One file of an artifact and where it lives locally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactFile {
    /// Path inside the repository at the artifact's revision.
    pub repo_path: String,
    /// Absolute local destination, in lexical normal form.
    pub destination: PathBuf,
    /// Exact size in bytes.
    pub size: u64,
    /// Publisher SHA-256, or `None` when the queue records none.
    pub sha256: Option<Sha256Hex>,
}

/// Why a manifest was not accepted.
#[derive(Debug)]
pub enum ManifestError {
    /// The manifest could not be read. The OS error is kept as it was raised.
    Read {
        path: PathBuf,
        error: std::io::Error,
    },
    /// The manifest is larger than [`MAX_MANIFEST_BYTES`] and was not parsed.
    TooLarge { path: PathBuf, limit: u64 },
    /// The manifest was read but does not satisfy the queue schema.
    Malformed {
        path: PathBuf,
        location: String,
        message: String,
    },
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, error } => {
                write!(f, "cannot read manifest {}: {error}", path.display())
            }
            Self::TooLarge { path, limit } => write!(
                f,
                "manifest {} is larger than {limit} bytes and was not parsed",
                path.display()
            ),
            Self::Malformed {
                path,
                location,
                message,
            } => write!(
                f,
                "malformed manifest {} at {location}: {message}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ManifestError {}

/// An `--artifact` selection that does not name the manifest's artifacts exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    UnknownArtifact(String),
    SelectedTwice(String),
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownArtifact(key) => write!(f, "no artifact with key {key:?} in the manifest"),
            Self::SelectedTwice(key) => write!(f, "artifact {key:?} was selected more than once"),
        }
    }
}

impl std::error::Error for SelectionError {}

impl QueueManifest {
    /// Read and validate a manifest, reading at most [`MAX_MANIFEST_BYTES`] + 1.
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        let read = |error| ManifestError::Read {
            path: path.to_path_buf(),
            error,
        };
        let file = std::fs::File::open(path).map_err(read)?;
        let mut bytes = Vec::new();
        file.take(MAX_MANIFEST_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(read)?;
        if bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(ManifestError::TooLarge {
                path: path.to_path_buf(),
                limit: MAX_MANIFEST_BYTES,
            });
        }
        Self::parse(path, &bytes)
    }

    /// Validate manifest bytes. `path` is used for identity and error messages only.
    pub fn parse(path: &Path, bytes: &[u8]) -> Result<Self, ManifestError> {
        let p = Parse { path };
        let root: Value = serde_json::from_slice(bytes)
            .map_err(|e| p.error("$", format!("not valid JSON: {e}")))?;
        let top = p.object(&root, "$")?;
        let schema = p.text(top, "schema", "$")?;
        if schema != QUEUE_SCHEMA {
            return Err(p.error(
                "$.schema",
                format!("expected {QUEUE_SCHEMA:?}, got {schema:?}"),
            ));
        }
        let built_at = match top.get("built_at") {
            None => None,
            Some(Value::String(text)) => Some(text.clone()),
            Some(_) => return Err(p.error("$.built_at", "expected a string")),
        };
        let total_bytes = p.count(top, "total_bytes", "$")?;
        let entries = p.array(top, "artifacts", "$", MAX_ARTIFACTS)?;

        let mut keys = BTreeSet::new();
        let mut owners: BTreeMap<PathBuf, String> = BTreeMap::new();
        let mut file_count = 0usize;
        let mut queue_sum = 0u64;
        let mut artifacts = Vec::with_capacity(entries.len());
        for (index, entry) in entries.iter().enumerate() {
            let at = format!("$.artifacts[{index}]");
            let object = p.object(entry, &at)?;
            let key = p.text(object, "key", &at)?;
            if !keys.insert(key.clone()) {
                return Err(p.error(
                    &format!("{at}.key"),
                    format!("artifact key {key:?} appears more than once"),
                ));
            }
            let capability = p.text(object, "capability", &at)?;
            let repository = p.text(object, "repository", &at)?;
            let parts: Vec<&str> = repository.split('/').collect();
            if parts.len() != 2
                || parts.iter().any(|part| part.is_empty())
                || repository.chars().any(char::is_whitespace)
            {
                return Err(p.error(
                    &format!("{at}.repository"),
                    format!("expected owner/name, got {repository:?}"),
                ));
            }
            let revision = p.text(object, "revision", &at)?;
            if revision.chars().any(char::is_whitespace) {
                return Err(p.error(
                    &format!("{at}.revision"),
                    format!("must not contain whitespace, got {revision:?}"),
                ));
            }
            let rows = p.array(object, "files", &at, MAX_FILES)?;
            let mut files = Vec::with_capacity(rows.len());
            let mut artifact_sum = 0u64;
            for (row_index, row) in rows.iter().enumerate() {
                let at = format!("{at}.files[{row_index}]");
                file_count += 1;
                if file_count > MAX_FILES {
                    return Err(p.error(&at, format!("more than {MAX_FILES} files")));
                }
                let fields = p.object(row, &at)?;
                let mut present: Vec<&str> = fields.keys().map(String::as_str).collect();
                present.sort_unstable();
                if present != FILE_KEYS {
                    return Err(p.error(
                        &at,
                        format!("expected exactly the keys {FILE_KEYS:?}, got {present:?}"),
                    ));
                }
                let repo_path = p.text(fields, "repo_path", &at)?;
                check_repo_path(&repo_path).map_err(|m| p.error(&format!("{at}.repo_path"), m))?;
                let destination = p.text(fields, "destination", &at)?;
                check_destination(&destination)
                    .map_err(|m| p.error(&format!("{at}.destination"), m))?;
                let destination = PathBuf::from(destination);
                let size = p.count(fields, "size", &at)?;
                let sha256 = match &fields["sha256"] {
                    Value::Null => None,
                    Value::String(text) => Some(Sha256Hex::parse(text).ok_or_else(|| {
                        p.error(
                            &format!("{at}.sha256"),
                            format!(
                                "expected 64 lowercase hexadecimal digits or null, got {text:?}"
                            ),
                        )
                    })?),
                    other => {
                        return Err(p.error(
                            &format!("{at}.sha256"),
                            format!(
                                "expected 64 lowercase hexadecimal digits or null, got {other}"
                            ),
                        ))
                    }
                };
                let owner = format!("{key}/{repo_path}");
                if let Some(previous) = owners.insert(destination.clone(), owner.clone()) {
                    return Err(p.error(
                        &format!("{at}.destination"),
                        format!(
                            "{} is claimed by both {previous} and {owner}",
                            destination.display()
                        ),
                    ));
                }
                artifact_sum = artifact_sum
                    .checked_add(size)
                    .ok_or_else(|| p.error(&at, "the artifact byte total overflows"))?;
                queue_sum = queue_sum
                    .checked_add(size)
                    .ok_or_else(|| p.error(&at, "the queue byte total overflows"))?;
                files.push(ArtifactFile {
                    repo_path,
                    destination,
                    size,
                    sha256,
                });
            }
            if object.contains_key("total_bytes") {
                let declared = p.count(object, "total_bytes", &at)?;
                if declared != artifact_sum {
                    return Err(p.error(
                        &format!("{at}.total_bytes"),
                        format!("declares {declared} bytes but its files sum to {artifact_sum}"),
                    ));
                }
            }
            artifacts.push(Artifact {
                key,
                capability,
                repository,
                revision,
                files,
            });
        }
        if total_bytes != queue_sum {
            return Err(p.error(
                "$.total_bytes",
                format!("declares {total_bytes} bytes but the files sum to {queue_sum}"),
            ));
        }
        Ok(Self {
            path: path.to_path_buf(),
            bytes: bytes.len() as u64,
            sha256: Sha256Hex::of(bytes),
            built_at,
            total_bytes,
            artifacts,
        })
    }

    /// The artifacts named by `keys`, in manifest order; every artifact when
    /// `keys` is empty. A key that matches nothing, or is given twice, is an
    /// error rather than a silently different selection.
    pub fn select(&self, keys: &[String]) -> Result<Vec<&Artifact>, SelectionError> {
        if keys.is_empty() {
            return Ok(self.artifacts.iter().collect());
        }
        let mut wanted = BTreeSet::new();
        for key in keys {
            if !self.artifacts.iter().any(|artifact| &artifact.key == key) {
                return Err(SelectionError::UnknownArtifact(key.clone()));
            }
            if !wanted.insert(key.as_str()) {
                return Err(SelectionError::SelectedTwice(key.clone()));
            }
        }
        Ok(self
            .artifacts
            .iter()
            .filter(|artifact| wanted.contains(artifact.key.as_str()))
            .collect())
    }

    /// Number of files across every artifact.
    pub fn file_count(&self) -> usize {
        self.artifacts
            .iter()
            .map(|artifact| artifact.files.len())
            .sum()
    }
}

/// Field extraction that reports the JSON location of the first problem.
struct Parse<'a> {
    path: &'a Path,
}

impl Parse<'_> {
    fn error(&self, location: &str, message: impl Into<String>) -> ManifestError {
        ManifestError::Malformed {
            path: self.path.to_path_buf(),
            location: location.to_string(),
            message: message.into(),
        }
    }

    fn object<'v>(
        &self,
        value: &'v Value,
        location: &str,
    ) -> Result<&'v Map<String, Value>, ManifestError> {
        value
            .as_object()
            .ok_or_else(|| self.error(location, "expected a JSON object"))
    }

    fn text(
        &self,
        object: &Map<String, Value>,
        key: &str,
        location: &str,
    ) -> Result<String, ManifestError> {
        let at = format!("{location}.{key}");
        match object.get(key) {
            None => Err(self.error(&at, "required field is missing")),
            Some(Value::String(text)) if text.trim().is_empty() => {
                Err(self.error(&at, "must not be empty"))
            }
            Some(Value::String(text)) if text.trim() != text.as_str() => {
                Err(self.error(&at, "must not have leading or trailing whitespace"))
            }
            Some(Value::String(text)) => Ok(text.clone()),
            Some(other) => Err(self.error(&at, format!("expected a string, got {other}"))),
        }
    }

    fn count(
        &self,
        object: &Map<String, Value>,
        key: &str,
        location: &str,
    ) -> Result<u64, ManifestError> {
        let at = format!("{location}.{key}");
        let value = object
            .get(key)
            .ok_or_else(|| self.error(&at, "required field is missing"))?;
        value
            .as_u64()
            .ok_or_else(|| self.error(&at, format!("expected a non-negative integer, got {value}")))
    }

    fn array<'v>(
        &self,
        object: &'v Map<String, Value>,
        key: &str,
        location: &str,
        limit: usize,
    ) -> Result<&'v Vec<Value>, ManifestError> {
        let at = format!("{location}.{key}");
        match object.get(key) {
            None => Err(self.error(&at, "required field is missing")),
            Some(Value::Array(items)) if items.is_empty() => {
                Err(self.error(&at, "must list at least one entry"))
            }
            Some(Value::Array(items)) if items.len() > limit => Err(self.error(
                &at,
                format!("lists {} entries; the limit is {limit}", items.len()),
            )),
            Some(Value::Array(items)) => Ok(items),
            Some(other) => Err(self.error(&at, format!("expected an array, got {other}"))),
        }
    }
}

/// A repository path is relative, with no empty, `.` or `..` component.
fn check_repo_path(text: &str) -> Result<(), String> {
    if text.contains('\0') {
        return Err("contains a NUL byte".to_string());
    }
    if text.starts_with('/') {
        return Err(format!("{text:?} must be relative to the repository"));
    }
    if text
        .split('/')
        .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(format!("{text:?} has an empty, '.' or '..' component"));
    }
    Ok(())
}

/// A destination is absolute and already in lexical normal form: no empty,
/// `.` or `..` component and no trailing `/`. Normal form is required, not
/// computed, so two spellings of one path cannot pass as two destinations.
fn check_destination(text: &str) -> Result<(), String> {
    if text.contains('\0') {
        return Err("contains a NUL byte".to_string());
    }
    let Some(rest) = text.strip_prefix('/') else {
        return Err(format!("{text:?} is not an absolute path"));
    };
    if rest
        .split('/')
        .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(format!(
            "{text:?} is not in normal form (an empty, '.' or '..' component, or a trailing '/')"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const DIGEST: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn file(repo_path: &str, destination: &str, size: u64, sha256: Value) -> Value {
        json!({"repo_path": repo_path, "destination": destination, "size": size, "sha256": sha256})
    }

    fn queue(files: Vec<Value>) -> Value {
        let total: u64 = files.iter().map(|f| f["size"].as_u64().unwrap()).sum();
        json!({
            "schema": QUEUE_SCHEMA,
            "built_at": "2026-09-01T09:39:54-0400",
            "source_of_truth": "informational",
            "policy": {"single_writer": true},
            "total_bytes": total,
            "artifacts": [{
                "key": "fixture",
                "capability": "test",
                "repository": "owner/repo",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "license": "apache-2.0",
                "gated": false,
                "total_bytes": total,
                "files": files,
            }],
        })
    }

    fn parse(value: &Value) -> Result<QueueManifest, ManifestError> {
        QueueManifest::parse(
            Path::new("fixture.json"),
            &serde_json::to_vec(value).unwrap(),
        )
    }

    fn malformed_at(value: &Value) -> (String, String) {
        match parse(value) {
            Err(ManifestError::Malformed {
                location, message, ..
            }) => (location, message),
            other => panic!("expected a malformed manifest, got {other:?}"),
        }
    }

    #[test]
    fn the_tracked_queue_shape_is_ingested_with_exact_identity() {
        let value = queue(vec![
            file("model.gguf", "/models/fixture/model.gguf", 3, json!(DIGEST)),
            file("config.json", "/models/fixture/config.json", 5, Value::Null),
        ]);
        let bytes = serde_json::to_vec(&value).unwrap();
        let manifest = QueueManifest::parse(Path::new("q.json"), &bytes).unwrap();
        assert_eq!(manifest.total_bytes, 8);
        assert_eq!(manifest.bytes, bytes.len() as u64);
        assert_eq!(manifest.sha256, Sha256Hex::of(&bytes));
        assert_eq!(
            manifest.built_at.as_deref(),
            Some("2026-09-01T09:39:54-0400")
        );
        let artifact = &manifest.artifacts[0];
        assert_eq!(artifact.key, "fixture");
        assert_eq!(artifact.repository, "owner/repo");
        assert_eq!(
            artifact.revision,
            "0123456789abcdef0123456789abcdef01234567"
        );
        assert_eq!(
            artifact.files,
            vec![
                ArtifactFile {
                    repo_path: "model.gguf".into(),
                    destination: "/models/fixture/model.gguf".into(),
                    size: 3,
                    sha256: Sha256Hex::parse(DIGEST),
                },
                ArtifactFile {
                    repo_path: "config.json".into(),
                    destination: "/models/fixture/config.json".into(),
                    size: 5,
                    sha256: None,
                },
            ]
        );
    }

    #[test]
    fn invalid_digests_are_rejected_not_treated_as_missing() {
        for bad in [
            json!(DIGEST.to_uppercase()),
            json!(&DIGEST[..63]),
            json!(format!("{DIGEST}0")),
            json!(format!("{}g", &DIGEST[..63])),
            json!(""),
            json!(42),
            json!(true),
        ] {
            let (location, message) =
                malformed_at(&queue(vec![file("m", "/models/m", 1, bad.clone())]));
            assert_eq!(location, "$.artifacts[0].files[0].sha256", "{bad}");
            assert!(message.contains("64 lowercase hexadecimal"), "{message}");
        }
    }

    #[test]
    fn a_file_entry_must_carry_exactly_the_four_queue_keys() {
        let mut missing = queue(vec![file("m", "/models/m", 1, json!(DIGEST))]);
        missing["artifacts"][0]["files"][0]
            .as_object_mut()
            .unwrap()
            .remove("sha256");
        assert!(malformed_at(&missing)
            .1
            .contains("expected exactly the keys"));

        let mut extra = queue(vec![file("m", "/models/m", 1, json!(DIGEST))]);
        extra["artifacts"][0]["files"][0]["md5"] = json!("abc");
        assert!(malformed_at(&extra).1.contains("expected exactly the keys"));
    }

    #[test]
    fn sizes_must_be_exact_non_negative_integers() {
        for bad in [json!(-1), json!(1.5), json!(3.0), json!("3"), Value::Null] {
            let mut value = queue(vec![file("m", "/models/m", 3, json!(DIGEST))]);
            value["artifacts"][0]["files"][0]["size"] = bad.clone();
            let (location, _) = malformed_at(&value);
            assert_eq!(location, "$.artifacts[0].files[0].size", "{bad}");
        }
    }

    #[test]
    fn unsafe_destinations_are_rejected() {
        for bad in [
            "models/m",
            "/models/../etc/passwd",
            "/models/./m",
            "/models//m",
            "/models/m/",
            "/",
            "",
        ] {
            let (location, _) = malformed_at(&queue(vec![file("m", bad, 1, json!(DIGEST))]));
            assert!(
                location == "$.artifacts[0].files[0].destination",
                "{bad:?} -> {location}"
            );
        }
    }

    #[test]
    fn unsafe_repository_paths_are_rejected() {
        for bad in ["/abs", "a/../b", "./a", "a//b", "a/"] {
            let (location, _) =
                malformed_at(&queue(vec![file(bad, "/models/m", 1, json!(DIGEST))]));
            assert_eq!(location, "$.artifacts[0].files[0].repo_path", "{bad:?}");
        }
    }

    #[test]
    fn duplicate_destinations_name_both_owners() {
        let (location, message) = malformed_at(&queue(vec![
            file("a.gguf", "/models/same.gguf", 1, json!(DIGEST)),
            file("b.gguf", "/models/same.gguf", 1, Value::Null),
        ]));
        assert_eq!(location, "$.artifacts[0].files[1].destination");
        assert!(
            message.contains("fixture/a.gguf") && message.contains("fixture/b.gguf"),
            "{message}"
        );
    }

    #[test]
    fn duplicate_destinations_across_artifacts_are_rejected() {
        let mut value = queue(vec![file("a", "/models/same", 1, json!(DIGEST))]);
        let mut second = value["artifacts"][0].clone();
        second["key"] = json!("second");
        value["artifacts"].as_array_mut().unwrap().push(second);
        value["total_bytes"] = json!(2);
        let (location, message) = malformed_at(&value);
        assert_eq!(location, "$.artifacts[1].files[0].destination");
        assert!(message.contains("claimed by both"), "{message}");
    }

    #[test]
    fn duplicate_artifact_keys_are_rejected() {
        let mut value = queue(vec![file("a", "/models/a", 1, json!(DIGEST))]);
        let mut second = value["artifacts"][0].clone();
        second["files"][0]["destination"] = json!("/models/b");
        value["artifacts"].as_array_mut().unwrap().push(second);
        value["total_bytes"] = json!(2);
        assert_eq!(malformed_at(&value).0, "$.artifacts[1].key");
    }

    #[test]
    fn byte_totals_must_match_their_files() {
        let mut queue_total = queue(vec![file("a", "/models/a", 4, json!(DIGEST))]);
        queue_total["total_bytes"] = json!(5);
        assert_eq!(malformed_at(&queue_total).0, "$.total_bytes");

        let mut artifact_total = queue(vec![file("a", "/models/a", 4, json!(DIGEST))]);
        artifact_total["artifacts"][0]["total_bytes"] = json!(3);
        assert_eq!(
            malformed_at(&artifact_total).0,
            "$.artifacts[0].total_bytes"
        );
    }

    #[test]
    fn structural_problems_fail_the_whole_manifest() {
        let mut schema = queue(vec![file("a", "/models/a", 1, json!(DIGEST))]);
        schema["schema"] = json!("hermes-hf-artifact-queue/2");
        assert_eq!(malformed_at(&schema).0, "$.schema");

        let mut no_artifacts = queue(vec![file("a", "/models/a", 1, json!(DIGEST))]);
        no_artifacts["artifacts"] = json!([]);
        no_artifacts["total_bytes"] = json!(0);
        assert_eq!(malformed_at(&no_artifacts).0, "$.artifacts");

        let mut no_files = queue(vec![file("a", "/models/a", 1, json!(DIGEST))]);
        no_files["artifacts"][0]["files"] = json!([]);
        assert_eq!(malformed_at(&no_files).0, "$.artifacts[0].files");

        let mut bad_repository = queue(vec![file("a", "/models/a", 1, json!(DIGEST))]);
        bad_repository["artifacts"][0]["repository"] = json!("no-owner");
        assert_eq!(malformed_at(&bad_repository).0, "$.artifacts[0].repository");

        assert_eq!(malformed_at(&json!([])).0, "$");
        match QueueManifest::parse(Path::new("x"), b"{\"schema\": ") {
            Err(ManifestError::Malformed { location, .. }) => assert_eq!(location, "$"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn selection_is_exact_and_keeps_manifest_order() {
        let mut value = queue(vec![file("a", "/models/a", 1, json!(DIGEST))]);
        for (key, destination) in [("second", "/models/b"), ("third", "/models/c")] {
            let mut next = value["artifacts"][0].clone();
            next["key"] = json!(key);
            next["files"][0]["destination"] = json!(destination);
            value["artifacts"].as_array_mut().unwrap().push(next);
        }
        value["total_bytes"] = json!(3);
        let manifest = parse(&value).unwrap();
        assert_eq!(manifest.select(&[]).unwrap().len(), 3);
        let picked: Vec<&str> = manifest
            .select(&["third".into(), "fixture".into()])
            .unwrap()
            .iter()
            .map(|a| a.key.as_str())
            .collect();
        assert_eq!(picked, ["fixture", "third"]);
        assert_eq!(
            manifest.select(&["absent".into()]).unwrap_err(),
            SelectionError::UnknownArtifact("absent".into())
        );
        assert_eq!(
            manifest
                .select(&["third".into(), "third".into()])
                .unwrap_err(),
            SelectionError::SelectedTwice("third".into())
        );
    }

    #[test]
    fn oversized_and_absent_manifests_are_refused_before_parsing() {
        let dir = std::env::temp_dir().join(format!("frankenctl-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let big = dir.join("big.json");
        std::fs::write(&big, vec![b' '; MAX_MANIFEST_BYTES as usize + 1]).unwrap();
        assert!(matches!(
            QueueManifest::load(&big),
            Err(ManifestError::TooLarge { .. })
        ));
        match QueueManifest::load(&dir.join("absent.json")) {
            Err(ManifestError::Read { error, .. }) => {
                assert_eq!(error.kind(), std::io::ErrorKind::NotFound)
            }
            other => panic!("{other:?}"),
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
