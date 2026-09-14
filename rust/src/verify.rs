//! Read-only verification of local artifacts against a download-queue manifest.
//!
//! This is the verification core only. It checks the files a validated
//! [`QueueManifest`] names: exact size always, and SHA-256 when the queue
//! records a publisher digest. It never downloads, repairs, deletes, renames or
//! promotes a file, never admits a model or grants qualification, and never
//! reads a completion stamp: every run judges the bytes on disk at that moment.
//!
//! It also does not take the download queues' locks yet. Coordinating with a
//! running downloader is the service integration's job, so the caller must make
//! sure nothing writes these destinations during a run. A file that changes
//! while it is read is still caught, and reported as such, rather than scored.
//!
//! Content is hashed through one fixed buffer of [`HASH_BUFFER_BYTES`], so
//! memory does not grow with file size. An I/O error keeps its operation, path,
//! OS error number and offset, and makes the run exit 74; it is never folded
//! into a mismatch or a missing file.

use std::fs::{File, Metadata};
use std::io::{self, Read};
use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::manifest::{to_hex, Artifact, ArtifactFile, ManifestError, QueueManifest, QUEUE_SCHEMA};

/// Size of the single read buffer used to hash a file.
pub const HASH_BUFFER_BYTES: usize = 8 * 1024 * 1024;

/// Every selected file matched: SHA-256 and size, or size where no digest is published.
pub const EXIT_OK: u8 = 0;
/// A file is missing, the wrong size, the wrong digest, unsafe to open, or changed while read.
pub const EXIT_INTEGRITY: u8 = 1;
/// Bad command line, including an `--artifact` key the manifest does not have.
pub const EXIT_USAGE: u8 = 2;
/// The manifest is malformed or too large; no artifact was touched.
pub const EXIT_DATA: u8 = 65;
/// The manifest does not exist.
pub const EXIT_NO_INPUT: u8 = 66;
/// An I/O error prevented a check; the report keeps the causal error.
pub const EXIT_IO: u8 = 74;

/// What a run did not do, stated in every run report.
pub const NOT_PERFORMED: &[&str] = &[
    "download",
    "repair, deletion, rename or promotion of any file",
    "model admission or functional qualification",
    "completion-stamp lookup",
    "download-lock coordination: no queue lock is taken, so no downloader may write these destinations during a run",
];

/// The exit status for a manifest that could not be accepted.
pub fn manifest_exit_code(error: &ManifestError) -> u8 {
    match error {
        ManifestError::Read { error, .. } if error.kind() == io::ErrorKind::NotFound => {
            EXIT_NO_INPUT
        }
        ManifestError::Read { .. } => EXIT_IO,
        ManifestError::TooLarge { .. } | ManifestError::Malformed { .. } => EXIT_DATA,
    }
}

/// Filesystem identity of one file at one moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct FileIdentity {
    pub dev: u64,
    pub ino: u64,
    pub size: u64,
    pub mtime_sec: i64,
    pub mtime_nsec: i64,
    pub ctime_sec: i64,
    pub ctime_nsec: i64,
}

impl FileIdentity {
    pub fn of(metadata: &Metadata) -> Self {
        Self {
            dev: metadata.dev(),
            ino: metadata.ino(),
            size: metadata.size(),
            mtime_sec: metadata.mtime(),
            mtime_nsec: metadata.mtime_nsec(),
            ctime_sec: metadata.ctime(),
            ctime_nsec: metadata.ctime_nsec(),
        }
    }
}

/// An I/O error, kept with the operation and place it happened.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IoFailure {
    /// `lstat`, `canonicalize-parent`, `open`, `fstat` or `read`.
    pub operation: &'static str,
    pub path: String,
    pub errno: Option<i32>,
    pub errno_name: Option<&'static str>,
    pub error: String,
    /// Byte offset of a failed read.
    pub offset: Option<u64>,
}

/// The result of checking one file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "kebab-case")]
pub enum Outcome {
    /// Exact size, and the publisher SHA-256 matched every byte.
    VerifiedSha256 {
        identity: FileIdentity,
        bytes_hashed: u64,
    },
    /// Exact size. The queue publishes no digest, so the content was not read.
    SizeOnlyNoPublishedHash { identity: FileIdentity },
    /// Nothing exists at the destination.
    Missing,
    /// The file exists with a different size; it was not read.
    SizeMismatch {
        actual_size: u64,
        identity: FileIdentity,
    },
    /// The file has the right size and a different SHA-256.
    Sha256Mismatch {
        actual_sha256: String,
        identity: FileIdentity,
    },
    /// An I/O error prevented the check.
    Unreadable { failure: IoFailure },
    /// The destination is a symlink, sits under a symlinked directory, or is
    /// not a regular file. It was not read.
    UnsafePath { reason: String },
    /// The file changed, grew, shrank or was replaced during the check, so no
    /// verdict about its content is given.
    ChangedDuringRead { reason: String },
}

/// Counts by outcome for one run.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct Summary {
    pub files: usize,
    pub verified_sha256: usize,
    pub size_only_no_published_hash: usize,
    pub missing: usize,
    pub size_mismatch: usize,
    pub sha256_mismatch: usize,
    pub unreadable: usize,
    pub unsafe_path: usize,
    pub changed_during_read: usize,
}

impl Summary {
    pub fn record(&mut self, outcome: &Outcome) {
        self.files += 1;
        let slot = match outcome {
            Outcome::VerifiedSha256 { .. } => &mut self.verified_sha256,
            Outcome::SizeOnlyNoPublishedHash { .. } => &mut self.size_only_no_published_hash,
            Outcome::Missing => &mut self.missing,
            Outcome::SizeMismatch { .. } => &mut self.size_mismatch,
            Outcome::Sha256Mismatch { .. } => &mut self.sha256_mismatch,
            Outcome::Unreadable { .. } => &mut self.unreadable,
            Outcome::UnsafePath { .. } => &mut self.unsafe_path,
            Outcome::ChangedDuringRead { .. } => &mut self.changed_during_read,
        };
        *slot += 1;
    }

    /// 74 when any check hit an I/O error, else 1 for any integrity problem, else 0.
    pub fn exit_code(&self) -> u8 {
        if self.unreadable > 0 {
            EXIT_IO
        } else if self.missing
            + self.size_mismatch
            + self.sha256_mismatch
            + self.unsafe_path
            + self.changed_during_read
            > 0
        {
            EXIT_INTEGRITY
        } else {
            EXIT_OK
        }
    }
}

/// Where a manifest came from and what it declares.
#[derive(Debug, Clone, Serialize)]
pub struct ManifestSummary {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub schema: &'static str,
    pub built_at: Option<String>,
    pub total_bytes: u64,
    pub artifacts: usize,
    pub files: usize,
}

impl ManifestSummary {
    pub fn of(manifest: &QueueManifest) -> Self {
        Self {
            path: manifest.path.to_string_lossy().into_owned(),
            bytes: manifest.bytes,
            sha256: manifest.sha256.as_str().to_string(),
            schema: QUEUE_SCHEMA,
            built_at: manifest.built_at.clone(),
            total_bytes: manifest.total_bytes,
            artifacts: manifest.artifacts.len(),
            files: manifest.file_count(),
        }
    }
}

/// The selected artifacts and what checking them involves.
#[derive(Debug, Clone, Serialize)]
pub struct Selection {
    pub all_artifacts: bool,
    pub artifact_keys: Vec<String>,
    pub files: usize,
    pub bytes: u64,
    pub publisher_sha256_files: usize,
    pub size_only_files: usize,
}

impl Selection {
    pub fn of(manifest: &QueueManifest, selected: &[&Artifact]) -> Self {
        let files = || selected.iter().flat_map(|artifact| artifact.files.iter());
        let hashed = files().filter(|file| file.sha256.is_some()).count();
        let total = files().count();
        Self {
            all_artifacts: selected.len() == manifest.artifacts.len(),
            artifact_keys: selected.iter().map(|a| a.key.clone()).collect(),
            files: total,
            bytes: files().map(|file| file.size).sum(),
            publisher_sha256_files: hashed,
            size_only_files: total - hashed,
        }
    }
}

/// A metadata-only plan. Building it reads nothing but the manifest.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub schema: &'static str,
    pub mode: &'static str,
    pub artifact_access: &'static str,
    pub manifest: ManifestSummary,
    pub selection: Selection,
    pub artifacts: Vec<PlannedArtifact>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlannedArtifact {
    pub key: String,
    pub capability: String,
    pub repository: String,
    pub revision: String,
    pub files: Vec<PlannedFile>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlannedFile {
    pub repo_path: String,
    pub destination: String,
    pub size: u64,
    pub sha256: Option<String>,
    /// `sha256-and-size` or `size-only-no-published-hash`.
    pub check: &'static str,
}

/// Describe what a run over `selected` would check, without touching any artifact.
pub fn plan(manifest: &QueueManifest, selected: &[&Artifact]) -> Plan {
    Plan {
        schema: "frankenctl-artifact-verification-plan/1",
        mode: "plan",
        artifact_access: "none: only the manifest was read; no artifact path was examined",
        manifest: ManifestSummary::of(manifest),
        selection: Selection::of(manifest, selected),
        artifacts: selected
            .iter()
            .map(|artifact| PlannedArtifact {
                key: artifact.key.clone(),
                capability: artifact.capability.clone(),
                repository: artifact.repository.clone(),
                revision: artifact.revision.clone(),
                files: artifact
                    .files
                    .iter()
                    .map(|file| PlannedFile {
                        repo_path: file.repo_path.clone(),
                        destination: file.destination.to_string_lossy().into_owned(),
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
    }
}

/// One file's identity from the manifest and what the check found.
#[derive(Debug, Clone, Serialize)]
pub struct FileResult {
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

/// The report of one run.
#[derive(Debug, Clone, Serialize)]
pub struct RunReport {
    pub schema: &'static str,
    pub mode: &'static str,
    pub manifest: ManifestSummary,
    pub selection: Selection,
    pub boot_id: Option<String>,
    pub started_at_unix: u64,
    pub finished_at_unix: u64,
    pub hash_buffer_bytes: usize,
    pub not_performed: &'static [&'static str],
    pub results: Vec<FileResult>,
    pub summary: Summary,
    pub exit_code: u8,
}

/// Check every file of the selected artifacts, in manifest order.
pub fn run(manifest: &QueueManifest, selected: &[&Artifact]) -> RunReport {
    let started_at_unix = unix_now();
    let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
    let mut summary = Summary::default();
    let mut results = Vec::new();
    for artifact in selected {
        for file in &artifact.files {
            let outcome = verify_file_with(file, &mut buffer, &mut |_| {});
            summary.record(&outcome);
            results.push(FileResult {
                key: artifact.key.clone(),
                capability: artifact.capability.clone(),
                repository: artifact.repository.clone(),
                revision: artifact.revision.clone(),
                repo_path: file.repo_path.clone(),
                destination: file.destination.to_string_lossy().into_owned(),
                expected_size: file.size,
                expected_sha256: file.sha256.as_ref().map(|d| d.as_str().to_string()),
                outcome,
            });
        }
    }
    RunReport {
        schema: "frankenctl-artifact-verification-run/1",
        mode: "run",
        manifest: ManifestSummary::of(manifest),
        selection: Selection::of(manifest, selected),
        boot_id: std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .ok()
            .map(|id| id.trim().to_string()),
        started_at_unix,
        finished_at_unix: unix_now(),
        hash_buffer_bytes: HASH_BUFFER_BYTES,
        not_performed: NOT_PERFORMED,
        exit_code: summary.exit_code(),
        results,
        summary,
    }
}

/// Check one file with a fresh buffer.
pub fn verify_file(file: &ArtifactFile) -> Outcome {
    let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
    verify_file_with(file, &mut buffer, &mut |_| {})
}

/// Check one file, hashing through `buffer`. `after_chunk` is called with the
/// running byte count after each chunk is hashed; production passes a no-op,
/// tests use it to change the file mid-read.
///
/// The file is examined with `lstat`, then opened without following a final
/// symlink, and its `fstat` identity must match the `lstat` one. After the last
/// byte, both `fstat` and a fresh `lstat` must still show the same device,
/// inode, size, mtime and ctime, and the byte count must equal the size.
/// Anything else is [`Outcome::ChangedDuringRead`], not a digest verdict.
pub fn verify_file_with(
    file: &ArtifactFile,
    buffer: &mut [u8],
    after_chunk: &mut dyn FnMut(u64),
) -> Outcome {
    let path = file.destination.as_path();
    let linked = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
            ) =>
        {
            return Outcome::Missing
        }
        Err(error) => return unreadable("lstat", path, &error, None),
    };
    if linked.file_type().is_symlink() {
        return Outcome::UnsafePath {
            reason: "the destination is a symbolic link; links are not followed".to_string(),
        };
    }
    if !linked.file_type().is_file() {
        return Outcome::UnsafePath {
            reason: format!(
                "the destination is {}, not a regular file",
                file_kind(&linked)
            ),
        };
    }
    if let Some(parent) = path.parent() {
        match std::fs::canonicalize(parent) {
            Ok(real) if real == parent => {}
            Ok(real) => {
                return Outcome::UnsafePath {
                    reason: format!(
                        "the parent directory {} resolves through a symbolic link to {}",
                        parent.display(),
                        real.display()
                    ),
                }
            }
            Err(error) => return unreadable("canonicalize-parent", parent, &error, None),
        }
    }
    let before = FileIdentity::of(&linked);
    if before.size != file.size {
        return Outcome::SizeMismatch {
            actual_size: before.size,
            identity: before,
        };
    }
    let Some(expected) = &file.sha256 else {
        return Outcome::SizeOnlyNoPublishedHash { identity: before };
    };

    // O_NONBLOCK keeps a FIFO swapped in after the lstat from blocking the open.
    let mut handle = match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
    {
        Ok(handle) => handle,
        Err(error) if error.raw_os_error() == Some(libc::ELOOP) => {
            return Outcome::UnsafePath {
                reason: "the destination became a symbolic link before it was opened".to_string(),
            }
        }
        Err(error) => return unreadable("open", path, &error, None),
    };
    let opened = match fstat(&handle) {
        Ok(identity) => identity,
        Err(error) => return unreadable("fstat", path, &error, None),
    };
    if opened != before {
        return Outcome::ChangedDuringRead {
            reason: "the path was replaced or modified between lstat and open".to_string(),
        };
    }
    let digest = match hash_stream(&mut handle, file.size, buffer, after_chunk) {
        Ok(digest) => digest,
        Err(StreamError::Io { offset, error }) => {
            return unreadable("read", path, &error, Some(offset))
        }
        Err(StreamError::LongerThanExpected { read }) => {
            return Outcome::ChangedDuringRead {
                reason: format!(
                    "read {read} bytes, more than the {} the file had when it was opened",
                    file.size
                ),
            }
        }
    };
    let after = match fstat(&handle) {
        Ok(identity) => identity,
        Err(error) => return unreadable("fstat", path, &error, None),
    };
    if after != opened {
        return Outcome::ChangedDuringRead {
            reason: "the file's size or timestamps changed while it was read".to_string(),
        };
    }
    if digest.bytes != file.size {
        return Outcome::ChangedDuringRead {
            reason: format!(
                "read {} bytes from a file of {} bytes",
                digest.bytes, file.size
            ),
        };
    }
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() && FileIdentity::of(&metadata) == after => {}
        Ok(_) => {
            return Outcome::ChangedDuringRead {
                reason: "after the read, the path names a different or modified file".to_string(),
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Outcome::ChangedDuringRead {
                reason: "the path was removed during the read".to_string(),
            }
        }
        Err(error) => return unreadable("lstat", path, &error, None),
    }
    if digest.sha256 == expected.as_str() {
        Outcome::VerifiedSha256 {
            identity: after,
            bytes_hashed: digest.bytes,
        }
    } else {
        Outcome::Sha256Mismatch {
            actual_sha256: digest.sha256,
            identity: after,
        }
    }
}

/// The digest and length of a completely read stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamDigest {
    pub sha256: String,
    pub bytes: u64,
}

/// Why a stream could not be hashed.
#[derive(Debug)]
pub enum StreamError {
    /// A read failed at `offset`. The error is the one the reader returned.
    Io { offset: u64, error: io::Error },
    /// The stream produced more than the expected number of bytes; reading
    /// stopped at the first chunk past the limit.
    LongerThanExpected { read: u64 },
}

/// Hash `reader` to EOF through `buffer`, the only memory the read uses.
///
/// Interrupted reads are retried. Any other error ends the hash and is
/// returned with its offset. Reading stops as soon as more than `expected_len`
/// bytes have arrived, so a growing file cannot make the read unbounded.
pub fn hash_stream<R: Read + ?Sized>(
    reader: &mut R,
    expected_len: u64,
    buffer: &mut [u8],
    after_chunk: &mut dyn FnMut(u64),
) -> Result<StreamDigest, StreamError> {
    assert!(!buffer.is_empty(), "the hash buffer must not be empty");
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    loop {
        let read = match reader.read(buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                return Err(StreamError::Io {
                    offset: total,
                    error,
                })
            }
        };
        total += read as u64;
        if total > expected_len {
            return Err(StreamError::LongerThanExpected { read: total });
        }
        hasher.update(&buffer[..read]);
        after_chunk(total);
    }
    Ok(StreamDigest {
        sha256: to_hex(&hasher.finalize()),
        bytes: total,
    })
}

fn fstat(handle: &File) -> io::Result<FileIdentity> {
    handle
        .metadata()
        .map(|metadata| FileIdentity::of(&metadata))
}

fn unreadable(
    operation: &'static str,
    path: &Path,
    error: &io::Error,
    offset: Option<u64>,
) -> Outcome {
    Outcome::Unreadable {
        failure: IoFailure {
            operation,
            path: path.to_string_lossy().into_owned(),
            errno: error.raw_os_error(),
            errno_name: error.raw_os_error().and_then(errno_name),
            error: error.to_string(),
            offset,
        },
    }
}

fn file_kind(metadata: &Metadata) -> &'static str {
    let kind = metadata.file_type();
    if kind.is_dir() {
        "a directory"
    } else if kind.is_fifo() {
        "a FIFO"
    } else if kind.is_socket() {
        "a socket"
    } else if kind.is_block_device() {
        "a block device"
    } else if kind.is_char_device() {
        "a character device"
    } else {
        "an unrecognized file type"
    }
}

fn errno_name(code: i32) -> Option<&'static str> {
    Some(match code {
        libc::EPERM => "EPERM",
        libc::ENOENT => "ENOENT",
        libc::EIO => "EIO",
        libc::ENXIO => "ENXIO",
        libc::EBADF => "EBADF",
        libc::ENOMEM => "ENOMEM",
        libc::EACCES => "EACCES",
        libc::ENODEV => "ENODEV",
        libc::ENOTDIR => "ENOTDIR",
        libc::EISDIR => "EISDIR",
        libc::EINVAL => "EINVAL",
        libc::EFBIG => "EFBIG",
        libc::ELOOP => "ELOOP",
        libc::EBADE => "EBADE",
        libc::EOVERFLOW => "EOVERFLOW",
        libc::ETIMEDOUT => "ETIMEDOUT",
        libc::ESTALE => "ESTALE",
        libc::EUCLEAN => "EUCLEAN",
        _ => return None,
    })
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
    use crate::manifest::Sha256Hex;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A disposable directory under a canonical temp root, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().canonicalize().expect("temp dir");
            let dir = root.join(format!(
                "frankenctl-verify-{tag}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&dir).expect("create scratch dir");
            Self(dir)
        }

        /// Write `bytes` and backdate its mtime, so an in-place write during a
        /// test changes the mtime even on a coarse filesystem clock.
        fn artifact(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            let old = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000);
            File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(old)
                .unwrap();
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn entry(destination: &Path, size: u64, sha256: Option<Sha256Hex>) -> ArtifactFile {
        ArtifactFile {
            repo_path: "fixture.bin".to_string(),
            destination: destination.to_path_buf(),
            size,
            sha256,
        }
    }

    fn running_as_root() -> bool {
        // SAFETY: geteuid has no preconditions and cannot fail.
        unsafe { libc::geteuid() == 0 }
    }

    /// Yields `total` copies of `byte`, recording the largest buffer offered.
    struct Repeat {
        byte: u8,
        remaining: u64,
        largest_request: usize,
    }

    impl Read for Repeat {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.largest_request = self.largest_request.max(buf.len());
            let n = buf.len().min(self.remaining as usize);
            buf[..n].fill(self.byte);
            self.remaining -= n as u64;
            Ok(n)
        }
    }

    #[test]
    fn hash_stream_matches_published_sha256_vectors() {
        let mut none = |_| {};
        let mut one = [0u8; 1];
        let abc = hash_stream(&mut &b"abc"[..], 3, &mut one, &mut none).unwrap();
        assert_eq!(
            abc.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let empty = hash_stream(&mut &b""[..], 0, &mut one, &mut none).unwrap();
        assert_eq!(
            empty.sha256,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let mut million = Repeat {
            byte: b'a',
            remaining: 1_000_000,
            largest_request: 0,
        };
        let mut buffer = vec![0u8; 4096];
        let digest = hash_stream(&mut million, 1_000_000, &mut buffer, &mut none).unwrap();
        assert_eq!(
            digest.sha256,
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
        assert_eq!(digest.bytes, 1_000_000);
    }

    #[test]
    fn hash_stream_memory_is_the_fixed_buffer_whatever_the_stream_length() {
        let length = 3 * 4096 + 17;
        let mut source = Repeat {
            byte: 7,
            remaining: length,
            largest_request: 0,
        };
        let mut buffer = vec![0u8; 4096];
        let mut chunks = Vec::new();
        let digest =
            hash_stream(&mut source, length, &mut buffer, &mut |n| chunks.push(n)).unwrap();
        assert_eq!(digest.bytes, length);
        assert_eq!(source.largest_request, 4096);
        assert_eq!(chunks, [4096, 8192, 12288, 12305]);
        assert_eq!(
            digest.sha256,
            Sha256Hex::of(&vec![7u8; length as usize]).as_str()
        );
        assert_eq!(HASH_BUFFER_BYTES, 8 * 1024 * 1024);
    }

    #[test]
    fn hash_stream_keeps_the_os_error_and_its_offset() {
        struct FailsAfter(usize);
        impl Read for FailsAfter {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if self.0 == 0 {
                    return Err(io::Error::from_raw_os_error(libc::EIO));
                }
                self.0 -= 1;
                buf.fill(1);
                Ok(buf.len())
            }
        }
        let mut buffer = [0u8; 8];
        match hash_stream(&mut FailsAfter(2), 64, &mut buffer, &mut |_| {}) {
            Err(StreamError::Io { offset, error }) => {
                assert_eq!(offset, 16);
                assert_eq!(error.raw_os_error(), Some(libc::EIO));
            }
            other => panic!("expected the EIO to be returned, got {other:?}"),
        }
    }

    #[test]
    fn hash_stream_retries_interrupted_reads_only() {
        struct InterruptOnce(bool, &'static [u8]);
        impl Read for InterruptOnce {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if !self.0 {
                    self.0 = true;
                    return Err(io::Error::from(io::ErrorKind::Interrupted));
                }
                self.1.read(buf)
            }
        }
        let mut buffer = [0u8; 2];
        let digest = hash_stream(
            &mut InterruptOnce(false, b"abc"),
            3,
            &mut buffer,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(digest.sha256, Sha256Hex::of(b"abc").as_str());
    }

    #[test]
    fn hash_stream_stops_once_a_stream_outgrows_its_expected_length() {
        let mut source = Repeat {
            byte: 0,
            remaining: 1 << 30,
            largest_request: 0,
        };
        let mut buffer = [0u8; 4];
        match hash_stream(&mut source, 10, &mut buffer, &mut |_| {}) {
            Err(StreamError::LongerThanExpected { read }) => assert_eq!(read, 12),
            other => panic!("{other:?}"),
        }
        assert_eq!(source.remaining, (1 << 30) - 12);
    }

    #[test]
    fn a_matching_publisher_digest_verifies() {
        let scratch = Scratch::new("ok");
        let path = scratch.artifact("ok.bin", b"frankenstein");
        let outcome = verify_file(&entry(&path, 12, Some(Sha256Hex::of(b"frankenstein"))));
        match outcome {
            Outcome::VerifiedSha256 {
                bytes_hashed,
                identity,
            } => {
                assert_eq!(bytes_hashed, 12);
                assert_eq!(identity.size, 12);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_file_without_a_published_digest_is_size_checked_and_never_read() {
        if running_as_root() {
            return;
        }
        let scratch = Scratch::new("size-only");
        let path = scratch.artifact("unreadable.json", b"{}");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
        // Unopenable, yet size-only: proof that no read was attempted.
        assert!(matches!(
            verify_file(&entry(&path, 2, None)),
            Outcome::SizeOnlyNoPublishedHash { .. }
        ));
        assert!(matches!(
            verify_file(&entry(&path, 3, None)),
            Outcome::SizeMismatch { actual_size: 2, .. }
        ));
    }

    #[test]
    fn missing_wrong_size_and_wrong_digest_are_distinct_outcomes() {
        let scratch = Scratch::new("integrity");
        let path = scratch.artifact("a.bin", b"abc");
        assert_eq!(
            verify_file(&entry(&scratch.0.join("absent.bin"), 3, None)),
            Outcome::Missing
        );
        assert_eq!(
            verify_file(&entry(&scratch.0.join("no-dir/absent.bin"), 3, None)),
            Outcome::Missing
        );
        assert!(matches!(
            verify_file(&entry(&path, 4, Some(Sha256Hex::of(b"abcd")))),
            Outcome::SizeMismatch { actual_size: 3, .. }
        ));
        match verify_file(&entry(&path, 3, Some(Sha256Hex::of(b"abd")))) {
            Outcome::Sha256Mismatch { actual_sha256, .. } => {
                assert_eq!(actual_sha256, Sha256Hex::of(b"abc").as_str())
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_open_failure_keeps_its_errno_and_path() {
        if running_as_root() {
            return;
        }
        let scratch = Scratch::new("eacces");
        let path = scratch.artifact("locked.bin", b"abc");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
        match verify_file(&entry(&path, 3, Some(Sha256Hex::of(b"abc")))) {
            Outcome::Unreadable { failure } => {
                assert_eq!(failure.operation, "open");
                assert_eq!(failure.errno, Some(libc::EACCES));
                assert_eq!(failure.errno_name, Some("EACCES"));
                assert_eq!(failure.path, path.to_string_lossy());
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_real_kernel_eio_is_reported_as_unreadable_with_its_cause() {
        // /proc/<pid>/mem is a regular file of size 0 whose read at offset 0
        // fails with EIO: a genuine kernel I/O error, no mock involved.
        let path = PathBuf::from(format!("/proc/{}/mem", std::process::id()));
        let outcome = verify_file(&entry(&path, 0, Some(Sha256Hex::of(b""))));
        match &outcome {
            Outcome::Unreadable { failure } => {
                assert_eq!(failure.operation, "read");
                assert_eq!(failure.errno, Some(libc::EIO));
                assert_eq!(failure.errno_name, Some("EIO"));
                assert_eq!(failure.offset, Some(0));
                assert_eq!(failure.path, path.to_string_lossy());
            }
            other => panic!("{other:?}"),
        }
        let mut summary = Summary::default();
        summary.record(&outcome);
        assert_eq!(summary.exit_code(), EXIT_IO);
    }

    #[test]
    fn symlinks_and_non_regular_files_are_refused_unread() {
        let scratch = Scratch::new("unsafe");
        let real = scratch.artifact("real.bin", b"abc");
        let link = scratch.0.join("link.bin");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let digest = Some(Sha256Hex::of(b"abc"));
        assert!(matches!(
            verify_file(&entry(&link, 3, digest.clone())),
            Outcome::UnsafePath { .. }
        ));

        let linked_dir = scratch.0.join("linked-dir");
        std::os::unix::fs::symlink(&scratch.0, &linked_dir).unwrap();
        match verify_file(&entry(&linked_dir.join("real.bin"), 3, digest.clone())) {
            Outcome::UnsafePath { reason } => assert!(reason.contains("symbolic link"), "{reason}"),
            other => panic!("{other:?}"),
        }

        let directory = scratch.0.join("dir.bin");
        std::fs::create_dir(&directory).unwrap();
        match verify_file(&entry(&directory, 3, digest)) {
            Outcome::UnsafePath { reason } => assert!(reason.contains("a directory"), "{reason}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_file_that_grows_during_the_read_gets_no_verdict() {
        let scratch = Scratch::new("grow");
        let path = scratch.artifact("grow.bin", &[1u8; 16]);
        let digest = Some(Sha256Hex::of(&[1u8; 16]));
        let mut buffer = [0u8; 4];
        let target = path.clone();
        let outcome = verify_file_with(&entry(&path, 16, digest), &mut buffer, &mut |n| {
            if n == 4 {
                let mut f = File::options().append(true).open(&target).unwrap();
                f.write_all(&[2u8; 4]).unwrap();
            }
        });
        assert!(
            matches!(outcome, Outcome::ChangedDuringRead { .. }),
            "{outcome:?}"
        );
    }

    #[test]
    fn an_in_place_rewrite_during_the_read_gets_no_verdict() {
        let scratch = Scratch::new("rewrite");
        let path = scratch.artifact("rewrite.bin", &[1u8; 16]);
        let digest = Some(Sha256Hex::of(&[1u8; 16]));
        let mut buffer = [0u8; 4];
        let target = path.clone();
        // Rewrites bytes that were already hashed, so the digest of what was
        // read still matches; only the metadata check can catch it.
        let outcome = verify_file_with(&entry(&path, 16, digest), &mut buffer, &mut |n| {
            if n == 8 {
                let mut f = File::options().write(true).open(&target).unwrap();
                f.write_all(&[9u8; 4]).unwrap();
            }
        });
        match outcome {
            Outcome::ChangedDuringRead { reason } => {
                assert!(reason.contains("changed while it was read"), "{reason}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_file_renamed_over_the_path_during_the_read_gets_no_verdict() {
        let scratch = Scratch::new("replace");
        let path = scratch.artifact("replace.bin", &[1u8; 16]);
        let other = scratch.artifact("other.bin", &[1u8; 16]);
        let digest = Some(Sha256Hex::of(&[1u8; 16]));
        let mut buffer = [0u8; 4];
        let (from, to) = (other.clone(), path.clone());
        // What a downloader's promotion does. Unlinking the original bumps its
        // ctime, so either identity check may be the one that fires.
        let outcome = verify_file_with(&entry(&path, 16, digest), &mut buffer, &mut |n| {
            if n == 4 {
                std::fs::rename(&from, &to).unwrap();
            }
        });
        assert!(
            matches!(outcome, Outcome::ChangedDuringRead { .. }),
            "{outcome:?}"
        );
    }

    #[test]
    fn a_parent_directory_swapped_during_the_read_gets_no_verdict() {
        let scratch = Scratch::new("swap");
        let current = scratch.0.join("current");
        let incoming = scratch.0.join("incoming");
        std::fs::create_dir(&current).unwrap();
        std::fs::create_dir(&incoming).unwrap();
        let path = scratch.artifact("current/weights.bin", &[1u8; 16]);
        scratch.artifact("incoming/weights.bin", &[1u8; 16]);
        let digest = Some(Sha256Hex::of(&[1u8; 16]));
        let mut buffer = [0u8; 4];
        let retired = scratch.0.join("retired");
        let (current_dir, incoming_dir) = (current.clone(), incoming.clone());
        // The inode being read is untouched, so its descriptor shows no change;
        // only the fresh lstat after the read sees that the path moved on.
        let outcome = verify_file_with(&entry(&path, 16, digest), &mut buffer, &mut |n| {
            if n == 4 {
                std::fs::rename(&current_dir, &retired).unwrap();
                std::fs::rename(&incoming_dir, &current_dir).unwrap();
            }
        });
        match outcome {
            Outcome::ChangedDuringRead { reason } => {
                assert!(reason.contains("different or modified file"), "{reason}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn io_errors_outrank_integrity_failures_in_the_exit_status() {
        let mut summary = Summary::default();
        summary.record(&Outcome::SizeOnlyNoPublishedHash {
            identity: FileIdentity {
                dev: 0,
                ino: 0,
                size: 0,
                mtime_sec: 0,
                mtime_nsec: 0,
                ctime_sec: 0,
                ctime_nsec: 0,
            },
        });
        assert_eq!(summary.exit_code(), EXIT_OK);
        summary.record(&Outcome::Missing);
        assert_eq!(summary.exit_code(), EXIT_INTEGRITY);
        summary.record(&Outcome::Unreadable {
            failure: IoFailure {
                operation: "read",
                path: "/x".into(),
                errno: Some(libc::EIO),
                errno_name: Some("EIO"),
                error: "Input/output error (os error 5)".into(),
                offset: Some(0),
            },
        });
        assert_eq!(summary.exit_code(), EXIT_IO);
        assert_eq!(summary.files, 3);
    }
}
