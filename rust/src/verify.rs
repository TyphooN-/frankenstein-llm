//! Read-only verification of one artifact file against its recorded identity.
//!
//! This is the per-file core of `frankenctl verify`: exact size always, and
//! SHA-256 when a publisher digest is recorded. It never downloads, repairs,
//! deletes, renames or promotes a file, never admits a model or grants
//! qualification, and never reads a completion stamp: every check judges the
//! bytes on disk at that moment. Writer locks, selection, cancellation and the
//! published report belong to [`crate::verification`].
//!
//! Content is hashed through one fixed buffer of [`HASH_BUFFER_BYTES`], so
//! memory does not grow with file size. An I/O error keeps its operation, path,
//! OS error number and offset, and makes the run exit 74; it is never folded
//! into a mismatch or a missing file. A cancellation seen while a file is read
//! ends that check with no verdict.

use std::fs::{File, Metadata};
use std::io::{self, Read};
use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt};
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::manifest::{to_hex, ArtifactFile, ManifestError};

/// Size of the single read buffer used to hash a file.
pub const HASH_BUFFER_BYTES: usize = 8 * 1024 * 1024;

/// Every selected file matched: SHA-256 and size, or size where no digest is published.
pub const EXIT_OK: u8 = 0;
/// A file is missing, the wrong size, the wrong digest, unsafe to open, or
/// changed while read; or a model file a preset or sidecar loads is unaccounted.
pub const EXIT_INTEGRITY: u8 = 1;
/// Bad command line, including a `--source` or `--artifact` that does not exist.
pub const EXIT_USAGE: u8 = 2;
/// The manifest or inventory is malformed or too large; no artifact was touched.
pub const EXIT_DATA: u8 = 65;
/// The manifest or inventory does not exist.
pub const EXIT_NO_INPUT: u8 = 66;
/// An internal failure, such as signal handlers that could not be installed.
pub const EXIT_SOFTWARE: u8 = 70;
/// The report could not be published durably; stdout still carries it.
pub const EXIT_CANT_CREATE: u8 = 73;
/// An I/O error prevented a check; the report keeps the causal error.
pub const EXIT_IO: u8 = 74;
/// A writer lock is held by another process; no artifact was examined.
pub const EXIT_LOCKED: u8 = 75;
/// Added to the number of the signal that interrupted a run.
pub const EXIT_SIGNAL_BASE: u8 = 128;

/// The exit status for a manifest or inventory that could not be accepted.
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
    /// Exact size. No digest is recorded, so the content was not read.
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
    /// The file changed, grew, shrank or was replaced during the check or
    /// before the verification window closed, so no verdict is given.
    ChangedDuringRead { reason: String },
    /// The run was cancelled before or while this file was read: no verdict.
    NotChecked { reason: String },
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
    pub not_checked: usize,
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
            Outcome::NotChecked { .. } => &mut self.not_checked,
        };
        *slot += 1;
    }

    /// 74 when any check hit an I/O error, else 1 for any integrity problem, else 0.
    ///
    /// Files not checked do not count here: only a cancelled run leaves them,
    /// and its exit status is the signal's.
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

/// Check one file with a fresh buffer and no cancellation.
pub fn verify_file(file: &ArtifactFile) -> Outcome {
    let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
    verify_file_with(file, &mut buffer, &mut |_| {}, &|| None)
}

/// Check one file, hashing through `buffer`. `after_chunk` is called with the
/// running byte count after each chunk is hashed; `cancelled` is asked before
/// every read and returns the number of a signal that should stop the check.
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
    cancelled: &dyn Fn() -> Option<i32>,
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
    let digest = match hash_stream(&mut handle, file.size, buffer, after_chunk, cancelled) {
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
        Err(StreamError::Cancelled { signal, read }) => {
            return Outcome::NotChecked {
                reason: format!(
                    "interrupted by signal {signal} after {read} of {} bytes were read",
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

/// Confirm, as the verification window closes, that `file`'s destination still
/// names the file `outcome` describes.
///
/// A verdict is about the bytes that were read. Writers that honour the locks
/// cannot change them during a run, but anything else could, and a report that
/// says `verified-sha256` for a path that names different bytes by the time the
/// report is published would be a false statement. So every outcome that
/// carries an identity is compared with a fresh `lstat`, and a missing file must
/// still be missing. Anything else becomes [`Outcome::ChangedDuringRead`].
pub fn confirm_at_window_close(file: &ArtifactFile, outcome: Outcome) -> Outcome {
    let identity = match &outcome {
        Outcome::VerifiedSha256 { identity, .. }
        | Outcome::SizeOnlyNoPublishedHash { identity }
        | Outcome::SizeMismatch { identity, .. }
        | Outcome::Sha256Mismatch { identity, .. } => Some(*identity),
        Outcome::Missing => None,
        _ => return outcome,
    };
    let path = file.destination.as_path();
    let absent = |error: &io::Error| {
        matches!(
            error.kind(),
            io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
        )
    };
    match (std::fs::symlink_metadata(path), identity) {
        (Ok(now), Some(then)) if now.file_type().is_file() && FileIdentity::of(&now) == then => {
            outcome
        }
        (Ok(_), Some(_)) => Outcome::ChangedDuringRead {
            reason: "when the verification window closed, the path named a different or modified file"
                .to_string(),
        },
        (Err(error), Some(_)) if absent(&error) => Outcome::ChangedDuringRead {
            reason: "the file was removed before the verification window closed".to_string(),
        },
        (Err(error), None) if absent(&error) => outcome,
        (Ok(_), None) => Outcome::ChangedDuringRead {
            reason: "a file appeared at the destination before the verification window closed"
                .to_string(),
        },
        (Err(error), _) => unreadable("lstat", path, &error, None),
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
    /// `cancelled` reported `signal` after `read` bytes.
    Cancelled { signal: i32, read: u64 },
}

/// Hash `reader` to EOF through `buffer`, the only memory the read uses.
///
/// `cancelled` is asked before every read, including a retry after an
/// interrupted one, so a signal stops the hash within one chunk. Interrupted
/// reads are otherwise retried. Any other error ends the hash and is returned
/// with its offset. Reading stops as soon as more than `expected_len` bytes have
/// arrived, so a growing file cannot make the read unbounded.
pub fn hash_stream<R: Read + ?Sized>(
    reader: &mut R,
    expected_len: u64,
    buffer: &mut [u8],
    after_chunk: &mut dyn FnMut(u64),
    cancelled: &dyn Fn() -> Option<i32>,
) -> Result<StreamDigest, StreamError> {
    assert!(!buffer.is_empty(), "the hash buffer must not be empty");
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    loop {
        if let Some(signal) = cancelled() {
            return Err(StreamError::Cancelled {
                signal,
                read: total,
            });
        }
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

/// The symbolic name of an errno value this verifier can meet.
pub fn errno_name(code: i32) -> Option<&'static str> {
    Some(match code {
        libc::EPERM => "EPERM",
        libc::ENOENT => "ENOENT",
        libc::EINTR => "EINTR",
        libc::EIO => "EIO",
        libc::ENXIO => "ENXIO",
        libc::EBADF => "EBADF",
        libc::EAGAIN => "EAGAIN",
        libc::ENOMEM => "ENOMEM",
        libc::EACCES => "EACCES",
        libc::EEXIST => "EEXIST",
        libc::ENODEV => "ENODEV",
        libc::ENOTDIR => "ENOTDIR",
        libc::EISDIR => "EISDIR",
        libc::EINVAL => "EINVAL",
        libc::EFBIG => "EFBIG",
        libc::ENOSPC => "ENOSPC",
        libc::EROFS => "EROFS",
        libc::ELOOP => "ELOOP",
        libc::ENOTEMPTY => "ENOTEMPTY",
        libc::EBADE => "EBADE",
        libc::EOVERFLOW => "EOVERFLOW",
        libc::ETIMEDOUT => "ETIMEDOUT",
        libc::ESTALE => "ESTALE",
        libc::EUCLEAN => "EUCLEAN",
        libc::EDQUOT => "EDQUOT",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Sha256Hex;
    use std::cell::Cell;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::SystemTime;

    fn never() -> Option<i32> {
        None
    }

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
        let abc = hash_stream(&mut &b"abc"[..], 3, &mut one, &mut none, &never).unwrap();
        assert_eq!(
            abc.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let empty = hash_stream(&mut &b""[..], 0, &mut one, &mut none, &never).unwrap();
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
        let digest =
            hash_stream(&mut million, 1_000_000, &mut buffer, &mut none, &never).unwrap();
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
        let digest = hash_stream(
            &mut source,
            length,
            &mut buffer,
            &mut |n| chunks.push(n),
            &never,
        )
        .unwrap();
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
        match hash_stream(&mut FailsAfter(2), 64, &mut buffer, &mut |_| {}, &never) {
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
            &never,
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
        match hash_stream(&mut source, 10, &mut buffer, &mut |_| {}, &never) {
            Err(StreamError::LongerThanExpected { read }) => assert_eq!(read, 12),
            other => panic!("{other:?}"),
        }
        assert_eq!(source.remaining, (1 << 30) - 12);
    }

    #[test]
    fn hash_stream_stops_within_one_chunk_of_a_cancellation() {
        let mut source = Repeat {
            byte: 0,
            remaining: 1 << 30,
            largest_request: 0,
        };
        let mut buffer = [0u8; 4];
        let signalled = Cell::new(None);
        let result = hash_stream(
            &mut source,
            1 << 30,
            &mut buffer,
            &mut |n| {
                if n == 8 {
                    signalled.set(Some(libc::SIGTERM));
                }
            },
            &|| signalled.get(),
        );
        match result {
            Err(StreamError::Cancelled { signal, read }) => {
                assert_eq!(signal, libc::SIGTERM);
                assert_eq!(read, 8);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(source.remaining, (1 << 30) - 8);
    }

    #[test]
    fn hash_stream_checks_cancellation_after_an_interrupted_read() {
        struct AlwaysInterrupted;
        impl Read for AlwaysInterrupted {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::from(io::ErrorKind::Interrupted))
            }
        }
        let asked = Cell::new(0);
        let result = hash_stream(
            &mut AlwaysInterrupted,
            4,
            &mut [0u8; 4],
            &mut |_| {},
            &|| {
                asked.set(asked.get() + 1);
                (asked.get() == 3).then_some(libc::SIGINT)
            },
        );
        assert!(matches!(
            result,
            Err(StreamError::Cancelled {
                signal: libc::SIGINT,
                read: 0
            })
        ));
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
        let outcome = verify_file_with(
            &entry(&path, 16, digest),
            &mut buffer,
            &mut |n| {
                if n == 4 {
                    let mut f = File::options().append(true).open(&target).unwrap();
                    f.write_all(&[2u8; 4]).unwrap();
                }
            },
            &never,
        );
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
        let outcome = verify_file_with(
            &entry(&path, 16, digest),
            &mut buffer,
            &mut |n| {
                if n == 8 {
                    let mut f = File::options().write(true).open(&target).unwrap();
                    f.write_all(&[9u8; 4]).unwrap();
                }
            },
            &never,
        );
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
        let outcome = verify_file_with(
            &entry(&path, 16, digest),
            &mut buffer,
            &mut |n| {
                if n == 4 {
                    std::fs::rename(&from, &to).unwrap();
                }
            },
            &never,
        );
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
        let outcome = verify_file_with(
            &entry(&path, 16, digest),
            &mut buffer,
            &mut |n| {
                if n == 4 {
                    std::fs::rename(&current_dir, &retired).unwrap();
                    std::fs::rename(&incoming_dir, &current_dir).unwrap();
                }
            },
            &never,
        );
        match outcome {
            Outcome::ChangedDuringRead { reason } => {
                assert!(reason.contains("different or modified file"), "{reason}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_cancelled_read_gets_no_verdict_and_names_the_signal() {
        let scratch = Scratch::new("cancel");
        let path = scratch.artifact("cancel.bin", &[3u8; 16]);
        let mut buffer = [0u8; 4];
        let signalled = Cell::new(None);
        let outcome = verify_file_with(
            &entry(&path, 16, Some(Sha256Hex::of(&[3u8; 16]))),
            &mut buffer,
            &mut |n| {
                if n == 4 {
                    signalled.set(Some(libc::SIGTERM));
                }
            },
            &|| signalled.get(),
        );
        match outcome {
            Outcome::NotChecked { reason } => {
                assert!(reason.contains("signal 15 after 4 of 16"), "{reason}")
            }
            other => panic!("{other:?}"),
        }
        let mut summary = Summary::default();
        summary.record(&Outcome::NotChecked {
            reason: String::new(),
        });
        assert_eq!((summary.not_checked, summary.exit_code()), (1, EXIT_OK));
    }

    #[test]
    fn the_window_close_check_catches_a_file_changed_after_its_verdict() {
        let scratch = Scratch::new("window");
        let path = scratch.artifact("weights.bin", b"abc");
        let file = entry(&path, 3, Some(Sha256Hex::of(b"abc")));
        let verified = verify_file(&file);
        assert!(matches!(verified, Outcome::VerifiedSha256 { .. }));
        assert_eq!(confirm_at_window_close(&file, verified.clone()), verified);

        // Same size, same bytes, new inode: only identity can tell.
        let replacement = scratch.artifact("replacement.bin", b"abc");
        std::fs::rename(&replacement, &path).unwrap();
        match confirm_at_window_close(&file, verified.clone()) {
            Outcome::ChangedDuringRead { reason } => {
                assert!(reason.contains("window closed"), "{reason}")
            }
            other => panic!("{other:?}"),
        }

        std::fs::remove_file(&path).unwrap();
        assert!(matches!(
            confirm_at_window_close(&file, verified),
            Outcome::ChangedDuringRead { .. }
        ));
        assert_eq!(confirm_at_window_close(&file, Outcome::Missing), Outcome::Missing);
        scratch.artifact("weights.bin", b"abc");
        assert!(matches!(
            confirm_at_window_close(&file, Outcome::Missing),
            Outcome::ChangedDuringRead { .. }
        ));
        let unsafe_path = Outcome::UnsafePath {
            reason: "kept".to_string(),
        };
        assert_eq!(confirm_at_window_close(&file, unsafe_path.clone()), unsafe_path);
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

    #[test]
    fn storage_errnos_have_names() {
        for (code, name) in [
            (libc::EIO, "EIO"),
            (libc::EROFS, "EROFS"),
            (libc::ENOSPC, "ENOSPC"),
            (libc::EDQUOT, "EDQUOT"),
            (libc::EUCLEAN, "EUCLEAN"),
        ] {
            assert_eq!(errno_name(code), Some(name));
        }
        assert_eq!(errno_name(99_999), None);
    }
}
