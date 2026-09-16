//! Writer locks: how verification and the artifact writers exclude each other.
//!
//! Every inventoried manifest has one writer lock, the manifest's own path with
//! its extension replaced by `lock`, and every writer of that manifest's
//! destinations holds it with `flock(2)` for as long as it may write:
//!
//! * `download_service.py` takes the four maintained queues' locks before its
//!   first queue and holds them for its whole run;
//! * `download_queue.py` takes `HERMES_DOWNLOAD_LOCK`, which defaults to the
//!   queue manifest's paired lock;
//! * `glm53flash-local/download_and_verify.py` takes `manifest.lock` beside
//!   `manifest.tsv`, and `download_obliterated_mmproj.py` takes
//!   `obliterated-mmproj-manifest.lock` beside its manifest.
//!
//! `frankenctl verify run` takes every inventoried source's lock, without
//! waiting, after the inventory is validated and before any artifact path is
//! examined, and holds all of them until its report is published. A writer that
//! starts meanwhile exits 75 rather than rewriting a file that is being read; a
//! verification that finds a lock held exits 75 rather than reading files that
//! are being written. The lock is `flock(2)`, called directly: the other party
//! is Python's `fcntl.flock`, and `File::try_lock` does not promise that
//! primitive.
//!
//! A lock excludes only processes that take it. A writer that ignores it, or a
//! lock file deleted while its holder still runs, defeats the exclusion; the
//! per-file identity checks in [`crate::verify`] still catch a file that changes
//! while it is read and report it as such rather than scoring it.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

/// The lock a writer holds while it writes `manifest`'s destinations: the
/// manifest's own path with its extension replaced by `lock`.
pub fn lock_path_for(manifest: &Path) -> PathBuf {
    manifest.with_extension("lock")
}

/// An exclusive `flock(2)` on a writer lock file, held until dropped.
///
/// Dropping closes the descriptor, which releases the lock; so does any exit of
/// the process, including death by a signal.
#[derive(Debug)]
pub struct QueueLock {
    path: PathBuf,
    // Never read or written: owning the descriptor is what holds the lock.
    _file: File,
}

/// Why a writer lock was not taken.
#[derive(Debug)]
pub enum LockError {
    /// Another process holds the lock: a writer or another verification.
    Held { path: PathBuf },
    /// The path stopped naming the opened file before the lock was taken, so
    /// that lock would not exclude a writer that opens the path now.
    Replaced { path: PathBuf },
    /// Something other than a regular file is at the lock path.
    NotRegular { path: PathBuf },
    /// Opening, creating, inspecting or locking the file failed.
    Io {
        operation: &'static str,
        path: PathBuf,
        error: io::Error,
    },
}

impl LockError {
    /// 75 when another process may still be writing, 74 for anything else.
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Held { .. } | Self::Replaced { .. } => crate::verify::EXIT_LOCKED,
            Self::NotRegular { .. } | Self::Io { .. } => crate::verify::EXIT_IO,
        }
    }

    /// The lock file the refusal is about.
    pub fn path(&self) -> &Path {
        match self {
            Self::Held { path }
            | Self::Replaced { path }
            | Self::NotRegular { path }
            | Self::Io { path, .. } => path,
        }
    }

    /// The operating-system error, when one caused the refusal.
    pub fn os_error(&self) -> Option<&io::Error> {
        match self {
            Self::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for LockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Held { path } => write!(
                f,
                "another process holds the writer lock {}",
                path.display()
            ),
            Self::Replaced { path } => write!(
                f,
                "the writer lock {} was replaced while it was being taken",
                path.display()
            ),
            Self::NotRegular { path } => write!(
                f,
                "the writer lock {} is not a regular file",
                path.display()
            ),
            Self::Io {
                operation,
                path,
                error,
            } => write!(
                f,
                "cannot {operation} the writer lock {}: {error} [{}]",
                path.display(),
                error
                    .raw_os_error()
                    .and_then(crate::verify::errno_name)
                    .unwrap_or("no errno name")
            ),
        }
    }
}

impl std::error::Error for LockError {}

impl QueueLock {
    /// Take the exclusive lock at `path` without waiting for it.
    ///
    /// An absent lock file is created empty, as the Python writers create it.
    /// A final symbolic link is refused rather than followed, and so is anything
    /// that is not a regular file.
    pub fn acquire(path: &Path) -> Result<Self, LockError> {
        Self::lock_opened(path, open_lock_file(path)?)
    }

    /// Lock `file`, already opened at `path`. After the lock is taken, `path`
    /// must still name that file: a lock on a file unlinked or replaced in the
    /// meantime excludes nobody.
    fn lock_opened(path: &Path, file: File) -> Result<Self, LockError> {
        let opened = file
            .metadata()
            .map_err(|error| io_failure("fstat", path, error))?;
        if !opened.file_type().is_file() {
            return Err(LockError::NotRegular {
                path: path.to_path_buf(),
            });
        }
        loop {
            // SAFETY: flock(2) takes a descriptor and flags and has no memory
            // preconditions. `file` owns the descriptor, so it stays open for
            // the whole call.
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
                break;
            }
            let error = io::Error::last_os_error();
            match error.raw_os_error() {
                Some(libc::EINTR) => continue,
                Some(libc::EWOULDBLOCK) => {
                    return Err(LockError::Held {
                        path: path.to_path_buf(),
                    })
                }
                _ => return Err(io_failure("flock", path, error)),
            }
        }
        match std::fs::symlink_metadata(path) {
            Ok(now) if now.dev() == opened.dev() && now.ino() == opened.ino() => {}
            Ok(_) => {
                return Err(LockError::Replaced {
                    path: path.to_path_buf(),
                })
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(LockError::Replaced {
                    path: path.to_path_buf(),
                })
            }
            Err(error) => return Err(io_failure("lstat", path, error)),
        }
        Ok(Self {
            path: path.to_path_buf(),
            _file: file,
        })
    }

    /// The lock file this lock is held on.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Open the lock file without following a final symlink, creating it when absent.
fn open_lock_file(path: &Path) -> Result<File, LockError> {
    // O_NONBLOCK keeps a FIFO at the lock path from blocking the open; it is then
    // refused as not a regular file.
    let flags = libc::O_NOFOLLOW | libc::O_NONBLOCK;
    match OpenOptions::new().read(true).custom_flags(flags).open(path) {
        Ok(file) => Ok(file),
        // The same open mode as Python's open(path, "a+"), which never truncates.
        Err(error) if error.kind() == io::ErrorKind::NotFound => OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .custom_flags(flags)
            .open(path)
            .map_err(|error| io_failure("create", path, error)),
        Err(error) => Err(io_failure("open", path, error)),
    }
}

fn io_failure(operation: &'static str, path: &Path, error: io::Error) -> LockError {
    LockError::Io {
        operation,
        path: path.to_path_buf(),
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A disposable directory under a canonical temp root, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().canonicalize().expect("temp dir");
            let dir = root.join(format!(
                "frankenctl-downloads-{tag}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&dir).expect("create scratch dir");
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_manifest_lock_sits_beside_it_with_the_extension_replaced() {
        assert_eq!(
            lock_path_for(Path::new("/q/download-queue-slot-uncensored-27b.json")),
            Path::new("/q/download-queue-slot-uncensored-27b.lock")
        );
        assert_eq!(
            lock_path_for(Path::new("/v/glm53flash-local/manifest.tsv")),
            Path::new("/v/glm53flash-local/manifest.lock")
        );
        assert_eq!(lock_path_for(Path::new("/q/queue")), Path::new("/q/queue.lock"));
    }

    #[test]
    fn an_absent_lock_is_created_empty_and_excludes_a_second_holder_until_dropped() {
        let scratch = Scratch::new("exclusive");
        let path = scratch.0.join("download-queue.lock");
        let first = QueueLock::acquire(&path).expect("first lock");
        assert_eq!(first.path(), path);
        let metadata = std::fs::symlink_metadata(&path).unwrap();
        assert!(metadata.file_type().is_file());
        assert_eq!(metadata.len(), 0);
        // flock(2) locks belong to an open file description, so a second open in
        // this same process conflicts exactly as another process would.
        match QueueLock::acquire(&path) {
            Err(error @ LockError::Held { .. }) => {
                assert_eq!(error.exit_code(), crate::verify::EXIT_LOCKED);
                assert_eq!(error.path(), path);
                assert!(error.os_error().is_none());
            }
            other => panic!("expected the lock to be held, got {other:?}"),
        }
        drop(first);
        QueueLock::acquire(&path).expect("the lock is free once dropped");
    }

    #[test]
    fn an_existing_lock_file_keeps_its_bytes() {
        let scratch = Scratch::new("keep");
        let path = scratch.0.join("download-queue.lock");
        std::fs::write(&path, b"left by a writer\n").unwrap();
        drop(QueueLock::acquire(&path).expect("lock"));
        assert_eq!(std::fs::read(&path).unwrap(), b"left by a writer\n");
    }

    #[test]
    fn a_symlink_or_a_directory_at_the_lock_path_fails_closed() {
        let scratch = Scratch::new("unsafe");
        let real = scratch.0.join("real.lock");
        std::fs::write(&real, b"").unwrap();
        let link = scratch.0.join("link.lock");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        match QueueLock::acquire(&link) {
            Err(
                error @ LockError::Io {
                    operation: "open", ..
                },
            ) => {
                assert_eq!(error.exit_code(), crate::verify::EXIT_IO);
                assert!(error.to_string().contains("ELOOP"), "{error}");
                assert_eq!(
                    error.os_error().and_then(io::Error::raw_os_error),
                    Some(libc::ELOOP)
                );
            }
            other => panic!("expected ELOOP, got {other:?}"),
        }
        let directory = scratch.0.join("dir.lock");
        std::fs::create_dir(&directory).unwrap();
        assert!(matches!(
            QueueLock::acquire(&directory),
            Err(LockError::NotRegular { .. })
        ));
        // Neither refusal created or changed anything at the real lock file.
        assert_eq!(std::fs::metadata(&real).unwrap().len(), 0);
    }

    #[test]
    fn a_lock_file_that_cannot_be_created_keeps_its_os_error() {
        let scratch = Scratch::new("create");
        let path = scratch.0.join("no-such-dir/download-queue.lock");
        match QueueLock::acquire(&path) {
            Err(LockError::Io {
                operation: "create",
                error,
                ..
            }) => assert_eq!(error.raw_os_error(), Some(libc::ENOENT)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_lock_file_replaced_before_it_was_locked_is_refused() {
        let scratch = Scratch::new("replaced");
        let path = scratch.0.join("download-queue.lock");
        std::fs::write(&path, b"").unwrap();
        let opened = open_lock_file(&path).unwrap();
        // What a writer that deletes and recreates the lock file would leave.
        let fresh = scratch.0.join("fresh.lock");
        std::fs::write(&fresh, b"").unwrap();
        std::fs::rename(&fresh, &path).unwrap();
        match QueueLock::lock_opened(&path, opened) {
            Err(error @ LockError::Replaced { .. }) => {
                assert_eq!(error.exit_code(), crate::verify::EXIT_LOCKED)
            }
            other => panic!("{other:?}"),
        }

        let opened = open_lock_file(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(
            QueueLock::lock_opened(&path, opened),
            Err(LockError::Replaced { .. })
        ));
    }
}
