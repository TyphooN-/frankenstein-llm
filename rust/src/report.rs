//! Durable publication of a verification report.
//!
//! A report a later reader acts on is written the way the download queue writes
//! its state: a new file beside the target, fsynced, renamed over the target,
//! and the directory fsynced. A reader sees the previous report or the new one,
//! never a torn file, and a published report survives a power loss. The
//! temporary name is created exclusively and without following a symbolic link,
//! and the rename replaces a link at the target instead of writing through it.

use std::ffi::OsStr;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

/// Why a report was not published, with the step and path that failed.
#[derive(Debug)]
pub struct ReportError {
    pub operation: &'static str,
    pub path: PathBuf,
    pub error: io::Error,
}

impl fmt::Display for ReportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot {} {}: {} [{}]",
            self.operation,
            self.path.display(),
            self.error,
            self.error
                .raw_os_error()
                .and_then(crate::verify::errno_name)
                .unwrap_or("no errno name")
        )
    }
}

impl std::error::Error for ReportError {}

fn failure(operation: &'static str, path: &Path, error: io::Error) -> ReportError {
    ReportError {
        operation,
        path: path.to_path_buf(),
        error,
    }
}

/// Replace `path` with `bytes` durably, creating missing parent directories.
///
/// On failure the previous file at `path`, if any, is left as it was, and the
/// temporary file is removed.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), ReportError> {
    let Some(name) = path.file_name() else {
        return Err(failure(
            "name",
            path,
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "the report path has no file name",
            ),
        ));
    };
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    };
    std::fs::create_dir_all(&parent)
        .map_err(|error| failure("create the directory", &parent, error))?;
    let (temporary, mut file) = create_temporary(&parent, name)?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = std::fs::remove_file(&temporary);
        return Err(failure("write and fsync", &temporary, error));
    }
    drop(file);
    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(failure("rename into place", path, error));
    }
    File::open(&parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| failure("fsync the directory", &parent, error))
}

/// Create `.<name>.tmp.<pid>.<n>` beside the target, exclusively.
fn create_temporary(parent: &Path, name: &OsStr) -> Result<(PathBuf, File), ReportError> {
    let mut attempt = 0u32;
    loop {
        let temporary = parent.join(format!(
            ".{}.tmp.{}.{attempt}",
            name.to_string_lossy(),
            std::process::id()
        ));
        // create_new is O_CREAT|O_EXCL, which already refuses an existing
        // symlink; O_NOFOLLOW states the intent.
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o644)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&temporary)
        {
            Ok(file) => return Ok((temporary, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists && attempt < 16 => {
                attempt += 1
            }
            Err(error) => return Err(failure("create", &temporary, error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().canonicalize().expect("temp dir");
            let dir = root.join(format!(
                "frankenctl-report-{tag}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&dir).expect("create scratch dir");
            Self(dir)
        }

        fn names(&self) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(&self.0)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_report_replaces_the_previous_one_and_leaves_no_temporary_file() {
        let scratch = Scratch::new("replace");
        let path = scratch.0.join("latest.json");
        write_atomic(&path, b"first\n").unwrap();
        write_atomic(&path, b"second\n").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second\n");
        assert_eq!(scratch.names(), ["latest.json"]);
    }

    #[test]
    fn missing_parent_directories_are_created() {
        let scratch = Scratch::new("parents");
        let path = scratch.0.join("proofs/verification/latest.json");
        write_atomic(&path, b"{}\n").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"{}\n");
    }

    #[test]
    fn a_symlink_at_the_target_is_replaced_not_written_through() {
        let scratch = Scratch::new("link");
        let victim = scratch.0.join("victim.txt");
        std::fs::write(&victim, b"untouched").unwrap();
        let path = scratch.0.join("latest.json");
        std::os::unix::fs::symlink(&victim, &path).unwrap();
        write_atomic(&path, b"report").unwrap();
        assert_eq!(std::fs::read(&victim).unwrap(), b"untouched");
        assert!(std::fs::symlink_metadata(&path).unwrap().is_file());
        assert_eq!(std::fs::read(&path).unwrap(), b"report");
    }

    #[test]
    fn a_planted_temporary_name_is_neither_followed_nor_reused() {
        let scratch = Scratch::new("planted");
        let victim = scratch.0.join("victim.txt");
        std::fs::write(&victim, b"untouched").unwrap();
        let planted = scratch
            .0
            .join(format!(".latest.json.tmp.{}.0", std::process::id()));
        std::os::unix::fs::symlink(&victim, &planted).unwrap();
        let path = scratch.0.join("latest.json");
        write_atomic(&path, b"report").unwrap();
        assert_eq!(std::fs::read(&victim).unwrap(), b"untouched");
        assert_eq!(std::fs::read(&path).unwrap(), b"report");
        assert!(std::fs::symlink_metadata(&planted)
            .unwrap()
            .file_type()
            .is_symlink());
    }

    #[test]
    fn a_failed_rename_keeps_the_previous_target_and_names_the_step() {
        let scratch = Scratch::new("fail");
        let path = scratch.0.join("latest.json");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("occupied"), b"x").unwrap();
        let error = write_atomic(&path, b"report").unwrap_err();
        assert_eq!(error.operation, "rename into place");
        assert_eq!(error.path, path);
        assert!(error.error.raw_os_error().is_some());
        assert_eq!(scratch.names(), ["latest.json"]);
        assert!(path.join("occupied").exists());
    }

    #[test]
    fn a_parent_that_is_a_file_fails_before_anything_is_written() {
        let scratch = Scratch::new("parent-file");
        let blocker = scratch.0.join("not-a-directory");
        std::fs::write(&blocker, b"").unwrap();
        let error = write_atomic(&blocker.join("latest.json"), b"report").unwrap_err();
        assert_eq!(error.operation, "create the directory");
        assert_eq!(scratch.names(), ["not-a-directory"]);
    }
}
