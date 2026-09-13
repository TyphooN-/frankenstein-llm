//! Typed errors for the control plane.
//!
//! Errors are classified so that callers can decide fail-closed behavior
//! uniformly rather than string-matching on messages. `Repo` errors are
//! mandatory-failure conditions for `frankenctl doctor`; `Config` errors are
//! malformed-operator-input conditions for the catalog/config loaders.

use std::path::PathBuf;

use thiserror::Error;

/// A repository/runtime check that failed. `doctor` treats every one of these
/// as a mandatory non-zero exit.
#[derive(Debug, Error)]
pub enum CheckError {
    #[error("required path is missing or unreadable: {0}")]
    MissingPath(PathBuf),
    #[error("pinned llama.cpp runtime binary is missing: {0}")]
    MissingLlamaBinary(PathBuf),
    #[error("upstream pin is not sane: {0}")]
    PinSane(String),
    #[error("ROCm sysfs visibility failed: {0}")]
    Rocm(String),
    #[error("GPU inventory mismatch: expected {expected}, found {found}")]
    GpuInventory { expected: usize, found: usize },
    #[error("required runtime path is absent: {0}")]
    RuntimePath(PathBuf),
}

/// A catalog/config file that was malformed.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("cannot read {0}: {1}")]
    Io(PathBuf, String),
    #[error("{0} is not valid JSON: {1}")]
    Json(PathBuf, String),
    #[error("invalid catalog entry: {0}")]
    CatalogEntry(String),
    #[error("serving config must contain only host=127.0.0.1 and port")]
    ServingHostPort,
    #[error("port must be an integer from 1024 to 65535")]
    ServingPort,
}

impl ConfigError {
    pub fn io(path: impl Into<PathBuf>, source: &std::io::Error) -> Self {
        Self::Io(path.into(), source.to_string())
    }
    pub fn json(path: impl Into<PathBuf>, source: &serde_json::Error) -> Self {
        Self::Json(path.into(), source.to_string())
    }
}

/// Convenience result alias used across the crate.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
