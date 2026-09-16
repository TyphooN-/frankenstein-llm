//! `frankenctl` library surface.
//!
//! Splitting the logic into a library (and a thin binary in `main.rs`) lets
//! the integration tests in `tests/` exercise the real config loaders, the
//! `llama-server` launch planner, the doctor checks and the artifact verifier
//! directly, and lets a future `frankend` daemon reuse the same code.

pub mod cli;
pub mod config;
pub mod doctor;
pub mod downloads;
pub mod error;
pub mod gpu;
pub mod inventory;
pub mod manifest;
pub mod report;
pub mod verification;
pub mod verify;
