//! The `frankenctl` command-line interface.
//!
//! Implements the first-real-slice of the command tree from the migration
//! brief: `doctor`, `config show`/`config check`, `model list`, and
//! `serve <alias>` (plan-only by default; `--execute` is reserved for later
//! once parity against the Python serving path is demonstrated in the
//! integration fixtures). The tree is deliberately small: no empty
//! subcommands, every subcommand does real work.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

/// Frankenstein LLM control plane.
#[derive(Debug, Parser)]
#[command(name = "frankenctl", version, about)]
pub struct Cli {
    /// The repository root. Defaults to the compiled-in parent of `rust/`.
    #[arg(long, global = true)]
    pub root: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Run repository and runtime health checks (fail-closed).
    Doctor {
        /// Expected number of GPUs; when given, a mismatch is a failure.
        #[arg(long)]
        expected_gpus: Option<usize>,
    },
    /// Inspect and validate configuration.
    #[command(subcommand)]
    Config(ConfigCmd),
    /// Inspect the model catalog and presets.
    #[command(subcommand)]
    Model(ModelCmd),
    /// Build or run a `llama-server` launch plan for a preset.
    Serve {
        /// The preset alias from `llama-models.ini`.
        alias: Option<String>,
        /// List all configured presets (default when no alias is given).
        #[arg(long)]
        list: bool,
        /// Execute the launch instead of printing the plan (later phase).
        #[arg(long)]
        execute: bool,
        /// Override the port for this invocation.
        #[arg(long)]
        port: Option<u16>,
    },
}

#[derive(Debug, Subcommand)]
pub enum ConfigCmd {
    /// Print the parsed serving config.
    Show,
    /// Validate the serving and catalog configuration files.
    Check,
}

#[derive(Debug, Subcommand)]
pub enum ModelCmd {
    /// List configured presets and their catalog entries.
    List,
}

/// Resolve the effective repository root, preferring an explicit `--root`.
pub fn resolve_root(explicit: Option<&Path>) -> PathBuf {
    explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(crate::doctor::repo_root)
        .canonicalize()
        .unwrap_or_else(|_| crate::doctor::repo_root())
}
