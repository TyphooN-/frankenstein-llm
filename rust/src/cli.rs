//! The `frankenctl` command-line interface.
//!
//! Implements the first-real-slice of the command tree from the migration
//! brief: `doctor`, `config show`/`config check`, `model list`, and
//! `serve <alias>` (plan-only by default; `--execute` is reserved for later
//! once parity against the Python serving path is demonstrated in the
//! integration fixtures), plus `verify plan`/`verify run`, the read-only
//! artifact verifier over one explicit download-queue manifest. The tree is
//! deliberately small: no empty subcommands, every subcommand does real work.

use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};

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
    /// Check local artifacts against one download-queue manifest, read-only.
    ///
    /// Exit status: 0 every selected file matched; 1 a file is missing, the
    /// wrong size or digest, unsafe to open, or changed while read; 2 bad
    /// usage; 65 malformed manifest; 66 manifest absent; 74 an I/O error
    /// prevented a check. Takes no download-queue lock yet: do not run it
    /// while a downloader may write the same destinations.
    #[command(subcommand)]
    Verify(VerifyCmd),
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

#[derive(Debug, Subcommand)]
pub enum VerifyCmd {
    /// Print what `run` would check, from the manifest alone.
    Plan(VerifyArgs),
    /// Check exact size, and SHA-256 where the queue records one.
    Run(VerifyArgs),
}

/// Arguments shared by `verify plan` and `verify run`.
#[derive(Debug, Args)]
pub struct VerifyArgs {
    /// A `hermes-hf-artifact-queue/1` download-queue manifest.
    #[arg(long, value_name = "PATH")]
    pub manifest: PathBuf,
    /// Only this artifact key; repeat for more. Default: every artifact.
    #[arg(long = "artifact", value_name = "KEY")]
    pub artifacts: Vec<String>,
}

/// Resolve the effective repository root, preferring an explicit `--root`.
pub fn resolve_root(explicit: Option<&Path>) -> PathBuf {
    explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(crate::doctor::repo_root)
        .canonicalize()
        .unwrap_or_else(|_| crate::doctor::repo_root())
}
