//! The `frankenctl` command-line interface.
//!
//! Implements the first-real-slice of the command tree from the migration
//! brief: `doctor`, `config show`/`config check`, `model list`, and
//! `serve <alias>` (plan-only by default; `--execute` is reserved for later
//! once parity against the Python serving path is demonstrated in the
//! integration fixtures), plus `verify plan`/`verify run`, the read-only
//! artifact verifier over the artifact inventory or one download-queue
//! manifest. The tree is deliberately small: no empty subcommands, every
//! subcommand does real work.

use std::path::{Path, PathBuf};

use clap::{ArgGroup, Args, Parser, Subcommand};

use crate::inventory::DEFAULT_INVENTORY;

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
    /// Check local artifacts read-only, against the artifact inventory or one
    /// download-queue manifest.
    ///
    /// Exit status: 0 every selected file matched; 1 a file is missing, the
    /// wrong size or digest, unsafe to open or changed during the run, or a
    /// model file a preset or sidecar loads is unaccounted for; 2 bad usage;
    /// 65 malformed inventory or manifest; 66 inventory, manifest or root
    /// absent; 70 internal failure; 73 the report could not be published; 74
    /// an I/O error prevented a check or a writer lock could not be taken; 75
    /// another process holds a writer lock, so nothing was examined; 128+N
    /// interrupted by signal N.
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
    /// Print what `run` would check and which writer locks it would take.
    /// Reads the inventory, its manifests and the preset and sidecar files,
    /// and no artifact path; takes no lock.
    Plan(ScopeArgs),
    /// Take every writer lock, then check exact size, and SHA-256 where one is
    /// recorded. Never downloads, repairs, deletes, renames or promotes a file.
    Run(RunArgs),
}

/// What `verify plan` and `verify run` cover.
#[derive(Debug, Args)]
#[command(group(ArgGroup::new("scope").required(true).args(["manifest", "inventory"])))]
pub struct ScopeArgs {
    /// One `hermes-hf-artifact-queue/1` download-queue manifest. Its writer
    /// lock is the same path with the extension `lock`.
    #[arg(long, value_name = "PATH")]
    pub manifest: Option<PathBuf>,
    /// The artifact inventory, `config/artifact-inventory.json` when given no
    /// path. A relative path is resolved against the repository root.
    #[arg(
        long,
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = DEFAULT_INVENTORY
    )]
    pub inventory: Option<PathBuf>,
    /// Only this inventory source; repeat for more. Default: every source.
    #[arg(long = "source", value_name = "ID", requires = "inventory")]
    pub sources: Vec<String>,
    /// Only this artifact key; repeat for more. Default: every artifact.
    #[arg(long = "artifact", value_name = "KEY")]
    pub artifacts: Vec<String>,
}

/// Arguments of `verify run`.
#[derive(Debug, Args)]
pub struct RunArgs {
    #[command(flatten)]
    pub scope: ScopeArgs,
    /// Also publish the report durably at PATH: as `running` once the locks
    /// are held and before any artifact is examined, then in its final state.
    #[arg(long, value_name = "PATH")]
    pub report: Option<PathBuf>,
}

/// Resolve the effective repository root, preferring an explicit `--root`.
pub fn resolve_root(explicit: Option<&Path>) -> PathBuf {
    explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(crate::doctor::repo_root)
        .canonicalize()
        .unwrap_or_else(|_| crate::doctor::repo_root())
}
