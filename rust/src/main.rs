//! `frankenctl` binary entry point.
//!
//! Thin dispatcher over the library modules. All real logic lives in
//! `lib.rs` and its modules so it can be unit-tested and later reused by a
//! `frankend` daemon.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::Parser;

use frankenctl::cli::{self, Cli, Commands};
use frankenctl::config::{self, Serving};
use frankenctl::doctor::Doctor;
use frankenctl::inventory::{self, Inventory};
use frankenctl::manifest::{ArtifactFile, ManifestError};
use frankenctl::verification::{self, ConsumerCoverage, Coverage, RunReport, RunState};
use frankenctl::verify::{self, Outcome};

fn main() -> ExitCode {
    let cli = Cli::parse();
    // `verify` has its own exit statuses and root handling, so it does not go
    // through the string reports.
    if let Commands::Verify(command) = &cli.command {
        return verify_command(cli.root.as_deref(), command);
    }
    let root = cli::resolve_root(cli.root.as_deref());

    match run(&cli.command, &root) {
        Ok(report) => {
            if !report.is_empty() {
                print!("{report}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("frankenctl: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Execute a parsed command. Returns the text to print and an `Option`-style
/// error via `Result`. The `report` string is empty when a command only needs
/// to signal success by exit code.
fn run(cmd: &Commands, root: &std::path::Path) -> std::result::Result<String, String> {
    match cmd {
        Commands::Doctor { expected_gpus } => {
            let d = Doctor::run(root.to_path_buf(), *expected_gpus);
            let report = d.render();
            if !d.ok() {
                eprint!("{report}");
                return Err("doctor failed".to_string());
            }
            Ok(report)
        }
        Commands::Config(sub) => match sub {
            cli::ConfigCmd::Show => {
                let s = load_serving(root)?;
                Ok(format!("host={} port={}\n", s.host, s.port))
            }
            cli::ConfigCmd::Check => {
                let serving = load_serving(root)?;
                let catalog = config::load_catalog(&root.join("config/model-catalog.json"))
                    .map_err(|e| e.to_string())?;
                Ok(format!(
                    "serving host={} port={} catalog_entries={} OK\n",
                    serving.host,
                    serving.port,
                    catalog.len()
                ))
            }
        },
        Commands::Model(cli::ModelCmd::List) => model_list(root),
        Commands::Serve {
            alias,
            list,
            execute,
            port,
        } => {
            if *list || alias.is_none() {
                return model_list(root);
            }
            serve_plan(root, alias.as_ref().unwrap(), *port, *execute)
        }
        Commands::Verify(_) => unreachable!("verify is dispatched in main"),
    }
}

/// `frankenctl verify plan|run`: the plan or report as JSON on stdout, one line
/// per problem on stderr, and the status from `verify::EXIT_*`.
fn verify_command(root: Option<&Path>, command: &cli::VerifyCmd) -> ExitCode {
    let (scope, report_path, execute) = match command {
        cli::VerifyCmd::Plan(scope) => (scope, None, false),
        cli::VerifyCmd::Run(args) => (&args.scope, args.report.as_deref(), true),
    };
    let (inventory, consumers) = match load_scope(root, scope) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("frankenctl: {error}");
            return ExitCode::from(verify::manifest_exit_code(&error));
        }
    };
    let selection = match verification::select(&inventory, &scope.sources, &scope.artifacts) {
        Ok(selection) => selection,
        Err(error) => {
            eprintln!("frankenctl: {error}");
            return ExitCode::from(verify::EXIT_USAGE);
        }
    };
    if !execute {
        let plan = verification::plan(&inventory, &selection, consumers);
        if let Some(coverage) = &plan.consumers {
            describe_consumers(coverage);
        }
        let code = plan.exit_code();
        return emit(&plan, code);
    }

    if let Err(error) = verification::install_cancellation_handlers() {
        eprintln!("frankenctl: cannot install the SIGINT, SIGTERM and SIGHUP handlers: {error}");
        return ExitCode::from(verify::EXIT_SOFTWARE);
    }
    let pause = test_pause_marker();
    let mut paused = false;
    let mut after_chunk = |_: &ArtifactFile, _: u64| {
        if let (Some(marker), false) = (pause.as_deref(), paused) {
            paused = true;
            hold_for_test(marker);
        }
    };
    let report = verification::run(
        &inventory,
        &selection,
        consumers,
        report_path,
        verification::Hooks {
            after_chunk: &mut after_chunk,
            cancelled: &verification::cancellation,
        },
    );
    describe_run(&report);
    let code = report.exit_code.unwrap_or(verify::EXIT_SOFTWARE);
    emit(&report, code)
}

/// The inventory named by `--inventory`, with every loaded model file accounted
/// for; or the one manifest named by `--manifest`, which has no such accounting.
fn load_scope(
    root: Option<&Path>,
    scope: &cli::ScopeArgs,
) -> Result<(Inventory, Option<ConsumerCoverage>), ManifestError> {
    if let Some(manifest) = &scope.manifest {
        return Ok((Inventory::single_manifest(manifest)?, None));
    }
    let root = verify_root(root)?;
    let named = scope
        .inventory
        .as_deref()
        .unwrap_or(Path::new(inventory::DEFAULT_INVENTORY));
    // Joining an absolute path replaces the root.
    let inventory = Inventory::load(&root, &root.join(named))?;
    let consumers = inventory::consumers(&root)?;
    let coverage = verification::consumer_coverage(&inventory, consumers);
    Ok((inventory, Some(coverage)))
}

/// The repository root for `verify`. Unlike the other commands, an explicit
/// `--root` that does not resolve is an error, never a silent fallback to the
/// compiled-in checkout: that would verify a different tree than was named.
fn verify_root(explicit: Option<&Path>) -> Result<PathBuf, ManifestError> {
    let named = explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(frankenctl::doctor::repo_root);
    named
        .canonicalize()
        .map_err(|error| ManifestError::Read { path: named, error })
}

/// The outcome of a run on stderr: the lock or publication failure, one line
/// per file that did not match, loaded-model accounting, and a count line.
fn describe_run(report: &RunReport) {
    if let Some(refused) = &report.writer_locks.refused {
        eprintln!("frankenctl: {}; no artifact was examined", refused.error);
    }
    if let Some(failure) = &report.report_failure {
        eprintln!(
            "frankenctl: cannot {} {}: {} [{}]",
            failure.operation,
            failure.path,
            failure.error,
            failure.errno_name.unwrap_or("no errno name")
        );
    }
    for result in &report.results {
        if let Some(problem) = problem(&result.outcome) {
            eprintln!(
                "frankenctl: {} {} {}: {problem}",
                result.source, result.key, result.destination
            );
        }
    }
    if let Some(coverage) = &report.consumers {
        describe_consumers(coverage);
    }
    let s = &report.summary;
    eprintln!(
        "frankenctl: {}: {} files: {} sha256 verified, {} size only (no published hash), \
         {} missing, {} size mismatch, {} sha256 mismatch, {} unreadable, {} unsafe path, \
         {} changed during the run, {} not checked; exit {}",
        state_name(report.state),
        s.files,
        s.verified_sha256,
        s.size_only_no_published_hash,
        s.missing,
        s.size_mismatch,
        s.sha256_mismatch,
        s.unreadable,
        s.unsafe_path,
        s.changed_during_read,
        s.not_checked,
        report.exit_code.unwrap_or(verify::EXIT_SOFTWARE)
    );
}

fn describe_consumers(coverage: &ConsumerCoverage) {
    for row in &coverage.rows {
        if row.coverage == Coverage::Unaccounted {
            eprintln!(
                "frankenctl: unaccounted loaded model file {} ({})",
                row.path,
                row.consumed_by.join(", ")
            );
        }
    }
    for path in &coverage.stale_declarations {
        eprintln!("frankenctl: declared uncovered, but nothing loads it: {path}");
    }
    let s = &coverage.summary;
    eprintln!(
        "frankenctl: {} loaded model files: {} in inventory sources, \
         {} declared unverifiable and not read, {} unaccounted",
        s.paths, s.in_inventory_sources, s.declared_unverifiable, s.unaccounted
    );
}

fn state_name(state: RunState) -> &'static str {
    match state {
        RunState::Running => "running",
        RunState::Complete => "complete",
        RunState::Interrupted => "interrupted",
        RunState::RefusedLocked => "refused-locked",
        RunState::LockFailed => "lock-failed",
        RunState::Aborted => "aborted",
    }
}

/// An operator-facing line for an outcome that is not a match.
fn problem(outcome: &Outcome) -> Option<String> {
    Some(match outcome {
        Outcome::VerifiedSha256 { .. } | Outcome::SizeOnlyNoPublishedHash { .. } => return None,
        Outcome::Missing => "missing".to_string(),
        Outcome::SizeMismatch { actual_size, .. } => {
            format!("size mismatch: {actual_size} bytes on disk")
        }
        Outcome::Sha256Mismatch { actual_sha256, .. } => {
            format!("sha256 mismatch: {actual_sha256} on disk")
        }
        Outcome::Unreadable { failure } => format!(
            "unreadable: {} of {}{} failed: {} [{}]",
            failure.operation,
            failure.path,
            failure
                .offset
                .map(|offset| format!(" at byte {offset}"))
                .unwrap_or_default(),
            failure.error,
            failure.errno_name.unwrap_or("no errno name")
        ),
        Outcome::UnsafePath { reason } => format!("unsafe path: {reason}"),
        Outcome::ChangedDuringRead { reason } => format!("changed during the run: {reason}"),
        Outcome::NotChecked { reason } => format!("not checked: {reason}"),
    })
}

/// Debug builds only: the integration tests stop a run after its first hashed
/// chunk, with every writer lock held, so they can act on it from outside. A
/// release build ignores the variable.
#[cfg(debug_assertions)]
fn test_pause_marker() -> Option<PathBuf> {
    std::env::var_os("FRANKENCTL_TEST_PAUSE_AFTER_FIRST_CHUNK").map(PathBuf::from)
}

#[cfg(not(debug_assertions))]
fn test_pause_marker() -> Option<PathBuf> {
    None
}

/// Write `marker`, then wait for a cancelling signal, for at most 30 seconds.
fn hold_for_test(marker: &Path) {
    let _ = std::fs::write(marker, b"paused\n");
    let deadline = Instant::now() + Duration::from_secs(30);
    while verification::cancellation().is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn emit<T: serde::Serialize>(value: &T, code: u8) -> ExitCode {
    match serde_json::to_string_pretty(value) {
        Ok(text) => {
            println!("{text}");
            ExitCode::from(code)
        }
        Err(error) => {
            eprintln!("frankenctl: cannot encode the report as JSON: {error}");
            ExitCode::from(verify::EXIT_SOFTWARE)
        }
    }
}

fn load_serving(root: &std::path::Path) -> std::result::Result<Serving, String> {
    config::load_serving(&root.join("config/serving.json")).map_err(|e| e.to_string())
}

fn model_list(root: &std::path::Path) -> std::result::Result<String, String> {
    let presets =
        config::load_presets(&root.join("llama-models.ini")).map_err(|e| e.to_string())?;
    let catalog =
        config::load_catalog(&root.join("config/model-catalog.json")).map_err(|e| e.to_string())?;

    let mut rows = Vec::new();
    for (alias, preset) in &presets {
        let mut names = Vec::new();
        if let Some(model) = preset.get("model") {
            names.push(
                std::path::Path::new(model)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(model)
                    .to_string(),
            );
        }
        if let Some(mmproj) = preset.get("mmproj") {
            names.push(
                std::path::Path::new(mmproj)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(mmproj)
                    .to_string(),
            );
        }
        let identity = if names.is_empty() {
            "not a configured preset".to_string()
        } else {
            names.join(" + ")
        };
        let purpose = catalog
            .get(alias)
            .map(|e| e.purpose.as_str())
            .unwrap_or("no catalog entry");
        rows.push((format!("{identity} ({purpose})"), alias.clone()));
    }
    rows.sort_by(|a, b| a.1.cmp(&b.1));
    let width = rows.iter().map(|(body, _)| body.len()).max().unwrap_or(0);
    let mut out = String::new();
    for (body, alias) in rows {
        out.push_str(&format!("{body:<width$}  [compatibility alias: {alias}]\n"));
    }
    Ok(out)
}

fn serve_plan(
    root: &std::path::Path,
    alias: &str,
    port: Option<u16>,
    execute: bool,
) -> std::result::Result<String, String> {
    let presets =
        config::load_presets(&root.join("llama-models.ini")).map_err(|e| e.to_string())?;
    let preset = presets
        .get(alias)
        .ok_or_else(|| format!("unknown preset alias: {alias}"))?;
    let mut serving = load_serving(root)?;
    if let Some(p) = port {
        serving.port = p;
    }

    let llama_bin = root.join("upstream/llama.cpp/build/bin/llama-server");
    let cmd = config::command(&llama_bin, alias, preset, &serving).map_err(|e| e.to_string())?;

    let model = preset.get("model").unwrap_or("<no-model>");
    let model_path = root.join(model);
    if !model_path.is_file() {
        eprintln!("frankenctl: model file not found: {}", model_path.display());
        // Not fatal for planning, but reported.
    }

    let mut out = String::new();
    out.push_str(&format!(
        "# plan for preset {alias}\n# host={} port={}\n",
        serving.host, serving.port
    ));
    out.push_str(&format!("# binary: {}\n", llama_bin.display()));
    out.push_str(&format!("{}\n", cmd.join(" ")));
    if execute {
        out.push_str("# execution reserved for the parity-verified phase; plan-only now\n");
    }
    Ok(out)
}
