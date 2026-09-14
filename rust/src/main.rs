//! `frankenctl` binary entry point.
//!
//! Thin dispatcher over the library modules. All real logic lives in
//! `lib.rs` and its modules so it can be unit-tested and later reused by a
//! `frankend` daemon.

use std::process::ExitCode;

use clap::Parser;

use frankenctl::cli::{self, Cli, Commands};
use frankenctl::config::{self, Serving};
use frankenctl::doctor::Doctor;
use frankenctl::manifest::QueueManifest;
use frankenctl::verify::{self, Outcome};

fn main() -> ExitCode {
    let cli = Cli::parse();
    // `verify` has its own exit statuses and takes an explicit manifest, so it
    // neither resolves a repository root nor goes through the string reports.
    if let Commands::Verify(command) = &cli.command {
        return verify_command(command);
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

/// `frankenctl verify plan|run`: the plan or report as JSON on stdout, one
/// line per problem on stderr, and the status from `verify::EXIT_*`.
fn verify_command(command: &cli::VerifyCmd) -> ExitCode {
    let (args, execute) = match command {
        cli::VerifyCmd::Plan(args) => (args, false),
        cli::VerifyCmd::Run(args) => (args, true),
    };
    let manifest = match QueueManifest::load(&args.manifest) {
        Ok(manifest) => manifest,
        Err(error) => {
            eprintln!("frankenctl: {error}");
            return ExitCode::from(verify::manifest_exit_code(&error));
        }
    };
    let selected = match manifest.select(&args.artifacts) {
        Ok(selected) => selected,
        Err(error) => {
            eprintln!("frankenctl: {error}");
            return ExitCode::from(verify::EXIT_USAGE);
        }
    };
    if !execute {
        return emit(&verify::plan(&manifest, &selected), verify::EXIT_OK);
    }
    let report = verify::run(&manifest, &selected);
    for result in &report.results {
        if let Some(problem) = problem(&result.outcome) {
            eprintln!(
                "frankenctl: {} {}: {problem}",
                result.key, result.destination
            );
        }
    }
    let s = &report.summary;
    eprintln!(
        "frankenctl: {} files: {} sha256 verified, {} size only (no published hash), \
         {} missing, {} size mismatch, {} sha256 mismatch, {} unreadable, {} unsafe path, \
         {} changed during read; exit {}",
        s.files,
        s.verified_sha256,
        s.size_only_no_published_hash,
        s.missing,
        s.size_mismatch,
        s.sha256_mismatch,
        s.unreadable,
        s.unsafe_path,
        s.changed_during_read,
        report.exit_code
    );
    emit(&report, report.exit_code)
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
        Outcome::ChangedDuringRead { reason } => format!("changed during read: {reason}"),
    })
}

fn emit<T: serde::Serialize>(value: &T, code: u8) -> ExitCode {
    match serde_json::to_string_pretty(value) {
        Ok(text) => {
            println!("{text}");
            ExitCode::from(code)
        }
        Err(error) => {
            eprintln!("frankenctl: cannot encode the report as JSON: {error}");
            ExitCode::from(70)
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
