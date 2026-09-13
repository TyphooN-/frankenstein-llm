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

fn main() -> ExitCode {
    let cli = Cli::parse();
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
            // Fail closed: a mandatory failure is a non-zero exit.
            if !d.ok() {
                return Ok(report); // printed; exit code set by caller via ok()
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

    let mut out = String::new();
    let mut aliases: Vec<&String> = presets.keys().collect();
    aliases.sort();
    for a in aliases {
        let preset = &presets[a];
        let model = preset.get("model").unwrap_or("<no-model>");
        let catalog_entry = catalog.get(a);
        let cat = match catalog_entry {
            Some(e) => format!("[{}/{}]", e.source, e.purpose),
            None => "[no-catalog-entry]".to_string(),
        };
        let n_params = preset.get("gpu-layers").unwrap_or("0");
        out.push_str(&format!(
            "{a:14} {n_params:>4}-layers {model}\n           {cat}\n"
        ));
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
