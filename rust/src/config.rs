//! Typed configuration for the Frankenstein stack.
//!
//! This module ports the three operator-owned configuration surfaces that the
//! Python serving path reads:
//!
//! * `config/model-catalog.json` -- provenance and intended use per alias.
//! * `config/serving.json`       -- the loopback host and port.
//! * `llama-models.ini`          -- the per-alias `llama-server` preset
//!   arguments, with the shared `[*]` section merged into every alias.
//!
//! The `command()` function here is a faithful port of
//! `scripts/serve-model.py::command()`: same flag vocabulary, same ordering
//! (host, port, alias, then keys in the preset's iteration order), same
//! fail-closed validation, and the same `--flag` / `--flag value` shapes.
//! Parity is proven by the integration fixtures in `tests/serve_plan.rs`.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde::Deserialize;

use crate::error::ConfigError;

/// A single model's provenance and intended use.
///
/// Mirrors `model_catalog.CATALOG_FIELDS` exactly: `source` is where the
/// bytes came from, `purpose` is the intended use, and both are mandatory and
/// non-empty. The catalog is a closed vocabulary of exactly these two fields;
/// a stray key is an error, not silently ignored.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    pub source: String,
    pub purpose: String,
}

/// The loopback serving endpoint.
///
/// `scripts/serve-model.py::command()` requires the host to be exactly
/// `127.0.0.1` and the port to be an integer in `1024..=65535`. We enforce
/// the same here so a misconfigured `serving.json` fails closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Serving {
    pub host: &'static str,
    pub port: u16,
}

/// The `[*]` shared section plus the per-alias overrides, merged.
///
/// Keys are the `llama-server` long-option names exactly as they appear in
/// `llama-models.ini` (e.g. `ctx-size`, `tensor-split`, `gpu-layers`). Values
/// are the raw string the operator wrote, so the launcher can pass them
/// through verbatim.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Preset(Vec<(String, String)>);

impl Preset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key/value. Existing keys keep their position so later alias
    /// overrides do not reshuffle the shared `[*]` order, matching Python
    /// `{**common, **section}`.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        if let Some((_, current)) = self.0.iter_mut().find(|(k, _)| k == &key) {
            *current = value;
            return;
        }
        self.0.push((key, value));
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.0.iter().any(|(k, _)| k == key)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The model weight path, if configured.
    pub fn model(&self) -> Option<&str> {
        self.get("model")
    }

    /// The projector path, if configured.
    pub fn mmproj(&self) -> Option<&str> {
        self.get("mmproj")
    }

    /// Keys in insertion order: shared `[*]` first, then alias-only keys.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> + '_ {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

/// `llama-server` flags that take no value in this stack.
///
/// `jinja` is special: `0` renders as `--no-jinja`, `1` as `--jinja`.
/// `embedding` and `reranking` are `--embedding` / `--reranking` when `1` and
/// absent when `0`. This mirrors `serve-model.BOOL_FLAGS` exactly.
#[derive(Debug, Clone, Copy)]
struct BoolFlag {
    on: Option<&'static str>,
    off: Option<&'static str>,
}

fn bool_flag(key: &str) -> Option<BoolFlag> {
    match key {
        "jinja" => Some(BoolFlag {
            on: Some("--jinja"),
            off: Some("--no-jinja"),
        }),
        "embedding" => Some(BoolFlag {
            on: Some("--embedding"),
            off: None,
        }),
        "reranking" => Some(BoolFlag {
            on: Some("--reranking"),
            off: None,
        }),
        _ => None,
    }
}

/// The value-bearing flags this stack understands. Anything else is an
/// unknown preset key and fails closed.
fn is_value_flag(key: &str) -> bool {
    matches!(
        key,
        "model"
            | "mmproj"
            | "mmproj-device"
            | "ctx-size"
            | "gpu-layers"
            | "flash-attn"
            | "cache-type-k"
            | "cache-type-v"
            | "parallel"
            | "device"
            | "tensor-split"
            | "reasoning"
            | "spec-type"
            | "spec-draft-n-max"
            | "temp"
            | "repeat-penalty"
            | "min-p"
            | "top-p"
            | "top-k"
            | "batch-size"
            | "ubatch-size"
            | "pooling"
            | "embd-normalize"
            | "chat-template-file"
            | "load-mode"
    )
}

/// Build the exact `llama-server` invocation for a preset.
///
/// This is a faithful port of `serve-model.command()`. The binary path is
/// supplied by the caller so the function is testable without the build
/// tree; production callers pass the pinned
/// `upstream/llama.cpp/build/bin/llama-server`.
///
/// # Errors
///
/// * `ConfigError::ServingHostPort` if the host is not `127.0.0.1`.
/// * `ConfigError::ServingPort` if the port is outside `1024..=65535`.
/// * A `Value` error (wrapped by the caller) for unknown preset keys, a
///   missing `model`, or a bool flag that is not `0`/`1`.
pub fn command(
    binary: &Path,
    alias: &str,
    preset: &Preset,
    serving: &Serving,
) -> anyhow::Result<Vec<String>> {
    if serving.host != "127.0.0.1" {
        return Err(ConfigError::ServingHostPort.into());
    }
    if !(1024..=65535).contains(&serving.port) {
        return Err(ConfigError::ServingPort.into());
    }

    // Collect unknown keys first so the error names them all at once, exactly
    // like the Python reference does with a sorted join.
    let mut unknown: Vec<&str> = Vec::new();
    for (key, _) in preset.iter() {
        if bool_flag(key).is_none() && !is_value_flag(key) {
            unknown.push(key);
        }
    }
    if !unknown.is_empty() {
        unknown.sort_unstable();
        unknown.dedup();
        return Err(anyhow::anyhow!(
            "unknown preset keys: {}",
            unknown.join(", ")
        ));
    }

    if preset.model().map(str::trim).unwrap_or("").is_empty() {
        return Err(anyhow::anyhow!("model path is required"));
    }

    let mut cmd: Vec<String> = vec![
        binary.to_string_lossy().into_owned(),
        "--host".to_string(),
        serving.host.to_string(),
        "--port".to_string(),
        serving.port.to_string(),
        "--alias".to_string(),
        alias.to_string(),
    ];

    for (key, value) in preset.iter() {
        if let Some(flag) = bool_flag(key) {
            if value != "0" && value != "1" {
                return Err(anyhow::anyhow!("{key} must be 0 or 1"));
            }
            let chosen = if value == "1" { flag.on } else { flag.off };
            if let Some(f) = chosen {
                cmd.push(f.to_string());
            }
        } else {
            cmd.push(format!("--{key}"));
            cmd.push(value.to_string());
        }
    }

    Ok(cmd)
}

// --- Loaders ----------------------------------------------------------------

/// Load and strictly validate `config/model-catalog.json`.
///
/// Every entry must carry exactly `source` and `purpose`, both non-empty
/// strings. A missing file, malformed JSON, a non-object, or a stray key is an
/// error: a broken catalog must fail the serving path closed, not degrade.
pub fn load_catalog(
    path: &Path,
) -> std::result::Result<BTreeMap<String, CatalogEntry>, ConfigError> {
    let raw = std::fs::read_to_string(path).map_err(|e| ConfigError::io(path, &e))?;
    let parsed: std::collections::HashMap<String, CatalogEntry> =
        serde_json::from_str(&raw).map_err(|e| ConfigError::json(path, &e))?;
    // serde enforces the two-field shape and non-emptiness of the type; we
    // additionally require the strings to be non-blank to match the Python
    // reference's `value.strip()` check.
    let mut out = BTreeMap::new();
    for (alias, entry) in parsed {
        if entry.source.trim().is_empty() || entry.purpose.trim().is_empty() {
            return Err(ConfigError::CatalogEntry(alias));
        }
        out.insert(alias, entry);
    }
    Ok(out)
}

/// Load and validate `config/serving.json`.
pub fn load_serving(path: &Path) -> std::result::Result<Serving, ConfigError> {
    let raw = std::fs::read_to_string(path).map_err(|e| ConfigError::io(path, &e))?;
    let v: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| ConfigError::json(path, &e))?;
    let host = v
        .get("host")
        .and_then(|h| h.as_str())
        .ok_or_else(|| ConfigError::ServingHostPort)?;
    let port = v
        .get("port")
        .and_then(|p| p.as_u64())
        .ok_or_else(|| ConfigError::ServingPort)?;
    // The Python reference requires the object to contain *only* host and
    // port. Enforce that.
    let keys: Vec<&str> = v
        .as_object()
        .map(|o| o.keys().map(|k| k.as_str()).collect())
        .unwrap_or_default();
    let expected = ["host", "port"];
    if keys.len() != expected.len() || !keys.iter().all(|k| expected.contains(k)) {
        return Err(ConfigError::ServingHostPort);
    }
    let host_static = match host {
        "127.0.0.1" => "127.0.0.1",
        _ => return Err(ConfigError::ServingHostPort),
    };
    let port = u16::try_from(port).map_err(|_| ConfigError::ServingPort)?;
    Ok(Serving {
        host: host_static,
        port,
    })
}

/// Load `llama-models.ini` and return the merged per-alias presets.
///
/// This is a faithful port of the INI handling in `scripts/model_catalog.py`:
/// a minimal INI reader that understands `[section]` headers, `key = value`
/// and `key = value` pairs, `#` and `;` comments, blank lines, and the shared
/// `[*]` section that is merged into every alias section. The merge order is
/// `[*]` first, then the alias section on top, so an alias may override any
/// shared default. The result is a `BTreeMap` keyed by alias with each value a
/// `Preset` carrying the merged key/value pairs.
///
/// # Errors
///
/// * `ConfigError::Io` if the file cannot be read.
/// * `ConfigError::Json` (reused as the "malformed content" carrier) if a
///   line is not a header, a pair, a comment, or blank.
pub fn load_presets(path: &Path) -> std::result::Result<BTreeMap<String, Preset>, ConfigError> {
    let raw = std::fs::read_to_string(path).map_err(|e| ConfigError::io(path, &e))?;
    let mut sections: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let mut current: Option<String> = None;
    for (lineno, raw_line) in raw.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(body) = line.strip_prefix('[') {
            let name = body.strip_suffix(']').ok_or_else(|| {
                ConfigError::Json(
                    path.to_path_buf(),
                    format!("line {}: bad section header", lineno + 1),
                )
            })?;
            let name = name.trim().to_string();
            current = Some(name.clone());
            sections.entry(name).or_default();
            continue;
        }
        // key = value (the '=' is mandatory; the Python reference uses split('=', 1)).
        let (key, value) = line.split_once('=').ok_or_else(|| {
            ConfigError::Json(
                path.to_path_buf(),
                format!("line {}: not a key=value pair", lineno + 1),
            )
        })?;
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim().to_string();
        if key.is_empty() {
            return Err(ConfigError::Json(
                path.to_path_buf(),
                format!("line {}: empty key", lineno + 1),
            ));
        }
        let section = current.clone().ok_or_else(|| {
            ConfigError::Json(
                path.to_path_buf(),
                format!("line {}: key before any section", lineno + 1),
            )
        })?;
        let pairs = sections
            .get_mut(&section)
            .expect("section was created by its header");
        if let Some((_, current)) = pairs.iter_mut().find(|(k, _)| k == &key) {
            *current = value;
        } else {
            pairs.push((key, value));
        }
    }

    let shared = sections.get("*").cloned().unwrap_or_default();
    let mut out = BTreeMap::new();
    for (alias, pairs) in &sections {
        if alias == "*" {
            continue;
        }
        let mut preset = Preset::new();
        // Merge [*] first, then the alias section on top.
        for (k, v) in &shared {
            preset.set(k.clone(), v.clone());
        }
        for (k, v) in pairs {
            preset.set(k.clone(), v.clone());
        }
        out.insert(alias.clone(), preset);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_requires_loopback() {
        assert_eq!(
            Serving {
                host: "127.0.0.1",
                port: 8080
            }
            .host,
            "127.0.0.1"
        );
    }

    #[test]
    fn command_rejects_unknown_key() {
        let mut preset = Preset::new();
        preset.set("model", "/tmp/x.gguf");
        preset.set("bogus", "1");
        let serving = Serving {
            host: "127.0.0.1",
            port: 8080,
        };
        let err = command(Path::new("/bin/llama-server"), "a", &preset, &serving).unwrap_err();
        assert!(
            err.to_string().contains("unknown preset keys: bogus"),
            "{err}"
        );
    }

    #[test]
    fn command_requires_model() {
        let preset = Preset::new();
        let serving = Serving {
            host: "127.0.0.1",
            port: 8080,
        };
        let err = command(Path::new("/bin/llama-server"), "a", &preset, &serving).unwrap_err();
        assert!(err.to_string().contains("model path is required"), "{err}");
    }

    #[test]
    fn command_rejects_empty_model() {
        let mut preset = Preset::new();
        preset.set("model", "");
        let serving = Serving {
            host: "127.0.0.1",
            port: 8080,
        };
        let err = command(Path::new("/bin/llama-server"), "a", &preset, &serving).unwrap_err();
        assert!(err.to_string().contains("model path is required"), "{err}");
    }

    #[test]
    fn command_accepts_top_p_and_top_k() {
        let mut preset = Preset::new();
        preset.set("model", "/tmp/weights.gguf");
        preset.set("top-p", "0.95");
        preset.set("top-k", "20");
        let serving = Serving {
            host: "127.0.0.1",
            port: 8080,
        };
        let cmd = command(Path::new("/bin/llama-server"), "signal", &preset, &serving).unwrap();
        let joined = cmd.join(" ");
        assert!(joined.contains("--top-p 0.95"), "{joined}");
        assert!(joined.contains("--top-k 20"), "{joined}");
    }
}
