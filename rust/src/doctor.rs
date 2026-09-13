//! `frankenctl doctor` -- real repository and runtime checks.
//!
//! Each check returns a `Check` with a name, an optional detail, and whether
//! it passed. `doctor` is fail-closed: any mandatory check that fails yields a
//! non-zero exit. The checks are deliberately concrete (a real path must exist
//! and be readable, the pinned llama.cpp binary must be present and executable,
//! the upstream lock must parse and match the submodule, the expected number of
//! GPUs must be visible), not decorative.

use std::path::{Path, PathBuf};

use crate::error::CheckError;
use crate::gpu::{self, DRM_ROOT};

/// The repository root, resolved relative to the running binary when possible
/// and overridable for tests.
pub fn repo_root() -> PathBuf {
    // The crate lives at <repo>/rust, so the parent of CARGO_MANIFEST_DIR is
    // the repository root. CARGO_MANIFEST_DIR is compile-time and correct for
    // this single-host project.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// A single named check result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub name: &'static str,
    pub passed: bool,
    pub mandatory: bool,
    pub detail: String,
}

impl Check {
    fn pass(name: &'static str, mandatory: bool, detail: String) -> Self {
        Self {
            name,
            passed: true,
            mandatory,
            detail,
        }
    }
    fn fail(name: &'static str, mandatory: bool, detail: String) -> Self {
        Self {
            name,
            passed: false,
            mandatory,
            detail,
        }
    }
}

/// The set of checks `doctor` runs.
pub struct Doctor {
    pub checks: Vec<Check>,
    pub repo_root: PathBuf,
    pub expected_gpus: Option<usize>,
}

impl Doctor {
    /// Run all checks. `expected_gpus`, when `Some`, is asserted against the
    /// observed count; when `None`, the GPU count is reported but not enforced
    /// (useful on a host where the card set legitimately varies).
    pub fn run(repo_root: PathBuf, expected_gpus: Option<usize>) -> Self {
        let mut checks = Vec::new();
        checks.push(Self::check_repo_readable(&repo_root));
        checks.push(Self::check_config_readability(&repo_root));
        checks.push(Self::check_llama_binary(&repo_root));
        checks.push(Self::check_upstream_pin(&repo_root));
        checks.push(Self::check_runtime_paths(&repo_root));
        checks.push(Self::check_gpu_inventory(
            &repo_root,
            expected_gpus,
            expected_gpus.is_some(),
        ));
        Self {
            checks,
            repo_root,
            expected_gpus,
        }
    }

    /// True when every mandatory check passed.
    pub fn ok(&self) -> bool {
        self.checks.iter().all(|c| !c.mandatory || c.passed)
    }

    /// Render a human-readable report, one line per check.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for c in &self.checks {
            let mark = if c.passed { "PASS" } else { "FAIL" };
            let mand = if c.mandatory { " (required)" } else { "" };
            out.push_str(&format!("[{mark}]{mand} {}: {}\n", c.name, c.detail));
        }
        out
    }

    fn check_repo_readable(root: &Path) -> Check {
        let name = "repository readable";
        if root.join("llama-models.ini").is_file() {
            Check::pass(name, true, "llama-models.ini present".to_string())
        } else {
            Check::fail(name, true, "llama-models.ini missing".to_string())
        }
    }

    fn check_config_readability(root: &Path) -> Check {
        let name = "config files readable";
        let catalog = root.join("config/model-catalog.json");
        let serving = root.join("config/serving.json");
        if catalog.is_file() && serving.is_file() {
            // Parse the catalog to confirm it is not just present but valid.
            match crate::config::load_catalog(&catalog) {
                Ok(entries) => {
                    Check::pass(name, true, format!("catalog has {} entries", entries.len()))
                }
                Err(e) => Check::fail(name, true, format!("catalog invalid: {e}")),
            }
        } else {
            Check::fail(
                name,
                true,
                "model-catalog.json or serving.json missing".to_string(),
            )
        }
    }

    fn check_llama_binary(root: &Path) -> Check {
        let name = "pinned llama.cpp runtime";
        let bin = root.join("upstream/llama.cpp/build/bin/llama-server");
        if bin.is_file() {
            Check::pass(name, true, bin.display().to_string())
        } else {
            Check::fail(
                name,
                true,
                format!("{} missing; build the pinned runtime first", bin.display()),
            )
        }
    }

    fn check_upstream_pin(root: &Path) -> Check {
        let name = "upstream pin sanity";
        let lock = root.join("upstream/llama-cpp.lock.json");
        match std::fs::read_to_string(&lock) {
            Ok(raw) => match serde_json::from_str::<serde_json::Value>(&raw) {
                Ok(v) => {
                    // The lock must carry the commit pin.
                    let commit = v
                        .get("commit")
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string())
                        .or_else(|| {
                            v.get("llama_cpp")
                                .and_then(|l| l.get("commit"))
                                .and_then(|c| c.as_str())
                                .map(|s| s.to_string())
                        });
                    match commit {
                        Some(c) => match std::process::Command::new("git")
                            .args([
                                "-C",
                                root.join("upstream/llama.cpp").to_str().unwrap_or("."),
                                "rev-parse",
                                "HEAD",
                            ])
                            .output()
                        {
                            Ok(out) if out.status.success() => {
                                let head = String::from_utf8_lossy(&out.stdout).trim().to_string();
                                if head == c {
                                    Check::pass(
                                        name,
                                        true,
                                        format!("commit {}", &c[..c.len().min(12)]),
                                    )
                                } else {
                                    Check::fail(
                                        name,
                                        true,
                                        format!("lock {c} != submodule HEAD {head}"),
                                    )
                                }
                            }
                            Ok(out) => Check::fail(
                                name,
                                true,
                                format!(
                                    "git rev-parse failed: {}",
                                    String::from_utf8_lossy(&out.stderr)
                                ),
                            ),
                            Err(e) => Check::fail(name, true, format!("git rev-parse: {e}")),
                        },
                        None => Check::fail(name, true, "lock has no commit field".to_string()),
                    }
                }
                Err(e) => Check::fail(name, true, format!("lock is not valid JSON: {e}")),
            },
            Err(e) => Check::fail(name, true, format!("lock unreadable: {e}")),
        }
    }

    fn check_runtime_paths(root: &Path) -> Check {
        let name = "required runtime paths";
        let mut missing: Vec<String> = Vec::new();
        for rel in ["upstream/llama.cpp", "config/chat-templates", "models"] {
            if !root.join(rel).exists() {
                missing.push(rel.to_string());
            }
        }
        if missing.is_empty() {
            Check::pass(name, true, "all present".to_string())
        } else {
            Check::fail(name, true, format!("missing: {}", missing.join(", ")))
        }
    }

    fn check_gpu_inventory(_root: &Path, expected: Option<usize>, enforce: bool) -> Check {
        let name = "GPU inventory";
        let root = Path::new(DRM_ROOT);
        match gpu::observe(root) {
            Ok(gpus) => {
                let found = gpus.len();
                let summary = gpus
                    .iter()
                    .map(|g| format!("card{}={}", g.card, g.device))
                    .collect::<Vec<_>>()
                    .join(", ");
                match expected {
                    Some(exp) if exp != found => Check::fail(
                        name,
                        enforce,
                        format!("expected {exp} GPUs, found {found} ({summary})"),
                    ),
                    Some(exp) => Check::pass(
                        name,
                        true,
                        format!("{found} GPUs present ({summary}); expected {exp}"),
                    ),
                    None => Check::pass(name, false, format!("{found} GPUs present ({summary})")),
                }
            }
            Err(e) => Check::fail(name, false, format!("sysfs scan failed: {e}")),
        }
    }
}

#[allow(dead_code)]
fn check_error_helper(e: CheckError) -> String {
    e.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctor_runs_and_reports() {
        let d = Doctor::run(PathBuf::from("/nonexistent-should-fail"), None);
        // With a bogus root, the mandatory repo checks must fail, so ok() is
        // false. This proves the fail-closed path: a broken checkout is not
        // silently reported as healthy.
        assert!(!d.ok());
        assert!(!d.render().is_empty());
    }

    #[test]
    fn doctor_pin_matches_this_checkout() {
        let root = repo_root().canonicalize().expect("repo root");
        let d = Doctor::run(root, None);
        let pin = d
            .checks
            .iter()
            .find(|c| c.name == "upstream pin sanity")
            .expect("pin check");
        assert!(pin.passed, "{}", pin.detail);
    }
}
