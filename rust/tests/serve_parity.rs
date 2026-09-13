//! Compare the Rust planner to live Python `scripts/serve-model.py::command()`.
//!
//! The Python module is loaded from the main checkout when this crate lives in
//! a worktree, so the test follows current presets rather than a duplicated
//! Rust oracle.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use frankenctl::config::{self, Serving};

fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.canonicalize().expect("repo root canonicalize")
}

fn python_root(repo: &Path) -> PathBuf {
    let main = PathBuf::from("/home/typhoon/git/frankenstein-llm");
    if main.join("scripts/serve-model.py").is_file() {
        main
    } else {
        repo.to_path_buf()
    }
}

fn python_commands(root: &Path) -> BTreeMap<String, Vec<String>> {
    let script = r#"
import importlib.util
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "scripts"))
from model_catalog import presets
spec = importlib.util.spec_from_file_location("serve_model", root / "scripts" / "serve-model.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
serving = json.loads((root / "config" / "serving.json").read_text())
out = {}
for alias, values in presets(root / "llama-models.ini").items():
    out[alias] = mod.command(alias, values, serving)
print(json.dumps(out))
"#;
    let output = Command::new("python3")
        .args(["-c", script, root.to_str().expect("utf8 path")])
        .output()
        .expect("run python reference");
    assert!(
        output.status.success(),
        "python reference failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("python json")
}

#[test]
fn rust_planner_matches_live_python_for_all_presets() {
    let rust_root = repo_root();
    let py_root = python_root(&rust_root);
    let ini = py_root.join("llama-models.ini");
    assert!(
        ini.is_file(),
        "llama-models.ini missing at {}",
        ini.display()
    );

    let presets = config::load_presets(&ini).expect("load presets");
    assert!(!presets.is_empty(), "no presets loaded");

    let serving = config::load_serving(&py_root.join("config/serving.json")).expect("serving");
    let llama_bin = py_root.join("upstream/llama.cpp/build/bin/llama-server");
    let python = python_commands(&py_root);
    assert_eq!(
        presets.keys().cloned().collect::<Vec<_>>(),
        python.keys().cloned().collect::<Vec<_>>(),
        "alias set mismatch"
    );

    for (alias, preset) in &presets {
        let got = config::command(&llama_bin, alias, preset, &serving)
            .unwrap_or_else(|e| panic!("rust planner failed for {alias}: {e}"));
        let want = python
            .get(alias)
            .unwrap_or_else(|| panic!("python missing {alias}"));
        assert_eq!(&got, want, "planner mismatch for preset {alias}");
    }
    let _ = Serving {
        host: serving.host,
        port: serving.port,
    };
}
