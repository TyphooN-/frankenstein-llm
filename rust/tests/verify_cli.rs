//! End-to-end checks of `frankenctl verify` on tiny disposable artifacts.
//!
//! Each test writes its own manifest and files under a fresh temporary
//! directory and runs the real binary. No model file is read. The only tracked
//! inputs are this checkout's own download-queue manifests, which `plan` reads
//! without examining any artifact path.

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::SystemTime;

use frankenctl::manifest::Sha256Hex;
use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_frankenctl");
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

/// A disposable directory under a canonical temp root, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().canonicalize().expect("temp dir");
        let dir = root.join(format!(
            "frankenctl-verify-cli-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).expect("create scratch dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        path
    }

    fn manifest(&self, name: &str, value: &Value) -> PathBuf {
        self.write(name, &serde_json::to_vec_pretty(value).unwrap())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn published(bytes: &[u8]) -> Option<String> {
    Some(Sha256Hex::of(bytes).as_str().to_string())
}

fn entry(repo_path: &str, destination: &Path, size: usize, sha256: Option<String>) -> Value {
    json!({
        "repo_path": repo_path,
        "destination": destination.to_str().unwrap(),
        "size": size,
        "sha256": sha256,
    })
}

fn artifact(key: &str, files: Vec<Value>) -> Value {
    json!({
        "key": key,
        "capability": "fixture",
        "repository": "owner/repo",
        "revision": REVISION,
        "files": files,
    })
}

fn queue(artifacts: Vec<Value>) -> Value {
    let total: u64 = artifacts
        .iter()
        .flat_map(|a| a["files"].as_array().unwrap().iter())
        .map(|f| f["size"].as_u64().unwrap())
        .sum();
    json!({
        "schema": "hermes-hf-artifact-queue/1",
        "built_at": "2026-09-14T00:00:00-0400",
        "total_bytes": total,
        "artifacts": artifacts,
    })
}

fn frankenctl(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("run frankenctl")
}

fn stdout_json(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "stdout is not JSON ({error}): {}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Size, mode and mtime of every entry below `root`.
fn snapshot(root: &Path) -> BTreeMap<PathBuf, (u64, u32, SystemTime)> {
    let mut seen = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for item in fs::read_dir(&dir).unwrap() {
            let path = item.unwrap().path();
            let metadata = fs::symlink_metadata(&path).unwrap();
            if metadata.is_dir() {
                pending.push(path.clone());
            }
            seen.insert(
                path,
                (
                    metadata.len(),
                    metadata.permissions().mode(),
                    metadata.modified().unwrap(),
                ),
            );
        }
    }
    seen
}

#[test]
fn plan_describes_every_check_from_the_manifest_alone() {
    let scratch = Scratch::new("plan");
    let locked = scratch.write("models/locked.json", b"{}");
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    let never_created = scratch.path().join("never-created/model.gguf");
    let path = scratch.manifest(
        "queue.json",
        &queue(vec![artifact(
            "weights",
            vec![
                entry("model.gguf", &never_created, 7, published(b"1234567")),
                entry("config.json", &locked, 2, None),
            ],
        )]),
    );
    let before = snapshot(scratch.path());

    let output = frankenctl(&["verify", "plan", "--manifest", path.to_str().unwrap()]);

    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert_eq!(stderr(&output), "");
    let plan = stdout_json(&output);
    assert_eq!(plan["mode"], "plan");
    assert_eq!(plan["manifest"]["total_bytes"], 9);
    assert_eq!(plan["selection"]["files"], 2);
    assert_eq!(plan["selection"]["publisher_sha256_files"], 1);
    assert_eq!(plan["selection"]["size_only_files"], 1);
    let files = plan["artifacts"][0]["files"].as_array().unwrap();
    assert_eq!(files[0]["check"], "sha256-and-size");
    assert_eq!(files[0]["destination"], never_created.to_str().unwrap());
    assert_eq!(files[1]["check"], "size-only-no-published-hash");
    assert!(files[1]["sha256"].is_null());
    assert!(!never_created.parent().unwrap().exists());
    assert_eq!(snapshot(scratch.path()), before);
}

#[test]
fn run_verifies_published_digests_and_labels_size_only_files() {
    let scratch = Scratch::new("run-ok");
    let weights = scratch.write("models/model.gguf", b"tiny weights");
    let config = scratch.write("models/config.json", b"{}");
    let path = scratch.manifest(
        "queue.json",
        &queue(vec![artifact(
            "weights",
            vec![
                entry("model.gguf", &weights, 12, published(b"tiny weights")),
                entry("config.json", &config, 2, None),
            ],
        )]),
    );
    let before = snapshot(scratch.path());

    let output = frankenctl(&["verify", "run", "--manifest", path.to_str().unwrap()]);

    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let report = stdout_json(&output);
    assert_eq!(report["exit_code"], 0);
    assert_eq!(report["hash_buffer_bytes"], 8 * 1024 * 1024);
    assert_eq!(report["summary"]["files"], 2);
    assert_eq!(report["summary"]["verified_sha256"], 1);
    assert_eq!(report["summary"]["size_only_no_published_hash"], 1);
    let results = report["results"].as_array().unwrap();
    assert_eq!(results[0]["outcome"], "verified-sha256");
    assert_eq!(results[0]["bytes_hashed"], 12);
    assert_eq!(results[0]["repository"], "owner/repo");
    assert_eq!(results[0]["revision"], REVISION);
    assert_eq!(results[0]["repo_path"], "model.gguf");
    assert_eq!(results[0]["destination"], weights.to_str().unwrap());
    assert_eq!(
        results[0]["expected_sha256"].as_str(),
        published(b"tiny weights").as_deref()
    );
    assert_eq!(results[1]["outcome"], "size-only-no-published-hash");
    assert!(results[1]["expected_sha256"].is_null());
    let not_performed = report["not_performed"].as_array().unwrap();
    for duty in ["download", "completion-stamp lookup"] {
        assert!(not_performed.iter().any(|v| v == duty), "{duty}");
    }
    assert!(not_performed.iter().any(|v| v
        .as_str()
        .unwrap()
        .starts_with("download-lock coordination")));
    assert_eq!(
        snapshot(scratch.path()),
        before,
        "run must not create, change, rename or delete anything"
    );
}

#[test]
fn run_reports_each_integrity_failure_and_ignores_completion_stamps() {
    let scratch = Scratch::new("run-bad");
    let short = scratch.write("models/short.bin", b"abc");
    let wrong = scratch.write("models/wrong.bin", b"abd");
    let missing = scratch.path().join("models/missing.bin");
    let value = queue(vec![artifact(
        "broken",
        vec![
            entry("short.bin", &short, 4, published(b"abcd")),
            entry("wrong.bin", &wrong, 3, published(b"abc")),
            entry("missing.bin", &missing, 3, published(b"abc")),
        ],
    )]);
    let path = scratch.manifest("download-queue.json", &value);
    // A completion stamp that claims this exact total must change nothing.
    let stamp = format!("{}\n", value["total_bytes"]);
    scratch.write("downloads-complete.ok", stamp.as_bytes());

    let output = frankenctl(&["verify", "run", "--manifest", path.to_str().unwrap()]);

    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let report = stdout_json(&output);
    let outcomes: Vec<&str> = report["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["outcome"].as_str().unwrap())
        .collect();
    assert_eq!(outcomes, ["size-mismatch", "sha256-mismatch", "missing"]);
    assert_eq!(report["results"][0]["actual_size"], 3);
    assert_eq!(
        report["results"][1]["actual_sha256"].as_str(),
        published(b"abd").as_deref()
    );
    assert_eq!(report["exit_code"], 1);
    let err = stderr(&output);
    for path in [&short, &wrong, &missing] {
        assert!(err.contains(path.to_str().unwrap()), "{err}");
    }
}

#[test]
fn run_reports_a_real_kernel_eio_with_its_path_and_exits_74() {
    let scratch = Scratch::new("eio");
    let template = scratch.manifest(
        "template.json",
        &queue(vec![artifact(
            "procmem",
            vec![json!({
                "repo_path": "mem",
                "destination": "/proc/@PID@/mem",
                "size": 0,
                "sha256": Sha256Hex::of(b"").as_str(),
            })],
        )]),
    );
    let manifest = scratch.path().join("queue.json");
    // The shell writes its own pid into the manifest and then execs frankenctl
    // under that pid. /proc/<pid>/mem is then the verifier's own address space:
    // a regular file of size 0 whose first read fails with a genuine EIO.
    let output = Command::new("sh")
        .arg("-c")
        .arg(r#"sed "s/@PID@/$$/" "$1" > "$2" && exec "$0" verify run --manifest "$2""#)
        .arg(BIN)
        .arg(&template)
        .arg(&manifest)
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(74), "{}", stderr(&output));
    let written: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    let destination = written["artifacts"][0]["files"][0]["destination"]
        .as_str()
        .unwrap()
        .to_string();
    assert_ne!(destination, "/proc/@PID@/mem");
    let report = stdout_json(&output);
    let result = &report["results"][0];
    assert_eq!(result["outcome"], "unreadable");
    assert_eq!(result["failure"]["operation"], "read");
    assert_eq!(result["failure"]["errno"], 5);
    assert_eq!(result["failure"]["errno_name"], "EIO");
    assert_eq!(result["failure"]["offset"], 0);
    assert_eq!(result["failure"]["path"], destination.as_str());
    assert_eq!(report["summary"]["unreadable"], 1);
    assert_eq!(report["exit_code"], 74);
    let err = stderr(&output);
    assert!(
        err.contains(&destination) && err.contains("Input/output error") && err.contains("EIO"),
        "{err}"
    );
}

#[test]
fn a_malformed_manifest_exits_65_and_checks_nothing() {
    let scratch = Scratch::new("malformed");
    let good = scratch.write("models/good.bin", b"abc");
    let base = queue(vec![artifact(
        "a",
        vec![entry("good.bin", &good, 3, published(b"abc"))],
    )]);
    let mut bad_hash = base.clone();
    bad_hash["artifacts"][0]["files"][0]["sha256"] =
        json!(Sha256Hex::of(b"abc").as_str().to_uppercase());
    let mut duplicate = base.clone();
    let copy = duplicate["artifacts"][0]["files"][0].clone();
    duplicate["artifacts"][0]["files"]
        .as_array_mut()
        .unwrap()
        .push(copy);
    duplicate["total_bytes"] = json!(6);
    let mut total = base.clone();
    total["total_bytes"] = json!(4);
    let mut relative = base.clone();
    relative["artifacts"][0]["files"][0]["destination"] = json!("models/good.bin");

    for (needle, value) in [
        ("sha256", bad_hash),
        ("claimed by both", duplicate),
        ("total_bytes", total),
        ("not an absolute path", relative),
    ] {
        let path = scratch.manifest("queue.json", &value);
        let output = frankenctl(&["verify", "run", "--manifest", path.to_str().unwrap()]);
        assert_eq!(
            output.status.code(),
            Some(65),
            "{needle}: {}",
            stderr(&output)
        );
        assert!(output.stdout.is_empty(), "{needle}");
        assert!(
            stderr(&output).contains(needle),
            "{needle}: {}",
            stderr(&output)
        );
    }

    let path = scratch.write("queue.json", b"{not json");
    let output = frankenctl(&["verify", "plan", "--manifest", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(65), "{}", stderr(&output));
    assert!(output.stdout.is_empty());
}

#[test]
fn artifact_selection_is_exact() {
    let scratch = Scratch::new("select");
    let present = scratch.write("models/present.bin", b"abc");
    let absent = scratch.path().join("models/absent.bin");
    let path = scratch.manifest(
        "queue.json",
        &queue(vec![
            artifact(
                "present",
                vec![entry("present.bin", &present, 3, published(b"abc"))],
            ),
            artifact(
                "absent",
                vec![entry("absent.bin", &absent, 3, published(b"abc"))],
            ),
        ]),
    );
    let manifest = path.to_str().unwrap();

    let everything = frankenctl(&["verify", "run", "--manifest", manifest]);
    assert_eq!(everything.status.code(), Some(1), "{}", stderr(&everything));

    let one = frankenctl(&[
        "verify",
        "run",
        "--manifest",
        manifest,
        "--artifact",
        "present",
    ]);
    assert_eq!(one.status.code(), Some(0), "{}", stderr(&one));
    let report = stdout_json(&one);
    assert_eq!(report["selection"]["artifact_keys"], json!(["present"]));
    assert_eq!(report["selection"]["all_artifacts"], false);
    assert_eq!(report["results"].as_array().unwrap().len(), 1);

    for extra in [
        vec!["--artifact", "nope"],
        vec!["--artifact", "present", "--artifact", "present"],
    ] {
        let mut args = vec!["verify", "run", "--manifest", manifest];
        args.extend(extra);
        let output = frankenctl(&args);
        assert_eq!(
            output.status.code(),
            Some(2),
            "{args:?}: {}",
            stderr(&output)
        );
        assert!(output.stdout.is_empty(), "{args:?}");
    }
}

#[test]
fn an_absent_manifest_and_bad_usage_have_their_own_exit_codes() {
    let scratch = Scratch::new("usage");
    let absent = scratch.path().join("absent.json");
    let output = frankenctl(&["verify", "run", "--manifest", absent.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(66), "{}", stderr(&output));
    assert!(stderr(&output).contains("No such file or directory"));
    assert!(output.stdout.is_empty());
    assert_eq!(frankenctl(&["verify", "run"]).status.code(), Some(2));
    assert_eq!(
        frankenctl(&["verify", "plan", "--manifest"]).status.code(),
        Some(2)
    );
}

#[test]
fn plan_ingests_every_tracked_download_queue_manifest_exactly() {
    let foundation =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../verification/local-coverage-foundation");
    let mut queues: Vec<PathBuf> = fs::read_dir(&foundation)
        .unwrap()
        .map(|item| item.unwrap().path())
        .filter(|path| {
            let name = path.file_name().unwrap().to_string_lossy();
            name.starts_with("download-queue") && name.ends_with(".json")
        })
        .collect();
    queues.sort();
    assert!(queues.len() >= 5, "{queues:?}");

    for queue_path in &queues {
        let raw: Value = serde_json::from_slice(&fs::read(queue_path).unwrap()).unwrap();
        let output = frankenctl(&["verify", "plan", "--manifest", queue_path.to_str().unwrap()]);
        assert_eq!(
            output.status.code(),
            Some(0),
            "{}: {}",
            queue_path.display(),
            stderr(&output)
        );
        let plan = stdout_json(&output);
        assert_eq!(plan["manifest"]["total_bytes"], raw["total_bytes"]);
        let wanted = raw["artifacts"].as_array().unwrap();
        let planned = plan["artifacts"].as_array().unwrap();
        assert_eq!(planned.len(), wanted.len());
        let (mut hashed, mut size_only) = (0, 0);
        for (want, got) in wanted.iter().zip(planned) {
            for field in ["key", "capability", "repository", "revision"] {
                assert_eq!(got[field], want[field], "{}", queue_path.display());
            }
            let want_files = want["files"].as_array().unwrap();
            let got_files = got["files"].as_array().unwrap();
            assert_eq!(got_files.len(), want_files.len());
            for (want_file, got_file) in want_files.iter().zip(got_files) {
                for field in ["repo_path", "destination", "size", "sha256"] {
                    assert_eq!(
                        got_file[field],
                        want_file[field],
                        "{}",
                        queue_path.display()
                    );
                }
                if want_file["sha256"].is_null() {
                    size_only += 1;
                    assert_eq!(got_file["check"], "size-only-no-published-hash");
                } else {
                    hashed += 1;
                    assert_eq!(got_file["check"], "sha256-and-size");
                }
            }
        }
        assert_eq!(plan["selection"]["publisher_sha256_files"], hashed);
        assert_eq!(plan["selection"]["size_only_files"], size_only);
    }
}
