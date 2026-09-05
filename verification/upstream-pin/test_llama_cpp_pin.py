#!/usr/bin/env python3
"""Offline contracts for the pinned llama.cpp submodule, lock, and build entry point.

None of these load a model, start a server, touch a GPU, or build anything. They
answer one question: does this workspace agree with itself about which llama.cpp
revision the stack runs, and does anything still point at the runtime layout that
was removed when the external checkout went away.
"""
from __future__ import annotations

import configparser
import json
from pathlib import Path
import re
import subprocess
import unittest

REPO = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO / "upstream" / "llama-cpp.lock.json"
GITMODULES_PATH = REPO / ".gitmodules"
BUILD_SCRIPT = REPO / "scripts" / "build-llama-cpp.sh"
PRESETS = REPO / "llama-models.ini"
# This file has to spell the forbidden paths out in order to search for them, so
# it is the one thing the search skips -- the same reason post_reboot_gate.py
# excludes itself from its own process sweep. Exactly one path, resolved rather
# than hardcoded, so the exemption cannot quietly widen.
SELF = str(Path(__file__).resolve().relative_to(REPO))

LOCK = json.loads(LOCK_PATH.read_text())
SUBMODULE = LOCK["submodule_path"]
BUILD_BIN = REPO / SUBMODULE / "build" / "bin"
LLAMA_SERVER = BUILD_BIN / "llama-server"

# Added by common_params_add_preset_options(); preset-only, so --help never lists
# them. "version" is reserved and skipped by the INI loader.
PRESET_ONLY_KEYS = ("load-on-startup", "stop-timeout", "dedup-cache-models", "version")

# The two runtime paths this migration removed: the deleted checkout, and the
# ~/.local/bin shims that now dangle because their target went with it.
STALE_RUNTIME_PATHS = ("/home/typhoon/src/llama.cpp", ".local/bin/llama")


def stale_hits(text: str) -> list[tuple[int, str]]:
    return [
        (number, line.strip())
        for number, line in enumerate(text.splitlines(), start=1)
        for stale in STALE_RUNTIME_PATHS
        if stale in line
    ]


def fenced_code(text: str) -> str:
    """The lines of a Markdown document that are inside a ``` fence."""
    inside = False
    kept = []
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            inside = not inside
            kept.append("")
            continue
        kept.append(line if inside else "")
    return "\n".join(kept)


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, text=True, check=False,
    )


def require_git_checkout(case: unittest.TestCase) -> None:
    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        case.skipTest(f"{REPO} is not a git checkout")


class LockTests(unittest.TestCase):
    def test_lock_pins_an_exact_upstream_revision(self):
        self.assertEqual("frankenstein-upstream-dependency/1", LOCK["schema"])
        self.assertEqual("llama.cpp", LOCK["name"])
        self.assertEqual("https://github.com/ggml-org/llama.cpp.git", LOCK["repository"])
        self.assertEqual("upstream/llama.cpp", SUBMODULE)
        self.assertEqual("v0.4.0", LOCK["tag"])
        self.assertRegex(LOCK["commit"], r"^[0-9a-f]{40}$")

    def test_lock_records_the_build_the_stack_actually_needs(self):
        build = LOCK["build"]
        self.assertEqual("ROCm/HIP", build["backend"])
        self.assertEqual(["gfx1030"], build["gpu_targets"])
        self.assertEqual("Release", build["build_type"])
        self.assertEqual("Ninja", build["generator"])
        self.assertEqual(
            ["llama-server", "llama-cli", "llama-quantize", "llama-gguf"],
            build["targets"],
        )

    def test_parallelism_is_discovered_not_hardcoded(self):
        # A fixed job count outlives the machine it was measured on. The lock
        # records the discovery mechanism so the build script has something to
        # be checked against.
        self.assertEqual("nproc", LOCK["build"]["parallelism"])


class GitlinkTests(unittest.TestCase):
    def test_gitmodules_declares_the_locked_submodule(self):
        parser = configparser.ConfigParser()
        parser.read_string(GITMODULES_PATH.read_text())
        section = f'submodule "{SUBMODULE}"'
        self.assertIn(section, parser.sections())
        self.assertEqual(SUBMODULE, parser[section]["path"])
        self.assertEqual(LOCK["repository"], parser[section]["url"])

    def test_staged_gitlink_matches_the_locked_commit(self):
        # The gitlink is what a fresh clone checks out, and it is recorded
        # independently of the lock, so the two drift silently. Staging the
        # submodule while its worktree sat on some other revision -- upstream
        # master, say -- pins that revision for everyone else while every local
        # binary and both documents still say v0.4.0.
        require_git_checkout(self)
        result = git("ls-files", "-s", "--", SUBMODULE)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertTrue(result.stdout.strip(), f"{SUBMODULE} is not registered in the index")
        mode, staged, _stage_and_path = result.stdout.split(maxsplit=2)
        self.assertEqual("160000", mode, f"{SUBMODULE} is not staged as a gitlink")
        self.assertEqual(LOCK["commit"], staged)

    def test_submodule_worktree_is_checked_out_at_the_locked_commit(self):
        require_git_checkout(self)
        head = subprocess.run(
            ["git", "-C", str(REPO / SUBMODULE), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        if head.returncode != 0:
            self.skipTest(f"{SUBMODULE} is not initialized")
        self.assertEqual(LOCK["commit"], head.stdout.strip())


class BuildScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = BUILD_SCRIPT.read_text()

    def test_build_script_is_executable_and_fails_fast(self):
        self.assertTrue(BUILD_SCRIPT.is_file())
        self.assertTrue(BUILD_SCRIPT.stat().st_mode & 0o111, "build script is not executable")
        self.assertIn("set -euo pipefail", self.source)

    def test_parallelism_comes_from_nproc_and_is_never_a_fixed_number(self):
        self.assertIn('JOBS="$(nproc)"', self.source)
        self.assertIn('--parallel "$JOBS"', self.source)
        fixed = re.findall(r"-j\s*\d+|--parallel\s+\d+", self.source)
        self.assertEqual([], fixed, f"build script hardcodes a job count: {fixed}")

    def test_build_refuses_a_revision_or_repository_the_lock_does_not_name(self):
        # Every value the build depends on is read back out of the lock and
        # compared, so a lock edit that is not matched by a checkout -- or a
        # checkout moved by hand -- stops the build before CMake runs.
        self.assertIn('rev-parse HEAD)" == "$EXPECTED"', self.source)
        self.assertIn("status --porcelain", self.source)
        self.assertIn(LOCK["repository"], self.source)
        self.assertIn('"$GPU_TARGETS" == "gfx1030"', self.source)

    def test_build_configures_the_locked_backend_and_targets(self):
        self.assertIn("-DGGML_HIP=ON", self.source)
        self.assertIn('-DGPU_TARGETS="$GPU_TARGETS"', self.source)
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", self.source)
        self.assertIn("-G Ninja", self.source)
        for target in LOCK["build"]["targets"]:
            self.assertIn(target, self.source)


class PresetCompatibilityTests(unittest.TestCase):
    """Every router preset key has to be an option this exact build accepts.

    llama.cpp loads --models-preset with ignore_unknown_keys unset, so a key this
    release does not define is not a warning: load_from_ini throws, the router
    never finishes starting, and every alias goes with it. Ask the binary what it
    accepts rather than keeping a second copy of the option list here.
    """

    @classmethod
    def setUpClass(cls):
        if not LLAMA_SERVER.is_file():
            raise unittest.SkipTest(f"{LLAMA_SERVER} is not built")
        result = subprocess.run(
            [str(LLAMA_SERVER), "--help"],
            capture_output=True, text=True, check=False, timeout=120,
        )
        if result.returncode != 0:
            raise unittest.SkipTest(f"{LLAMA_SERVER} --help exited {result.returncode}")
        cls.accepted = set(PRESET_ONLY_KEYS)
        for line in result.stdout.splitlines():
            match = re.match(r"^\s{0,6}(-{1,2}[A-Za-z0-9][\w-]*(?:\s*,\s*-{1,2}[A-Za-z0-9][\w-]*)*)", line)
            if match:
                cls.accepted.update(token.lstrip("-") for token in re.split(r"\s*,\s*", match.group(1)))

    def preset_keys(self) -> list[str]:
        return sorted(set(re.findall(r"(?m)^([a-z0-9][a-z0-9-]*)\s*=", PRESETS.read_text())))

    def test_the_help_output_was_parsed_into_something_usable(self):
        # A parser that silently matched nothing would make the next test vacuous.
        self.assertGreater(len(self.accepted), 100)
        for known in ("ctx-size", "model", "flash-attn"):
            self.assertIn(known, self.accepted)
        self.assertNotIn("definitely-not-an-option", self.accepted)

    def test_every_router_preset_key_is_accepted_by_the_pinned_build(self):
        keys = self.preset_keys()
        self.assertGreater(len(keys), 10, "preset key extraction found almost nothing")
        self.assertEqual([], [key for key in keys if key not in self.accepted])


class RuntimePathTests(unittest.TestCase):
    def tracked_files(self) -> list[str]:
        result = git("ls-files", "--", ".", f":!{SUBMODULE}")
        self.assertEqual(0, result.returncode, result.stderr)
        return [line for line in result.stdout.splitlines() if line]

    def test_nothing_tracked_still_executes_the_removed_checkout(self):
        # Markdown is exempt because the documents have to be able to name these
        # paths in order to forbid them, and because prose executes nothing. Its
        # code blocks are not exempt: those get copied into a shell.
        require_git_checkout(self)
        offenders = []
        for relative in self.tracked_files():
            if relative == SELF:
                continue
            try:
                text = (REPO / relative).read_text(errors="replace")
            except OSError:
                continue
            if relative.endswith(".md"):
                text = fenced_code(text)
            offenders.extend(
                f"{relative}:{number}: {line}" for number, line in stale_hits(text)
            )
        self.assertEqual([], offenders)

    def test_the_stale_path_scan_would_actually_catch_a_regression(self):
        # A scan that silently matches nothing is indistinguishable from a clean
        # tree, so pin both halves: prose is allowed to name the paths, a command
        # block is not.
        prose = "Do not recreate /home/typhoon/src/llama.cpp or its shims.\n"
        self.assertEqual([], stale_hits(fenced_code(prose)))
        block = "Run it:\n\n```\n/home/typhoon/.local/bin/llama-server --version\n```\n"
        self.assertEqual(1, len(stale_hits(fenced_code(block))))
        unit = "ExecStart=/home/typhoon/src/llama.cpp/build/bin/llama-server\n"
        self.assertEqual(1, len(stale_hits(unit)))

    def test_units_and_launchers_run_the_submodule_build(self):
        expected = str(BUILD_BIN / "llama-server")
        units = [
            REPO / "services" / "systemd" / "llama-router.service",
            REPO / "services" / "systemd" / "llama-sidecar@.service",
            REPO / "verification" / "local-coverage-foundation" / "services" / "llama-sidecar@.service",
        ]
        for unit in units:
            with self.subTest(unit=unit.name):
                exec_lines = [
                    line for line in unit.read_text().splitlines()
                    if line.startswith("ExecStart=")
                ]
                self.assertEqual(1, len(exec_lines), f"{unit} has {len(exec_lines)} ExecStart lines")
                self.assertIn(expected, exec_lines[0])

    def test_launcher_scripts_default_to_the_submodule_build(self):
        # convert_qwen3_reranker_gguf.sh reaches the binary through $LLAMA_SRC, so
        # assert the two halves rather than one literal absolute path: the script
        # names the tracked submodule, and it runs the binary out of that build.
        for relative, binary in (
            ("scripts/serve-ridge.sh", "llama-server"),
            ("verification/local-coverage-foundation/validators/run_probe_server.sh", "llama-server"),
            ("verification/local-coverage-foundation/validators/run_vision_probe_server.sh", "llama-server"),
            ("verification/local-coverage-foundation/scripts/convert_qwen3_reranker_gguf.sh", "llama-quantize"),
        ):
            with self.subTest(script=relative):
                text = (REPO / relative).read_text()
                if relative == "scripts/serve-ridge.sh":
                    plan = json.loads(subprocess.check_output(
                        ["bash", str(REPO / relative)], text=True))
                    self.assertTrue(plan["plan_only"])
                    self.assertEqual(plan["command"][0], str(REPO / SUBMODULE / "build/bin/llama-server"))
                    continue
                self.assertIn(str(REPO / SUBMODULE), text)
                self.assertIn(f"build/bin/{binary}", text)


if __name__ == "__main__":
    unittest.main()
