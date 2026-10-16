"""Installer behavior in real disposable Git trees; no live Hermes state."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('install_runtime_under_test', ROOT / 'install_runtime.py')
assert spec is not None and spec.loader is not None
INSTALL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(INSTALL)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='worker-install-')
        self.addCleanup(self.temp.cleanup)
        self.area = Path(self.temp.name)
        self.repo = self.area / 'checkout'
        self.repo.mkdir()
        self.bundle = self.area / 'bundle'
        self.bundle.mkdir()
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        self.env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
        self.git('init', '-q')
        (self.repo / 'agent').mkdir()
        self.target = self.repo / 'agent/fixture.py'
        self.target.write_bytes(b'before\n')
        self.git('add', 'agent/fixture.py')
        self.git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
        self.base = self.git('rev-parse', 'HEAD').decode().strip()
        self.target.write_bytes(b'after\n')
        patch = self.git('diff', '--no-ext-diff', '--src-prefix=a/', '--dst-prefix=b/')
        self.target.write_bytes(b'before\n')
        (self.bundle / 'hermes-local-helper.patch').write_bytes(patch)
        self.manifest = {'schema_version': 1, 'base_commit': self.base,
            'patch': 'hermes-local-helper.patch', 'patch_sha256': sha(patch),
            'files': [{'path': 'agent/fixture.py', 'base_sha256': sha(b'before\n'),
                'sha256': sha(b'after\n'), 'base_mode': '100644', 'mode': '100644'}]}
        self.save_manifest()

    def git(self, *args):
        result = subprocess.run(['git', *args], cwd=self.repo, env=self.env, capture_output=True, check=True)
        return result.stdout

    def save_manifest(self):
        (self.bundle / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_check_reports_ready_without_writing_checkout_or_index(self):
        check = getattr(INSTALL, 'check', lambda *args: {'status': 'missing'})
        before = (self.repo / '.git/index').read_bytes()
        self.assertEqual(check(self.repo, self.bundle)['status'], 'ready')
        self.assertEqual(self.target.read_bytes(), b'before\n')
        self.assertEqual((self.repo / '.git/index').read_bytes(), before)

    def test_apply_changes_only_declared_files_and_is_idempotent(self):
        apply = getattr(INSTALL, 'apply', lambda *args: {'status': 'missing'})
        unrelated = self.repo / 'unrelated.txt'
        unrelated.write_text('preserve this staged addition')
        self.git('add', 'unrelated.txt')
        before = (self.repo / '.git/index').read_bytes()
        self.assertEqual(apply(self.repo, self.bundle)['status'], 'already_applied')
        self.assertEqual(self.target.read_bytes(), b'after\n')
        self.assertEqual(apply(self.repo, self.bundle)['status'], 'already_applied')
        self.assertEqual(unrelated.read_text(), 'preserve this staged addition')
        self.assertEqual((self.repo / '.git/index').read_bytes(), before)

    def test_tampered_payload_refuses_before_any_write(self):
        self.manifest['patch_sha256'] = '0' * 64
        self.save_manifest()
        index = (self.repo / '.git/index').read_bytes()
        with self.assertRaises(INSTALL.InstallError):
            INSTALL.apply(self.repo, self.bundle)
        self.assertEqual(self.target.read_bytes(), b'before\n')
        self.assertEqual((self.repo / '.git/index').read_bytes(), index)

    def test_wrong_base_refuses_without_writing(self):
        self.manifest['base_commit'] = '0' * 40
        self.save_manifest()
        with self.assertRaises(INSTALL.InstallError):
            INSTALL.apply(self.repo, self.bundle)
        self.assertEqual(self.target.read_bytes(), b'before\n')

    def test_dirty_target_refuses_even_when_index_hides_the_edit(self):
        self.git('update-index', '--assume-unchanged', 'agent/fixture.py')
        self.target.write_bytes(b'private work\n')
        with self.assertRaises(INSTALL.InstallError):
            INSTALL.apply(self.repo, self.bundle)
        self.assertEqual(self.target.read_bytes(), b'private work\n')

    def test_staged_target_refuses_even_when_worktree_is_base(self):
        self.target.write_bytes(b'staged work\n')
        self.git('add', 'agent/fixture.py')
        self.target.write_bytes(b'before\n')
        index = (self.repo / '.git/index').read_bytes()
        with self.assertRaises(INSTALL.InstallError):
            INSTALL.apply(self.repo, self.bundle)
        self.assertEqual((self.repo / '.git/index').read_bytes(), index)
        self.assertEqual(self.target.read_bytes(), b'before\n')

    def test_unsafe_manifest_paths_and_duplicates_refuse(self):
        original = json.loads(json.dumps(self.manifest))
        for name in ('../escape.py', '/agent/fixture.py', '.git/config',
                     'agent/../fixture.py', 'agent//fixture.py', 'agent/config.yaml'):
            with self.subTest(name=name):
                self.manifest = json.loads(json.dumps(original))
                self.manifest['files'][0]['path'] = name
                self.save_manifest()
                with self.assertRaises(INSTALL.InstallError):
                    INSTALL.apply(self.repo, self.bundle)
        self.manifest = original
        self.manifest['files'].append(dict(self.manifest['files'][0]))
        self.save_manifest()
        with self.assertRaises(INSTALL.InstallError):
            INSTALL.apply(self.repo, self.bundle)
        self.assertEqual(self.target.read_bytes(), b'before\n')

    def test_patch_cannot_write_an_undeclared_file(self):
        extra = b'diff --git a/agent/extra.py b/agent/extra.py\nnew file mode 100644\n--- /dev/null\n+++ b/agent/extra.py\n@@ -0,0 +1 @@\n+extra\n'
        path = self.bundle / 'hermes-local-helper.patch'
        data = path.read_bytes() + extra
        path.write_bytes(data)
        self.manifest['patch_sha256'] = sha(data)
        self.save_manifest()
        with self.assertRaises(INSTALL.InstallError):
            INSTALL.apply(self.repo, self.bundle)
        self.assertFalse((self.repo / 'agent/extra.py').exists())
        self.assertEqual(self.target.read_bytes(), b'before\n')

    def test_symlink_target_refuses_without_touching_destination(self):
        other = self.area / 'outside.py'
        other.write_bytes(b'before\n')
        self.target.unlink()
        self.target.symlink_to(other)
        with self.assertRaises(INSTALL.InstallError):
            INSTALL.apply(self.repo, self.bundle)
        self.assertEqual(other.read_bytes(), b'before\n')

    def test_wrong_shaped_and_oversized_manifests_refuse(self):
        path = self.bundle / 'manifest.json'
        for value in ([], None, False, {'schema_version': True}):
            with self.subTest(value=value):
                path.write_text(json.dumps(value))
                with self.assertRaises(INSTALL.InstallError):
                    INSTALL.check(self.repo, self.bundle)
        path.write_bytes(b' ' * 65537)
        with self.assertRaises(INSTALL.InstallError):
            INSTALL.check(self.repo, self.bundle)

    def test_content_and_executable_mode_must_both_match(self):
        self.target.chmod(0o755)
        with self.assertRaises(INSTALL.InstallError):
            INSTALL.apply(self.repo, self.bundle)
        self.assertEqual(self.target.read_bytes(), b'before\n')


if __name__ == '__main__':
    unittest.main()
