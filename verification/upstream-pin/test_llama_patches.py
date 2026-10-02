"""Use real temporary Git trees to verify the declared llama.cpp patch policy."""
import hashlib
import json
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
VERIFY = REPO / 'scripts/verify_llama_patches.py'

class PatchPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.source = self.root / 'upstream/llama.cpp'
        self.source.mkdir(parents=True)
        self.file = self.source / 'conv.cpp'
        self.file.write_text('raw convolution\n')
        self.other = self.source / 'other.cpp'
        self.other.write_text('unchanged unrelated source\n')
        self.git('init', '-q')
        self.git('add', 'conv.cpp', 'other.cpp')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@invalid',
                 '-c', 'commit.gpgSign=false', 'commit', '-qm', 'fixture base')
        self.head = self.git('rev-parse', 'HEAD').strip()
        self.file.write_text('converted convolution\n')
        patch = self.git('diff', '--binary', 'HEAD').encode()
        self.file.write_text('raw convolution\n')
        self.patch = self.root / 'upstream/patches/0001.patch'
        self.patch.parent.mkdir()
        self.patch.write_bytes(patch)
        self.lock_path = self.root / 'upstream/llama-cpp.lock.json'
        self.lock = {'commit': self.head, 'local_patches': [{
            'path': 'upstream/patches/0001.patch',
            'sha256': hashlib.sha256(patch).hexdigest()}]}
        self.write_lock()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.source), *args], text=True)

    def write_lock(self):
        self.lock_path.write_text(json.dumps(self.lock))

    def verify(self):
        self.assertTrue(VERIFY.is_file(), 'Declared patch verifier is missing')
        return subprocess.run(['python3', str(VERIFY), '--root', str(self.root)],
                              capture_output=True, text=True, timeout=15)

    def test_required_patch_cannot_be_missing_from_clean_source(self):
        result = self.verify()
        self.assertNotEqual(0, result.returncode)
        self.assertIn('does not match declared patch', result.stderr)

    def test_exact_patch_is_accepted_without_changing_real_index_or_source(self):
        self.git('apply', str(self.patch))
        before = self.git('diff', '--binary', 'HEAD')
        index = self.git('write-tree')
        result = self.verify()
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(index, self.git('write-tree'))
        self.assertEqual(before, self.git('diff', '--binary', 'HEAD'))

    def test_extra_edit_inside_owned_file_is_rejected(self):
        self.git('apply', str(self.patch))
        self.file.write_text('converted convolution\nextra code\n')
        result = self.verify()
        self.assertNotEqual(0, result.returncode)
        self.assertIn('does not match declared patch', result.stderr)

    def test_unrelated_tracked_edit_is_rejected_and_preserved(self):
        self.git('apply', str(self.patch))
        other = self.source / 'other.cpp'
        other.write_text('user change\n')
        self.git('add', 'other.cpp')
        result = self.verify()
        self.assertNotEqual(0, result.returncode)
        self.assertIn('outside declared patch', result.stderr)
        self.assertEqual('user change\n', other.read_text())

    def test_changed_patch_bytes_are_rejected(self):
        self.patch.write_bytes(self.patch.read_bytes() + b'\n')
        result = self.verify()
        self.assertNotEqual(0, result.returncode)
        self.assertIn('SHA256 mismatch', result.stderr)

    def test_wrong_base_is_rejected(self):
        self.lock['commit'] = '0' * 40
        self.write_lock()
        result = self.verify()
        self.assertNotEqual(0, result.returncode)
        self.assertIn('HEAD does not match', result.stderr)

    def test_escaping_patch_path_is_rejected(self):
        self.lock['local_patches'][0]['path'] = '../outside.patch'
        self.write_lock()
        result = self.verify()
        self.assertNotEqual(0, result.returncode)
        self.assertIn('patch path', result.stderr)

    def test_symlinked_patch_directory_cannot_escape_manifest_scope(self):
        self.git('apply', str(self.patch))
        displaced = self.root / 'outside-patches'
        self.patch.parent.rename(displaced)
        self.patch.parent.symlink_to(displaced, target_is_directory=True)
        result = self.verify()
        self.assertNotEqual(0, result.returncode)
        self.assertIn('patch path', result.stderr)

    def test_undeclared_dirty_source_is_rejected(self):
        self.lock['local_patches'] = []
        self.write_lock()
        self.file.write_text('user change\n')
        result = self.verify()
        self.assertNotEqual(0, result.returncode)
        self.assertIn('dirty', result.stderr)

    def test_index_flags_cannot_hide_source_edits_with_or_without_patches(self):
        declared = self.lock['local_patches']
        for patched in (True, False):
            self.file.write_text('converted convolution\n' if patched else 'raw convolution\n')
            self.lock['local_patches'] = declared if patched else []
            self.write_lock()
            for flag in ('assume-unchanged', 'skip-worktree'):
                with self.subTest(patched=patched, flag=flag):
                    self.git('update-index', '--' + flag, 'other.cpp')
                    self.other.write_text('hidden unauthorized source change\n')
                    try:
                        result = self.verify()
                        self.assertNotEqual(0, result.returncode)
                        self.assertIn('visibility-suppressing index flags', result.stderr)
                    finally:
                        self.git('update-index', '--no-' + flag, 'other.cpp')
                        self.other.write_text('unchanged unrelated source\n')

    def test_no_patch_clean_source_remains_supported(self):
        self.lock['local_patches'] = []
        self.write_lock()
        result = self.verify()
        self.assertEqual(0, result.returncode, result.stderr)

if __name__ == '__main__':
    unittest.main()
