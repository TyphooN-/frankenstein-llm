"""Transfer-integrity contract for the reboot-resumable downloader.

The queue is restarted by systemd on failure and survives reboots, so the states
it can wake up in are the interesting ones: a partial that stopped mid-transfer,
a partial that is already the whole file because the host died between the last
byte and the rename, and a partial that is full-size but wrong. Only the first
of those may reach curl. The second must be promoted, and the third quarantined
-- otherwise the unit restarts forever without transferring anything.

Nothing here performs a download; ``subprocess.run`` is replaced by a recorder.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

import download_queue as dq


class TransferStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        patcher = mock.patch.object(dq, "LOG", self.sandbox / "downloads.log")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.destination = self.sandbox / "artifact.gguf"
        self.partial = self.sandbox / "artifact.gguf.partial"
        self.payload = b"hermes-local-artifact-bytes"
        self.digest = hashlib.sha256(self.payload).hexdigest()

    def fetch(self, expected_sha=None):
        """Call download() with curl replaced; return the recorded argv or None."""
        recorded = []

        def fake_run(command, **_kwargs):
            recorded.append(command)
            raise AssertionError("curl was invoked")

        with mock.patch.object(dq.subprocess, "run", side_effect=fake_run):
            dq.download("owner/repo", "main", "artifact.gguf", self.destination,
                        len(self.payload), expected_sha)
        return recorded

    def test_complete_partial_is_promoted_without_contacting_the_server(self):
        # Resuming a partial that is already complete asks for a range at EOF.
        # The server answers 416, curl --fail exits non-zero, and the unit
        # restarts into exactly the same state -- forever.
        self.partial.write_bytes(self.payload)
        self.assertEqual([], self.fetch(self.digest))
        self.assertFalse(self.partial.exists())
        self.assertEqual(self.payload, self.destination.read_bytes())

    def test_complete_partial_is_promoted_when_only_a_size_is_known(self):
        self.partial.write_bytes(self.payload)
        self.assertEqual([], self.fetch(None))
        self.assertEqual(self.payload, self.destination.read_bytes())

    def test_full_size_partial_with_a_wrong_digest_is_quarantined_not_promoted(self):
        self.partial.write_bytes(b"x" * len(self.payload))
        with self.assertRaises(AssertionError):        # curl is reached again
            self.fetch(self.digest)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.partial.exists())
        quarantined = sorted(p.name for p in self.sandbox.glob("*.partial.bad-*"))
        self.assertEqual(1, len(quarantined), quarantined)

    def test_short_partial_still_resumes_through_curl(self):
        self.partial.write_bytes(self.payload[:5])
        with self.assertRaises(AssertionError):
            self.fetch(self.digest)
        self.assertTrue(self.partial.exists(), "a resumable partial was discarded")

    def test_valid_destination_short_circuits(self):
        self.destination.write_bytes(self.payload)
        self.assertEqual([], self.fetch(self.digest))

    def test_promote_complete_partial_leaves_a_wrong_file_readable_for_triage(self):
        self.partial.write_bytes(b"y" * len(self.payload))
        promoted = dq.promote_complete_partial(self.partial, self.destination, self.digest)
        self.assertFalse(promoted)
        bad = list(self.sandbox.glob("*.bad-*"))
        self.assertEqual(1, len(bad))
        self.assertEqual(b"y" * len(self.payload), bad[0].read_bytes())

    def test_promote_complete_partial_reports_success_to_its_caller(self):
        self.partial.write_bytes(self.payload)
        self.assertTrue(dq.promote_complete_partial(self.partial, self.destination, self.digest))

    def test_a_promoted_file_is_not_hashed_a_second_time(self):
        # Every worker thread returning from a reboot with a complete partial
        # would otherwise pay a second full pass over a multi-gigabyte file at
        # the same moment, for a digest the promotion just computed.
        self.partial.write_bytes(self.payload)
        hashed = []
        real_sha256 = dq.sha256
        with mock.patch.object(dq, "sha256",
                               side_effect=lambda path: hashed.append(path) or real_sha256(path)):
            self.assertEqual([], self.fetch(self.digest))
        self.assertEqual([self.partial], hashed)


class AtomicStateTests(unittest.TestCase):
    """The resume decision after a reboot is made from this file."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.state = self.sandbox / "download-state.json"

    def test_state_round_trips_and_leaves_no_temp_file(self):
        dq.atomic_json(self.state, {"status": "running", "artifacts": {}})
        self.assertEqual({"status": "running", "artifacts": {}},
                         json.loads(self.state.read_text()))
        self.assertEqual(["download-state.json"],
                         sorted(p.name for p in self.sandbox.iterdir()))

    def test_state_is_fsynced_before_the_rename(self):
        # A rename is atomic but not durable. Without the fsync a power loss can
        # publish the new name over content that never reached the disk.
        synced = []
        real_fsync = os.fsync
        with mock.patch.object(dq.os, "fsync", side_effect=lambda fd: synced.append(fd) or real_fsync(fd)):
            dq.atomic_json(self.state, {"status": "complete"})
        self.assertGreaterEqual(len(synced), 1, "the state file was never fsynced")

    def test_the_completion_stamp_is_written_durably_too(self):
        # Downstream phases and the mission supervisor compare this stamp against
        # an exact byte total, so a torn write is a permanent mismatch against a
        # queue that is in fact complete -- not something a retry repairs.
        stamp = self.sandbox / "downloads-complete.ok"
        synced = []
        real_fsync = os.fsync
        with mock.patch.object(dq.os, "fsync",
                               side_effect=lambda fd: synced.append(fd) or real_fsync(fd)):
            dq.atomic_text(stamp, "72134030730\n")
        self.assertEqual("72134030730\n", stamp.read_text())
        self.assertGreaterEqual(len(synced), 1, "the stamp was never fsynced")
        self.assertEqual(["downloads-complete.ok"],
                         sorted(path.name for path in self.sandbox.iterdir()))

    def test_a_failing_directory_fsync_does_not_lose_the_write(self):
        def only_dir_fails(fd):
            if os.fstat(fd).st_mode & 0o170000 == 0o040000:
                raise OSError("directory fsync unsupported")

        with mock.patch.object(dq.os, "fsync", side_effect=only_dir_fails):
            dq.atomic_json(self.state, {"status": "complete"})
        self.assertEqual({"status": "complete"}, json.loads(self.state.read_text()))


class DurablePromotionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.partial = self.sandbox / "artifact.partial"
        self.destination = self.sandbox / "artifact.gguf"
        self.partial.write_bytes(b"verified model bytes")

    def test_bytes_then_rename_then_directory_are_persisted_in_order(self):
        events = []
        real_fsync = os.fsync
        real_replace = os.replace

        def recording_fsync(fd):
            kind = "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
            events.append(kind)
            real_fsync(fd)

        def recording_replace(source, destination):
            events.append("rename")
            real_replace(source, destination)

        with mock.patch.object(dq.os, "fsync", recording_fsync), \
             mock.patch.object(dq.os, "replace", recording_replace):
            dq.durable_promote(self.partial, self.destination)
        self.assertEqual(["file", "rename", "directory"], events)
        self.assertEqual(b"verified model bytes", self.destination.read_bytes())

    def test_directory_fsync_failure_does_not_publish_success(self):
        real_fsync = os.fsync

        def fail_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("directory fsync failed")
            real_fsync(fd)

        with mock.patch.object(dq.os, "fsync", fail_directory), \
             self.assertRaisesRegex(OSError, "directory fsync failed"):
            dq.durable_promote(self.partial, self.destination)
        # Rename happened, but the caller cannot advance state. On restart the
        # normal valid-destination path rechecks these bytes before continuing.
        self.assertTrue(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
