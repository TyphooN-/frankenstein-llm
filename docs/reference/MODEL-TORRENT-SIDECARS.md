# Model torrent sidecars

Generate a BitTorrent v1 `.torrent` and `.magnet` **beside each complete model-weight file**. This is metadata generation only: the tool has no peer client, trackers, uploads, firewall integration or model execution. Python 3.10+ on Linux is sufficient; no runtime packages are required.

## On-disk layout and scope

```text
models/Phr00tyMix-v4-32B-imat-Q6_K.gguf
models/Phr00tyMix-v4-32B-imat-Q6_K.gguf.torrent
models/Phr00tyMix-v4-32B-imat-Q6_K.gguf.magnet
models/Phr00tyMix-v4-32B-imat-Q6_K.gguf.torrent.json
```

The same layout applies to `.safetensors`, `.bin`, `.pt`, `.pth`, `.onnx` and `.ckpt`. Each shard and vision projector gets its own single-file torrent. Hidden/staging paths, partial files, zero-length files and symlinks are excluded or refused. Files must have been unmodified for at least five minutes. Metadata contains only the literal basename, byte length and SHA-1 piece hashes, not the host's absolute directory. Torrents are trackerless/DHT-compatible; they define new swarms, not automatically existing publisher swarms.

The `.torrent.json` is a local ownership/cache receipt, **not** a public license or publisher-integrity attestation. It records the local SHA-256, source device/inode/size/mtime/ctime, metainfo checksum and infohash. Unchanged source and sidecars are checked without rereading model bytes. Changed sources and damaged tool-owned sidecars are regenerated. Existing unrelated torrents/magnets/receipts are not overwritten. Remove a stale transferred ownership receipt before running the generator under a different model-root layout.

## Run or inspect

From the repository root:

```sh
python3 scripts/model-torrent-sidecars.py --root models --dry-run
python3 scripts/model-torrent-sidecars.py --root models --report proofs/torrents/latest.json
```

The first generation reads each selected weight once, serially, computing BitTorrent pieces and a local SHA-256 in the same pass. Piece buffers are capped at 16 MiB and piece counts at 65,536; the per-run file catalog is proportional to the number of files. Models are never opened for writing. Outputs use atomic, fsynced replacement. A root lock prevents overlapping generator jobs. Source changes during hashing and read errors are reported rather than published as successful torrents.

The report records per-file status and computed totals. Exit `0` means no failed/skipped selected files (or a separately reported busy lock), `1` means an error, and `2` means empty/still-settling files were skipped. The service treats `2` as a normal deferred retry; inspect the report for actual coverage.

## Hourly automation

The repository supplies `services/systemd/model-torrent-sidecars.service` and `.timer`. They are **user units**, scoped to this checkout's model tree. The job runs at low CPU and idle I/O priority, waits up to five minutes after each hourly boundary, and resumes scanning after login using the timer's persistent catch-up behavior. It creates metadata only; it never starts a torrent client.

Install the two unit files into `~/.config/systemd/user/`, reload the user manager, then enable `model-torrent-sidecars.timer`. Verify installation and the exact next run with:

```sh
systemctl --user status model-torrent-sidecars.timer
systemctl --user list-timers --all model-torrent-sidecars.timer
systemctl --user start model-torrent-sidecars.service
journalctl --user -u model-torrent-sidecars.service -n 20 --no-pager
```

Stop future scans with `systemctl --user disable --now model-torrent-sidecars.timer`. No root, paid services or new firewall rules are needed. These user units run while the user manager is running; they do not enable account lingering or guarantee execution while logged out. On a different checkout location, edit the unit's paths rather than assuming the example location applies.

## Transfer to a NAS and seed deliberately

1. Copy the model file and its matching `.torrent` to the NAS. Preserve the original filename. For sharded models, copy **all** shards, plus configuration, tokenizer/projector and other required companion files. Torrent metadata is not the model payload; copying only `.torrent` files does not move weights.
2. Before public redistribution, establish that the **exact derivative and its base** permit it and preserve any required license, NOTICE, attribution, acceptable-use or access conditions. Stored models and research-inventory license tags are not blanket redistribution permission. A single-file torrent does not carry a license/NOTICE file as additional payload. Use a compliant multi-file bundle or an appropriately licensed accompanying distribution when terms require those files to accompany the weights. Do not seed proprietary, private, gated or unverified-license artifacts merely because a sidecar exists.
3. Add the torrent to the NAS client **paused**, with the save location set to the existing weight's parent directory. Force a recheck and require **100%** before starting seeding; otherwise the client can download into the wrong destination or try to replace incomplete data.
4. Enable the NAS client's DHT/peer discovery for trackerless torrents. Publish/share the exact magnet or `.torrent` deliberately; generation does not publish or advertise it. Seeding exposes the client's public IP and model filename/content fingerprint to peers.
5. Configure the client and any forwarding/firewall rules yourself. A `.torrent` does not specify your listening port. Copy updated sidecars again if you replace a model's bytes; the infohash changes.

Runtime/source/qualification and filesystem integrity gates elsewhere in this repository remain separate from these local byte fingerprints. No model is admitted, qualified, licensed for redistribution or deployed by this feature.
