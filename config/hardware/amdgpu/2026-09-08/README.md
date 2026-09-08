# AMDGPU profile backup

> **Superseded 2026-09-08T19:24-0400.** `card0` and `card1` were retuned
> after this capture; the current bytes are in
> [`../2026-09-08T1924-0400/`](../2026-09-08T1924-0400/README.md). The
> checksums and offsets below remain correct for the 12:37/13:03 state and
> are kept as the record of it.

Captured from all files matching `/etc/default/amdgpu*` at
2026-09-08T13:03:14-04:00. All three copies matched their source SHA-256.

The operator reports these configurations are now stable. This is an
operator-reported baseline, not new stability or performance qualification.
No GPU settings were applied, no services restarted, and no tests of GPU
stability were performed during this backup.

| File | Description in source | Configured voltage offset | Power cap |
| --- | --- | --- | --- |
| `amdgpu-custom-state.card0` | ASRock Phantom 6900 XT | -55 mV | 323000000 microwatts |
| `amdgpu-custom-state.card1` | Radeon Pro V620 | -140 mV | 250000000 microwatts |
| `amdgpu-custom-state.card2` | 6900 XT Liquid Devil | -90 mV | 346000000 microwatts |

Files are preserved byte-for-byte, including crash notes and older placeholder
comments. In particular, card1's header still describes an unqualified zero-offset
placeholder, but its actual configured offset is now -140 mV. This backup does
not silently correct those historical comments.

Verify copies with `sha256sum -c SHA256SUMS` from this directory. Before restoring,
verify PCI identities and current card numbering; do not apply a profile by
assuming that DRM numbering survived a reboot or hardware change. Installation
and application are separate operations; this directory is archival only.
