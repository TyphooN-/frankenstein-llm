# AMDGPU profile backup — 2026-09-08T19:24-0400

Byte-for-byte copies of every file matching `/etc/default/amdgpu*`, taken at
2026-09-08T19:24:41-04:00. Each file was hashed at the source, copied, then both
source and destination were re-hashed and compared, so a mid-capture edit would
have failed the capture rather than produced a silently mixed copy. All three
destinations matched their source SHA-256.

**Nothing was applied and nothing was written outside this directory.** No GPU
setting was changed, no service was restarted, no tuning helper was run, and no
stability or performance test was performed. This directory is archival.

| File | Description in source | Configured voltage offset | Power cap | Changed since the 12:37/13:03 archive? |
| --- | --- | --- | --- | --- |
| `amdgpu-custom-state.card0` | ASRock Phantom 6900 XT | -45 mV | 323000000 microwatts | yes, was -55 mV |
| `amdgpu-custom-state.card1` | Radeon Pro V620 | -125 mV | 250000000 microwatts | yes, was -140 mV |
| `amdgpu-custom-state.card2` | 6900 XT Liquid Devil | -90 mV | 346000000 microwatts | no, identical bytes |

Source modification times at capture: card0 2026-09-08T15:38:42-0400, card1
2026-09-08T16:15:42-0400, card2 2026-09-08T08:44:30-0400.

## What changed, and what that does and does not mean

Both changed files moved **toward** their nominal voltage, and both gained a
crash annotation immediately above the new value: card0 now reads
`#-50mV crash` above `-45mV`, card1 `#-130mV crash` above `-125mV`. Those
comments are the operator's own record that the previous, more aggressive
offsets crashed.

Read that as a live tuning history, not as a verdict about the host. It does not
establish that the current values are stable, and this repository has no
qualification evidence for any of them. Gate results recorded while the host was
faulting describe the host, not the model under test — see
[a gate fails while the kernel is faulting](../../../../docs/reference/TROUBLESHOOTING.md#a-gate-fails-while-the-kernel-is-faulting).
Do not treat this capture as a stability claim in either direction.

The card1 header still describes an unqualified zero-offset placeholder while
the file configures -125 mV. That contradiction is preserved deliberately:
these are byte-for-byte copies, including stale comments.

## Using this directory

```bash
sha256sum -c SHA256SUMS   # from inside this directory
```

The previous capture is kept at [`../2026-09-08/`](../2026-09-08/README.md) and
remains accurate for the state it recorded. Keep both: the pair is the only
record that card0 and card1 were retuned between 12:37 and 16:15 on 2026-09-08.

Before restoring anything, re-verify PCI identities and current DRM card
numbering; this host has already come up with a card missing and the rest
renumbered. Installing a profile is not applying it, and applying one is an
operator decision that is out of scope for this repository's automation.
