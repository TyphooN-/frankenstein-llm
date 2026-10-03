# Current AMDGPU profiles

One current copy per PCI address, copied byte-for-byte from `/etc/default/` on 2026-10-03 using the original filenames:

- `amdgpu-custom-state.pci:0000:03:00.0`
- `amdgpu-custom-state.pci:0000:07:00.0`
- `amdgpu-custom-state.pci:0000:0a:00.0`

Refresh these files in place when the operator changes the persisted profiles; do not create dated duplicates. The superseded `amdgpu-custom-state.card0`, `.card1`, and `.card2` copies were removed. Older versions remain in Git history. PCI-keyed filenames avoid depending on DRM card numbers, which can change after a reboot.

These files record persisted configuration, not verified effective settings or stability. Copying them does not apply GPU settings or restart services.

Verify the copies with `sha256sum -c SHA256SUMS` from this directory.
