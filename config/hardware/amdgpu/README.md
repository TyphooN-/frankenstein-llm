# Current AMDGPU profiles

One current copy per card, copied byte-for-byte from `/etc/default/` using the original filenames. Refresh these files in place when the operator changes the persisted profiles; do not create dated duplicates. Older versions remain in Git history.

These files record persisted configuration, not verified effective settings or stability. Copying them does not apply GPU settings or restart services.

Verify the copies with `sha256sum -c SHA256SUMS` from this directory.
