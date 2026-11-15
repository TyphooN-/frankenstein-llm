# Current AMDGPU profiles

One current copy per PCI address, copied byte-for-byte from `/etc/default/` on 2026-11-12 using the original filenames:

- `amdgpu-custom-state.pci:0000:03:00.0`
- `amdgpu-custom-state.pci:0000:07:00.0`
- `amdgpu-custom-state.pci:0000:0a:00.0`

Refresh these files in place when the operator changes the persisted profiles; do not create dated duplicates. The superseded `amdgpu-custom-state.card0`, `.card1`, and `.card2` copies were removed. Older versions remain in Git history. PCI-keyed filenames avoid depending on DRM card numbers, which can change after a reboot.

These files record persisted configuration, not verified effective settings or stability. Copying them does not apply GPU settings or restart services.

The 2026-11-12 capture declares:

| PCI address | GPU | Maximum SCLK | Maximum MCLK | Voltage offset | Power cap | Performance level |
|---|---|---|---|---|---|---|
| `0000:03:00.0` | ASRock Phantom 6900 XT | 2640 MHz | 1000 MHz | -76 mV | 323 W | `manual` |
| `0000:07:00.0` | Radeon Pro V620 | not declared | not declared | -68 mV | 250 W | `auto` |
| `0000:0a:00.0` | Liquid Devil 6900 XT | 2710 MHz | 1000 MHz | -56 mV | 346 W | `manual` |

The V620 export retains its existing warning comments alongside the requested negative offset. That value is operator configuration intent, not proof that the driver supports or applies it. Profile comments and recorded fault history are preserved verbatim; none of these settings is newly qualified by this capture.

Verify the copies with `sha256sum -c SHA256SUMS` from this directory.
