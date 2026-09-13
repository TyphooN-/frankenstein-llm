//! GPU observation and placement.
//!
//! Separates **hardware observation** (reading what is actually present via
//! sysfs) from the **placement algorithm** (deciding where a model fits).
//! The observer reads `/sys/class/drm/cardN/device/mem_info_vram_total` and
//! the device name, so a card that is absent simply does not appear. The
//! planner operates on a pure, fixture-testable input type with no physical
//! GPUs required, so it can be tested on a laptop with no ROCm.
//!
//! Domain types carry units so a `Bytes` never masquerades as a layer count.

use std::path::{Path, PathBuf};

use thiserror::Error;

/// A byte quantity. Newtype so callers cannot mix bytes with counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Bytes(pub u64);

impl Bytes {
    /// MiB constructor for readable call sites.
    pub const fn from_mib(mib: u64) -> Self {
        Self(mib * 1024 * 1024)
    }

    pub const fn as_bytes(self) -> u64 {
        self.0
    }
}

/// A single physical GPU as observed from sysfs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gpu {
    /// The kernel card number (`card0`, `card1`, ...).
    pub card: u32,
    /// The `device_name` from sysfs, e.g. `gfx1030` or `gfx1031`.
    pub device: String,
    /// Total VRAM in bytes.
    pub vram_total: Bytes,
    /// The sysfs path, retained so callers can resolve the card to a UUID or
    /// other attribute without re-scanning.
    pub path: PathBuf,
}

/// An error from observing GPU hardware.
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("cannot read sysfs card list: {0}")]
    Scan(String),
    #[error("cannot read {0}: {1}")]
    Read(PathBuf, String),
}

/// The sysfs root for DRM devices. Overridable for tests so the observer can
/// be pointed at a fixture tree without physical GPUs.
pub const DRM_ROOT: &str = "/sys/class/drm";

/// Observe the physical GPUs present under `root`.
///
/// A card that lacks a `device/mem_info_vram_total` attribute is not a render
/// device (it is a connector or a placeholder) and is skipped, mirroring how
/// `scripts/gpu_vram.py` only reports cards with VRAM.
pub fn observe(root: &Path) -> Result<Vec<Gpu>, GpuError> {
    let entries = std::fs::read_dir(root).map_err(|e| GpuError::Scan(e.to_string()))?;
    let mut gpus = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let name = entry.file_name();
        let name = match name.to_str() {
            Some(n) => n,
            None => continue,
        };
        // Card directories are named `cardN`; ignore `cardN-<mode>` connectors
        // and anything else (renderD*, card*-DP-*, etc.).
        let card = match parse_card_dir(name) {
            Some(c) => c,
            None => continue,
        };
        let device_dir = entry.path().join("device");
        let vram_path = device_dir.join("mem_info_vram_total");
        let device_path = device_dir.join("device");
        let vram = match std::fs::read_to_string(&vram_path) {
            Ok(v) => match v.trim().parse::<u64>() {
                Ok(b) => Bytes(b),
                Err(_) => continue,
            },
            Err(_) => continue,
        };
        let device = std::fs::read_to_string(&device_path)
            .map(|d| d.trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        gpus.push(Gpu {
            card,
            device,
            vram_total: vram,
            path: entry.path(),
        });
    }
    gpus.sort_by_key(|g| g.card);
    Ok(gpus)
}

/// Parse a `cardN` directory name into `N`. Returns `None` for anything that
/// is not a bare `card` plus digits (connectors, render nodes, etc.).
fn parse_card_dir(name: &str) -> Option<u32> {
    let n = name.strip_prefix("card")?;
    if n.is_empty() || !n.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    n.parse::<u32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_card_dir_accepts_bare_cards() {
        assert_eq!(parse_card_dir("card0"), Some(0));
        assert_eq!(parse_card_dir("card12"), Some(12));
        assert_eq!(parse_card_dir("card"), None);
        assert_eq!(parse_card_dir("card0-DP-1"), None);
        assert_eq!(parse_card_dir("renderD128"), None);
        assert_eq!(parse_card_dir("card-"), None);
    }

    #[test]
    fn observe_reads_a_fixture_tree() {
        let tmp = std::env::temp_dir().join(format!("frankenctl-gpu-test-{}", std::process::id()));
        let card0 = tmp.join("card0/device");
        let card1 = tmp.join("card1/device");
        std::fs::create_dir_all(&card0).unwrap();
        std::fs::create_dir_all(&card1).unwrap();
        std::fs::write(card0.join("mem_info_vram_total"), "16368215552\n").unwrap();
        std::fs::write(card0.join("device"), "1002:73bf\n").unwrap();
        std::fs::write(card1.join("mem_info_vram_total"), "30704175104\n").unwrap();
        std::fs::write(card1.join("device"), "1002:74a0\n").unwrap();

        let gpus = observe(&tmp).expect("observe should succeed");
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].card, 0);
        assert_eq!(gpus[0].vram_total, Bytes(16_368_215_552));
        assert_eq!(gpus[1].card, 1);

        // Cleanup.
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
