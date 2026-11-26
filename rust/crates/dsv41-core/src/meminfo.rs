//! Fast `/proc/meminfo` and `/proc/zoneinfo` parsers matching `scripts/memavail.sh`
//! and TensorFold's `native/cuda_memory.zig` (`parkedBytes` / `counting`).

use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;

const REST_DEFAULT_BYTES: u64 = 536_870_912; // 512 MiB
const GIB: f64 = 1_073_741_824.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemCounting {
    pub available_bytes: u64,
    pub parked_bytes: u64,
}

impl MemCounting {
    pub fn available_gib(&self) -> f64 {
        self.available_bytes as f64 / GIB
    }

    pub fn parked_gib(&self) -> f64 {
        self.parked_bytes as f64 / GIB
    }

    /// Formats as `"<available GiB> <parked GiB> <parked bytes>"` (one decimal on GiB),
    /// matching `scripts/memavail.sh`'s `mem_counting`.
    pub fn format_line(&self) -> String {
        format!(
            "{:.1} {:.1} {}",
            self.available_gib(),
            self.parked_gib(),
            self.parked_bytes
        )
    }
}

/// Fast, fixed-buffer read of `MemAvailable:` (in KiB) from `/proc/meminfo` (or synthetic path).
pub fn read_mem_available_kb(path: &Path, buf: &mut [u8]) -> io::Result<u64> {
    let mut f = File::open(path)?;
    let n = f.read(buf)?;
    let text = std::str::from_utf8(&buf[..n])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    parse_mem_available_kb(text)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "MemAvailable not found"))
}

pub fn parse_mem_available_kb(text: &str) -> Option<u64> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            if let Some(tok) = rest.split_whitespace().next() {
                if let Ok(kb) = tok.parse::<u64>() {
                    return Some(kb);
                }
            }
        }
    }
    None
}

pub fn parse_meminfo_bytes(text: &str) -> Option<(u64, u64)> {
    let mut total: u64 = 0;
    let mut avail: Option<u64> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            if let Some(tok) = rest.split_whitespace().next() {
                if let Ok(kb) = tok.parse::<u64>() {
                    total = kb.saturating_mul(1024);
                }
            }
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            if let Some(tok) = rest.split_whitespace().next() {
                if let Ok(kb) = tok.parse::<u64>() {
                    avail = Some(kb.saturating_mul(1024));
                }
            }
        }
    }
    avail.map(|a| (total, a))
}

pub fn parse_zoneinfo_parked_bytes(text: &str, page_size: u64) -> u64 {
    let mut listed = false;
    let mut count: u64 = 0;
    let mut rest: u64 = 0;
    let mut bad = false;

    for raw_line in text.lines() {
        // Strip trailing CR if present, matching awk `gsub(/\r/, " ", row)`
        let t = raw_line.replace('\r', " ");
        let trimmed = t.trim();
        if trimmed == "pagesets" {
            listed = true;
            continue;
        }
        if listed && !raw_line.starts_with("    ") && !raw_line.starts_with("  vm stats") {
            listed = false;
            continue;
        }
        if !listed {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let Some(key) = parts.next() else {
            continue;
        };
        if key != "count:" && key != "high_min:" {
            continue;
        }
        let Some(val_s) = parts.next() else {
            bad = true;
            continue;
        };
        if !val_s.bytes().all(|b| b.is_ascii_digit()) {
            bad = true;
            continue;
        }
        match val_s.parse::<u64>() {
            Ok(v) => {
                if key == "count:" {
                    count = count.saturating_add(v);
                } else {
                    rest = rest.saturating_add(v);
                }
            }
            Err(_) => {
                bad = true;
            }
        }
    }

    if bad || count == 0 {
        return 0;
    }
    let resting = if rest > 0 {
        rest.saturating_mul(page_size)
    } else {
        REST_DEFAULT_BYTES
    };
    count.saturating_mul(page_size).saturating_sub(resting)
}

pub fn system_page_size() -> u64 {
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if ps > 0 {
        ps as u64
    } else {
        4096
    }
}

/// Reads meminfo first, then zoneinfo right after (pages draining in between are never double-counted).
pub fn mem_counting(
    meminfo_path: &Path,
    zoneinfo_path: Option<&Path>,
    page_size: Option<u64>,
) -> io::Result<MemCounting> {
    let mi_text = fs::read_to_string(meminfo_path)?;
    let (total, avail) = parse_meminfo_bytes(&mi_text)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "MemAvailable missing"))?;
    let pg = page_size.unwrap_or_else(system_page_size);
    let parked = match zoneinfo_path {
        Some(zp) => match fs::read_to_string(zp) {
            Ok(zi_text) => parse_zoneinfo_parked_bytes(&zi_text, pg),
            Err(_) => 0,
        },
        None => 0,
    };
    let mut a = avail.saturating_add(parked);
    if total > 0 && a > total {
        a = total;
    }
    Ok(MemCounting {
        available_bytes: a,
        parked_bytes: parked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mem_counting_with_parked_pages() {
        let mi = "MemTotal: 126877696 kB\nMemAvailable: 10485760 kB\n";
        let zi = "\
Node 0, zone Normal
  pagesets
    cpu: 0
              count: 262144
              high_min: 1024
  vm stats threshold: 125
    cpu: 1
              count: 262144
              high_min: 1024
  node_unreclaimable: 0
";
        let (total, avail) = parse_meminfo_bytes(mi).unwrap();
        let parked = parse_zoneinfo_parked_bytes(zi, 4096);
        assert_eq!(total, 126_877_696 * 1024);
        assert_eq!(avail, 10_485_760 * 1024);
        assert_eq!(parked, (524_288 - 2048) * 4096);
        let mc = MemCounting {
            available_bytes: avail + parked,
            parked_bytes: parked,
        };
        assert_eq!(mc.format_line(), "12.0 2.0 2139095040");
    }

    #[test]
    fn test_zoneinfo_fallback_rest_default() {
        let zi = "  pagesets\n    cpu: 0\n      count: 262144\n";
        // 262144 * 4096 = 1 GiB; minus 512 MiB default = 512 MiB
        assert_eq!(parse_zoneinfo_parked_bytes(zi, 4096), 536_870_912);
    }

    #[test]
    fn test_zoneinfo_malformed_counts_zero() {
        let zi = "  pagesets\n    cpu: 0\n      count: oops\n      high_min: 10\n";
        assert_eq!(parse_zoneinfo_parked_bytes(zi, 4096), 0);
    }
}
