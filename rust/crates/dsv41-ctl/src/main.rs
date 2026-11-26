//! High-speed native Rust control & preparation utility (`dsv41-ctl`).
//!
//! Replaces all Python and awk helpers used by `start.sh`, `stop.sh`, `scripts/prepare.sh`,
//! `scripts/nodes.sh`, `scripts/config.sh`, and `scripts/port.sh`:
//! - `zig-flags`: validate serve flags against `tensorfold-native capabilities --json` on stdin
//! - `ablit-stock-dtypes`: restore stock safetensors dtypes without loading PyTorch
//! - `memavail`: compute MemAvailable + per-CPU parked pages (`/proc/meminfo` + `/proc/zoneinfo`)
//! - `make-bias-vl`: extract `*.ffn.gate.bias_vl` tensors byte-for-byte into one safetensors file
//! - `cidr-network` / `cidr-contains`: IPv4 interface/network math without Python `ipaddress`
//! - `gen-api-key`: generate `sk-tf-<token>` via OS CSPRNG
//! - `check-smoke`: validate smoke-test chat completion JSON on stdin
//! - `health-busy`: extract live connection / running request count from `/health` JSON on stdin
//! - `json-field`: read/validate a top-level field from a JSON file
//! - `fsops`: spill directory file operations (`list`, `corrupt`, `truncate`, `remove`, `statvfs`)

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use clap::{Parser, Subcommand};
use dsv41_core::meminfo::mem_counting;
use dsv41_core::rank_dir::execute_fsop;
use dsv41_core::safetensors::{is_pure_cast, mmap_file, read_header, same_file, write_safetensors};
use dsv41_core::sha256_hex;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::{self, Read};
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};
use std::process;

#[derive(Parser)]
#[command(
    name = "dsv41-ctl",
    version,
    about = "Fast native Rust control helpers for DeepSeek-V4.1-Flash on 2x DGX Sparks"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Check serve arguments against `tensorfold-native capabilities --json` on stdin
    ZigFlags {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Restore stock dtypes on an ABLIT overlay checkpoint without loading PyTorch
    AblitStockDtypes {
        stock_dir: PathBuf,
        ablit_dir: PathBuf,
    },
    /// Compute MemAvailable + per-CPU parked pages ("<avail GiB> <parked GiB> <parked bytes>")
    Memavail {
        #[arg(default_value = "/proc/meminfo")]
        meminfo: PathBuf,
        #[arg(default_value = "/proc/zoneinfo")]
        zoneinfo: PathBuf,
        pagesize: Option<u64>,
    },
    /// Extract *.ffn.gate.bias_vl tensors byte-for-byte from a checkpoint into one safetensors file
    MakeBiasVl { model_dir: PathBuf, out: PathBuf },
    /// Print the IPv4 network CIDR for an interface address (e.g. 192.168.100.10/24 -> 192.168.100.0/24)
    CidrNetwork { iface_cidr: String },
    /// Exit 0 if IPv4 interface/IP belongs to the given IPv4 network CIDR, else exit 1
    CidrContains {
        iface_cidr: String,
        net_cidr: String,
    },
    /// Generate a cryptographically random `sk-tf-...` API key
    GenApiKey,
    /// Validate a chat completion smoke-test response JSON on stdin and print a summary
    CheckSmoke,
    /// Print live connections / running requests count from `/health` JSON on stdin
    HealthBusy,
    /// Read a top-level field from a JSON file (or `-` for stdin), optionally asserting its value
    JsonField {
        file: PathBuf,
        key: String,
        #[arg(long)]
        expect_int: Option<i64>,
        #[arg(long)]
        expect_str: Option<String>,
    },
    /// Read `<name> <rc>` summary lines on stdin and write `summary.json` for `tools/gpu_validate.sh`
    SummaryJson { out: PathBuf },
    /// Execute one spill-directory filesystem operation specified as a JSON object
    Fsops { json_arg: String },
}

fn cmd_zig_flags(args: &[String]) -> i32 {
    let mut input = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut input) {
        eprintln!("tensorfold serve: error: failed to read capabilities JSON: {e}");
        return 1;
    }
    let doc: Value = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("tensorfold serve: error: invalid capabilities JSON: {e}");
            return 1;
        }
    };
    let Some(flags) = doc
        .get("serve")
        .and_then(|s| s.get("flags"))
        .and_then(|f| f.as_object())
    else {
        eprintln!("tensorfold serve: error: missing serve.flags in capabilities JSON");
        return 1;
    };

    let value_flags: HashSet<&str> = [
        "--host",
        "--port",
        "--name",
        "--context",
        "--parallel",
        "--max-tokens",
        "--drafter",
        "--tp",
        "--rank",
        "--master",
        "--master-port",
        "--prompt-cache-gib",
        "--backend",
        "--temperature",
        "--top-p",
        "--top-k",
        "--min-p",
        "--reasoning-effort",
        "--thinking-budget",
        "--alias",
        "--api-key",
        "--api-key-file",
        "--chat-template",
        "--device",
        "--spill-dir",
        "--spill-gib",
        "--spill-min-free-gib",
        "--spill-min-tokens",
        "--spill-quota",
    ]
    .into_iter()
    .collect();

    let mut errors = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        i += 1;
        if !a.starts_with("--") {
            continue;
        }
        let (name, eq, mut value) = match a.split_once('=') {
            Some((n, v)) => (n, true, v.to_string()),
            None => (a.as_str(), false, String::new()),
        };
        let Some(spec) = flags.get(name) else {
            errors.push(format!(
                "{name} is not served by this tensorfold-native yet"
            ));
            continue;
        };
        let values: Option<Vec<&str>> = spec
            .get("values")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_str()).collect());
        let takes_value = eq || values.is_some() || value_flags.contains(name);
        if takes_value && !eq {
            if i < args.len() {
                value = args[i].clone();
                i += 1;
            } else {
                value.clear();
            }
        }
        if let Some(allowed) = values {
            if !allowed.iter().any(|&v| v == value) {
                errors.push(format!(
                    "{name} takes one of {}, not {value}",
                    allowed.join(", ")
                ));
            }
        }
    }
    if !errors.is_empty() {
        eprintln!("tensorfold serve: error: {}", errors.join("; "));
        return 1;
    }
    0
}

fn cmd_ablit_stock_dtypes(stock_dir: &Path, ablit_dir: &Path) -> i32 {
    let idx_path = ablit_dir.join("model.safetensors.index.json");
    let idx_text = match fs::read_to_string(&idx_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {}: {e}", idx_path.display());
            return 1;
        }
    };
    let idx: Value = match serde_json::from_str(&idx_text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("invalid JSON in {}: {e}", idx_path.display());
            return 1;
        }
    };
    let Some(wmap) = idx.get("weight_map").and_then(|v| v.as_object()) else {
        eprintln!("missing weight_map in {}", idx_path.display());
        return 1;
    };
    let shards: BTreeSet<String> = wmap
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    let mut given = 0usize;
    for shard_name in shards {
        let s_path = stock_dir.join(&shard_name);
        let a_path = ablit_dir.join(&shard_name);
        if same_file(&s_path, &a_path) {
            continue;
        }
        let hs = match read_header(&s_path) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("{shard_name}: cannot read stock header: {e}");
                return 1;
            }
        };
        let ha = match read_header(&a_path) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("{shard_name}: cannot read ablit header: {e}");
                return 1;
            }
        };
        let cast_keys: Vec<String> = ha
            .tensors
            .iter()
            .filter_map(|(k, ma)| {
                hs.tensors
                    .get(k)
                    .filter(|ms| ms.dtype != ma.dtype)
                    .map(|_| k.clone())
            })
            .collect();
        if cast_keys.is_empty() {
            continue;
        }

        let s_map = match mmap_file(&s_path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{shard_name}: cannot mmap stock shard: {e}");
                return 1;
            }
        };
        let a_map = match mmap_file(&a_path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{shard_name}: cannot mmap ablit shard: {e}");
                return 1;
            }
        };

        let s_base = hs.data_start as usize;
        let a_base = ha.data_start as usize;
        let cast_set: HashSet<&str> = cast_keys.iter().map(String::as_str).collect();

        for k in &cast_keys {
            let ms = &hs.tensors[k];
            let ma = &ha.tensors[k];
            let s_bytes = &s_map[s_base + ms.data_offsets[0]..s_base + ms.data_offsets[1]];
            let a_bytes = &a_map[a_base + ma.data_offsets[0]..a_base + ma.data_offsets[1]];
            let shape_ok = ms.shape == ma.shape;
            let cast_ok =
                shape_ok && is_pure_cast(&ms.dtype, s_bytes, &ma.dtype, a_bytes).unwrap_or(false);
            if !cast_ok {
                eprintln!(
                    "{shard_name}: {k} is {} {:?}, stock {} {:?}, and not a cast of stock's values",
                    ma.dtype, ma.shape, ms.dtype, ms.shape
                );
                return 1;
            }
            println!("{k}: {} -> {} (stock's bytes)", ma.dtype, ms.dtype);
        }

        let mut out_tensors: Vec<(&str, &str, &[usize], &[u8])> =
            Vec::with_capacity(ha.tensors.len());
        for (k, ma) in &ha.tensors {
            if cast_set.contains(k.as_str()) {
                let ms = &hs.tensors[k];
                let bytes = &s_map[s_base + ms.data_offsets[0]..s_base + ms.data_offsets[1]];
                out_tensors.push((k.as_str(), ms.dtype.as_str(), &ms.shape, bytes));
            } else {
                let bytes = &a_map[a_base + ma.data_offsets[0]..a_base + ma.data_offsets[1]];
                out_tensors.push((k.as_str(), ma.dtype.as_str(), &ma.shape, bytes));
            }
        }
        if let Err(e) = write_safetensors(&a_path, ha.metadata.as_ref(), &out_tensors) {
            eprintln!("{shard_name}: failed to write restored shard: {e}");
            return 1;
        }
        given += cast_keys.len();
    }

    if given == 0 {
        println!("every tensor already has its stock dtype");
    }
    0
}

fn cmd_memavail(meminfo: &Path, zoneinfo: &Path, pagesize: Option<u64>) -> i32 {
    let zi = if zoneinfo.exists() {
        Some(zoneinfo)
    } else {
        None
    };
    match mem_counting(meminfo, zi, pagesize) {
        Ok(mc) => {
            println!("{}", mc.format_line());
            0
        }
        Err(_) => 1,
    }
}

fn cmd_make_bias_vl(model_dir: &Path, out: &Path) -> i32 {
    let idx_path = model_dir.join("model.safetensors.index.json");
    let idx_text = match fs::read_to_string(&idx_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {}: {e}", idx_path.display());
            return 1;
        }
    };
    let idx: Value = match serde_json::from_str(&idx_text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("invalid JSON in {}: {e}", idx_path.display());
            return 1;
        }
    };
    let Some(wmap) = idx.get("weight_map").and_then(|v| v.as_object()) else {
        eprintln!("missing weight_map");
        return 1;
    };
    let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut names: Vec<String> = wmap
        .keys()
        .filter(|k| k.ends_with(".ffn.gate.bias_vl"))
        .cloned()
        .collect();
    names.sort();
    if names.is_empty() {
        eprintln!("no *.ffn.gate.bias_vl tensors in the index: is MODEL_DIR the original (not EXL3) checkpoint?");
        return 1;
    }
    for k in &names {
        if let Some(shard) = wmap.get(k).and_then(|v| v.as_str()) {
            by_shard
                .entry(shard.to_string())
                .or_default()
                .push(k.clone());
        }
    }

    let mut extracted: BTreeMap<String, (String, Vec<usize>, Vec<u8>)> = BTreeMap::new();
    for (shard, ks) in by_shard {
        let sp = model_dir.join(&shard);
        let hdr = match read_header(&sp) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("{shard}: {e}");
                return 1;
            }
        };
        let map = match mmap_file(&sp) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{shard}: {e}");
                return 1;
            }
        };
        let base = hdr.data_start as usize;
        for k in ks {
            let Some(tm) = hdr.tensors.get(&k) else {
                eprintln!("{shard}: missing tensor {k}");
                return 1;
            };
            let raw = map[base + tm.data_offsets[0]..base + tm.data_offsets[1]].to_vec();
            extracted.insert(k, (tm.dtype.clone(), tm.shape.clone(), raw));
        }
    }

    if let Some(parent) = out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut meta = BTreeMap::new();
    meta.insert(
        "source".to_string(),
        "deepseek-ai/DeepSeek-V4.1-Flash (MIT): the ffn.gate.bias_vl tensors, byte for byte"
            .to_string(),
    );
    let specs: Vec<(&str, &str, &[usize], &[u8])> = extracted
        .iter()
        .map(|(k, (dt, sh, raw))| (k.as_str(), dt.as_str(), sh.as_slice(), raw.as_slice()))
        .collect();
    if let Err(e) = write_safetensors(out, Some(&meta), &specs) {
        eprintln!("cannot write {}: {e}", out.display());
        return 1;
    }
    if let Ok(bytes) = fs::read(out) {
        println!("{}  {}", sha256_hex(&bytes), out.display());
    }
    0
}

fn parse_ipv4_cidr(s: &str) -> Result<(u32, u8), String> {
    let (ip_s, pfx_s) = match s.split_once('/') {
        Some((a, b)) => (a, Some(b)),
        None => (s, None),
    };
    let ip: Ipv4Addr = ip_s
        .parse()
        .map_err(|_| format!("invalid IPv4 address: {ip_s}"))?;
    let pfx: u8 = match pfx_s {
        Some(p) => p
            .parse::<u8>()
            .ok()
            .filter(|&v| v <= 32)
            .ok_or_else(|| format!("invalid prefix length: {p}"))?,
        None => 32,
    };
    Ok((u32::from(ip), pfx))
}

fn netmask(pfx: u8) -> u32 {
    if pfx == 0 {
        0
    } else {
        !0u32 << (32 - pfx)
    }
}

fn cmd_cidr_network(iface_cidr: &str) -> i32 {
    match parse_ipv4_cidr(iface_cidr) {
        Ok((ip, pfx)) => {
            let net = Ipv4Addr::from(ip & netmask(pfx));
            println!("{net}/{pfx}");
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

fn cmd_cidr_contains(iface_cidr: &str, net_cidr: &str) -> i32 {
    let Ok((ip, _)) = parse_ipv4_cidr(iface_cidr) else {
        return 1;
    };
    let Ok((net_ip, net_pfx)) = parse_ipv4_cidr(net_cidr) else {
        return 1;
    };
    let mask = netmask(net_pfx);
    if (ip & mask) == (net_ip & mask) {
        0
    } else {
        1
    }
}

fn cmd_gen_api_key() -> i32 {
    let mut bytes = [0u8; 32];
    if let Err(e) = getrandom::getrandom(&mut bytes) {
        eprintln!("getrandom failed: {e}");
        return 1;
    }
    println!("sk-tf-{}", URL_SAFE_NO_PAD.encode(bytes));
    0
}

fn cmd_check_smoke() -> i32 {
    let mut input = String::new();
    if io::stdin().read_to_string(&mut input).is_err() {
        return 1;
    }
    let Ok(r) = serde_json::from_str::<Value>(&input) else {
        return 1;
    };
    let content = r
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return 1;
    }
    let uniq: HashSet<char> = trimmed.chars().collect();
    if trimmed.chars().count() >= 8 && uniq.len() <= 2 {
        eprintln!("one repeated character");
        return 1;
    }
    let Some(toks) = r
        .get("usage")
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|v| v.as_u64())
    else {
        return 1;
    };
    let preview: String = trimmed.chars().take(40).collect();
    println!("{preview:?}, {toks} tokens");
    0
}

fn cmd_health_busy() -> i32 {
    let mut input = String::new();
    if io::stdin().read_to_string(&mut input).is_err() {
        println!("0");
        return 0;
    }
    let Ok(d) = serde_json::from_str::<Value>(&input) else {
        println!("0");
        return 0;
    };
    let busy = d
        .get("live")
        .and_then(|l| l.get("connections"))
        .and_then(|v| v.as_u64())
        .or_else(|| d.get("requests_running").and_then(|v| v.as_u64()))
        .unwrap_or(0);
    println!("{busy}");
    0
}

fn cmd_json_field(
    file: &Path,
    key: &str,
    expect_int: Option<i64>,
    expect_str: Option<&str>,
) -> i32 {
    let text = if file == Path::new("-") {
        let mut s = String::new();
        if io::stdin().read_to_string(&mut s).is_err() {
            eprintln!("cannot read stdin");
            return 1;
        }
        s
    } else {
        match fs::read_to_string(file) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("cannot read {}", file.display());
                return 1;
            }
        }
    };
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
        eprintln!("invalid JSON in {}", file.display());
        return 1;
    };
    let Some(val) = v.get(key) else {
        eprintln!("missing key {key} in {}", file.display());
        return 1;
    };
    if let Some(expected) = expect_int {
        if val.as_i64() != Some(expected) {
            eprintln!("expected {key} == {expected}, got {val}");
            return 1;
        }
        return 0;
    }
    if let Some(expected) = expect_str {
        if val.as_str() != Some(expected) {
            eprintln!("expected {key} == {expected:?}, got {val}");
            return 1;
        }
        return 0;
    }
    match val {
        Value::String(s) => println!("{s}"),
        other => println!("{other}"),
    }
    0
}

fn cmd_summary_json(out: &Path) -> i32 {
    let mut input = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut input) {
        eprintln!("cannot read stdin: {e}");
        return 1;
    }
    let mut suites = serde_json::Map::new();
    let mut failed = Vec::new();
    for line in input.lines() {
        let mut parts = line.split_whitespace();
        let (Some(name), Some(code)) = (parts.next(), parts.next()) else {
            continue;
        };
        let val = if let Ok(n) = code.parse::<i64>() {
            Value::Number(n.into())
        } else {
            Value::String(code.to_string())
        };
        suites.insert(name.to_string(), val);
        if code != "0" && code != "dry-run" {
            failed.push(Value::String(name.to_string()));
        }
    }
    let doc = serde_json::json!({"suites": suites, "failed": failed});
    let text = serde_json::to_string_pretty(&doc).unwrap_or_default() + "\n";
    if let Err(e) = fs::write(out, text) {
        eprintln!("cannot write {}: {e}", out.display());
        return 1;
    }
    0
}

fn cmd_fsops(json_arg: &str) -> i32 {
    let req: Value = match serde_json::from_str(json_arg) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("invalid fsops json: {e}");
            return 1;
        }
    };
    match execute_fsop(&req) {
        Ok(res) => {
            println!("{}", serde_json::to_string(&res).unwrap_or_default());
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let code = match cli.cmd {
        Cmd::ZigFlags { args } => cmd_zig_flags(&args),
        Cmd::AblitStockDtypes {
            stock_dir,
            ablit_dir,
        } => cmd_ablit_stock_dtypes(&stock_dir, &ablit_dir),
        Cmd::Memavail {
            meminfo,
            zoneinfo,
            pagesize,
        } => cmd_memavail(&meminfo, &zoneinfo, pagesize),
        Cmd::MakeBiasVl { model_dir, out } => cmd_make_bias_vl(&model_dir, &out),
        Cmd::CidrNetwork { iface_cidr } => cmd_cidr_network(&iface_cidr),
        Cmd::CidrContains {
            iface_cidr,
            net_cidr,
        } => cmd_cidr_contains(&iface_cidr, &net_cidr),
        Cmd::GenApiKey => cmd_gen_api_key(),
        Cmd::CheckSmoke => cmd_check_smoke(),
        Cmd::HealthBusy => cmd_health_busy(),
        Cmd::JsonField {
            file,
            key,
            expect_int,
            expect_str,
        } => cmd_json_field(&file, &key, expect_int, expect_str.as_deref()),
        Cmd::SummaryJson { out } => cmd_summary_json(&out),
        Cmd::Fsops { json_arg } => cmd_fsops(&json_arg),
    };
    process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cidr_math() {
        assert_eq!(
            cmd_cidr_contains("192.168.100.11/24", "192.168.100.0/24"),
            0
        );
        assert_eq!(cmd_cidr_contains("10.0.0.1/24", "192.168.100.0/24"), 1);
        let (ip, pfx) = parse_ipv4_cidr("192.168.100.10/24").unwrap();
        assert_eq!(
            Ipv4Addr::from(ip & netmask(pfx)),
            Ipv4Addr::new(192, 168, 100, 0)
        );
    }
}
