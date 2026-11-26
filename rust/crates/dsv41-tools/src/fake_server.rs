//! High-throughput async mock TensorFold server (`dsv41-tools fake-server`)
//! replacing `tools/fake_server.py`, simulating prompt cache, spill v2 disk tier,
//! KV sharding, and all `--mutate` fault injections.

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use clap::Args;
use dsv41_core::metrics::{KV_FIELDS, KV_GAUGES, KV_PART_FIELDS, SPILL_FIELDS};
use dsv41_core::{canonical_json_py, sha256_hex, DEFAULT_MODEL};
use regex::Regex;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Notify, Semaphore};

const FAKE_WORDS: &[&str] = &[
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "xray", "yankee", "zulu",
];
const BLOCK_BYTES: usize = 4096;
const SMALL_BYTES: usize = 1024;
const KV_BYTES_PER_TOKEN: usize = 720;

#[derive(Args, Clone)]
pub struct FakeServerArgs {
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 18888)]
    pub port: u16,
    #[arg(long, default_value = DEFAULT_MODEL)]
    pub model: String,
    #[arg(long, default_value_t = 8)]
    pub parallel: usize,
    #[arg(long, default_value_t = 1_048_576)]
    pub context: usize,
    #[arg(long, default_value_t = 0)]
    pub pool: usize,
    #[arg(long = "kv-shard")]
    pub kv_shard: bool,
    #[arg(long = "twin-min", default_value_t = 16384)]
    pub twin_min: usize,
    #[arg(long = "kv-carry", default_value_t = 1)]
    pub kv_carry: u8,
    #[arg(long = "kv-twins", default_value_t = 1)]
    pub kv_twins: u8,
    #[arg(long = "kv-decode", default_value = "fetch")]
    pub kv_decode: String,
    #[arg(long = "carry-ms", default_value_t = 30.0)]
    pub carry_ms: f64,
    #[arg(long, default_value_t = 2048)]
    pub grid: usize,
    #[arg(long, default_value_t = 128)]
    pub window: usize,
    #[arg(long = "mem-points", default_value_t = 8)]
    pub mem_points: usize,
    #[arg(long = "spill-dir")]
    pub spill_dir: Option<PathBuf>,
    #[arg(long = "spill-dir-rank1")]
    pub spill_dir_rank1: Option<PathBuf>,
    #[arg(long = "spill-min-tokens", default_value_t = 8192)]
    pub spill_min_tokens: usize,
    #[arg(long = "spill-min-free-gib", default_value_t = 0.0)]
    pub spill_min_free_gib: f64,
    #[arg(long = "disk-free-bytes", default_value_t = 0)]
    pub disk_free_bytes: u64,
    #[arg(long = "startup-log")]
    pub startup_log: Option<PathBuf>,
    #[arg(long = "write-ms", default_value_t = 40.0)]
    pub write_ms: f64,
    #[arg(long = "restore-ms-per-block", default_value_t = 15.0)]
    pub restore_ms_per_block: f64,
    #[arg(long = "serial-prefill")]
    pub serial_prefill: bool,
    #[arg(long = "prefill-us-per-token", default_value_t = 20.0)]
    pub prefill_us_per_token: f64,
    #[arg(long = "decode-ms-per-token", default_value_t = 2.0)]
    pub decode_ms_per_token: f64,
    #[arg(long, default_value = "")]
    pub mutate: String,
}

#[derive(Clone)]
struct MemEntry {
    id: u64,
    at: usize,
    hashes: Vec<String>,
    part: usize,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Manifest {
    hash: String,
    at: usize,
    blocks: Vec<(String, usize, u32)>,
    small: (usize, u32),
}

#[derive(Clone)]
struct WriteJob {
    hash: String,
    at: usize,
    blocks: Vec<String>,
    ranks: Vec<usize>,
    needs: HashMap<usize, u64>,
    pictures: bool,
}

struct LiveStream {
    part: usize,
    need: usize,
}

struct FakeInner {
    a: FakeServerArgs,
    part_size: usize,
    window: usize,
    roots: Vec<PathBuf>,
    c: HashMap<String, i64>,
    t: [HashMap<String, i64>; 2],
    k: HashMap<String, i64>,
    mem: VecDeque<MemEntry>,
    index: HashMap<String, (u64, usize)>,
    next_id: u64,
    disk: HashMap<String, Manifest>,
    have: [HashSet<String>; 2],
    stored: [u64; 2],
    queued: [u64; 2],
    streams: HashMap<u64, LiveStream>,
    next_sid: u64,
    kept_rows: [usize; 2],
    twin: [usize; 2],
    hold: [usize; 2],
    rank1_dead: bool,
    active: usize,
    writes: VecDeque<WriteJob>,
    pending_restores: HashMap<String, Arc<Notify>>,
}

pub struct FakeState {
    inner: Mutex<FakeInner>,
    prefill_gate: tokio::sync::Mutex<()>,
    slots: Semaphore,
    write_notify: Notify,
}

impl FakeInner {
    fn cnt_add(&mut self, k: &str, v: i64) {
        *self.c.entry(k.to_string()).or_insert(0) += v;
    }
    fn t_add(&mut self, r: usize, k: &str, v: i64) {
        *self.t[r].entry(k.to_string()).or_insert(0) += v;
    }
    fn k_add(&mut self, k: &str, v: i64) {
        *self.k.entry(k.to_string()).or_insert(0) += v;
    }

    fn point_dir(&self, r: usize) -> PathBuf {
        self.roots[r].join("points")
    }

    fn free_bytes(&self, r: usize) -> u64 {
        if self.a.disk_free_bytes > 0 {
            return self.a.disk_free_bytes.saturating_sub(self.stored[r]);
        }
        #[cfg(unix)]
        {
            use std::ffi::CString;
            if let Ok(cp) = CString::new(self.roots[r].to_string_lossy().as_bytes()) {
                let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
                if unsafe { libc::statvfs(cp.as_ptr(), &mut st) } == 0 {
                    return (st.f_bavail as u64).saturating_mul(st.f_frsize as u64);
                }
            }
        }
        100 * (1u64 << 30)
    }

    fn recover(&mut self) {
        for r in 0..self.roots.len() {
            let root = self.roots[r].clone();
            for sub in ["blocks", "points"] {
                let sd = root.join(sub);
                let _ = fs::create_dir_all(&sd);
                if let Ok(rd) = fs::read_dir(&sd) {
                    for ent in rd.flatten() {
                        let name = ent.file_name().to_string_lossy().into_owned();
                        let p = ent.path();
                        if name.starts_with('.') || name.contains(".tmp") {
                            let _ = fs::remove_file(&p);
                            self.t_add(r, "dropped_temporaries", 1);
                        } else if let Ok(md) = fs::metadata(&p) {
                            self.stored[r] += md.len();
                        }
                    }
                }
            }
            let pd = self.point_dir(r);
            if let Ok(rd) = fs::read_dir(&pd) {
                for ent in rd.flatten() {
                    let p = ent.path();
                    if p.extension().and_then(|s| s.to_str()) == Some("man") {
                        if let Ok(txt) = fs::read_to_string(&p) {
                            if let Ok(man) = serde_json::from_str::<Manifest>(&txt) {
                                self.have[r].insert(man.hash.clone());
                                self.t_add(r, "recovered_points", 1);
                                if r == 0 {
                                    self.disk.insert(man.hash.clone(), man);
                                } else {
                                    self.disk.entry(man.hash.clone()).or_insert(man);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn mem_best(&self, bounds: &[(usize, String)]) -> usize {
        let mut best = 0;
        for (at, h) in bounds {
            if self.index.contains_key(h) {
                best = *at;
            }
        }
        best
    }

    fn mem_entry_part(&self, bounds: &[(usize, String)], at: usize) -> Option<(u64, usize)> {
        for (b, h) in bounds {
            if *b == at {
                if let Some(&(id, _)) = self.index.get(h) {
                    if let Some(e) = self.mem.iter().find(|e| e.id == id) {
                        return Some((id, e.part));
                    }
                }
            }
        }
        None
    }

    fn move_to_end(&mut self, id: u64) {
        if let Some(pos) = self.mem.iter().position(|e| e.id == id) {
            if let Some(item) = self.mem.remove(pos) {
                self.mem.push_back(item);
            }
        }
    }

    fn evict_one(&mut self, part: Option<usize>) -> bool {
        let pos = self
            .mem
            .iter()
            .position(|e| part.map(|p| e.part == p).unwrap_or(true));
        let Some(idx) = pos else { return false };
        let e = self.mem.remove(idx).unwrap();
        for h in &e.hashes {
            if self.index.get(h).map(|&(id, _)| id) == Some(e.id) {
                self.index.remove(h);
            }
        }
        self.cnt_add("evicted", 1);
        if self.a.kv_shard {
            self.kept_rows[e.part] = self.kept_rows[e.part].saturating_sub(e.at);
            if part.is_some() {
                self.k_add("staled", 1);
            }
        }
        true
    }

    fn mem_keep(&mut self, bounds: &[(usize, String)], at: usize, part: usize) {
        let hs: Vec<String> = bounds
            .iter()
            .filter(|(b, _)| *b <= at)
            .map(|(_, h)| h.clone())
            .collect();
        let Some(last_h) = hs.last() else { return };
        if let Some(&(id, _)) = self.index.get(last_h) {
            if self.mem.iter().any(|e| e.id == id && e.at >= at) {
                self.move_to_end(id);
                return;
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        for (b, h) in bounds {
            if *b <= at {
                self.index.insert(h.clone(), (id, *b));
            }
        }
        self.mem.push_back(MemEntry {
            id,
            at,
            hashes: hs,
            part,
        });
        self.cnt_add("kept", 1);
        if self.a.kv_shard {
            self.kept_rows[part] += at;
        }
        while self.mem.len() > self.a.mem_points {
            self.evict_one(None);
        }
    }

    fn live(&self, p: usize) -> usize {
        self.streams.values().filter(|s| s.part == p).count()
    }
    fn live_rows(&self, p: usize) -> usize {
        self.streams
            .values()
            .filter(|s| s.part == p)
            .map(|s| s.need)
            .sum()
    }
    fn room(&self, p: usize) -> isize {
        self.part_size as isize
            - self.live_rows(p) as isize
            - self.kept_rows[p] as isize
            - self.twin[p] as isize
            - self.hold[p] as isize
    }

    fn make_room(&mut self, part: usize, need: usize) -> usize {
        if self.a.kv_carry != 0 {
            for p in [part, 1 - part] {
                let q = 1 - p;
                let moving_ids: Vec<u64> = self
                    .streams
                    .iter()
                    .filter(|(_, s)| s.part == p)
                    .map(|(&id, _)| id)
                    .collect();
                let rows: usize = moving_ids.iter().map(|id| self.streams[id].need).sum();
                let spare = self.part_size as isize
                    - self.live_rows(q) as isize
                    - self.twin[q] as isize
                    - self.hold[q] as isize;
                if !moving_ids.is_empty()
                    && rows as isize <= spare
                    && (self.part_size as isize - self.twin[p] as isize - self.hold[p] as isize)
                        >= need as isize
                {
                    while self.room(q) < rows as isize && self.evict_one(Some(q)) {}
                    for id in &moving_ids {
                        if let Some(s) = self.streams.get_mut(id) {
                            s.part = q;
                        }
                    }
                    let nm = moving_ids.len() as i64;
                    self.k_add("moves", nm);
                    self.k_add("moved_tokens", rows as i64);
                    self.k_add("moved_bytes", (rows * KV_BYTES_PER_TOKEN) as i64);
                    self.k_add("copies", nm);
                    self.k_add("copy_bytes", (rows * KV_BYTES_PER_TOKEN) as i64);
                    while self.room(p) < need as isize && self.evict_one(Some(p)) {}
                    return p;
                }
            }
        }
        if self.room(part) < need as isize {
            self.k_add("no_part", 1);
        }
        part
    }

    fn need_bytes(&self, r: usize, hashes: &[String]) -> u64 {
        let mut new_b = 0u64;
        for bh in hashes {
            let p = self.roots[r]
                .join("blocks")
                .join(format!("{}.blk", &bh[..32]));
            if !p.exists() {
                new_b += BLOCK_BYTES as u64;
            }
        }
        new_b + SMALL_BYTES as u64 + 65 * (hashes.len() as u64) + 64 + 80 * (hashes.len() as u64)
    }

    fn enqueue_write(&mut self, bounds: &[(usize, String)], at: usize, pictures: bool) -> bool {
        if self.a.spill_dir.is_none() || at < self.a.spill_min_tokens {
            return false;
        }
        if pictures && self.a.mutate == "no-pictures" {
            return false;
        }
        let Some((_, h)) = bounds.iter().find(|(b, _)| *b == at) else {
            return false;
        };
        if self.disk.contains_key(h) || self.writes.iter().any(|w| &w.hash == h) {
            return false;
        }
        let hashes: Vec<String> = bounds
            .iter()
            .filter(|(b, _)| *b <= at)
            .map(|(_, x)| x.clone())
            .collect();
        let floor = (self.a.spill_min_free_gib * (1u64 << 30) as f64) as u64;
        let mut ranks = Vec::new();
        let mut needs = HashMap::new();
        for r in 0..2 {
            let need = self.need_bytes(r, &hashes);
            if floor.saturating_add(need).saturating_add(self.queued[r]) > self.free_bytes(r) {
                self.t_add(r, "no_space", 1);
                continue;
            }
            ranks.push(r);
            needs.insert(r, need);
            self.queued[r] += need;
            self.t_add(r, "writes_queued", 1);
        }
        if ranks.is_empty() {
            return false;
        }
        self.writes.push_back(WriteJob {
            hash: h.clone(),
            at,
            blocks: hashes,
            ranks,
            needs,
            pictures,
        });
        self.cnt_add("writes_enqueued", 1);
        true
    }

    fn pending(&self, r: usize) -> usize {
        self.writes
            .iter()
            .filter(|w| w.ranks.contains(&r) && !self.have[r].contains(&w.hash))
            .count()
    }

    fn disk_best(&self, bounds: &[(usize, String)], beyond: usize) -> Option<(usize, String)> {
        let mut best = None;
        for (at, h) in bounds {
            if *at > beyond && self.disk.contains_key(h) && self.have[0].contains(h) {
                best = Some((*at, h.clone()));
            }
        }
        best
    }

    fn forget(&mut self, h: &str) {
        let man = self.disk.remove(h);
        for r in 0..2 {
            self.have[r].remove(h);
        }
        if man.is_some() {
            for r in 0..2 {
                let pd = self.point_dir(r);
                for ext in [".man", ".st", ".ids"] {
                    let _ = fs::remove_file(pd.join(format!("{}{ext}", &h[..32])));
                }
            }
        }
    }

    fn health_json(&self) -> Value {
        let mut h = json!({
            "status": "ok",
            "model": self.a.model,
            "model_ids": [self.a.model],
            "max_batch_size": self.a.parallel,
            "warming": false,
            "live": {"connections": self.active, "waiting": 0}
        });
        let on = self.a.spill_dir.is_some() || self.a.serial_prefill || self.a.kv_shard;
        if on {
            let mut pc = Map::new();
            for k in [
                "memory_hits",
                "memory_misses",
                "kept",
                "evicted",
                "refused",
                "failed",
                "disk_lookups",
                "disk_hits",
                "disk_hits_committed",
                "restores_begun",
                "restores_shared",
                "restores_failed",
                "restores_cancelled",
                "restores_dropped",
                "restores_active",
                "restore_blocks_done",
                "restore_blocks_total",
                "restored_tokens",
                "restore_waits",
                "restore_wait_ns",
                "restore_wait_max_ns",
                "writes_enqueued",
                "parked",
                "cached_tokens",
                "computed_tokens",
            ] {
                pc.insert(k.to_string(), json!(self.c.get(k).copied().unwrap_or(0)));
            }
            pc.insert("entries".to_string(), json!(self.mem.len()));
            if self.a.spill_dir.is_some() {
                let mut ranks = Vec::new();
                for r in 0..2 {
                    let mut sp = Map::new();
                    sp.insert("rank".to_string(), json!(r));
                    for &k in SPILL_FIELDS {
                        sp.insert(k.to_string(), json!(self.t[r].get(k).copied().unwrap_or(0)));
                    }
                    let blk_cnt = fs::read_dir(self.roots[r].join("blocks"))
                        .map(|rd| rd.count())
                        .unwrap_or(0);
                    sp.insert("points".to_string(), json!(self.have[r].len()));
                    sp.insert("stored_bytes".to_string(), json!(self.stored[r]));
                    sp.insert("cap_bytes".to_string(), json!(200u64 * (1u64 << 30)));
                    sp.insert("disk_free_bytes".to_string(), json!(self.free_bytes(r)));
                    sp.insert("writes_pending".to_string(), json!(self.pending(r)));
                    sp.insert("blocks".to_string(), json!(blk_cnt));
                    ranks.push(Value::Object(sp));
                }
                pc.insert("spill".to_string(), Value::Array(ranks));
            }
            h["chats_continued"] = json!(0);
            h["chats_continued_prefill_tokens"] = json!(0);
            h["prompt_cache"] = Value::Object(pc);
        }
        if self.a.kv_shard {
            let mut ks = Map::new();
            for &k in KV_FIELDS {
                let v = if k == "parts" {
                    2
                } else if k == "part_rows" {
                    self.part_size as i64
                } else {
                    self.k.get(k).copied().unwrap_or(0)
                };
                ks.insert(k.to_string(), json!(v));
            }
            let per_part: Vec<Value> = [0usize, 1]
                .into_iter()
                .map(|p| {
                    let free = self.room(p).max(0) as usize;
                    let avail_for_kept = self
                        .part_size
                        .saturating_sub(self.live_rows(p) + self.twin[p] + self.hold[p]);
                    json!({
                        "part": p,
                        "free": free,
                        "live_rows": self.live_rows(p),
                        "kept_rows": self.kept_rows[p].min(avail_for_kept),
                        "twin_rows": self.twin[p],
                        "hold_rows": self.hold[p],
                        "live": self.live(p)
                    })
                })
                .collect();
            ks.insert("per_part".to_string(), Value::Array(per_part));
            h["kv_shard"] = Value::Object(ks);
        }
        h
    }

    fn metrics_text(&self) -> String {
        let h = self.health_json();
        let mut lines = Vec::new();
        if let Some(pc) = h.get("prompt_cache").and_then(|v| v.as_object()) {
            for (k, v) in pc {
                if !v.is_array() {
                    lines.push(format!("# TYPE tensorfold:prompt_cache_{k} counter\ntensorfold:prompt_cache_{k} {v}"));
                }
            }
            if let Some(sp) = pc.get("spill").and_then(|v| v.as_array()) {
                for &k in SPILL_FIELDS {
                    lines.push(format!("# TYPE tensorfold:spill_{k} counter"));
                    for o in sp {
                        lines.push(format!(
                            "tensorfold:spill_{k}{{rank=\"{}\"}} {}",
                            o["rank"], o[k]
                        ));
                    }
                }
            }
        }
        if let Some(ks) = h.get("kv_shard").and_then(|v| v.as_object()) {
            for &k in KV_FIELDS {
                let ty = if KV_GAUGES.contains(&k) {
                    "gauge"
                } else {
                    "counter"
                };
                lines.push(format!(
                    "# TYPE tensorfold:kv_shard_{k} {ty}\ntensorfold:kv_shard_{k} {}",
                    ks[k]
                ));
            }
            if let Some(pp) = ks.get("per_part").and_then(|v| v.as_array()) {
                for &row in KV_PART_FIELDS {
                    lines.push(format!("# TYPE tensorfold:kv_shard_part_{row} gauge"));
                    for o in pp {
                        lines.push(format!(
                            "tensorfold:kv_shard_part_{row}{{part=\"{}\"}} {}",
                            o["part"], o[row]
                        ));
                    }
                }
            }
        }
        lines.join("\n") + "\n"
    }
}

fn tokenize_messages(messages: &[Value]) -> Vec<String> {
    let mut out = Vec::new();
    for m in messages {
        out.push(format!("<{}>", m["role"].as_str().unwrap_or("user")));
        match m.get("content") {
            Some(Value::String(s)) => {
                for w in s.split_whitespace() {
                    out.push(w.to_string());
                }
            }
            Some(Value::Array(arr)) => {
                for p in arr {
                    if p["type"].as_str().unwrap_or("text") == "text" {
                        if let Some(t) = p["text"].as_str() {
                            for w in t.split_whitespace() {
                                out.push(w.to_string());
                            }
                        }
                    } else {
                        let d = sha256_hex(canonical_json_py(p).as_bytes());
                        out.push(format!("<picture:{}>", &d[..16]));
                    }
                }
            }
            _ => {}
        }
    }
    out.push("<assistant>".to_string());
    out
}

fn boundaries(toks: &[String], grid: usize, window: usize) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let limit = toks.len().saturating_sub(window);
    let mut h = Sha256::new();
    for (i, tk) in toks.iter().enumerate() {
        h.update(tk.as_bytes());
        h.update([0u8]);
        let at = i + 1;
        if at % grid == 0 && at <= limit {
            out.push((at, hex::encode(h.clone().finalize())));
        }
    }
    out
}

fn synth_reply(toks: &[String], max_tokens: usize, salt: &str) -> Vec<String> {
    let mut input = toks.join("\0");
    input.push_str(&format!("|{max_tokens}|{salt}"));
    let ph = sha256_hex(input.as_bytes());
    let b0 = u8::from_str_radix(&ph[0..2], 16).unwrap_or(0) as usize;
    let n = (12 + (b0 % 20)).min(max_tokens).max(1);
    (0..n)
        .map(|i| {
            let off = (2 * i) % 60;
            let v = u8::from_str_radix(&ph[off..off + 2], 16).unwrap_or(0) as usize;
            FAKE_WORDS[v % FAKE_WORDS.len()].to_string()
        })
        .collect()
}

fn write_atomic_file(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path.file_name().unwrap().to_string_lossy();
    let tmp = dir.join(format!(".{name}.tmp"));
    let mut f = fs::File::create(&tmp)?;
    f.write_all(data)?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, path)
}

async fn writer_loop(state: Arc<FakeState>) {
    loop {
        let job_opt = { state.inner.lock().unwrap().writes.front().cloned() };
        let Some(job) = job_opt else {
            state.write_notify.notified().await;
            continue;
        };
        let (write_ms, roots) = {
            let g = state.inner.lock().unwrap();
            (g.a.write_ms, g.roots.clone())
        };
        if write_ms > 0.0 {
            tokio::time::sleep(Duration::from_secs_f64(write_ms / 1000.0)).await;
        }
        let mut man_blocks = Vec::with_capacity(job.blocks.len());
        let mut block_payloads = Vec::with_capacity(job.blocks.len());
        for bh in &job.blocks {
            let rep = bh.repeat(BLOCK_BYTES / 64 + 1);
            let data = rep.as_bytes()[..BLOCK_BYTES].to_vec();
            let crc = crc32fast::hash(&data);
            man_blocks.push((bh[..32].to_string(), data.len(), crc));
            block_payloads.push((bh[..32].to_string(), data));
        }
        let small = job.hash.repeat(SMALL_BYTES / 64).into_bytes();
        let man = Manifest {
            hash: job.hash.clone(),
            at: job.at,
            blocks: man_blocks,
            small: (small.len(), crc32fast::hash(&small)),
        };
        let man_bytes = serde_json::to_vec(&man).unwrap();
        let ids_bytes = job.blocks.join(" ").into_bytes();

        for &r in &job.ranks {
            let root = &roots[r];
            for (bname, data) in &block_payloads {
                let p = root.join("blocks").join(format!("{bname}.blk"));
                if !p.exists() {
                    if write_atomic_file(&p, data).is_ok() {
                        let mut g = state.inner.lock().unwrap();
                        g.stored[r] += data.len() as u64;
                        g.t_add(r, "written_bytes", data.len() as i64);
                    }
                } else {
                    state.inner.lock().unwrap().t_add(r, "dedup_blocks", 1);
                }
            }
            let pd = root.join("points");
            for (ext, payload) in [(".ids", &ids_bytes[..]), (".st", &small[..])] {
                let p = pd.join(format!("{}{ext}", &job.hash[..32]));
                if write_atomic_file(&p, payload).is_ok() {
                    let mut g = state.inner.lock().unwrap();
                    g.stored[r] += payload.len() as u64;
                    g.t_add(r, "written_bytes", payload.len() as i64);
                }
            }
            if write_ms > 0.0 {
                tokio::time::sleep(Duration::from_secs_f64(write_ms / 4000.0)).await;
            }
            let mp = pd.join(format!("{}.man", &job.hash[..32]));
            if write_atomic_file(&mp, &man_bytes).is_ok() {
                let mut g = state.inner.lock().unwrap();
                g.stored[r] += man_bytes.len() as u64;
                g.t_add(r, "written_bytes", man_bytes.len() as i64);
                g.have[r].insert(job.hash.clone());
                g.queued[r] = g.queued[r].saturating_sub(job.needs[&r]);
                g.t_add(r, "writes_committed", 1);
                if job.pictures {
                    g.t_add(r, "pictures_kept", 1);
                }
            }
        }
        let mut g = state.inner.lock().unwrap();
        g.writes.pop_front();
        g.disk.insert(job.hash.clone(), man);
    }
}

async fn handle_health(State(st): State<Arc<FakeState>>) -> Json<Value> {
    Json(st.inner.lock().unwrap().health_json())
}

async fn handle_metrics(State(st): State<Arc<FakeState>>) -> impl IntoResponse {
    let txt = st.inner.lock().unwrap().metrics_text();
    ([(header::CONTENT_TYPE, "text/plain")], txt)
}

async fn handle_models(State(st): State<Arc<FakeState>>) -> Json<Value> {
    let model = st.inner.lock().unwrap().a.model.clone();
    Json(
        json!({"object": "list", "data": [{"id": model, "object": "model", "owned_by": "tensorfold"}]}),
    )
}

async fn handle_tokenize(State(st): State<Arc<FakeState>>, Json(body): Json<Value>) -> Json<Value> {
    let msgs = body["messages"].as_array().cloned().unwrap_or_default();
    let count = tokenize_messages(&msgs).len();
    let w = st.inner.lock().unwrap().window;
    Json(json!({"count": count, "max_model_len": w}))
}

async fn handle_stop() -> Json<Value> {
    std::process::exit(0);
}

async fn handle_kill_rank1(State(st): State<Arc<FakeState>>) -> Json<Value> {
    let mutate = {
        let mut g = st.inner.lock().unwrap();
        g.rank1_dead = true;
        g.a.mutate.clone()
    };
    if mutate != "kill-hang" {
        tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(1)).await;
            std::process::exit(1);
        });
    }
    Json(json!({"killed": "rank1"}))
}

struct RequestGuard {
    st: Arc<FakeState>,
    sid: Option<u64>,
    part: usize,
    twin: usize,
    held_need: usize,
    active_inc: bool,
    restore_cleanup: Option<(String, usize)>,
    parked_inc: bool,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        let mut g = self.st.inner.lock().unwrap();
        if let Some((h, nblocks)) = self.restore_cleanup.take() {
            g.cnt_add("restores_cancelled", 1);
            g.cnt_add("restores_active", -1);
            g.cnt_add("restore_blocks_total", -(nblocks as i64));
            g.c.insert("restore_blocks_done".to_string(), 0);
            if let Some(ev) = g.pending_restores.remove(&h) {
                ev.notify_waiters();
            }
        }
        if self.parked_inc {
            g.cnt_add("parked", -1);
        }
        if self.held_need > 0 && g.a.mutate != "leak-hold" {
            g.hold[self.part] = g.hold[self.part].saturating_sub(self.held_need);
        }
        if self.active_inc {
            g.active = g.active.saturating_sub(1);
        }
        if g.a.kv_shard {
            if let Some(sid) = self.sid {
                g.streams.remove(&sid);
            }
            if self.twin > 0 {
                g.twin[1 - self.part] = g.twin[1 - self.part].saturating_sub(self.twin);
            }
        }
    }
}

async fn run_request(
    st: Arc<FakeState>,
    messages: Vec<Value>,
    max_tokens: usize,
    emit: Option<mpsc::Sender<Result<Bytes, std::io::Error>>>,
) -> Result<Value, (StatusCode, Value)> {
    let (a, window, roots) = {
        let g = st.inner.lock().unwrap();
        (g.a.clone(), g.window, g.roots.clone())
    };
    let toks = tokenize_messages(&messages);
    if (toks.len() + 1 > window || toks.len() + max_tokens.min(32) > window + 32)
        && a.mutate != "admit-over-window"
    {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({"error": {
                "message": format!("This server's maximum context length is {window} tokens, prompt has {}", toks.len()),
                "type": "invalid_request_error",
                "code": "context_length_exceeded"
            }}),
        ));
    }
    let bounds = boundaries(&toks, a.grid, a.window);
    let mut guard = RequestGuard {
        st: st.clone(),
        sid: None,
        part: 0,
        twin: 0,
        held_need: 0,
        active_inc: false,
        restore_cleanup: None,
        parked_inc: false,
    };

    let (mut best, disk) = {
        let mut g = st.inner.lock().unwrap();
        let b = g.mem_best(&bounds);
        if b > 0 {
            g.cnt_add("memory_hits", 1);
            if let Some((id, _)) = g.mem_entry_part(&bounds, b) {
                g.move_to_end(id);
            }
        } else {
            g.cnt_add("memory_misses", 1);
        }
        let d = if a.spill_dir.is_some() && a.mutate != "no-disk" {
            g.cnt_add("disk_lookups", 1);
            let found = g.disk_best(&bounds, b);
            if found.is_none() {
                g.t_add(0, "compatible_misses", 1);
            }
            found
        } else {
            None
        };
        (b, d)
    };

    let mut restored = false;
    let mut corrupt_flag = false;

    if let Some((at, h)) = disk {
        let (waiter, man_opt) = {
            let mut g = st.inner.lock().unwrap();
            g.cnt_add("disk_hits", 1);
            g.cnt_add("parked", 1);
            guard.parked_inc = true;
            if let Some(w) = g.pending_restores.get(&h).cloned() {
                g.cnt_add("restores_shared", 1);
                (Some(w), None)
            } else {
                let ev = Arc::new(Notify::new());
                g.pending_restores.insert(h.clone(), ev);
                g.cnt_add("restores_begun", 1);
                g.cnt_add("restores_active", 1);
                let man = g.disk[&h].clone();
                g.cnt_add("restore_blocks_total", man.blocks.len() as i64);
                guard.restore_cleanup = Some((h.clone(), man.blocks.len()));
                (None, Some(man))
            }
        };

        if let Some(w) = waiter {
            let _ = tokio::time::timeout(Duration::from_secs(600), w.notified()).await;
            let mut g = st.inner.lock().unwrap();
            g.cnt_add("parked", -1);
            guard.parked_inc = false;
            best = g.mem_best(&bounds);
        } else if let Some(man) = man_opt {
            let t0 = Instant::now();
            let mut why: Option<(&'static str, usize)> = None;
            {
                let g = st.inner.lock().unwrap();
                for r in 0..2 {
                    if !g.have[r].contains(&man.hash) && a.mutate != "ignore-presence" {
                        why = Some(("presence_refused", 0));
                        break;
                    }
                }
            }
            if why.is_none() {
                for (r, root) in roots.iter().enumerate().take(2) {
                    let p = root.join("points").join(format!("{}.st", &man.hash[..32]));
                    match fs::read(&p) {
                        Err(_) => {
                            why = Some(("read_failed", r));
                            break;
                        }
                        Ok(small) => {
                            if small.len() != man.small.0 {
                                why = Some(("size_errors", r));
                                break;
                            }
                            if crc32fast::hash(&small) != man.small.1 {
                                if a.mutate != "ignore-crc" {
                                    why = Some(("crc_errors", r));
                                    break;
                                }
                                corrupt_flag = true;
                            }
                        }
                    }
                }
            }
            if why.is_none() {
                'blk: for (bname, bsz, bcrc) in &man.blocks {
                    if let Some(tx) = &emit {
                        if tx.is_closed() {
                            return Err((StatusCode::REQUEST_TIMEOUT, json!({"error": "gone"})));
                        }
                    }
                    if a.restore_ms_per_block > 0.0 {
                        tokio::time::sleep(Duration::from_secs_f64(
                            a.restore_ms_per_block / 1000.0,
                        ))
                        .await;
                    }
                    st.inner.lock().unwrap().cnt_add("restore_blocks_done", 1);
                    for (r, root) in roots.iter().enumerate().take(2) {
                        let p = root.join("blocks").join(format!("{bname}.blk"));
                        match fs::read(&p) {
                            Err(_) => {
                                why = Some(("read_failed", r));
                                break 'blk;
                            }
                            Ok(data) => {
                                if data.len() != *bsz {
                                    why = Some(("size_errors", r));
                                    break 'blk;
                                }
                                if crc32fast::hash(&data) != *bcrc {
                                    if a.mutate != "ignore-crc" {
                                        why = Some(("crc_errors", r));
                                        break 'blk;
                                    }
                                    corrupt_flag = true;
                                }
                            }
                        }
                    }
                }
            }
            let mut g = st.inner.lock().unwrap();
            guard.restore_cleanup = None;
            g.cnt_add("parked", -1);
            guard.parked_inc = false;
            g.cnt_add("restores_active", -1);
            g.cnt_add("restore_blocks_total", -(man.blocks.len() as i64));
            g.c.insert("restore_blocks_done".to_string(), 0);
            let wait_ns = t0.elapsed().as_nanos() as i64;
            g.cnt_add("restore_waits", 1);
            g.cnt_add("restore_wait_ns", wait_ns);
            let prev_max = g.c.get("restore_wait_max_ns").copied().unwrap_or(0);
            g.c.insert("restore_wait_max_ns".to_string(), prev_max.max(wait_ns));
            if let Some((cnt, r)) = why {
                g.t_add(r, cnt, 1);
                g.cnt_add("restores_failed", 1);
                if cnt != "presence_refused" {
                    g.forget(&h);
                }
            } else {
                for r in 0..2 {
                    g.t_add(r, "reads", 1);
                }
                g.cnt_add("disk_hits_committed", 1);
                g.cnt_add("restored_tokens", at as i64);
                g.mem_keep(&bounds, at, 0);
                best = at;
                restored = true;
            }
            if let Some(ev) = g.pending_restores.remove(&h) {
                ev.notify_waiters();
            }
        }
    }

    // KV shard placement
    let need = toks.len() + max_tokens;
    let mut carried = false;
    {
        let mut g = st.inner.lock().unwrap();
        if a.kv_shard {
            if let Some((_, p)) = if best > 0 {
                g.mem_entry_part(&bounds, best)
            } else {
                None
            } {
                let mut part = p;
                let other = 1 - part;
                if g.room(part) < (need.saturating_sub(best)) as isize {
                    if a.kv_carry != 0 && g.room(other) >= need as isize {
                        carried = true;
                        part = other;
                    } else {
                        while g.room(part) < (need.saturating_sub(best)) as isize
                            && g.evict_one(Some(part))
                        {}
                        if g.mem_best(&bounds) < best {
                            best = 0;
                        }
                    }
                }
                guard.part = part;
            } else {
                let mut part = if g.live(0) <= g.live(1) { 0 } else { 1 };
                if g.room(part) < need as isize {
                    part = g.make_room(part, need);
                }
                if a.kv_twins != 0
                    && toks.len() >= a.twin_min
                    && g.room(1 - part) >= toks.len() as isize
                {
                    guard.twin = toks.len();
                    g.k_add("twins_made", 1);
                    g.twin[1 - part] += guard.twin;
                }
                guard.part = part;
            }
            let sid = g.next_sid;
            g.next_sid += 1;
            guard.sid = Some(sid);
            if carried {
                g.k_add("carries", 1);
                g.k_add("carry_pressure", 1);
                g.k_add("moved_tokens", best as i64);
                g.k_add("moved_bytes", (best * KV_BYTES_PER_TOKEN) as i64);
                g.k_add("copies", 1);
                g.k_add("copy_bytes", (best * KV_BYTES_PER_TOKEN) as i64);
                g.hold[guard.part] += need;
                guard.held_need = need;
            } else {
                g.streams.insert(
                    sid,
                    LiveStream {
                        part: guard.part,
                        need,
                    },
                );
            }
            g.k_add("plans", 1);
        }
    }

    let mut prefill_guard = if a.serial_prefill {
        Some(st.prefill_gate.lock().await)
    } else {
        None
    };
    if prefill_guard.is_some() {
        let g = st.inner.lock().unwrap();
        best = best.max(g.mem_best(&bounds));
    }

    if carried {
        let steps = (best / a.grid).max(1);
        for _ in 0..steps {
            if st.inner.lock().unwrap().rank1_dead {
                if a.mutate == "kill-hang" {
                    loop {
                        if emit.as_ref().map(|tx| tx.is_closed()).unwrap_or(false) {
                            return Err((StatusCode::REQUEST_TIMEOUT, json!({"error": "gone"})));
                        }
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                }
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"error": {"message": "dsv41: rank 1 is gone (PairBroken)"}}),
                ));
            }
            if emit.as_ref().map(|tx| tx.is_closed()).unwrap_or(false) {
                return Err((StatusCode::REQUEST_TIMEOUT, json!({"error": "gone"})));
            }
            if a.carry_ms > 0.0 {
                tokio::time::sleep(Duration::from_secs_f64(a.carry_ms / 1000.0)).await;
            }
        }
        let mut g = st.inner.lock().unwrap();
        g.hold[guard.part] = g.hold[guard.part].saturating_sub(guard.held_need);
        guard.held_need = 0;
        if let Some(sid) = guard.sid {
            g.streams.insert(
                sid,
                LiveStream {
                    part: guard.part,
                    need,
                },
            );
        }
    }

    if a.kv_shard && (guard.twin == 0 || a.mutate == "twin-fetch") && toks.len() > best {
        let mut g = st.inner.lock().unwrap();
        let rows = toks.len() - best;
        let chunks = rows.div_ceil(a.grid);
        g.k_add("prompt_fetches", chunks as i64);
        g.k_add("prompt_syncs", chunks as i64);
        g.k_add("prompt_rows", rows as i64);
        g.k_add("prompt_picks", (2 * rows) as i64);
        g.k_add("prompt_bytes", (rows * KV_BYTES_PER_TOKEN) as i64);
    }

    {
        let mut g = st.inner.lock().unwrap();
        g.cnt_add("cached_tokens", best as i64);
        g.cnt_add("computed_tokens", (toks.len() - best) as i64);
    }
    let prefill_s = (toks.len().saturating_sub(best)) as f64 * a.prefill_us_per_token / 1e6;
    if prefill_s > 0.0 {
        tokio::time::sleep(Duration::from_secs_f64(prefill_s)).await;
    }

    let notify_writer = {
        let mut g = st.inner.lock().unwrap();
        if guard.twin > 0 {
            g.twin[1 - guard.part] = g.twin[1 - guard.part].saturating_sub(guard.twin);
            guard.twin = 0;
        }
        let keep_at = bounds.last().map(|(b, _)| *b).unwrap_or(0);
        if keep_at > 0 {
            let cur_part = guard
                .sid
                .and_then(|id| g.streams.get(&id).map(|s| s.part))
                .unwrap_or(guard.part);
            g.mem_keep(&bounds, keep_at, cur_part);
            let has_pic = toks[..keep_at].iter().any(|t| t.starts_with("<picture:"));
            g.enqueue_write(&bounds, keep_at, has_pic)
        } else {
            false
        }
    };
    if notify_writer {
        st.write_notify.notify_one();
    }
    drop(prefill_guard.take());

    // Decode
    let mut salt = String::new();
    if (restored && a.mutate == "restore-wrong") || corrupt_flag {
        salt = "restored-differently".to_string();
    }
    {
        let mut g = st.inner.lock().unwrap();
        g.active += 1;
        guard.active_inc = true;
        if g.active > 4 && a.mutate == "conc-wrong" {
            salt = "crowded".to_string();
        }
        let cur_part = guard
            .sid
            .and_then(|id| g.streams.get(&id).map(|s| s.part))
            .unwrap_or(guard.part);
        if a.kv_shard && cur_part == 1 && a.mutate == "shard-wrong" {
            salt = "part1".to_string();
        }
    }

    let mut out = synth_reply(&toks, max_tokens, &salt);
    let joined_toks = toks.join(" ");
    if joined_toks.contains("SPECIAL ENTRY:") {
        let re = Regex::new(r"SPECIAL ENTRY: (\w+) secret value is (\d+)").unwrap();
        let mut found = BTreeMap::new();
        for cap in re.captures_iter(&joined_toks) {
            found.insert(cap[1].to_string(), cap[2].to_string());
        }
        if !found.is_empty() {
            out = serde_json::to_string(&found)
                .unwrap()
                .split(' ')
                .map(String::from)
                .collect();
        }
    }

    {
        let _permit = st.slots.acquire().await.unwrap();
        for tk in &out {
            if st.inner.lock().unwrap().rank1_dead {
                if a.mutate == "kill-hang" {
                    loop {
                        if emit.as_ref().map(|tx| tx.is_closed()).unwrap_or(false) {
                            return Err((StatusCode::REQUEST_TIMEOUT, json!({"error": "gone"})));
                        }
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                }
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"error": {"message": "dsv41: rank 1 is gone (PairBroken)"}}),
                ));
            }
            if a.decode_ms_per_token > 0.0 {
                tokio::time::sleep(Duration::from_secs_f64(a.decode_ms_per_token / 1000.0)).await;
            }
            if a.kv_shard {
                let mut g = st.inner.lock().unwrap();
                let rk = if a.kv_decode == "owner" {
                    "owner_rounds"
                } else {
                    "fetch_rounds"
                };
                g.k_add(rk, 1);
            }
            if let Some(tx) = &emit {
                let chunk = format!(
                    "data: {}\n\n",
                    json!({"choices": [{"index": 0, "delta": {"content": format!("{tk} ")}, "finish_reason": null}]})
                );
                if tx.send(Ok(Bytes::from(chunk))).await.is_err() {
                    return Err((StatusCode::REQUEST_TIMEOUT, json!({"error": "gone"})));
                }
            }
        }
    }

    let text = format!("{} ", out.join(" "));
    let mut tf = Map::new();
    tf.insert("cached_tokens".to_string(), json!(best));
    tf.insert(
        "prefill_seconds".to_string(),
        json!((prefill_s * 10000.0).round() / 10000.0),
    );
    if a.mutate != "no-token-sha" {
        tf.insert("token_sha".to_string(), json!(sha256_hex(text.as_bytes())));
    }
    if a.kv_shard {
        tf.insert("part".to_string(), json!(guard.part));
    }
    let finish = if out.len() >= max_tokens {
        "length"
    } else {
        "stop"
    };
    Ok(json!({
        "text": text,
        "prompt_tokens": toks.len(),
        "completion_tokens": out.len(),
        "cached": best,
        "finish": finish,
        "tensorfold": Value::Object(tf)
    }))
}

async fn handle_chat(State(st): State<Arc<FakeState>>, Json(body): Json<Value>) -> Response {
    let messages = body["messages"].as_array().cloned().unwrap_or_default();
    let max_tokens = body["max_tokens"].as_u64().unwrap_or(64) as usize;
    let stream = body["stream"].as_bool().unwrap_or(false);
    let model = st.inner.lock().unwrap().a.model.clone();

    if stream {
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
        let st_c = st.clone();
        tokio::spawn(async move {
            match run_request(st_c, messages, max_tokens, Some(tx.clone())).await {
                Ok(r) => {
                    let fin = format!(
                        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": r["finish"]}]}),
                        json!({
                            "choices": [],
                            "usage": {
                                "prompt_tokens": r["prompt_tokens"],
                                "completion_tokens": r["completion_tokens"],
                                "prompt_tokens_details": {"cached_tokens": r["cached"]}
                            },
                            "tensorfold": r["tensorfold"]
                        })
                    );
                    let _ = tx.send(Ok(Bytes::from(fin))).await;
                }
                Err((_, err_json)) => {
                    let _ = tx
                        .send(Ok(Bytes::from(format!("data: {err_json}\n\n"))))
                        .await;
                }
            }
        });
        let s = futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from_stream(s))
            .unwrap();
    }

    match run_request(st, messages, max_tokens, None).await {
        Ok(r) => Json(json!({
            "id": "fake",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]}, "finish_reason": r["finish"]}],
            "usage": {
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["prompt_tokens"].as_u64().unwrap_or(0) + r["completion_tokens"].as_u64().unwrap_or(0),
                "prompt_tokens_details": {"cached_tokens": r["cached"]}
            },
            "tensorfold": r["tensorfold"]
        }))
        .into_response(),
        Err((code, v)) => (code, Json(v)).into_response(),
    }
}

pub async fn run(mut a: FakeServerArgs) -> i32 {
    if a.pool == 0 {
        a.pool = a.context;
    }
    if a.serial_prefill && a.kv_shard {
        eprintln!("--serial-prefill cannot be combined with --kv-shard");
        return 2;
    }
    let part_size = if a.kv_shard { a.pool / 2 } else { a.pool };
    let window = if a.kv_shard {
        a.context.min(part_size.saturating_sub(18))
    } else {
        a.context
    };
    if let Some(sl) = &a.startup_log {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(sl) {
            let _ = writeln!(
                f,
                "info: dsv41: pool {} positions, {} {part_size} a part; a conversation may use up to {window} tokens; compressed planes 1 MiB a rank (2 MiB replicated); arena 3 MiB used",
                a.pool,
                if a.kv_shard { "fetch" } else { "off" }
            );
        }
    }
    let roots = if let Some(d0) = &a.spill_dir {
        let d1 = a
            .spill_dir_rank1
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("{}-rank1", d0.display())));
        vec![
            d0.join("native-spill-fake").join("rank0"),
            d1.join("native-spill-fake").join("rank1"),
        ]
    } else {
        Vec::new()
    };

    let mut inner = FakeInner {
        a: a.clone(),
        part_size,
        window,
        roots,
        c: HashMap::new(),
        t: [HashMap::new(), HashMap::new()],
        k: HashMap::new(),
        mem: VecDeque::new(),
        index: HashMap::new(),
        next_id: 1,
        disk: HashMap::new(),
        have: [HashSet::new(), HashSet::new()],
        stored: [0, 0],
        queued: [0, 0],
        streams: HashMap::new(),
        next_sid: 1,
        kept_rows: [0, 0],
        twin: [0, 0],
        hold: [0, 0],
        rank1_dead: false,
        active: 0,
        writes: VecDeque::new(),
        pending_restores: HashMap::new(),
    };
    if !inner.roots.is_empty() {
        inner.recover();
    }
    let state = Arc::new(FakeState {
        inner: Mutex::new(inner),
        prefill_gate: tokio::sync::Mutex::new(()),
        slots: Semaphore::new(a.parallel.max(1)),
        write_notify: Notify::new(),
    });
    if a.spill_dir.is_some() {
        tokio::spawn(writer_loop(state.clone()));
    }

    let app = Router::new()
        .route("/health", get(handle_health))
        .route("/metrics", get(handle_metrics))
        .route("/v1/models", get(handle_models))
        .route("/tokenize", post(handle_tokenize))
        .route("/v1/chat/completions", post(handle_chat))
        .route("/fake/stop", post(handle_stop))
        .route("/fake/kill-rank1", post(handle_kill_rank1))
        .with_state(state);

    let addr = format!("{}:{}", a.host, a.port);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("fake-server bind {addr} failed: {e}");
            return 1;
        }
    };
    println!(
        "fake_server: http://{addr} spill={:?} serial_prefill={} kv_shard={} mutate={}",
        a.spill_dir,
        a.serial_prefill,
        a.kv_shard,
        if a.mutate.is_empty() {
            "none"
        } else {
            &a.mutate
        }
    );
    let _ = axum::serve(listener, app).await;
    0
}
