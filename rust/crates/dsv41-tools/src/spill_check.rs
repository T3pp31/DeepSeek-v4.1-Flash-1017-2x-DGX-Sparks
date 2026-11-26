//! End-to-end spill tier checks (`dsv41-tools spill-check record|run|drop|estimate`)
//! replacing `tools/spill_check.py`.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use clap::{Args, Subcommand};
use dsv41_core::auth::resolve_strict;
use dsv41_core::client::{same_reply, Server};
use dsv41_core::metrics::{delta, deltas, rank_deltas, rank_values, Counters};
use dsv41_core::png::encode_rgb_png;
use dsv41_core::prose::{words, words_for};
use dsv41_core::rank_dir::{kind_of, RankDir};
use dsv41_core::receipt::Receipt;
use dsv41_core::{now_iso8601, random_hex, DEFAULT_MODEL};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Notify;

const Q_SEED: &str = "Summarize the text above in one sentence.";
const Q_HIT: &str = "List the three most frequent words in the text above.";
const ITEM_TOKENS: usize = 16_000;
const BLOCK_TOKENS: usize = 2048;
const BYTES_PER_POSITION: usize = 890;
const SMALL_STATE_BYTES: usize = 6_684_672;
const KEEP_WINDOW: usize = 128;
const DECODER: &str = "DEC";

const FAIL_KEYS: &[&str] = &[
    "spill.crc_errors",
    "spill.size_errors",
    "prompt_cache.restores_failed",
    "spill.presence_refused",
    "spill.read_failed",
];

const MAIN_CHECKS: &[&str] = &[
    "evict",
    "corrupt-block",
    "corrupt-small",
    "truncate-block",
    "truncate-small",
    "remove-small",
    "cancel",
    "dual-restore",
    "sizes",
    "settled",
];

fn check_item(chk: &str) -> Option<&'static str> {
    match chk {
        "evict" => Some("E1"),
        "corrupt-block" => Some("CB"),
        "corrupt-small" => Some("CS"),
        "truncate-block" => Some("TB"),
        "truncate-small" => Some("TS"),
        "remove-small" => Some("RS"),
        "cancel" => Some("X1"),
        "dual-restore" => Some("D1"),
        "pictures" => Some("P1"),
        _ => None,
    }
}

fn default_url() -> String {
    env::var("API_URL").unwrap_or_else(|_| {
        format!(
            "http://127.0.0.1:{}",
            env::var("PORT").unwrap_or_else(|_| "8888".to_string())
        )
    })
}

#[derive(Args)]
pub struct SpillCheckArgs {
    #[command(subcommand)]
    pub mode: SpillMode,
}

#[derive(Subcommand)]
pub enum SpillMode {
    Record(SpillRecordArgs),
    Run(Box<SpillRunArgs>),
    Estimate(SpillEstimateArgs),
    Drop(SpillDropArgs),
}

#[derive(Args)]
pub struct SpillRecordArgs {
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value_t = 1800.0)]
    pub timeout: f64,
    #[arg(long, default_value = "")]
    pub label: String,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long)]
    pub salt: Option<String>,
    #[arg(long, default_value_t = 1234)]
    pub seed: i64,
    #[arg(long = "max-tokens", default_value_t = 48)]
    pub max_tokens: u64,
    #[arg(long = "decoder-tokens", default_value_t = 400)]
    pub decoder_tokens: u64,
    #[arg(long = "cancel-tokens", default_value_t = 500_000)]
    pub cancel_tokens: usize,
    #[arg(long = "dual-tokens", default_value_t = 40_000)]
    pub dual_tokens: usize,
    #[arg(long = "kill-tokens", default_value_t = 200_000)]
    pub kill_tokens: usize,
    #[arg(long = "floor-tokens", default_value_t = 200_000)]
    pub floor_tokens: usize,
    #[arg(long, default_value = "16000,40000,100000")]
    pub sizes: String,
    #[arg(long = "api-key-file")]
    pub api_key_file: Option<PathBuf>,
}

#[derive(Args)]
pub struct SpillRunArgs {
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value_t = 1800.0)]
    pub timeout: f64,
    #[arg(long, default_value = "")]
    pub label: String,
    #[arg(long)]
    pub baseline: PathBuf,
    #[arg(long, value_parser = ["floor", "main", "postkill", "postrestart", "prekill", "prerestart"])]
    pub phase: String,
    #[arg(long)]
    pub receipt: Option<PathBuf>,
    #[arg(long)]
    pub checks: Option<String>,
    #[arg(long = "rank0-dir")]
    pub rank0_dir: Option<String>,
    #[arg(long = "rank1-dir")]
    pub rank1_dir: Option<String>,
    #[arg(long = "rank0-shell", default_value = "")]
    pub rank0_shell: String,
    #[arg(long = "rank1-shell", default_value = "")]
    pub rank1_shell: String,
    #[arg(long)]
    pub sudo: bool,
    #[arg(long = "fault-rank")]
    pub fault_rank: Option<usize>,
    #[arg(long = "max-fillers", default_value_t = 120)]
    pub max_fillers: usize,
    #[arg(long, default_value_t = 24)]
    pub fillers: usize,
    #[arg(long = "filler-tokens", default_value_t = 32_000)]
    pub filler_tokens: usize,
    #[arg(long = "commit-timeout", default_value_t = 300.0)]
    pub commit_timeout: f64,
    #[arg(long = "health-timeout", default_value_t = 900.0)]
    pub health_timeout: f64,
    #[arg(long = "cancel-wait", default_value_t = 120.0)]
    pub cancel_wait: f64,
    #[arg(long = "cancel-delay", default_value_t = 0.0)]
    pub cancel_delay: f64,
    #[arg(long = "max-gap", default_value_t = 0.0)]
    pub max_gap: f64,
    #[arg(long = "spill-min-tokens", default_value_t = 8192)]
    pub spill_min_tokens: usize,
    #[arg(long = "kill-cmd")]
    pub kill_cmd: Option<String>,
    #[arg(long = "resolve-cmd")]
    pub resolve_cmd: Option<String>,
    #[arg(long)]
    pub state: Option<PathBuf>,
    #[arg(long = "floor-margin-bytes", default_value_t = 0)]
    pub floor_margin_bytes: usize,
    #[arg(long)]
    pub pictures: bool,
    #[arg(long = "allow-weak")]
    pub allow_weak: bool,
    #[arg(long)]
    pub strict: bool,
    #[arg(long = "bytes-per-position", default_value_t = BYTES_PER_POSITION)]
    pub bytes_per_position: usize,
    #[arg(long = "small-state-bytes", default_value_t = SMALL_STATE_BYTES)]
    pub small_state_bytes: usize,
    #[arg(long = "api-key-file")]
    pub api_key_file: Option<PathBuf>,
}

#[derive(Args)]
pub struct SpillEstimateArgs {
    #[arg(long)]
    pub baseline: PathBuf,
    #[arg(long, default_value = "F1")]
    pub item: String,
    #[arg(long)]
    pub bytes: bool,
    #[arg(long = "bytes-per-position", default_value_t = BYTES_PER_POSITION)]
    pub bytes_per_position: usize,
    #[arg(long = "small-state-bytes", default_value_t = SMALL_STATE_BYTES)]
    pub small_state_bytes: usize,
}

#[derive(Args)]
pub struct SpillDropArgs {
    #[arg(long)]
    pub state: PathBuf,
    #[arg(long, default_value = "MM")]
    pub item: String,
    #[arg(long = "rank1-dir")]
    pub rank1_dir: String,
    #[arg(long = "rank1-shell", default_value = "")]
    pub rank1_shell: String,
    #[arg(long)]
    pub sudo: bool,
}

fn doc(salt: &str, name: &str, tokens: usize, seed: i64) -> String {
    let sum: i64 = name.chars().map(|c| c as i64).sum();
    format!(
        "[ref {salt}-{name}]\n{}",
        words(words_for(tokens), seed + sum * 7919)
    )
}

fn picture_data_url(name: &str) -> String {
    let k: u32 = name.chars().map(|c| c as u32).sum();
    let col = [
        ((k * 37) % 256) as u8,
        ((k * 91) % 256) as u8,
        ((k * 53) % 256) as u8,
    ];
    let cx = 16 + (k % 32) as i32;
    let img = encode_rgb_png(64, 64, |x, y| {
        let dx = x as i32 - cx;
        let dy = y as i32 - 32;
        if dx * dx + dy * dy < 300 {
            col
        } else {
            [255, 255, 255]
        }
    });
    format!("data:image/png;base64,{}", STANDARD.encode(&img))
}

fn msgs(salt: &str, name: &str, tokens: usize, seed: i64, q: &str) -> Value {
    if name.starts_with('P') {
        json!([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": picture_data_url(name)}},
            {"type": "text", "text": format!("{}\n\n{q}", doc(salt, name, tokens, seed))}
        ]}])
    } else if name == DECODER {
        json!([{"role": "user", "content": format!("[ref {salt}-dec] Write a long, detailed story about a lighthouse keeper who finds a map. Do not stop early.")}])
    } else {
        json!([{"role": "user", "content": format!("{}\n\n{q}", doc(salt, name, tokens, seed))}])
    }
}

pub fn estimate_bytes(tokens: usize, bytes_per_position: usize, small: usize) -> Value {
    let at = tokens.saturating_sub(KEEP_WINDOW) / BLOCK_TOKENS * BLOCK_TOKENS;
    let blocks = at / BLOCK_TOKENS;
    let need = at * bytes_per_position + small + 4 * at + 256 + 80 * blocks;
    json!({"tokens": tokens, "at": at, "blocks": blocks, "need_bytes": need})
}

fn load_state(path: Option<&Path>) -> Value {
    let Some(p) = path else { return json!({}) };
    fs::read_to_string(p)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}))
}

fn save_state(path: Option<&Path>, st: &Value) {
    if let Some(p) = path {
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(p, serde_json::to_string_pretty(st).unwrap_or_default());
    }
}

fn run_estimate(a: SpillEstimateArgs) -> i32 {
    let base: Value = match fs::read_to_string(&a.baseline)
        .and_then(|s| serde_json::from_str(&s).map_err(Into::into))
    {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot read {}: {e}", a.baseline.display());
            return 1;
        }
    };
    let tokens = base["meta"]["items"][&a.item].as_u64().unwrap_or(0) as usize;
    let e = estimate_bytes(tokens, a.bytes_per_position, a.small_state_bytes);
    if a.bytes {
        println!("{}", e["need_bytes"].as_u64().unwrap_or(0));
    } else {
        println!("{e}");
    }
    0
}

fn run_drop(a: SpillDropArgs) -> i32 {
    let mut st = load_state(Some(&a.state));
    let man: Vec<String> = st["files"][&a.item]["1"]
        .as_object()
        .map(|m| {
            let mut v: Vec<String> = m
                .keys()
                .filter(|p| kind_of(p) == "manifest")
                .cloned()
                .collect();
            v.sort();
            v
        })
        .unwrap_or_default();
    if man.is_empty() {
        eprintln!(
            "drop: no manifest of {} on rank 1 in {}",
            a.item,
            a.state.display()
        );
        st["dropped"] = json!({"item": a.item, "error": "nothing to drop"});
        save_state(Some(&a.state), &st);
        return 1;
    }
    let target = man.last().unwrap().clone();
    let d = RankDir::new(&a.rank1_dir, &a.rank1_shell, a.sudo, 1);
    let mut extra = Map::new();
    extra.insert("path".to_string(), json!(target));
    match d.run("remove", extra) {
        Ok(mut info) => {
            if let Some(m) = info.as_object_mut() {
                m.insert("item".to_string(), json!(a.item));
                m.insert("rank".to_string(), json!(1));
                m.insert("time".to_string(), json!(now_iso8601()));
            }
            st["dropped"] = info;
            save_state(Some(&a.state), &st);
            eprintln!("drop: removed rank 1's {target} ({})", a.item);
            0
        }
        Err(e) => {
            eprintln!("drop: {e}");
            st["dropped"] = json!({"item": a.item, "error": e});
            save_state(Some(&a.state), &st);
            1
        }
    }
}

async fn run_record(a: SpillRecordArgs) -> i32 {
    let key = match resolve_strict(a.api_key_file.as_deref()) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let model = a
        .model
        .or_else(|| env::var("MODEL").ok())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let srv = Server::new(
        &a.url.unwrap_or_else(default_url),
        &model,
        a.timeout,
        a.seed,
        key,
    );
    if !srv.alive().await {
        eprintln!("record: {} not healthy", srv.url);
        return 2;
    }
    let salt = a.salt.unwrap_or_else(|| random_hex(6));
    let sizes: Vec<usize> = a
        .sizes
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    let mut its: BTreeMap<String, usize> = BTreeMap::new();
    for name in [
        "E1", "CB", "CS", "TB", "TS", "RS", "X1", "D1", "D2", "P1", "R1", "MM", "P2", "K1", "K2",
        "F1",
    ] {
        its.insert(name.to_string(), ITEM_TOKENS);
    }
    its.insert("X1".to_string(), a.cancel_tokens);
    its.insert("D1".to_string(), a.dual_tokens);
    its.insert("D2".to_string(), a.dual_tokens);
    its.insert("K2".to_string(), a.kill_tokens);
    its.insert("F1".to_string(), a.floor_tokens);
    for &z in &sizes {
        its.insert(format!("Z{z}"), z);
    }
    its.insert(DECODER.to_string(), 400);

    let mut rows = Map::new();
    let mut errors = Vec::new();
    let mut weak = Vec::new();
    for (name, &tokens) in &its {
        let entry = if name == DECODER {
            let s = srv
                .chat(&msgs(&salt, name, tokens, a.seed, ""), a.decoder_tokens)
                .await;
            json!({"seed": s})
        } else {
            let s = srv
                .chat(&msgs(&salt, name, tokens, a.seed, Q_SEED), a.max_tokens)
                .await;
            let h = srv
                .chat(&msgs(&salt, name, tokens, a.seed, Q_HIT), a.max_tokens)
                .await;
            json!({"seed": s, "hit": h})
        };
        if entry
            .as_object()
            .unwrap()
            .values()
            .any(|v| v.get("error").is_some())
        {
            errors.push(name.clone());
        }
        if entry
            .as_object()
            .unwrap()
            .values()
            .any(|v| v.get("error").is_none() && v["token_sha_source"].as_str() != Some("server"))
        {
            weak.push(name.clone());
        }
        rows.insert(name.clone(), entry);
    }
    let doc_v = json!({
        "meta": {
            "tool": "spill_check", "schema": 1, "salt": salt, "seed": a.seed,
            "model": model, "url": srv.url, "max_tokens": a.max_tokens,
            "decoder_tokens": a.decoder_tokens, "items": its, "sizes": sizes,
            "label": a.label, "weak": weak, "time": now_iso8601(),
            "health": srv.health().await
        },
        "rows": Value::Object(rows),
    });
    if let Some(p) = a.out.parent() {
        let _ = fs::create_dir_all(p);
    }
    let _ = fs::write(
        &a.out,
        serde_json::to_string_pretty(&doc_v).unwrap_or_default(),
    );
    eprintln!(
        "wrote {}: {} items, errors {errors:?}",
        a.out.display(),
        its.len()
    );
    if errors.is_empty() {
        0
    } else {
        1
    }
}

struct SpillRunner {
    a: SpillRunArgs,
    base: Value,
    salt: String,
    seed: i64,
    its: BTreeMap<String, usize>,
    mt: u64,
    dt: u64,
    srv: Server,
    rc: Receipt,
    ranks: BTreeMap<usize, RankDir>,
    files: BTreeMap<String, BTreeMap<usize, BTreeMap<String, Value>>>,
}

impl SpillRunner {
    fn m(&self, name: &str, q: &str) -> Value {
        msgs(&self.salt, name, self.its[name], self.seed, q)
    }

    async fn hit(&self, name: &str) -> Value {
        self.srv.chat(&self.m(name, Q_HIT), self.mt).await
    }

    fn same(&self, name: &str, kind: &str, row: &Value) -> (bool, String) {
        same_reply(
            self.base["rows"].get(name).and_then(|x| x.get(kind)),
            row,
            self.a.allow_weak,
        )
    }

    async fn wait_commit(&self, before: &Counters, n: f64) -> BTreeMap<String, f64> {
        let deadline = Instant::now() + Duration::from_secs_f64(self.a.commit_timeout);
        loop {
            let c = self.srv.counters().await;
            let rd = rank_deltas(before, &c, "writes_committed");
            let d: BTreeMap<String, f64> = if !rd.is_empty() {
                rd.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
            } else if let Some(v) = delta(before, &c, "spill.writes_committed") {
                [("all".to_string(), v)].into_iter().collect()
            } else {
                BTreeMap::new()
            };
            if !d.is_empty()
                && d.values().copied().fold(f64::INFINITY, f64::min) >= n
                && c.get("spill.writes_pending").copied().unwrap_or(0.0) == 0.0
            {
                return d;
            }
            if Instant::now() >= deadline {
                return d;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    async fn seed_item(&mut self, name: &str) {
        let before: BTreeMap<usize, BTreeMap<String, Value>> = self
            .ranks
            .iter()
            .map(|(&r, d)| (r, d.list().unwrap_or_default()))
            .collect();
        let c0 = self.srv.counters().await;
        let row = self.srv.chat(&self.m(name, Q_SEED), self.mt).await;
        let (ok, detail) = self.same(name, "seed", &row);
        let committed = self.wait_commit(&c0, 1.0).await;
        let mut new_map = BTreeMap::new();
        for (&r, d) in &self.ranks {
            let nf = d.new_files(&before[&r]).unwrap_or_default();
            new_map.insert(r, nf);
        }
        let counts: BTreeMap<usize, usize> = new_map.iter().map(|(&r, m)| (r, m.len())).collect();
        self.files.insert(name.to_string(), new_map);
        let mut ex = Map::new();
        ex.insert("row".to_string(), row);
        ex.insert("committed".to_string(), json!(committed));
        self.rc.check(
            &format!("seed {name} == baseline"),
            ok,
            &format!("{detail}; committed {committed:?}; new files {counts:?}"),
            ex,
        );
        if name.starts_with('P') {
            let c1 = self.srv.counters().await;
            let pk = rank_deltas(&c0, &c1, "pictures_kept");
            let pk_map: BTreeMap<String, f64> = if !pk.is_empty() {
                pk.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
            } else if let Some(v) = delta(&c0, &c1, "spill.pictures_kept") {
                [("all".to_string(), v)].into_iter().collect()
            } else {
                BTreeMap::new()
            };
            let ok_pk =
                !pk_map.is_empty() && pk_map.values().copied().fold(f64::INFINITY, f64::min) >= 1.0;
            let mut ex2 = Map::new();
            ex2.insert("pictures_kept".to_string(), json!(pk_map));
            self.rc.check(
                &format!("seed {name} kept with its picture"),
                ok_pk,
                &format!("pictures_kept {pk_map:?}"),
                ex2,
            );
        }
    }

    async fn evict(&self) -> Value {
        let c0 = self.srv.counters().await;
        let entries = c0.get("prompt_cache.entries").copied();
        let target = entries.map(|e| e as usize);
        let mut sent = 0usize;
        while sent < self.a.max_fillers {
            let nm = format!("fill-{}", random_hex(4));
            let m = json!([{"role": "user", "content": format!("{}\n\n{Q_SEED}", doc(&format!("{}-f", self.salt), &nm, self.a.filler_tokens, self.seed))}]);
            let _ = self.srv.chat(&m, 8).await;
            sent += 1;
            let d = delta(&c0, &self.srv.counters().await, "prompt_cache.evicted");
            match target {
                None => {
                    if sent >= self.a.fillers {
                        break;
                    }
                }
                Some(t) => {
                    if d.unwrap_or(0.0) >= t as f64 {
                        break;
                    }
                }
            }
        }
        let d = delta(&c0, &self.srv.counters().await, "prompt_cache.evicted");
        let complete = target.map(|t| d.unwrap_or(0.0) >= t as f64).unwrap_or(true);
        json!({"fillers": sent, "evicted": d, "entries_before": entries, "complete": complete})
    }

    fn file_of(&self, name: &str, rank: usize, kind: &str) -> Option<String> {
        let m = self.files.get(name)?.get(&rank)?;
        let mut cands: Vec<String> = m.keys().filter(|p| kind_of(p) == kind).cloned().collect();
        cands.sort();
        cands.pop()
    }

    async fn disk_hit(&mut self, name: &str, check: &str, mut extra: Map<String, Value>) {
        let c0 = self.srv.counters().await;
        let row = self.hit(name).await;
        let c1 = self.srv.counters().await;
        let (ok, detail) = self.same(name, "hit", &row);
        let mut keys = vec![
            "prompt_cache.disk_hits_committed",
            "prompt_cache.restored_tokens",
            "prompt_cache.memory_hits",
            "prompt_cache.restore_wait_ns",
        ];
        keys.extend_from_slice(FAIL_KEYS);
        let d = deltas(&c0, &c1, keys);
        let Some(dh) = d["prompt_cache.disk_hits_committed"] else {
            extra.insert("row".to_string(), row);
            self.rc
                .add(check, "FAIL", "no prompt_cache.disk_hits_committed", extra);
            return;
        };
        let rt = d["prompt_cache.restored_tokens"].unwrap_or(0.0);
        let from_disk = dh >= 1.0 && rt > 0.0;
        let mut why = Vec::new();
        if !from_disk {
            why.push(format!("not from disk (disk_hits_committed +{dh})"));
        }
        if !ok {
            why.push(detail.clone());
        }
        extra.insert("row".to_string(), row);
        extra.insert("deltas".to_string(), json!(d));
        self.rc.check(
            check,
            from_disk && ok,
            &if why.is_empty() {
                format!("from disk ({rt} tokens), {detail}")
            } else {
                why.join("; ")
            },
            extra,
        );
    }

    async fn fault(&mut self, check: &str, name: &str, rank: usize, kind: &str, op: &str) {
        let Some(d_dir) = self.ranks.get(&rank).cloned() else {
            self.rc.add(
                check,
                "SKIP",
                &format!("no directory for rank {rank}"),
                Map::new(),
            );
            return;
        };
        let Some(path) = self.file_of(name, rank, kind) else {
            let st = if self
                .files
                .get(name)
                .and_then(|m| m.get(&rank))
                .map(|m| m.is_empty())
                .unwrap_or(true)
            {
                "SKIP"
            } else {
                "FAIL"
            };
            self.rc.add(
                check,
                st,
                &format!("no {kind} file of {name} on rank {rank}"),
                Map::new(),
            );
            return;
        };
        let mut op_args = Map::new();
        op_args.insert("path".to_string(), json!(path));
        let info = match d_dir.run(op, op_args) {
            Ok(v) => v,
            Err(e) => {
                self.rc.add(
                    check,
                    "FAIL",
                    &format!("could not {op} {path} on rank {rank}: {e}"),
                    Map::new(),
                );
                return;
            }
        };
        let c0 = self.srv.counters().await;
        let row = self.hit(name).await;
        let c1 = self.srv.counters().await;
        let (ok, detail) = self.same(name, "hit", &row);
        let mut keys = vec![
            "prompt_cache.disk_hits_committed",
            "prompt_cache.restores_begun",
        ];
        keys.extend_from_slice(FAIL_KEYS);
        let d = deltas(&c0, &c1, keys);
        let refused: f64 = FAIL_KEYS
            .iter()
            .filter_map(|k| d.get(*k).and_then(|v| *v))
            .sum();
        let no_hit = d["prompt_cache.disk_hits_committed"].unwrap_or(0.0) == 0.0;
        let good = ok && no_hit && refused >= 1.0;
        let mut why = Vec::new();
        if !no_hit {
            why.push("a damaged point was restored".to_string());
        }
        if refused < 1.0 {
            why.push("no refusal/CRC/size/presence/failed counter moved".to_string());
        }
        if !ok {
            why.push(detail.clone());
        }
        let mut ex = Map::new();
        ex.insert("fault".to_string(), info);
        ex.insert("row".to_string(), row);
        ex.insert("deltas".to_string(), json!(d));
        self.rc.check(
            check,
            good,
            &if why.is_empty() {
                format!("{op} rank {rank} {path}: refused, {detail}")
            } else {
                why.join("; ")
            },
            ex,
        );
    }

    async fn cancel(&mut self) {
        let name = "X1";
        let c0 = self.srv.counters().await;
        let cancel = Arc::new(Notify::new());
        let s = self.srv.clone();
        let m = self.m(name, Q_HIT);
        let mt = self.mt;
        let cn = cancel.clone();
        let handle =
            tokio::spawn(async move { s.stream(&m, mt, None, false, Some(cn), None, false).await });

        let deadline = Instant::now() + Duration::from_secs_f64(self.a.cancel_wait);
        let mut seen = false;
        while Instant::now() < deadline {
            if handle.is_finished() {
                break;
            }
            let c = self.srv.counters().await;
            let begun = delta(&c0, &c, "prompt_cache.restores_begun").unwrap_or(0.0) >= 1.0
                || c.get("prompt_cache.restores_active")
                    .copied()
                    .unwrap_or(0.0)
                    >= 1.0
                || c.get("prompt_cache.restore_blocks_done")
                    .copied()
                    .unwrap_or(0.0)
                    < c.get("prompt_cache.restore_blocks_total")
                        .copied()
                        .unwrap_or(0.0);
            if begun {
                seen = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let active = !handle.is_finished();
        if active && self.a.cancel_delay > 0.0 {
            tokio::time::sleep(Duration::from_secs_f64(self.a.cancel_delay)).await;
        }
        cancel.notify_waiters();
        let cancelled_row = handle.await.unwrap_or(Value::Null);

        let settle_end = Instant::now() + Duration::from_secs(120);
        let mut settled = false;
        while Instant::now() < settle_end {
            if self
                .srv
                .counters()
                .await
                .get("prompt_cache.restores_active")
                .copied()
                .unwrap_or(0.0)
                == 0.0
            {
                settled = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        let alive = self.srv.alive().await;
        let c1 = self.srv.counters().await;
        let d = deltas(
            &c0,
            &c1,
            [
                "prompt_cache.restores_cancelled",
                "prompt_cache.restores_begun",
                "prompt_cache.disk_hits_committed",
                "spill.staging_bytes",
            ],
        );
        let again = self.hit(name).await;
        let (ok, detail) = self.same(name, "hit", &again);
        let mut ex = Map::new();
        ex.insert("deltas".to_string(), json!(d));
        ex.insert("cancelled_row".to_string(), cancelled_row);
        ex.insert("again".to_string(), again);
        ex.insert("restore_seen".to_string(), json!(seen && active));
        if !alive || !settled {
            self.rc.add(
                "cancel",
                "FAIL",
                &format!("alive={alive}, settled={settled}"),
                ex,
            );
        } else if !ok {
            self.rc
                .add("cancel", "FAIL", &format!("after cancel: {detail}"), ex);
        } else if d["prompt_cache.restores_cancelled"].unwrap_or(0.0) < 1.0 {
            self.rc
                .add("cancel", "SKIP", "no restore was cancelled", ex);
        } else {
            self.rc.add(
                "cancel",
                "PASS",
                &format!("restore cancelled mid-flight, {detail}"),
                ex,
            );
        }
    }

    async fn dual(&mut self) {
        let c0 = self.srv.counters().await;
        let first = Arc::new(Notify::new());
        let sd = self.srv.clone();
        let md = self.m(DECODER, "");
        let dt = self.dt;
        let fn_ = first.clone();
        let dec_h = tokio::spawn(async move {
            sd.stream(&md, dt, None, false, None, Some(fn_), false)
                .await
        });
        let _ =
            tokio::time::timeout(Duration::from_secs_f64(self.a.timeout), first.notified()).await;
        let t0 = Instant::now();
        let s1 = self.srv.clone();
        let m1 = self.m("D1", Q_HIT);
        let s2 = self.srv.clone();
        let m2 = self.m("D2", Q_HIT);
        let mt = self.mt;
        let h1 = tokio::spawn(async move { s1.chat(&m1, mt).await });
        let h2 = tokio::spawn(async move { s2.chat(&m2, mt).await });
        let (r1, r2) = (h1.await.unwrap(), h2.await.unwrap());
        let restores_s = t0.elapsed().as_secs_f64();
        let rd = dec_h.await.unwrap();
        let c1 = self.srv.counters().await;
        let d = deltas(
            &c0,
            &c1,
            [
                "prompt_cache.disk_hits_committed",
                "prompt_cache.restores_begun",
                "prompt_cache.restore_wait_max_ns",
            ],
        );
        let (ok1, d1) = self.same("D1", "hit", &r1);
        let (ok2, d2) = self.same("D2", "hit", &r2);
        let (okd, dd) = same_reply(self.base["rows"][DECODER].get("seed"), &rd, true);
        let both = d["prompt_cache.disk_hits_committed"].unwrap_or(0.0) >= 2.0;
        let mut why = Vec::new();
        if !both {
            why.push("not both from disk".to_string());
        }
        if !ok1 {
            why.push(format!("D1: {d1}"));
        }
        if !ok2 {
            why.push(format!("D2: {d2}"));
        }
        if !okd {
            why.push(format!("decoder: {dd}"));
        }
        let gap = rd["max_gap"].as_f64();
        if self.a.max_gap > 0.0 && gap.map(|g| g > self.a.max_gap).unwrap_or(false) {
            why.push(format!("decoder max_gap {:?} > {}", gap, self.a.max_gap));
        }
        let mut ex = Map::new();
        ex.insert("d1".to_string(), r1);
        ex.insert("d2".to_string(), r2);
        ex.insert("decoder".to_string(), rd);
        ex.insert("deltas".to_string(), json!(d));
        self.rc.check(
            "dual-restore",
            why.is_empty(),
            &if why.is_empty() {
                format!("both from disk in {restores_s:.2}s while decoding; gap {gap:?}s")
            } else {
                why.join("; ")
            },
            ex,
        );
    }

    async fn sizes(&mut self) {
        let names: Vec<String> = self
            .its
            .keys()
            .filter(|n| n.starts_with('Z'))
            .cloned()
            .collect();
        if names.is_empty() {
            self.rc.add("sizes", "SKIP", "no --sizes items", Map::new());
            return;
        }
        let mut out = Vec::new();
        let mut bad = Vec::new();
        for n in &names {
            let c0 = self.srv.counters().await;
            let row = self
                .srv
                .stream(&self.m(n, Q_HIT), self.mt, None, false, None, None, false)
                .await;
            let c1 = self.srv.counters().await;
            let (ok, detail) = same_reply(self.base["rows"][n].get("hit"), &row, true);
            let dh = delta(&c0, &c1, "prompt_cache.disk_hits_committed").unwrap_or(0.0);
            out.push(json!({"item": n, "tokens": self.its[n], "restore_ttft": row["ttft"], "from_disk": dh > 0.0, "equal": ok}));
            if !ok {
                bad.push(format!("{n}: {detail}"));
            } else if dh <= 0.0 && self.its[n] >= self.a.spill_min_tokens + 2048 + 128 {
                bad.push(format!("{n}: not from disk"));
            }
        }
        let mut ex = Map::new();
        ex.insert("sizes".to_string(), Value::Array(out));
        self.rc.check(
            "sizes",
            bad.is_empty(),
            &if bad.is_empty() {
                format!("{} sizes restored, replies equal", names.len())
            } else {
                bad.join("; ")
            },
            ex,
        );
    }

    async fn settled(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(120);
        let mut ok = false;
        let mut last = Counters::new();
        while Instant::now() < deadline {
            let c = self.srv.counters().await;
            let clear = c
                .get("prompt_cache.restores_active")
                .copied()
                .unwrap_or(0.0)
                == 0.0
                && c.get("prompt_cache.parked").copied().unwrap_or(0.0) == 0.0
                && c.get("spill.writes_pending").copied().unwrap_or(0.0) == 0.0;
            last = c;
            if clear {
                ok = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let alive = self.srv.alive().await;
        self.rc.check(
            "settled",
            ok && alive,
            &format!(
                "restores_active={:?} parked={:?} writes_pending={:?}",
                last.get("prompt_cache.restores_active"),
                last.get("prompt_cache.parked"),
                last.get("spill.writes_pending")
            ),
            Map::new(),
        );
    }

    async fn temporaries(&mut self, check: &str) {
        if self.ranks.is_empty() {
            self.rc
                .add(check, "SKIP", "no rank directory given", Map::new());
            return;
        }
        // Let any background spill writes triggered by the preceding probe commit before scanning for stray .tmp files.
        let deadline = Instant::now() + Duration::from_secs_f64(self.a.commit_timeout);
        while Instant::now() < deadline {
            let pending = self
                .srv
                .counters()
                .await
                .get("spill.writes_pending")
                .copied()
                .unwrap_or(0.0);
            if pending <= 0.0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let mut found: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        for (&r, d) in &self.ranks {
            match d.list() {
                Ok(files) => {
                    let temps: Vec<String> = files
                        .keys()
                        .filter(|p| kind_of(p) == "temp")
                        .cloned()
                        .collect();
                    found.insert(r, temps);
                }
                Err(e) => {
                    self.rc.add(
                        check,
                        "FAIL",
                        &format!("could not list rank {r}: {e}"),
                        Map::new(),
                    );
                    return;
                }
            }
        }
        let n: usize = found.values().map(Vec::len).sum();
        let mut ex = Map::new();
        ex.insert("temporaries".to_string(), json!(found));
        self.rc
            .check(check, n == 0, &format!("temporaries: {n}"), ex);
    }

    async fn phase_main(&mut self) {
        let checks: Vec<String> = match &self.a.checks {
            Some(s) => s
                .split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect(),
            None => MAIN_CHECKS.iter().map(|s| s.to_string()).collect(),
        };
        let mut names: Vec<String> = checks
            .iter()
            .filter_map(|c| check_item(c).map(String::from))
            .collect();
        if checks.iter().any(|c| c == "dual-restore") {
            names.push("D2".to_string());
        }
        if checks.iter().any(|c| c == "sizes") {
            names.extend(self.its.keys().filter(|n| n.starts_with('Z')).cloned());
        }
        for n in names {
            self.seed_item(&n).await;
        }
        let ev = self.evict().await;
        self.rc.doc.insert("eviction".to_string(), ev.clone());
        let r_corrupt = self.a.fault_rank.unwrap_or(0);
        let r_trunc = self
            .a
            .fault_rank
            .unwrap_or(if self.ranks.contains_key(&1) { 1 } else { 0 });
        let r_rem = if self.ranks.contains_key(&1) { 1 } else { 0 };

        for chk in checks {
            let mut ev_map = Map::new();
            ev_map.insert("eviction".to_string(), ev.clone());
            match chk.as_str() {
                "evict" => self.disk_hit("E1", "evict", ev_map).await,
                "corrupt-block" => self.fault(&chk, "CB", r_corrupt, "block", "corrupt").await,
                "corrupt-small" => self.fault(&chk, "CS", r_corrupt, "small", "corrupt").await,
                "truncate-block" => self.fault(&chk, "TB", r_trunc, "block", "truncate").await,
                "truncate-small" => self.fault(&chk, "TS", r_trunc, "small", "truncate").await,
                "remove-small" => self.fault(&chk, "RS", r_rem, "small", "remove").await,
                "pictures" => self.disk_hit("P1", "pictures", ev_map).await,
                "cancel" => self.cancel().await,
                "dual-restore" => self.dual().await,
                "sizes" => self.sizes().await,
                "settled" => self.settled().await,
                _ => self.rc.add(&chk, "FAIL", "unknown check", Map::new()),
            }
        }
    }

    async fn phase_prerestart(&mut self) {
        let mut st = load_state(self.a.state.as_deref());
        let names: Vec<&str> = if self.a.pictures {
            vec!["R1", "MM", "P2"]
        } else {
            vec!["R1", "MM"]
        };
        for &n in &names {
            self.seed_item(n).await;
        }
        let mut files_obj = Map::new();
        for &n in &names {
            let mut by_r = Map::new();
            if let Some(rm) = self.files.get(n) {
                for (r, fs_map) in rm {
                    by_r.insert(r.to_string(), json!(fs_map));
                }
            }
            files_obj.insert(n.to_string(), Value::Object(by_r));
        }
        st["files"] = Value::Object(files_obj);
        if let Some(m) = st.as_object_mut() {
            m.remove("dropped");
        }
        save_state(self.a.state.as_deref(), &st);
    }

    async fn phase_postrestart(&mut self) {
        let c = self.srv.counters().await;
        let rp = rank_values(&c, "recovered_points");
        let ok_rp = if !rp.is_empty() {
            rp.values().copied().fold(f64::INFINITY, f64::min) >= 1.0
        } else {
            c.get("spill.recovered_points").copied().unwrap_or(0.0) >= 1.0
        };
        self.rc.check(
            "recovered points",
            ok_rp,
            &format!("recovered_points {rp:?}"),
            Map::new(),
        );
        self.disk_hit("R1", "restart", Map::new()).await;
        if self.a.pictures {
            let st = load_state(self.a.state.as_deref());
            if st["files"].get("P2").is_some() {
                self.disk_hit("P2", "restart: picture point restores", Map::new())
                    .await;
            } else {
                self.rc.add(
                    "restart: picture point restores",
                    "FAIL",
                    "no P2 in --state",
                    Map::new(),
                );
            }
        }
        let st = load_state(self.a.state.as_deref());
        let dr = &st["dropped"];
        if dr["item"].as_str() != Some("MM")
            || dr.get("error").is_some()
            || dr.get("path").is_none()
        {
            self.rc.add(
                "missing-rank1",
                "SKIP",
                "MM manifest not dropped",
                Map::new(),
            );
        } else {
            let c0 = self.srv.counters().await;
            let row = self.hit("MM").await;
            let c1 = self.srv.counters().await;
            let (ok, detail) = self.same("MM", "hit", &row);
            let mut keys = vec![
                "prompt_cache.disk_hits_committed",
                "prompt_cache.restores_begun",
            ];
            keys.extend_from_slice(FAIL_KEYS);
            let d = deltas(&c0, &c1, keys);
            let pts = rank_values(&c1, "points");
            let fewer = pts.len() >= 2
                && pts.get(&1).copied().unwrap_or(0.0) < pts.get(&0).copied().unwrap_or(0.0);
            let refused: f64 = FAIL_KEYS
                .iter()
                .filter_map(|k| d.get(*k).and_then(|v| *v))
                .sum();
            let no_hit = d["prompt_cache.disk_hits_committed"].unwrap_or(0.0) == 0.0;
            let mut why = Vec::new();
            if !no_hit {
                why.push("restored from disk though rank 1 lacks it".to_string());
            }
            if refused < 1.0 && !fewer {
                why.push(format!("no evidence rank 1 lacks it (points {pts:?})"));
            }
            if !ok {
                why.push(detail.clone());
            }
            self.rc.check(
                "missing-rank1",
                why.is_empty(),
                &if why.is_empty() {
                    format!("not restored, {detail}")
                } else {
                    why.join("; ")
                },
                Map::new(),
            );
        }
        self.temporaries("no temporaries after a clean restart")
            .await;
    }

    async fn phase_prekill(&mut self) {
        let mut st = load_state(self.a.state.as_deref());
        self.seed_item("K1").await;
        if let Some(rcmd) = self.a.resolve_cmd.clone() {
            let out = Command::new("sh").args(["-c", &rcmd]).output();
            let ok = out.as_ref().map(|o| o.status.success()).unwrap_or(false);
            if !self
                .rc
                .check("kill: PIDs resolved", ok, &format!("{rcmd:?}"), Map::new())
            {
                return;
            }
        }
        let c0 = self.srv.counters().await;
        let s = self.srv.clone();
        let m = self.m("K2", Q_SEED);
        let mt = self.mt;
        let handle =
            tokio::spawn(async move { s.stream(&m, mt, None, false, None, None, false).await });

        let deadline = Instant::now() + Duration::from_secs_f64(self.a.commit_timeout);
        let mut at_kill = Counters::new();
        while Instant::now() < deadline {
            if handle.is_finished() {
                at_kill = self.srv.counters().await;
                break;
            }
            let c = self.srv.counters().await;
            if delta(&c0, &c, "spill.writes_queued").unwrap_or(0.0) >= 1.0
                && c.get("spill.writes_pending").copied().unwrap_or(0.0) > 0.0
            {
                at_kill = c;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let pending = at_kill.get("spill.writes_pending").copied().unwrap_or(0.0);
        let kinfo = json!({
            "writes_pending_at_kill": pending,
            "writes_pending_per_rank": rank_values(&at_kill, "writes_pending"),
            "committed_k2_before_kill": rank_deltas(&c0, &at_kill, "writes_committed"),
            "reply_done_before_kill": handle.is_finished(),
        });
        if let Some(kcmd) = self.a.kill_cmd.clone() {
            let ok = Command::new("sh")
                .args(["-c", &kcmd])
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            self.rc.check(
                "kill",
                ok,
                &format!("{kcmd:?}; pending {pending}"),
                Map::new(),
            );
        } else {
            println!("KILL NOW: kill -9 both ranks, then restart and run --phase postkill");
        }
        let _ = tokio::time::timeout(Duration::from_secs_f64(self.a.timeout), handle).await;
        st["prekill"] = kinfo.clone();
        save_state(self.a.state.as_deref(), &st);
        self.rc.doc.insert("prekill".to_string(), kinfo);
    }

    async fn phase_postkill(&mut self) {
        let st = load_state(self.a.state.as_deref());
        self.disk_hit("K1", "kill: committed point restores", Map::new())
            .await;
        let c0 = self.srv.counters().await;
        let row = self.hit("K2").await;
        let (ok, detail) = self.same("K2", "hit", &row);
        let dh = delta(
            &c0,
            &self.srv.counters().await,
            "prompt_cache.disk_hits_committed",
        )
        .unwrap_or(0.0);
        let pending = st["prekill"]["writes_pending_at_kill"]
            .as_f64()
            .unwrap_or(0.0);
        let check_name = "kill: in-flight point hits or misses safely";
        let how = format!(
            "{}; {detail}",
            if dh > 0.0 { "from disk" } else { "recomputed" }
        );
        if !ok {
            self.rc.add(check_name, "FAIL", &how, Map::new());
        } else if pending <= 0.0 {
            self.rc.add(
                check_name,
                "SKIP",
                &format!("K2 write not pending at kill ({how})"),
                Map::new(),
            );
        } else {
            self.rc.add(
                check_name,
                "PASS",
                &format!("killed with {pending} pending; {how}"),
                Map::new(),
            );
        }
        self.temporaries("no temporaries after recovery").await;
    }

    async fn phase_floor(&mut self) {
        let est = estimate_bytes(
            self.its["F1"],
            self.a.bytes_per_position,
            self.a.small_state_bytes,
        );
        let need = est["need_bytes"].as_u64().unwrap_or(0) as usize;
        if self.a.floor_margin_bytes > 0 && need <= self.a.floor_margin_bytes {
            self.rc.add(
                "floor: sizing",
                "FAIL",
                &format!(
                    "F1 need ({need} B) <= margin ({} B)",
                    self.a.floor_margin_bytes
                ),
                Map::new(),
            );
        }
        let c0 = self.srv.counters().await;
        let before: BTreeMap<usize, BTreeMap<String, Value>> = self
            .ranks
            .iter()
            .map(|(&r, d)| (r, d.list().unwrap_or_default()))
            .collect();
        let row = self.srv.chat(&self.m("F1", Q_SEED), self.mt).await;
        let (ok, detail) = self.same("F1", "seed", &row);
        let deadline = Instant::now() + Duration::from_secs_f64(self.a.commit_timeout);
        while Instant::now() < deadline {
            if self
                .srv
                .counters()
                .await
                .get("spill.writes_pending")
                .copied()
                .unwrap_or(0.0)
                == 0.0
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
        let c1 = self.srv.counters().await;
        let d = deltas(
            &c0,
            &c1,
            [
                "spill.no_space",
                "spill.writes_committed",
                "spill.stored_bytes",
            ],
        );
        let ns = rank_deltas(&c0, &c1, "no_space");
        let wc = rank_deltas(&c0, &c1, "writes_committed");
        let mut any_new = false;
        for (&r, d_dir) in &self.ranks {
            for p in d_dir.new_files(&before[&r]).unwrap_or_default().keys() {
                if matches!(kind_of(p), "manifest" | "block" | "small") {
                    any_new = true;
                }
            }
        }
        let refused = if !ns.is_empty() {
            ns.values().copied().fold(f64::INFINITY, f64::min) >= 1.0
                && wc.values().copied().fold(0.0f64, f64::max) == 0.0
        } else {
            d["spill.no_space"].unwrap_or(0.0) >= 1.0
                && d["spill.writes_committed"].unwrap_or(0.0) == 0.0
        };
        self.rc.check(
            "floor",
            refused && !any_new && ok,
            &format!("no_space={ns:?} committed={wc:?} any_new={any_new}; {detail}"),
            Map::new(),
        );
        let r2 = self.hit("F1").await;
        let (ok2, d2) = self.same("F1", "hit", &r2);
        self.rc
            .check("floor: memory still serves", ok2, &d2, Map::new());
    }

    async fn go(mut self) -> i32 {
        if !self.srv.wait_healthy(self.a.health_timeout).await {
            self.rc.add("health", "FAIL", "not healthy", Map::new());
            return self.rc.finish(self.a.receipt.as_deref(), self.a.strict);
        }
        let c0 = self.srv.counters().await;
        self.rc.doc.insert("counters_before".to_string(), json!(c0));
        if !c0.contains_key("prompt_cache.disk_lookups") && !c0.contains_key("spill.points") {
            self.rc.add(
                "tier on",
                "FAIL",
                "no disk tier counters reported",
                Map::new(),
            );
            return self.rc.finish(self.a.receipt.as_deref(), self.a.strict);
        }
        match self.a.phase.as_str() {
            "main" => self.phase_main().await,
            "prerestart" => self.phase_prerestart().await,
            "postrestart" => self.phase_postrestart().await,
            "prekill" => self.phase_prekill().await,
            "postkill" => self.phase_postkill().await,
            "floor" => self.phase_floor().await,
            _ => {}
        }
        let c_after = self.srv.counters().await;
        self.rc
            .doc
            .insert("counters_after".to_string(), json!(c_after));
        self.rc.finish(self.a.receipt.as_deref(), self.a.strict)
    }
}

pub async fn run(a: SpillCheckArgs) -> i32 {
    match a.mode {
        SpillMode::Estimate(e) => run_estimate(e),
        SpillMode::Drop(d) => run_drop(d),
        SpillMode::Record(r) => run_record(r).await,
        SpillMode::Run(u) => {
            let key = match resolve_strict(u.api_key_file.as_deref()) {
                Ok(k) => k,
                Err(e) => {
                    eprintln!("{e}");
                    return 2;
                }
            };
            let base: Value = match fs::read_to_string(&u.baseline)
                .and_then(|s| serde_json::from_str(&s).map_err(Into::into))
            {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("cannot read baseline {}: {e}", u.baseline.display());
                    return 2;
                }
            };
            let m = &base["meta"];
            let salt = m["salt"].as_str().unwrap_or("").to_string();
            let seed = m["seed"].as_i64().unwrap_or(1234);
            let mt = m["max_tokens"].as_u64().unwrap_or(48);
            let dt = m["decoder_tokens"].as_u64().unwrap_or(400);
            let mut its = BTreeMap::new();
            if let Some(im) = m["items"].as_object() {
                for (k, v) in im {
                    its.insert(k.clone(), v.as_u64().unwrap_or(0) as usize);
                }
            }
            let model = u
                .model
                .clone()
                .or_else(|| m["model"].as_str().map(String::from))
                .unwrap_or_else(|| DEFAULT_MODEL.to_string());
            let url = u.url.clone().unwrap_or_else(default_url);
            let srv = Server::new(&url, &model, u.timeout, seed, key.clone());
            let mut meta = Map::new();
            meta.insert("phase".to_string(), json!(u.phase));
            meta.insert("url".to_string(), json!(srv.url));
            meta.insert(
                "baseline".to_string(),
                json!(u.baseline.display().to_string()),
            );
            meta.insert("salt".to_string(), json!(salt));
            meta.insert("label".to_string(), json!(u.label));
            let rc = Receipt::new("spill_check", meta, key);
            let mut ranks = BTreeMap::new();
            if let Some(d0) = &u.rank0_dir {
                ranks.insert(0, RankDir::new(d0, &u.rank0_shell, u.sudo, 0));
            }
            if let Some(d1) = &u.rank1_dir {
                ranks.insert(1, RankDir::new(d1, &u.rank1_shell, u.sudo, 1));
            }
            let runner = SpillRunner {
                a: *u,
                base,
                salt,
                seed,
                its,
                mt,
                dt,
                srv,
                rc,
                ranks,
                files: BTreeMap::new(),
            };
            runner.go().await
        }
    }
}
