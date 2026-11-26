//! End-to-end KV sharding checks (`dsv41-tools kv-shard-check record|run`)
//! replacing `tools/kv_shard_check_e2e.py`.

use clap::{Args, Subcommand};
use dsv41_core::auth::resolve_strict;
use dsv41_core::client::{same_reply, Server};
use dsv41_core::metrics::{delta, deltas, kv_parts, kv_pool, Counters};
use dsv41_core::prose::{words, words_for};
use dsv41_core::receipt::Receipt;
use dsv41_core::{now_iso8601, random_hex, DEFAULT_MODEL};
use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Notify};

const KV_Q: &[&str] = &[
    "Summarize the text above in one sentence.",
    "Which word comes last in the text above?",
    "Give a title for the text above.",
    "List three nouns from the text above.",
    "Name two words that repeat in the text above.",
    "Reply with a number between one and nine.",
    "Say one short sentence about the text above.",
    "What is the first topic in the text above?",
    "Write one question a reader could ask about the text above.",
    "Describe the text above in five words.",
    "Pick one word from the text above and define it.",
    "How long is the text above, roughly?",
];
const Q_AFTER: &str = "After the interruption: summarize the text above in one sentence.";
const MIXED_SIZES: &[usize] = &[1200, 2600, 4100, 5300, 6800, 8200, 9900, 12000];
const FILLER: &str =
    "Archive note: The old stone bridge crosses a quiet river beside the northern forest.\n";
const SECRETS: &[(&str, &str)] = &[("ALPHA", "384912"), ("BETA", "725063"), ("GAMMA", "491857")];

fn default_url() -> String {
    env::var("API_URL").unwrap_or_else(|_| {
        format!(
            "http://127.0.0.1:{}",
            env::var("PORT").unwrap_or_else(|_| "8888".to_string())
        )
    })
}

#[derive(Args)]
pub struct KvShardArgs {
    #[command(subcommand)]
    pub mode: KvMode,
}

#[derive(Subcommand)]
pub enum KvMode {
    Record(KvRecordArgs),
    Run(KvRunArgs),
}

#[derive(Args)]
pub struct KvRecordArgs {
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value_t = 3600.0)]
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
    #[arg(long = "decode-tokens", default_value_t = 300)]
    pub decode_tokens: u64,
    #[arg(long = "owner-tokens", default_value_t = 6000)]
    pub owner_tokens: usize,
    #[arg(long = "carry-tokens", default_value_t = 64_000)]
    pub carry_tokens: usize,
    #[arg(long = "carry-rounds", default_value_t = 10)]
    pub carry_rounds: usize,
    #[arg(long = "twin-tokens", default_value_t = 96_000)]
    pub twin_tokens: usize,
    #[arg(long = "filler-tokens", default_value_t = 64_000)]
    pub filler_tokens: usize,
    #[arg(long, default_value_t = 4)]
    pub burst: usize,
    #[arg(long = "burst-tokens", default_value_t = 24_000)]
    pub burst_tokens: usize,
    #[arg(long = "move-side-tokens", default_value_t = 8000)]
    pub move_side_tokens: usize,
    #[arg(long = "move-tokens", default_value_t = 256_000)]
    pub move_tokens: usize,
    #[arg(long = "api-key-file")]
    pub api_key_file: Option<PathBuf>,
}

#[derive(Args)]
pub struct KvRunArgs {
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value_t = 3600.0)]
    pub timeout: f64,
    #[arg(long, default_value = "")]
    pub label: String,
    #[arg(long)]
    pub baseline: PathBuf,
    #[arg(long)]
    pub receipt: Option<PathBuf>,
    #[arg(long)]
    pub checks: Option<String>,
    #[arg(long = "expect-window", default_value_t = 0)]
    pub expect_window: i64,
    #[arg(long = "expect-pool", default_value_t = 0)]
    pub expect_pool: i64,
    #[arg(long = "startup-log")]
    pub startup_log: Option<PathBuf>,
    #[arg(long = "reported-pool", default_value_t = 0)]
    pub reported_pool: i64,
    #[arg(long = "reported-window", default_value_t = 0)]
    pub reported_window: i64,
    #[arg(long = "over-window", default_value_t = 1024)]
    pub over_window: i64,
    #[arg(long = "needle-sizes", default_value = "524288,max")]
    pub needle_sizes: String,
    #[arg(long = "needle-answer", default_value_t = 128)]
    pub needle_answer: u64,
    #[arg(long = "cancel-wait", default_value_t = 120.0)]
    pub cancel_wait: f64,
    #[arg(long = "cancel-after", default_value_t = 0.5)]
    pub cancel_after: f64,
    #[arg(long = "settle-wait", default_value_t = 120.0)]
    pub settle_wait: f64,
    #[arg(long = "move-wait", default_value_t = 600.0)]
    pub move_wait: f64,
    #[arg(long = "kill-cmd")]
    pub kill_cmd: Option<String>,
    #[arg(long = "kill-deadline", default_value_t = 90.0)]
    pub kill_deadline: f64,
    #[arg(long = "sample-ms", default_value_t = 100.0)]
    pub sample_ms: f64,
    #[arg(long = "health-timeout", default_value_t = 900.0)]
    pub health_timeout: f64,
    #[arg(long = "allow-weak")]
    pub allow_weak: bool,
    #[arg(long)]
    pub strict: bool,
    #[arg(long = "api-key-file")]
    pub api_key_file: Option<PathBuf>,
}

fn kv_doc(salt: &str, name: &str, tokens: usize, seed: i64) -> String {
    let sum: i64 = name.chars().map(|c| c as i64).sum();
    format!(
        "[ref {salt}-{name}]\n{}",
        words(words_for(tokens), seed + sum * 104_729)
    )
}

fn msg(text: &str, q: &str) -> Value {
    let content = if q.is_empty() {
        text.to_string()
    } else {
        format!("{text}\n\n{q}")
    };
    json!([{"role": "user", "content": content}])
}

fn parse_startup_line(path: &Path) -> Option<(i64, String, i64, i64)> {
    let text = fs::read_to_string(path).ok()?;
    let re = Regex::new(r"dsv41: pool (\d+) positions, (\w+) (\d+) a part; a conversation may use up to (\d+) tokens").ok()?;
    let cap = re.captures_iter(&text).last()?;
    Some((
        cap[1].parse().ok()?,
        cap[2].to_string(),
        cap[3].parse().ok()?,
        cap[4].parse().ok()?,
    ))
}

struct Plan {
    salt: String,
    seed: i64,
    filler_tokens: usize,
    owners: BTreeMap<String, Value>,
    mixed: BTreeMap<String, Value>,
    carry_seed: Value,
    carry_hits: Vec<Value>,
    cancel_seed: Value,
    cancel_after: Value,
    kill_seed: Value,
    kill_after: Value,
    twin: Value,
    burst: BTreeMap<String, Value>,
    move_side: BTreeMap<String, Value>,
    move_big: Option<Value>,
}

impl Plan {
    fn from_meta(meta: &Value) -> Self {
        let s = meta["salt"].as_str().unwrap_or("").to_string();
        let seed = meta["seed"].as_i64().unwrap_or(1234);
        let owner_tokens = meta["owner_tokens"].as_u64().unwrap_or(6000) as usize;
        let carry_tokens = meta["carry_tokens"].as_u64().unwrap_or(64_000) as usize;
        let carry_rounds = meta["carry_rounds"].as_u64().unwrap_or(10) as usize;
        let twin_tokens = meta["twin_tokens"].as_u64().unwrap_or(96_000) as usize;
        let filler_tokens = meta["filler_tokens"].as_u64().unwrap_or(64_000) as usize;
        let burst_n = meta["burst"].as_u64().unwrap_or(0) as usize;
        let burst_tokens = meta["burst_tokens"].as_u64().unwrap_or(0) as usize;
        let move_side_tokens = meta["move_side_tokens"].as_u64().unwrap_or(0) as usize;
        let move_tokens = meta["move_tokens"].as_u64().unwrap_or(0) as usize;

        let mut owners = BTreeMap::new();
        for i in [1, 2] {
            owners.insert(
                format!("O{i}"),
                msg(
                    &kv_doc(&s, &format!("O{i}"), owner_tokens, seed),
                    "Write a long, detailed story inspired by the text above. Do not stop early.",
                ),
            );
        }
        let mut mixed = BTreeMap::new();
        for (i, &t) in MIXED_SIZES.iter().enumerate() {
            mixed.insert(
                format!("M{}", i + 1),
                msg(
                    &kv_doc(&s, &format!("M{}", i + 1), t, seed),
                    KV_Q[i % KV_Q.len()],
                ),
            );
        }
        let ca = kv_doc(&s, "CA", carry_tokens, seed);
        let carry_seed = msg(&ca, KV_Q[0]);
        let carry_hits = (0..carry_rounds)
            .map(|i| msg(&ca, KV_Q[(i + 1) % KV_Q.len()]))
            .collect();
        let cb = kv_doc(&s, "CB", carry_tokens, seed);
        let cancel_seed = msg(&cb, KV_Q[0]);
        let cancel_after = msg(&cb, Q_AFTER);
        let ck = kv_doc(&s, "CK", carry_tokens, seed);
        let kill_seed = msg(&ck, KV_Q[0]);
        let kill_after = msg(&ck, Q_AFTER);
        let twin = msg(&kv_doc(&s, "TW", twin_tokens, seed), KV_Q[2]);

        let mut burst = BTreeMap::new();
        if burst_n > 0 {
            let bu = kv_doc(&s, "BU", burst_tokens, seed);
            for i in 0..burst_n {
                burst.insert(format!("BU{}", i + 1), msg(&bu, KV_Q[i % KV_Q.len()]));
            }
        }
        let mut move_side = BTreeMap::new();
        let move_big = if move_tokens > 0 {
            for x in ["A", "B"] {
                move_side.insert(
                    format!("MV_{x}"),
                    msg(
                        &kv_doc(&s, &format!("MV_{x}"), move_side_tokens, seed),
                        "Write a long, detailed story inspired by the text above. Do not stop early.",
                    ),
                );
            }
            Some(msg(&kv_doc(&s, "MV_C", move_tokens, seed), KV_Q[3]))
        } else {
            None
        };
        Self {
            salt: s,
            seed,
            filler_tokens,
            owners,
            mixed,
            carry_seed,
            carry_hits,
            cancel_seed,
            cancel_after,
            kill_seed,
            kill_after,
            twin,
            burst,
            move_side,
            move_big,
        }
    }

    fn filler(&self, tag: &str) -> Value {
        msg(
            &kv_doc(
                &format!("{}-f", self.salt),
                tag,
                self.filler_tokens,
                self.seed,
            ),
            KV_Q[0],
        )
    }
}

async fn run_record(a: KvRecordArgs) -> i32 {
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
    let mut meta = json!({
        "tool": "kv_shard_check_e2e", "schema": 2, "salt": salt, "seed": a.seed, "model": model,
        "max_tokens": a.max_tokens, "decode_tokens": a.decode_tokens, "owner_tokens": a.owner_tokens,
        "carry_tokens": a.carry_tokens, "carry_rounds": a.carry_rounds, "twin_tokens": a.twin_tokens,
        "filler_tokens": a.filler_tokens, "burst": a.burst, "burst_tokens": a.burst_tokens,
        "move_side_tokens": a.move_side_tokens, "move_tokens": a.move_tokens, "label": a.label,
        "time": now_iso8601(), "health": srv.health().await
    });
    let plan = Plan::from_meta(&meta);
    let mut rows = Map::new();
    let mut jobs: Vec<(String, Value, u64)> = Vec::new();
    for (n, m) in &plan.owners {
        jobs.push((n.clone(), m.clone(), a.decode_tokens));
    }
    for (n, m) in &plan.mixed {
        jobs.push((n.clone(), m.clone(), a.max_tokens));
    }
    for (n, m) in &plan.burst {
        jobs.push((n.clone(), m.clone(), a.max_tokens));
    }
    jobs.push(("CA".to_string(), plan.carry_seed.clone(), a.max_tokens));
    for (i, m) in plan.carry_hits.iter().enumerate() {
        jobs.push((format!("CA{}", i + 1), m.clone(), a.max_tokens));
    }
    jobs.push(("CB".to_string(), plan.cancel_seed.clone(), a.max_tokens));
    jobs.push((
        "CB_after".to_string(),
        plan.cancel_after.clone(),
        a.max_tokens,
    ));
    jobs.push(("CK".to_string(), plan.kill_seed.clone(), a.max_tokens));
    jobs.push(("TW".to_string(), plan.twin.clone(), a.max_tokens));
    for (n, m) in &plan.move_side {
        jobs.push((n.clone(), m.clone(), a.decode_tokens));
    }
    if let Some(mb) = &plan.move_big {
        jobs.push(("MV_C".to_string(), mb.clone(), a.max_tokens));
    }
    let mut errors = Vec::new();
    let mut weak = Vec::new();
    for (name, msgs, mt) in jobs {
        let r = srv.chat(&msgs, mt).await;
        if r.get("error").is_some() {
            errors.push(name.clone());
        } else if r["token_sha_source"].as_str() != Some("server") {
            weak.push(name.clone());
        }
        rows.insert(name, r);
    }
    meta["weak"] = json!(weak);
    let doc = json!({"meta": meta, "rows": Value::Object(rows)});
    if let Some(p) = a.out.parent() {
        let _ = fs::create_dir_all(p);
    }
    let _ = fs::write(
        &a.out,
        serde_json::to_string_pretty(&doc).unwrap_or_default(),
    );
    eprintln!("wrote {}: errors {errors:?}", a.out.display());
    if errors.is_empty() {
        0
    } else {
        1
    }
}

#[derive(Default, Clone)]
struct SampleData {
    samples: usize,
    both: usize,
    parts: usize,
    max_live: Vec<i64>,
    max_hold: i64,
    max_twin: i64,
}

impl SampleData {
    fn summary(&self) -> Value {
        json!({
            "samples": self.samples, "parts": self.parts, "max_live": self.max_live,
            "both_live": self.both, "max_hold_rows": self.max_hold, "max_twin_rows": self.max_twin
        })
    }
}

struct SamplerHandle {
    stop: Arc<AtomicBool>,
    data: Arc<Mutex<SampleData>>,
    task: tokio::task::JoinHandle<()>,
}

impl SamplerHandle {
    fn start(srv: Server, every: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let data = Arc::new(Mutex::new(SampleData::default()));
        let st = stop.clone();
        let dt = data.clone();
        let task = tokio::spawn(async move {
            while !st.load(Ordering::Relaxed) {
                let c = srv.counters().await;
                let ps = kv_parts(&c);
                if !ps.is_empty() {
                    let mut d = dt.lock().await;
                    d.samples += 1;
                    d.parts = d.parts.max(ps.len());
                    if d.max_live.len() < ps.len() {
                        d.max_live.resize(ps.len(), 0);
                    }
                    let lives: Vec<i64> = ps
                        .iter()
                        .map(|p| p.get("live").copied().unwrap_or(0.0) as i64)
                        .collect();
                    for (i, &l) in lives.iter().enumerate() {
                        d.max_live[i] = d.max_live[i].max(l);
                    }
                    if lives.len() >= 2 && lives[0] > 0 && lives[1] > 0 {
                        d.both += 1;
                    }
                    for p in &ps {
                        d.max_hold = d
                            .max_hold
                            .max(p.get("hold_rows").copied().unwrap_or(0.0) as i64);
                        d.max_twin = d
                            .max_twin
                            .max(p.get("twin_rows").copied().unwrap_or(0.0) as i64);
                    }
                }
                tokio::time::sleep(every).await;
            }
        });
        Self { stop, data, task }
    }

    async fn finish(self) -> SampleData {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.task.await;
        self.data.lock().await.clone()
    }
}

struct KvRunner {
    a: KvRunArgs,
    base: Value,
    p: Plan,
    srv: Server,
    rc: Receipt,
    mt: u64,
    dt: u64,
    window: Option<i64>,
    rounds_to_carry: Option<usize>,
    c_start: Counters,
}

impl KvRunner {
    fn same(&self, name: &str, row: &Value) -> (bool, String) {
        same_reply(self.base["rows"].get(name), row, self.a.allow_weak)
    }

    fn rounds(&self, c0: &Counters, c1: &Counters) -> Option<f64> {
        let d = deltas(c0, c1, ["kv_shard.fetch_rounds", "kv_shard.owner_rounds"]);
        let f = d["kv_shard.fetch_rounds"]?;
        let o = d["kv_shard.owner_rounds"]?;
        Some(f + o)
    }

    async fn size_prompt<F>(
        &self,
        content: F,
        target: i64,
        lo: i64,
        hi: i64,
    ) -> (usize, Option<i64>)
    where
        F: Fn(usize) -> String,
    {
        let (Some(ca), _, _) = self
            .srv
            .tokenize(&json!([{"role": "user", "content": content(256)}]))
            .await
        else {
            return (0, None);
        };
        let (Some(cb), _, _) = self
            .srv
            .tokenize(&json!([{"role": "user", "content": content(512)}]))
            .await
        else {
            return (0, None);
        };
        let per = ((cb - ca) as f64 / 256.0).max(0.001);
        let mid = (lo + hi) as f64 / 2.0;
        let mut n = (256.0 + (mid - ca as f64) / per).max(1.0) as usize;
        let mut got = None;
        for _ in 0..8 {
            let (Some(g), _, _) = self
                .srv
                .tokenize(&json!([{"role": "user", "content": content(n)}]))
                .await
            else {
                break;
            };
            got = Some(g);
            if g >= lo && g <= hi {
                break;
            }
            let adj = ((mid - g as f64) / per) as isize;
            let step = if adj != 0 {
                adj
            } else if g > hi {
                -1
            } else {
                1
            };
            n = ((n as isize) + step).max(1) as usize;
        }
        let _ = target;
        (n, got)
    }

    async fn at_once(
        &self,
        named: &BTreeMap<String, Value>,
        mt: u64,
    ) -> (BTreeMap<String, Value>, SampleData) {
        let smp = SamplerHandle::start(
            self.srv.clone(),
            Duration::from_secs_f64(self.a.sample_ms / 1000.0),
        );
        let mut handles = Vec::with_capacity(named.len());
        for (n, m) in named {
            let s = self.srv.clone();
            let nm = n.clone();
            let mm = m.clone();
            handles.push(tokio::spawn(async move { (nm, s.chat(&mm, mt).await) }));
        }
        let mut rows = BTreeMap::new();
        for h in handles {
            let (n, r) = h.await.unwrap();
            rows.insert(n, r);
        }
        let sd = smp.finish().await;
        (rows, sd)
    }

    async fn limits(&mut self) {
        let models = self
            .srv
            .get_json("/v1/models", Duration::from_secs(15))
            .await;
        let c = self.srv.counters().await;
        let h = self.srv.health().await;
        let mml = models["data"][0]["max_model_len"].as_i64();
        let (_, tok_limit, _) = self
            .srv
            .tokenize(&json!([{"role": "user", "content": "hi"}]))
            .await;
        let line = self.a.startup_log.as_deref().and_then(parse_startup_line);
        let (lp, lmode, lpart, lw) = match &line {
            Some((p, m, pt, w)) => (Some(*p), Some(m.clone()), Some(*pt), Some(*w)),
            None => (None, None, None, None),
        };
        let hpool = kv_pool(&c);
        let h_cw = h["context_window"].as_i64();
        let rep_w = (self.a.reported_window > 0).then_some(self.a.reported_window);
        let rep_p = (self.a.reported_pool > 0).then_some(self.a.reported_pool);
        let window = tok_limit.or(mml).or(h_cw).or(rep_w).or(lw);
        let pool = hpool.or(rep_p).or(lp);
        self.window = window;

        let mut why = Vec::new();
        if window.is_none() {
            why.push("no single-conversation limit reported".to_string());
        }
        let lims: BTreeSet<i64> = [tok_limit, mml, h_cw, lw, rep_w]
            .into_iter()
            .flatten()
            .collect();
        if lims.len() > 1 {
            why.push(format!("window differs between sources: {lims:?}"));
        }
        if self.a.expect_window > 0 && window.unwrap_or(0) < self.a.expect_window {
            why.push(format!(
                "window {:?} < --expect-window {}",
                window, self.a.expect_window
            ));
        }
        self.rc.check(
            "limits: window",
            why.is_empty(),
            &if why.is_empty() {
                format!("one conversation up to {:?} tokens", window)
            } else {
                why.join("; ")
            },
            Map::new(),
        );

        if let Some(p) = pool {
            let mut why_p = Vec::new();
            let pools: BTreeSet<i64> = [hpool, rep_p, lp].into_iter().flatten().collect();
            if pools.len() > 1 {
                why_p.push(format!("pool differs between sources: {pools:?}"));
            }
            if let (Some(lp_part), Some(&h_part)) = (lpart, c.get("kv_shard.part_rows")) {
                if lp_part != h_part as i64 {
                    why_p.push(format!("part {h_part} != {lp_part}"));
                }
            }
            if let Some(w) = window {
                if w > p {
                    why_p.push(format!("window {w} > pool {p}"));
                }
            }
            if self.a.expect_pool > 0 && p != self.a.expect_pool {
                why_p.push(format!("pool {p} != --expect-pool {}", self.a.expect_pool));
            }
            self.rc.check(
                "limits: pool",
                why_p.is_empty(),
                &if why_p.is_empty() {
                    format!("pool {p} ({:?})", lmode)
                } else {
                    why_p.join("; ")
                },
                Map::new(),
            );
        } else {
            self.rc
                .add("limits: pool", "SKIP", "no total pool reported", Map::new());
        }

        if self.a.over_window > 0 {
            if let Some(w) = window {
                let target = w + self.a.over_window;
                let (n, got) = self
                    .size_prompt(
                        |k| FILLER.repeat(k),
                        target,
                        target,
                        target + 2 * self.a.over_window,
                    )
                    .await;
                match got {
                    None => self
                        .rc
                        .add("limits: over-window", "SKIP", "no /tokenize", Map::new()),
                    Some(g) if g <= w => self.rc.add(
                        "limits: over-window",
                        "SKIP",
                        "could not size past window",
                        Map::new(),
                    ),
                    Some(g) => {
                        let body = self.srv.body(
                            &json!([{"role": "user", "content": FILLER.repeat(n)}]),
                            16,
                            false,
                        );
                        let r = self
                            .srv
                            .post_json(
                                "/v1/chat/completions",
                                &body,
                                Some(Duration::from_secs(600)),
                            )
                            .await;
                        let err = r["error"].as_str().unwrap_or("");
                        let refused = r["http_status"].as_u64() == Some(400)
                            && err.to_lowercase().contains("context");
                        self.rc.check(
                            "limits: over-window",
                            refused,
                            &format!(
                                "{g}-token prompt (window {w}): status={} {err}",
                                r["http_status"]
                            ),
                            Map::new(),
                        );
                    }
                }
            }
        }
    }

    async fn owners(&mut self) {
        let c0 = self.srv.counters().await;
        let (rows, smp) = self.at_once(&self.p.owners, self.dt).await;
        let c1 = self.srv.counters().await;
        let bad: Vec<String> = rows
            .iter()
            .filter_map(|(n, r)| {
                let (ok, d) = self.same(n, r);
                (!ok).then(|| format!("{n}: {d}"))
            })
            .collect();
        self.rc.check(
            "owners: replies == baseline",
            bad.is_empty(),
            &if bad.is_empty() {
                "both equal".into()
            } else {
                bad.join("; ")
            },
            Map::new(),
        );
        let r = self.rounds(&c0, &c1);
        if smp.parts == 0 || r.is_none() {
            self.rc.add(
                "owners: both parts decoded",
                "SKIP",
                "no kv_shard per_part/rounds",
                Map::new(),
            );
            return;
        }
        let ok = smp.max_live.first().copied().unwrap_or(0) >= 1
            && smp.max_live.get(1).copied().unwrap_or(0) >= 1
            && r.unwrap_or(0.0) > 0.0;
        let mut ex = Map::new();
        ex.insert("sampler".to_string(), smp.summary());
        self.rc.check(
            "owners: both parts decoded",
            ok,
            &format!("max_live={:?} rounds=+{:?}", smp.max_live, r),
            ex,
        );
    }

    async fn mixed(&mut self) {
        let c0 = self.srv.counters().await;
        let (rows, smp) = self.at_once(&self.p.mixed, self.mt).await;
        let c1 = self.srv.counters().await;
        let bad: Vec<String> = rows
            .iter()
            .filter_map(|(n, r)| {
                let (ok, d) = self.same(n, r);
                (!ok).then(|| format!("{n}: {d}"))
            })
            .collect();
        self.rc.check(
            "mixed: 8 at once == baseline",
            bad.is_empty(),
            &if bad.is_empty() {
                "8 of 8 equal".into()
            } else {
                bad.join("; ")
            },
            Map::new(),
        );
        let r = self.rounds(&c0, &c1);
        if smp.parts == 0 || r.is_none() {
            self.rc.add(
                "mixed: rounds spanned both owners",
                "SKIP",
                "no kv_shard per_part/rounds",
                Map::new(),
            );
            return;
        }
        self.rc.check(
            "mixed: rounds spanned both owners",
            smp.both > 0 && r.unwrap_or(0.0) > 0.0,
            &format!(
                "{} of {} samples both live, rounds +{:?}",
                smp.both, smp.samples, r
            ),
            Map::new(),
        );
    }

    async fn burst(&mut self) {
        if self.p.burst.is_empty() {
            self.rc
                .add("burst", "SKIP", "baseline has no burst rows", Map::new());
            return;
        }
        let c0 = self.srv.counters().await;
        let (rows, _smp) = self.at_once(&self.p.burst, self.mt).await;
        let c1 = self.srv.counters().await;
        let bad: Vec<String> = rows
            .iter()
            .filter_map(|(n, r)| {
                let (ok, d) = self.same(n, r);
                (!ok).then(|| format!("{n}: {d}"))
            })
            .collect();
        self.rc.check(
            &format!("burst: {} on one prompt == baseline", rows.len()),
            bad.is_empty(),
            &if bad.is_empty() {
                format!("{} equal", rows.len())
            } else {
                bad.join("; ")
            },
            Map::new(),
        );
        let d = deltas(
            &c0,
            &c1,
            [
                "kv_shard.transfers_failed",
                "kv_shard.copies",
                "kv_shard.carries",
            ],
        );
        match d["kv_shard.transfers_failed"] {
            None => self.rc.add(
                "burst: no transfer failed",
                "SKIP",
                "transfers_failed not reported",
                Map::new(),
            ),
            Some(tf) => {
                self.rc.check(
                    "burst: no transfer failed",
                    tf == 0.0,
                    &format!("{d:?}"),
                    Map::new(),
                );
            }
        }
    }

    async fn carry(&mut self) {
        let r = self.srv.chat(&self.p.carry_seed, self.mt).await;
        let (ok, d) = self.same("CA", &r);
        let mut bad = if ok {
            Vec::new()
        } else {
            vec![format!("CA: {d}")]
        };
        let mut carried_at = None;
        for (i, msgs) in self.p.carry_hits.clone().into_iter().enumerate() {
            let _ = self
                .srv
                .chat(&self.p.filler(&format!("carry-{i}-{}", random_hex(3))), 8)
                .await;
            let cb = self.srv.counters().await;
            let h = self.srv.chat(&msgs, self.mt).await;
            let ca = self.srv.counters().await;
            let (ok_h, dh) = self.same(&format!("CA{}", i + 1), &h);
            if !ok_h {
                bad.push(format!("CA{}: {dh}", i + 1));
            }
            if delta(&cb, &ca, "kv_shard.carries").unwrap_or(0.0) > 0.0 {
                carried_at = Some(i + 1);
                break;
            }
        }
        self.rounds_to_carry = carried_at;
        if !bad.is_empty() {
            self.rc.add("carry", "FAIL", &bad.join("; "), Map::new());
        } else if let Some(at) = carried_at {
            self.rc.add(
                "carry",
                "PASS",
                &format!("carried at round {at}"),
                Map::new(),
            );
        } else {
            self.rc
                .add("carry", "SKIP", "no carry provoked", Map::new());
        }
    }

    async fn twins(&mut self) {
        let c0 = self.srv.counters().await;
        let smp = SamplerHandle::start(
            self.srv.clone(),
            Duration::from_secs_f64(self.a.sample_ms / 1000.0),
        );
        let r = self.srv.chat(&self.p.twin, self.mt).await;
        let _sd = smp.finish().await;
        let c1 = self.srv.counters().await;
        let (ok, d) = self.same("TW", &r);
        self.rc
            .check("twins: reply == baseline", ok, &d, Map::new());
        let dd = deltas(
            &c0,
            &c1,
            [
                "kv_shard.twins_made",
                "kv_shard.prompt_fetches",
                "kv_shard.twins_revoked",
            ],
        );
        let Some(tm) = dd["kv_shard.twins_made"] else {
            self.rc.add(
                "twins: borrowed",
                "SKIP",
                "twins_made not reported",
                Map::new(),
            );
            return;
        };
        self.rc
            .check("twins: borrowed", tm >= 1.0, &format!("{dd:?}"), Map::new());
        if tm >= 1.0 {
            if let Some(pf) = dd["kv_shard.prompt_fetches"] {
                if dd["kv_shard.twins_revoked"].unwrap_or(0.0) > 0.0 {
                    self.rc.add(
                        "twins: the twinned pass fetched nothing",
                        "PASS",
                        "revoked then fetched",
                        Map::new(),
                    );
                } else {
                    self.rc.check(
                        "twins: the twinned pass fetched nothing",
                        pf == 0.0,
                        &format!("prompt_fetches +{pf}"),
                        Map::new(),
                    );
                }
            }
        }
        let left: Vec<f64> = kv_parts(&c1)
            .iter()
            .map(|p| p.get("twin_rows").copied().unwrap_or(0.0))
            .collect();
        if !left.is_empty() {
            self.rc.check(
                "twins: released",
                left.iter().all(|&x| x == 0.0),
                &format!("twin_rows {left:?}"),
                Map::new(),
            );
        }
    }

    async fn move_chk(&mut self) {
        let Some(move_big) = self.p.move_big.clone() else {
            self.rc
                .add("move", "SKIP", "no move rows in baseline", Map::new());
            return;
        };
        let c0 = self.srv.counters().await;
        let smp = SamplerHandle::start(
            self.srv.clone(),
            Duration::from_secs_f64(self.a.sample_ms / 1000.0),
        );
        let mut handles = Vec::new();
        for (n, msgs) in self.p.move_side.clone() {
            let first = Arc::new(Notify::new());
            let fn_ = first.clone();
            let s = self.srv.clone();
            let dt = self.dt;
            let nm = n.clone();
            let h = tokio::spawn(async move {
                (
                    nm,
                    s.stream(&msgs, dt, None, false, None, Some(fn_), false)
                        .await,
                )
            });
            let _ =
                tokio::time::timeout(Duration::from_secs_f64(self.a.move_wait), first.notified())
                    .await;
            handles.push(h);
        }
        let mut rows = BTreeMap::new();
        rows.insert("MV_C".to_string(), self.srv.chat(&move_big, self.mt).await);
        for h in handles {
            let (n, r) = h.await.unwrap();
            rows.insert(n, r);
        }
        let _sd = smp.finish().await;
        let c1 = self.srv.counters().await;
        let bad: Vec<String> = ["MV_A", "MV_B", "MV_C"]
            .iter()
            .filter_map(|&n| {
                let (ok, d) = self.same(n, rows.get(n).unwrap_or(&json!({"error": "no reply"})));
                (!ok).then(|| format!("{n}: {d}"))
            })
            .collect();
        self.rc.check(
            "move: replies == baseline",
            bad.is_empty(),
            &if bad.is_empty() {
                "3 of 3 equal".into()
            } else {
                bad.join("; ")
            },
            Map::new(),
        );
        match delta(&c0, &c1, "kv_shard.moves") {
            None => self.rc.add(
                "move: a live stream moved",
                "SKIP",
                "kv_shard.moves not reported",
                Map::new(),
            ),
            Some(m) if m <= 0.0 => {
                self.rc
                    .add("move: a live stream moved", "SKIP", "moves +0", Map::new())
            }
            Some(m) => self.rc.add(
                "move: a live stream moved",
                "PASS",
                &format!("moves +{m}"),
                Map::new(),
            ),
        }
    }

    async fn provoke(&self, seed_msgs: &Value, seed_name: &str, tag: &str) -> (bool, String) {
        let r = self.srv.chat(seed_msgs, self.mt).await;
        let res = self.same(seed_name, &r);
        let rounds = self.rounds_to_carry.unwrap_or(0);
        for i in 0..rounds {
            let _ = self
                .srv
                .chat(&self.p.filler(&format!("{tag}-{i}-{}", random_hex(3))), 8)
                .await;
        }
        res
    }

    async fn cancel_transfer(&mut self) {
        if self.rounds_to_carry.is_none() {
            self.rc
                .add("cancel-transfer", "SKIP", "no carry provoked", Map::new());
            return;
        }
        let (ok0, d0) = self
            .provoke(&self.p.cancel_seed.clone(), "CB", "cancel")
            .await;
        let c0 = self.srv.counters().await;
        let cancel = Arc::new(Notify::new());
        let cn = cancel.clone();
        let s = self.srv.clone();
        let m = self.p.cancel_after.clone();
        let mt = self.mt;
        let h =
            tokio::spawn(async move { s.stream(&m, mt, None, false, Some(cn), None, false).await });

        let deadline = Instant::now() + Duration::from_secs_f64(self.a.cancel_wait);
        let mut seen = false;
        while Instant::now() < deadline && !h.is_finished() {
            let c = self.srv.counters().await;
            let holds = kv_parts(&c)
                .iter()
                .any(|p| p.get("hold_rows").copied().unwrap_or(0.0) > 0.0);
            if delta(&c0, &c, "kv_shard.carries").unwrap_or(0.0) > 0.0 || holds {
                seen = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        if !seen {
            tokio::time::sleep(Duration::from_secs_f64(self.a.cancel_after)).await;
        }
        let in_flight = !h.is_finished();
        cancel.notify_waiters();
        let cancelled_row = h.await.unwrap_or(Value::Null);

        let s_end = Instant::now() + Duration::from_secs_f64(self.a.settle_wait);
        let mut settled = false;
        while Instant::now() < s_end {
            let ps = kv_parts(&self.srv.counters().await);
            if !ps.is_empty()
                && ps.iter().all(|p| {
                    p.get("hold_rows").copied().unwrap_or(0.0) == 0.0
                        && p.get("live").copied().unwrap_or(0.0) == 0.0
                })
            {
                settled = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        let alive = self.srv.alive().await;
        let c1 = self.srv.counters().await;
        let again = self.srv.chat(&self.p.cancel_after, self.mt).await;
        let (ok, d) = self.same("CB_after", &again);
        let dc = delta(&c0, &c1, "kv_shard.carries").unwrap_or(0.0);
        if !(alive && settled && ok && ok0) {
            self.rc.add(
                "cancel-transfer",
                "FAIL",
                &format!("alive={alive} settled={settled} seed={d0} again={d}"),
                Map::new(),
            );
        } else if !in_flight || !cancelled_row["cancelled"].as_bool().unwrap_or(false) {
            self.rc.add(
                "cancel-transfer",
                "SKIP",
                "request finished before cancel",
                Map::new(),
            );
        } else if dc <= 0.0 {
            self.rc
                .add("cancel-transfer", "SKIP", "no carry began", Map::new());
        } else {
            self.rc.add(
                "cancel-transfer",
                "PASS",
                &format!("carry +{dc} cancelled cleanly, {d}"),
                Map::new(),
            );
        }
    }

    async fn kill_rank1(&mut self) {
        let Some(kcmd) = self.a.kill_cmd.clone() else {
            self.rc
                .add("kill-rank1", "SKIP", "no --kill-cmd", Map::new());
            return;
        };
        if self.rounds_to_carry.is_none() {
            self.rc
                .add("kill-rank1", "SKIP", "no carry provoked", Map::new());
            return;
        }
        let _ = self.provoke(&self.p.kill_seed.clone(), "CK", "kill").await;
        let c0 = self.srv.counters().await;
        let s = self.srv.clone();
        let m = self.p.kill_after.clone();
        let mt = self.mt;
        let h = tokio::spawn(async move { s.stream(&m, mt, None, false, None, None, false).await });

        let deadline = Instant::now() + Duration::from_secs_f64(self.a.cancel_wait);
        while Instant::now() < deadline && !h.is_finished() {
            let c = self.srv.counters().await;
            let holds = kv_parts(&c)
                .iter()
                .any(|p| p.get("hold_rows").copied().unwrap_or(0.0) > 0.0);
            if delta(&c0, &c, "kv_shard.carries").unwrap_or(0.0) > 0.0 || holds {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        if h.is_finished() {
            self.rc.add(
                "kill-rank1",
                "SKIP",
                "request finished before kill",
                Map::new(),
            );
            return;
        }
        let t0 = Instant::now();
        let k_ok = Command::new("sh")
            .args(["-c", &kcmd])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        let ended = tokio::time::timeout(Duration::from_secs_f64(self.a.kill_deadline), h)
            .await
            .is_ok();
        let secs = t0.elapsed().as_secs_f64();
        if !k_ok {
            self.rc
                .add("kill-rank1", "FAIL", "--kill-cmd failed", Map::new());
        } else if !ended {
            self.rc.add(
                "kill-rank1",
                "FAIL",
                &format!("request hung past {}s", self.a.kill_deadline),
                Map::new(),
            );
        } else {
            self.rc.add(
                "kill-rank1",
                "PASS",
                &format!("ended {secs:.1}s after kill"),
                Map::new(),
            );
        }
    }

    async fn idle(&mut self) {
        let end = Instant::now() + Duration::from_secs_f64(self.a.settle_wait);
        while Instant::now() < end {
            let ps = kv_parts(&self.srv.counters().await);
            if !ps.is_empty()
                && ps
                    .iter()
                    .all(|p| p.get("live").copied().unwrap_or(0.0) == 0.0)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let c = self.srv.counters().await;
        let ps = kv_parts(&c);
        if ps.is_empty() {
            self.rc
                .add("idle", "SKIP", "no kv_shard per_part", Map::new());
            return;
        }
        let rows = c.get("kv_shard.part_rows").copied().unwrap_or(0.0) as i64;
        let mut why = Vec::new();
        if c.get("kv_shard.parts").copied().unwrap_or(0.0) != 2.0 || ps.len() != 2 {
            why.push(format!("parts != 2 ({})", ps.len()));
        }
        for (i, p) in ps.iter().enumerate() {
            for k in ["live", "live_rows", "twin_rows", "hold_rows"] {
                if p.get(k).copied().unwrap_or(0.0) != 0.0 {
                    why.push(format!("part {i} {k} != 0"));
                }
            }
            let free = p.get("free").copied().unwrap_or(0.0) as i64;
            let kept = p.get("kept_rows").copied().unwrap_or(0.0) as i64;
            if rows > 0 && free + kept > rows {
                why.push(format!("part {i}: free+kept > part_rows"));
            }
        }
        if delta(&self.c_start, &c, "kv_shard.transfers_failed").unwrap_or(0.0) > 0.0 {
            why.push("transfers_failed grew".to_string());
        }
        self.rc.check(
            "idle",
            why.is_empty(),
            &if why.is_empty() {
                format!("2 parts of {rows}, nothing held")
            } else {
                why.join("; ")
            },
            Map::new(),
        );
    }

    async fn needle(&mut self) {
        let sizes: Vec<String> = self
            .a
            .needle_sizes
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        for s in sizes {
            let name = format!("needle {s}");
            let target = if s == "max" {
                self.window.map(|w| w - self.a.needle_answer as i64 - 64)
            } else {
                s.parse::<i64>().ok()
            };
            let Some(tgt) = target else {
                self.rc.add(&name, "SKIP", "no window reported", Map::new());
                continue;
            };
            if let Some(w) = self.window {
                if tgt + self.a.needle_answer as i64 > w {
                    self.rc.add(&name, "SKIP", "over window", Map::new());
                    continue;
                }
            }
            let intro = "Read this synthetic archive. Three special entries give the secret values for ALPHA, BETA and GAMMA. Remember their exact values and answer the final question.\n";
            let q = "\nWhat are the secret values for ALPHA, BETA and GAMMA? Return only one JSON object with these three keys and their six-digit string values.\n";
            let make = |n: usize| {
                let a = n / 10;
                let b = n * 4 / 10;
                let c = n * 4 / 10;
                format!(
                    "{intro}{}\nSPECIAL ENTRY: ALPHA secret value is {}.\n{}\nSPECIAL ENTRY: BETA secret value is {}.\n{}\nSPECIAL ENTRY: GAMMA secret value is {}.\n{}{q}",
                    FILLER.repeat(a), SECRETS[0].1, FILLER.repeat(b), SECRETS[1].1, FILLER.repeat(c), SECRETS[2].1, FILLER.repeat(n.saturating_sub(a + b + c))
                )
            };
            let (n, got) = self.size_prompt(make, tgt, tgt - 256, tgt).await;
            if got.is_none() {
                self.rc.add(&name, "SKIP", "no /tokenize", Map::new());
                continue;
            }
            let row = self
                .srv
                .stream(
                    &json!([{"role": "user", "content": make(n)}]),
                    self.a.needle_answer,
                    None,
                    false,
                    None,
                    None,
                    true,
                )
                .await;
            let txt = row["text"].as_str().unwrap_or("");
            let ok = row.get("error").is_none() && SECRETS.iter().all(|&(_, v)| txt.contains(v));
            self.rc.check(
                &name,
                ok,
                &format!("prompt {} tok", row["prompt_tokens"]),
                Map::new(),
            );
        }
    }

    async fn go(mut self) -> i32 {
        if !self.srv.wait_healthy(self.a.health_timeout).await {
            self.rc.add("health", "FAIL", "not healthy", Map::new());
            return self.rc.finish(self.a.receipt.as_deref(), self.a.strict);
        }
        self.c_start = self.srv.counters().await;
        self.rc
            .doc
            .insert("counters_before".to_string(), json!(self.c_start));
        let checks: Vec<String> = match &self.a.checks {
            Some(s) => s
                .split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect(),
            None => vec![
                "limits",
                "owners",
                "mixed",
                "burst",
                "carry",
                "twins",
                "move",
                "cancel-transfer",
                "needle",
                "idle",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        };
        let has_kill = checks.iter().any(|c| c == "kill-rank1");
        for chk in checks {
            match chk.as_str() {
                "limits" => self.limits().await,
                "owners" => self.owners().await,
                "mixed" => self.mixed().await,
                "burst" => self.burst().await,
                "carry" => self.carry().await,
                "twins" => self.twins().await,
                "move" => self.move_chk().await,
                "cancel-transfer" => self.cancel_transfer().await,
                "kill-rank1" => self.kill_rank1().await,
                "needle" => self.needle().await,
                "idle" => self.idle().await,
                _ => self.rc.add(&chk, "FAIL", "unknown check", Map::new()),
            }
        }
        if !has_kill {
            let ca = self.srv.counters().await;
            self.rc.doc.insert("counters_after".to_string(), json!(ca));
        }
        self.rc.finish(self.a.receipt.as_deref(), self.a.strict)
    }
}

pub async fn run(a: KvShardArgs) -> i32 {
    match a.mode {
        KvMode::Record(r) => run_record(r).await,
        KvMode::Run(u) => {
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
            let meta = &base["meta"];
            let p = Plan::from_meta(meta);
            let mt = meta["max_tokens"].as_u64().unwrap_or(48);
            let dt = meta["decode_tokens"].as_u64().unwrap_or(300);
            let model = u
                .model
                .clone()
                .or_else(|| meta["model"].as_str().map(String::from))
                .unwrap_or_else(|| DEFAULT_MODEL.to_string());
            let url = u.url.clone().unwrap_or_else(default_url);
            let srv = Server::new(&url, &model, u.timeout, p.seed, key.clone());
            let mut rmeta = Map::new();
            rmeta.insert("url".to_string(), json!(srv.url));
            rmeta.insert(
                "baseline".to_string(),
                json!(u.baseline.display().to_string()),
            );
            rmeta.insert("label".to_string(), json!(u.label));
            rmeta.insert("salt".to_string(), json!(p.salt));
            let rc = Receipt::new("kv_shard_check_e2e", rmeta, key);
            let runner = KvRunner {
                a: u,
                base,
                p,
                srv,
                rc,
                mt,
                dt,
                window: None,
                rounds_to_carry: None,
                c_start: Counters::new(),
            };
            runner.go().await
        }
    }
}
