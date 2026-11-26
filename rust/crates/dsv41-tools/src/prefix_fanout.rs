//! `prefix-fanout`, `pair-check`, and `resume-bench` subcommands replacing
//! `tools/prefix_fanout.py`, `tools/pair_check.py`, and `tools/resume_bench.py`.

use clap::{Args, ValueEnum};
use dsv41_core::auth::resolve_strict;
use dsv41_core::client::{same_reply, Server};
use dsv41_core::metrics::{delta, deltas, health_counters, rank_deltas, Counters};
use dsv41_core::prose::{words, words_for};
use dsv41_core::receipt::Receipt;
use dsv41_core::{now_hms, now_iso8601, random_hex, DEFAULT_MODEL};
use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Notify, Semaphore};

fn default_url() -> String {
    env::var("API_URL").unwrap_or_else(|_| {
        format!(
            "http://127.0.0.1:{}",
            env::var("PORT").unwrap_or_else(|_| "8888".to_string())
        )
    })
}

// ====================================================================== prefix-fanout
const FANOUT_QUESTIONS: &[&str] = &[
    "Summarize the text above in one sentence.",
    "Which word comes last in the text above?",
    "Give a title for the text above.",
    "List three nouns from the text above.",
    "Name two words that repeat in the text above.",
    "Describe the text above in five words.",
    "Write one question a reader could ask about the text above.",
    "Pick one word from the text and define it.",
    "What is the first topic in the text above?",
    "Say one short sentence about the text above.",
    "Which word appears most often above?",
    "Reply with a number between one and nine.",
];

const FANOUT_COUNTERS: &[&str] = &[
    "prompt_cache.computed_tokens",
    "prompt_cache.cached_tokens",
    "prompt_cache.memory_hits",
    "chats_continued",
    "chats_continued_prefill_tokens",
];

#[derive(Args)]
pub struct PrefixFanoutArgs {
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value = "1,2,4,8")]
    pub levels: String,
    #[arg(long, default_value_t = 8)]
    pub fanout: usize,
    #[arg(long = "system-tokens", default_value_t = 30_000)]
    pub system_tokens: usize,
    #[arg(long = "max-tokens", default_value_t = 32)]
    pub max_tokens: u64,
    #[arg(long, default_value_t = 1234)]
    pub seed: i64,
    #[arg(long)]
    pub salt: Option<String>,
    #[arg(long, default_value_t = 2048)]
    pub grid: i64,
    #[arg(long = "dup-tolerance", default_value_t = 2048)]
    pub dup_tolerance: i64,
    #[arg(long)]
    pub out: Option<PathBuf>,
    #[arg(long)]
    pub baseline: Option<PathBuf>,
    #[arg(long)]
    pub receipt: Option<PathBuf>,
    #[arg(long = "expect-one-fill")]
    pub expect_one_fill: bool,
    #[arg(long = "allow-weak")]
    pub allow_weak: bool,
    #[arg(long = "strict-cold")]
    pub strict_cold: bool,
    #[arg(long, default_value_t = 1800.0)]
    pub timeout: f64,
    #[arg(long, default_value = "")]
    pub label: String,
    #[arg(long = "api-key-file")]
    pub api_key_file: Option<PathBuf>,
}

pub async fn run_prefix_fanout(mut a: PrefixFanoutArgs) -> i32 {
    let key = match resolve_strict(a.api_key_file.as_deref()) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let base: Option<Value> = match &a.baseline {
        Some(p) => {
            match fs::read_to_string(p).and_then(|s| serde_json::from_str(&s).map_err(Into::into)) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("cannot read baseline {}: {e}", p.display());
                    return 2;
                }
            }
        }
        None => None,
    };
    if let Some(b) = &base {
        let bm = &b["meta"];
        a.salt = bm["salt"].as_str().map(String::from);
        a.seed = bm["seed"].as_i64().unwrap_or(a.seed);
        a.system_tokens = bm["system_tokens"]
            .as_u64()
            .unwrap_or(a.system_tokens as u64) as usize;
        a.fanout = bm["fanout"].as_u64().unwrap_or(a.fanout as u64) as usize;
        a.max_tokens = bm["max_tokens"].as_u64().unwrap_or(a.max_tokens);
        if a.model.is_none() {
            a.model = bm["model"].as_str().map(String::from);
        }
    }
    let model = a
        .model
        .or_else(|| env::var("MODEL").ok())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let salt = a.salt.unwrap_or_else(|| random_hex(6));
    let levels: Vec<usize> = a
        .levels
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    let srv = Server::new(
        &a.url.unwrap_or_else(default_url),
        &model,
        a.timeout,
        a.seed,
        key.clone(),
    );

    let mut meta = Map::new();
    meta.insert("url".to_string(), json!(srv.url));
    meta.insert("salt".to_string(), json!(salt));
    meta.insert("levels".to_string(), json!(levels));
    meta.insert("fanout".to_string(), json!(a.fanout));
    meta.insert("system_tokens".to_string(), json!(a.system_tokens));
    meta.insert(
        "baseline".to_string(),
        json!(a.baseline.as_ref().map(|p| p.display().to_string())),
    );
    meta.insert("label".to_string(), json!(a.label));
    let mut rc = Receipt::new("prefix_fanout", meta, key);

    if !srv.wait_healthy(600.0).await {
        rc.add(
            "health",
            "FAIL",
            &format!("{} not healthy: {}", srv.url, srv.health().await),
            Map::new(),
        );
        return rc.finish(a.receipt.as_deref(), a.strict_cold);
    }

    let mut rows_out = Map::new();
    let mut summary = Vec::new();
    for &lvl in &levels {
        let sys_content = format!(
            "[ref {salt}-c{lvl}] You are a careful assistant. Read the reference text, then answer briefly.\n{}",
            words(words_for(a.system_tokens), a.seed + (lvl as i64) * 7907)
        );
        let msgs: Vec<Value> = (0..a.fanout)
            .map(|i| {
                json!([
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": FANOUT_QUESTIONS[i % FANOUT_QUESTIONS.len()]}
                ])
            })
            .collect();
        let (tok_count, _, _) = srv.tokenize(&json!([msgs[0][0]])).await;
        let c0 = srv.counters().await;
        let t0 = Instant::now();
        let sem = Arc::new(Semaphore::new(lvl.max(1)));
        let mut handles = Vec::with_capacity(msgs.len());
        for m in msgs {
            let s = srv.clone();
            let sp = sem.clone();
            let mt = a.max_tokens;
            handles.push(tokio::spawn(async move {
                let _p = sp.acquire().await.unwrap();
                s.chat(&m, mt).await
            }));
        }
        let mut rows = Vec::with_capacity(handles.len());
        for h in handles {
            rows.push(h.await.unwrap());
        }
        let wall = t0.elapsed().as_secs_f64();
        let c1 = srv.counters().await;
        let d = deltas(&c0, &c1, FANOUT_COUNTERS.iter().copied());
        let name = format!("c{lvl}");
        rows_out.insert(name.clone(), Value::Array(rows.clone()));

        let errors: Vec<String> = rows
            .iter()
            .filter_map(|r| r.get("error").map(|e| e.to_string()))
            .collect();
        let prompts: Vec<i64> = rows
            .iter()
            .map(|r| r["prompt_tokens"].as_i64().unwrap_or(0))
            .collect();
        let cached: Vec<Option<i64>> = rows.iter().map(|r| r["cached_tokens"].as_i64()).collect();
        let count = tok_count.unwrap_or_else(|| prompts.iter().copied().min().unwrap_or(0) - 48);
        let cut = (((count - 16) / a.grid) * a.grid).max(0);
        let computed: i64 = if cached.iter().any(|c| c.is_none()) {
            if let Some(Some(ct)) = d.get("prompt_cache.computed_tokens") {
                *ct as i64
            } else {
                prompts
                    .iter()
                    .zip(&cached)
                    .map(|(p, c)| p - c.unwrap_or(0))
                    .sum()
            }
        } else {
            prompts
                .iter()
                .zip(&cached)
                .map(|(p, c)| p - c.unwrap_or(0))
                .sum()
        };
        let ideal: i64 =
            prompts.iter().sum::<i64>() - ((rows.len().saturating_sub(1)) as i64) * cut;
        let dup = computed - ideal;
        let warm = !cached.is_empty() && cached.iter().all(|c| c.unwrap_or(0) > 0);
        let info = json!({
            "level": lvl,
            "requests": rows.len(),
            "wall_seconds": (wall * 1000.0).round() / 1000.0,
            "system_tokens": count,
            "cut": cut,
            "computed_tokens": computed,
            "ideal_tokens": ideal,
            "duplicate_prefill_tokens": dup,
            "duplicate_per_follower": ((dup as f64 / (rows.len().saturating_sub(1).max(1) as f64)) * 10.0).round() / 10.0,
            "cached_tokens": cached,
            "seconds": rows.iter().map(|r| r.get("seconds").cloned().unwrap_or(Value::Null)).collect::<Vec<_>>(),
            "deltas": d,
            "warm": warm,
        });
        summary.push(info.clone());
        eprintln!(
            "  c{lvl}: {} requests in {wall:.2}s, duplicate prefill {dup} tokens (cut {cut})",
            rows.len()
        );

        let mut why = Vec::new();
        if !errors.is_empty() {
            why.push(format!("{} errors: {}", errors.len(), errors[0]));
        }
        if let Some(b) = &base {
            let brows = b["rows"][&name].as_array();
            let mut bad_idx = Vec::new();
            for (i, r) in rows.iter().enumerate() {
                let br = brows.and_then(|arr| arr.get(i));
                if !same_reply(br, r, a.allow_weak).0 {
                    bad_idx.push(i);
                }
            }
            if !bad_idx.is_empty() {
                let first = bad_idx[0];
                let detail = same_reply(
                    brows.and_then(|arr| arr.get(first)),
                    &rows[first],
                    a.allow_weak,
                )
                .1;
                why.push(format!(
                    "replies {bad_idx:?} differ from the baseline: {detail}"
                ));
            }
        }
        let tol = a.dup_tolerance * (rows.len().saturating_sub(1).max(1) as i64);
        let judged = (lvl == 1 || a.expect_one_fill) && !warm;
        let extra = info.as_object().cloned().unwrap_or_default();
        if warm && why.is_empty() {
            rc.add(
                &format!("fan-out c{lvl}"),
                "SKIP",
                &format!(
                    "not cold: every request found the prefix cached ({:?} tokens)",
                    cached[0]
                ),
                extra,
            );
            continue;
        }
        if judged && dup > tol {
            why.push(format!(
                "the shared prefix was computed again: {dup} duplicate tokens > {tol}"
            ));
        }
        rc.check(
            &format!("fan-out c{lvl}"),
            why.is_empty(),
            &if why.is_empty() {
                format!(
                    "duplicate prefill {dup} tokens ({}), {wall:.2}s",
                    if judged { "judged" } else { "measured" }
                )
            } else {
                why.join("; ")
            },
            extra,
        );
    }

    rc.doc
        .insert("summary".to_string(), Value::Array(summary.clone()));
    if let Some(out) = a.out {
        let doc = json!({
            "meta": {
                "tool": "prefix_fanout", "schema": 1, "salt": salt, "seed": a.seed,
                "model": model, "system_tokens": a.system_tokens, "fanout": a.fanout,
                "max_tokens": a.max_tokens, "levels": levels, "label": a.label,
                "time": now_iso8601()
            },
            "rows": Value::Object(rows_out),
            "summary": summary
        });
        let _ = fs::write(&out, serde_json::to_string_pretty(&doc).unwrap_or_default());
        eprintln!("wrote {}", out.display());
    }
    rc.finish(a.receipt.as_deref(), a.strict_cold)
}

// ====================================================================== pair-check
#[derive(Args)]
pub struct PairCheckArgs {
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value = "")]
    pub label: String,
    #[arg(long, default_value_t = 8)]
    pub streams: usize,
    #[arg(long = "max-tokens", default_value_t = 8000)]
    pub max_tokens: u64,
    #[arg(long = "decode-s", default_value_t = 10.0)]
    pub decode_s: f64,
    #[arg(long = "kill-cmd")]
    pub kill_cmd: String,
    #[arg(long = "end-deadline", default_value_t = 60.0)]
    pub end_deadline: f64,
    #[arg(long = "exit-cmd")]
    pub exit_cmd: Option<String>,
    #[arg(long = "exit-code", default_value_t = 75)]
    pub exit_code: i32,
    #[arg(long = "expect-log")]
    pub expect_log: Vec<String>,
    #[arg(long)]
    pub receipt: Option<PathBuf>,
    #[arg(long = "api-key-file")]
    pub api_key_file: Option<PathBuf>,
}

fn run_shell_cmd(cmd: &str) -> (i32, String) {
    match Command::new("sh").args(["-c", cmd]).output() {
        Ok(o) => {
            let mut txt = String::from_utf8_lossy(&o.stdout).into_owned();
            txt.push_str(&String::from_utf8_lossy(&o.stderr));
            (o.status.code().unwrap_or(-1), txt)
        }
        Err(e) => (-1, e.to_string()),
    }
}

fn poll_container_state(cmd: &str) -> (Option<String>, Option<i32>, String) {
    let (rc, out) = run_shell_cmd(cmd);
    let ws: Vec<&str> = out.split_whitespace().collect();
    if rc == 0 && ws.len() == 2 {
        if let Ok(code) = ws[1].parse::<i32>() {
            return (Some(ws[0].to_string()), Some(code), out);
        }
    }
    (
        None,
        None,
        out.trim()
            .chars()
            .rev()
            .take(200)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect(),
    )
}

pub async fn run_pair_check(a: PairCheckArgs) -> i32 {
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
        1800.0,
        1234,
        key.clone(),
    );
    let mut meta = Map::new();
    meta.insert("url".to_string(), json!(srv.url));
    meta.insert("label".to_string(), json!(a.label));
    meta.insert("streams".to_string(), json!(a.streams));
    let mut rc = Receipt::new("pair_check", meta, key);

    if !srv.wait_healthy(900.0).await {
        rc.add(
            "health",
            "FAIL",
            &format!("{} not healthy", srv.url),
            Map::new(),
        );
        return rc.finish(a.receipt.as_deref(), false);
    }

    let salt = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let rows_shared = Arc::new(tokio::sync::Mutex::new(vec![Value::Null; a.streams]));
    let ended_shared = Arc::new(tokio::sync::Mutex::new(vec![None::<Instant>; a.streams]));
    let mut first_notifies = Vec::with_capacity(a.streams);

    for i in 0..a.streams {
        let fn_ = Arc::new(Notify::new());
        first_notifies.push(fn_.clone());
        let s = srv.clone();
        let rs = rows_shared.clone();
        let es = ended_shared.clone();
        let mt = a.max_tokens;
        let msgs = json!([{"role": "user", "content": format!("[pair {salt}-{i}] Write a long, detailed story about a lighthouse keeper who finds a map. Do not stop early.")}]);
        tokio::spawn(async move {
            let row = s
                .stream(&msgs, mt, None, false, None, Some(fn_), false)
                .await;
            rs.lock().await[i] = row;
            es.lock().await[i] = Some(Instant::now());
        });
    }

    for n in &first_notifies {
        let _ = tokio::time::timeout(Duration::from_secs(600), n.notified()).await;
    }
    tokio::time::sleep(Duration::from_secs_f64(a.decode_s)).await;

    let early: Vec<usize> = {
        let es = ended_shared.lock().await;
        (0..a.streams).filter(|&i| es[i].is_some()).collect()
    };
    rc.check(
        "decoding",
        early.is_empty(),
        &if early.is_empty() {
            format!("{} streams decoding at the kill", a.streams)
        } else {
            format!("streams {early:?} had already ended before the kill")
        },
        Map::new(),
    );

    let t_kill = Instant::now();
    let (krc, kout) = run_shell_cmd(&a.kill_cmd);
    rc.check(
        "kill",
        krc == 0,
        &format!(
            "{:?} exit {krc} {}",
            a.kill_cmd,
            kout.trim()
                .chars()
                .rev()
                .take(200)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<String>()
        ),
        Map::new(),
    );

    let mut down: Option<f64> = None;
    let mut said: Option<Value> = None;
    let mut exited: Option<f64> = None;
    let mut last_state: Option<(Option<String>, Option<i32>, String)> = None;
    let deadline = t_kill + Duration::from_secs_f64(a.end_deadline);

    while Instant::now() < deadline {
        if down.is_none() {
            let h = srv.health().await;
            if h.get("status").and_then(|v| v.as_str()) != Some("ok") {
                down = Some(t_kill.elapsed().as_secs_f64());
                said = Some(h);
            }
        }
        if exited.is_none() {
            if let Some(ec) = &a.exit_cmd {
                let st = poll_container_state(ec);
                let done_st =
                    !matches!(st.0.as_deref(), None | Some("running") | Some("restarting"));
                last_state = Some(st);
                if done_st {
                    exited = Some(t_kill.elapsed().as_secs_f64());
                }
            }
        }
        let all_ended = ended_shared.lock().await.iter().all(|x| x.is_some());
        if all_ended && down.is_some() && (exited.is_some() || a.exit_cmd.is_none()) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    let es = ended_shared.lock().await.clone();
    let rows = rows_shared.lock().await.clone();
    let hung: Vec<usize> = (0..a.streams).filter(|&i| es[i].is_none()).collect();
    let last_end = es.iter().filter_map(|&x| x).max();
    let streams_ended = if hung.is_empty() {
        last_end
            .map(|e| (e.saturating_duration_since(t_kill).as_secs_f64() * 100.0).round() / 100.0)
    } else {
        None
    };
    let mut extra_rows = Map::new();
    extra_rows.insert("rows".to_string(), Value::Array(rows.clone()));
    rc.check(
        "streams end",
        hung.is_empty() && streams_ended.map(|s| s <= a.end_deadline).unwrap_or(true),
        &if !hung.is_empty() {
            format!(
                "streams {hung:?} still open {}s after the kill",
                a.end_deadline
            )
        } else {
            format!(
                "all {} ended {:?}s after the kill",
                a.streams, streams_ended
            )
        },
        extra_rows,
    );
    rc.check(
        "down",
        down.is_some(),
        &match down {
            Some(d) => format!(
                "/health stopped answering ok {d:.2}s after the kill: {:?}",
                said
            ),
            None => format!(
                "/health still answered ok {}s after the kill",
                a.end_deadline
            ),
        },
        Map::new(),
    );
    if a.exit_cmd.is_some() {
        let (st, code, raw) = last_state.unwrap_or((None, None, "not read".to_string()));
        let mut ex_m = Map::new();
        ex_m.insert("state".to_string(), json!(st));
        ex_m.insert("code".to_string(), json!(code));
        rc.check(
            "exited",
            exited.is_some() && code == Some(a.exit_code),
            &match exited {
                Some(e) => format!("rank 0 container {:?} {:?}, {e:.2}s after kill", st, code),
                None => format!(
                    "rank 0 container still {:?} ({raw}) {}s after kill",
                    st, a.end_deadline
                ),
            },
            ex_m,
        );
    } else {
        rc.add("exited", "SKIP", "no --exit-cmd", Map::new());
    }
    for spec in &a.expect_log {
        if let Some((cmd, pat)) = spec.rsplit_once("::") {
            let (lrc, lout) = run_shell_cmd(cmd);
            let re = Regex::new(pat).ok();
            let matched = re.as_ref().and_then(|r| r.find(&lout));
            rc.check(
                &format!("log /{pat}/"),
                matched.is_some(),
                &match matched {
                    Some(m) => m.as_str().chars().take(200).collect(),
                    None => format!("{cmd:?} (exit {lrc}) has no /{pat}/"),
                },
                Map::new(),
            );
        }
    }
    rc.doc.insert(
        "times".to_string(),
        json!({
            "streams_ended": streams_ended,
            "down": down.map(|d| (d * 100.0).round() / 100.0),
            "exited": exited.map(|e| (e * 100.0).round() / 100.0),
        }),
    );
    rc.finish(a.receipt.as_deref(), false)
}

// ====================================================================== resume-bench
const RB_Q: &str = "List the three most frequent words in the text above.";
const RB_KEYS: &[&str] = &[
    "prompt_cache.memory_hits",
    "prompt_cache.memory_misses",
    "prompt_cache.disk_hits",
    "prompt_cache.disk_hits_committed",
    "prompt_cache.restored_tokens",
    "prompt_cache.restore_wait_ns",
    "prompt_cache.restores_failed",
    "prompt_cache.restores_refused",
    "prompt_cache.restores_begun",
    "prompt_cache.kept",
    "prompt_cache.evicted",
    "prompt_cache.cached_tokens",
    "prompt_cache.computed_tokens",
];
const RB_RANK_KEYS: &[&str] = &[
    "reads",
    "read_bytes",
    "read_failed",
    "crc_errors",
    "size_errors",
    "presence_refused",
    "writes_committed",
    "written_bytes",
    "writes_failed",
    "no_space",
];

#[derive(Clone, ValueEnum)]
pub enum ResumeMode {
    Seed,
    Resume,
    Report,
}

#[derive(Args)]
pub struct ResumeBenchArgs {
    #[arg(value_enum)]
    pub mode: ResumeMode,
    #[arg(long)]
    pub state: PathBuf,
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value_t = 1800.0)]
    pub timeout: f64,
    #[arg(long = "health-timeout", default_value_t = 900.0)]
    pub health_timeout: f64,
    #[arg(long = "commit-timeout", default_value_t = 600.0)]
    pub commit_timeout: f64,
    #[arg(long, default_value = "100000,150000,200000,250000")]
    pub sizes: String,
    #[arg(long, default_value_t = 2)]
    pub reps: usize,
    #[arg(long = "warm-tokens", default_value_t = 20_000)]
    pub warm_tokens: usize,
    #[arg(long = "max-tokens", default_value_t = 32)]
    pub max_tokens: u64,
    #[arg(long)]
    pub salt: Option<String>,
    #[arg(long, default_value_t = 1234)]
    pub seed: i64,
    #[arg(long, default_value = "")]
    pub label: String,
    #[arg(long = "overlap-retries", default_value_t = 2)]
    pub overlap_retries: usize,
    #[arg(long = "idle-wait", default_value_t = 300.0)]
    pub idle_wait: f64,
    #[arg(long)]
    pub md: Option<PathBuf>,
    #[arg(long = "api-key-file")]
    pub api_key_file: Option<PathBuf>,
}

fn rb_msgs(salt: &str, name: &str, tokens: usize, seed: i64) -> Value {
    let name_sum: i64 = name.chars().map(|c| c as i64).sum();
    let doc = format!(
        "[ref {salt}-{name}]\n{}",
        words(words_for(tokens), seed + name_sum * 7919)
    );
    json!([{"role": "user", "content": format!("{doc}\n\n{RB_Q}")}])
}

fn save_state_atomic(path: &PathBuf, st: &Value) {
    let tmp = path.with_extension("json.tmp");
    if let Ok(s) = serde_json::to_string_pretty(st) {
        if fs::write(&tmp, s).is_ok() {
            let _ = fs::rename(&tmp, path);
        }
    }
}

async fn server_busy(srv: &Server) -> bool {
    let h = srv.health().await;
    let conns = h["live"]["connections"].as_u64().unwrap_or(0);
    let c = health_counters(&h);
    conns > 0
        || c.get("prompt_cache.restores_active")
            .copied()
            .unwrap_or(0.0)
            > 0.0
        || c.get("spill.writes_pending").copied().unwrap_or(0.0) > 0.0
}

async fn timed_rb(
    srv: &Server,
    st: &mut Value,
    state_path: &PathBuf,
    idx: usize,
    kind: &str,
    overlap_retries: usize,
    idle_wait: f64,
) -> (Value, Counters) {
    let mut last_c0 = Counters::new();
    let mut last_row = Value::Null;
    for attempt in 0..=overlap_retries {
        if attempt > 0 && kind == "cold" {
            st["items"][idx]["alt"] = json!(format!("~{attempt}"));
            save_state_atomic(state_path, st);
        }
        let deadline = Instant::now() + Duration::from_secs_f64(idle_wait);
        while Instant::now() < deadline {
            if !server_busy(srv).await {
                break;
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        let c0 = srv.counters().await;
        last_c0 = c0.clone();

        let h0 = srv.health().await;
        let before_conns = h0["live"]["connections"].as_u64().unwrap_or(0);
        let peak = Arc::new(AtomicU64::new(0));
        let waiting = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let wsrv = srv.clone();
        let wp = peak.clone();
        let ww = waiting.clone();
        let wst = stop.clone();
        let watch = tokio::spawn(async move {
            while !wst.load(Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_millis(500)).await;
                if wst.load(Ordering::Relaxed) {
                    break;
                }
                let h = wsrv.health().await;
                wp.fetch_max(
                    h["live"]["connections"].as_u64().unwrap_or(0),
                    Ordering::Relaxed,
                );
                ww.fetch_max(
                    h["live"]["waiting"].as_u64().unwrap_or(0),
                    Ordering::Relaxed,
                );
            }
        });

        let salt = format!(
            "{}{}",
            st["meta"]["salt"].as_str().unwrap_or(""),
            st["items"][idx]["alt"].as_str().unwrap_or("")
        );
        let name = st["items"][idx]["name"].as_str().unwrap_or("").to_string();
        let tokens = st["items"][idx]["tokens"].as_u64().unwrap_or(0) as usize;
        let seed = st["meta"]["seed"].as_i64().unwrap_or(1234);
        let mt = st["meta"]["max_tokens"].as_u64().unwrap_or(32);

        let mut row = srv
            .stream(
                &rb_msgs(&salt, &name, tokens, seed),
                mt,
                None,
                false,
                None,
                None,
                false,
            )
            .await;
        stop.store(true, Ordering::Relaxed);
        let _ = watch.await;
        tokio::time::sleep(Duration::from_millis(1300)).await;
        let c1 = srv.counters().await;
        let d = deltas(&c0, &c1, RB_KEYS.iter().copied());
        let mut per = Map::new();
        for &k in RB_RANK_KEYS {
            per.insert(k.to_string(), json!(rank_deltas(&c0, &c1, k)));
        }
        let pk = peak.load(Ordering::Relaxed);
        let wt = waiting.load(Ordering::Relaxed);
        let pt = row["prompt_tokens"].as_i64().unwrap_or(0);
        let ct = row["cached_tokens"].as_i64().unwrap_or(0);
        let mh = d
            .get("prompt_cache.memory_hits")
            .and_then(|x| *x)
            .unwrap_or(0.0);
        let mm = d
            .get("prompt_cache.memory_misses")
            .and_then(|x| *x)
            .unwrap_or(0.0);
        let dh = d
            .get("prompt_cache.disk_hits_committed")
            .and_then(|x| *x)
            .unwrap_or(0.0);
        let overlap = before_conns > 0 || pk > 1 || (mh + mm) > 1.0;

        if let Some(obj) = row.as_object_mut() {
            obj.insert("kind".to_string(), json!(kind));
            obj.insert("attempt".to_string(), json!(attempt));
            obj.insert("deltas".to_string(), json!(d));
            obj.insert("rank".to_string(), Value::Object(per));
            obj.insert("peak_connections".to_string(), json!(pk));
            obj.insert("peak_waiting".to_string(), json!(wt));
            obj.insert("connections_before".to_string(), json!(before_conns));
            obj.insert("tail".to_string(), json!(pt - ct));
            obj.insert("overlap".to_string(), json!(overlap));
            obj.insert("at".to_string(), json!(now_hms()));
        }
        last_row = row;
        if !overlap || kind == "warm" || (kind == "disk" && dh >= 1.0) {
            return (last_row, last_c0);
        }
    }
    (last_row, last_c0)
}

pub async fn run_resume_bench(a: ResumeBenchArgs) -> i32 {
    let key = match resolve_strict(a.api_key_file.as_deref()) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let url = a.url.unwrap_or_else(default_url);
    let model = a
        .model
        .or_else(|| env::var("MODEL").ok())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let mut st: Value = if a.state.exists() {
        fs::read_to_string(&a.state)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| json!({}))
    } else {
        json!({})
    };

    if !matches!(a.mode, ResumeMode::Seed) && st.as_object().map(|m| m.is_empty()).unwrap_or(true) {
        eprintln!("no state in {}: run seed first", a.state.display());
        return 2;
    }

    match a.mode {
        ResumeMode::Seed => {
            let sizes: Vec<usize> = a
                .sizes
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            let salt = a.salt.unwrap_or_else(|| random_hex(4));
            let mut items =
                vec![json!({"name": "W", "tokens": a.warm_tokens, "rep": -1, "warm": true})];
            for r in 0..a.reps {
                for &s in &sizes {
                    items.push(json!({"name": format!("R{r}-{s}"), "tokens": s, "rep": r}));
                }
            }
            st = json!({
                "meta": {
                    "salt": salt, "seed": a.seed, "model": model, "max_tokens": a.max_tokens,
                    "sizes": sizes, "reps": a.reps, "url": url, "label": a.label,
                    "seeded": now_iso8601()
                },
                "items": items,
                "rows": {}
            });
            let srv = Server::new(&url, &model, a.timeout, a.seed, key);
            if !srv.wait_healthy(a.health_timeout).await {
                eprintln!("server not healthy");
                return 1;
            }
            save_state_atomic(&a.state, &st);
            let n_items = st["items"].as_array().map(|v| v.len()).unwrap_or(0);
            for i in 0..n_items {
                let name = st["items"][i]["name"].as_str().unwrap_or("").to_string();
                let is_warm = st["items"][i]["warm"].as_bool().unwrap_or(false);
                let (cold, c0) = timed_rb(
                    &srv,
                    &mut st,
                    &a.state,
                    i,
                    "cold",
                    a.overlap_retries,
                    a.idle_wait,
                )
                .await;
                let t_wait = Instant::now();
                let deadline = t_wait + Duration::from_secs_f64(a.commit_timeout);
                let mut committed = BTreeMap::<String, f64>::new();
                while Instant::now() < deadline {
                    let c = srv.counters().await;
                    let rd = rank_deltas(&c0, &c, "writes_committed");
                    let d: BTreeMap<String, f64> = if !rd.is_empty() {
                        rd.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
                    } else if let Some(v) = delta(&c0, &c, "spill.writes_committed") {
                        [("all".to_string(), v)].into_iter().collect()
                    } else {
                        BTreeMap::new()
                    };
                    if !d.is_empty()
                        && d.values().copied().fold(f64::INFINITY, f64::min) >= 1.0
                        && c.get("spill.writes_pending").copied().unwrap_or(0.0) == 0.0
                    {
                        committed = d;
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                let mut item_rows = Map::new();
                item_rows.insert("cold".to_string(), cold.clone());
                item_rows.insert(
                    "commit".to_string(),
                    json!({"per_rank": committed, "waited_s": (t_wait.elapsed().as_secs_f64() * 100.0).round() / 100.0}),
                );
                if cold.get("error").is_some() || committed.is_empty() {
                    st["rows"][&name] = Value::Object(item_rows);
                    save_state_atomic(&a.state, &st);
                    if is_warm {
                        return 1;
                    }
                    continue;
                }
                if !is_warm {
                    let (mem, _) = timed_rb(
                        &srv,
                        &mut st,
                        &a.state,
                        i,
                        "memory",
                        a.overlap_retries,
                        a.idle_wait,
                    )
                    .await;
                    item_rows.insert("memory".to_string(), mem);
                }
                st["rows"][&name] = Value::Object(item_rows);
                save_state_atomic(&a.state, &st);
            }
            0
        }
        ResumeMode::Resume => {
            let m_model = st["meta"]["model"].as_str().unwrap_or(&model).to_string();
            let seed = st["meta"]["seed"].as_i64().unwrap_or(1234);
            let srv = Server::new(&url, &m_model, a.timeout, seed, key);
            if !srv.wait_healthy(a.health_timeout).await {
                eprintln!("server not healthy");
                return 1;
            }
            let salt = st["meta"]["salt"].as_str().unwrap_or("").to_string();
            let _ = srv
                .stream(
                    &rb_msgs(
                        &format!("{salt}-x"),
                        &format!("first-{}", random_hex(3)),
                        600,
                        seed,
                    ),
                    8,
                    None,
                    false,
                    None,
                    None,
                    false,
                )
                .await;
            let n_items = st["items"].as_array().map(|v| v.len()).unwrap_or(0);
            for i in 0..n_items {
                let name = st["items"][i]["name"].as_str().unwrap_or("").to_string();
                let is_warm = st["items"][i]["warm"].as_bool().unwrap_or(false);
                let kind = if is_warm { "warm" } else { "disk" };
                let (row, _) = timed_rb(
                    &srv,
                    &mut st,
                    &a.state,
                    i,
                    kind,
                    a.overlap_retries,
                    a.idle_wait,
                )
                .await;
                if !st["rows"][&name].is_object() {
                    st["rows"][&name] = json!({});
                }
                st["rows"][&name]["disk"] = row;
                save_state_atomic(&a.state, &st);
            }
            0
        }
        ResumeMode::Report => {
            let mut bad = Vec::new();
            let mut by: BTreeMap<u64, Vec<Value>> = BTreeMap::new();
            if let Some(items) = st["items"].as_array() {
                for it in items {
                    if it["warm"].as_bool().unwrap_or(false) {
                        continue;
                    }
                    let name = it["name"].as_str().unwrap_or("");
                    let toks = it["tokens"].as_u64().unwrap_or(0);
                    let rows = &st["rows"][name];
                    by.entry(toks).or_default().push(rows.clone());
                    for k in ["cold", "memory", "disk"] {
                        if rows.get(k).map(|v| v.is_null()).unwrap_or(true) {
                            bad.push(format!("{name} {k}: no row"));
                        }
                    }
                    let cold = rows.get("cold");
                    if let Some(c) = cold {
                        if c["cached_tokens"].as_i64().unwrap_or(0) > 0 {
                            bad.push(format!("{name} cold: not cold"));
                        }
                        if c["overlap"].as_bool().unwrap_or(false) {
                            bad.push(format!("{name} cold: overlapped"));
                        }
                        for kind in ["memory", "disk"] {
                            if let Some(row) = rows.get(kind) {
                                let (ok, detail) = same_reply(Some(c), row, false);
                                if !ok {
                                    bad.push(format!("{name} {kind}: {detail}"));
                                }
                                let reused = row["cached_tokens"].as_i64().unwrap_or(0) > 0;
                                let d = &row["deltas"];
                                let dh = d["prompt_cache.disk_hits_committed"]
                                    .as_f64()
                                    .unwrap_or(0.0);
                                let mh = d["prompt_cache.memory_hits"].as_f64().unwrap_or(0.0);
                                let rt = d["prompt_cache.restored_tokens"].as_i64().unwrap_or(0);
                                let src = reused
                                    && if kind == "memory" {
                                        mh >= 1.0 && dh == 0.0
                                    } else {
                                        dh >= 1.0
                                            && row["cached_tokens"].as_i64().unwrap_or(0) == rt
                                    };
                                if !src {
                                    bad.push(format!("{name} {kind}: not from {kind}"));
                                }
                                if row["overlap"].as_bool().unwrap_or(false) {
                                    bad.push(format!("{name} {kind}: overlapped"));
                                }
                            }
                        }
                    }
                }
            }
            let mut lines = vec![
                "| prompt | cold TTFT | memory TTFT | disk TTFT | equal |".to_string(),
                "|---:|---:|---:|---:|:---:|".to_string(),
            ];
            for (toks, rs) in &by {
                let c0 = rs
                    .first()
                    .and_then(|r| r["cold"]["ttft"].as_f64())
                    .unwrap_or(0.0);
                let m0 = rs
                    .first()
                    .and_then(|r| r["memory"]["ttft"].as_f64())
                    .unwrap_or(0.0);
                let d0 = rs
                    .first()
                    .and_then(|r| r["disk"]["ttft"].as_f64())
                    .unwrap_or(0.0);
                let eq = rs.iter().all(|r| {
                    same_reply(r.get("cold"), &r["memory"], false).0
                        && same_reply(r.get("cold"), &r["disk"], false).0
                });
                lines.push(format!(
                    "| {toks} | {c0:.2} s | {m0:.2} s | {d0:.2} s | {} |",
                    if eq { "yes" } else { "NO" }
                ));
            }
            let md = lines.join("\n");
            println!("{md}");
            for b in &bad {
                println!("FAIL {b}");
            }
            println!(
                "{} resume_bench: {} sizes, {} problems",
                if bad.is_empty() { "PASS" } else { "FAIL" },
                by.len(),
                bad.len()
            );
            if let Some(mp) = a.md {
                let _ = fs::write(&mp, format!("{md}\n"));
            }
            if bad.is_empty() {
                0
            } else {
                1
            }
        }
    }
}
