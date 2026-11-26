//! Streaming benchmark (`dsv41-tools bench`) replacing `tools/bench.py`.

use clap::Args;
use dsv41_core::auth::{default_base_url, resolve_standard};
use dsv41_core::prose::{prose_str_seed, TOK_PER_SIZE};
use dsv41_core::{now_iso8601, random_hex, DEFAULT_MODEL};
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Barrier;

const SCHEMA: u64 = 1;
const SECRET: &str = "violet-harbor-7291";
const DECODE_PROMPT: &str = "Write a long, detailed essay about the history of the printing press, with many paragraphs. Do not stop early. Variation tag: ";

#[derive(Args)]
pub struct BenchArgs {
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value = "prefill,decode,multi,needle")]
    pub suite: String,
    #[arg(long, default_value = "")]
    pub label: String,
    #[arg(long)]
    pub out: Option<PathBuf>,
    #[arg(long, default_value_t = 3)]
    pub reps: usize,
    #[arg(long = "prefill-sizes", default_value = "2048,8192,32768")]
    pub prefill_sizes: String,
    #[arg(long = "decode-tokens", default_value_t = 256)]
    pub decode_tokens: u64,
    #[arg(long, default_value = "1,2,4")]
    pub streams: String,
    #[arg(long = "needle-sizes", default_value = "8k,32k,128k")]
    pub needle_sizes: String,
    #[arg(long, default_value = "0.6")]
    pub depths: String,
    #[arg(long)]
    pub thinking: bool,
    #[arg(long, default_value_t = 1234)]
    pub seed: i64,
    #[arg(long)]
    pub salt: Option<String>,
    #[arg(long, default_value_t = 3600.0)]
    pub timeout: f64,
    #[arg(long)]
    pub health: bool,
    #[arg(long)]
    pub metrics: bool,
    #[arg(long, num_args = 2, value_names = ["A.json", "B.json"])]
    pub compare: Option<Vec<PathBuf>>,
}

#[derive(Clone)]
struct Ctx {
    url: String,
    model: String,
    reps: usize,
    seed: i64,
    salt: String,
    thinking: bool,
    timeout: Duration,
    decode_tokens: u64,
    prefill_sizes: Vec<usize>,
    needle_sizes: Vec<usize>,
    streams: Vec<usize>,
    depths: Vec<f64>,
    http: reqwest::Client,
}

fn parse_size_list(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if let Some(prefix) = p.strip_suffix('k').or_else(|| p.strip_suffix('K')) {
            if !prefix.is_empty() && prefix.bytes().all(|b| b.is_ascii_digit()) {
                let v: usize = prefix.parse().map_err(|e| format!("bad size {p}: {e}"))?;
                out.push(v * 1000);
                continue;
            }
        }
        let v: usize = p.parse().map_err(|e| format!("bad size {p}: {e}"))?;
        out.push(v);
    }
    Ok(out)
}

fn collect_numeric_leaves(val: &Value, prefix: &str, out: &mut BTreeMap<String, f64>) {
    match val {
        Value::Object(map) => {
            for (k, v) in map {
                let next = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                collect_numeric_leaves(v, &next, out);
            }
        }
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                out.insert(prefix.to_string(), f);
            }
        }
        _ => {}
    }
}

fn draft_figures(server: &Value) -> Map<String, Value> {
    let mut leaves = BTreeMap::new();
    collect_numeric_leaves(server, "", &mut leaves);
    let mut d = Map::new();
    let mut drafted: Option<f64> = None;
    let mut accepted: Option<f64> = None;
    for (k, v) in leaves {
        let kl = k.to_lowercase();
        if kl.contains("draft") || kl.contains("accept") {
            if drafted.is_none()
                && kl.contains("draft")
                && !kl.contains("accept")
                && !kl.contains("rate")
            {
                drafted = Some(v);
            }
            if accepted.is_none() && kl.contains("accept") && !kl.contains("rate") {
                accepted = Some(v);
            }
            d.insert(k, json!(v));
        }
    }
    if let (Some(dr), Some(ac)) = (drafted, accepted) {
        if dr > 0.0 {
            d.insert("accepted_over_drafted".to_string(), json!(ac / dr));
        }
    }
    d
}

fn median(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let mid = s.len() / 2;
    if s.len() % 2 == 1 {
        Some(s[mid])
    } else {
        Some((s[mid - 1] + s[mid]) / 2.0)
    }
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.1}"),
        None => "n/a".to_string(),
    }
}

async fn stream_chat(
    c: &Ctx,
    messages: Value,
    max_tokens: u64,
    gate: Option<Arc<Barrier>>,
    base_instant: Instant,
) -> Map<String, Value> {
    let body = json!({
        "model": c.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": c.seed,
        "stream": true,
        "stream_options": {"include_usage": true},
        "chat_template_kwargs": {"enable_thinking": c.thinking}
    });
    if let Some(g) = gate {
        g.wait().await;
    }
    let t0 = Instant::now();
    let resp = match c
        .http
        .post(format!("{}/v1/chat/completions", c.url))
        .timeout(c.timeout)
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            let mut m = Map::new();
            m.insert("error".to_string(), json!(format!("RequestError: {e}")));
            return m;
        }
    };
    if !resp.status().is_success() {
        let code = resp.status().as_u16();
        let bytes = resp.bytes().await.unwrap_or_default();
        let short: String = String::from_utf8_lossy(&bytes).chars().take(300).collect();
        let mut m = Map::new();
        m.insert("error".to_string(), json!(format!("HTTP {code}: {short}")));
        return m;
    }

    let mut first: Option<Instant> = None;
    let mut last: Option<Instant> = None;
    let mut usage: Option<Value> = None;
    let mut server: Option<Value> = None;
    let mut text = String::new();
    let mut events: u64 = 0;
    let mut finish: Option<String> = None;

    let mut stream = resp.bytes_stream();
    let mut buf = Vec::<u8>::with_capacity(4096);
    'read: while let Some(item) = stream.next().await {
        let chunk = match item {
            Ok(b) => b,
            Err(e) => {
                let mut m = Map::new();
                m.insert("error".to_string(), json!(format!("StreamError: {e}")));
                return m;
            }
        };
        buf.extend_from_slice(&chunk);
        while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
            let raw: Vec<u8> = buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&raw);
            let line = line.trim();
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                break 'read;
            }
            let Ok(ev) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if let Some(u) = ev.get("usage") {
                if !u.is_null() {
                    usage = Some(u.clone());
                }
            }
            if let Some(tf) = ev.get("tensorfold") {
                if !tf.is_null() {
                    server = Some(tf.clone());
                }
            }
            if let Some(choices) = ev.get("choices").and_then(|v| v.as_array()) {
                for ch in choices {
                    let delta = ch.get("delta");
                    let piece = delta
                        .and_then(|d| d.get("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let has_reason = delta
                        .and_then(|d| d.get("reasoning_content").or_else(|| d.get("reasoning")))
                        .and_then(|v| v.as_str())
                        .map(|s| !s.is_empty())
                        .unwrap_or(false);
                    if !piece.is_empty() || has_reason {
                        let now = Instant::now();
                        if first.is_none() {
                            first = Some(now);
                        }
                        last = Some(now);
                        events += 1;
                        text.push_str(piece);
                    }
                    if let Some(fr) = ch.get("finish_reason").and_then(|v| v.as_str()) {
                        if !fr.is_empty() {
                            finish = Some(fr.to_string());
                        }
                    }
                }
            }
        }
    }

    let end = Instant::now();
    let (Some(first_t), Some(last_t), Some(u)) = (first, last, usage) else {
        let mut m = Map::new();
        m.insert(
            "error".to_string(),
            json!(format!("no tokens or no usage chunk (events={events})")),
        );
        return m;
    };

    let pt = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    let ct = u
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let mut cached = u
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64());
    if cached.is_none() {
        if let Some(smap) = server.as_ref().and_then(|s| s.as_object()) {
            for (k, v) in smap {
                if k.to_lowercase().contains("cached") {
                    if let Some(cv) = v.as_u64() {
                        cached = Some(cv);
                        break;
                    }
                }
            }
        }
    }
    let ttft = first_t.duration_since(t0).as_secs_f64();
    let uncached = pt.saturating_sub(cached.unwrap_or(0)) as f64;
    let prefill_tok_s = if ttft > 0.0 {
        Some(uncached / ttft)
    } else {
        None
    };
    let dec_span = last_t.duration_since(first_t).as_secs_f64();
    let decode_tok_s = if ct > 1 && dec_span > 0.0 {
        Some((ct - 1) as f64 / dec_span)
    } else {
        None
    };

    let mut r = Map::new();
    r.insert("prompt_tokens".to_string(), json!(pt));
    r.insert("completion_tokens".to_string(), json!(ct));
    r.insert("cached_tokens".to_string(), json!(cached));
    r.insert("ttft_s".to_string(), json!(ttft));
    r.insert("prefill_tok_s".to_string(), json!(prefill_tok_s));
    r.insert("decode_tok_s".to_string(), json!(decode_tok_s));
    r.insert(
        "total_s".to_string(),
        json!(end.duration_since(t0).as_secs_f64()),
    );
    r.insert("sse_events".to_string(), json!(events));
    r.insert("finish_reason".to_string(), json!(finish));
    r.insert(
        "t_first".to_string(),
        json!(first_t.duration_since(base_instant).as_secs_f64()),
    );
    r.insert(
        "t_last".to_string(),
        json!(last_t.duration_since(base_instant).as_secs_f64()),
    );
    r.insert("text".to_string(), json!(text));
    if let Some(srv) = server {
        let df = draft_figures(&srv);
        r.insert("server".to_string(), srv);
        r.insert("draft".to_string(), Value::Object(df));
    }
    r
}

fn show_row(r: &Map<String, Value>) {
    let suite = r.get("suite").and_then(|v| v.as_str()).unwrap_or("");
    if let Some(err) = r.get("error").and_then(|v| v.as_str()) {
        println!("  {suite} ERROR {err}");
        return;
    }
    let pt = r.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    let ct = r
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cached = r
        .get("cached_tokens")
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".to_string());
    let ttft = r.get("ttft_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let pref = fmt_opt(r.get("prefill_tok_s").and_then(|v| v.as_f64()));
    let dec = fmt_opt(r.get("decode_tok_s").and_then(|v| v.as_f64()));
    let extra = if suite == "needle" {
        let depth = r.get("depth").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ok = r.get("correct").and_then(|v| v.as_bool()).unwrap_or(false);
        format!(" depth {depth} {}", if ok { "CORRECT" } else { "WRONG" })
    } else {
        String::new()
    };
    let dr = r
        .get("draft")
        .and_then(|d| d.get("accepted_over_drafted"))
        .and_then(|v| v.as_f64())
        .map(|x| format!(" accepted/drafted {x:.2}"))
        .unwrap_or_default();
    println!(
        "  {suite:7} prompt {pt:7} (cached {cached}) ttft {ttft:7.2}s prefill {pref:>8} tok/s | out {ct:4} decode {dec:>6} tok/s{extra}{dr}"
    );
}

fn summarize(runs: &[Map<String, Value>]) -> Map<String, Value> {
    let mut s = Map::new();
    let prefill_sizes: BTreeSet<u64> = runs
        .iter()
        .filter(|r| {
            r.get("suite").and_then(|v| v.as_str()) == Some("prefill") && !r.contains_key("error")
        })
        .filter_map(|r| r.get("target_tokens").and_then(|v| v.as_u64()))
        .collect();
    for size in prefill_sizes {
        let ttfts: Vec<f64> = runs
            .iter()
            .filter(|r| {
                r.get("suite").and_then(|v| v.as_str()) == Some("prefill")
                    && r.get("target_tokens").and_then(|v| v.as_u64()) == Some(size)
                    && !r.contains_key("error")
            })
            .filter_map(|r| r.get("ttft_s").and_then(|v| v.as_f64()))
            .collect();
        let toks: Vec<f64> = runs
            .iter()
            .filter(|r| {
                r.get("suite").and_then(|v| v.as_str()) == Some("prefill")
                    && r.get("target_tokens").and_then(|v| v.as_u64()) == Some(size)
                    && !r.contains_key("error")
            })
            .filter_map(|r| r.get("prefill_tok_s").and_then(|v| v.as_f64()))
            .collect();
        s.insert(format!("prefill.{size}.ttft_s"), json!(median(&ttfts)));
        s.insert(format!("prefill.{size}.tok_s"), json!(median(&toks)));
    }

    let dec_runs: Vec<&Map<String, Value>> = runs
        .iter()
        .filter(|r| {
            r.get("suite").and_then(|v| v.as_str()) == Some("decode") && !r.contains_key("error")
        })
        .collect();
    if !dec_runs.is_empty() {
        let dt: Vec<f64> = dec_runs
            .iter()
            .filter_map(|r| r.get("decode_tok_s").and_then(|v| v.as_f64()))
            .collect();
        let tt: Vec<f64> = dec_runs
            .iter()
            .filter_map(|r| r.get("ttft_s").and_then(|v| v.as_f64()))
            .collect();
        let acc: Vec<f64> = dec_runs
            .iter()
            .filter_map(|r| {
                r.get("draft")
                    .and_then(|d| d.get("accepted_over_drafted"))
                    .and_then(|v| v.as_f64())
            })
            .collect();
        s.insert("decode.tok_s".to_string(), json!(median(&dt)));
        s.insert("decode.ttft_s".to_string(), json!(median(&tt)));
        s.insert(
            "decode.accepted_over_drafted".to_string(),
            json!(median(&acc)),
        );
    }

    let multi_streams: BTreeSet<u64> = runs
        .iter()
        .filter(|r| r.get("suite").and_then(|v| v.as_str()) == Some("multi"))
        .filter_map(|r| r.get("streams").and_then(|v| v.as_u64()))
        .collect();
    for n in multi_streams {
        let rs: Vec<&Map<String, Value>> = runs
            .iter()
            .filter(|r| {
                r.get("suite").and_then(|v| v.as_str()) == Some("multi")
                    && r.get("streams").and_then(|v| v.as_u64()) == Some(n)
                    && r.get("errors")
                        .and_then(|e| e.as_array())
                        .map(|a| a.is_empty())
                        .unwrap_or(true)
            })
            .collect();
        let agg: Vec<f64> = rs
            .iter()
            .filter_map(|r| r.get("aggregate_tok_s").and_then(|v| v.as_f64()))
            .collect();
        let per: Vec<f64> = rs
            .iter()
            .filter_map(|r| r.get("mean_stream_decode_tok_s").and_then(|v| v.as_f64()))
            .collect();
        s.insert(format!("multi.{n}.aggregate_tok_s"), json!(median(&agg)));
        s.insert(format!("multi.{n}.per_stream_tok_s"), json!(median(&per)));
    }

    let needle_sizes: BTreeSet<u64> = runs
        .iter()
        .filter(|r| r.get("suite").and_then(|v| v.as_str()) == Some("needle"))
        .filter_map(|r| r.get("target_tokens").and_then(|v| v.as_u64()))
        .collect();
    for size in needle_sizes {
        let rs: Vec<&Map<String, Value>> = runs
            .iter()
            .filter(|r| {
                r.get("suite").and_then(|v| v.as_str()) == Some("needle")
                    && r.get("target_tokens").and_then(|v| v.as_u64()) == Some(size)
            })
            .collect();
        let ok_n = rs
            .iter()
            .filter(|r| r.get("correct").and_then(|v| v.as_bool()).unwrap_or(false))
            .count();
        let pts: Vec<f64> = rs
            .iter()
            .filter_map(|r| r.get("prompt_tokens").and_then(|v| v.as_f64()))
            .collect();
        let ttfts: Vec<f64> = rs
            .iter()
            .filter(|r| !r.contains_key("error"))
            .filter_map(|r| r.get("ttft_s").and_then(|v| v.as_f64()))
            .collect();
        s.insert(
            format!("needle.{size}.correct"),
            json!(format!("{ok_n}/{}", rs.len())),
        );
        s.insert(
            format!("needle.{size}.actual_prompt_tokens"),
            json!(median(&pts)),
        );
        s.insert(format!("needle.{size}.ttft_s"), json!(median(&ttfts)));
    }

    let errors = runs
        .iter()
        .filter(|r| {
            r.contains_key("error")
                || r.get("errors")
                    .and_then(|e| e.as_array())
                    .map(|a| !a.is_empty())
                    .unwrap_or(false)
        })
        .count();
    s.insert("errors".to_string(), json!(errors));
    s
}

fn compare_runs(a_path: &PathBuf, b_path: &PathBuf) -> i32 {
    let a: Value = match fs::read_to_string(a_path)
        .and_then(|s| serde_json::from_str(&s).map_err(Into::into))
    {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot read {}: {e}", a_path.display());
            return 1;
        }
    };
    let b: Value = match fs::read_to_string(b_path)
        .and_then(|s| serde_json::from_str(&s).map_err(Into::into))
    {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot read {}: {e}", b_path.display());
            return 1;
        }
    };
    println!(
        "A = {} ({})\nB = {} ({})",
        a["label"],
        a_path.display(),
        b["label"],
        b_path.display()
    );
    println!("{:36} {:>12} {:>12} {:>9}", "metric", "A", "B", "B vs A");
    let sa = a.get("summary").and_then(|v| v.as_object());
    let sb = b.get("summary").and_then(|v| v.as_object());
    let mut keys = BTreeSet::new();
    if let Some(m) = sa {
        keys.extend(m.keys().cloned());
    }
    if let Some(m) = sb {
        keys.extend(m.keys().cloned());
    }
    let fmt_val = |v: Option<&Value>| -> String {
        match v {
            None | Some(Value::Null) => "-".to_string(),
            Some(Value::Number(n)) => {
                if let Some(i) = n.as_i64() {
                    i.to_string()
                } else if let Some(f) = n.as_f64() {
                    format!("{f:.2}")
                } else {
                    n.to_string()
                }
            }
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
        }
    };
    for k in keys {
        let x = sa.and_then(|m| m.get(&k));
        let y = sb.and_then(|m| m.get(&k));
        let delta = match (x.and_then(|v| v.as_f64()), y.and_then(|v| v.as_f64())) {
            (Some(xf), Some(yf)) if xf != 0.0 => format!("{:+.1}%", (yf - xf) / xf * 100.0),
            _ => {
                if x == y {
                    String::new()
                } else {
                    "differs".to_string()
                }
            }
        };
        println!("{k:36} {:>12} {:>12} {:>9}", fmt_val(x), fmt_val(y), delta);
    }
    for k in ["model", "salt", "seed"] {
        if a.get(k) != b.get(k) {
            println!("note: {k} differs: {:?} vs {:?}", a.get(k), b.get(k));
        }
    }
    0
}

pub async fn run(a: BenchArgs) -> i32 {
    if let Some(pair) = &a.compare {
        return compare_runs(&pair[0], &pair[1]);
    }
    let cwd = env::current_dir().ok();
    let url = a
        .url
        .unwrap_or_else(|| default_base_url(cwd.as_deref()))
        .trim_end_matches('/')
        .to_string();
    let model = a
        .model
        .or_else(|| env::var("MODEL").ok())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let salt = a.salt.unwrap_or_else(|| random_hex(4));
    let prefill_sizes = match parse_size_list(&a.prefill_sizes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let needle_sizes = match parse_size_list(&a.needle_sizes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let streams = match parse_size_list(&a.streams) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let depths: Vec<f64> = a
        .depths
        .split(',')
        .filter_map(|x| x.trim().parse::<f64>().ok())
        .collect();

    let key = resolve_standard(None);
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if !key.is_empty() {
        if let Ok(v) = HeaderValue::from_str(&format!("Bearer {key}")) {
            headers.insert(AUTHORIZATION, v);
        }
    }
    let http = reqwest::Client::builder()
        .no_proxy()
        .default_headers(headers)
        .build()
        .expect("reqwest client");

    let c = Ctx {
        url: url.clone(),
        model: model.clone(),
        reps: a.reps,
        seed: a.seed,
        salt: salt.clone(),
        thinking: a.thinking,
        timeout: Duration::from_secs_f64(a.timeout.max(1.0)),
        decode_tokens: a.decode_tokens,
        prefill_sizes,
        needle_sizes,
        streams,
        depths,
        http,
    };

    let suites: Vec<&str> = a
        .suite
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    for s in &suites {
        if !matches!(*s, "prefill" | "decode" | "multi" | "needle") {
            eprintln!("unknown suite {s}; choose from decode, multi, needle, prefill");
            return 2;
        }
    }

    let base_instant = Instant::now();
    let started = now_iso8601();
    let mut snapshots = Map::new();
    for (name, on) in [("health", a.health), ("metrics", a.metrics)] {
        if on {
            snapshots.insert(format!("{name}_before"), get_snapshot(&c, name).await);
        }
    }

    let mut runs: Vec<Map<String, Value>> = Vec::new();
    for s in suites {
        println!("[{s}]");
        match s {
            "prefill" => {
                for &size in &c.prefill_sizes {
                    for rep in 0..c.reps {
                        let hay = prose_str_seed(
                            (size as f64 / TOK_PER_SIZE) as usize,
                            &format!("{}-p-{size}-{rep}-{}", c.seed, c.salt),
                        );
                        let msgs = json!([{"role": "user", "content": format!("{hay}\n\nReply with the single word: ok")}]);
                        let mut r = stream_chat(&c, msgs, 1, None, base_instant).await;
                        r.insert("suite".to_string(), json!("prefill"));
                        r.insert("target_tokens".to_string(), json!(size));
                        r.insert("rep".to_string(), json!(rep));
                        show_row(&r);
                        runs.push(r);
                    }
                }
            }
            "decode" => {
                for rep in 0..c.reps {
                    let msgs = json!([{"role": "user", "content": format!("{DECODE_PROMPT}{}-{rep}", c.salt)}]);
                    let mut r = stream_chat(&c, msgs, c.decode_tokens, None, base_instant).await;
                    r.insert("suite".to_string(), json!("decode"));
                    r.insert("rep".to_string(), json!(rep));
                    show_row(&r);
                    runs.push(r);
                }
            }
            "multi" => {
                for &n in &c.streams {
                    for rep in 0..c.reps {
                        let gate = Arc::new(Barrier::new(n));
                        let mut handles = Vec::with_capacity(n);
                        for i in 0..n {
                            let cc = c.clone();
                            let gg = gate.clone();
                            let msgs = json!([{"role": "user", "content": format!("{DECODE_PROMPT}{}-m{n}-{rep}-{i}", c.salt)}]);
                            handles.push(tokio::spawn(async move {
                                stream_chat(&cc, msgs, cc.decode_tokens, Some(gg), base_instant)
                                    .await
                            }));
                        }
                        let mut res = Vec::with_capacity(n);
                        for h in handles {
                            res.push(h.await.unwrap_or_else(|e| {
                                let mut m = Map::new();
                                m.insert("error".to_string(), json!(format!("JoinError: {e}")));
                                m
                            }));
                        }
                        let ok: Vec<&Map<String, Value>> =
                            res.iter().filter(|r| !r.contains_key("error")).collect();
                        let errors: Vec<Value> =
                            res.iter().filter_map(|r| r.get("error").cloned()).collect();
                        let per_stream: Vec<Value> = ok
                            .iter()
                            .map(|r| r.get("decode_tok_s").cloned().unwrap_or(Value::Null))
                            .collect();
                        let ttfts: Vec<Value> = ok
                            .iter()
                            .map(|r| r.get("ttft_s").cloned().unwrap_or(Value::Null))
                            .collect();
                        let mut row = Map::new();
                        row.insert("suite".to_string(), json!("multi"));
                        row.insert("streams".to_string(), json!(n));
                        row.insert("rep".to_string(), json!(rep));
                        row.insert("errors".to_string(), Value::Array(errors.clone()));
                        row.insert(
                            "per_stream_decode_tok_s".to_string(),
                            Value::Array(per_stream),
                        );
                        row.insert("ttft_s".to_string(), Value::Array(ttfts));
                        if !ok.is_empty() {
                            let t_min = ok
                                .iter()
                                .filter_map(|r| r.get("t_first").and_then(|v| v.as_f64()))
                                .fold(f64::INFINITY, f64::min);
                            let t_max = ok
                                .iter()
                                .filter_map(|r| r.get("t_last").and_then(|v| v.as_f64()))
                                .fold(f64::NEG_INFINITY, f64::max);
                            let window = t_max - t_min;
                            let total: u64 = ok
                                .iter()
                                .filter_map(|r| r.get("completion_tokens").and_then(|v| v.as_u64()))
                                .sum();
                            let agg = if window > 0.0 {
                                Some((total.saturating_sub(ok.len() as u64)) as f64 / window)
                            } else {
                                None
                            };
                            let valid_per: Vec<f64> = ok
                                .iter()
                                .filter_map(|r| r.get("decode_tok_s").and_then(|v| v.as_f64()))
                                .collect();
                            let mean_per = if !valid_per.is_empty() {
                                Some(valid_per.iter().sum::<f64>() / valid_per.len() as f64)
                            } else {
                                None
                            };
                            row.insert("completion_tokens".to_string(), json!(total));
                            row.insert("window_s".to_string(), json!(window));
                            row.insert("aggregate_tok_s".to_string(), json!(agg));
                            row.insert("mean_stream_decode_tok_s".to_string(), json!(mean_per));
                            let mut draft_sum = BTreeMap::<String, f64>::new();
                            for r in &ok {
                                if let Some(dm) = r.get("draft").and_then(|d| d.as_object()) {
                                    for (k, v) in dm {
                                        if k != "accepted_over_drafted" {
                                            if let Some(fv) = v.as_f64() {
                                                *draft_sum.entry(k.clone()).or_insert(0.0) += fv;
                                            }
                                        }
                                    }
                                }
                            }
                            row.insert(
                                "draft".to_string(),
                                Value::Object(
                                    draft_sum.into_iter().map(|(k, v)| (k, json!(v))).collect(),
                                ),
                            );
                        }
                        println!(
                            "  multi x{n} rep{rep}: aggregate {} tok/s, mean per-stream {}, errors {}",
                            fmt_opt(row.get("aggregate_tok_s").and_then(|v| v.as_f64())),
                            fmt_opt(row.get("mean_stream_decode_tok_s").and_then(|v| v.as_f64())),
                            errors.len()
                        );
                        runs.push(row);
                    }
                }
            }
            "needle" => {
                for &size in &c.needle_sizes {
                    for &depth in &c.depths {
                        let hay_str = prose_str_seed(
                            (size as f64 / TOK_PER_SIZE) as usize,
                            &format!("{}-n-{size}-{}", c.seed, c.salt),
                        );
                        let mut parts: Vec<&str> = hay_str.split(". ").collect();
                        let pos = ((parts.len() as f64) * depth) as usize;
                        let needle_line =
                            format!("Remember this: the secret passphrase is {SECRET}");
                        let mut joined: Vec<String> = parts.drain(..).map(String::from).collect();
                        joined.insert(pos.min(joined.len()), needle_line);
                        let prompt = format!(
                            "{}\n\nWhat is the secret passphrase mentioned in the text above? Reply with the passphrase only.",
                            joined.join(". ")
                        );
                        let msgs = json!([{"role": "user", "content": prompt}]);
                        let mut r = stream_chat(&c, msgs, 64, None, base_instant).await;
                        let correct = !r.contains_key("error")
                            && r.get("text")
                                .and_then(|v| v.as_str())
                                .map(|t| t.contains(SECRET))
                                .unwrap_or(false);
                        r.insert("suite".to_string(), json!("needle"));
                        r.insert("target_tokens".to_string(), json!(size));
                        r.insert("depth".to_string(), json!(depth));
                        r.insert("correct".to_string(), json!(correct));
                        show_row(&r);
                        runs.push(r);
                    }
                }
            }
            _ => {}
        }
    }

    for (name, on) in [("health", a.health), ("metrics", a.metrics)] {
        if on {
            snapshots.insert(format!("{name}_after"), get_snapshot(&c, name).await);
        }
    }
    for r in &mut runs {
        if r.get("suite").and_then(|v| v.as_str()) != Some("multi") {
            r.remove("t_first");
        }
    }

    let summary = summarize(&runs);
    println!("\nsummary (medians):");
    for (k, v) in &summary {
        let disp = match v.as_f64() {
            Some(f) if !v.is_i64() => format!("{:.2}", (f * 100.0).round() / 100.0),
            _ => v.to_string(),
        };
        println!("  {k:36} {disp}");
    }

    let wrong = runs.iter().any(|r| {
        r.get("suite").and_then(|v| v.as_str()) == Some("needle")
            && !r.get("correct").and_then(|v| v.as_bool()).unwrap_or(false)
    });
    let err_count = summary.get("errors").and_then(|v| v.as_u64()).unwrap_or(0);

    let doc = json!({
        "schema": SCHEMA,
        "label": a.label,
        "model": c.model,
        "url": c.url,
        "salt": c.salt,
        "seed": c.seed,
        "thinking": c.thinking,
        "started": started,
        "finished": now_iso8601(),
        "engine_tool": format!("dsv41-tools {}", env!("CARGO_PKG_VERSION")),
        "snapshots": Value::Object(snapshots),
        "runs": runs.into_iter().map(Value::Object).collect::<Vec<_>>(),
        "summary": Value::Object(summary),
    });

    if let Some(out) = a.out {
        if let Some(parent) = out.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Err(e) = fs::write(&out, serde_json::to_string_pretty(&doc).unwrap_or_default()) {
            eprintln!("failed to write {}: {e}", out.display());
            return 1;
        }
        println!("wrote {}", out.display());
    }
    if wrong || err_count > 0 {
        1
    } else {
        0
    }
}

async fn get_snapshot(c: &Ctx, name: &str) -> Value {
    match c
        .http
        .get(format!("{}/{name}", c.url))
        .timeout(Duration::from_secs(15))
        .send()
        .await
    {
        Ok(r) => match r.bytes().await {
            Ok(b) => {
                let raw = String::from_utf8_lossy(&b);
                serde_json::from_str(&raw)
                    .unwrap_or_else(|_| Value::String(raw.chars().take(20_000).collect()))
            }
            Err(e) => json!({"error": format!("ReadError: {e}")}),
        },
        Err(e) => json!({"error": format!("RequestError: {e}")}),
    }
}
