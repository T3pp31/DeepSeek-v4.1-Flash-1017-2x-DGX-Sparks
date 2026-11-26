//! Bit-exactness harness (`dsv41-tools exact record|compare`) replacing `tools/exact.py`.

use clap::{Args, Subcommand};
use dsv41_core::auth::resolve_standard;
use dsv41_core::client::Server;
use dsv41_core::prose::words;
use dsv41_core::{canonical_json_py, now_iso8601, random_hex, sha256_hex, DEFAULT_MODEL};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;

const SCHEMA: u64 = 1;
const PROMPT_SET: u64 = 2;
const PREDICT_FROM: usize = 30_000;
const CONV_SYS_TARGET: usize = 6_000;
const RESUME_TARGET: usize = 8_000;
const OLD_STREAMS: usize = 4;

const COLD_SIZES: &[(&str, usize)] = &[
    ("len_1", 1),
    ("len_100", 100),
    ("len_2048", 2048),
    ("len_2049", 2049),
    ("len_4096", 4096),
    ("len_8000", 8000),
    ("len_20000", 20000),
    ("len_30000", 30000),
    ("len_50000", 50000),
    ("len_64000", 64000),
    ("len_100000", 100000),
];
const CONC_TARGETS: &[usize] = &[600, 1200, 1800, 2400, 3000, 3600, 4200, 4800];
const CONV_ORDER: &[&str] = &["A1", "B1", "C1", "A2", "B2", "C2", "A3"];
const SUFFIXES: &[&str] = &[
    "Summarize the text above in one sentence.",
    "List the three most frequent words in the text above.",
    "Write one question a reader could ask about the text above.",
];

fn conv_question(k: char, turn: usize) -> &'static str {
    match (k, turn) {
        ('A', 0) => "What is the first topic in the text above?",
        ('A', 1) => "Name two words that repeat in the text.",
        ('A', _) => "Say one short sentence about the weather.",
        ('B', 0) => "Summarize the text above in five words.",
        ('B', 1) => "Which word comes last in the text?",
        ('B', _) => "Reply with a number between one and nine.",
        (_, 0) => "Give a title for the text above.",
        (_, 1) => "List three nouns from the text.",
        (_, _) => "Say hello in a different way.",
    }
}

#[derive(Args)]
pub struct ExactArgs {
    #[command(subcommand)]
    pub mode: ExactMode,
}

#[derive(Subcommand)]
pub enum ExactMode {
    Record(RecordArgs),
    Compare(CompareArgs),
}

#[derive(Args)]
pub struct RecordArgs {
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value = "")]
    pub label: String,
    #[arg(long, default_value = "auto", value_parser = ["auto", "on", "off"])]
    pub drafts: String,
    #[arg(long)]
    pub like: Option<PathBuf>,
    #[arg(long, default_value_t = 1234)]
    pub seed: i64,
    #[arg(long = "max-tokens", default_value_t = 192)]
    pub max_tokens: u64,
    #[arg(long, default_value_t = 900.0)]
    pub timeout: f64,
    #[arg(long = "no-calibrate")]
    pub no_calibrate: bool,
    #[arg(long, default_value_t = 8)]
    pub streams: usize,
    #[arg(long = "keep-text")]
    pub keep_text: bool,
}

#[derive(Args)]
pub struct CompareArgs {
    pub a: PathBuf,
    pub b: PathBuf,
    #[arg(long = "allow-weak")]
    pub allow_weak: bool,
}

fn prompt_text(salt: &str, n: usize, seed: i64, suffix: &str) -> String {
    if suffix.is_empty() {
        format!("[ref {salt}]\n{}", words(n, seed))
    } else {
        format!("[ref {salt}]\n{}\n\n{suffix}", words(n, seed))
    }
}

async fn ask(
    srv: &Server,
    messages: Value,
    max_tokens: u64,
    keep_text: bool,
) -> (Map<String, Value>, String) {
    let body = srv.body(&messages, max_tokens, false);
    let r = srv.post_json("/v1/chat/completions", &body, None).await;
    let mut out = Map::new();
    out.insert(
        "prompt_sha".to_string(),
        Value::String(sha256_hex(canonical_json_py(&messages).as_bytes())),
    );
    if let Some(err) = r.get("error") {
        out.insert("error".to_string(), err.clone());
        return (out, String::new());
    }
    let Some(ch) = r
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
    else {
        out.insert("error".to_string(), json!("unexpected reply shape"));
        return (out, String::new());
    };
    let text = ch
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let u = r.get("usage").and_then(|v| v.as_object());
    let tf = r.get("tensorfold").and_then(|v| v.as_object());
    let tsha = tf
        .and_then(|t| t.get("token_sha"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    let text_sha = sha256_hex(text.as_bytes());
    let cached = u
        .and_then(|u| u.get("prompt_tokens_details"))
        .and_then(|d| d.get("cached_tokens"))
        .cloned()
        .or_else(|| tf.and_then(|t| t.get("cached_tokens")).cloned())
        .unwrap_or(Value::Null);
    let secs = r.get("_seconds").and_then(|v| v.as_f64()).unwrap_or(0.0);

    out.insert(
        "prompt_tokens".to_string(),
        u.and_then(|u| u.get("prompt_tokens"))
            .cloned()
            .unwrap_or(Value::Null),
    );
    out.insert(
        "completion_tokens".to_string(),
        u.and_then(|u| u.get("completion_tokens"))
            .cloned()
            .unwrap_or(Value::Null),
    );
    out.insert(
        "finish_reason".to_string(),
        ch.get("finish_reason").cloned().unwrap_or(Value::Null),
    );
    out.insert(
        "token_sha".to_string(),
        Value::String(tsha.map(String::from).unwrap_or_else(|| text_sha.clone())),
    );
    out.insert(
        "token_sha_source".to_string(),
        Value::String(if tsha.is_some() { "server" } else { "text" }.to_string()),
    );
    out.insert("text_sha".to_string(), Value::String(text_sha));
    out.insert("text_len".to_string(), json!(text.chars().count()));
    out.insert("cached_tokens".to_string(), cached);
    out.insert(
        "drafts".to_string(),
        tf.and_then(|t| t.get("drafts"))
            .cloned()
            .unwrap_or(Value::Null),
    );
    out.insert(
        "sampling".to_string(),
        tf.and_then(|t| t.get("sampling"))
            .cloned()
            .unwrap_or(Value::Null),
    );
    out.insert(
        "seconds".to_string(),
        json!((secs * 1000.0).round() / 1000.0),
    );
    if keep_text {
        out.insert("text".to_string(), Value::String(text.clone()));
    }
    (out, text)
}

async fn calibrate(
    srv: &Server,
    seed: i64,
    target: usize,
    tag: &str,
    tol: usize,
) -> (usize, Option<usize>) {
    if target <= 1 {
        return (0, Some(1));
    }
    let mut n = ((target as f64 / 1.2) as usize).max(1);
    let mut best: (usize, usize, Option<usize>) = (usize::MAX, n, None);
    for _ in 0..8 {
        let content = prompt_text(
            &format!("cal-{}-{tag}", random_hex(4)),
            n,
            seed,
            SUFFIXES[0],
        );
        let msgs = json!([{"role": "user", "content": content}]);
        let (r, _) = ask(srv, msgs, 1, false).await;
        let Some(pt) = r
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
        else {
            break;
        };
        let diff = pt as isize - target as isize;
        let adiff = diff.unsigned_abs();
        if adiff < best.0 {
            best = (adiff, n, Some(pt));
        }
        if adiff <= tol {
            return (n, Some(pt));
        }
        let step = ((adiff as f64 / 1.15).round() as usize).max(1);
        n = if diff > 0 {
            n.saturating_sub(step).max(1)
        } else {
            n.saturating_add(step)
        };
    }
    eprintln!(
        "calibrate {tag}: target {target} not hit exactly; using {} words (off by {})",
        best.1, best.0
    );
    (best.1, best.2)
}

fn log_case(name: &str, r: &Map<String, Value>) {
    if let Some(err) = r.get("error").and_then(|v| v.as_str()) {
        eprintln!("  {name}: {err}");
    } else {
        let sha = r.get("token_sha").and_then(|v| v.as_str()).unwrap_or("");
        let src = r
            .get("token_sha_source")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        eprintln!(
            "  {name}: prompt={} out={} {} cached={} sha={}({src}) {}s",
            r.get("prompt_tokens").unwrap_or(&Value::Null),
            r.get("completion_tokens").unwrap_or(&Value::Null),
            r.get("finish_reason")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            r.get("cached_tokens").unwrap_or(&Value::Null),
            &sha[..sha.len().min(12)],
            r.get("seconds").unwrap_or(&Value::Null),
        );
    }
}

async fn run_record(a: RecordArgs) -> i32 {
    if a.streams != 4 && a.streams != 8 {
        eprintln!("--streams must be 4 or 8");
        return 2;
    }
    let like: Option<Value> = match &a.like {
        Some(p) => {
            match fs::read_to_string(p).and_then(|s| serde_json::from_str(&s).map_err(Into::into)) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("cannot read --like {}: {e}", p.display());
                    return 2;
                }
            }
        }
        None => None,
    };
    if let Some(lk) = &like {
        let ps = lk
            .get("meta")
            .and_then(|m| m.get("prompt_set"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1);
        if ps != PROMPT_SET {
            eprintln!("--like is prompt set {ps}, this tool is set {PROMPT_SET}");
            return 2;
        }
    }

    let url = a.url.unwrap_or_else(|| {
        env::var("API_URL").unwrap_or_else(|_| {
            format!(
                "http://127.0.0.1:{}",
                env::var("PORT").unwrap_or_else(|_| "8888".to_string())
            )
        })
    });
    let model = a
        .model
        .or_else(|| env::var("MODEL").ok())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let key = resolve_standard(None);
    let srv = Server::new(&url, &model, a.timeout, a.seed, key);

    let salt = like
        .as_ref()
        .and_then(|l| l["meta"]["salt"].as_str())
        .map(String::from)
        .unwrap_or_else(|| random_hex(6));

    let mut shape: BTreeMap<String, usize> = BTreeMap::new();
    if let Some(sm) = like.as_ref().and_then(|l| l["meta"]["shape"].as_object()) {
        for (k, v) in sm {
            if let Some(n) = v.as_u64() {
                shape.insert(k.clone(), n as usize);
            }
        }
    }

    let health = srv.health().await;
    eprintln!(
        "record: label={:?} url={} salt={salt} like={:?}",
        a.label, srv.url, a.like
    );

    let mut todo: Vec<(String, usize, usize)> = COLD_SIZES
        .iter()
        .map(|&(n, t)| {
            (
                n.to_string(),
                t,
                if t <= 2049 { 0 } else { (t / 100).max(2) },
            )
        })
        .collect();
    todo.push(("resume".to_string(), RESUME_TARGET, 80));
    for (i, &t) in CONC_TARGETS[..a.streams].iter().enumerate() {
        todo.push((format!("conc_{}", i + 1), t, 20));
    }
    todo.push(("convsys".to_string(), CONV_SYS_TARGET, 60));

    let mut cal: BTreeMap<String, (usize, Option<usize>)> = BTreeMap::new();
    for (name, tgt, tol) in todo {
        if shape.contains_key(&name) {
            continue;
        }
        if let Some(lk) = &like {
            let old_s = lk["meta"]["conc_streams"]
                .as_u64()
                .unwrap_or(OLD_STREAMS as u64) as usize;
            if let Some(num_s) = name.strip_prefix("conc_") {
                if let Ok(k) = num_s.parse::<usize>() {
                    if k > old_s {
                        let mut sum_w = 0.0;
                        let mut sum_t = 0.0;
                        for (i, &conc_t) in CONC_TARGETS.iter().enumerate().take(OLD_STREAMS) {
                            if let Some(&w) = shape.get(&format!("conc_{}", i + 1)) {
                                sum_w += w as f64;
                                sum_t += conc_t as f64;
                            }
                        }
                        let ratio = if sum_t > 0.0 {
                            sum_w / sum_t
                        } else {
                            1.0 / 1.15
                        };
                        let derived = ((tgt as f64 * ratio) as usize).max(1);
                        shape.insert(name.clone(), derived);
                        eprintln!("  shape {name}: {derived} words (derived from --like)");
                        continue;
                    }
                }
            }
            eprintln!("--like file has no shape for {name}");
            return 2;
        }
        if a.no_calibrate {
            let w = if tgt > 1 {
                (tgt as f64 / 1.15) as usize
            } else {
                0
            };
            shape.insert(name.clone(), w);
        } else if tgt >= PREDICT_FROM
            && cal.contains_key("len_8000")
            && cal.contains_key("len_20000")
        {
            let (n1, p1) = cal["len_8000"];
            let (n2, p2) = cal["len_20000"];
            if let (Some(p1), Some(p2)) = (p1, p2) {
                let b = (p2 as f64 - p1 as f64) / (n2 as f64 - n1 as f64).max(1.0);
                let est = ((tgt as f64 - (p1 as f64 - b * n1 as f64)) / b) as usize;
                shape.insert(name.clone(), est.max(1));
            } else {
                shape.insert(name.clone(), ((tgt as f64 / 1.15) as usize).max(1));
            }
        } else {
            let use_tol = if name == "len_100" { 1 } else { tol };
            let (w, pt) = calibrate(&srv, a.seed, tgt, &name, use_tol).await;
            shape.insert(name.clone(), w);
            cal.insert(name.clone(), (w, pt));
        }
        eprintln!(
            "  shape {name}: {} words (target {tgt} tokens)",
            shape[&name]
        );
    }

    let mut cases: BTreeMap<String, Map<String, Value>> = BTreeMap::new();
    let mut case_order: Vec<String> = Vec::new();
    let mut record_one = |name: String, target: usize, mut r: Map<String, Value>| {
        r.insert("target_tokens".to_string(), json!(target));
        log_case(&name, &r);
        case_order.push(name.clone());
        cases.insert(name, r);
    };

    for &(name, tgt) in COLD_SIZES {
        let n = shape[name];
        let content = if tgt == 1 {
            "Hi".to_string()
        } else {
            prompt_text(&format!("{salt}-{name}"), n, a.seed, SUFFIXES[0])
        };
        let (r, _) = ask(
            &srv,
            json!([{"role": "user", "content": content}]),
            a.max_tokens,
            a.keep_text,
        )
        .await;
        record_one(name.to_string(), tgt, r);
    }

    let rep_content = prompt_text(
        &format!("{salt}-len_2048"),
        shape["len_2048"],
        a.seed,
        SUFFIXES[0],
    );
    let (r, _) = ask(
        &srv,
        json!([{"role": "user", "content": rep_content}]),
        a.max_tokens,
        a.keep_text,
    )
    .await;
    record_one("repeat_2048".to_string(), 2048, r);

    let rn = shape["resume"];
    for (i, sfx) in SUFFIXES.iter().enumerate() {
        let c = prompt_text(&format!("{salt}-resume"), rn, a.seed, sfx);
        let (r, _) = ask(
            &srv,
            json!([{"role": "user", "content": c}]),
            a.max_tokens,
            a.keep_text,
        )
        .await;
        record_one(format!("resume_{}", i + 1), RESUME_TARGET, r);
    }
    let c = prompt_text(&format!("{salt}-resume"), rn, a.seed, SUFFIXES[0]);
    let (r, _) = ask(
        &srv,
        json!([{"role": "user", "content": c}]),
        a.max_tokens,
        a.keep_text,
    )
    .await;
    record_one("resume_1_again".to_string(), RESUME_TARGET, r);

    // Concurrent batch
    let conc_prompts: Vec<(String, usize, String)> = (0..a.streams)
        .map(|i| {
            let nm = format!("conc_{}", i + 1);
            let content = prompt_text(
                &format!("{salt}-conc{}", i + 1),
                shape[&nm],
                a.seed,
                SUFFIXES[0],
            );
            (nm, CONC_TARGETS[i], content)
        })
        .collect();
    let mut handles = Vec::with_capacity(a.streams);
    for (nm, tgt, content) in &conc_prompts {
        let s = srv.clone();
        let c = content.clone();
        let mt = a.max_tokens;
        let kt = a.keep_text;
        let nm_c = nm.clone();
        let tgt_c = *tgt;
        handles.push(tokio::spawn(async move {
            let (r, _) = ask(&s, json!([{"role": "user", "content": c}]), mt, kt).await;
            (nm_c, tgt_c, r)
        }));
    }
    for h in handles {
        let (nm, tgt, r) = h.await.unwrap();
        record_one(nm, tgt, r);
    }
    for (i, (_, tgt, content)) in conc_prompts.iter().enumerate() {
        let (r, _) = ask(
            &srv,
            json!([{"role": "user", "content": content}]),
            a.max_tokens,
            a.keep_text,
        )
        .await;
        record_one(format!("solo_{}", i + 1), *tgt, r);
    }

    // Interleaved 3 conversations
    let mut convs: BTreeMap<char, Vec<Value>> = BTreeMap::new();
    for k in ['A', 'B', 'C'] {
        let sys = format!(
            "[ref {salt}-conv{k}]\n{}",
            words(shape["convsys"], a.seed + (k as i64))
        );
        convs.insert(k, vec![json!({"role": "system", "content": sys})]);
    }
    let mut dead: BTreeSet<char> = BTreeSet::new();
    for &turn in CONV_ORDER {
        let k = turn.chars().next().unwrap();
        let t: usize = turn[1..].parse::<usize>().unwrap() - 1;
        if dead.contains(&k) {
            continue;
        }
        let hist = convs.get_mut(&k).unwrap();
        hist.push(json!({"role": "user", "content": conv_question(k, t)}));
        let (r, text) = ask(&srv, Value::Array(hist.clone()), a.max_tokens, a.keep_text).await;
        if r.contains_key("error") {
            dead.insert(k);
        } else {
            hist.push(json!({"role": "assistant", "content": text}));
        }
        record_one(format!("conv_{turn}"), CONV_SYS_TARGET, r);
    }

    // Intra-run checks
    let mut checks: Vec<Value> = Vec::new();
    let same_case = |x: &str, y: &str| -> bool {
        match (cases.get(x), cases.get(y)) {
            (Some(a), Some(b)) => {
                !a.contains_key("error")
                    && !b.contains_key("error")
                    && a.get("token_sha") == b.get("token_sha")
            }
            _ => false,
        }
    };
    let add_chk = |checks: &mut Vec<Value>, name: String, ok: bool, detail: String| {
        eprintln!(
            "  check {name}: {} {detail}",
            if ok { "ok" } else { "FAIL" }
        );
        checks.push(json!({"check": name, "ok": ok, "detail": detail}));
    };
    add_chk(
        &mut checks,
        "repeat_2048 == len_2048".to_string(),
        same_case("repeat_2048", "len_2048"),
        String::new(),
    );
    add_chk(
        &mut checks,
        "resume_1_again == resume_1".to_string(),
        same_case("resume_1_again", "resume_1"),
        String::new(),
    );
    for i in 1..=a.streams {
        add_chk(
            &mut checks,
            format!("solo_{i} == conc_{i}"),
            same_case(&format!("solo_{i}"), &format!("conc_{i}")),
            String::new(),
        );
    }
    for nm in ["resume_2", "resume_3"] {
        match cases
            .get(nm)
            .and_then(|c| c.get("cached_tokens"))
            .and_then(|v| v.as_u64())
        {
            None => checks.push(json!({
                "check": format!("{nm} cache hit"),
                "ok": true,
                "detail": "server reports no cached_tokens (not checked)"
            })),
            Some(c) => add_chk(
                &mut checks,
                format!("{nm} cache hit (cached_tokens>0)"),
                c > 0,
                format!("cached={c}"),
            ),
        }
    }

    let drafts_seen: BTreeSet<String> = cases
        .values()
        .filter(|c| !c.contains_key("error"))
        .map(|c| {
            c.get("drafts")
                .map(|v| v.to_string())
                .unwrap_or_else(|| "None".to_string())
        })
        .collect();
    let weak: Vec<String> = cases
        .iter()
        .filter(|(_, c)| {
            !c.contains_key("error")
                && c.get("token_sha_source").and_then(|v| v.as_str()) != Some("server")
        })
        .map(|(n, _)| n.clone())
        .collect();
    if !weak.is_empty() {
        eprintln!(
            "WARNING: weak record ({} cases without server token_sha)",
            weak.len()
        );
    }

    let mut health_sub = Map::new();
    for k in ["status", "model", "max_batch_size", "error"] {
        if let Some(v) = health.get(k) {
            health_sub.insert(k.to_string(), v.clone());
        }
    }
    let meta = json!({
        "schema": SCHEMA,
        "prompt_set": PROMPT_SET,
        "conc_streams": a.streams,
        "weak": !weak.is_empty(),
        "label": a.label,
        "drafts_label": a.drafts,
        "drafts_observed": drafts_seen.into_iter().collect::<Vec<_>>(),
        "url": srv.url,
        "model": srv.model,
        "salt": salt,
        "seed": a.seed,
        "max_tokens": a.max_tokens,
        "shape": shape,
        "like": a.like.as_ref().map(|p| p.display().to_string()),
        "time": now_iso8601(),
        "health": Value::Object(health_sub),
    });

    let errors: Vec<&String> = cases
        .iter()
        .filter(|(_, c)| c.contains_key("error"))
        .map(|(n, _)| n)
        .collect();
    let bad_checks: Vec<String> = checks
        .iter()
        .filter(|c| !c["ok"].as_bool().unwrap_or(false))
        .filter_map(|c| c["check"].as_str().map(String::from))
        .collect();

    let cases_obj: Map<String, Value> = cases
        .iter()
        .map(|(k, v)| (k.clone(), Value::Object(v.clone())))
        .collect();
    let doc = json!({
        "meta": meta,
        "cases": Value::Object(cases_obj),
        "checks": checks,
    });
    if let Some(parent) = a.out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Err(e) = fs::write(
        &a.out,
        serde_json::to_string_pretty(&doc).unwrap_or_default(),
    ) {
        eprintln!("failed to write {}: {e}", a.out.display());
        return 1;
    }
    eprintln!(
        "wrote {}: {} cases, {} errors, {} failed checks",
        a.out.display(),
        cases.len(),
        errors.len(),
        bad_checks.len()
    );
    if !errors.is_empty() || !bad_checks.is_empty() {
        1
    } else {
        0
    }
}

fn run_compare(a: CompareArgs) -> i32 {
    let doc_a: Value =
        match fs::read_to_string(&a.a).and_then(|s| serde_json::from_str(&s).map_err(Into::into)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("cannot read {}: {e}", a.a.display());
                return 1;
            }
        };
    let doc_b: Value =
        match fs::read_to_string(&a.b).and_then(|s| serde_json::from_str(&s).map_err(Into::into)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("cannot read {}: {e}", a.b.display());
                return 1;
            }
        };
    let mut bad = 0usize;
    let ma = &doc_a["meta"];
    let mb = &doc_b["meta"];
    println!("A: {} label={}", a.a.display(), ma["label"]);
    println!("B: {} label={}", a.b.display(), mb["label"]);
    for k in ["prompt_set", "salt", "seed", "max_tokens", "model"] {
        if ma.get(k) != mb.get(k) {
            println!("MISMATCH meta.{k}: {} vs {}", ma[k], mb[k]);
            bad += 1;
        }
    }
    for (who, d) in [("A", &doc_a), ("B", &doc_b)] {
        let is_weak = d["meta"]["weak"].as_bool().unwrap_or(false)
            || d["cases"]
                .as_object()
                .map(|m| {
                    m.values().any(|c| {
                        c.get("error").is_none()
                            && c.get("token_sha_source").and_then(|v| v.as_str()) == Some("text")
                    })
                })
                .unwrap_or(false);
        if is_weak {
            if a.allow_weak {
                println!("note: {who} is WEAK (--allow-weak)");
            } else {
                println!(
                    "MISMATCH {who} is a WEAK record (no tensorfold.token_sha); use --allow-weak"
                );
                bad += 1;
            }
        }
        if let Some(chks) = d.get("checks").and_then(|v| v.as_array()) {
            for c in chks {
                if !c["ok"].as_bool().unwrap_or(false) {
                    println!(
                        "MISMATCH {who} intra-run check failed: {} {}",
                        c["check"], c["detail"]
                    );
                    bad += 1;
                }
            }
        }
    }

    let ca = doc_a["cases"].as_object();
    let cb = doc_b["cases"].as_object();
    let mut names = BTreeSet::new();
    if let Some(m) = ca {
        names.extend(m.keys().cloned());
    }
    if let Some(m) = cb {
        names.extend(m.keys().cloned());
    }
    for n in &names {
        let ra = ca.and_then(|m| m.get(n));
        let rb = cb.and_then(|m| m.get(n));
        match (ra, rb) {
            (None, Some(_)) | (Some(_), None) => {
                let lacking = if ra.is_none() { ma } else { mb };
                let lim = lacking["conc_streams"]
                    .as_u64()
                    .unwrap_or(OLD_STREAMS as u64) as usize;
                let idx = n
                    .strip_prefix("conc_")
                    .or_else(|| n.strip_prefix("solo_"))
                    .and_then(|s| s.parse::<usize>().ok());
                if let Some(k) = idx {
                    if k > lim {
                        println!("note     {n}: outside conc_streams={lim}");
                        continue;
                    }
                }
                println!(
                    "MISMATCH {n}: present only in {}",
                    if ra.is_none() { "B" } else { "A" }
                );
                bad += 1;
            }
            (Some(va), Some(vb)) => {
                if va.get("error").is_some() || vb.get("error").is_some() {
                    println!(
                        "MISMATCH {n}: request error A={} B={}",
                        va["error"], vb["error"]
                    );
                    bad += 1;
                    continue;
                }
                let mut diffs = Vec::new();
                for k in [
                    "prompt_sha",
                    "prompt_tokens",
                    "completion_tokens",
                    "finish_reason",
                    "text_sha",
                ] {
                    if va.get(k) != vb.get(k) {
                        diffs.push(format!("{k}: {} vs {}", va[k], vb[k]));
                    }
                }
                if va.get("token_sha_source") == vb.get("token_sha_source")
                    && va.get("token_sha") != vb.get("token_sha")
                {
                    diffs.push(format!(
                        "token_sha: {} vs {}",
                        va["token_sha"], vb["token_sha"]
                    ));
                }
                if !diffs.is_empty() {
                    bad += 1;
                    println!("MISMATCH {n}: {}", diffs.join("; "));
                } else {
                    let sha = va["token_sha"].as_str().unwrap_or("");
                    println!(
                        "ok       {n}: prompt={} out={} sha={} cached(A/B)={}/{}",
                        va["prompt_tokens"],
                        va["completion_tokens"],
                        &sha[..sha.len().min(12)],
                        va["cached_tokens"],
                        vb["cached_tokens"]
                    );
                }
            }
            (None, None) => {}
        }
    }
    println!(
        "RESULT: {} ({bad} problem(s), {} cases)",
        if bad == 0 { "IDENTICAL" } else { "DIFFERENT" },
        names.len()
    );
    if bad > 0 {
        1
    } else {
        0
    }
}

pub async fn run(a: ExactArgs) -> i32 {
    match a.mode {
        ExactMode::Record(r) => run_record(r).await,
        ExactMode::Compare(c) => run_compare(c),
    }
}
