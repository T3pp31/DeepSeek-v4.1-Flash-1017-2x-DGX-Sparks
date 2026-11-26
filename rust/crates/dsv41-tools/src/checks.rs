//! Single-purpose verification checks replacing `needle.py`, `toolcheck.py`,
//! `anthropiccheck.py`, `end_of_turn.py`, `prompt_reuse.py`, and `prefill_yield_check.py`.

use clap::Args;
use dsv41_core::auth::resolve_standard;
use dsv41_core::client::Server;
use dsv41_core::prose::{prose_int_seed, words};
use dsv41_core::{sha256_hex, DEFAULT_MODEL};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

fn default_api_url() -> String {
    env::var("API_URL").unwrap_or_else(|_| {
        format!(
            "http://127.0.0.1:{}",
            env::var("PORT").unwrap_or_else(|_| "8888".to_string())
        )
    })
}

fn default_model() -> String {
    env::var("MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

// ------------------------------------------------------------------ needle
#[derive(Args)]
pub struct NeedleArgs {
    #[arg(default_value = "needle")]
    pub label: String,
    #[arg(default_value_t = 248_000)]
    pub size: usize,
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
}

pub async fn run_needle(a: NeedleArgs) -> i32 {
    const SECRET: &str = "violet-harbor-7291";
    let srv = Server::new(
        &a.url.unwrap_or_else(default_api_url),
        &a.model.unwrap_or_else(default_model),
        3600.0,
        1234,
        resolve_standard(None),
    );
    let hay_s = prose_int_seed(a.size, 777);
    let mut hay: Vec<String> = hay_s.split(". ").map(String::from).collect();
    let at = ((hay.len() as f64) * 0.6) as usize;
    hay.insert(
        at.min(hay.len()),
        format!("Remember this: the secret passphrase is {SECRET}"),
    );
    let prompt = format!(
        "{}\n\nWhat is the secret passphrase mentioned in the text above? Reply with the passphrase only.",
        hay.join(". ")
    );
    let body = json!({
        "model": srv.model,
        "max_tokens": 2048,
        "temperature": 0,
        "seed": 1234,
        "messages": [{"role": "user", "content": prompt}]
    });
    let t0 = Instant::now();
    let r = srv
        .post_json(
            "/v1/chat/completions",
            &body,
            Some(Duration::from_secs(3600)),
        )
        .await;
    if let Some(err) = r.get("error") {
        eprintln!("{}: error: {err}", a.label);
        return 1;
    }
    let content = r["choices"][0]["message"]["content"].as_str().unwrap_or("");
    let ok = content.contains(SECRET);
    let tail: String = content
        .trim()
        .chars()
        .rev()
        .take(80)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    println!(
        "{}: needle {} tok, prefill {} s, total {:.1} s, answer {:?}: {}",
        a.label,
        r["usage"]["prompt_tokens"],
        r["tensorfold"]["prefill_seconds"],
        t0.elapsed().as_secs_f64(),
        tail,
        if ok { "CORRECT" } else { "WRONG" }
    );
    if ok {
        0
    } else {
        1
    }
}

// ------------------------------------------------------------------ toolcheck
#[derive(Args)]
pub struct ToolcheckArgs {
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
}

pub async fn run_toolcheck(a: ToolcheckArgs) -> i32 {
    let srv = Server::new(
        &a.url.unwrap_or_else(default_api_url),
        &a.model.unwrap_or_else(default_model),
        600.0,
        1234,
        resolve_standard(None),
    );
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "add_tags",
            "description": "Attach tags to a document.",
            "parameters": {
                "type": "object",
                "required": ["doc_id", "tags"],
                "properties": {
                    "doc_id": {"type": "integer", "description": "The document id."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "The tags to attach."}
                }
            }
        }
    }]);
    let body = json!({
        "model": srv.model,
        "max_tokens": 1024,
        "temperature": 0,
        "tools": tools,
        "messages": [{"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool."}]
    });
    let r = srv
        .post_json(
            "/v1/chat/completions",
            &body,
            Some(Duration::from_secs(600)),
        )
        .await;
    if let Some(err) = r.get("error") {
        eprintln!("toolcheck: FAILED: {err}");
        return 1;
    }
    let msg = &r["choices"][0]["message"];
    let calls = msg
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut ok = false;
    for c in &calls {
        if let Ok(args) =
            serde_json::from_str::<Value>(c["function"]["arguments"].as_str().unwrap_or(""))
        {
            println!(
                "tool call {}({args})",
                c["function"]["name"].as_str().unwrap_or("")
            );
            if c["function"]["name"].as_str() == Some("add_tags")
                && args
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len() >= 3)
                    .unwrap_or(false)
                && args.get("doc_id").and_then(|v| v.as_i64()).is_some()
            {
                ok = true;
            }
        }
    }
    println!(
        "toolcheck: {}",
        if ok {
            "OK: tags is a JSON array".to_string()
        } else {
            format!("FAILED: {msg}")
        }
    );
    if ok {
        0
    } else {
        1
    }
}

// ------------------------------------------------------------------ anthropiccheck
#[derive(Args)]
pub struct AnthropicArgs {
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
}

fn anthropic_text(msg: &Value) -> String {
    msg.get("content")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default()
}

pub async fn run_anthropic(a: AnthropicArgs) -> i32 {
    let url = a
        .url
        .unwrap_or_else(default_api_url)
        .trim_end_matches('/')
        .to_string();
    let model = a.model.unwrap_or_else(default_model);
    let key = resolve_standard(None);
    let srv = Server::new(&url, &model, 600.0, 1234, key.clone());
    let tools = json!([{
        "name": "add_tags",
        "description": "Attach tags to a document.",
        "input_schema": {
            "type": "object",
            "required": ["doc_id", "tags"],
            "properties": {
                "doc_id": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}}
            }
        }
    }]);

    let mut fails = Vec::new();
    let mut check = |name: &str, ok: bool, detail: String| {
        println!("{name}: {} ({detail})", if ok { "ok" } else { "FAIL" });
        if !ok {
            fails.push(name.to_string());
        }
    };

    let post_a = |path: &str, body: Value| {
        let http = srv.client().clone();
        let full = format!("{url}{path}");
        async move {
            let resp = http
                .post(&full)
                .header("anthropic-version", "2023-06-01")
                .timeout(Duration::from_secs(600))
                .json(&body)
                .send()
                .await;
            match resp {
                Ok(r) => {
                    let st = r.status().as_u16();
                    let b = r.bytes().await.unwrap_or_default();
                    let v = serde_json::from_slice::<Value>(&b).unwrap_or_else(|_| json!({}));
                    Ok((st, v))
                }
                Err(e) => Err(e.to_string()),
            }
        }
    };

    // 1. message
    let text_body = json!({
        "model": model, "max_tokens": 64, "temperature": 0, "system": "You are terse.",
        "messages": [{"role": "user", "content": "Say exactly: hello from tensorfold"}]
    });
    let (st, msg) = match post_a("/v1/messages", text_body.clone()).await {
        Ok(x) => x,
        Err(e) => {
            eprintln!("anthropiccheck: {e}");
            return 1;
        }
    };
    let texts = anthropic_text(&msg);
    let in_tok = msg["usage"]["input_tokens"].as_i64().unwrap_or(0);
    let out_tok = msg["usage"]["output_tokens"].as_i64().unwrap_or(0);
    check(
        "message",
        st == 200
            && msg["type"].as_str() == Some("message")
            && texts.contains("hello from tensorfold")
            && in_tok > 0
            && out_tok > 0,
        format!("stop_reason {}, usage {}", msg["stop_reason"], msg["usage"]),
    );

    // 2. count_tokens
    if let Ok((_, count)) = post_a("/v1/messages/count_tokens", text_body).await {
        let c = count["input_tokens"].as_i64().unwrap_or(-1);
        check(
            "count_tokens",
            (c - in_tok).abs() <= 8,
            format!("count {c} vs usage {in_tok}"),
        );
    } else {
        check("count_tokens", false, "request failed".to_string());
    }

    // 3. streaming tool_use
    let stream_body = json!({
        "model": model, "max_tokens": 512, "temperature": 0, "tools": tools, "stream": true,
        "messages": [{"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool."}]
    });
    match srv
        .client()
        .post(format!("{url}/v1/messages"))
        .header("anthropic-version", "2023-06-01")
        .timeout(Duration::from_secs(600))
        .json(&stream_body)
        .send()
        .await
    {
        Err(e) => check("streaming tool_use", false, e.to_string()),
        Ok(resp) => {
            let mut events = Vec::new();
            let mut call_args: BTreeMap<u64, String> = BTreeMap::new();
            let mut tool_name = String::new();
            let mut stop_reason = String::new();
            let mut st = resp.bytes_stream();
            let mut buf = Vec::new();
            while let Some(Ok(chunk)) = st.next().await {
                buf.extend_from_slice(&chunk);
                while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                    let raw: Vec<u8> = buf.drain(..=nl).collect();
                    let line = String::from_utf8_lossy(&raw).trim().to_string();
                    let Some(d) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let Ok(ev) = serde_json::from_str::<Value>(d.trim()) else {
                        continue;
                    };
                    let ty = ev["type"].as_str().unwrap_or("").to_string();
                    events.push(ty.clone());
                    if ty == "content_block_start"
                        && ev["content_block"]["type"].as_str() == Some("tool_use")
                    {
                        tool_name = ev["content_block"]["name"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();
                        call_args.insert(ev["index"].as_u64().unwrap_or(0), String::new());
                    }
                    if ty == "content_block_delta"
                        && ev["delta"]["type"].as_str() == Some("input_json_delta")
                    {
                        let idx = ev["index"].as_u64().unwrap_or(0);
                        call_args
                            .entry(idx)
                            .or_default()
                            .push_str(ev["delta"]["partial_json"].as_str().unwrap_or(""));
                    }
                    if ty == "message_delta" {
                        if let Some(sr) = ev["delta"]["stop_reason"].as_str() {
                            stop_reason = sr.to_string();
                        }
                    }
                }
            }
            let first_json = call_args
                .values()
                .next()
                .map(String::as_str)
                .unwrap_or("{}");
            let call: Value = serde_json::from_str(first_json).unwrap_or_else(|_| json!({}));
            let ok = events.iter().any(|e| e == "message_start")
                && tool_name == "add_tags"
                && call
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len() >= 3)
                    .unwrap_or(false)
                && stop_reason == "tool_use"
                && events.iter().any(|e| e == "message_stop");
            check(
                "streaming tool_use",
                ok,
                format!(
                    "{} events, {tool_name}({call}), stop {stop_reason}",
                    events.len()
                ),
            );
        }
    }

    // 4. tool_result turn
    let loop_body = json!({
        "model": model, "max_tokens": 128, "temperature": 0,
        "messages": [
            {"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool."},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "add_tags", "input": {"doc_id": 42, "tags": ["urgent", "finance", "q3"]}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "saved tags: urgent, finance, q3"}]}
        ]
    });
    if let Ok((_, msg2)) = post_a("/v1/messages", loop_body).await {
        let t2 = anthropic_text(&msg2);
        check(
            "tool_result turn",
            msg2["type"].as_str() == Some("message") && !t2.trim().is_empty(),
            format!(
                "stop {}, reply {:?}",
                msg2["stop_reason"],
                t2.trim().chars().take(60).collect::<String>()
            ),
        );
    } else {
        check("tool_result turn", false, "request failed".to_string());
    }

    // 5. stop_sequences
    let stop_body = json!({
        "model": model, "max_tokens": 64, "temperature": 0, "stop_sequences": ["END"],
        "messages": [{"role": "user", "content": "Reply with exactly this and nothing else: alpha beta END gamma"}]
    });
    if let Ok((_, msg3)) = post_a("/v1/messages", stop_body).await {
        let t3 = anthropic_text(&msg3);
        let ok = msg3["stop_reason"].as_str() == Some("stop_sequence")
            && msg3["stop_sequence"].as_str() == Some("END")
            && !t3.contains("gamma");
        check(
            "stop_sequences",
            ok,
            format!(
                "stop {} {:?}, text {:?}",
                msg3["stop_reason"],
                msg3["stop_sequence"],
                t3.trim().chars().take(40).collect::<String>()
            ),
        );
    } else {
        check("stop_sequences", false, "request failed".to_string());
    }

    // 6. 400 error shape
    let bad =
        json!({"model": model, "messages": [{"role": "user", "content": "no max_tokens here"}]});
    if let Ok((code, err)) = post_a("/v1/messages", bad).await {
        let ok = code == 400
            && err["type"].as_str() == Some("error")
            && err["error"]["type"].as_str() == Some("invalid_request_error");
        check(
            "400 error shape",
            ok,
            format!("{code} {}", err["error"]["type"]),
        );
    } else {
        check("400 error shape", false, "request failed".to_string());
    }

    // 7. raw HTTP/1.1 Transfer-Encoding: chunked body
    let raw_body = serde_json::to_vec(&json!({
        "model": model, "max_tokens": 32, "temperature": 0,
        "messages": [{"role": "user", "content": "Say exactly: chunked ok"}]
    }))
    .unwrap();
    let parsed_url = url::Url::parse(&url).ok();
    let host = parsed_url
        .as_ref()
        .and_then(|u| u.host_str())
        .unwrap_or("127.0.0.1");
    let port = parsed_url.as_ref().and_then(|u| u.port()).unwrap_or(8888);
    let mut chunk_ok = false;
    let mut chunk_detail = String::new();
    if let Ok(mut tcp) = TcpStream::connect((host, port)).await {
        let auth_hdr = if !key.is_empty() {
            format!("Authorization: Bearer {key}\r\n")
        } else {
            String::new()
        };
        let req = format!(
            "POST /v1/messages HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nanthropic-version: 2023-06-01\r\n{auth_hdr}Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n",
            raw_body.len()
        );
        let _ = tcp.write_all(req.as_bytes()).await;
        let _ = tcp.write_all(&raw_body).await;
        let _ = tcp.write_all(b"\r\n0\r\n\r\n").await;
        let mut resp_bytes = Vec::new();
        let _ = tcp.read_to_end(&mut resp_bytes).await;
        let resp_str = String::from_utf8_lossy(&resp_bytes);
        if let Some((head, body_part)) = resp_str.split_once("\r\n\r\n") {
            let status_code = head
                .lines()
                .next()
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(0);
            let json_str = if head.to_lowercase().contains("transfer-encoding: chunked") {
                dechunk(body_part)
            } else {
                body_part.to_string()
            };
            let v: Value = serde_json::from_str(&json_str).unwrap_or_else(|_| json!({}));
            let t4 = anthropic_text(&v);
            chunk_ok = status_code == 200 && t4.contains("chunked ok");
            chunk_detail = format!(
                "{status_code} {:?}",
                t4.trim().chars().take(40).collect::<String>()
            );
        }
    }
    check("chunked body", chunk_ok, chunk_detail);

    println!(
        "anthropiccheck: {}",
        if fails.is_empty() {
            "all ok".to_string()
        } else {
            format!("FAILED {}", fails.join(", "))
        }
    );
    if fails.is_empty() {
        0
    } else {
        1
    }
}

fn dechunk(mut s: &str) -> String {
    let mut out = String::new();
    while let Some((sz_hex, rest)) = s.split_once("\r\n") {
        let Ok(sz) = usize::from_str_radix(sz_hex.trim(), 16) else {
            break;
        };
        if sz == 0 || rest.len() < sz {
            break;
        }
        out.push_str(&rest[..sz]);
        s = rest[sz..].strip_prefix("\r\n").unwrap_or("");
    }
    out
}

// ------------------------------------------------------------------ end-of-turn
#[derive(Args)]
pub struct EndOfTurnArgs {
    pub label: Option<String>,
    #[arg(default_value_t = 4)]
    pub max_cut: usize,
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
}

const EOT_TASKS: &[&str] = &[
    "Écris une fonction Python `fusionne_tries(listes)` qui fusionne une liste de listes triées en une seule liste triée, en O(N log k) avec heapq. Réponds uniquement avec le code.",
    "Écris une classe Python `LRU(capacite)` avec `get(cle)` (renvoie -1 si absente) et `put(cle, valeur)`, éviction du moins récemment utilisé, O(1). Réponds uniquement avec le code.",
    "Écris une fonction Python `vers_romain(n)` (1 ≤ n ≤ 3999) qui renvoie le nombre en chiffres romains. Réponds uniquement avec le code.",
    "Écris une fonction Python `equilibre(s)` qui renvoie True si les parenthèses (), [], {} de la chaîne sont correctement imbriquées. Réponds uniquement avec le code.",
    "Écris une fonction Python `tri_topologique(n, aretes)` qui renvoie un ordre topologique des sommets 0..n-1 (aretes = liste de (u,v) signifiant u avant v), ou None s'il y a un cycle. Réponds uniquement avec le code.",
    "Écris une fonction Python `spirale(m)` qui renvoie les éléments d'une matrice (liste de listes) parcourue en spirale dans le sens horaire depuis le coin haut-gauche. Réponds uniquement avec le code.",
    "Écris deux fonctions Python `rle_encode(s)` et `rle_decode(s)` : encodage par plages sous la forme '3a2b1c' pour 'aaabbc' (compte puis caractère, pas de chiffres dans l'entrée), et l'inverse. Réponds uniquement avec le code.",
    "Écris une fonction Python `duree_secondes(s)` qui convertit une durée ISO 8601 de la forme PnDTnHnMnS (chaque composante optionnelle, entiers) en secondes. Réponds uniquement avec le code.",
];

pub async fn run_end_of_turn(a: EndOfTurnArgs) -> i32 {
    let url = a.url.unwrap_or_else(default_api_url);
    let label = a.label.unwrap_or_else(|| url.clone());
    let srv = Server::new(
        &url,
        &a.model.unwrap_or_else(default_model),
        900.0,
        1234,
        resolve_standard(None),
    );

    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(4));
    let mut jobs = Vec::new();
    for &p in EOT_TASKS {
        for i in 0..5 {
            jobs.push((p, Some(1000 + i), 0.7f64));
        }
    }
    for &p in EOT_TASKS {
        jobs.push((p, None, 0.0f64));
    }
    let total = jobs.len();
    let mut handles = Vec::with_capacity(total);
    for (p, seed, temp) in jobs {
        let s = srv.clone();
        let permit_sem = sem.clone();
        handles.push(tokio::spawn(async move {
            let _permit = permit_sem.acquire().await.unwrap();
            let mut body = json!({
                "model": s.model,
                "messages": [{"role": "user", "content": p}],
                "max_tokens": 1024,
                "temperature": temp,
                "chat_template_kwargs": {"enable_thinking": false}
            });
            if let Some(sd) = seed {
                body.as_object_mut()
                    .unwrap()
                    .insert("seed".to_string(), json!(sd));
            }
            let r = s
                .post_json(
                    "/v1/chat/completions",
                    &body,
                    Some(Duration::from_secs(900)),
                )
                .await;
            r["choices"][0]["finish_reason"].as_str() == Some("length")
        }));
    }
    let mut cut_n = 0usize;
    for h in handles {
        if h.await.unwrap_or(false) {
            cut_n += 1;
        }
    }

    let mut ps = Vec::new();
    for &prompt in EOT_TASKS {
        let msgs = json!([{"role": "user", "content": prompt}]);
        let r = srv
            .post_json(
                "/v1/chat/completions",
                &json!({
                    "model": srv.model, "messages": msgs, "max_tokens": 1024,
                    "temperature": 0, "return_token_ids": true,
                    "chat_template_kwargs": {"enable_thinking": false}
                }),
                Some(Duration::from_secs(900)),
            )
            .await;
        let gen: Vec<i64> = r["tensorfold"]["token_ids"]
            .as_array()
            .or_else(|| r["choices"][0]["token_ids"].as_array())
            .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
            .unwrap_or_default();
        let mut text = String::new();
        let mut cut = None;
        for (k, &t) in gen.iter().enumerate() {
            let dr = srv
                .post_json(
                    "/detokenize",
                    &json!({"model": srv.model, "tokens": [t]}),
                    None,
                )
                .await;
            text.push_str(dr["prompt"].as_str().unwrap_or(""));
            if text.matches("```").count() >= 2 {
                cut = Some(k + 1);
                break;
            }
        }
        let Some(cut_idx) = cut else { continue };
        let tr = srv
            .post_json(
                "/tokenize",
                &json!({"model": srv.model, "messages": msgs, "chat_template_kwargs": {"enable_thinking": false}}),
                None,
            )
            .await;
        let mut ids: Vec<i64> = tr["tokens"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
            .unwrap_or_default();
        ids.extend_from_slice(&gen[..cut_idx]);
        let mut stops = 0usize;
        for seed in 0..40 {
            let c = srv
                .post_json(
                    "/v1/completions",
                    &json!({"model": srv.model, "prompt": ids, "max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "seed": seed}),
                    None,
                )
                .await;
            let ch = &c["choices"][0];
            if ch["finish_reason"].as_str() == Some("stop")
                && ch["text"].as_str().unwrap_or("").is_empty()
            {
                stops += 1;
            }
        }
        ps.push(stops as f64 / 40.0);
    }

    let mean_str = if !ps.is_empty() {
        let mean = ps.iter().sum::<f64>() / ps.len() as f64;
        let min = ps.iter().copied().fold(f64::INFINITY, f64::min);
        format!("mean {mean:.2}, min {min:.2} ({} tasks)", ps.len())
    } else {
        "no closed fence".to_string()
    };
    println!("{label}: replies cut by max_tokens {cut_n}/{total}; P(end of turn) after the closing fence: {mean_str}");
    if cut_n > a.max_cut {
        1
    } else {
        0
    }
}

// ------------------------------------------------------------------ prompt-reuse
#[derive(Args)]
pub struct PromptReuseArgs {
    #[arg(default_value_t = 40_000)]
    pub size: usize,
    #[arg(long = "require-counter")]
    pub require_counter: bool,
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
}

pub async fn run_prompt_reuse(a: PromptReuseArgs) -> i32 {
    let srv = Server::new(
        &a.url.unwrap_or_else(default_api_url),
        &a.model.unwrap_or_else(default_model),
        1800.0,
        1234,
        resolve_standard(None),
    );
    let system = json!({
        "role": "system",
        "content": format!("You are a careful coding agent. Read the material, then answer briefly. {}", prose_int_seed(2000, 1))
    });

    let ask_turn = |msgs: Vec<Value>| {
        let s = srv.clone();
        async move {
            let body = json!({
                "model": s.model,
                "messages": msgs,
                "max_tokens": 64,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": false}
            });
            let t0 = Instant::now();
            let r = s
                .post_json(
                    "/v1/chat/completions",
                    &body,
                    Some(Duration::from_secs(1800)),
                )
                .await;
            let secs = t0.elapsed().as_secs_f64();
            let content = r["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let pt = r["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
            let mut cached = r["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64();
            if cached.is_none() {
                if let Some(tf) = r.get("tensorfold").and_then(|v| v.as_object()) {
                    for (k, v) in tf {
                        if k.to_lowercase().contains("cached") {
                            if let Some(cv) = v.as_u64() {
                                cached = Some(cv);
                                break;
                            }
                        }
                    }
                }
            }
            (content, pt, cached, secs, r.get("error").cloned())
        }
    };

    let mut agent = vec![
        system.clone(),
        json!({"role": "user", "content": format!("Summarize this log in one line.\n{}", prose_int_seed(a.size, 2))}),
    ];
    let (mut reply_txt, pt, cached, secs, err) = ask_turn(agent.clone()).await;
    if let Some(e) = err {
        eprintln!("prompt_reuse: error: {e}");
        return 1;
    }
    println!("agent turn 0: {pt} prompt tokens, {cached:?} cached, {secs:.1} s (cold)");
    let mut worst = 1.0f64;
    let mut seen = cached.is_some();

    for turn in 1..=3 {
        let _ = ask_turn(vec![
            system.clone(),
            json!({"role": "user", "content": format!("Sub-task {turn}: name three uses of a hash map.")}),
        ])
        .await;
        agent.push(json!({"role": "assistant", "content": reply_txt}));
        agent.push(json!({"role": "user", "content": format!("Note {turn}: {}", prose_int_seed(200, 10 + turn))}));
        let (txt, pt, cached, secs, err) = ask_turn(agent.clone()).await;
        if let Some(e) = err {
            eprintln!("prompt_reuse: error: {e}");
            return 1;
        }
        reply_txt = txt;
        match cached {
            None => println!("agent turn {turn} (after another conversation): {pt} prompt tokens, no cached count, {secs:.1} s"),
            Some(c) => {
                seen = true;
                let ratio = if pt > 0 { c as f64 / pt as f64 } else { 0.0 };
                worst = worst.min(ratio);
                println!("agent turn {turn} (after another conversation): {pt} prompt tokens, {c} cached ({:.0}%), {secs:.1} s", ratio * 100.0);
            }
        }
    }

    if !seen {
        println!("prompt_reuse: NO COUNTER: the server reported no cached tokens");
        return if a.require_counter { 2 } else { 0 };
    }
    if worst >= 0.9 {
        println!("prompt_reuse: OK");
        0
    } else {
        println!(
            "prompt_reuse: FAILED: a turn resumed only {:.0}% of its prompt",
            worst * 100.0
        );
        1
    }
}

// ------------------------------------------------------------------ prefill-yield
#[derive(Args)]
pub struct PrefillYieldArgs {
    #[arg(long, default_value = "http://127.0.0.1:8888")]
    pub url: String,
    #[arg(long, default_value = DEFAULT_MODEL)]
    pub model: String,
    #[arg(long = "long-words", default_value_t = 64_000)]
    pub long_words: usize,
    #[arg(long, default_value_t = 8.0)]
    pub delay: f64,
    #[arg(long, default_value = "yieldtest")]
    pub salt: String,
    #[arg(long, default_value_t = 7)]
    pub seed: i64,
    #[arg(long, default_value_t = 900.0)]
    pub timeout: f64,
}

pub async fn run_prefill_yield(a: PrefillYieldArgs) -> i32 {
    let srv = Server::new(&a.url, &a.model, a.timeout, a.seed, resolve_standard(None));
    let long_text = format!(
        "[ref {}]\n{}\n\nSummarize the text above in one sentence.",
        a.salt,
        words(a.long_words, a.seed)
    );
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let short_text = format!("[ref short-{ts}] What is the capital of France? Answer in one word.");

    let ask_one = |s: Server, content: String, max_tokens: u64| async move {
        let body = s.body(
            &json!([{"role": "user", "content": content}]),
            max_tokens,
            false,
        );
        let t0 = Instant::now();
        let r = s.post_json("/v1/chat/completions", &body, None).await;
        if let Some(err) = r.get("error") {
            return json!({"error": err, "seconds": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0});
        }
        let text = r["choices"][0]["message"]["content"].as_str().unwrap_or("");
        let tsha = r["tensorfold"]["token_sha"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| sha256_hex(text.as_bytes()));
        let secs = r["_seconds"]
            .as_f64()
            .unwrap_or_else(|| t0.elapsed().as_secs_f64());
        json!({
            "seconds": (secs * 100.0).round() / 100.0,
            "prompt_tokens": r["usage"]["prompt_tokens"],
            "cached_tokens": r["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0),
            "finish": r["choices"][0]["finish_reason"],
            "token_sha": &tsha[..tsha.len().min(16)],
            "text": text.chars().take(60).collect::<String>(),
        })
    };

    let t0 = Instant::now();
    let long_h = tokio::spawn(ask_one(srv.clone(), long_text, 24));
    tokio::time::sleep(Duration::from_secs_f64(a.delay)).await;
    let sent_at = t0.elapsed().as_secs_f64();
    let mut short_res = ask_one(srv.clone(), short_text, 16).await;
    short_res.as_object_mut().unwrap().insert(
        "sent_after_s".to_string(),
        json!((sent_at * 10.0).round() / 10.0),
    );
    let mut long_res = long_h.await.unwrap();
    long_res.as_object_mut().unwrap().insert(
        "total_wall_s".to_string(),
        json!((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0),
    );

    let out = json!({"long": long_res, "short": short_res});
    println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    if out["long"].get("error").is_some() || out["short"].get("error").is_some() {
        return 1;
    }
    if out["long"]["cached_tokens"].as_u64().unwrap_or(0) > 0 {
        eprintln!(
            "NOTE: the long prompt was not cold (cached_tokens={})",
            out["long"]["cached_tokens"]
        );
    }
    if out["short"]["seconds"].as_f64().unwrap_or(0.0)
        > out["long"]["seconds"].as_f64().unwrap_or(0.0)
    {
        eprintln!("NOTE: the short request finished after the long one");
    }
    println!(
        "short request: {} s (sent {} s in); long: {} s, {} tokens, sha {}",
        out["short"]["seconds"],
        out["short"]["sent_after_s"],
        out["long"]["seconds"],
        out["long"]["prompt_tokens"],
        out["long"]["token_sha"]
    );
    0
}
