//! End-to-end validation (`validate`) and correctness smoke (`smoke`) replacing
//! `tools/validate.py` and `tools/smoke.py`.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use clap::Args;
use dsv41_core::auth::resolve_standard;
use dsv41_core::client::Server;
use dsv41_core::png::encode_rgb_png;
use dsv41_core::DEFAULT_MODEL;
use futures_util::StreamExt;
use regex::Regex;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Args)]
pub struct ValidateArgs {
    #[arg(default_value = "http://127.0.0.1:8888")]
    pub url: String,
    #[arg(long, default_value = DEFAULT_MODEL)]
    pub model: String,
    #[arg(long)]
    pub save: Option<PathBuf>,
    #[arg(long)]
    pub compare: Option<PathBuf>,
}

#[derive(Args)]
pub struct SmokeArgs {
    #[arg(long)]
    pub url: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long)]
    pub json: Option<PathBuf>,
}

fn assistant_message(r: &Value) -> Option<&serde_json::Map<String, Value>> {
    let obj = r.as_object()?;
    if obj.contains_key("error") {
        return None;
    }
    let ch = obj.get("choices")?.as_array()?.first()?.as_object()?;
    let m = ch.get("message")?.as_object()?;
    if m.get("role")
        .and_then(|v| v.as_str())
        .unwrap_or("assistant")
        != "assistant"
    {
        return None;
    }
    Some(m)
}

fn msg_text(r: &Value) -> String {
    assistant_message(r)
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn well_formed_call(c: &Value) -> bool {
    let Some(obj) = c.as_object() else {
        return false;
    };
    let id_ok = obj
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let ty_ok = obj.get("type").and_then(|v| v.as_str()) == Some("function");
    if !id_ok || !ty_ok {
        return false;
    }
    let Some(f) = obj.get("function").and_then(|v| v.as_object()) else {
        return false;
    };
    let name_ok = f
        .get("name")
        .and_then(|v| v.as_str())
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let Some(args_s) = f.get("arguments").and_then(|v| v.as_str()) else {
        return false;
    };
    if !name_ok {
        return false;
    }
    // serde_json strictly rejects NaN / Infinity (RFC 8259)
    serde_json::from_str::<Value>(args_s)
        .map(|v| v.is_object())
        .unwrap_or(false)
}

fn tool_calls_ok(r: &Value, name: &str, city: &str) -> (bool, Vec<Value>) {
    let Some(calls) = assistant_message(r)
        .and_then(|m| m.get("tool_calls"))
        .and_then(|v| v.as_array())
    else {
        return (false, Vec::new());
    };
    if calls.is_empty() || !calls.iter().all(well_formed_call) {
        return (false, Vec::new());
    }
    let mut cities = Vec::new();
    for c in calls {
        if c["function"]["name"].as_str() != Some(name) {
            return (false, Vec::new());
        }
        let Ok(args) =
            serde_json::from_str::<Value>(c["function"]["arguments"].as_str().unwrap_or(""))
        else {
            return (false, Vec::new());
        };
        let Some(g) = args
            .get("city")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        else {
            return (false, Vec::new());
        };
        cities.push(g.to_lowercase());
    }
    (cities[0].contains(city), calls.clone())
}

pub async fn run_validate(a: ValidateArgs) -> i32 {
    let key = resolve_standard(None);
    let srv = Server::new(&a.url, &a.model, 900.0, 1234, key);
    let mut failed: Vec<String> = Vec::new();
    let mut check = |name: &str, ok: bool, detail: String| {
        if detail.is_empty() {
            println!("{}  {name}", if ok { "PASS" } else { "FAIL" });
        } else {
            println!("{}  {name}  {detail}", if ok { "PASS" } else { "FAIL" });
        }
        if !ok {
            failed.push(name.to_string());
        }
    };

    let chat = |srv: Server, msgs: Value, max_tokens: u64, thinking: bool, tools: Option<Value>| async move {
        let mut body = json!({
            "model": srv.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": thinking}
        });
        if let Some(t) = tools {
            body.as_object_mut().unwrap().insert("tools".to_string(), t);
        }
        srv.post_json(
            "/v1/chat/completions",
            &body,
            Some(Duration::from_secs(900)),
        )
        .await
    };

    // 1. text
    let t0 = Instant::now();
    let r = chat(
        srv.clone(),
        json!([{"role": "user", "content": "What is the capital of France? One word."}]),
        20,
        false,
        None,
    )
    .await;
    let tx = msg_text(&r);
    check(
        "text",
        tx.to_lowercase().contains("paris"),
        format!(
            "{:?} {:.1}s",
            tx.chars().take(60).collect::<String>(),
            t0.elapsed().as_secs_f64()
        ),
    );

    // 2. thinking
    let r = chat(
        srv.clone(),
        json!([{"role": "user", "content": "What is 17 * 23? Answer with the number."}]),
        2000,
        true,
        None,
    )
    .await;
    let reasoning = assistant_message(&r)
        .and_then(|m| m.get("reasoning_content"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let tx = msg_text(&r);
    check(
        "thinking",
        tx.contains("391") && !reasoning.is_empty(),
        format!(
            "reasoning {} chars, answer {:?}",
            reasoning.chars().count(),
            tx.chars().take(40).collect::<String>()
        ),
    );

    // 3. tool call + tool result
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather of a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]);
    let r = chat(
        srv.clone(),
        json!([{"role": "user", "content": "What's the weather in Tokyo? Use the tool."}]),
        300,
        false,
        Some(tools.clone()),
    )
    .await;
    let (ok, calls) = tool_calls_ok(&r, "get_weather", "tokyo");
    check(
        "tool call",
        ok,
        if !calls.is_empty() {
            serde_json::to_string(&calls)
                .unwrap_or_default()
                .chars()
                .take(120)
                .collect()
        } else {
            r.to_string().chars().take(160).collect()
        },
    );
    if ok {
        let cid = calls[0].get("id").and_then(|v| v.as_str()).unwrap_or("0");
        let msgs = json!([
            {"role": "user", "content": "What's the weather in Tokyo? Use the tool."},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "tool_call_id": cid, "content": "{\"temp_c\": 21, \"sky\": \"clear\"}"}
        ]);
        let r2 = chat(srv.clone(), msgs, 200, false, Some(tools)).await;
        let tx2 = msg_text(&r2);
        check(
            "tool result",
            tx2.contains("21"),
            format!("{:?}", tx2.chars().take(80).collect::<String>()),
        );
    }

    // 4. image
    let img = encode_rgb_png(64, 64, |x, y| {
        let dx = x as i32 - 32;
        let dy = y as i32 - 32;
        if dx * dx + dy * dy < 400 {
            [240, 200, 0]
        } else {
            [255, 255, 255]
        }
    });
    let data_url = format!("data:image/png;base64,{}", STANDARD.encode(&img));
    let r = chat(
        srv.clone(),
        json!([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "What shape and color is in this picture? Short answer."}
        ]}]),
        60,
        false,
        None,
    )
    .await;
    let tx = msg_text(&r);
    let txl = tx.to_lowercase();
    check(
        "image",
        txl.contains("yellow") || txl.contains("circle"),
        format!("{:?}", tx.chars().take(80).collect::<String>()),
    );

    println!("SKIP  video (native Rust check omits PyAV video generation)");

    // 5. long prompt
    let mut long = String::with_capacity(60_000);
    for i in 0..1500 {
        if i > 0 {
            long.push(' ');
        }
        long.push_str(&format!("Item {i}: fox {i} jumps over dog {}.", i * 7));
    }
    let t0 = Instant::now();
    let r = chat(
        srv.clone(),
        json!([{"role": "user", "content": format!("{long}\nWhat is item 1234's dog number? The number only.")}]),
        20,
        false,
        None,
    )
    .await;
    let tx = msg_text(&r);
    let pt = r
        .get("usage")
        .and_then(|u| u.get("prompt_tokens"))
        .cloned()
        .unwrap_or(Value::Null);
    check(
        "long prompt",
        tx.contains("8638"),
        format!(
            "{pt} tokens, {:?}, {:.1}s",
            tx.chars().take(20).collect::<String>(),
            t0.elapsed().as_secs_f64()
        ),
    );

    // 6. 8 concurrent == solo
    let prompts = [
        "Write a haiku about the sea.",
        "List the first ten prime numbers.",
        "Explain TCP in two sentences.",
        "Write a Python function that reverses a string.",
        "Name five rivers in Europe.",
        "What causes the seasons on Earth? Two sentences.",
        "Translate 'good morning' into French, Spanish and German.",
        "Give three tips for writing clear documentation.",
    ];
    let mut solo = Vec::with_capacity(prompts.len());
    for p in prompts {
        let r = chat(
            srv.clone(),
            json!([{"role": "user", "content": p}]),
            160,
            false,
            None,
        )
        .await;
        solo.push(msg_text(&r));
    }
    let mut handles = Vec::with_capacity(prompts.len());
    for p in prompts {
        let s = srv.clone();
        handles.push(tokio::spawn(async move {
            let r = chat(s, json!([{"role": "user", "content": p}]), 160, false, None).await;
            msg_text(&r)
        }));
    }
    let mut burst = Vec::with_capacity(prompts.len());
    for h in handles {
        burst.push(h.await.unwrap_or_default());
    }
    let same = solo
        .iter()
        .zip(&burst)
        .filter(|(a, b)| !a.is_empty() && a == b)
        .count();
    check(
        "concurrent == solo",
        same == prompts.len(),
        format!("{same} of {} equal", prompts.len()),
    );

    if let Some(save_path) = a.save {
        let _ = fs::write(&save_path, serde_json::to_string(&solo).unwrap_or_default());
    }
    if let Some(cmp_path) = a.compare {
        let (ok, detail) = match fs::read_to_string(&cmp_path)
            .and_then(|s| serde_json::from_str::<Value>(&s).map_err(Into::into))
        {
            Err(e) => (false, format!("cannot read {}: {e}", cmp_path.display())),
            Ok(ref_val) => match ref_val.as_array() {
                None => (
                    false,
                    format!("{} does not hold a list of replies", cmp_path.display()),
                ),
                Some(arr) if arr.len() != prompts.len() => (
                    false,
                    format!(
                        "{} holds {} replies, not {}",
                        cmp_path.display(),
                        arr.len(),
                        prompts.len()
                    ),
                ),
                Some(arr) => {
                    let ref_strs: Option<Vec<&str>> = arr
                        .iter()
                        .map(|v| v.as_str().filter(|s| !s.is_empty()))
                        .collect();
                    match ref_strs {
                        None => (
                            false,
                            format!("{} has an empty or non-text reply", cmp_path.display()),
                        ),
                        Some(_) if solo.iter().any(|s| s.is_empty()) => {
                            (false, "an empty reply here".to_string())
                        }
                        Some(refs) => {
                            let eq = solo
                                .iter()
                                .zip(refs)
                                .filter(|(a, b)| a.as_str() == *b)
                                .count();
                            (
                                eq == prompts.len(),
                                format!("{eq} of {} equal", prompts.len()),
                            )
                        }
                    }
                }
            },
        };
        check("equal to the saved replies", ok, detail);
    }

    if failed.is_empty() {
        println!("\nall passed");
        0
    } else {
        println!("\nFAILED: {}", failed.join(", "));
        1
    }
}

pub async fn run_smoke(a: SmokeArgs) -> i32 {
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
    let srv = Server::new(&url, &model, 600.0, 7, key);

    let mut results: Vec<Value> = Vec::new();
    let mut record = |name: &str, status: &str, detail: String| {
        results.push(json!({"check": name, "status": status, "detail": detail}));
        if detail.is_empty() {
            println!("{status:4} {name}");
        } else {
            println!("{status:4} {name}: {detail}");
        }
    };

    let leak_re = Regex::new(
        r"(?i)<\|?[^>\s]{0,24}(?:begin|end|im_|eos|bos|think|User|Assistant|tool)[^>]{0,24}\|?>|<｜[^>]*｜>",
    )
    .unwrap();
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

    // 1. health
    match srv.get_text("/health", Duration::from_secs(15)).await {
        Some(body) => record("health", "PASS", body.chars().take(300).collect()),
        None => {
            record("health", "FAIL", "unreachable".to_string());
            return 1;
        }
    }

    let post = |srv: Server, extra: Value| async move {
        let mut body = json!({
            "model": srv.model,
            "temperature": 0,
            "seed": 7,
            "chat_template_kwargs": {"enable_thinking": false}
        });
        if let (Some(dst), Some(src)) = (body.as_object_mut(), extra.as_object()) {
            for (k, v) in src {
                dst.insert(k.clone(), v.clone());
            }
        }
        srv.post_json(
            "/v1/chat/completions",
            &body,
            Some(Duration::from_secs(600)),
        )
        .await
    };

    // 2. plain reply + leak check
    let r = post(
        srv.clone(),
        json!({"messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 200}),
    )
    .await;
    if let Some(err) = r.get("error").and_then(|v| v.as_str()) {
        record("plain reply", "FAIL", err.to_string());
    } else {
        let txt = msg_text(&r);
        let fin = r["choices"][0]["finish_reason"].as_str().unwrap_or("");
        let ok = fin == "stop" && !txt.trim().is_empty() && r.get("usage").is_some();
        record(
            "plain reply",
            if ok { "PASS" } else { "FAIL" },
            format!(
                "finish={fin} usage={} text={:?}",
                r["usage"],
                txt.chars().take(60).collect::<String>()
            ),
        );
        if let Some(m) = leak_re.find(&txt) {
            record("no special-token leak", "FAIL", m.as_str().to_string());
        } else {
            record("no special-token leak", "PASS", String::new());
        }
    }

    // 3. max_tokens
    let r = post(
        srv.clone(),
        json!({"messages": [{"role": "user", "content": "Write a long story about a dragon."}], "max_tokens": 5}),
    )
    .await;
    if let Some(err) = r.get("error").and_then(|v| v.as_str()) {
        record("max_tokens", "FAIL", err.to_string());
    } else {
        let fin = r["choices"][0]["finish_reason"].as_str().unwrap_or("");
        let n = r["usage"]["completion_tokens"].as_u64().unwrap_or(999);
        record(
            "max_tokens",
            if fin == "length" && n <= 5 {
                "PASS"
            } else {
                "FAIL"
            },
            format!("finish={fin} completion_tokens={n}"),
        );
    }

    // 4. stop string
    let r = post(
        srv.clone(),
        json!({"messages": [{"role": "user", "content": "Count from 1 to 10 as digits separated by spaces, nothing else."}], "max_tokens": 64, "stop": ["5"]}),
    )
    .await;
    if let Some(err) = r.get("error").and_then(|v| v.as_str()) {
        record("stop string", "FAIL", err.to_string());
    } else {
        let fin = r["choices"][0]["finish_reason"].as_str().unwrap_or("");
        let t = msg_text(&r);
        let ok = !t.contains('5') && fin == "stop" && t.contains('4');
        record(
            "stop string",
            if ok { "PASS" } else { "FAIL" },
            format!("finish={fin} text={t:?}"),
        );
    }

    // 5. stream
    let stream_body = json!({
        "model": srv.model,
        "temperature": 0,
        "seed": 7,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 16,
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    match srv
        .client()
        .post(format!("{}/v1/chat/completions", srv.url))
        .timeout(Duration::from_secs(600))
        .json(&stream_body)
        .send()
        .await
    {
        Err(e) => record("stream", "FAIL", e.to_string()),
        Ok(resp) if !resp.status().is_success() => {
            record("stream", "FAIL", format!("HTTP {}", resp.status().as_u16()))
        }
        Ok(resp) => {
            let mut done = false;
            let mut usage = false;
            let mut text = String::new();
            let mut st = resp.bytes_stream();
            let mut buf = Vec::new();
            while let Some(Ok(chunk)) = st.next().await {
                buf.extend_from_slice(&chunk);
                while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                    let raw: Vec<u8> = buf.drain(..=nl).collect();
                    let line = String::from_utf8_lossy(&raw).trim().to_string();
                    if line == "data: [DONE]" {
                        done = true;
                    } else if let Some(d) = line.strip_prefix("data:") {
                        if let Ok(ev) = serde_json::from_str::<Value>(d.trim()) {
                            if ev.get("usage").map(|u| !u.is_null()).unwrap_or(false) {
                                usage = true;
                            }
                            if let Some(chs) = ev.get("choices").and_then(|v| v.as_array()) {
                                for ch in chs {
                                    if let Some(p) = ch
                                        .get("delta")
                                        .and_then(|d| d.get("content"))
                                        .and_then(|v| v.as_str())
                                    {
                                        text.push_str(p);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            record(
                "stream",
                if done && usage { "PASS" } else { "FAIL" },
                format!("[DONE]={done} usage_chunk={usage} text={text:?}"),
            );
        }
    }

    // 6. tool call + round trip
    let tool_msgs = json!([{"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool."}]);
    let r = post(
        srv.clone(),
        json!({"messages": tool_msgs, "tools": tools, "max_tokens": 1024}),
    )
    .await;
    let mut saved_calls: Vec<Value> = Vec::new();
    if let Some(err) = r.get("error").and_then(|v| v.as_str()) {
        record("tool call", "FAIL", err.to_string());
    } else {
        let fin = r["choices"][0]["finish_reason"].as_str().unwrap_or("");
        let calls = r["choices"][0]["message"]["tool_calls"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut ok = false;
        for c in &calls {
            if let Ok(args) =
                serde_json::from_str::<Value>(c["function"]["arguments"].as_str().unwrap_or(""))
            {
                if c["function"]["name"].as_str() == Some("add_tags")
                    && args
                        .get("tags")
                        .and_then(|t| t.as_array())
                        .map(|a| a.len() >= 3)
                        .unwrap_or(false)
                    && args.get("doc_id").and_then(|v| v.as_i64()).is_some()
                {
                    ok = true;
                }
            }
        }
        record(
            "tool call",
            if ok { "PASS" } else { "FAIL" },
            format!(
                "finish={fin} calls={}",
                serde_json::to_string(&calls)
                    .unwrap_or_default()
                    .chars()
                    .take(200)
                    .collect::<String>()
            ),
        );
        if ok && fin != "tool_calls" {
            record(
                "tool call finish_reason",
                "FAIL",
                format!("{fin:?}, expected 'tool_calls'"),
            );
        }
        saved_calls = calls;
    }
    if !saved_calls.is_empty() {
        let cid = saved_calls[0]
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("call_0");
        let follow = json!([
            {"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool."},
            {"role": "assistant", "content": null, "tool_calls": saved_calls},
            {"role": "tool", "tool_call_id": cid, "content": "ok: 3 tags attached"}
        ]);
        let r2 = post(
            srv.clone(),
            json!({"messages": follow, "tools": tools, "max_tokens": 200}),
        )
        .await;
        if let Some(err) = r2.get("error").and_then(|v| v.as_str()) {
            record("tool result round trip", "FAIL", err.to_string());
        } else {
            let fin = r2["choices"][0]["finish_reason"].as_str().unwrap_or("");
            let t = msg_text(&r2);
            record(
                "tool result round trip",
                if !t.trim().is_empty() && fin == "stop" {
                    "PASS"
                } else {
                    "FAIL"
                },
                format!(
                    "finish={fin} text={:?}",
                    t.chars().take(80).collect::<String>()
                ),
            );
        }
    }

    // 7. greedy repeat
    let b = json!({"messages": [{"role": "user", "content": "Name three primary colors, comma separated."}], "max_tokens": 60});
    let r1 = post(srv.clone(), b.clone()).await;
    let r2 = post(srv.clone(), b).await;
    if r1.get("error").is_some() || r2.get("error").is_some() {
        record(
            "greedy repeat",
            "FAIL",
            format!("{} {}", r1["error"], r2["error"]),
        );
    } else {
        let t1 = msg_text(&r1);
        let t2 = msg_text(&r2);
        record(
            "greedy repeat",
            if t1 == t2 { "PASS" } else { "FAIL" },
            format!("{t1:?} vs {t2:?}"),
        );
    }

    // 8. thinking on/off (INFO)
    for think in [true, false] {
        let r = post(
            srv.clone(),
            json!({"messages": [{"role": "user", "content": "What is 17 * 23?"}], "max_tokens": 600, "chat_template_kwargs": {"enable_thinking": think}}),
        )
        .await;
        let label = format!("thinking {}", if think { "on" } else { "off" });
        if let Some(err) = r.get("error").and_then(|v| v.as_str()) {
            record(&label, "FAIL", err.to_string());
        } else {
            let fin = r["choices"][0]["finish_reason"].as_str().unwrap_or("");
            let m = assistant_message(&r);
            let reason = m
                .and_then(|m| m.get("reasoning_content").or_else(|| m.get("reasoning")))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let txt = msg_text(&r);
            record(
                &label,
                "INFO",
                format!(
                    "finish={fin} reasoning_chars={} content={:?}",
                    reason.chars().count(),
                    txt.chars().take(60).collect::<String>()
                ),
            );
        }
    }

    let fails: Vec<&str> = results
        .iter()
        .filter(|x| x["status"].as_str() == Some("FAIL"))
        .filter_map(|x| x["check"].as_str())
        .collect();
    println!(
        "smoke: {}",
        if fails.is_empty() {
            "OK".to_string()
        } else {
            format!("FAILED: {}", fails.join(", "))
        }
    );
    if let Some(jp) = a.json {
        let _ = fs::write(
            &jp,
            serde_json::to_string_pretty(&results).unwrap_or_default(),
        );
    }
    if fails.is_empty() {
        0
    } else {
        1
    }
}
