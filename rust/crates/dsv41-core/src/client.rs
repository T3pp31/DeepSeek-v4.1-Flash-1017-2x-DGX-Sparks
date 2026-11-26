//! Async OpenAI / TensorFold HTTP & SSE client with cancellation and inter-token gap metrics.

use crate::metrics::{health_counters, metrics_counters, parse_metrics, Counters};
use crate::{canonical_json_py, sha256_hex, DEFAULT_MODEL};
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Map, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Notify;

#[derive(Clone)]
pub struct Server {
    pub url: String,
    pub model: String,
    pub timeout: Duration,
    pub seed: i64,
    pub api_key: String,
    http: reqwest::Client,
}

impl Server {
    pub fn new(url: &str, model: &str, timeout_s: f64, seed: i64, api_key: String) -> Self {
        let mut clean = url.trim_end_matches('/').to_string();
        if let Some(stripped) = clean.strip_suffix("/v1") {
            clean = stripped.to_string();
        }
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if !api_key.is_empty() {
            if let Ok(v) = HeaderValue::from_str(&format!("Bearer {api_key}")) {
                headers.insert(AUTHORIZATION, v);
            }
        }
        let http = reqwest::Client::builder()
            .no_proxy()
            .default_headers(headers)
            .build()
            .expect("reqwest client build");
        Self {
            url: clean,
            model: if model.is_empty() {
                DEFAULT_MODEL.to_string()
            } else {
                model.to_string()
            },
            timeout: Duration::from_secs_f64(timeout_s.max(1.0)),
            seed,
            api_key,
            http,
        }
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.http
    }

    pub async fn get_json(&self, path: &str, timeout: Duration) -> Value {
        match self
            .http
            .get(format!("{}{}", self.url, path))
            .timeout(timeout)
            .send()
            .await
        {
            Ok(resp) => match resp.bytes().await {
                Ok(b) => serde_json::from_slice(&b)
                    .unwrap_or_else(|e| json!({"error": format!("JSONDecodeError: {e}")})),
                Err(e) => json!({"error": format!("ReadError: {e}")}),
            },
            Err(e) => json!({"error": format!("RequestError: {e}")}),
        }
    }

    pub async fn get_text(&self, path: &str, timeout: Duration) -> Option<String> {
        let resp = self
            .http
            .get(format!("{}{}", self.url, path))
            .timeout(timeout)
            .send()
            .await
            .ok()?;
        let bytes = resp.bytes().await.ok()?;
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub async fn post_json(&self, path: &str, body: &Value, timeout: Option<Duration>) -> Value {
        let t0 = Instant::now();
        let req = self
            .http
            .post(format!("{}{}", self.url, path))
            .timeout(timeout.unwrap_or(self.timeout))
            .json(body);
        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                match resp.bytes().await {
                    Ok(b) => {
                        if !status.is_success() {
                            let msg = String::from_utf8_lossy(&b);
                            let short: String = msg.chars().take(300).collect();
                            return json!({
                                "error": format!("HTTP {}: {}", status.as_u16(), short),
                                "http_status": status.as_u16()
                            });
                        }
                        match serde_json::from_slice::<Value>(&b) {
                            Ok(mut v) => {
                                if let Some(obj) = v.as_object_mut() {
                                    obj.insert(
                                        "_seconds".to_string(),
                                        json!(t0.elapsed().as_secs_f64()),
                                    );
                                }
                                v
                            }
                            Err(e) => json!({"error": format!("JSONDecodeError: {e}")}),
                        }
                    }
                    Err(e) => json!({"error": format!("ReadError: {e}")}),
                }
            }
            Err(e) => json!({"error": format!("RequestError: {e}")}),
        }
    }

    pub async fn health(&self) -> Value {
        self.get_json("/health", Duration::from_secs(15)).await
    }

    pub async fn alive(&self) -> bool {
        self.health().await.get("status").and_then(|s| s.as_str()) == Some("ok")
    }

    pub async fn wait_healthy(&self, timeout_s: f64) -> bool {
        let deadline = Instant::now() + Duration::from_secs_f64(timeout_s.max(0.1));
        while Instant::now() < deadline {
            if self.alive().await {
                return true;
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        false
    }

    pub async fn counters(&self) -> Counters {
        let mut out = health_counters(&self.health().await);
        if let Some(text) = self.get_text("/metrics", Duration::from_secs(15)).await {
            for (k, v) in metrics_counters(&parse_metrics(&text)) {
                out.entry(k).or_insert(v);
            }
        }
        out
    }

    pub async fn tokenize(&self, messages: &Value) -> (Option<i64>, Option<i64>, Value) {
        let body = json!({
            "model": self.model,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": false}
        });
        let r = self
            .post_json("/tokenize", &body, Some(Duration::from_secs(600)))
            .await;
        let count = r.get("count").and_then(|v| v.as_i64());
        let max_len = r.get("max_model_len").and_then(|v| v.as_i64());
        (count, max_len, r)
    }

    pub fn body(&self, messages: &Value, max_tokens: u64, stream: bool) -> Value {
        let mut b = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": self.seed,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": false}
        });
        if stream {
            b.as_object_mut()
                .unwrap()
                .insert("stream_options".to_string(), json!({"include_usage": true}));
        }
        b
    }

    pub async fn chat(&self, messages: &Value, max_tokens: u64) -> Value {
        let b = self.body(messages, max_tokens, false);
        let r = self.post_json("/v1/chat/completions", &b, None).await;
        result_row(messages, &r)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn stream(
        &self,
        messages: &Value,
        max_tokens: u64,
        cancel_after: Option<f64>,
        cancel_on_first: bool,
        cancel_notify: Option<Arc<Notify>>,
        first_notify: Option<Arc<Notify>>,
        keep_text: bool,
    ) -> Value {
        let mut row = Map::new();
        row.insert(
            "prompt_sha".to_string(),
            Value::String(sha256_hex(canonical_json_py(messages).as_bytes())),
        );
        let t0 = Instant::now();
        let body = self.body(messages, max_tokens, true);
        let mut text = String::new();
        let mut last: Option<Instant> = None;
        let mut gaps: Vec<f64> = Vec::new();
        let mut events: u64 = 0;
        let mut usage: Option<Value> = None;
        let mut tf = Map::new();
        let mut finish: Option<String> = None;
        let mut ttft: Option<f64> = None;
        let mut cut = false;

        let cancel_timer = cancel_after.map(|s| tokio::time::sleep(Duration::from_secs_f64(s)));
        tokio::pin!(cancel_timer);

        let req_fut = self
            .http
            .post(format!("{}/v1/chat/completions", self.url))
            .timeout(self.timeout)
            .json(&body)
            .send();

        let resp_res = tokio::select! {
            r = req_fut => Some(r),
            _ = async {
                if let Some(n) = &cancel_notify {
                    n.notified().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {
                cut = true;
                None
            }
            _ = async {
                if let Some(t) = cancel_timer.as_mut().as_pin_mut() {
                    t.await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {
                cut = true;
                None
            }
        };

        if let Some(resp_res) = resp_res {
            match resp_res {
                Err(e) => {
                    if !cut {
                        row.insert(
                            "error".to_string(),
                            Value::String(format!("RequestError: {e}")),
                        );
                    }
                }
                Ok(resp) => {
                    if !resp.status().is_success() {
                        let code = resp.status().as_u16();
                        let b = resp.bytes().await.unwrap_or_default();
                        let msg: String = String::from_utf8_lossy(&b).chars().take(300).collect();
                        row.insert(
                            "error".to_string(),
                            Value::String(format!("HTTP {code}: {msg}")),
                        );
                        row.insert("http_status".to_string(), json!(code));
                        if let Some(f) = &first_notify {
                            f.notify_one();
                        }
                        return Value::Object(row);
                    }
                    let mut stream = resp.bytes_stream();
                    let mut buf = Vec::<u8>::with_capacity(4096);
                    'outer: loop {
                        let next_chunk = tokio::select! {
                            c = stream.next() => c,
                            _ = async {
                                if let Some(n) = &cancel_notify {
                                    n.notified().await;
                                } else {
                                    std::future::pending::<()>().await;
                                }
                            } => {
                                cut = true;
                                break 'outer;
                            }
                            _ = async {
                                if let Some(t) = cancel_timer.as_mut().as_pin_mut() {
                                    t.await;
                                } else {
                                    std::future::pending::<()>().await;
                                }
                            } => {
                                cut = true;
                                break 'outer;
                            }
                        };
                        let Some(chunk_res) = next_chunk else {
                            break;
                        };
                        match chunk_res {
                            Err(e) => {
                                if !cut {
                                    row.insert(
                                        "error".to_string(),
                                        Value::String(format!("StreamError: {e}")),
                                    );
                                }
                                break;
                            }
                            Ok(bytes) => {
                                buf.extend_from_slice(&bytes);
                                while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                                    let line_bytes: Vec<u8> = buf.drain(..=nl).collect();
                                    let line = String::from_utf8_lossy(&line_bytes);
                                    let line = line.trim();
                                    let Some(data) = line.strip_prefix("data:") else {
                                        continue;
                                    };
                                    let data = data.trim();
                                    if data == "[DONE]" {
                                        break 'outer;
                                    }
                                    let Ok(ev) = serde_json::from_str::<Value>(data) else {
                                        continue;
                                    };
                                    events += 1;
                                    if let Some(err) = ev.get("error") {
                                        if !err.is_null() {
                                            let s: String =
                                                err.to_string().chars().take(300).collect();
                                            row.insert("error".to_string(), Value::String(s));
                                        }
                                    }
                                    if let Some(u) = ev.get("usage") {
                                        if !u.is_null() {
                                            usage = Some(u.clone());
                                        }
                                    }
                                    if let Some(tmap) =
                                        ev.get("tensorfold").and_then(|v| v.as_object())
                                    {
                                        for (k, v) in tmap {
                                            tf.insert(k.clone(), v.clone());
                                        }
                                    }
                                    if let Some(choices) =
                                        ev.get("choices").and_then(|v| v.as_array())
                                    {
                                        for ch in choices {
                                            let piece = ch
                                                .get("delta")
                                                .and_then(|d| d.get("content"))
                                                .and_then(|c| c.as_str())
                                                .unwrap_or("");
                                            if !piece.is_empty() {
                                                let now = Instant::now();
                                                if ttft.is_none() {
                                                    ttft =
                                                        Some(now.duration_since(t0).as_secs_f64());
                                                    if let Some(f) = &first_notify {
                                                        f.notify_one();
                                                    }
                                                } else if let Some(prev) = last {
                                                    gaps.push(
                                                        now.duration_since(prev).as_secs_f64(),
                                                    );
                                                }
                                                last = Some(now);
                                                text.push_str(piece);
                                                if cancel_on_first {
                                                    cut = true;
                                                    break 'outer;
                                                }
                                            }
                                            if let Some(fr) =
                                                ch.get("finish_reason").and_then(|v| v.as_str())
                                            {
                                                if !fr.is_empty() {
                                                    finish = Some(fr.to_string());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(f) = &first_notify {
            f.notify_one();
        }

        row.insert("cancelled".to_string(), json!(cut && finish.is_none()));
        let u = usage.as_ref().and_then(|v| v.as_object());
        let tsha = tf
            .get("token_sha")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty());
        let cached = u
            .and_then(|u| u.get("prompt_tokens_details"))
            .and_then(|d| d.get("cached_tokens"))
            .cloned()
            .or_else(|| tf.get("cached_tokens").cloned())
            .unwrap_or(Value::Null);
        let text_sha = sha256_hex(text.as_bytes());
        row.insert(
            "prompt_tokens".to_string(),
            u.and_then(|u| u.get("prompt_tokens"))
                .cloned()
                .unwrap_or(Value::Null),
        );
        row.insert(
            "completion_tokens".to_string(),
            u.and_then(|u| u.get("completion_tokens"))
                .cloned()
                .unwrap_or(Value::Null),
        );
        row.insert("cached_tokens".to_string(), cached);
        row.insert(
            "finish_reason".to_string(),
            finish.map(Value::String).unwrap_or(Value::Null),
        );
        row.insert(
            "token_sha".to_string(),
            Value::String(tsha.map(String::from).unwrap_or_else(|| text_sha.clone())),
        );
        row.insert(
            "token_sha_source".to_string(),
            Value::String(if tsha.is_some() { "server" } else { "text" }.to_string()),
        );
        row.insert("text_sha".to_string(), Value::String(text_sha));
        row.insert("text_len".to_string(), json!(text.chars().count()));
        row.insert("events".to_string(), json!(events));
        row.insert(
            "ttft".to_string(),
            ttft.map(|v| json!((v * 10000.0).round() / 10000.0))
                .unwrap_or(Value::Null),
        );
        let max_gap = gaps.iter().copied().reduce(f64::max);
        row.insert(
            "max_gap".to_string(),
            max_gap
                .map(|v| json!((v * 10000.0).round() / 10000.0))
                .unwrap_or(Value::Null),
        );
        row.insert("gaps".to_string(), json!(gaps.len()));
        row.insert(
            "seconds".to_string(),
            json!((t0.elapsed().as_secs_f64() * 1000.0).round() / 1000.0),
        );
        if keep_text {
            row.insert("text".to_string(), Value::String(text));
        }
        Value::Object(row)
    }
}

pub fn result_row(messages: &Value, r: &Value) -> Value {
    let mut out = Map::new();
    out.insert(
        "prompt_sha".to_string(),
        Value::String(sha256_hex(canonical_json_py(messages).as_bytes())),
    );
    if let Some(err) = r.get("error") {
        out.insert("error".to_string(), err.clone());
        if let Some(st) = r.get("http_status") {
            out.insert("http_status".to_string(), st.clone());
        }
        return Value::Object(out);
    }
    let ch = match r
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
    {
        Some(c) => c,
        None => {
            out.insert(
                "error".to_string(),
                Value::String(format!(
                    "unexpected reply shape: {}",
                    &r.to_string().chars().take(200).collect::<String>()
                )),
            );
            return Value::Object(out);
        }
    };
    let text = ch
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
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
        "prefill_seconds".to_string(),
        tf.and_then(|t| t.get("prefill_seconds"))
            .cloned()
            .unwrap_or(Value::Null),
    );
    let secs = r.get("_seconds").and_then(|v| v.as_f64()).unwrap_or(0.0);
    out.insert(
        "seconds".to_string(),
        json!((secs * 1000.0).round() / 1000.0),
    );
    Value::Object(out)
}

pub fn same_reply(base: Option<&Value>, got: &Value, allow_weak: bool) -> (bool, String) {
    let Some(base) = base else {
        return (false, "no baseline row".to_string());
    };
    if let Some(err) = got.get("error") {
        return (false, format!("request error: {err}"));
    }
    if let Some(err) = base.get("error") {
        return (false, format!("baseline row has an error: {err}"));
    }
    if base.get("prompt_sha") != got.get("prompt_sha") {
        return (
            false,
            "prompt differs from the baseline's (different salt or shapes)".to_string(),
        );
    }
    let b_src = base
        .get("token_sha_source")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let g_src = got
        .get("token_sha_source")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if b_src == "server" && g_src == "server" {
        let bs = base.get("token_sha").and_then(|v| v.as_str()).unwrap_or("");
        let gs = got.get("token_sha").and_then(|v| v.as_str()).unwrap_or("");
        let ok = bs == gs;
        return (
            ok,
            format!(
                "token_sha {} ({} vs baseline {})",
                if ok { "equal" } else { "DIFFERS" },
                &gs[..gs.len().min(12)],
                &bs[..bs.len().min(12)]
            ),
        );
    }
    let ok = base.get("text_sha") == got.get("text_sha");
    if !allow_weak {
        let side = if b_src != "server" && g_src != "server" {
            "both"
        } else {
            "one"
        };
        return (
            false,
            format!(
                "WEAK: no server token_sha on {side} side(s); text {} (use --allow-weak to accept)",
                if ok { "equal" } else { "differs" }
            ),
        );
    }
    (
        ok,
        format!(
            "text_sha {} (weak: --allow-weak)",
            if ok { "equal" } else { "DIFFERS" }
        ),
    )
}
