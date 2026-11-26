use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use dsv41_core::auth::{port_from_local_sh, redact_str, redact_value, resolve_strict};
use dsv41_core::receipt::Receipt;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

fn tools_bin() -> &'static str {
    env!("CARGO_BIN_EXE_dsv41-tools")
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "dsv41-tools-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        Self(p)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_checklib_credentials_and_receipt_redaction() {
    let _g = ENV_LOCK.lock().unwrap();
    let tmp = TempDir::new("creds");
    let key = "sk-tf-test-0123456789abcdef";

    let saved_api = std::env::var("API_KEY").ok();
    let saved_file = std::env::var("API_KEY_FILE").ok();
    let saved_home = std::env::var("HOME").ok();

    let restore = || {
        match &saved_api {
            Some(v) => std::env::set_var("API_KEY", v),
            None => std::env::remove_var("API_KEY"),
        }
        match &saved_file {
            Some(v) => std::env::set_var("API_KEY_FILE", v),
            None => std::env::remove_var("API_KEY_FILE"),
        }
        match &saved_home {
            Some(v) => std::env::set_var("HOME", v),
            None => std::env::remove_var("HOME"),
        }
    };

    // 1. No default path and no API_KEY_FILE variable in strict mode
    let home = tmp.path().join("home");
    let state = home.join(".local/state/dsv41-tensorfold");
    fs::create_dir_all(&state).unwrap();
    let def_file = state.join("api-key");
    fs::write(&def_file, format!("{key}\n")).unwrap();
    std::env::remove_var("API_KEY");
    std::env::set_var("HOME", &home);
    std::env::set_var("API_KEY_FILE", &def_file);
    assert_eq!(resolve_strict(None).unwrap(), "");

    // 2. API_KEY env var & exclusion of recipe switches "0", "1", ""
    std::env::set_var("API_KEY", format!(" {key} "));
    assert_eq!(resolve_strict(None).unwrap(), key);
    for sw in ["0", "1", ""] {
        std::env::set_var("API_KEY", sw);
        assert_eq!(resolve_strict(None).unwrap(), "");
    }

    // 3. Explicit --api-key-file: comments, header prefix, env precedence, unreadable error
    std::env::remove_var("API_KEY");
    let f1 = tmp.path().join("k1");
    fs::write(&f1, format!("# comment\n\n{key}\n")).unwrap();
    assert_eq!(resolve_strict(Some(&f1)).unwrap(), key);

    let f2 = tmp.path().join("k2");
    fs::write(&f2, format!("Authorization: Bearer-less {key}\n")).unwrap();
    assert_eq!(
        resolve_strict(Some(&f2)).unwrap(),
        format!("Bearer-less {key}")
    );

    std::env::set_var("API_KEY", "sk-env");
    assert_eq!(resolve_strict(Some(&f1)).unwrap(), "sk-env");
    std::env::remove_var("API_KEY");

    let missing = tmp.path().join("missing-key");
    let err = resolve_strict(Some(&missing)).unwrap_err();
    assert!(err.contains("cannot read it"), "{err}");

    // 4. Receipt redaction
    let mut meta = Map::new();
    meta.insert("note".to_string(), json!(format!("url with {key}")));
    let mut r = Receipt::new("t", meta, key.to_string());
    let mut extra = Map::new();
    extra.insert("body".to_string(), json!({"echo": [key, 3]}));
    r.add("c1", "FAIL", &format!("server said: bad key {key}"), extra);
    let out = tmp.path().join("receipt.json");
    let rc = r.finish(Some(&out), false);
    assert_eq!(rc, 1);
    let text = fs::read_to_string(&out).unwrap();
    assert!(!text.contains(key), "key leaked into receipt: {text}");
    let doc: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(doc["checks"][0]["body"]["echo"], json!(["<api-key>", 3]));
    assert_eq!(
        redact_str(&format!("bearer {key}"), key),
        "bearer <api-key>"
    );
    assert_eq!(
        redact_value(&json!({"a": key}), key),
        json!({"a": "<api-key>"})
    );

    restore();
}

#[test]
fn test_bench_local_sh_parsing_never_executes_shell() {
    let tmp = TempDir::new("localsh");
    let local = tmp.path().join("local.sh");
    assert_eq!(port_from_local_sh(&local), None);

    fs::write(&local, "export PORT=\"8007\" # local\n").unwrap();
    assert_eq!(port_from_local_sh(&local).as_deref(), Some("8007"));

    fs::write(&local, "PORT=8006\nPORT='8007'\n").unwrap();
    assert_eq!(port_from_local_sh(&local).as_deref(), Some("8007"));

    let marker = tmp.path().join("executed");
    fs::write(
        &local,
        format!(
            "# PORT=9001\nSOME_PORT=9002\nPORT=8007\"\nPORT=$(touch {})\n",
            marker.display()
        ),
    )
    .unwrap();
    assert_eq!(port_from_local_sh(&local), None);
    assert!(!marker.exists(), "local.sh must never be executed");
}

// ---------------------------------------------------------------- Mock validate server (port of tools/test_validate.py)

const PROMPTS: [&str; 8] = [
    "Write a haiku about the sea.",
    "List the first ten prime numbers.",
    "Explain TCP in two sentences.",
    "Write a Python function that reverses a string.",
    "Name five rivers in Europe.",
    "What causes the seasons on Earth? Two sentences.",
    "Translate 'good morning' into French, Spanish and German.",
    "Give three tips for writing clear documentation.",
];

fn reply_to(p: &str) -> String {
    format!("Reply to: {p}")
}

fn completion(content: Value, extra: Map<String, Value>) -> (StatusCode, Value) {
    let mut msg = Map::new();
    msg.insert("role".to_string(), json!("assistant"));
    msg.insert("content".to_string(), content);
    for (k, v) in extra {
        msg.insert(k, v);
    }
    (
        StatusCode::OK,
        json!({
            "choices": [{"message": Value::Object(msg)}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }),
    )
}

type FeatureFn = Arc<dyn Fn(StatusCode, Value) -> (StatusCode, Value) + Send + Sync>;

#[derive(Clone, Default)]
struct MockConfig {
    overrides: HashMap<String, (StatusCode, Value)>,
    feature_hook: Option<FeatureFn>,
}

fn feature_answer(body: &Value) -> Option<(StatusCode, Value)> {
    let msgs = body["messages"].as_array()?;
    let last = msgs.last()?;
    let content = last.get("content")?;
    if let Some(arr) = content.as_array() {
        let has_video = arr
            .iter()
            .any(|p| p.get("type").and_then(|v| v.as_str()) == Some("video_url"));
        return Some(completion(
            json!(if has_video {
                "Red at first, then blue."
            } else {
                "A yellow circle."
            }),
            Map::new(),
        ));
    }
    if last.get("role").and_then(|v| v.as_str()) == Some("tool") {
        return Some(completion(
            json!("It is 21 degrees and clear in Tokyo."),
            Map::new(),
        ));
    }
    let s = content.as_str().unwrap_or("");
    if s.contains("capital of France") {
        return Some(completion(json!("Paris"), Map::new()));
    }
    if s.contains("17 * 23") {
        let mut ex = Map::new();
        ex.insert("reasoning_content".to_string(), json!("17 * 23 = 391."));
        return Some(completion(json!("391"), ex));
    }
    if s.contains("weather in Tokyo") {
        let mut ex = Map::new();
        ex.insert(
            "tool_calls".to_string(),
            json!([{
                "id": "call_0",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}
            }]),
        );
        return Some(completion(json!(""), ex));
    }
    if s.contains("item 1234") {
        return Some(completion(json!("8638"), Map::new()));
    }
    None
}

async fn mock_chat(
    State(cfg): State<Arc<MockConfig>>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    if let Some((st, val)) = feature_answer(&body) {
        let (st2, val2) = match &cfg.feature_hook {
            Some(f) => f(st, val),
            None => (st, val),
        };
        return (st2, Json(val2));
    }
    if let Some(c) = body["messages"]
        .as_array()
        .and_then(|a| a.last())
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
    {
        if PROMPTS.contains(&c) {
            if let Some(ov) = cfg.overrides.get(c) {
                return (ov.0, Json(ov.1.clone()));
            }
            let (st, val) = completion(json!(reply_to(c)), Map::new());
            return (st, Json(val));
        }
    }
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": {"message": "unexpected prompt"}})),
    )
}

async fn spawn_mock(cfg: MockConfig) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = Router::new()
        .route("/v1/chat/completions", post(mock_chat))
        .with_state(Arc::new(cfg));
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (format!("http://{addr}"), handle)
}

fn run_validate_cli(args: &[&str]) -> (i32, String) {
    let out = Command::new(tools_bin())
        .arg("validate")
        .args(args)
        .output()
        .unwrap();
    let mut combined = String::from_utf8_lossy(&out.stdout).to_string();
    combined.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), combined)
}

fn check_line(out: &str, name: &str) -> String {
    out.lines()
        .find(|l| l.contains(name) && (l.starts_with("PASS") || l.starts_with("FAIL")))
        .unwrap_or("")
        .to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_validate_pass_compare_and_failure_modes() {
    let tmp = TempDir::new("validate");

    // 1. Clean pass + --save + --compare
    let (url, h) = spawn_mock(MockConfig::default()).await;
    let saved = tmp.path().join("ref.json");
    let (code, out) = run_validate_cli(&[&url, "--save", saved.to_str().unwrap()]);
    assert_eq!(code, 0, "{out}");
    for nm in [
        "text",
        "thinking",
        "tool call",
        "tool result",
        "image",
        "long prompt",
        "concurrent == solo",
    ] {
        assert!(check_line(&out, nm).starts_with("PASS"), "{nm}:\n{out}");
    }

    let (code2, out2) = run_validate_cli(&[&url, "--compare", saved.to_str().unwrap()]);
    assert_eq!(code2, 0, "{out2}");
    assert!(
        check_line(&out2, "equal to the saved replies").starts_with("PASS"),
        "{out2}"
    );

    // 2. Compare failure cases
    let bad_refs: Vec<(&str, String)> = vec![
        ("empty list", "[]".to_string()),
        ("short list", json!([reply_to(PROMPTS[0])]).to_string()),
        ("not a list", json!({"a": 1}).to_string()),
        ("null", "null".to_string()),
        ("invalid json", "not json".to_string()),
        (
            "reordered",
            json!(PROMPTS
                .iter()
                .rev()
                .map(|p| reply_to(p))
                .collect::<Vec<_>>())
            .to_string(),
        ),
        (
            "non-string element",
            json!(PROMPTS[..7]
                .iter()
                .map(|p| json!(reply_to(p)))
                .chain(std::iter::once(Value::Null))
                .collect::<Vec<_>>())
            .to_string(),
        ),
        (
            "whitespace diff",
            json!(PROMPTS[..7]
                .iter()
                .map(|p| reply_to(p))
                .chain(std::iter::once(format!("{} ", reply_to(PROMPTS[7]))))
                .collect::<Vec<_>>())
            .to_string(),
        ),
    ];
    for (label, raw) in bad_refs {
        let p = tmp.path().join(format!("{}.json", label.replace(' ', "_")));
        fs::write(&p, raw).unwrap();
        let (c, o) = run_validate_cli(&[&url, "--compare", p.to_str().unwrap()]);
        assert_eq!(c, 1, "case {label} should fail:\n{o}");
        assert!(
            check_line(&o, "equal to the saved replies").starts_with("FAIL"),
            "{label}:\n{o}"
        );
    }
    h.abort();

    // 3. Error-marked feature replies fail
    let cfg_err = MockConfig {
        feature_hook: Some(Arc::new(|st, mut val| {
            val["error"] = json!({"message": "engine failed"});
            (st, val)
        })),
        ..Default::default()
    };
    let (url_e, h_e) = spawn_mock(cfg_err).await;
    let (c_e, o_e) = run_validate_cli(&[&url_e]);
    assert_eq!(c_e, 1, "{o_e}");
    for nm in ["text", "thinking", "tool call", "image", "long prompt"] {
        assert!(check_line(&o_e, nm).starts_with("FAIL"), "{nm}:\n{o_e}");
    }
    h_e.abort();

    // 4. Tool call with null content passes
    let cfg_null = MockConfig {
        feature_hook: Some(Arc::new(|st, mut val| {
            if val["choices"][0]["message"].get("tool_calls").is_some() {
                val["choices"][0]["message"]["content"] = Value::Null;
            }
            (st, val)
        })),
        ..Default::default()
    };
    let (url_n, h_n) = spawn_mock(cfg_null).await;
    let (c_n, o_n) = run_validate_cli(&[&url_n]);
    assert_eq!(c_n, 0, "{o_n}");
    assert!(check_line(&o_n, "tool call").starts_with("PASS"), "{o_n}");
    h_n.abort();

    // 5. Malformed tool call (NaN in arguments, missing id, bad type, nonexistent tool) fails cleanly
    let bad_calls = vec![
        json!([{"id": "c0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\", \"x\": NaN}"}}]),
        json!([{"id": "", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}]),
        json!([{"id": "c0", "type": "not-a-function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}]),
        json!([{"id": "c0", "type": "function", "function": {"name": "nonexistent_tool", "arguments": "{\"city\": \"Tokyo\"}"}}]),
    ];
    for calls in bad_calls {
        let cfg_tc = MockConfig {
            feature_hook: Some(Arc::new(move |st, mut val| {
                if val["choices"][0]["message"].get("tool_calls").is_some() {
                    val["choices"][0]["message"]["tool_calls"] = calls.clone();
                }
                (st, val)
            })),
            ..Default::default()
        };
        let (u_tc, h_tc) = spawn_mock(cfg_tc).await;
        let (c_tc, o_tc) = run_validate_cli(&[&u_tc]);
        assert_eq!(c_tc, 1, "{o_tc}");
        assert!(check_line(&o_tc, "tool call").starts_with("FAIL"), "{o_tc}");
        h_tc.abort();
    }

    // 6. Option missing value stops cleanly without reaching server
    let (c_arg, _) = run_validate_cli(&["http://127.0.0.1:9", "--compare"]);
    assert_ne!(c_arg, 0);
}

#[test]
fn test_bench_compare_cli() {
    let tmp = TempDir::new("bench-cmp");
    let a = tmp.path().join("a.json");
    let b = tmp.path().join("b.json");
    let sample = |label: &str, decode: f64| {
        json!({
            "schema": 1,
            "meta": {"label": label, "model": "DeepSeek-V4.1-Flash-1017"},
            "decode": {"tok_s": decode, "ttft_ms": 100.0, "tokens": 200},
            "prefill": [{"target": 4096, "prompt_tokens": 4100, "tok_s": 2100.0, "ttft_s": 1.95}],
            "multi": [{"streams": 4, "agg_tok_s": 150.0, "per_stream_tok_s": 37.5}]
        })
    };
    fs::write(&a, serde_json::to_string(&sample("before", 80.0)).unwrap()).unwrap();
    fs::write(&b, serde_json::to_string(&sample("after", 100.0)).unwrap()).unwrap();

    let out = Command::new(tools_bin())
        .args([
            "bench",
            "--compare",
            a.to_str().unwrap(),
            b.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
