pub mod auth;
pub mod client;
pub mod docker_sock;
pub mod meminfo;
pub mod metrics;
pub mod png;
pub mod prose;
pub mod rank_dir;
pub mod receipt;
pub mod safetensors;

use sha2::{Digest, Sha256};

pub const DEFAULT_MODEL: &str = "DeepSeek-V4.1-Flash-1017";
pub const SPILL_MARKER: &str = ".dsv41-tensorfold-spill";

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

pub fn random_hex(n_bytes: usize) -> String {
    let mut buf = vec![0u8; n_bytes];
    getrandom::getrandom(&mut buf).expect("OS CSPRNG failed");
    hex::encode(buf)
}

pub fn now_iso8601() -> String {
    chrono::Local::now()
        .format("%Y-%m-%dT%H:%M:%S%z")
        .to_string()
}

pub fn now_hms() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

pub fn now_stamp() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Canonical JSON representation matching Python's `json.dumps(obj, sort_keys=True)`
/// (default separators `", "` and `": "`, sorted keys) so prompt_sha matches legacy records.
pub fn canonical_json_py(val: &serde_json::Value) -> String {
    let mut out = String::new();
    write_py_json(val, &mut out);
    out
}

fn write_py_json(val: &serde_json::Value, out: &mut String) {
    match val {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => out.push_str(&n.to_string()),
        serde_json::Value::String(s) => {
            out.push_str(&serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string()));
        }
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(item, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            out.push('{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_else(|_| "\"\"".to_string()));
                out.push_str(": ");
                write_py_json(&map[*k], out);
            }
            out.push('}');
        }
    }
}
