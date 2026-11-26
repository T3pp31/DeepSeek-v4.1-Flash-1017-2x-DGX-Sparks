//! API key credential resolution and secret redaction.
//!
//! Two credential modes exist in the recipe:
//! 1. Strict checklib mode (`resolve_strict`): reads ONLY `API_KEY` env var (excluding `"0"`/`"1"`
//!    recipe switches) or an explicit `--api-key-file` path; never reads default paths or `API_KEY_FILE`.
//! 2. Standard tool mode (`resolve_standard`): reads `API_KEY` env var (excluding `"0"`/`"1"`),
//!    else `API_KEY_FILE` env var, else `~/.local/state/dsv41-tensorfold/api-key`.

use serde_json::Value;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn parse_key_file_contents(contents: &str) -> String {
    for line in contents.lines() {
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        return match s.split_once(':') {
            Some((_, rhs)) => rhs.trim().to_string(),
            None => s.to_string(),
        };
    }
    String::new()
}

pub fn resolve_strict(api_key_file: Option<&Path>) -> Result<String, String> {
    if let Ok(v) = env::var("API_KEY") {
        let k = v.trim();
        if !k.is_empty() && k != "0" && k != "1" {
            return Ok(k.to_string());
        }
    }
    let Some(path) = api_key_file else {
        return Ok(String::new());
    };
    let text = fs::read_to_string(path).map_err(|e| {
        format!(
            "--api-key-file {}: cannot read it ({})",
            path.display(),
            e.kind()
        )
    })?;
    Ok(parse_key_file_contents(&text))
}

pub fn resolve_standard(api_key_file: Option<&Path>) -> String {
    if let Ok(v) = env::var("API_KEY") {
        let k = v.trim();
        if !k.is_empty() && k != "0" && k != "1" {
            return k.to_string();
        }
    }
    if let Some(p) = api_key_file {
        if let Ok(text) = fs::read_to_string(p) {
            return parse_key_file_contents(&text);
        }
        return String::new();
    }
    let path = match env::var("API_KEY_FILE") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => match env::var("HOME") {
            Ok(h) => PathBuf::from(h).join(".local/state/dsv41-tensorfold/api-key"),
            Err(_) => return String::new(),
        },
    };
    match fs::read_to_string(path) {
        Ok(text) => parse_key_file_contents(&text),
        Err(_) => String::new(),
    }
}

pub fn redact_str(s: &str, key: &str) -> String {
    if key.is_empty() {
        s.to_string()
    } else {
        s.replace(key, "<api-key>")
    }
}

pub fn redact_value(val: &Value, key: &str) -> Value {
    if key.is_empty() {
        return val.clone();
    }
    match val {
        Value::String(s) => Value::String(s.replace(key, "<api-key>")),
        Value::Array(arr) => Value::Array(arr.iter().map(|x| redact_value(x, key)).collect()),
        Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                out.insert(k.replace(key, "<api-key>"), redact_value(v, key));
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// Resolve default server base URL: `API_URL`, else `PORT` env, else literal `PORT=...` in
/// `<repo_root>/scripts/local.sh` (never executed), else `http://127.0.0.1:8888`.
pub fn default_base_url(repo_root: Option<&Path>) -> String {
    if let Ok(url) = env::var("API_URL") {
        if !url.is_empty() {
            return url;
        }
    }
    if let Ok(port) = env::var("PORT") {
        if !port.is_empty() {
            return format!("http://127.0.0.1:{port}");
        }
    }
    if let Some(root) = repo_root {
        if let Some(port) = port_from_local_sh(&root.join("scripts/local.sh")) {
            return format!("http://127.0.0.1:{port}");
        }
    }
    "http://127.0.0.1:8888".to_string()
}

pub fn port_from_local_sh(path: &Path) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    let mut port: Option<String> = None;
    for raw in text.lines() {
        let mut s = raw.trim();
        if let Some(rest) = s.strip_prefix("export") {
            if rest.starts_with(char::is_whitespace) {
                s = rest.trim_start();
            }
        }
        let Some(rest) = s.strip_prefix("PORT") else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix('=') else {
            continue;
        };
        let rest = rest.trim_start();
        let (val, after) = if let Some(r) = rest.strip_prefix('"') {
            let end = r.find('"')?;
            (&r[..end], &r[end + 1..])
        } else if let Some(r) = rest.strip_prefix('\'') {
            let end = r.find('\'')?;
            (&r[..end], &r[end + 1..])
        } else {
            let end = rest
                .find(|c: char| c.is_whitespace() || c == '#')
                .unwrap_or(rest.len());
            (&rest[..end], &rest[end..])
        };
        let after = after.trim_start();
        if !after.is_empty() && !after.starts_with('#') {
            continue;
        }
        if !val.is_empty() && val.len() <= 5 && val.bytes().all(|b| b.is_ascii_digit()) {
            port = Some(val.to_string());
        }
    }
    port
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_port_from_local_sh_literal_and_safety() {
        let dir = std::env::temp_dir().join(format!("dsv41-localsh-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let f = dir.join("local.sh");

        fs::write(&f, "export PORT=\"8007\" # local\n").unwrap();
        assert_eq!(port_from_local_sh(&f).as_deref(), Some("8007"));

        fs::write(&f, "PORT=8006\nPORT='8007'\n").unwrap();
        assert_eq!(port_from_local_sh(&f).as_deref(), Some("8007"));

        fs::write(
            &f,
            "# PORT=9001\nSOME_PORT=9002\nPORT=8007\"\nPORT=$(touch /tmp/x)\n",
        )
        .unwrap();
        assert_eq!(port_from_local_sh(&f), None);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_redact_value() {
        let v =
            serde_json::json!({"body": {"echo": ["sk-secret", 3]}, "sk-secret": "hi sk-secret"});
        let r = redact_value(&v, "sk-secret");
        assert_eq!(r["body"]["echo"][0], "<api-key>");
        assert_eq!(r["<api-key>"], "hi <api-key>");
    }
}
