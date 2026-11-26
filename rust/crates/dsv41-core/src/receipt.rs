//! Structured PASS/FAIL/SKIP check receipts with automatic API key redaction.

use crate::auth::{redact_str, redact_value};
use crate::now_iso8601;
use serde_json::{Map, Value};
use std::fs;
use std::path::Path;

pub struct Receipt {
    pub tool: String,
    pub doc: Map<String, Value>,
    pub api_key: String,
}

impl Receipt {
    pub fn new(tool: &str, mut meta: Map<String, Value>, api_key: String) -> Self {
        meta.insert("time".to_string(), Value::String(now_iso8601()));
        let mut doc = Map::new();
        doc.insert("tool".to_string(), Value::String(tool.to_string()));
        doc.insert("meta".to_string(), Value::Object(meta));
        doc.insert("checks".to_string(), Value::Array(Vec::new()));
        Self {
            tool: tool.to_string(),
            doc,
            api_key,
        }
    }

    pub fn add(&mut self, name: &str, status: &str, detail: &str, extra: Map<String, Value>) {
        debug_assert!(matches!(status, "PASS" | "FAIL" | "SKIP"));
        let mut entry = Map::new();
        entry.insert("check".to_string(), Value::String(name.to_string()));
        entry.insert("status".to_string(), Value::String(status.to_string()));
        entry.insert("detail".to_string(), Value::String(detail.to_string()));
        for (k, v) in extra {
            entry.insert(k, v);
        }
        let redacted = redact_value(&Value::Object(entry), &self.api_key);
        if let Some(Value::Array(arr)) = self.doc.get_mut("checks") {
            arr.push(redacted);
        }
        let rd = redact_str(detail, &self.api_key);
        if rd.is_empty() {
            println!("{status:4} {name}");
        } else {
            println!("{status:4} {name}: {rd}");
        }
    }

    pub fn check(&mut self, name: &str, ok: bool, detail: &str, extra: Map<String, Value>) -> bool {
        self.add(name, if ok { "PASS" } else { "FAIL" }, detail, extra);
        ok
    }

    pub fn failed(&self) -> Vec<String> {
        self.doc
            .get("checks")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter(|c| c.get("status").and_then(|s| s.as_str()) == Some("FAIL"))
                    .filter_map(|c| c.get("check").and_then(|s| s.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn skipped(&self) -> Vec<String> {
        self.doc
            .get("checks")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter(|c| c.get("status").and_then(|s| s.as_str()) == Some("SKIP"))
                    .filter_map(|c| c.get("check").and_then(|s| s.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn finish(&mut self, path: Option<&Path>, strict_skip: bool) -> i32 {
        let fails = self.failed();
        let skips = self.skipped();
        let bad_count = fails.len() + if strict_skip { skips.len() } else { 0 };
        let result = if bad_count == 0 { "PASS" } else { "FAIL" };
        self.doc
            .insert("result".to_string(), Value::String(result.to_string()));
        self.doc.insert(
            "failed".to_string(),
            Value::Array(fails.iter().cloned().map(Value::String).collect()),
        );
        self.doc.insert(
            "skipped".to_string(),
            Value::Array(skips.iter().cloned().map(Value::String).collect()),
        );

        let total = self
            .doc
            .get("checks")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        if let Some(p) = path {
            if let Some(parent) = p.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let redacted = redact_value(&Value::Object(self.doc.clone()), &self.api_key);
            if let Ok(text) = serde_json::to_string_pretty(&redacted) {
                let _ = fs::write(p, text);
            }
        }
        let skip_s = if skips.is_empty() {
            String::new()
        } else {
            format!(", {} skipped", skips.len())
        };
        let path_s = path
            .map(|p| format!(" -> {}", p.display()))
            .unwrap_or_default();
        println!(
            "{}: {} ({} checks, {} failed{}){}",
            self.tool,
            result,
            total,
            fails.len(),
            skip_s,
            path_s
        );
        if bad_count > 0 {
            1
        } else {
            0
        }
    }
}
