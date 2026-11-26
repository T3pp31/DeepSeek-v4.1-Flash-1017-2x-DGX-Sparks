//! `/health` JSON and Prometheus `/metrics` counter flattening & aggregation
//! matching `tools/checklib.py` (`SPILL_FIELDS`, `KV_FIELDS`, `KV_PART_FIELDS`).

use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

pub const SPILL_FIELDS: &[&str] = &[
    "points",
    "blocks",
    "stored_bytes",
    "cap_bytes",
    "disk_free_bytes",
    "staging_bytes",
    "staging_peak",
    "writes_pending",
    "writes_queued",
    "writes_committed",
    "writes_cancelled",
    "writes_cancelled_guard",
    "writes_cancelled_staging",
    "writes_cancelled_shutdown",
    "writes_failed",
    "forced_staging",
    "no_space",
    "dedup_blocks",
    "dedup_bytes",
    "written_bytes",
    "presence_refused",
    "compatible_misses",
    "crc_errors",
    "size_errors",
    "reads",
    "read_bytes",
    "read_failed",
    "evicted_turns",
    "evicted_forks",
    "evicted_conversations",
    "evicted_shared",
    "quota_trims",
    "recovered_points",
    "dropped_temporaries",
    "pictures_kept",
    "pictures_refused",
];

pub const SPILL_GAUGES: &[&str] = &[
    "points",
    "blocks",
    "stored_bytes",
    "cap_bytes",
    "disk_free_bytes",
    "staging_bytes",
    "staging_peak",
    "writes_pending",
];

pub const SPILL_MIN: &[&str] = &["disk_free_bytes", "cap_bytes"];
pub const SPILL_MAX: &[&str] = &["staging_peak"];

pub const KV_FIELDS: &[&str] = &[
    "parts",
    "part_rows",
    "carries",
    "carry_rescued",
    "carry_pressure",
    "carry_others",
    "carry_paused",
    "moves",
    "moved_tokens",
    "moved_bytes",
    "transfers_failed",
    "staged",
    "staled",
    "twins_made",
    "twins_revoked",
    "refused_pinned",
    "refused_twin",
    "refused_loading",
    "refused_partial",
    "refused_no_target",
    "refused_small",
    "no_part",
    "plans",
    "checks",
    "prompt_fetches",
    "prompt_rows",
    "prompt_picks",
    "prompt_bytes",
    "prompt_syncs",
    "fetch_rounds",
    "owner_rounds",
    "copies",
    "copy_bytes",
];

pub const KV_GAUGES: &[&str] = &["parts", "part_rows"];
pub const KV_PART_FIELDS: &[&str] = &[
    "free",
    "live_rows",
    "kept_rows",
    "twin_rows",
    "hold_rows",
    "live",
];

pub type Counters = BTreeMap<String, f64>;

pub fn kv_parts(c: &Counters) -> Vec<BTreeMap<String, f64>> {
    let mut out: BTreeMap<usize, BTreeMap<String, f64>> = BTreeMap::new();
    for (k, &v) in c {
        if let Some(rest) = k.strip_prefix("kv_shard.part") {
            if let Some((p_str, row)) = rest.split_once('.') {
                if !row.contains('.')
                    && !p_str.is_empty()
                    && p_str.bytes().all(|b| b.is_ascii_digit())
                    && KV_PART_FIELDS.contains(&row)
                {
                    if let Ok(p) = p_str.parse::<usize>() {
                        out.entry(p).or_default().insert(row.to_string(), v);
                    }
                }
            }
        }
    }
    out.into_values().collect()
}

pub fn kv_pool(c: &Counters) -> Option<i64> {
    let parts = *c.get("kv_shard.parts")?;
    let rows = *c.get("kv_shard.part_rows")?;
    if parts > 0.0 && rows > 0.0 {
        Some((parts * rows) as i64)
    } else {
        None
    }
}

pub fn aggregate_spill(out: &mut Counters) {
    let mut per: HashMap<String, Vec<f64>> = HashMap::new();
    for (k, &v) in out.iter() {
        if let Some(rest) = k.strip_prefix("spill.") {
            if let Some((name, r)) = rest.rsplit_once(".rank") {
                if !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()) {
                    per.entry(name.to_string()).or_default().push(v);
                }
            }
        }
    }
    for (name, vs) in per {
        let key = format!("spill.{name}");
        if !out.contains_key(&key) && !vs.is_empty() {
            let agg = if SPILL_MIN.contains(&name.as_str()) {
                vs.iter().copied().fold(f64::INFINITY, f64::min)
            } else if SPILL_MAX.contains(&name.as_str()) {
                vs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            } else {
                vs.iter().sum()
            };
            out.insert(key, agg);
        }
    }
}

pub fn rank_values(c: &Counters, name: &str) -> BTreeMap<usize, f64> {
    let pre = format!("spill.{name}.rank");
    let mut out = BTreeMap::new();
    for (k, &v) in c {
        if let Some(rest) = k.strip_prefix(&pre) {
            if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                if let Ok(r) = rest.parse::<usize>() {
                    out.insert(r, v);
                }
            }
        }
    }
    out
}

pub fn rank_deltas(before: &Counters, after: &Counters, name: &str) -> BTreeMap<usize, f64> {
    rank_values(after, name)
        .into_iter()
        .map(|(r, v)| {
            let b = before
                .get(&format!("spill.{name}.rank{r}"))
                .copied()
                .unwrap_or(0.0);
            (r, v - b)
        })
        .collect()
}

pub fn delta(before: &Counters, after: &Counters, key: &str) -> Option<f64> {
    let a = *after.get(key)?;
    let b = before.get(key).copied().unwrap_or(0.0);
    Some(a - b)
}

pub fn deltas<'a>(
    before: &Counters,
    after: &Counters,
    keys: impl IntoIterator<Item = &'a str>,
) -> BTreeMap<String, Option<f64>> {
    keys.into_iter()
        .map(|k| (k.to_string(), delta(before, after, k)))
        .collect()
}

fn flatten(prefix: &str, obj: &serde_json::Map<String, Value>, out: &mut Counters) {
    for (k, v) in obj {
        let key = format!("{prefix}{k}");
        match v {
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    out.insert(key, f);
                }
            }
            Value::Object(sub) => flatten(&format!("{key}."), sub, out),
            _ => {}
        }
    }
}

pub fn parse_metrics(text: &str) -> Counters {
    let mut out = Counters::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((raw_name, raw_val)) = line.rsplit_once(' ') {
            if let Ok(v) = raw_val.parse::<f64>() {
                let name = raw_name
                    .strip_prefix("tensorfold:")
                    .unwrap_or(raw_name)
                    .to_string();
                out.insert(name, v);
            }
        }
    }
    out
}

pub fn metrics_counters(m: &Counters) -> Counters {
    let mut out = Counters::new();
    for (raw_k, &v) in m {
        let (k, labels) = if raw_k.ends_with('}') {
            if let Some((base, lab_end)) = raw_k.split_once('{') {
                let lab = &lab_end[..lab_end.len() - 1];
                let mut lmap = HashMap::new();
                for part in lab.split(',') {
                    if let Some((n, val)) = part.split_once('=') {
                        lmap.insert(n.trim(), val.trim().trim_matches('"'));
                    }
                }
                (base, lmap)
            } else {
                (raw_k.as_str(), HashMap::new())
            }
        } else {
            (raw_k.as_str(), HashMap::new())
        };

        if !labels.is_empty() {
            if let Some(suffix) = k.strip_prefix("spill_") {
                if labels.len() == 1 {
                    if let Some(&r) = labels.get("rank") {
                        if !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()) {
                            out.insert(format!("spill.{suffix}.rank{r}"), v);
                        }
                    }
                }
            } else if let Some(suffix) = k.strip_prefix("kv_shard_part_") {
                if labels.len() == 1 {
                    if let Some(&p) = labels.get("part") {
                        if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) {
                            out.insert(format!("kv_shard.part{p}.{suffix}"), v);
                        }
                    }
                }
            }
            continue;
        }

        if let Some(suffix) = k.strip_prefix("prompt_cache_") {
            out.insert(format!("prompt_cache.{suffix}"), v);
        } else if let Some(suffix) = k.strip_prefix("spill_") {
            out.insert(format!("spill.{suffix}"), v);
        } else if let Some(suffix) = k.strip_prefix("kv_shard_") {
            out.insert(format!("kv_shard.{suffix}"), v);
        } else if k == "chats_continued" || k == "chats_continued_prefill_tokens" {
            out.insert(k.to_string(), v);
        }
    }
    aggregate_spill(&mut out);
    out
}

pub fn health_counters(h: &Value) -> Counters {
    let mut out = Counters::new();
    let Some(obj) = h.as_object() else {
        return out;
    };
    if obj.contains_key("error") {
        return out;
    }
    flatten("", obj, &mut out);

    if let Some(pc) = obj.get("prompt_cache").and_then(|v| v.as_object()) {
        if let Some(sp) = pc.get("spill") {
            let ranks: Vec<&serde_json::Map<String, Value>> = match sp {
                Value::Array(arr) => arr.iter().filter_map(|v| v.as_object()).collect(),
                Value::Object(m) => vec![m],
                _ => vec![],
            };
            for (i, o) in ranks.into_iter().enumerate() {
                let r = o.get("rank").and_then(|v| v.as_u64()).unwrap_or(i as u64);
                for (k, v) in o {
                    if k != "rank" {
                        if let Some(f) = v.as_f64() {
                            out.insert(format!("spill.{k}.rank{r}"), f);
                        }
                    }
                }
            }
        }
    }
    out.retain(|k, _| !k.starts_with("prompt_cache.spill."));

    if let Some(ks) = obj.get("kv_shard").and_then(|v| v.as_object()) {
        if let Some(pp) = ks.get("per_part").and_then(|v| v.as_array()) {
            for (i, v) in pp.iter().enumerate() {
                if let Some(o) = v.as_object() {
                    let p = o.get("part").and_then(|v| v.as_u64()).unwrap_or(i as u64);
                    for (k, val) in o {
                        if k != "part" {
                            if let Some(f) = val.as_f64() {
                                out.insert(format!("kv_shard.part{p}.{k}"), f);
                            }
                        }
                    }
                }
            }
        }
    }
    aggregate_spill(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_health_and_metrics_spill_and_kv() {
        let h = json!({
            "status": "ok",
            "prompt_cache": {
                "disk_lookups": 3,
                "spill": [
                    {"rank": 0, "writes_committed": 2, "disk_free_bytes": 100},
                    {"rank": 1, "writes_committed": 3, "disk_free_bytes": 50}
                ]
            },
            "kv_shard": {
                "parts": 2,
                "part_rows": 1050624,
                "carries": 3,
                "per_part": [
                    {"part": 0, "free": 10, "live": 1},
                    {"part": 1, "free": 20, "live": 0}
                ]
            }
        });
        let hc = health_counters(&h);
        assert_eq!(hc.get("spill.writes_committed"), Some(&5.0));
        assert_eq!(hc.get("spill.disk_free_bytes"), Some(&50.0));
        assert_eq!(hc.get("prompt_cache.disk_lookups"), Some(&3.0));
        assert_eq!(kv_pool(&hc), Some(2_101_248));
        assert_eq!(kv_parts(&hc).len(), 2);

        let m = "\
# HELP tensorfold:spill_writes_committed x
tensorfold:spill_writes_committed{rank=\"0\"} 2
tensorfold:spill_writes_committed{rank=\"1\"} 3
tensorfold:spill_disk_free_bytes{rank=\"0\"} 100
tensorfold:spill_disk_free_bytes{rank=\"1\"} 50
tensorfold:prompt_cache_disk_lookups 3
tensorfold:kv_shard_parts 2
tensorfold:kv_shard_part_rows 1050624
tensorfold:kv_shard_part_free{part=\"0\"} 10
tensorfold:kv_shard_part_free{part=\"1\"} 20
";
        let mc = metrics_counters(&parse_metrics(m));
        assert_eq!(mc.get("spill.writes_committed"), Some(&5.0));
        assert_eq!(mc.get("spill.disk_free_bytes"), Some(&50.0));
        assert_eq!(kv_pool(&mc), Some(2_101_248));
    }
}
