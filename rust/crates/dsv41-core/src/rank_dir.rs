//! Local and remote (over SSH / optional sudo) spill directory filesystem operations.

use crate::SPILL_MARKER;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

pub fn kind_of(path: &str) -> &'static str {
    if path == SPILL_MARKER {
        return "marker";
    }
    for (ext, kind) in [
        (".blk", "block"),
        (".st", "small"),
        (".man", "manifest"),
        (".ids", "ids"),
    ] {
        if path.ends_with(ext) {
            return kind;
        }
    }
    let base = path.rsplit('/').next().unwrap_or(path);
    if path.contains(".tmp") || base.starts_with('.') {
        "temp"
    } else {
        "other"
    }
}

fn walk_files(root: &Path, dir: &Path, out: &mut BTreeMap<String, Value>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            walk_files(root, &p, out);
        } else if let Ok(md) = fs::metadata(&p) {
            if let Ok(rel) = p.strip_prefix(root) {
                let mtime = md
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                out.insert(
                    rel.to_string_lossy().replace('\\', "/"),
                    json!([md.len(), mtime]),
                );
            }
        }
    }
}

pub fn execute_fsop(req: &Value) -> Result<Value, String> {
    let op = req.get("op").and_then(|v| v.as_str()).ok_or("missing op")?;
    let root_s = req
        .get("root")
        .and_then(|v| v.as_str())
        .ok_or("missing root")?;
    let root = Path::new(root_s);

    match op {
        "list" => {
            let mut out = BTreeMap::new();
            walk_files(root, root, &mut out);
            let map: Map<String, Value> = out.into_iter().collect();
            Ok(Value::Object(map))
        }
        "corrupt" => {
            let rel = req
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or("missing path")?;
            let p = root.join(rel);
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&p)
                .map_err(|e| format!("open {}: {e}", p.display()))?;
            let n = f.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
            let at = req
                .get("at")
                .and_then(|v| v.as_u64())
                .unwrap_or(n / 2)
                .min(n.saturating_sub(1));
            f.seek(SeekFrom::Start(at)).map_err(|e| e.to_string())?;
            let mut b = [0u8; 1];
            let read_n = f.read(&mut b).map_err(|e| e.to_string())?;
            let orig = if read_n > 0 { b[0] } else { 0 };
            f.seek(SeekFrom::Start(at)).map_err(|e| e.to_string())?;
            f.write_all(&[orig ^ 0xFF]).map_err(|e| e.to_string())?;
            f.sync_all().map_err(|e| e.to_string())?;
            Ok(json!({"path": rel, "size": n, "at": at}))
        }
        "truncate" => {
            let rel = req
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or("missing path")?;
            let p = root.join(rel);
            let n = fs::metadata(&p).map_err(|e| e.to_string())?.len();
            let to = req.get("to").and_then(|v| v.as_u64()).unwrap_or(n / 2);
            let f = OpenOptions::new()
                .write(true)
                .open(&p)
                .map_err(|e| e.to_string())?;
            f.set_len(to).map_err(|e| e.to_string())?;
            Ok(json!({"path": rel, "size": n, "to": to}))
        }
        "remove" => {
            let rel = req
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or("missing path")?;
            let p = root.join(rel);
            fs::remove_file(&p).map_err(|e| e.to_string())?;
            Ok(json!({"path": rel}))
        }
        "statvfs" => {
            #[cfg(unix)]
            {
                use std::ffi::CString;
                let cpath = CString::new(root_s).map_err(|e| e.to_string())?;
                let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
                let rc = unsafe { libc::statvfs(cpath.as_ptr(), &mut st) };
                if rc != 0 {
                    return Err(std::io::Error::last_os_error().to_string());
                }
                let frsize = st.f_frsize as u64;
                Ok(json!({
                    "free": (st.f_bavail as u64).saturating_mul(frsize),
                    "total": (st.f_blocks as u64).saturating_mul(frsize),
                }))
            }
            #[cfg(not(unix))]
            {
                Err("statvfs unsupported on non-unix".to_string())
            }
        }
        other => Err(format!("unknown op {other}")),
    }
}

#[derive(Clone, Debug)]
pub struct RankDir {
    pub root: PathBuf,
    pub shell: Vec<String>,
    pub sudo: bool,
    pub rank: usize,
}

impl RankDir {
    pub fn new(root: &str, shell: &str, sudo: bool, rank: usize) -> Self {
        Self {
            root: PathBuf::from(root),
            shell: shell.split_whitespace().map(String::from).collect(),
            sudo,
            rank,
        }
    }

    pub fn run(&self, op: &str, mut extra: Map<String, Value>) -> Result<Value, String> {
        extra.insert("op".to_string(), Value::String(op.to_string()));
        extra.insert(
            "root".to_string(),
            Value::String(self.root.to_string_lossy().into_owned()),
        );
        let req = Value::Object(extra);
        if self.shell.is_empty() && !self.sudo {
            return execute_fsop(&req);
        }
        let arg = serde_json::to_string(&req).map_err(|e| e.to_string())?;
        let ctl_bin = std::env::var("DSV41_CTL_BIN").unwrap_or_else(|_| "dsv41-ctl".to_string());
        let mut cmd = if !self.shell.is_empty() {
            let mut c = Command::new(&self.shell[0]);
            for a in &self.shell[1..] {
                c.arg(a);
            }
            let quoted_arg = shell_quote(&arg);
            let mut remote = String::new();
            if self.sudo {
                remote.push_str("sudo -n ");
            }
            remote.push_str(&format!(
                "PATH=\"$HOME/.cache/dsv41-tensorfold/bin:$HOME/.local/bin:$PATH\" {} fsops {}",
                shell_quote(&ctl_bin),
                quoted_arg
            ));
            c.arg(remote);
            c
        } else {
            let mut c = Command::new("sudo");
            c.arg("-n").arg(&ctl_bin).arg("fsops").arg(&arg);
            c
        };
        let out = cmd
            .output()
            .map_err(|e| format!("rank {} {op} spawn failed: {e}", self.rank))?;
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            return Err(format!(
                "rank {} {op} failed ({:?}): {}",
                self.rank,
                out.status.code(),
                err.trim()
            ));
        }
        serde_json::from_slice(&out.stdout)
            .map_err(|e| format!("rank {} {op} bad json: {e}", self.rank))
    }

    pub fn list(&self) -> Result<BTreeMap<String, Value>, String> {
        let v = self.run("list", Map::new())?;
        let obj = v.as_object().ok_or("list did not return object")?;
        Ok(obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }

    pub fn new_files(
        &self,
        before: &BTreeMap<String, Value>,
    ) -> Result<BTreeMap<String, Value>, String> {
        let now = self.list()?;
        Ok(now
            .into_iter()
            .filter(|(k, _)| !before.contains_key(k))
            .collect())
    }
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kind_of() {
        assert_eq!(kind_of("blocks/ab12.blk"), "block");
        assert_eq!(kind_of("points/k.st"), "small");
        assert_eq!(kind_of("points/k.man"), "manifest");
        assert_eq!(kind_of("points/k.ids"), "ids");
        assert_eq!(kind_of("points/k.man.tmp"), "temp");
        assert_eq!(kind_of("points/.k"), "temp");
        assert_eq!(kind_of(".lost"), "temp");
        assert_eq!(kind_of("LOCK"), "other");
        assert_eq!(kind_of(SPILL_MARKER), "marker");
        assert_eq!(kind_of(&format!("points/{SPILL_MARKER}")), "temp");
    }
}
