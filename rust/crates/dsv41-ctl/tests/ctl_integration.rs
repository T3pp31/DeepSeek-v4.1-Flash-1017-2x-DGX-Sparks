use dsv41_core::safetensors::{read_header, same_file, write_safetensors};
use half::f16;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn ctl_bin() -> &'static str {
    env!("CARGO_BIN_EXE_dsv41-ctl")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repo root")
        .to_path_buf()
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "dsv41-ctl-test-{tag}-{}-{}",
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

fn run_ctl_with_stdin(args: &[&str], stdin_data: &str) -> (i32, String, String) {
    let mut child = Command::new(ctl_bin())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin_data.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn test_zig_flags_accepts_and_rejects() {
    let caps = json!({
        "serve": {
            "flags": {
                "--backend": {"values": ["cuda"]},
                "--host": {},
                "--port": {},
                "--context": {},
                "--thinking": {},
                "--reasoning-effort": {"values": ["low", "medium", "high", "xhigh", "max"]}
            }
        }
    })
    .to_string();

    let (code, _, err) = run_ctl_with_stdin(
        &[
            "zig-flags",
            "--backend",
            "cuda",
            "--host=0.0.0.0",
            "--port",
            "8888",
            "--thinking",
            "--reasoning-effort",
            "max",
        ],
        &caps,
    );
    assert_eq!(code, 0, "valid flags rejected: {err}");

    let (code_bad, _, err_bad) = run_ctl_with_stdin(
        &["zig-flags", "--reasoning-effort", "ultra", "--bogus-flag"],
        &caps,
    );
    assert_eq!(code_bad, 1);
    assert!(err_bad.contains("ultra"), "{err_bad}");
    assert!(err_bad.contains("--bogus-flag"), "{err_bad}");
}

#[test]
fn test_ablit_stock_dtypes_restore_hardlink_noop_and_tamper() {
    let head = "mtp.2.confidence_head.proj.weight";
    let bias = "mtp.0.ffn.gate.bias";

    let build_case = |root: &Path, tamper: bool| -> (PathBuf, PathBuf) {
        let stock = root.join("stock");
        let ablit = root.join("ablit");
        fs::create_dir_all(&stock).unwrap();
        fs::create_dir_all(&ablit).unwrap();

        let mut meta = BTreeMap::new();
        meta.insert("format".to_string(), "pt".to_string());

        let keep_vals: Vec<f16> = (0..16)
            .map(|i| f16::from_f32((i as f32) * 0.25 - 2.0))
            .collect();
        let keep_bytes: Vec<u8> = keep_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_safetensors(
            &stock.join("model-00001-of-00002.safetensors"),
            Some(&meta),
            &[("layers.0.attn.wo_b.weight", "F16", &[4, 4], &keep_bytes)],
        )
        .unwrap();

        let head_f16: Vec<f16> = (0..8)
            .map(|i| f16::from_f32((i as f32) * 0.125 - 0.5))
            .collect();
        let bias_f16: Vec<f16> = (0..8)
            .map(|i| f16::from_f32((i as f32) * -0.5 + 1.0))
            .collect();
        let head_f16_bytes: Vec<u8> = head_f16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bias_f16_bytes: Vec<u8> = bias_f16.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_safetensors(
            &stock.join("model-00002-of-00002.safetensors"),
            Some(&meta),
            &[
                (bias, "F16", &[8], &bias_f16_bytes),
                (head, "F16", &[1, 8], &head_f16_bytes),
            ],
        )
        .unwrap();

        // Shard 1 hardlinked to stock
        fs::hard_link(
            stock.join("model-00001-of-00002.safetensors"),
            ablit.join("model-00001-of-00002.safetensors"),
        )
        .unwrap();

        // Shard 2 cast F16 -> F32 (and optionally tampered)
        let mut head_f32: Vec<f32> = head_f16.iter().map(|v| v.to_f32()).collect();
        if tamper {
            head_f32[0] += 1.0;
        }
        let bias_f32: Vec<f32> = bias_f16.iter().map(|v| v.to_f32()).collect();
        let head_f32_bytes: Vec<u8> = head_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bias_f32_bytes: Vec<u8> = bias_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_safetensors(
            &ablit.join("model-00002-of-00002.safetensors"),
            Some(&meta),
            &[
                (head, "F32", &[1, 8], &head_f32_bytes),
                (bias, "F32", &[8], &bias_f32_bytes),
            ],
        )
        .unwrap();

        let idx = json!({
            "weight_map": {
                "layers.0.attn.wo_b.weight": "model-00001-of-00002.safetensors",
                head: "model-00002-of-00002.safetensors",
                bias: "model-00002-of-00002.safetensors"
            }
        });
        fs::write(
            ablit.join("model.safetensors.index.json"),
            serde_json::to_string(&idx).unwrap(),
        )
        .unwrap();
        (stock, ablit)
    };

    // Case 1: clean F16 -> F32 cast restored, hardlinked shard untouched, second run is noop
    let tmp = TempDir::new("ablit-ok");
    let (stock, ablit) = build_case(tmp.path(), false);
    let s1 = stock.join("model-00001-of-00002.safetensors");
    let s1_before = fs::read(&s1).unwrap();

    let out = Command::new(ctl_bin())
        .args([
            "ablit-stock-dtypes",
            stock.to_str().unwrap(),
            ablit.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "restore failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let got_path = ablit.join("model-00002-of-00002.safetensors");
    let got_hdr = read_header(&got_path).unwrap();
    assert_eq!(got_hdr.tensors[head].dtype, "F16");
    assert_eq!(got_hdr.tensors[bias].dtype, "F16");
    assert_eq!(
        fs::read(&got_path).unwrap(),
        fs::read(stock.join("model-00002-of-00002.safetensors")).unwrap(),
        "restored shard bytes differ from stock"
    );
    assert_eq!(fs::read(&s1).unwrap(), s1_before);
    assert!(same_file(
        &s1,
        &ablit.join("model-00001-of-00002.safetensors")
    ));

    // Second run: noop
    let out2 = Command::new(ctl_bin())
        .args([
            "ablit-stock-dtypes",
            stock.to_str().unwrap(),
            ablit.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out2.status.success());
    let stdout2 = String::from_utf8_lossy(&out2.stdout);
    assert!(stdout2.contains("already has its stock dtype"), "{stdout2}");

    // Case 2: tampered F32 tensor must be refused and left untouched
    let tmp_bad = TempDir::new("ablit-tamper");
    let (stock_b, ablit_b) = build_case(tmp_bad.path(), true);
    let out_b = Command::new(ctl_bin())
        .args([
            "ablit-stock-dtypes",
            stock_b.to_str().unwrap(),
            ablit_b.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(out_b.status.code(), Some(1));
    let err_b = String::from_utf8_lossy(&out_b.stderr);
    assert!(err_b.contains("not a cast"), "{err_b}");
    let hdr_b = read_header(&ablit_b.join("model-00002-of-00002.safetensors")).unwrap();
    assert_eq!(
        hdr_b.tensors[head].dtype, "F32",
        "refused shard must stay untouched"
    );
}

#[test]
fn test_make_bias_vl_extracts_only_gate_bias_vl_tensors() {
    let tmp = TempDir::new("bias-vl");
    let model = tmp.path().join("model");
    fs::create_dir_all(&model).unwrap();
    let out_file = tmp.path().join("vision/gate_bias_vl.safetensors");

    let b0: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let b1: Vec<u8> = [-1.5f32, 0.25, 0.5, 8.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let other: Vec<u8> = [9.0f32, 9.0].iter().flat_map(|v| v.to_le_bytes()).collect();

    write_safetensors(
        &model.join("shard-1.safetensors"),
        None,
        &[
            ("model.layers.0.ffn.gate.bias_vl", "F32", &[4], &b0),
            ("model.layers.0.ffn.gate.weight", "F32", &[2], &other),
        ],
    )
    .unwrap();
    write_safetensors(
        &model.join("shard-2.safetensors"),
        None,
        &[("model.layers.1.ffn.gate.bias_vl", "F32", &[4], &b1)],
    )
    .unwrap();
    fs::write(
        model.join("model.safetensors.index.json"),
        serde_json::to_string(&json!({
            "weight_map": {
                "model.layers.0.ffn.gate.bias_vl": "shard-1.safetensors",
                "model.layers.0.ffn.gate.weight": "shard-1.safetensors",
                "model.layers.1.ffn.gate.bias_vl": "shard-2.safetensors"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let out = Command::new(ctl_bin())
        .args([
            "make-bias-vl",
            model.to_str().unwrap(),
            out_file.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let hdr = read_header(&out_file).unwrap();
    let data_start = hdr.data_start as usize;
    assert_eq!(hdr.tensors.len(), 2);
    assert!(hdr.tensors.contains_key("model.layers.0.ffn.gate.bias_vl"));
    assert!(hdr.tensors.contains_key("model.layers.1.ffn.gate.bias_vl"));
    let raw = fs::read(&out_file).unwrap();
    let e0 = &hdr.tensors["model.layers.0.ffn.gate.bias_vl"];
    assert_eq!(
        &raw[data_start + e0.data_offsets[0]..data_start + e0.data_offsets[1]],
        b0.as_slice()
    );
    let e1 = &hdr.tensors["model.layers.1.ffn.gate.bias_vl"];
    assert_eq!(
        &raw[data_start + e1.data_offsets[0]..data_start + e1.data_offsets[1]],
        b1.as_slice()
    );
}

#[test]
fn test_ctl_helpers_cidr_key_smoke_health_json_summary_fsops() {
    // CIDR
    let out = Command::new(ctl_bin())
        .args(["cidr-network", "192.168.100.11/24"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "192.168.100.0/24"
    );

    assert!(Command::new(ctl_bin())
        .args(["cidr-contains", "192.168.100.12/24", "192.168.100.0/24"])
        .status()
        .unwrap()
        .success());
    assert!(!Command::new(ctl_bin())
        .args(["cidr-contains", "10.0.0.1/24", "192.168.100.0/24"])
        .status()
        .unwrap()
        .success());

    // gen-api-key
    let k1 = String::from_utf8_lossy(
        &Command::new(ctl_bin())
            .arg("gen-api-key")
            .output()
            .unwrap()
            .stdout,
    )
    .trim()
    .to_string();
    let k2 = String::from_utf8_lossy(
        &Command::new(ctl_bin())
            .arg("gen-api-key")
            .output()
            .unwrap()
            .stdout,
    )
    .trim()
    .to_string();
    assert!(k1.starts_with("sk-tf-") && k1.len() >= 40);
    assert_ne!(k1, k2);

    // check-smoke
    let (c, out, _) = run_ctl_with_stdin(
        &["check-smoke"],
        r#"{"choices":[{"message":{"content":"OK."}}],"usage":{"completion_tokens":2}}"#,
    );
    assert_eq!(c, 0);
    assert!(out.contains("OK.") && out.contains("2 tokens"), "{out}");

    let (c_rep, _, _) = run_ctl_with_stdin(
        &["check-smoke"],
        r#"{"choices":[{"message":{"content":"aaaaaaaaaaaa"}}],"usage":{"completion_tokens":12}}"#,
    );
    assert_eq!(c_rep, 1);

    // health-busy
    let (c, out, _) = run_ctl_with_stdin(&["health-busy"], r#"{"live":{"connections":3}}"#);
    assert_eq!(c, 0);
    assert_eq!(out.trim(), "3");
    let (_, out2, _) = run_ctl_with_stdin(&["health-busy"], r#"{"requests_running":2}"#);
    assert_eq!(out2.trim(), "2");
    let (_, out3, _) = run_ctl_with_stdin(&["health-busy"], "not-json");
    assert_eq!(out3.trim(), "0");

    // json-field (stdin + file)
    let (c, _, _) = run_ctl_with_stdin(
        &["json-field", "-", "status", "--expect-str", "ok"],
        r#"{"status":"ok"}"#,
    );
    assert_eq!(c, 0);
    let (c_bad, _, _) = run_ctl_with_stdin(
        &["json-field", "-", "status", "--expect-str", "ok"],
        r#"{"status":"degraded"}"#,
    );
    assert_eq!(c_bad, 1);

    let tmp = TempDir::new("misc");
    let meta = tmp.path().join("ABLIT_META.json");
    fs::write(&meta, r#"{"n_edited": 104, "token_map_sha256": "abc123"}"#).unwrap();
    assert!(Command::new(ctl_bin())
        .args([
            "json-field",
            meta.to_str().unwrap(),
            "n_edited",
            "--expect-int",
            "104"
        ])
        .status()
        .unwrap()
        .success());
    assert!(!Command::new(ctl_bin())
        .args([
            "json-field",
            meta.to_str().unwrap(),
            "n_edited",
            "--expect-int",
            "103"
        ])
        .status()
        .unwrap()
        .success());

    // summary-json
    let sum_path = tmp.path().join("summary.json");
    let (c, _, _) = run_ctl_with_stdin(
        &["summary-json", sum_path.to_str().unwrap()],
        "off-exact 0\nspill-main 1\nshard-dry dry-run\n",
    );
    assert_eq!(c, 0);
    let doc: Value = serde_json::from_str(&fs::read_to_string(&sum_path).unwrap()).unwrap();
    assert_eq!(doc["suites"]["off-exact"], 0);
    assert_eq!(doc["suites"]["spill-main"], 1);
    assert_eq!(doc["suites"]["shard-dry"], "dry-run");
    assert_eq!(doc["failed"], json!(["spill-main"]));

    // fsops (list + corrupt + truncate + remove + statvfs)
    let fdir = tmp.path().join("spill");
    fs::create_dir_all(fdir.join("blocks")).unwrap();
    fs::write(fdir.join("blocks/a.blk"), b"0123456789abcdef").unwrap();
    let list_arg = json!({"op": "list", "root": fdir.to_str().unwrap()}).to_string();
    let list_out = Command::new(ctl_bin())
        .args(["fsops", &list_arg])
        .output()
        .unwrap();
    assert!(list_out.status.success());
    let listed: Value = serde_json::from_slice(&list_out.stdout).unwrap();
    assert_eq!(listed["blocks/a.blk"][0], 16);
}

#[test]
fn test_checkpoint_pins_and_mixed_pack_patches_and_systemd_unit() {
    let root = repo_root();

    // Port of scripts/test_checkpoint_pin.py
    let cfg = fs::read_to_string(root.join("scripts/config.sh")).unwrap();
    assert!(cfg.contains(r#"EAQ_V1_1="${EAQ_V1_1:-0}""#));
    assert!(cfg.contains("_default_id=Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw"));
    assert!(cfg.contains("_default_rev=45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b"));
    assert!(cfg.contains("_default_id=Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw"));
    assert!(cfg.contains("_default_rev=aee640e2afba93be683bb5b9b420115fe7fbac20"));

    // Port of scripts/test_mixed_pack_patch.py + new patches 0069 & 0070
    let checks: &[(&str, &[&str])] = &[
        (
            "zig/patches/0024-mixed-pack.patch",
            &[
                "mixed pack: {d} float linears",
                "plain: u64 = 0",
                "plain_tmp",
                "e.blas.linearBf16",
                "groupedRotated",
            ],
        ),
        (
            "zig/patches/0025-widen-bf16.patch",
            &[
                "widenBFloat",
                "from == .bf16 and k.dtype.? == .f32",
                "a bf16 tensor read as f32 widens exactly",
            ],
        ),
        (
            "zig/patches/0026-plain-rows.patch",
            &[
                "plainRows",
                "TF_DS_PLAIN_ROWS",
                "e.plain_rows <= m",
                "e.round_rows",
            ],
        ),
        (
            "zig/patches-inputs/0001-mixed-pack-rank-cache.patch",
            &["wantsShards", "plain = true", "plainKey", "t.shape[1]"],
        ),
        (
            "zig/patches-inputs/0002-widen-bf16.patch",
            &[
                "widenBFloat",
                "from == .bf16 and k.dtype.? == .f32",
                "a bf16 tensor read as f32 widens exactly",
            ],
        ),
        ("zig/patches/0069-max-reasoning-effort.patch", &["\"max\""]),
        (
            "zig/patches/0070-native-image-decoder.patch",
            &["TENSORFOLD_IMAGE_DECODER", "pil.zig"],
        ),
    ];
    for (rel, needles) in checks {
        let text = fs::read_to_string(root.join(rel)).unwrap();
        for needle in *needles {
            assert!(text.contains(needle), "{rel} missing marker {needle:?}");
        }
    }

    // Port of scripts/test_systemd_unit.py (offline structure + optional systemd-analyze)
    let unit = fs::read_to_string(root.join("systemd/dsv41-tensorfold.service")).unwrap();
    assert!(unit.contains("ExecStart=/usr/bin/bash ./start.sh"));
    assert!(unit.contains("ExecStop=/usr/bin/bash ./stop.sh"));
    assert!(unit.contains("Restart=on-failure"));
    assert!(unit.contains("FOREGROUND=1"));
    assert!(unit.contains("PREPARE=0"));

    // Port of scripts/test_worker_files.py call-site checks
    let start = fs::read_to_string(root.join("start.sh")).unwrap();
    let check_pos = start
        .find("worker_files_check 1")
        .expect("worker_files_check in start.sh");
    assert!(start.find("./scripts/prepare.sh\n").unwrap() < check_pos);
    assert!(
        check_pos
            < start
                .find("if exists_here || exists_worker 1; then ./stop.sh; fi")
                .unwrap()
    );
    let prepare = fs::read_to_string(root.join("scripts/prepare.sh")).unwrap();
    assert!(prepare.contains("ensure_nfs_volume 1\nworker_files_check 1\n"));
}

#[test]
fn test_round_rows_and_pool_settings_and_worker_rank_cache_cleanup() {
    let root = repo_root();
    let start = fs::read_to_string(root.join("start.sh")).unwrap();

    // Port of scripts/test_round_rows.py
    let rr_line = start
        .lines()
        .find(|l| l.starts_with(r#"[[ -z "$ROUND_ROWS" ]] || {"#))
        .expect("ROUND_ROWS line in start.sh");
    for val in ["", "16", "17", "32", "48", "0", "garbage"] {
        let st = Command::new("bash")
            .args([
                "-c",
                &format!("die() {{ exit 1; }}; ROUND_ROWS=$1; {rr_line}"),
                "test",
                val,
            ])
            .status()
            .unwrap();
        assert_eq!(
            st.success(),
            val.is_empty() || val == "16",
            "ROUND_ROWS={val:?}"
        );
    }

    // Port of tools/test_pool_settings.py
    let eng_start = start.find("ENGINE_ENV=(").unwrap();
    let eng_end = start.find("envs() {").unwrap();
    let engine_block = &start[eng_start..eng_end];
    let follow_line = start.lines().find(|l| l.starts_with("FOLLOW=\"")).unwrap();
    for pool in ["", "131072"] {
        let out = Command::new("bash")
            .args([
                "-euc",
                &format!("{engine_block}\n{follow_line}\nprintf \"%s\\n\" \"${{ENGINE_ENV[@]}}\" \"$FOLLOW\""),
            ])
            .env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("CONTEXT", "65536")
            .env("POOL", pool)
            .env("REPLAY_FLOOR", "0")
            .env("PARALLEL", "4")
            .env("HF_HUB_OFFLINE", "1")
            .env("ROUND_ROWS", "")
            .env("PREFILL_INTERLEAVE_MS", "0")
            .env("RDMA_IDLE_S", "2")
            .env("RDMA_NAP_US", "500")
            .env("DRAFTS", "1")
            .env("GRAPHS", "1")
            .env("GRAPH_CAP", "128")
            .env("SIDE", "0")
            .env("PREFETCH", "0")
            .env("ENGRAM_AIO", "0")
            .env("TF_DS_POOL", "999999")
            .env("TF_DS_WINDOW", "1")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let lines: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(String::from)
            .collect();
        assert!(!lines.iter().any(|l| l == "TF_DS_POOL=999999"));
        assert!(!lines.iter().any(|l| l == "TF_DS_WINDOW=1"));
        if pool.is_empty() {
            assert!(!lines.iter().any(|l| l.starts_with("TF_DS_POOL=")));
            assert!(lines
                .last()
                .unwrap()
                .starts_with("--pool 65536 --streams 4"));
        } else {
            assert!(lines.iter().any(|l| l == "TF_DS_POOL=131072"));
            assert!(lines.iter().any(|l| l == "TF_DS_WINDOW=65536"));
            assert!(lines
                .last()
                .unwrap()
                .starts_with("--pool 131072 --streams 4"));
        }
    }

    // Port of scripts/test_worker_rank_cache.py (bash -O failglob)
    let prepare = fs::read_to_string(root.join("scripts/prepare.sh")).unwrap();
    let matches: Vec<&str> = prepare
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with(r#"worker 1 "mkdir -p $WORKER_CACHE_DIR &&"#))
        .collect();
    assert_eq!(matches.len(), 1);
    let inner = &matches[0][r#"worker 1 ""#.len()..matches[0].len() - 1];

    let tmp = TempDir::new("rank-cache");
    let cache = tmp.path().join("rank-cache");
    let cmd = inner.replace("$WORKER_CACHE_DIR", cache.to_str().unwrap());
    let out1 = Command::new("bash")
        .args(["-O", "failglob", "-c", &cmd])
        .output()
        .unwrap();
    assert!(
        out1.status.success(),
        "{}",
        String::from_utf8_lossy(&out1.stderr)
    );
    assert!(cache.is_dir());

    for name in ["rank1of2-old.bin", ".rank1of2.partial", "keep.bin"] {
        fs::write(cache.join(name), "fixture").unwrap();
    }
    let out2 = Command::new("bash")
        .args(["-O", "failglob", "-c", &cmd])
        .output()
        .unwrap();
    assert!(
        out2.status.success(),
        "{}",
        String::from_utf8_lossy(&out2.stderr)
    );
    let left: Vec<String> = fs::read_dir(&cache)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    assert_eq!(left, vec!["keep.bin"]);
}
