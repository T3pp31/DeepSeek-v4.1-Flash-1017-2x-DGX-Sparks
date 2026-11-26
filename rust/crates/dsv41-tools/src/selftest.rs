//! Offline self-test harness (`dsv41-tools selftest`) replacing `tools/selftest.py`.
//! Exercises `exact`, `spill-check`, `kv-shard-check`, `prefix-fanout`, `pair-check`, and `bench`
//! against `dsv41-tools fake-server` including positive and `--mutate` negative fault detection.

use clap::Args;
use dsv41_core::client::Server;
use dsv41_core::metrics::{
    health_counters, kv_parts, kv_pool, metrics_counters, parse_metrics, KV_FIELDS, KV_PART_FIELDS,
    SPILL_FIELDS,
};
use dsv41_core::rank_dir::kind_of;
use dsv41_core::SPILL_MARKER;
use serde_json::json;
use std::env;
use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Args)]
pub struct SelftestArgs {
    #[arg(long)]
    pub only: Option<String>,
    #[arg(long)]
    pub keep: bool,
}

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    l.local_addr().unwrap().port()
}

struct FakeProc {
    port: u16,
    url: String,
    args: Vec<String>,
    child: Option<Child>,
    exe: PathBuf,
    log: PathBuf,
}

impl FakeProc {
    async fn start(exe: &Path, work: &Path, name: &str, args: &[&str]) -> Self {
        let port = free_port();
        let url = format!("http://127.0.0.1:{port}");
        let log = work.join(format!("fake-{name}.log"));
        let mut f = Self {
            port,
            url,
            args: args.iter().map(|s| s.to_string()).collect(),
            child: None,
            exe: exe.to_path_buf(),
            log,
        };
        f.spawn().await;
        f
    }

    async fn spawn(&mut self) {
        let lf = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log)
            .unwrap();
        let lf2 = lf.try_clone().unwrap();
        let mut cmd = Command::new(&self.exe);
        cmd.arg("fake-server")
            .arg("--port")
            .arg(self.port.to_string())
            .args(&self.args)
            .stdout(Stdio::from(lf))
            .stderr(Stdio::from(lf2));
        self.child = Some(cmd.spawn().expect("spawn fake-server"));
        let srv = Server::new(
            &self.url,
            "DeepSeek-V4.1-Flash-1017",
            5.0,
            1234,
            String::new(),
        );
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if srv.alive().await {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("fake-server failed to start (log: {})", self.log.display());
    }

    fn stop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }

    async fn restart(&mut self) {
        self.stop();
        self.spawn().await;
    }

    fn pid(&self) -> u32 {
        self.child.as_ref().map(|c| c.id()).unwrap_or(0)
    }
}

impl Drop for FakeProc {
    fn drop(&mut self) {
        self.stop();
    }
}

struct Suite {
    exe: PathBuf,
    work: PathBuf,
    results: Vec<(String, bool)>,
}

impl Suite {
    fn record(&mut self, label: &str, ok: bool, detail: &str) {
        self.results.push((label.to_string(), ok));
        println!("{}  {label}: {detail}", if ok { "PASS" } else { "FAIL" });
    }

    fn run_tool(&mut self, label: &str, args: &[&str], expect_ok: bool) -> bool {
        let safe = label.replace([' ', '/', ':'], "_");
        let log = self.work.join(format!("{safe}.log"));
        let t0 = Instant::now();
        let out = Command::new(&self.exe)
            .args(args)
            .output()
            .expect("spawn tool");
        let mut combined = out.stdout;
        combined.extend_from_slice(&out.stderr);
        let _ = fs::write(&log, &combined);
        let ok = out.status.success() == expect_ok;
        self.results.push((label.to_string(), ok));
        println!(
            "{}  {label}: exit {:?} (expected {}), {:.2}s",
            if ok { "PASS" } else { "FAIL" },
            out.status.code(),
            if expect_ok { "0" } else { "non-zero" },
            t0.elapsed().as_secs_f64()
        );
        if !ok {
            let s = String::from_utf8_lossy(&combined);
            for line in s
                .lines()
                .rev()
                .take(15)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
            {
                println!("      {line}");
            }
        }
        ok
    }
}

pub async fn run(a: SelftestArgs) -> i32 {
    let exe = env::current_exe().expect("current_exe");
    let work = env::temp_dir().join(format!("dsv41-selftest-{}", std::process::id()));
    let _ = fs::remove_dir_all(&work);
    fs::create_dir_all(&work).unwrap();
    let mut st = Suite {
        exe: exe.clone(),
        work: work.clone(),
        results: Vec::new(),
    };

    let selected: Vec<String> = match &a.only {
        Some(s) => s
            .split(',')
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect(),
        None => vec!["counters", "exact", "spill", "kv", "prefix", "pair"]
            .into_iter()
            .map(String::from)
            .collect(),
    };
    let fast = ["--prefill-us-per-token", "3", "--decode-ms-per-token", "1"];

    for name in selected {
        match name.as_str() {
            "counters" => {
                // kind_of
                let cases = [
                    ("blocks/ab12.blk", "block"),
                    ("points/k.st", "small"),
                    ("points/k.man", "manifest"),
                    ("points/k.ids", "ids"),
                    ("points/k.man.tmp", "temp"),
                    ("points/.k", "temp"),
                    (".lost", "temp"),
                    ("LOCK", "other"),
                    (SPILL_MARKER, "marker"),
                ];
                let ok = cases.iter().all(|&(p, k)| kind_of(p) == k);
                st.record("checklib: kind_of", ok, &format!("{} cases", cases.len()));

                let f = FakeProc::start(
                    &exe,
                    &work,
                    "cnt-kv",
                    &[
                        "--kv-shard",
                        "--pool",
                        "131072",
                        "--prefill-us-per-token",
                        "3",
                        "--decode-ms-per-token",
                        "1",
                    ],
                )
                .await;
                let srv = Server::new(
                    &f.url,
                    "DeepSeek-V4.1-Flash-1017",
                    15.0,
                    1234,
                    String::new(),
                );
                let c = srv.counters().await;
                let mc = metrics_counters(&parse_metrics(
                    &srv.get_text("/metrics", Duration::from_secs(5))
                        .await
                        .unwrap_or_default(),
                ));
                let ok_kv = KV_FIELDS
                    .iter()
                    .all(|k| c.contains_key(&format!("kv_shard.{k}")))
                    && kv_pool(&c) == Some(131_072)
                    && kv_parts(&c).len() == 2
                    && KV_PART_FIELDS
                        .iter()
                        .all(|r| mc.contains_key(&format!("kv_shard.part0.{r}")));
                st.record(
                    "counters: kv_shard /health + /metrics",
                    ok_kv,
                    "all KV fields present",
                );
                drop(f);

                let d0 = work.join("cnt-r0");
                let d1 = work.join("cnt-r1");
                let f2 = FakeProc::start(
                    &exe,
                    &work,
                    "cnt-spill",
                    &[
                        "--spill-dir",
                        d0.to_str().unwrap(),
                        "--spill-dir-rank1",
                        d1.to_str().unwrap(),
                        "--prefill-us-per-token",
                        "3",
                        "--decode-ms-per-token",
                        "1",
                    ],
                )
                .await;
                let srv2 = Server::new(
                    &f2.url,
                    "DeepSeek-V4.1-Flash-1017",
                    15.0,
                    1234,
                    String::new(),
                );
                let c2 = srv2.counters().await;
                let h2 = health_counters(&srv2.health().await);
                let ok_sp = SPILL_FIELDS.iter().all(|k| {
                    c2.contains_key(&format!("spill.{k}"))
                        && h2.contains_key(&format!("spill.{k}.rank0"))
                });
                st.record(
                    "counters: spill /health + /metrics",
                    ok_sp,
                    "all spill fields present",
                );
            }
            "exact" => {
                let base = FakeProc::start(&exe, &work, "exact-base", &fast).await;
                let cand = FakeProc::start(&exe, &work, "exact-cand", &fast).await;
                let bad = FakeProc::start(
                    &exe,
                    &work,
                    "exact-bad",
                    &[
                        "--mutate",
                        "conc-wrong",
                        "--prefill-us-per-token",
                        "3",
                        "--decode-ms-per-token",
                        "1",
                    ],
                )
                .await;
                let weak = FakeProc::start(
                    &exe,
                    &work,
                    "exact-weak",
                    &[
                        "--mutate",
                        "no-token-sha",
                        "--prefill-us-per-token",
                        "3",
                        "--decode-ms-per-token",
                        "1",
                    ],
                )
                .await;
                let jb = work.join("x-base.json");
                let jc = work.join("x-cand.json");
                let j4 = work.join("x-cand4.json");
                let jbad = work.join("x-bad.json");
                let jw = work.join("x-weak.json");
                st.run_tool(
                    "exact: record baseline (8 streams)",
                    &[
                        "exact",
                        "record",
                        "--url",
                        &base.url,
                        "--out",
                        jb.to_str().unwrap(),
                        "--no-calibrate",
                    ],
                    true,
                );
                st.run_tool(
                    "exact: record candidate --like",
                    &[
                        "exact",
                        "record",
                        "--url",
                        &cand.url,
                        "--out",
                        jc.to_str().unwrap(),
                        "--like",
                        jb.to_str().unwrap(),
                    ],
                    true,
                );
                st.run_tool(
                    "exact: compare identical",
                    &[
                        "exact",
                        "compare",
                        jb.to_str().unwrap(),
                        jc.to_str().unwrap(),
                    ],
                    true,
                );
                st.run_tool(
                    "exact: 4-stream record",
                    &[
                        "exact",
                        "record",
                        "--url",
                        &cand.url,
                        "--out",
                        j4.to_str().unwrap(),
                        "--like",
                        jb.to_str().unwrap(),
                        "--streams",
                        "4",
                    ],
                    true,
                );
                st.run_tool(
                    "exact: compare 8 vs 4",
                    &[
                        "exact",
                        "compare",
                        jb.to_str().unwrap(),
                        j4.to_str().unwrap(),
                    ],
                    true,
                );
                st.run_tool(
                    "exact: fault conc-wrong caught",
                    &[
                        "exact",
                        "record",
                        "--url",
                        &bad.url,
                        "--out",
                        jbad.to_str().unwrap(),
                        "--like",
                        jb.to_str().unwrap(),
                    ],
                    false,
                );
                st.run_tool(
                    "exact: weak refused",
                    &[
                        "exact",
                        "record",
                        "--url",
                        &weak.url,
                        "--out",
                        jw.to_str().unwrap(),
                        "--like",
                        jb.to_str().unwrap(),
                    ],
                    true,
                );
                st.run_tool(
                    "exact: compare refuses weak",
                    &[
                        "exact",
                        "compare",
                        jb.to_str().unwrap(),
                        jw.to_str().unwrap(),
                    ],
                    false,
                );
            }
            "spill" => {
                let d0 = work.join("sp-r0");
                let d1 = work.join("sp-r1");
                let rec = work.join("sp-base.json");
                let state = work.join("sp-state.json");
                let base = FakeProc::start(&exe, &work, "sp-base", &fast).await;
                let mut cand = FakeProc::start(
                    &exe,
                    &work,
                    "sp-cand",
                    &[
                        "--spill-dir",
                        d0.to_str().unwrap(),
                        "--spill-dir-rank1",
                        d1.to_str().unwrap(),
                        "--mem-points",
                        "6",
                        "--write-ms",
                        "80",
                        "--prefill-us-per-token",
                        "3",
                        "--decode-ms-per-token",
                        "1",
                    ],
                )
                .await;
                st.run_tool(
                    "spill: record baseline",
                    &[
                        "spill-check",
                        "record",
                        "--url",
                        &base.url,
                        "--out",
                        rec.to_str().unwrap(),
                        "--sizes",
                        "16000,40000",
                        "--cancel-tokens",
                        "60000",
                        "--dual-tokens",
                        "30000",
                        "--kill-tokens",
                        "30000",
                        "--floor-tokens",
                        "20000",
                    ],
                    true,
                );
                st.run_tool(
                    "spill: main phase",
                    &[
                        "spill-check",
                        "run",
                        "--url",
                        &cand.url,
                        "--baseline",
                        rec.to_str().unwrap(),
                        "--phase",
                        "main",
                        "--rank0-dir",
                        d0.to_str().unwrap(),
                        "--rank1-dir",
                        d1.to_str().unwrap(),
                        "--filler-tokens",
                        "3000",
                        "--commit-timeout",
                        "30",
                    ],
                    true,
                );
                st.run_tool(
                    "spill: pictures",
                    &[
                        "spill-check",
                        "run",
                        "--url",
                        &cand.url,
                        "--baseline",
                        rec.to_str().unwrap(),
                        "--phase",
                        "main",
                        "--checks",
                        "pictures,settled",
                        "--strict",
                        "--rank0-dir",
                        d0.to_str().unwrap(),
                        "--rank1-dir",
                        d1.to_str().unwrap(),
                        "--filler-tokens",
                        "3000",
                        "--commit-timeout",
                        "30",
                    ],
                    true,
                );
                st.run_tool(
                    "spill: prerestart",
                    &[
                        "spill-check",
                        "run",
                        "--url",
                        &cand.url,
                        "--baseline",
                        rec.to_str().unwrap(),
                        "--phase",
                        "prerestart",
                        "--state",
                        state.to_str().unwrap(),
                        "--pictures",
                        "--rank0-dir",
                        d0.to_str().unwrap(),
                        "--rank1-dir",
                        d1.to_str().unwrap(),
                        "--commit-timeout",
                        "30",
                    ],
                    true,
                );
                cand.stop();
                st.run_tool(
                    "spill: drop rank 1 manifest",
                    &[
                        "spill-check",
                        "drop",
                        "--state",
                        state.to_str().unwrap(),
                        "--rank1-dir",
                        d1.to_str().unwrap(),
                    ],
                    true,
                );
                cand.spawn().await;
                st.run_tool(
                    "spill: postrestart --strict",
                    &[
                        "spill-check",
                        "run",
                        "--url",
                        &cand.url,
                        "--baseline",
                        rec.to_str().unwrap(),
                        "--phase",
                        "postrestart",
                        "--state",
                        state.to_str().unwrap(),
                        "--strict",
                        "--pictures",
                        "--rank0-dir",
                        d0.to_str().unwrap(),
                        "--rank1-dir",
                        d1.to_str().unwrap(),
                    ],
                    true,
                );
                let kill_cmd = format!("kill -9 {}", cand.pid());
                st.run_tool(
                    "spill: prekill",
                    &[
                        "spill-check",
                        "run",
                        "--url",
                        &cand.url,
                        "--baseline",
                        rec.to_str().unwrap(),
                        "--phase",
                        "prekill",
                        "--state",
                        state.to_str().unwrap(),
                        "--resolve-cmd",
                        "true",
                        "--kill-cmd",
                        &kill_cmd,
                        "--rank0-dir",
                        d0.to_str().unwrap(),
                        "--rank1-dir",
                        d1.to_str().unwrap(),
                        "--commit-timeout",
                        "30",
                    ],
                    true,
                );
                cand.restart().await;
                st.run_tool(
                    "spill: postkill --strict",
                    &[
                        "spill-check",
                        "run",
                        "--url",
                        &cand.url,
                        "--baseline",
                        rec.to_str().unwrap(),
                        "--phase",
                        "postkill",
                        "--state",
                        state.to_str().unwrap(),
                        "--strict",
                        "--rank0-dir",
                        d0.to_str().unwrap(),
                        "--rank1-dir",
                        d1.to_str().unwrap(),
                    ],
                    true,
                );
                st.run_tool(
                    "spill: estimate",
                    &[
                        "spill-check",
                        "estimate",
                        "--baseline",
                        rec.to_str().unwrap(),
                        "--bytes",
                    ],
                    true,
                );

                // Negative fault test: restore-wrong
                let m0 = work.join("sp-mut-r0");
                let m1 = work.join("sp-mut-r1");
                let bad = FakeProc::start(
                    &exe,
                    &work,
                    "sp-mut",
                    &[
                        "--spill-dir",
                        m0.to_str().unwrap(),
                        "--spill-dir-rank1",
                        m1.to_str().unwrap(),
                        "--mem-points",
                        "6",
                        "--mutate",
                        "restore-wrong",
                        "--prefill-us-per-token",
                        "3",
                        "--decode-ms-per-token",
                        "1",
                    ],
                )
                .await;
                st.run_tool(
                    "spill: fault restore-wrong caught",
                    &[
                        "spill-check",
                        "run",
                        "--url",
                        &bad.url,
                        "--baseline",
                        rec.to_str().unwrap(),
                        "--phase",
                        "main",
                        "--checks",
                        "evict",
                        "--rank0-dir",
                        m0.to_str().unwrap(),
                        "--rank1-dir",
                        m1.to_str().unwrap(),
                        "--filler-tokens",
                        "3000",
                        "--commit-timeout",
                        "30",
                    ],
                    false,
                );
            }
            "kv" => {
                let rec = work.join("kv-base.json");
                let sl = work.join("kv-startup.log");
                let base = FakeProc::start(
                    &exe,
                    &work,
                    "kv-base",
                    &[
                        "--context",
                        "65518",
                        "--prefill-us-per-token",
                        "3",
                        "--decode-ms-per-token",
                        "1",
                    ],
                )
                .await;
                let sh_args = [
                    "--kv-shard",
                    "--pool",
                    "131072",
                    "--mem-points",
                    "16",
                    "--prefill-us-per-token",
                    "3",
                    "--decode-ms-per-token",
                    "30",
                    "--startup-log",
                    sl.to_str().unwrap(),
                ];
                let cand = FakeProc::start(&exe, &work, "kv-cand", &sh_args).await;
                let bad = FakeProc::start(
                    &exe,
                    &work,
                    "kv-bad",
                    &[
                        "--kv-shard",
                        "--pool",
                        "131072",
                        "--mem-points",
                        "16",
                        "--prefill-us-per-token",
                        "3",
                        "--decode-ms-per-token",
                        "20",
                        "--mutate",
                        "shard-wrong",
                    ],
                )
                .await;
                st.run_tool(
                    "kv: record baseline",
                    &[
                        "kv-shard-check",
                        "record",
                        "--url",
                        &base.url,
                        "--out",
                        rec.to_str().unwrap(),
                        "--carry-tokens",
                        "20000",
                        "--twin-tokens",
                        "20000",
                        "--filler-tokens",
                        "12000",
                        "--owner-tokens",
                        "2000",
                        "--decode-tokens",
                        "100",
                        "--burst-tokens",
                        "6000",
                        "--move-side-tokens",
                        "8000",
                        "--move-tokens",
                        "60000",
                    ],
                    true,
                );
                st.run_tool(
                    "kv: run all checks --strict",
                    &[
                        "kv-shard-check",
                        "run",
                        "--url",
                        &cand.url,
                        "--baseline",
                        rec.to_str().unwrap(),
                        "--needle-sizes",
                        "20000,max",
                        "--startup-log",
                        sl.to_str().unwrap(),
                        "--expect-pool",
                        "131072",
                        "--expect-window",
                        "65518",
                        "--strict",
                        "--sample-ms",
                        "20",
                        "--settle-wait",
                        "5",
                    ],
                    true,
                );
                st.run_tool(
                    "kv: fault shard-wrong caught",
                    &[
                        "kv-shard-check",
                        "run",
                        "--url",
                        &bad.url,
                        "--baseline",
                        rec.to_str().unwrap(),
                        "--checks",
                        "owners,mixed",
                        "--sample-ms",
                        "20",
                        "--settle-wait",
                        "5",
                    ],
                    false,
                );
            }
            "prefix" => {
                let fb = work.join("fan-base.json");
                let f1 = FakeProc::start(
                    &exe,
                    &work,
                    "pfx-1",
                    &[
                        "--mem-points",
                        "32",
                        "--serial-prefill",
                        "--prefill-us-per-token",
                        "3",
                        "--decode-ms-per-token",
                        "1",
                    ],
                )
                .await;
                st.run_tool(
                    "prefix: serial one-fill record",
                    &[
                        "prefix-fanout",
                        "--url",
                        &f1.url,
                        "--out",
                        fb.to_str().unwrap(),
                        "--system-tokens",
                        "30000",
                        "--expect-one-fill",
                    ],
                    true,
                );
                drop(f1);

                let f2 = FakeProc::start(
                    &exe,
                    &work,
                    "pfx-2",
                    &[
                        "--mem-points",
                        "32",
                        "--serial-prefill",
                        "--prefill-us-per-token",
                        "3",
                        "--decode-ms-per-token",
                        "1",
                    ],
                )
                .await;
                st.run_tool(
                    "prefix: serial baseline match",
                    &[
                        "prefix-fanout",
                        "--url",
                        &f2.url,
                        "--baseline",
                        fb.to_str().unwrap(),
                        "--expect-one-fill",
                        "--strict-cold",
                    ],
                    true,
                );
                st.run_tool(
                    "prefix: warm server caught (--strict-cold)",
                    &[
                        "prefix-fanout",
                        "--url",
                        &f2.url,
                        "--baseline",
                        fb.to_str().unwrap(),
                        "--strict-cold",
                    ],
                    false,
                );
                drop(f2);

                let f3 = FakeProc::start(
                    &exe,
                    &work,
                    "pfx-3",
                    &[
                        "--mem-points",
                        "32",
                        "--prefill-us-per-token",
                        "3",
                        "--decode-ms-per-token",
                        "1",
                    ],
                )
                .await;
                st.run_tool(
                    "prefix: overlapping prefill caught",
                    &[
                        "prefix-fanout",
                        "--url",
                        &f3.url,
                        "--baseline",
                        fb.to_str().unwrap(),
                        "--expect-one-fill",
                    ],
                    false,
                );
            }
            "pair" => {
                let mut f = FakeProc::start(
                    &exe,
                    &work,
                    "pair",
                    &[
                        "--prefill-us-per-token",
                        "3",
                        "--decode-ms-per-token",
                        "250",
                    ],
                )
                .await;
                let st_file = work.join("pair-state");
                let rlog = work.join("pair-rank0.log");
                fs::write(&st_file, "running 0\n").unwrap();
                let kill75 = format!(
                    "printf 'tensorfold: the rank pair broke; exiting with status 75\\n' > '{}'; printf 'exited 75\\n' > '{}'; kill -9 {}",
                    rlog.display(), st_file.display(), f.pid()
                );
                st.run_tool(
                    "pair: exit 75 + log matched",
                    &[
                        "pair-check",
                        "--url",
                        &f.url,
                        "--streams",
                        "3",
                        "--max-tokens",
                        "2000",
                        "--decode-s",
                        "0.4",
                        "--exit-cmd",
                        &format!("cat '{}'", st_file.display()),
                        "--kill-cmd",
                        &kill75,
                        "--end-deadline",
                        "15",
                        "--expect-log",
                        &format!("cat '{}'::exiting with status 75", rlog.display()),
                    ],
                    true,
                );
                f.restart().await;
                fs::write(&st_file, "running 0\n").unwrap();
                let kill137 = format!(
                    "printf 'exited 137\\n' > '{}'; kill -9 {}",
                    st_file.display(),
                    f.pid()
                );
                st.run_tool(
                    "pair: exit 137 caught",
                    &[
                        "pair-check",
                        "--url",
                        &f.url,
                        "--streams",
                        "3",
                        "--max-tokens",
                        "2000",
                        "--decode-s",
                        "0.4",
                        "--exit-cmd",
                        &format!("cat '{}'", st_file.display()),
                        "--kill-cmd",
                        &kill137,
                        "--end-deadline",
                        "15",
                    ],
                    false,
                );
                let _ = json!({});
            }
            other => {
                eprintln!("unknown selftest suite: {other}");
                return 2;
            }
        }
    }

    let fails: Vec<&String> = st
        .results
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(n, _)| n)
        .collect();
    println!(
        "\nselftest: {} ({} checks, {} failed)",
        if fails.is_empty() { "PASS" } else { "FAIL" },
        st.results.len(),
        fails.len()
    );
    if fails.is_empty() && !a.keep {
        let _ = fs::remove_dir_all(&work);
    } else {
        println!("artifacts kept in {}", work.display());
    }
    if fails.is_empty() {
        0
    } else {
        1
    }
}
