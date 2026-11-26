//! Zero-fork native Rust memory watchdog for NVIDIA DGX Spark (GB10 unified memory).
//!
//! Replaces `scripts/memguard.sh`: reads `/proc/meminfo` directly into a fixed stack buffer
//! with zero process forks per tick, optionally locks itself into RAM via `mlockall`, and kills
//! the target container directly via Docker's Unix domain socket (`POST /containers/{id}/kill`).
//!
//! Usage:
//!   dsv41-memguard start <container> <floor GiB> <log file> <pid file> [hold seconds [hard floor GiB]]
//!   dsv41-memguard run   <container> <floor GiB> <log file> <pid file> [hold seconds [hard floor GiB]]
//!   dsv41-memguard stop  <pid file>
//!   dsv41-memguard --version

use dsv41_core::docker_sock::{default_docker_socket, kill_container_via_socket};
use dsv41_core::meminfo::read_mem_available_kb;
use dsv41_core::now_stamp;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

static STOP_FLAG: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_signal(_sig: libc::c_int) {
    STOP_FLAG.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = handle_signal as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0; // no SA_RESTART so sleep wakes immediately on signal
        libc::sigaction(libc::SIGTERM, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
    }
}

fn lock_memory_best_effort() {
    #[cfg(target_os = "linux")]
    unsafe {
        let _ = libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE);
    }
}

fn sleep_interruptible(dur: Duration) -> bool {
    let mut req = libc::timespec {
        tv_sec: dur.as_secs() as libc::time_t,
        tv_nsec: dur.subsec_nanos() as libc::c_long,
    };
    let mut rem = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    while !STOP_FLAG.load(Ordering::SeqCst) {
        let rc = unsafe { libc::nanosleep(&req, &mut rem) };
        if rc == 0 {
            return !STOP_FLAG.load(Ordering::SeqCst);
        }
        if STOP_FLAG.load(Ordering::SeqCst) {
            return false;
        }
        req = rem;
    }
    false
}

fn is_valid_number(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut dot = false;
    let mut digits = false;
    for b in s.bytes() {
        if b.is_ascii_digit() {
            digits = true;
        } else if b == b'.' && !dot {
            dot = true;
        } else {
            return false;
        }
    }
    digits && !s.starts_with('.') && !s.ends_with('.')
}

fn process_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    unsafe { libc::kill(pid, 0) == 0 || *errno_ptr() == libc::EPERM }
}

#[cfg(target_os = "linux")]
unsafe fn errno_ptr() -> *mut libc::c_int {
    libc::__errno_location()
}

#[cfg(target_os = "macos")]
unsafe fn errno_ptr() -> *mut libc::c_int {
    libc::__error()
}

fn looks_like_memguard(pid: i32) -> bool {
    #[cfg(target_os = "linux")]
    {
        if let Ok(raw) = fs::read(format!("/proc/{pid}/cmdline")) {
            let s = String::from_utf8_lossy(&raw);
            return s.contains("memguard");
        }
        false
    }
    #[cfg(not(target_os = "linux"))]
    {
        let Ok(out) = Command::new("ps")
            .args(["-o", "command=", "-p", &pid.to_string()])
            .output()
        else {
            return false;
        };
        String::from_utf8_lossy(&out.stdout).contains("memguard")
    }
}

fn stop_guard(pidfile: &Path) {
    if !pidfile.exists() {
        return;
    }
    if let Ok(text) = fs::read_to_string(pidfile) {
        if let Ok(pid) = text.trim().parse::<i32>() {
            if pid > 0 && looks_like_memguard(pid) {
                unsafe {
                    libc::kill(pid, libc::SIGTERM);
                }
                for _ in 0..50 {
                    if !process_alive(pid) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }
    let _ = fs::remove_file(pidfile);
}

fn append_log(log_path: &Path, msg: &str) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log_path) {
        let _ = writeln!(f, "{msg}");
    }
}

/// Collects top-5 RSS processes directly from `/proc` without forking `ps`.
fn top_rss_snapshot() -> Vec<String> {
    #[cfg(target_os = "linux")]
    {
        let page_kb = (dsv41_core::meminfo::system_page_size() / 1024).max(4);
        let mut procs: Vec<(u64, u32, String)> = Vec::new();
        if let Ok(rd) = fs::read_dir("/proc") {
            for ent in rd.flatten() {
                let name = ent.file_name();
                let Some(s) = name.to_str() else { continue };
                let Ok(pid) = s.parse::<u32>() else { continue };
                let statm_p = ent.path().join("statm");
                let Ok(statm) = fs::read_to_string(statm_p) else {
                    continue;
                };
                let rss_pages = statm
                    .split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(0);
                if rss_pages == 0 {
                    continue;
                }
                let comm = fs::read_to_string(ent.path().join("comm"))
                    .map(|c| c.trim().to_string())
                    .unwrap_or_else(|_| "?".to_string());
                procs.push((rss_pages * page_kb, pid, comm));
            }
        }
        procs.sort_by(|a, b| b.0.cmp(&a.0));
        let mut lines = vec!["      RSS     PID COMMAND".to_string()];
        for (rss, pid, comm) in procs.into_iter().take(5) {
            lines.push(format!("    {rss:>7} {pid:>7} {comm}"));
        }
        lines
    }
    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

fn kill_container(container: &str) -> bool {
    let sock = default_docker_socket();
    let timeout = Duration::from_secs(20);
    match kill_container_via_socket(&sock, container, timeout) {
        Ok(code) => (200..300).contains(&code),
        Err(_) => {
            // Fallback to `docker kill` only if Unix socket is unavailable
            Command::new("docker")
                .args(["kill", container])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        }
    }
}

fn run_guard(
    container: &str,
    floor_gib_str: &str,
    log_path: &Path,
    pidfile: &Path,
    hold_str: &str,
    hard_gib_str: &str,
) -> i32 {
    let floor_gib: f64 = floor_gib_str.parse().unwrap_or(2.5);
    let hold_s: f64 = hold_str.parse().unwrap_or(4.0);
    let hard_gib: f64 = hard_gib_str.parse().unwrap_or(1.0);
    let floor_kb = (floor_gib * 1_048_576.0) as u64;
    let hard_kb = (hard_gib * 1_048_576.0) as u64;
    let hold_dur = Duration::from_secs_f64(hold_s.max(0.0));
    let poll_ms: u64 = env::var("MEMGUARD_POLL_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(250);
    let poll_dur = Duration::from_millis(poll_ms);
    let meminfo_path = env::var("MEMGUARD_MEMINFO")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/proc/meminfo"));

    if let Some(p) = log_path.parent() {
        let _ = fs::create_dir_all(p);
    }
    if let Some(p) = pidfile.parent() {
        let _ = fs::create_dir_all(p);
    }
    let my_pid = std::process::id();
    let _ = fs::write(pidfile, format!("{my_pid}\n"));

    install_signal_handlers();
    lock_memory_best_effort();

    let mut buf = [0u8; 4096];
    let init_kb = read_mem_available_kb(&meminfo_path, &mut buf).unwrap_or(0);
    append_log(
        log_path,
        &format!(
            "{} memguard: watching {}, floor {} GiB for {} s, hard floor {} GiB (MemAvailable {} GiB now)",
            now_stamp(),
            container,
            floor_gib_str,
            hold_str,
            hard_gib_str,
            init_kb / 1_048_576
        ),
    );

    let mut since: Option<Instant> = None;
    let mut low_min: u64 = 0;
    let mut hard_low: u32 = 0;

    while sleep_interruptible(poll_dur) {
        let Ok(a) = read_mem_available_kb(&meminfo_path, &mut buf) else {
            continue;
        };
        let now = Instant::now();
        if a >= floor_kb {
            if let Some(s) = since.take() {
                let dur_ms = now.duration_since(s).as_millis();
                append_log(
                    log_path,
                    &format!(
                        "{} memguard: MemAvailable was under the {} GiB floor for {} ms (lowest {} MiB) and recovered: no kill",
                        now_stamp(),
                        floor_gib_str,
                        dur_ms,
                        low_min / 1024
                    ),
                );
            }
            hard_low = 0;
            continue;
        }
        let s = match since {
            Some(s) => {
                if a < low_min {
                    low_min = a;
                }
                s
            }
            None => {
                since = Some(now);
                low_min = a;
                now
            }
        };
        if a < hard_kb {
            hard_low += 1;
        } else {
            hard_low = 0;
        }
        if hard_low < 2 && now.duration_since(s) < hold_dur {
            continue;
        }
        since = None;
        hard_low = 0;

        // Kill first, log afterwards: under heavy thrashing every syscall before kill costs time.
        let killed = kill_container(container);
        if killed {
            let snap = top_rss_snapshot();
            let mut msg = format!(
                "{} memguard: MemAvailable {} MiB (lowest {} MiB) was under the {} GiB floor for {} s or under the {} GiB hard floor: killed container {}",
                now_stamp(),
                a / 1024,
                low_min / 1024,
                floor_gib_str,
                hold_str,
                hard_gib_str,
                container
            );
            for line in snap {
                msg.push('\n');
                msg.push_str(&line);
            }
            append_log(log_path, &msg);
            if !sleep_interruptible(Duration::from_secs(5)) {
                break;
            }
        }
    }

    if let Ok(text) = fs::read_to_string(pidfile) {
        if text.trim() == my_pid.to_string() {
            let _ = fs::remove_file(pidfile);
        }
    }
    0
}

fn start_daemon(args: &[String]) -> i32 {
    if args.len() < 4 || args.len() > 6 {
        eprintln!(
            "usage: dsv41-memguard start <container> <floor GiB> <log file> <pid file> [hold seconds [hard floor GiB]]"
        );
        return 2;
    }
    if !is_valid_number(&args[1]) {
        eprintln!(
            "memguard: floor is a number of GiB (2 or 2.5), not {}",
            args[1]
        );
        return 2;
    }
    if let Some(h) = args.get(4) {
        if !h.is_empty() && !is_valid_number(h) {
            eprintln!("memguard: hold is a number of seconds, not {h}");
            return 2;
        }
    }
    if let Some(hf) = args.get(5) {
        if !hf.is_empty() && !is_valid_number(hf) {
            eprintln!("memguard: the hard floor is a number of GiB, not {hf}");
            return 2;
        }
    }
    let pidfile = Path::new(&args[3]);
    stop_guard(pidfile);

    let exe = match env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("memguard: cannot locate own binary: {e}");
            return 1;
        }
    };
    let mut cmd = Command::new(exe);
    cmd.arg("run")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            // Close inherited extra fds (such as start.sh's flock fd 8 / prepare.sh's fd 9)
            for fd in 3..64 {
                libc::close(fd);
            }
            Ok(())
        });
    }
    if let Err(e) = cmd.spawn() {
        eprintln!("memguard: did not start ({e})");
        return 1;
    }
    for _ in 0..40 {
        if pidfile.exists() {
            return 0;
        }
        thread::sleep(Duration::from_millis(50));
    }
    eprintln!("memguard: did not start");
    1
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("dsv41-memguard {}", env!("CARGO_PKG_VERSION"));
            0
        }
        Some("start") => start_daemon(&args[1..]),
        Some("run") => {
            if args.len() < 5 {
                eprintln!(
                    "usage: dsv41-memguard run <container> <floor GiB> <log file> <pid file> [hold seconds [hard floor GiB]]"
                );
                2
            } else {
                let hold = args.get(5).map(String::as_str).unwrap_or("4");
                let hard = args.get(6).map(String::as_str).unwrap_or("1");
                run_guard(
                    &args[1],
                    &args[2],
                    Path::new(&args[3]),
                    Path::new(&args[4]),
                    hold,
                    hard,
                )
            }
        }
        Some("stop") => {
            if let Some(pf) = args.get(1) {
                stop_guard(Path::new(pf));
                0
            } else {
                eprintln!("usage: dsv41-memguard stop <pid file>");
                2
            }
        }
        _ => {
            eprintln!("usage: dsv41-memguard start|run|stop ...");
            2
        }
    };
    std::process::exit(code);
}
