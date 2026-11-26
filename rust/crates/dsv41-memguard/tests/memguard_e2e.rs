use std::fs;
use std::io::{Read, Write};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_dsv41-memguard")
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        // Keep Unix socket path well under macOS SUN_LEN (104 bytes)
        let p = PathBuf::from(format!("/tmp/d41mg-{tag}-{}", std::process::id()));
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

fn write_meminfo(path: &Path, avail_gib: f64) {
    let kb = (avail_gib * 1_048_576.0) as u64;
    let tmp = path.with_extension("tmp");
    fs::write(
        &tmp,
        format!(
            "MemTotal:       134217728 kB\nMemFree:        1048576 kB\nMemAvailable:   {kb} kB\n"
        ),
    )
    .unwrap();
    fs::rename(&tmp, path).unwrap();
}

struct MockDockerSocket {
    path: PathBuf,
    requests: Arc<Mutex<Vec<String>>>,
}

impl MockDockerSocket {
    fn start(dir: &Path) -> Self {
        let path = dir.join("docker.sock");
        let _ = fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        listener.set_nonblocking(true).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let reqs_clone = Arc::clone(&requests);
        thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(15);
            while Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                        let mut buf = [0u8; 4096];
                        if let Ok(n) = stream.read(&mut buf) {
                            let s = String::from_utf8_lossy(&buf[..n]).to_string();
                            reqs_clone.lock().unwrap().push(s);
                            let _ = stream.write_all(
                                b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                            );
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });
        Self { path, requests }
    }

    fn received(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

#[test]
fn test_soft_floor_dip_recovers_without_kill_and_clean_stop() {
    let tmp = TempDir::new("dip");
    let meminfo = tmp.path().join("meminfo");
    let log = tmp.path().join("memguard.log");
    let pid = tmp.path().join("memguard.pid");
    write_meminfo(&meminfo, 10.0);
    let sock = MockDockerSocket::start(tmp.path());

    let st = Command::new(bin())
        .args([
            "start",
            "test-rank0",
            "2.5",
            log.to_str().unwrap(),
            pid.to_str().unwrap(),
            "3.0",
            "1.0",
        ])
        .env("MEMGUARD_MEMINFO", &meminfo)
        .env("MEMGUARD_DOCKER_SOCK", &sock.path)
        .env("MEMGUARD_POLL_MS", "40")
        .status()
        .unwrap();
    assert!(st.success(), "memguard start failed");
    assert!(pid.exists(), "pidfile missing after start");

    // Dip below soft floor (2.5 GiB) but above hard floor (1.0 GiB) briefly (< 3.0s)
    write_meminfo(&meminfo, 1.8);
    thread::sleep(Duration::from_millis(250));
    // Recover above soft floor
    write_meminfo(&meminfo, 10.0);
    thread::sleep(Duration::from_millis(250));

    // Stop cleanly
    let st_stop = Command::new(bin())
        .args(["stop", pid.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(st_stop.success(), "memguard stop failed");
    thread::sleep(Duration::from_millis(100));
    assert!(!pid.exists(), "pidfile should be removed on stop");

    assert!(
        sock.received().is_empty(),
        "must not kill container on transient dip: {:?}",
        sock.received()
    );
    let log_text = fs::read_to_string(&log).unwrap();
    assert!(log_text.contains("watching test-rank0"), "{log_text}");
    assert!(
        log_text.contains("recovered"),
        "expected dip recovery log: {log_text}"
    );
}

#[test]
fn test_soft_floor_hold_kills_via_docker_socket() {
    let tmp = TempDir::new("soft-kill");
    let meminfo = tmp.path().join("meminfo");
    let log = tmp.path().join("memguard.log");
    let pid = tmp.path().join("memguard.pid");
    write_meminfo(&meminfo, 1.8); // below 2.5 GiB soft floor, above 1.0 GiB hard floor
    let sock = MockDockerSocket::start(tmp.path());

    let mut child = Command::new(bin())
        .args([
            "run",
            "test-rank0",
            "2.5",
            log.to_str().unwrap(),
            pid.to_str().unwrap(),
            "0.25",
            "1.0",
        ])
        .env("MEMGUARD_MEMINFO", &meminfo)
        .env("MEMGUARD_DOCKER_SOCK", &sock.path)
        .env("MEMGUARD_POLL_MS", "40")
        .spawn()
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && sock.received().is_empty() {
        thread::sleep(Duration::from_millis(20));
    }
    let reqs = sock.received();
    thread::sleep(Duration::from_millis(80));
    let _ = Command::new(bin())
        .args(["stop", pid.to_str().unwrap()])
        .status();
    let _ = child.wait();
    assert_eq!(
        reqs.len(),
        1,
        "expected 1 Docker kill request, got {reqs:?}"
    );
    assert!(
        reqs[0].starts_with("POST /containers/test-rank0/kill HTTP/1.1"),
        "unexpected request: {}",
        reqs[0]
    );
    let log_text = fs::read_to_string(&log).unwrap();
    assert!(
        log_text.contains("killed container test-rank0"),
        "{log_text}"
    );
}

#[test]
fn test_hard_floor_two_samples_kills_immediately() {
    let tmp = TempDir::new("hard-kill");
    let meminfo = tmp.path().join("meminfo");
    let log = tmp.path().join("memguard.log");
    let pid = tmp.path().join("memguard.pid");
    write_meminfo(&meminfo, 0.5); // below 1.0 GiB hard floor
    let sock = MockDockerSocket::start(tmp.path());

    let t0 = Instant::now();
    let mut child = Command::new(bin())
        .args([
            "run",
            "test-rank1",
            "2.5",
            log.to_str().unwrap(),
            pid.to_str().unwrap(),
            "30.0", // long soft hold: hard floor must trigger immediately!
            "1.0",
        ])
        .env("MEMGUARD_MEMINFO", &meminfo)
        .env("MEMGUARD_DOCKER_SOCK", &sock.path)
        .env("MEMGUARD_POLL_MS", "40")
        .spawn()
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && sock.received().is_empty() {
        thread::sleep(Duration::from_millis(20));
    }
    let elapsed = t0.elapsed();
    let reqs = sock.received();
    thread::sleep(Duration::from_millis(80));
    let _ = Command::new(bin())
        .args(["stop", pid.to_str().unwrap()])
        .status();
    let _ = child.wait();
    assert_eq!(
        reqs.len(),
        1,
        "expected 1 Docker kill request, got {reqs:?}"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "hard floor waited too long: {elapsed:?}"
    );
    assert!(
        reqs[0].starts_with("POST /containers/test-rank1/kill HTTP/1.1"),
        "unexpected request: {}",
        reqs[0]
    );
    let log_text = fs::read_to_string(&log).unwrap();
    assert!(
        log_text.contains("killed container test-rank1"),
        "{log_text}"
    );
}
