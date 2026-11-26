//! Zero-fork Docker Engine API client over Unix domain socket (`/var/run/docker.sock`).

use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub fn default_docker_socket() -> PathBuf {
    if let Ok(p) = std::env::var("MEMGUARD_DOCKER_SOCK") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    if let Ok(host) = std::env::var("DOCKER_HOST") {
        if let Some(p) = host.strip_prefix("unix://") {
            if !p.is_empty() {
                return PathBuf::from(p);
            }
        }
    }
    PathBuf::from("/var/run/docker.sock")
}

fn percent_encode_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    out
}

fn parse_http_status(resp: &[u8]) -> io::Result<u16> {
    let line_end = resp
        .windows(2)
        .position(|w| w == b"\r\n")
        .unwrap_or(resp.len());
    let line = std::str::from_utf8(&resp[..line_end])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let code_str = line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("bad status: {line}")))?;
    code_str
        .parse::<u16>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Sends `POST /containers/{name}/kill` directly over Docker's Unix domain socket.
/// Returns the HTTP status code (204 = killed, 404 = not found, 409 = not running).
pub fn kill_container_via_socket(
    sock_path: &Path,
    container: &str,
    timeout: Duration,
) -> io::Result<u16> {
    let mut stream = UnixStream::connect(sock_path)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let req = format!(
        "POST /containers/{}/kill HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        percent_encode_segment(container),
    );
    stream.write_all(req.as_bytes())?;
    stream.flush()?;
    let mut buf = [0u8; 512];
    let n = stream.read(&mut buf)?;
    if n == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "empty response from docker socket",
        ));
    }
    parse_http_status(&buf[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;
    use std::thread;

    #[test]
    fn test_kill_container_mock_socket() {
        let dir = std::env::temp_dir().join(format!("dsv41-sock-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("docker.sock");
        let listener = UnixListener::bind(&sock).unwrap();

        let h = thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let mut req = [0u8; 512];
            let n = conn.read(&mut req).unwrap();
            let s = String::from_utf8_lossy(&req[..n]).to_string();
            conn.write_all(b"HTTP/1.1 204 No Content\r\n\r\n").unwrap();
            s
        });

        let code =
            kill_container_via_socket(&sock, "dsv41-1017-exl3-tf", Duration::from_secs(2)).unwrap();
        let req = h.join().unwrap();
        assert_eq!(code, 204);
        assert!(req.starts_with("POST /containers/dsv41-1017-exl3-tf/kill HTTP/1.1\r\n"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
