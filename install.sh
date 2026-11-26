#!/usr/bin/env bash
# Install the Rust toolchain (if missing), compile optimized release binaries for
# DeepSeek-V4.1-Flash-1017 on 2x DGX Sparks, install them atomically on this host,
# and optionally distribute them to the worker Spark (WORKER).
#
# Usage: ./install.sh [options]
#   --prefix DIR              Install directory on this host (default: $HOME/.local/bin)
#   --rust-version VER        Rust toolchain version to install/use (default: stable)
#   --target-cpu CPU          RUSTFLAGS -C target-cpu value (default: native; use generic for cross-CPU)
#   --local-only              Skip distribution to WORKER
#   --worker-install-rust     Install rustup and build natively on WORKER instead of copying head binaries
#   --skip-tests              Skip `cargo test --workspace` after building
#   --check                   Verify installed Rust toolchain and binaries without building
#   -h, --help                Show this help text
#
# Kept compatible with bash 3.2+ (macOS and Linux).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT="$SCRIPT_DIR"
MANIFEST="$ROOT/rust/Cargo.toml"

PREFIX="${PREFIX:-$HOME/.local/bin}"
RUST_VERSION="stable"
RUST_VERSION_EXPLICIT=0
TARGET_CPU="${TARGET_CPU:-native}"
LOCAL_ONLY=0
WORKER_INSTALL_RUST=0
SKIP_TESTS=0
CHECK_ONLY=0

ALL_BINS="dsv41-memguard dsv41-imgdec dsv41-ctl dsv41-tools"
WORKER_BINS="dsv41-memguard dsv41-ctl dsv41-tools"
WORKER_BIN_DIR='$HOME/.cache/dsv41-tensorfold/bin'

usage() {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
}

log()  { printf '[install.sh] %s\n' "$*"; }
warn() { printf '[install.sh] WARN: %s\n' "$*" >&2; }
die()  { printf '[install.sh] ERROR: %s\n' "$*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --prefix)
      [ "$#" -ge 2 ] || die "--prefix requires a directory"
      PREFIX="$2"; shift 2 ;;
    --rust-version)
      [ "$#" -ge 2 ] || die "--rust-version requires a version (e.g. stable or 1.88.0)"
      RUST_VERSION="$2"; RUST_VERSION_EXPLICIT=1; shift 2 ;;
    --target-cpu)
      [ "$#" -ge 2 ] || die "--target-cpu requires a CPU name (e.g. native or generic)"
      TARGET_CPU="$2"; shift 2 ;;
    --local-only)
      LOCAL_ONLY=1; shift ;;
    --worker-install-rust)
      WORKER_INSTALL_RUST=1; shift ;;
    --skip-tests)
      SKIP_TESTS=1; shift ;;
    --check)
      CHECK_ONLY=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die "unknown option: $1 (try --help)" ;;
  esac
done

# Extract WORKER safely without executing scripts/local.sh (bash 3.2 compatible)
read_kv_from_file() {
  local file=$1 key=$2 val=""
  [ -f "$file" ] || return 0
  val=$(awk -v k="$key" '
    /^[[:space:]]*#/ { next }
    {
      line = $0
      sub(/^[[:space:]]*(export[[:space:]]+)?/, "", line)
      if (index(line, k "=") == 1) {
        v = substr(line, length(k) + 2)
        sub(/[[:space:]]*#.*$/, "", v)
        sub(/[[:space:]]+$/, "", v)
        if ((substr(v, 1, 1) == "\"" && substr(v, length(v), 1) == "\"") ||
            (substr(v, 1, 1) == "\047" && substr(v, length(v), 1) == "\047")) {
          v = substr(v, 2, length(v) - 2)
        }
        out = v
      }
    }
    END { if (out != "") print out }
  ' "$file")
  printf '%s' "$val"
}

if [ -z "${WORKER:-}" ]; then
  WORKER=$(read_kv_from_file "$ROOT/scripts/local.sh" WORKER)
fi
if [ -z "${WORKER:-}" ]; then
  WORKER=$(read_kv_from_file "$ROOT/.env" WORKER)
fi

export PATH="$HOME/.cargo/bin:$PATH"

check_installed() {
  local ok=0 b
  if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
    log "rustc: $(rustc --version)"
    log "cargo: $(cargo --version)"
  else
    warn "rustc / cargo not found on PATH (including \$HOME/.cargo/bin)"
    ok=1
  fi
  for b in $ALL_BINS; do
    if [ -x "$PREFIX/$b" ]; then
      log "head $PREFIX/$b: $("$PREFIX/$b" --version 2>&1 || echo '<failed>')"
    else
      warn "missing binary: $PREFIX/$b"
      ok=1
    fi
  done
  if [ "$LOCAL_ONLY" -eq 0 ] && [ -n "${WORKER:-}" ]; then
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" true 2>/dev/null; then
      for b in $WORKER_BINS; do
        local ver
        if ver=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" \
          "[ -x $WORKER_BIN_DIR/$b ] && $WORKER_BIN_DIR/$b --version" 2>/dev/null); then
          log "worker ($WORKER) $b: $ver"
        else
          warn "worker ($WORKER) missing binary: $WORKER_BIN_DIR/$b"
          ok=1
        fi
      done
    else
      warn "cannot reach worker $WORKER over ssh (use --local-only to skip worker check)"
      ok=1
    fi
  fi
  return "$ok"
}

if [ "$CHECK_ONLY" -eq 1 ]; then
  check_installed
  exit $?
fi

[ -f "$MANIFEST" ] || die "Rust workspace manifest not found: $MANIFEST"

# 1. Prerequisite checks
has_cc=0
for cc in cc gcc clang; do
  if command -v "$cc" >/dev/null 2>&1; then has_cc=1; break; fi
done
if [ "$has_cc" -eq 0 ]; then
  die "C compiler / linker (cc, gcc or clang) not found. On Ubuntu/Debian run: sudo apt-get update && sudo apt-get install -y build-essential curl"
fi

# 2. Ensure Rust toolchain is installed
if ! command -v cargo >/dev/null 2>&1 || ! command -v rustc >/dev/null 2>&1; then
  command -v curl >/dev/null 2>&1 || die "curl is required to install rustup (sudo apt-get install -y curl)"
  log "Rust toolchain not found; installing rustup (toolchain: $RUST_VERSION, profile: minimal)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain "$RUST_VERSION"
  # shellcheck disable=SC1091
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
  export PATH="$HOME/.cargo/bin:$PATH"
fi

CARGO_TOOLCHAIN=""
if [ "$RUST_VERSION_EXPLICIT" -eq 1 ] && command -v rustup >/dev/null 2>&1; then
  log "Ensuring Rust toolchain '$RUST_VERSION' is installed via rustup"
  rustup toolchain install "$RUST_VERSION" --profile minimal
  CARGO_TOOLCHAIN="+$RUST_VERSION"
fi

log "Using $(rustc --version) / $(cargo --version)"

# 3. Release build (native CPU tuning by default)
LOCK_FLAG="--locked"
[ -f "$ROOT/rust/Cargo.lock" ] || LOCK_FLAG=""

log "Building release binaries (target-cpu=$TARGET_CPU)..."
RUSTFLAGS="-C target-cpu=${TARGET_CPU}" \
  cargo ${CARGO_TOOLCHAIN:+$CARGO_TOOLCHAIN} build --release ${LOCK_FLAG:+$LOCK_FLAG} \
    --manifest-path "$MANIFEST" --bins

if [ "$SKIP_TESTS" -eq 0 ]; then
  log "Running workspace tests..."
  cargo ${CARGO_TOOLCHAIN:+$CARGO_TOOLCHAIN} test ${LOCK_FLAG:+$LOCK_FLAG} \
    --manifest-path "$MANIFEST" --workspace
fi

# 4. Atomic installation into PREFIX (safe even while dsv41-memguard is running)
mkdir -p "$PREFIX"
for b in $ALL_BINS; do
  src="$ROOT/rust/target/release/$b"
  [ -x "$src" ] || die "expected built binary missing: $src"
  tmp="$PREFIX/$b.tmp.$$"
  cp "$src" "$tmp"
  chmod 0755 "$tmp"
  mv -f "$tmp" "$PREFIX/$b"
  log "Installed $PREFIX/$b ($("$PREFIX/$b" --version))"
done

# 5. Optional worker distribution
if [ "$LOCAL_ONLY" -eq 0 ] && [ -n "${WORKER:-}" ]; then
  log "Checking worker $WORKER..."
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" true 2>/dev/null || \
    die "cannot ssh to WORKER=$WORKER without a password (pass --local-only to install on this host only)"
  head_arch=$(uname -m)
  worker_arch=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" uname -m)
  if [ "$WORKER_INSTALL_RUST" -eq 1 ] || [ "$head_arch" != "$worker_arch" ]; then
    log "Building natively on worker $WORKER (head=$head_arch, worker=$worker_arch)..."
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "bash -s" -- "$RUST_VERSION" "$TARGET_CPU" <<'REMOTE_RUST'
set -euo pipefail
RUST_VERSION=$1
TARGET_CPU=$2
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v cargo >/dev/null 2>&1 || ! command -v rustc >/dev/null 2>&1; then
  command -v curl >/dev/null 2>&1 || { echo "curl is missing on worker" >&2; exit 1; }
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$RUST_VERSION"
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
fi
REMOTE_RUST
    remote_tmp=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" 'mktemp -d "${TMPDIR:-/tmp}/dsv41-rust.XXXXXX"')
    tar -C "$ROOT" --exclude='rust/target' -cf - rust | \
      ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "tar -C '$remote_tmp' -xf -"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" \
      "export PATH=\"\$HOME/.cargo/bin:\$PATH\"; \
       RUSTFLAGS='-C target-cpu=${TARGET_CPU}' cargo build --release ${LOCK_FLAG:+$LOCK_FLAG} --manifest-path '$remote_tmp/rust/Cargo.toml' --bins && \
       mkdir -p $WORKER_BIN_DIR && \
       for b in $WORKER_BINS; do \
         cp '$remote_tmp/rust/target/release/'\"\$b\" $WORKER_BIN_DIR/\"\$b\".tmp && \
         chmod 0755 $WORKER_BIN_DIR/\"\$b\".tmp && \
         mv -f $WORKER_BIN_DIR/\"\$b\".tmp $WORKER_BIN_DIR/\"\$b\"; \
       done; \
       rm -rf '$remote_tmp'"
  else
    log "Distributing $head_arch binaries atomically to $WORKER ($WORKER_BIN_DIR)..."
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "mkdir -p $WORKER_BIN_DIR"
    for b in $WORKER_BINS; do
      ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" \
        "cat > $WORKER_BIN_DIR/$b.tmp && chmod 0755 $WORKER_BIN_DIR/$b.tmp && mv -f $WORKER_BIN_DIR/$b.tmp $WORKER_BIN_DIR/$b" \
        < "$PREFIX/$b"
    done
  fi
  for b in $WORKER_BINS; do
    ver=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "$WORKER_BIN_DIR/$b --version")
    log "Verified on $WORKER: $ver"
  done
elif [ "$LOCAL_ONLY" -eq 0 ]; then
  log "WORKER is not configured (scripts/local.sh); installed on this host only"
fi

case ":$PATH:" in
  *":$PREFIX:"*) ;;
  *) warn "$PREFIX is not currently in PATH; add 'export PATH=\"$PREFIX:\$PATH\"' to your shell profile" ;;
esac
log "Done."
