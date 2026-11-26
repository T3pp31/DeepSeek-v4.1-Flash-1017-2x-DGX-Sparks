//! Zero-copy safetensors reader/writer and exact F16/BF16/F32 cast verification.
//! Replaces PyTorch-heavy `scripts/ablit_stock_dtypes.py` and `make_bias_vl.py`.

use half::{bf16, f16};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    pub data_start: u64,
    pub metadata: Option<BTreeMap<String, String>>,
    pub tensors: BTreeMap<String, TensorMeta>,
}

pub fn read_header(path: &Path) -> io::Result<SafetensorsHeader> {
    let mut f = File::open(path)?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let hdr_len = u64::from_le_bytes(len_buf);
    if hdr_len > 256 * 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "safetensors header too large",
        ));
    }
    let mut hdr_bytes = vec![0u8; hdr_len as usize];
    f.read_exact(&mut hdr_bytes)?;
    let raw: BTreeMap<String, Value> = serde_json::from_slice(&hdr_bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut metadata = None;
    let mut tensors = BTreeMap::new();
    for (k, v) in raw {
        if k == "__metadata__" {
            if let Ok(m) = serde_json::from_value::<BTreeMap<String, String>>(v) {
                metadata = Some(m);
            }
        } else {
            let tm: TensorMeta = serde_json::from_value(v)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            tensors.insert(k, tm);
        }
    }
    Ok(SafetensorsHeader {
        data_start: 8 + hdr_len,
        metadata,
        tensors,
    })
}

/// Checks if `dst_bytes` (with `dst_dtype`) equals `src_bytes` (with `src_dtype`) cast to `dst_dtype`.
/// Matches `torch.equal(t.to(dst_dtype), dst_tensor)` (NaN != NaN, exact bit equality after cast).
pub fn is_pure_cast(
    src_dtype: &str,
    src: &[u8],
    dst_dtype: &str,
    dst: &[u8],
) -> Result<bool, String> {
    if src_dtype == dst_dtype {
        return Ok(src == dst);
    }
    match (src_dtype, dst_dtype) {
        ("F16", "F32") => {
            if src.len() * 2 != dst.len() || src.len() % 2 != 0 {
                return Ok(false);
            }
            for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact(4)) {
                let sv = f16::from_le_bytes([s[0], s[1]]).to_f32();
                let dv = f32::from_le_bytes([d[0], d[1], d[2], d[3]]);
                if sv != dv {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        ("BF16", "F32") => {
            if src.len() * 2 != dst.len() || src.len() % 2 != 0 {
                return Ok(false);
            }
            for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact(4)) {
                let sv = bf16::from_le_bytes([s[0], s[1]]).to_f32();
                let dv = f32::from_le_bytes([d[0], d[1], d[2], d[3]]);
                if sv != dv {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        ("F32", "F16") => {
            if src.len() != dst.len() * 2 || dst.len() % 2 != 0 {
                return Ok(false);
            }
            for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact(2)) {
                let sv = f16::from_f32(f32::from_le_bytes([s[0], s[1], s[2], s[3]]));
                let dv = f16::from_le_bytes([d[0], d[1]]);
                if sv.to_f32() != dv.to_f32() {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        ("F32", "BF16") => {
            if src.len() != dst.len() * 2 || dst.len() % 2 != 0 {
                return Ok(false);
            }
            for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact(2)) {
                let sv = bf16::from_f32(f32::from_le_bytes([s[0], s[1], s[2], s[3]]));
                let dv = bf16::from_le_bytes([d[0], d[1]]);
                if sv.to_f32() != dv.to_f32() {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        ("F16", "BF16") => {
            if src.len() != dst.len() || src.len() % 2 != 0 {
                return Ok(false);
            }
            for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact(2)) {
                let sv = bf16::from_f32(f16::from_le_bytes([s[0], s[1]]).to_f32());
                let dv = bf16::from_le_bytes([d[0], d[1]]);
                if sv.to_f32() != dv.to_f32() {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        ("BF16", "F16") => {
            if src.len() != dst.len() || src.len() % 2 != 0 {
                return Ok(false);
            }
            for (s, d) in src.chunks_exact(2).zip(dst.chunks_exact(2)) {
                let sv = f16::from_f32(bf16::from_le_bytes([s[0], s[1]]).to_f32());
                let dv = f16::from_le_bytes([d[0], d[1]]);
                if sv.to_f32() != dv.to_f32() {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Err(format!(
            "unsupported dtype cast check: {src_dtype} -> {dst_dtype}"
        )),
    }
}

pub fn write_safetensors(
    out_path: &Path,
    metadata: Option<&BTreeMap<String, String>>,
    tensors: &[(&str, &str, &[usize], &[u8])],
) -> io::Result<()> {
    let mut hdr = serde_json::Map::new();
    if let Some(meta) = metadata {
        let mut m = serde_json::Map::new();
        for (k, v) in meta {
            m.insert(k.clone(), Value::String(v.clone()));
        }
        hdr.insert("__metadata__".to_string(), Value::Object(m));
    }
    let mut off = 0usize;
    for &(name, dtype, shape, bytes) in tensors {
        let mut t = serde_json::Map::new();
        t.insert("dtype".to_string(), Value::String(dtype.to_string()));
        t.insert(
            "shape".to_string(),
            Value::Array(shape.iter().map(|&d| Value::from(d as u64)).collect()),
        );
        t.insert(
            "data_offsets".to_string(),
            Value::Array(vec![
                Value::from(off as u64),
                Value::from((off + bytes.len()) as u64),
            ]),
        );
        hdr.insert(name.to_string(), Value::Object(t));
        off += bytes.len();
    }
    let hdr_bytes = serde_json::to_vec(&Value::Object(hdr))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let tmp_path = out_path.with_extension(format!(
        "{}.tmp",
        out_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("safetensors")
    ));
    let f = File::create(&tmp_path)?;
    let mut w = BufWriter::with_capacity(4 * 1024 * 1024, f);
    w.write_all(&(hdr_bytes.len() as u64).to_le_bytes())?;
    w.write_all(&hdr_bytes)?;
    for &(_, _, _, bytes) in tensors {
        w.write_all(bytes)?;
    }
    w.flush()?;
    drop(w);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o644))?;
    }
    fs::rename(&tmp_path, out_path)?;
    Ok(())
}

pub fn mmap_file(path: &Path) -> io::Result<Mmap> {
    let f = File::open(path)?;
    unsafe { Mmap::map(&f) }
}

pub fn same_file(a: &Path, b: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (fs::metadata(a), fs::metadata(b)) {
            (Ok(ma), Ok(mb)) => ma.dev() == mb.dev() && ma.ino() == mb.ino(),
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        a == b
    }
}
