//! Minimal deterministic RGB PNG encoder (zero external image-crate overhead for tests/tools).

use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::Write;

fn chunk(tag: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + data.len());
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    let mut h = crc32fast::Hasher::new();
    h.update(tag);
    h.update(data);
    out.extend_from_slice(&h.finalize().to_be_bytes());
    out
}

pub fn encode_rgb_png<F>(w: u32, h: u32, mut pixel: F) -> Vec<u8>
where
    F: FnMut(u32, u32) -> [u8; 3],
{
    let mut raw = Vec::with_capacity((h as usize) * (1 + (w as usize) * 3));
    for y in 0..h {
        raw.push(0);
        for x in 0..w {
            let p = pixel(x, y);
            raw.extend_from_slice(&p);
        }
    }
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(&raw).unwrap();
    let compressed = enc.finish().unwrap();

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);

    let mut out = Vec::with_capacity(8 + 25 + compressed.len() + 12 + 12);
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    out.extend_from_slice(&chunk(b"IHDR", &ihdr));
    out.extend_from_slice(&chunk(b"IDAT", &compressed));
    out.extend_from_slice(&chunk(b"IEND", b""));
    out
}
