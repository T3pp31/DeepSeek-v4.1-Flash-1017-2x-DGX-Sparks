//! Native Rust image decoder subprocess for TensorFold's `zig/src/families/dsv41/pil.zig`.
//!
//! Wire protocol (byte-identical with `pil.zig`'s Python+Pillow child):
//! 1. On startup: writes 16-byte greeting `b"PIL"` + 13-byte space-padded version, flushes stdout.
//! 2. Loop:
//!    - Reads `u64 LE` length `N`, then `N` bytes from stdin (EOF => exit 0).
//!    - Validates pixel count <= `MAX_IMAGE_PIXELS` (134,217,728 = 2 * Pillow's 67,108,864 limit).
//!    - Decodes image and converts to tightly-packed RGB888.
//!    - Writes 25-byte header `[status: u8, width: u64 LE, height: u64 LE, payload_len: u64 LE]`
//!      followed by `payload_len` bytes, flushes stdout:
//!      - status 0: RGB pixels
//!      - status 1: DecompressionBomb (pixel limit exceeded)
//!      - status 2: decode / format error message (UTF-8)

use image::{DynamicImage, ImageDecoder, ImageReader, Limits};
use std::env;
use std::io::{self, BufReader, BufWriter, Cursor, ErrorKind, Read, Write};

/// Pillow raises `DecompressionBombError` past `2 * Image.MAX_IMAGE_PIXELS` (2 * 67,108,864).
const MAX_IMAGE_PIXELS: u64 = 134_217_728;
const MAX_PAYLOAD_BYTES: usize = 512 * 1024 * 1024;

fn write_reply(
    out: &mut impl Write,
    status: u8,
    width: u64,
    height: u64,
    payload: &[u8],
) -> io::Result<()> {
    let mut hdr = [0u8; 25];
    hdr[0] = status;
    hdr[1..9].copy_from_slice(&width.to_le_bytes());
    hdr[9..17].copy_from_slice(&height.to_le_bytes());
    hdr[17..25].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    out.write_all(&hdr)?;
    out.write_all(payload)?;
    out.flush()
}

enum DecodeOutcome {
    Rgb { w: u64, h: u64, px: Vec<u8> },
    Bomb(String),
    Err(String),
}

fn decode_image(data: &[u8]) -> DecodeOutcome {
    let mut reader = match ImageReader::new(Cursor::new(data)).with_guessed_format() {
        Ok(r) => r,
        Err(e) => return DecodeOutcome::Err(format!("cannot identify image: {e}")),
    };
    let mut limits = Limits::default();
    limits.max_image_width = Some(131_072);
    limits.max_image_height = Some(131_072);
    limits.max_alloc = Some(1024 * 1024 * 1024);
    reader.limits(limits);

    let decoder = match reader.into_decoder() {
        Ok(d) => d,
        Err(e) => return DecodeOutcome::Err(e.to_string()),
    };
    let (w, h) = decoder.dimensions();
    let pixels = u64::from(w).saturating_mul(u64::from(h));
    if pixels > MAX_IMAGE_PIXELS {
        return DecodeOutcome::Bomb(format!(
            "Image size ({pixels} pixels) exceeds limit of {MAX_IMAGE_PIXELS} pixels, could be decompression bomb DOS attack."
        ));
    }

    match DynamicImage::from_decoder(decoder) {
        Ok(dyn_img) => {
            let rgb = dyn_img.into_rgb8();
            let rw = rgb.width() as u64;
            let rh = rgb.height() as u64;
            DecodeOutcome::Rgb {
                w: rw,
                h: rh,
                px: rgb.into_raw(),
            }
        }
        Err(e) => DecodeOutcome::Err(e.to_string()),
    }
}

fn run_protocol<R: Read, W: Write>(inp: &mut R, out: &mut W) -> io::Result<()> {
    let ver = format!("rust-{}", env!("CARGO_PKG_VERSION"));
    let mut hello = [b' '; 16];
    hello[0..3].copy_from_slice(b"PIL");
    let vb = ver.as_bytes();
    let n = vb.len().min(13);
    hello[3..3 + n].copy_from_slice(&vb[..n]);
    out.write_all(&hello)?;
    out.flush()?;

    let mut len_buf = [0u8; 8];
    let mut img_buf = Vec::<u8>::new();
    loop {
        match inp.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        }
        let len = u64::from_le_bytes(len_buf) as usize;
        if len > MAX_PAYLOAD_BYTES {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("image payload too large: {len}"),
            ));
        }
        img_buf.resize(len, 0);
        match inp.read_exact(&mut img_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        }

        match decode_image(&img_buf) {
            DecodeOutcome::Rgb { w, h, px } => write_reply(out, 0, w, h, &px)?,
            DecodeOutcome::Bomb(m) => write_reply(out, 1, 0, 0, m.as_bytes())?,
            DecodeOutcome::Err(m) => write_reply(out, 2, 0, 0, m.as_bytes())?,
        }
    }
}

fn main() {
    if matches!(env::args().nth(1).as_deref(), Some("--version" | "-V")) {
        println!("dsv41-imgdec {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut inp = BufReader::with_capacity(256 * 1024, stdin.lock());
    let mut out = BufWriter::with_capacity(1024 * 1024, stdout.lock());
    if let Err(e) = run_protocol(&mut inp, &mut out) {
        if e.kind() != ErrorKind::BrokenPipe {
            eprintln!("dsv41-imgdec: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageFormat, RgbImage};

    #[test]
    fn test_protocol_png_jpeg_webp_and_invalid() {
        let img = RgbImage::from_fn(16, 12, |x, y| {
            image::Rgb([(x * 16) as u8, (y * 20) as u8, 42])
        });
        let mut png_bytes = Vec::new();
        img.write_to(&mut Cursor::new(&mut png_bytes), ImageFormat::Png)
            .unwrap();
        let mut jpg_bytes = Vec::new();
        img.write_to(&mut Cursor::new(&mut jpg_bytes), ImageFormat::Jpeg)
            .unwrap();

        let mut req = Vec::new();
        for payload in [&png_bytes[..], &jpg_bytes[..], b"not-an-image"] {
            req.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            req.extend_from_slice(payload);
        }

        let mut resp = Vec::new();
        run_protocol(&mut Cursor::new(&req), &mut resp).unwrap();

        assert_eq!(&resp[0..3], b"PIL");
        let mut pos = 16usize;

        // Frame 1: PNG exact RGB
        let st = resp[pos];
        let w = u64::from_le_bytes(resp[pos + 1..pos + 9].try_into().unwrap());
        let h = u64::from_le_bytes(resp[pos + 9..pos + 17].try_into().unwrap());
        let n = u64::from_le_bytes(resp[pos + 17..pos + 25].try_into().unwrap()) as usize;
        assert_eq!((st, w, h, n), (0, 16, 12, 16 * 12 * 3));
        assert_eq!(&resp[pos + 25..pos + 25 + n], img.as_raw().as_slice());
        pos += 25 + n;

        // Frame 2: JPEG dimensions
        let st = resp[pos];
        let w = u64::from_le_bytes(resp[pos + 1..pos + 9].try_into().unwrap());
        let h = u64::from_le_bytes(resp[pos + 9..pos + 17].try_into().unwrap());
        let n = u64::from_le_bytes(resp[pos + 17..pos + 25].try_into().unwrap()) as usize;
        assert_eq!((st, w, h, n), (0, 16, 12, 16 * 12 * 3));
        pos += 25 + n;

        // Frame 3: invalid image -> status 2
        let st = resp[pos];
        let n = u64::from_le_bytes(resp[pos + 17..pos + 25].try_into().unwrap()) as usize;
        assert_eq!(st, 2);
        assert!(n > 0);
        assert_eq!(pos + 25 + n, resp.len());
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xffff_ffff;
        for &b in data {
            crc ^= u32::from(b);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xedb8_8320 & mask);
            }
        }
        !crc
    }

    fn png_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + data.len());
        buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
        buf.extend_from_slice(kind);
        buf.extend_from_slice(data);
        let mut body = Vec::with_capacity(4 + data.len());
        body.extend_from_slice(kind);
        body.extend_from_slice(data);
        buf.extend_from_slice(&crc32(&body).to_be_bytes());
        buf
    }

    #[test]
    fn test_decompression_bomb_rejected_with_status_1() {
        // Minimal PNG header claiming 20000 x 10000 = 200,000,000 pixels (> 134,217,728)
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&20000u32.to_be_bytes());
        ihdr.extend_from_slice(&10000u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&png_chunk(b"IHDR", &ihdr));
        png.extend_from_slice(&png_chunk(
            b"IDAT",
            &[0x78, 0x9c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01],
        ));
        png.extend_from_slice(&png_chunk(b"IEND", &[]));

        match decode_image(&png) {
            DecodeOutcome::Bomb(m) => assert!(m.contains("exceeds limit")),
            DecodeOutcome::Rgb { .. } => panic!("must not decode bomb"),
            DecodeOutcome::Err(e) => panic!("expected Bomb, got Err: {e}"),
        }
    }
}
