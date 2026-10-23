//! Image loading, PIL-compatible Lanczos resizing, and PNG output.
use anyhow::{Context, Result};
use std::path::Path;

/// 8-bit RGB image, row-major, interleaved.
#[derive(Clone)]
pub struct Rgb8 {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Rgb8 {
    pub fn load(path: &Path) -> Result<Rgb8> {
        let img = image::ImageReader::open(path)
            .with_context(|| format!("opening {}", path.display()))?
            .with_guessed_format()?
            .decode()
            .with_context(|| format!("decoding {}", path.display()))?;
        let rgb = img.to_rgb8();
        Ok(Rgb8 { w: rgb.width() as usize, h: rgb.height() as usize, data: rgb.into_raw() })
    }
    /// Decode an image from memory (any format the `image` crate was built with).
    pub fn from_bytes(bytes: &[u8]) -> Result<Rgb8> {
        let img = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?.decode().context("decoding image")?;
        let rgb = img.to_rgb8();
        Ok(Rgb8 { w: rgb.width() as usize, h: rgb.height() as usize, data: rgb.into_raw() })
    }
    /// PNG-encoded bytes (for writing to stdout). `text` entries become `iTXt` chunks (keyword, UTF-8 text) placed
    /// before the image data, where metadata readers look without decoding the picture.
    pub fn png_bytes(&self, text: &[(String, String)]) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.data.len() / 2);
        let mut enc = png::Encoder::new(&mut out, self.w as u32, self.h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        for (k, v) in text {
            enc.add_itxt_chunk(k.clone(), v.clone()).with_context(|| format!("png text chunk {k}"))?;
        }
        let mut w = enc.write_header().context("encoding png")?;
        w.write_image_data(&self.data).context("encoding png")?;
        w.finish().context("encoding png")?;
        Ok(out)
    }
    pub fn save_png(&self, path: &Path, text: &[(String, String)]) -> Result<()> {
        let bytes = self.png_bytes(text)?;
        std::fs::write(path, bytes).with_context(|| format!("saving {}", path.display()))?;
        Ok(())
    }
    /// Pixels as f32 in [0, 1], HWC.
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|v| *v as f32 / 255.0).collect()
    }
    pub fn from_f32(w: usize, h: usize, px: &[f32]) -> Rgb8 {
        let data = px.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
        Rgb8 { w, h, data }
    }

    /// Lanczos (a=3) resize reproducing PIL's `Image.resize(..., LANCZOS)` on 8-bit data:
    /// separable horizontal then vertical pass, each rounded to u8, antialiasing on downscale.
    pub fn resize_lanczos(&self, nw: usize, nh: usize) -> Rgb8 {
        if nw == self.w && nh == self.h {
            return self.clone();
        }
        let mut cur = self.clone();
        if nw != cur.w {
            cur = resample_axis(&cur, nw, true);
        }
        if nh != cur.h {
            cur = resample_axis(&cur, nh, false);
        }
        cur
    }
}

fn lanczos3(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    if x.abs() >= 3.0 {
        return 0.0;
    }
    let px = std::f64::consts::PI * x;
    (px.sin() / px) * ((px / 3.0).sin() / (px / 3.0))
}

/// Coefficients per output index: (start, weights)
fn coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 3.0 * filterscale;
    let mut out = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ww_lo = (center - support + 0.5).floor().max(0.0) as isize;
        let xmin = ww_lo as usize;
        let mut xmax = (center + support + 0.5).floor() as usize;
        if xmax > in_size {
            xmax = in_size;
        }
        let mut w = Vec::with_capacity(xmax - xmin);
        let mut sum = 0.0;
        for x in xmin..xmax {
            let v = lanczos3((x as f64 - center + 0.5) / filterscale);
            w.push(v);
            sum += v;
        }
        if sum != 0.0 {
            for v in w.iter_mut() {
                *v /= sum;
            }
        }
        out.push((xmin, w));
    }
    out
}

fn resample_axis(img: &Rgb8, new_size: usize, horizontal: bool) -> Rgb8 {
    let (w, h) = (img.w, img.h);
    let (nw, nh) = if horizontal { (new_size, h) } else { (w, new_size) };
    let co = coeffs(if horizontal { w } else { h }, new_size);
    let mut data = vec![0u8; nw * nh * 3];
    if horizontal {
        for y in 0..h {
            let row = &img.data[y * w * 3..(y + 1) * w * 3];
            for (ox, (xmin, ws)) in co.iter().enumerate() {
                for c in 0..3 {
                    let mut acc = 0.0f64;
                    for (k, wv) in ws.iter().enumerate() {
                        acc += row[(xmin + k) * 3 + c] as f64 * wv;
                    }
                    data[(y * nw + ox) * 3 + c] = clip8(acc);
                }
            }
        }
    } else {
        for (oy, (ymin, ws)) in co.iter().enumerate() {
            for x in 0..w {
                for c in 0..3 {
                    let mut acc = 0.0f64;
                    for (k, wv) in ws.iter().enumerate() {
                        acc += img.data[((ymin + k) * w + x) * 3 + c] as f64 * wv;
                    }
                    data[(oy * nw + x) * 3 + c] = clip8(acc);
                }
            }
        }
    }
    Rgb8 { w: nw, h: nh, data }
}

fn clip8(v: f64) -> u8 {
    // PIL: fixed-point with rounding half up
    let r = (v + 0.5).floor();
    r.clamp(0.0, 255.0) as u8
}

/// A soft single-channel mask in [0, 1] (white = 1), row-major.
#[derive(Clone)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Mask {
    /// Load any image as a mask: its luminance (an image with an alpha channel and no visible content uses the alpha).
    pub fn load(path: &Path) -> Result<Mask> {
        let img = image::ImageReader::open(path)
            .with_context(|| format!("opening {}", path.display()))?
            .with_guessed_format()?
            .decode()
            .with_context(|| format!("decoding {}", path.display()))?;
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgba = img.to_rgba8();
        let px: Vec<[u8; 4]> = rgba.pixels().map(|p| p.0).collect();
        let rgb_flat = px.iter().all(|p| p[0] == px[0][0] && p[1] == px[0][1] && p[2] == px[0][2]);
        let has_alpha = img.color().has_alpha() && px.iter().any(|p| p[3] != 255);
        let data = if has_alpha && rgb_flat {
            px.iter().map(|p| p[3] as f32 / 255.0).collect()
        } else {
            px.iter().map(|p| (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0).collect()
        };
        Ok(Mask { w, h, data })
    }
    /// Lanczos resize (through the 8-bit RGB path, which is plenty for a mask).
    pub fn resize(&self, nw: usize, nh: usize) -> Mask {
        if (nw, nh) == (self.w, self.h) {
            return self.clone();
        }
        let rgb = Rgb8 { w: self.w, h: self.h, data: self.data.iter().flat_map(|v| { let b = (v.clamp(0.0, 1.0) * 255.0).round() as u8; [b, b, b] }).collect() };
        let r = rgb.resize_lanczos(nw, nh);
        Mask { w: nw, h: nh, data: r.data.chunks_exact(3).map(|c| c[0] as f32 / 255.0).collect() }
    }
    /// Max over `f` x `f` cells (w, h multiples of f): the mask at latent resolution. A cell touched by the mask is
    /// regenerated as strongly as its strongest pixel, so a soft edge never pins the old content inside the region (the
    /// soft blend itself comes from the full-resolution composite).
    pub fn max_down(&self, f: usize) -> Mask {
        let (nw, nh) = (self.w / f, self.h / f);
        let mut data = vec![0f32; nw * nh];
        for y in 0..nh {
            for x in 0..nw {
                let mut s = 0f32;
                for dy in 0..f {
                    for dx in 0..f {
                        s = s.max(self.data[(y * f + dy) * self.w + x * f + dx]);
                    }
                }
                data[y * nw + x] = s;
            }
        }
        Mask { w: nw, h: nh, data }
    }
    /// Average over `f` x `f` cells (w, h multiples of f).
    pub fn box_down(&self, f: usize) -> Mask {
        let (nw, nh) = (self.w / f, self.h / f);
        let mut data = vec![0f32; nw * nh];
        for y in 0..nh {
            for x in 0..nw {
                let mut s = 0.0;
                for dy in 0..f {
                    for dx in 0..f {
                        s += self.data[(y * f + dy) * self.w + x * f + dx];
                    }
                }
                data[y * nw + x] = s / (f * f) as f32;
            }
        }
        Mask { w: nw, h: nh, data }
    }
}

/// out = m * a + (1 - m) * b per pixel (soft composite), all the same size.
pub fn composite(a: &Rgb8, b: &Rgb8, m: &Mask) -> Rgb8 {
    let mut data = vec![0u8; a.data.len()];
    for i in 0..a.w * a.h {
        let k = m.data[i].clamp(0.0, 1.0);
        for c in 0..3 {
            let v = k * a.data[i * 3 + c] as f32 + (1.0 - k) * b.data[i * 3 + c] as f32;
            data[i * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    Rgb8 { w: a.w, h: a.h, data }
}

/// ComfyUI TextEncodeQwenImage21 sizing: resolution 0 keeps the image size rounded to a multiple of 32.
pub fn reference_size(w: usize, h: usize, resolution: usize) -> (usize, usize) {
    let (mut nw, mut nh);
    if resolution > 0 {
        let ratio = w as f64 / h as f64;
        let r2 = (resolution * resolution) as f64;
        nw = ((r2 * ratio).sqrt() / 32.0).round() as usize * 32;
        nh = ((r2 / ratio).sqrt() / 32.0).round() as usize * 32;
    } else {
        nw = (w as f64 / 32.0).round() as usize * 32;
        nh = (h as f64 / 32.0).round() as usize * 32;
    }
    nw = nw.max(32);
    nh = nh.max(32);
    (nw, nh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_round_trips_pixels_and_text() {
        let img = Rgb8 { w: 3, h: 2, data: (0..18u8).collect() };
        let text = vec![("loki".to_string(), r#"{"loki":1,"tool":"loki-retouch","prompt":"über — ☃"}"#.to_string())];
        let bytes = img.png_bytes(&text).unwrap();
        let back = Rgb8::from_bytes(&bytes).unwrap();
        assert_eq!((back.w, back.h), (3, 2));
        assert_eq!(back.data, img.data);
        // the text chunk is an iTXt before IDAT, readable without decoding the image
        let idat = bytes.windows(4).position(|w| w == b"IDAT").unwrap();
        let itxt = bytes.windows(4).position(|w| w == b"iTXt").unwrap();
        assert!(itxt < idat);
        let dec = png::Decoder::new(std::io::Cursor::new(&bytes)).read_info().unwrap();
        let chunks = &dec.info().utf8_text;
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].keyword, "loki");
        assert_eq!(chunks[0].get_text().unwrap(), text[0].1);
        // no text: plain PNG
        assert!(!img.png_bytes(&[]).unwrap().windows(4).any(|w| w == b"iTXt"));
    }
}
