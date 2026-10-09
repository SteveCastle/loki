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
    pub fn save_png(&self, path: &Path) -> Result<()> {
        let img = image::RgbImage::from_raw(self.w as u32, self.h as u32, self.data.clone()).context("image buffer")?;
        img.save_with_format(path, image::ImageFormat::Png).with_context(|| format!("saving {}", path.display()))?;
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
