//! Supported aspect ratios and canvas fitting (port of the custom ComfyUI `MiniMaxH3AutoRatio` node).
use crate::image::Rgb8;
use crate::pipeline::adapt_canvas;

/// The ratios the model is built around (square, 4:3, 16:9 in both orientations).
pub const SUPPORTED_RATIOS: [(&str, f64); 5] = [("1:1", 1.0), ("4:3", 4.0 / 3.0), ("3:4", 3.0 / 4.0), ("16:9", 16.0 / 9.0), ("9:16", 9.0 / 16.0)];

/// Supported ratio with the smallest log-distance to w/h.
pub fn best_ratio(w: usize, h: usize) -> (&'static str, f64) {
    let a = w as f64 / h as f64;
    *SUPPORTED_RATIOS.iter().min_by(|x, y| (a / x.1).ln().abs().partial_cmp(&(a / y.1).ln().abs()).unwrap()).unwrap()
}

/// Native canvas for a ratio: 768 short edge, area capped at 768*1344, per-axis rounded to 32
/// (1:1 768x768, 4:3 1024x768, 3:4 768x1024, 16:9 1344x768, 9:16 768x1344).
pub fn canvas_for_ratio(ratio: f64) -> (usize, usize) {
    adapt_canvas((ratio * 10000.0).round() as usize, 10000)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    Pad,
    Crop,
    Stretch,
}

/// Size the image occupies inside the canvas with `Fit::Pad` (aspect kept, centred).
pub fn pad_extent(ow: usize, oh: usize, cw: usize, ch: usize) -> (usize, usize) {
    let s = (cw as f64 / ow as f64).min(ch as f64 / oh as f64);
    (((ow as f64 * s).round() as usize).max(1), ((oh as f64 * s).round() as usize).max(1))
}

/// Fit an image to the canvas by padding with black bars, centre-cropping or stretching. Returns (image, padded?).
pub fn fit_to_canvas(img: &Rgb8, cw: usize, ch: usize, fit: Fit) -> (Rgb8, bool) {
    match fit {
        Fit::Stretch => (img.resize_lanczos(cw, ch), false),
        Fit::Pad => {
            let (nw, nh) = pad_extent(img.w, img.h, cw, ch);
            let r = img.resize_lanczos(nw, nh);
            if (nw, nh) == (cw, ch) {
                return (r, false);
            }
            let mut out = Rgb8 { w: cw, h: ch, data: vec![0u8; cw * ch * 3] };
            let (x0, y0) = ((cw - nw) / 2, (ch - nh) / 2);
            for y in 0..nh {
                let d = ((y0 + y) * cw + x0) * 3;
                out.data[d..d + nw * 3].copy_from_slice(&r.data[y * nw * 3..(y + 1) * nw * 3]);
            }
            (out, true)
        }
        Fit::Crop => {
            // scale to cover, then centre-crop
            let s = (cw as f64 / img.w as f64).max(ch as f64 / img.h as f64);
            let (nw, nh) = (((img.w as f64 * s).round() as usize).max(cw), ((img.h as f64 * s).round() as usize).max(ch));
            let r = img.resize_lanczos(nw, nh);
            let (x0, y0) = ((nw - cw) / 2, (nh - ch) / 2);
            let mut out = Rgb8 { w: cw, h: ch, data: vec![0u8; cw * ch * 3] };
            for y in 0..ch {
                let sidx = ((y0 + y) * nw + x0) * 3;
                out.data[y * cw * 3..(y + 1) * cw * 3].copy_from_slice(&r.data[sidx..sidx + cw * 3]);
            }
            (out, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn auto_ratio() {
        assert_eq!(best_ratio(973, 1646).0, "9:16");
        assert_eq!(canvas_for_ratio(best_ratio(973, 1646).1), (768, 1344));
        assert_eq!(canvas_for_ratio(1.0), (768, 768));
        assert_eq!(canvas_for_ratio(4.0 / 3.0), (1024, 768));
        assert_eq!(canvas_for_ratio(3.0 / 4.0), (768, 1024));
        assert_eq!(canvas_for_ratio(16.0 / 9.0), (1344, 768));
        assert_eq!(best_ratio(1080, 1350).0, "3:4");
        assert_eq!(pad_extent(973, 1646, 768, 1344), (768, 1299));
    }
}
