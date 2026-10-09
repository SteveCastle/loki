//! Named presets: a preset bundles a prompt template, an output size policy and a reference-sizing policy.
//! The engine itself (`pipeline`) is a generic Qwen Image 2.1 edit pipeline; `--preset 4kify` is just one of these.
use crate::pipeline::RefPolicy;
use anyhow::{bail, Result};

/// (name, one-line description) of every preset, for `--list-presets` and the help text.
pub const PRESETS: &[(&str, &str)] = &[
    ("4kify", "restore the photo and outpaint it into a 3840x2160 desktop wallpaper"),
    ("4kify-phone", "restore the photo and outpaint it into a 1296x2800 vertical phone wallpaper"),
    ("upscale", "faithful super-resolution of the input (default 2x; set the factor with --scale or --upscale)"),
    ("restore", "faithful restoration (denoise, deblock, de-blur) at the input's own size, no outpainting"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    Landscape,
    Portrait,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Desktop,
    Phone,
    Restore,
    Upscale,
}

#[derive(Clone, Copy, Debug)]
pub struct Preset {
    kind: Kind,
    /// sequence mode (the frames of a clip): stricter "faithful enlargement" wording, one seed for every input,
    /// and (for the wallpaper presets) a framing paragraph that pins the placement
    pub seq: bool,
}

impl Preset {
    pub fn parse(name: &str, seq: bool) -> Result<Preset> {
        let kind = match name {
            "4kify" | "desktop" => Kind::Desktop,
            "4kify-phone" | "phone" => Kind::Phone,
            "restore" => Kind::Restore,
            "upscale" => Kind::Upscale,
            other => bail!("unknown preset '{other}' (available: {})", PRESETS.iter().map(|p| p.0).collect::<Vec<_>>().join(", ")),
        };
        Ok(Preset { kind, seq })
    }

    /// Fixed output size of the preset, if it has one (`restore` follows the input).
    pub fn size(&self) -> Option<(usize, usize)> {
        match self.kind {
            Kind::Desktop => Some((3840, 2160)),
            Kind::Phone => Some((1296, 2800)),
            Kind::Restore | Kind::Upscale => None,
        }
    }

    /// Default output size factor relative to the input when neither --size nor a scaling flag is given.
    pub fn default_scale(&self) -> Option<f64> {
        match self.kind {
            Kind::Upscale => Some(2.0),
            _ => None,
        }
    }

    /// Output file suffix: `<stem><suffix>.png`.
    pub fn suffix(&self) -> &'static str {
        match self.kind {
            Kind::Desktop => "_4k",
            Kind::Phone => "_phone",
            Kind::Restore => "_restored",
            Kind::Upscale => "_up",
        }
    }

    pub fn ref_policy(&self) -> RefPolicy {
        match self.kind {
            Kind::Desktop | Kind::Phone | Kind::Upscale => RefPolicy::Wallpaper,
            Kind::Restore => RefPolicy::Native,
        }
    }

    /// The prompt for an output of `w` x `h`.
    pub fn prompt(&self, w: usize, h: usize) -> String {
        let o = if h > w { Orientation::Portrait } else { Orientation::Landscape };
        match self.kind {
            Kind::Desktop | Kind::Phone => wallpaper_prompt(o, w, h, self.seq),
            Kind::Restore => restore_prompt(self.seq),
            Kind::Upscale => upscale_prompt(),
        }
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn canvas_name(o: Orientation, w: usize, h: usize) -> String {
    let g = gcd(w, h);
    let ratio = format!("{}:{}", w / g, h / g);
    match o {
        Orientation::Landscape => format!("{ratio} landscape desktop wallpaper"),
        Orientation::Portrait => format!("{ratio} portrait phone wallpaper"),
    }
}

/// The restoration paragraph. `fidelity` (sequence mode) asks for a geometrically faithful enlargement: a
/// free restoration re-draws eyes, fingers and contours of a low-resolution frame, which reads as morphing
/// between the frames of a clip.
fn restoration_paragraph(fidelity: bool) -> &'static str {
    if fidelity {
        "Enlarge <image1> into a clean, high-resolution version of the same photograph, as a faithful super-resolution of exactly what <image1> shows. Every edge, contour, and object must stay exactly where it is in <image1>, with the same shape, size, position, and angle: do not redraw, re-pose, re-light, or reinterpret anything, do not open or close eyes or mouths, and do not change the expression, the fingers, the teeth, the tongue, the hair, or any object. Only add the fine detail that the low resolution lost, removing compression artifacts, blockiness, and noise, and keeping intentional blur as it is."
    } else {
        "Restore <image1> into a clean, clear, high-resolution version of the original photograph. Remove grain, unwanted noise, compression artifacts, blockiness, and muddy detail. Correct unintended softness and defocus so the main subject appears naturally in focus, with believable texture and smooth tonal transitions. Preserve intentional depth of field and background blur."
    }
}

const KEEP_PARAGRAPH: &str = "Keep the original subject identity, facial features, expression, pose, proportions, clothing, objects, and scene details. Preserve the photograph\u{2019}s original lighting, colors, exposure, and atmosphere. Use faithful restoration without beauty retouching, stylization, artificial textures, digital sharpening, halos, or exaggerated contrast.";

/// The framing paragraph of sequence mode: every frame of a clip must land on the canvas identically, so
/// the placement is spelled out (independently restored frames otherwise drift in zoom by about +-15%).
fn framing_paragraph(o: Orientation) -> &'static str {
    match o {
        Orientation::Landscape => "This is one frame of a video clip whose frames are processed one by one with the same settings, so every frame must be framed identically. Place <image1> at the centre of the canvas at the one scale where its full height exactly fills the height of the output, cropping nothing of it, so that the added surroundings lie only to its left and right. Do not zoom in or out, shift, rotate, or re-crop: scaled to the same size, the output must align exactly with <image1>.",
        Orientation::Portrait => "This is one frame of a video clip whose frames are processed one by one with the same settings, so every frame must be framed identically. Place <image1> at the centre of the canvas at the one scale where its full width exactly fills the width of the output, cropping nothing of it, so that the added surroundings lie only above and below it. Do not zoom in or out, shift, rotate, or re-crop: scaled to the same size, the output must align exactly with <image1>.",
    }
}

/// The restoration + outpainting prompt of the wallpaper presets, parameterized by the target orientation;
/// `seq` picks the strict restoration wording and adds the framing paragraph.
pub fn wallpaper_prompt(o: Orientation, w: usize, h: usize, seq: bool) -> String {
    let canvas = canvas_name(o, w, h);
    let extend = match o {
        Orientation::Landscape => "to the left and right of the original image, and above and below where needed",
        Orientation::Portrait => "above and below the original image, and along the sides where needed",
    };
    let composition = match o {
        Orientation::Landscape => "landscape",
        Orientation::Portrait => "portrait",
    };
    let restoration = restoration_paragraph(seq);
    let framing = if seq { format!("{}\n\n", framing_paragraph(o)) } else { String::new() };
    format!(
        "{restoration}\n\n\
{KEEP_PARAGRAPH}\n\n\
Expand the canvas into a {canvas} by outpainting beyond the boundaries of <image1>. Keep the original image region intact in its content and geometry, applying only the requested restoration within it. Preserve the subjects\u{2019} original size and spatial relationships. Do not stretch, squeeze, warp, crop, reposition, or redraw the original scene to fit the new ratio.\n\n\
Create the additional space by extending the surrounding environment {extend}. Continue the existing perspective, lighting, textures, depth of field, and background structures seamlessly. Add only plausible environmental continuation, without introducing new focal subjects or duplicating existing people or objects.\n\n\
{framing}\
Deliver a clean, faithfully restored photograph within a naturally expanded {composition} composition, with seamless transitions between the original image and the added surroundings."
    )
}

/// Restoration only: same framing and size as the input.
pub fn restore_prompt(seq: bool) -> String {
    format!(
        "{}\n\n{KEEP_PARAGRAPH}\n\nKeep the framing, composition, and aspect ratio of <image1> exactly: do not crop, extend, reposition, or redraw the scene. Deliver a clean, faithfully restored photograph.",
        restoration_paragraph(seq)
    )
}

/// Faithful super-resolution: same content and framing at a larger size, only the lost detail is added.
pub fn upscale_prompt() -> String {
    format!(
        "{}\n\n{KEEP_PARAGRAPH}\n\nKeep the framing, composition, and aspect ratio of <image1> exactly: every edge and contour must stay exactly where it is, and the output must align with <image1> when scaled to the same size. Do not crop, extend, reposition, or redraw anything.",
        restoration_paragraph(true)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn presets_parse_and_describe() {
        for (name, _) in PRESETS {
            let p = Preset::parse(name, false).unwrap();
            assert!(p.prompt(1000, 600).contains("<image1>"));
        }
        assert!(Preset::parse("nope", false).is_err());
        assert_eq!(Preset::parse("4kify", false).unwrap().size(), Some((3840, 2160)));
        assert!(Preset::parse("4kify", false).unwrap().prompt(3840, 2160).contains("16:9 landscape desktop wallpaper"));
        assert!(Preset::parse("4kify-phone", true).unwrap().prompt(1296, 2800).contains("one frame of a video clip"));
    }
}
