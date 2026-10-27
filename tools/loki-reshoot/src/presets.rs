//! Prompt presets for common jobs, written in the structure of MiniMax's full-reference prompt guide
//! (`--prompt-guide`). The first reference image is `<Picture 1>`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shake {
    None,
    Subtle,
    Handheld,
}

/// "Bring a still photo to life": natural ambient life, subtle resting movement, optional camera shake, no new content.
/// `describe` optionally names the subject/setting (strongly recommended: it anchors identity and what may move);
/// `extra` is appended as additional direction. When either says the backdrop is plain (solid white, studio
/// backdrop, isolated...) the environment and sound lines switch to a plain-backdrop variant: the generic "the place
/// is alive" wording otherwise makes the model paint a room or scenery behind the subject.
pub fn animate_prompt(describe: Option<&str>, shake: Shake, extra: Option<&str>, padded: bool) -> String {
    let plain = describe.is_some_and(plain_backdrop) || extra.is_some_and(plain_backdrop);
    let what = describe.map(|d| d.trim().trim_end_matches('.')).filter(|d| !d.is_empty()).unwrap_or("the person or subject and the setting shown");
    let camera = match shake {
        Shake::None => "The camera is perfectly steady, as if locked off on a tripod, with no movement at all.",
        Shake::Subtle => "The camera has a very subtle, natural handheld sway: tiny, slow, low-amplitude drift and micro-shake like a person holding a phone still, only a few pixels of motion, with no zoom, no pan and no change of framing.",
        Shake::Handheld => "The camera is held by hand with a clearly visible but gentle natural shake and slow drift, keeping the same framing, with no zoom and no pan.",
    };
    let extra = extra.map(|e| format!(" {}", e.trim())).unwrap_or_default();
    let bars = if padded {
        " The thin black bars at the edges of <Picture 1> are padding, not part of the scene: from the first frame on, the video seamlessly extends the photographed scene into those areas, matching its lighting, textures and perspective, so the frame is completely filled with no black bars."
    } else {
        ""
    };
    let (life, environment, sound, soundscape) = if plain {
        (
            "<Subject 1> shows only natural life and subtle resting movement against the unchanged plain backdrop",
            "The background is the same plain, uniform, solid backdrop as in <Picture 1> and stays exactly that in every frame: completely empty and flat in the same color, with no room, walls, floor, horizon, scenery, furniture, props, texture, gradient, vignette or new shadows ever appearing; only the subject moves.",
            "The sound is a soft, clean, neutral studio ambience: quiet room tone and the subtle rustle of skin and fabric as the subject breathes and shifts; it is soft but present throughout, there are no environmental sounds, nobody speaks, and there is no music.",
            "Soft, clean neutral studio room tone throughout, with subtle breathing and fabric rustle; no environmental sounds, no speech.",
        )
    } else {
        (
            "<Subject 1> and the environment show only natural ambient life and subtle resting movement",
            "The environment is alive too, but only through elements already visible in <Picture 1>: light shifts softly and what is there moves subtly the way it would in the real place, always gentle and realistic; a plain or empty background stays plain and empty.",
            "The sound is a clearly audible, natural, realistic ambient soundscape of the depicted place: soft room tone, gentle air movement, faint distant sounds that fit the setting, the subtle rustle of skin and fabric as the subject breathes and shifts, and slight handling sounds of the camera or phone; it is soft but present throughout, nobody speaks, and there is no music.",
            "Clearly audible, soft natural ambience of the depicted place throughout: room tone, gentle air movement, faint distant environmental sounds, subtle breathing and fabric rustle; no speech.",
        )
    };
    format!(
"subject_definitions:
<Subject 1> is {what} in <Picture 1>, to be kept exactly as shown: identity, face, expression, body proportions, hair, skin, clothing, accessories, pose, framing, colors, lighting and photographic rendering.

summary:
[keyframe completion] The target video is a faithful, photorealistic living-photo animation of <Picture 1>: one continuous shot that starts exactly from <Picture 1> in which {life}.{bars}{extra}

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identity, appearance, clothing, pose, setting and framing are retained.
<Picture 1> ([Shot 1] first frame): fully_preserved - the video begins from this exact composition and keeps its look.

detailed_description:
The target video keeps the photorealistic look, colors, natural lighting and framing of <Picture 1>, like a few seconds of real footage of the same moment.
[Shot 1] The shot begins exactly from <Picture 1> and stays one uninterrupted, unedited take with the same composition. <Subject 1> is at rest and shows only small natural life: slow relaxed breathing that gently lifts the chest and shoulders, tiny weight shifts, soft natural blinks, a faint subtle change of expression, small relaxed head and hand micro-movements, and hair and clothing reacting slightly to the air. {environment} Nothing new enters the frame, no new people or objects appear, the pose is not changed, and there is no dramatic motion. {camera} {sound}

overall_soundscape:
{soundscape}

non_diegetic_music:
N/A
")
}

/// Does this text say the photo's backdrop is plain (solid colour, studio sweep, cut-out)?
fn plain_backdrop(text: &str) -> bool {
    const PHRASES: &[&str] = &[
        "solid white", "solid black", "solid grey", "solid gray", "solid colo", "solid background", "solid backdrop",
        "plain white", "plain background", "plain backdrop", "blank background", "empty background", "no background",
        "white background", "black background", "grey background", "gray background", "white backdrop",
        "studio backdrop", "seamless backdrop", "seamless background", "white void", "white space", "white studio",
        "isolated on", "transparent background", "green screen", "greenscreen", "chroma key",
    ];
    let t = text.to_lowercase();
    PHRASES.iter().any(|p| t.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompt_has_sections() {
        let p = animate_prompt(Some("the young woman taking a mirror selfie"), Shake::Subtle, None, true);
        for s in ["subject_definitions:", "summary:", "retention_analysis:", "detailed_description:", "overall_soundscape:", "non_diegetic_music:"] {
            assert!(p.contains(s));
        }
        assert!(p.contains("<Subject 1> is the young woman taking a mirror selfie in <Picture 1>"));
        assert!(!p.contains("plain, uniform, solid backdrop"));
    }

    #[test]
    fn plain_backdrop_drops_scenery() {
        for (d, e) in [
            (Some("an old man reading a newspaper. The background is a solid white space"), None),
            (Some("an old man reading a newspaper"), Some("Keep the plain white background.")),
        ] {
            let p = animate_prompt(d, Shake::Subtle, e, false);
            assert!(p.contains("plain, uniform, solid backdrop"), "{p}");
            for bad in ["foliage", "curtains", "outdoor", "depicted place"] {
                assert!(!p.contains(bad), "{bad} in {p}");
            }
        }
    }
}
