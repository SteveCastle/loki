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
/// `extra` is appended as additional direction.
pub fn animate_prompt(describe: Option<&str>, shake: Shake, extra: Option<&str>, padded: bool) -> String {
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
    format!(
"subject_definitions:
<Subject 1> is {what} in <Picture 1>, to be kept exactly as shown: identity, face, expression, body proportions, hair, skin, clothing, accessories, pose, framing, colors, lighting and photographic rendering.

summary:
[keyframe completion] The target video is a faithful, photorealistic living-photo animation of <Picture 1>: one continuous shot that starts exactly from <Picture 1> in which <Subject 1> and the environment show only natural ambient life and subtle resting movement.{bars}{extra}

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identity, appearance, clothing, pose, setting and framing are retained.
<Picture 1> ([Shot 1] first frame): fully_preserved - the video begins from this exact composition and keeps its look.

detailed_description:
The target video keeps the photorealistic look, colors, natural lighting and framing of <Picture 1>, like a few seconds of real footage of the same moment.
[Shot 1] The shot begins exactly from <Picture 1> and stays one uninterrupted, unedited take with the same composition. <Subject 1> is at rest and shows only small natural life: slow relaxed breathing that gently lifts the chest and shoulders, tiny weight shifts, soft natural blinks, a faint subtle change of expression, small relaxed head and hand micro-movements, and hair and clothing reacting slightly to the air. The environment is alive too: light shifts softly, background elements move subtly the way they would in the real place (foliage, curtains, reflections, dust or haze in the light, distant movement outside), always gentle and realistic. Nothing new enters the frame, no new people or objects appear, the pose is not changed, and there is no dramatic motion. {camera} The sound is a clearly audible, natural, realistic ambient soundscape of the depicted place: soft room tone, gentle air movement, faint distant outdoor sounds, the subtle rustle of skin and fabric as the subject breathes and shifts, and slight handling sounds of the camera or phone; it is soft but present throughout, nobody speaks, and there is no music.

overall_soundscape:
Clearly audible, soft natural ambience of the depicted place throughout: room tone, gentle air movement, faint distant environmental sounds, subtle breathing and fabric rustle; no speech.

non_diegetic_music:
N/A
")
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
    }
}
