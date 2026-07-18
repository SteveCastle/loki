#version 450

// beat-strobe — tempo-synced strobe / zoom-punch, single pass.
//
// A VJ beat hit: on every beat the frame flashes brighter, punches inward
// (quick zoom that eases back out), and the RGB channels snap apart, then
// settle before the next beat. Every Nth beat can invert as a downbeat accent.
//
// Timing comes from a synthesized beat clock: seconds = FrameCount / fps,
// beats = seconds * bpm/60 * beat_div. Set `fps` to your footage/output frame
// rate so the pulse lands on the actual beat. amount = 0 is a clean pass.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float bpm;
    float fps;
    float beat_div;
    float beat_phase;
    float strobe;
    float zoom_punch;
    float punch_decay;
    float rgb_kick;
    float invert_every;
} params;

#pragma parameter amount       "Strobe: amount"            1.0   0.0  1.0   0.01
#pragma parameter bpm          "Strobe: BPM"               120.0 40.0 220.0 0.5
#pragma parameter fps          "Strobe: footage fps"       24.0  1.0  120.0 1.0
#pragma parameter beat_div     "Strobe: beat subdivision"  1.0   0.25 8.0   0.25
#pragma parameter beat_phase   "Strobe: beat phase (beats)" 0.0 -1.0 1.0 0.01
#pragma parameter strobe       "Strobe: flash strength"    0.6   0.0  3.0   0.05
#pragma parameter zoom_punch   "Strobe: zoom punch"        0.12  0.0  0.6   0.01
#pragma parameter punch_decay  "Strobe: pulse tightness"   6.0   1.0  20.0  0.5
#pragma parameter rgb_kick     "Strobe: RGB snap"          0.4   0.0  3.0   0.05
#pragma parameter invert_every "Strobe: invert each N (0=off)" 0.0 0.0 16.0 1.0

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;

void main()
{
    vec2  uv = vTexCoord;

    /* Beat clock + sharp-attack / exp-decay envelope (1 at the beat, →0). */
    float t     = float(params.FrameCount) / max(params.fps, 1.0);
    float beats = t * params.bpm / 60.0 * params.beat_div - params.beat_phase;
    float phase = fract(beats);
    float idx   = floor(beats);
    float env   = exp(-phase * params.punch_decay);

    /* Zoom punch toward centre. */
    vec2 c = (uv - 0.5) / (1.0 + params.zoom_punch * env);
    vec2 uvz = c + 0.5;

    /* RGB snap: channels separate horizontally on the hit. */
    float ca = params.rgb_kick * params.SourceSize.z * 10.0 * env * params.amount;
    vec3 col;
    col.r = texture(Source, clamp(uvz + vec2(ca, 0.0), 0.0, 1.0)).r;
    col.g = texture(Source, clamp(uvz, 0.0, 1.0)).g;
    col.b = texture(Source, clamp(uvz - vec2(ca, 0.0), 0.0, 1.0)).b;

    /* Strobe flash. */
    col *= 1.0 + params.strobe * env * params.amount;

    /* Downbeat invert accent: brief flip at the top of every Nth beat. */
    float n = max(params.invert_every, 1.0);
    float inv = step(0.5, params.invert_every)
              * step(mod(idx, n), 0.5)
              * step(phase, 0.10) * params.amount;
    col = mix(col, vec3(1.0) - col, inv);

    /* Blend back toward the clean frame by `amount`. */
    vec3 clean = texture(Source, uv).rgb;
    col = mix(clean, col, params.amount);
    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
