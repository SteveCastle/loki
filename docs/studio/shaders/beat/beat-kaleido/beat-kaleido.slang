#version 450

// beat-kaleido — tempo-synced kaleidoscope, single pass.
//
// Mirrors the frame into `segments` wedges around the centre. The rotation
// drifts continuously and then snaps forward by `beat_kick` on each beat
// (eased over the start of the beat), while the radius pulses inward on the
// hit — a kaleidoscope that dances to the tempo.
//
// Timing: seconds = FrameCount / fps, beats = seconds * bpm/60 * beat_div.
// Set `fps` to your real frame rate. amount = 0 is a clean pass.

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
    float segments;
    float spin;
    float beat_kick;
    float zoom_pulse;
    float pulse_decay;
} params;

#pragma parameter amount      "Kaleido: amount"           1.0   0.0  1.0   0.01
#pragma parameter bpm         "Kaleido: BPM"              120.0 40.0 220.0 0.5
#pragma parameter fps         "Kaleido: footage fps"      24.0  1.0  120.0 1.0
#pragma parameter beat_div    "Kaleido: beat subdivision" 1.0   0.25 8.0   0.25
#pragma parameter beat_phase   "BeatKaleido: beat phase (beats)" 0.0 -1.0 1.0 0.01
#pragma parameter segments    "Kaleido: mirror segments"  6.0   2.0  24.0  1.0
#pragma parameter spin        "Kaleido: drift spin"       0.2   -2.0 2.0   0.05
#pragma parameter beat_kick   "Kaleido: rotation snap"    0.5   0.0  3.14  0.05
#pragma parameter zoom_pulse  "Kaleido: radius pulse"     0.18  0.0  0.8   0.01
#pragma parameter pulse_decay "Kaleido: pulse tightness"  5.0   1.0  20.0  0.5

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
    vec2 uv = vTexCoord;

    /* Beat clock + envelope. */
    float t     = float(params.FrameCount) / max(params.fps, 1.0);
    float beats = t * params.bpm / 60.0 * params.beat_div - params.beat_phase;
    float phase = fract(beats);
    float idx   = floor(beats);
    float env   = exp(-phase * params.pulse_decay);

    /* Polar coords about centre (correct for aspect so wedges stay radial). */
    float aspect = params.SourceSize.x * params.SourceSize.w;
    vec2  p = (uv - 0.5) * vec2(aspect, 1.0);
    float r = length(p);
    float a = atan(p.y, p.x);

    /* Rotation: continuous drift + a forward snap each beat, eased over the
     * first quarter of the beat. */
    float rot = t * params.spin
              + params.beat_kick * (idx + smoothstep(0.0, 0.25, phase));
    a += rot;

    /* Fold into one mirrored wedge. */
    float seg = 6.28318530718 / max(params.segments, 1.0);
    a = mod(a, seg);
    a = abs(a - seg * 0.5);

    /* Radius pulses inward on the beat. */
    r *= 1.0 - params.zoom_pulse * env;

    vec2 kp = vec2(cos(a), sin(a)) * r;
    kp.x /= max(aspect, 0.0001);
    vec2 kuv = clamp(kp + 0.5, 0.0, 1.0);

    vec3 col = texture(Source, kuv).rgb;
    col *= 1.0 + 0.3 * env;                 /* slight flash on the hit */

    vec3 clean = texture(Source, uv).rgb;
    col = mix(clean, col, params.amount);
    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
