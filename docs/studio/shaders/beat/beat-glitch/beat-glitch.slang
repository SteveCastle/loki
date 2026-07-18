#version 450

// beat-glitch — tempo-gated databend burst, single pass.
//
// Clean between beats; on every beat (or subdivision) a glitch burst fires
// with a sharp attack and exponential decay: horizontal slices jump, RGB
// channels split, and blocks scramble/displace. Every `accent_every` beats the
// hit is stronger (downbeat accent). The whole thing is locked to a beat clock
// so it cuts to the music.
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
    float attack_decay;
    float accent_every;
    float slice_amount;
    float rgb_split;
    float block_scramble;
    float bands;
    float seed;
} params;

#pragma parameter amount         "BeatGlitch: amount"          1.0   0.0  1.0   0.01
#pragma parameter bpm            "BeatGlitch: BPM"             120.0 40.0 220.0 0.5
#pragma parameter fps            "BeatGlitch: footage fps"     24.0  1.0  120.0 1.0
#pragma parameter beat_div       "BeatGlitch: beat subdivision" 1.0  0.25 8.0   0.25
#pragma parameter beat_phase   "BeatGlitch: beat phase (beats)" 0.0 -1.0 1.0 0.01
#pragma parameter attack_decay   "BeatGlitch: hit tightness"   7.0   1.0  24.0  0.5
#pragma parameter accent_every   "BeatGlitch: accent each N"   4.0   1.0  16.0  1.0
#pragma parameter slice_amount   "BeatGlitch: slice shift"     0.15  0.0  0.5   0.005
#pragma parameter rgb_split      "BeatGlitch: RGB split"       2.0   0.0  8.0   0.1
#pragma parameter block_scramble "BeatGlitch: block scramble"  0.5   0.0  1.0   0.01
#pragma parameter bands          "BeatGlitch: slab count"      40.0  4.0 200.0  1.0
#pragma parameter seed           "BeatGlitch: seed"            13.0  0.0 64.0   1.0

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

float hash1(float n) { return fract(sin(n * 17.3217) * 43758.5453); }

void main()
{
    vec2  uv  = vTexCoord;
    float amt = params.amount;

    /* Beat clock + hit envelope, accented on the downbeat. */
    float t      = float(params.FrameCount) / max(params.fps, 1.0);
    float beats  = t * params.bpm / 60.0 * params.beat_div - params.beat_phase;
    float phase  = fract(beats);
    float idx    = floor(beats);
    float accent = mix(1.0, 1.7, step(mod(idx, max(params.accent_every, 1.0)), 0.5));
    float hit    = exp(-phase * params.attack_decay) * accent * amt;

    /* Horizontal slabs jump sideways on the hit. */
    float sid  = floor(uv.y * params.bands);
    float soff = (hash1(sid * 5.1 + idx * 0.7 + params.seed) - 0.5) * 2.0
               * params.slice_amount * hit;
    float x = uv.x + soff;

    /* Block scramble: on strong hits, whole blocks grab from a random offset. */
    vec2  bg   = vec2(28.0, 16.0);
    vec2  blk  = floor(vec2(x, uv.y) * bg);
    float bkey = hash1(dot(blk, vec2(1.0, 57.0)) + idx * 0.31 + params.seed);
    float bgate = step(1.0 - params.block_scramble * clamp(hit, 0.0, 1.0), bkey);
    vec2  bjump = (vec2(hash1(bkey * 7.1), hash1(bkey * 3.7)) - 0.5) * 0.25 * bgate * hit;
    vec2  suv = vec2(x, uv.y) + bjump;

    /* RGB split, exploding on the hit. */
    float ca = params.rgb_split * params.SourceSize.z * (1.0 + hit * 6.0) * amt;
    vec3 col;
    col.r = texture(Source, clamp(suv + vec2(ca, 0.0), 0.0, 1.0)).r;
    col.g = texture(Source, clamp(suv, 0.0, 1.0)).g;
    col.b = texture(Source, clamp(suv - vec2(ca, 0.0), 0.0, 1.0)).b;

    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
