#version 450

// slice-glitch — databending / signal-corruption look, single pass.
//
// A clean digital glitch (as opposed to the analog scrambled-crt): the frame
// is cut into horizontal slabs that jump sideways, colour channels separate,
// and rectangular blocks invert during corruption bursts. A time-gated burst
// envelope keeps it mostly intact with sudden hits, like a corrupted stream
// losing sync — turn `burst_freq` up for constant chaos.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float slice_amount;
    float slice_density;
    float bands;
    float rgb_split;
    float burst_freq;
    float block_invert;
    float speed;
    float seed;
} params;

#pragma parameter amount       "Slice: amount"             1.0  0.0  1.0   0.01
#pragma parameter slice_amount "Slice: max h-shift"        0.12 0.0  0.5   0.005
#pragma parameter slice_density "Slice: glitched fraction" 0.45 0.0  1.0   0.01
#pragma parameter bands        "Slice: slab count"         48.0 4.0 200.0  1.0
#pragma parameter rgb_split    "Slice: RGB separation"     1.2  0.0  6.0   0.05
#pragma parameter burst_freq   "Slice: bursts/sec"         3.5  0.0 20.0   0.1
#pragma parameter block_invert "Slice: invert blocks"      0.5  0.0  1.0   0.01
#pragma parameter speed        "Slice: speed"              1.0  0.0  4.0   0.05
#pragma parameter seed         "Slice: seed"               11.0 0.0 64.0   1.0

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
    float t   = float(params.FrameCount) * (1.0 / 60.0) * params.speed;

    /* Burst envelope: ~1/3 of time slots fire, each with its own strength. */
    float slot  = floor(t * max(params.burst_freq, 0.001));
    float fire  = step(0.66, hash1(slot * 3.1 + params.seed));
    float burst = fire * (0.4 + 0.6 * hash1(slot * 1.7 + params.seed));

    /* Horizontal slabs jump sideways; only a fraction are active. */
    float sid    = floor(uv.y * params.bands);
    float sgate  = step(1.0 - params.slice_density, hash1(sid * 1.7 + slot * 2.3 + params.seed));
    float soff   = (hash1(sid * 5.1 + slot * 0.7 + params.seed) - 0.5) * 2.0
                 * params.slice_amount * sgate * (0.25 + burst);

    /* Subtle sync-wave wobble, strongest during bursts. */
    float wave = sin(uv.y * 60.0 + t * 5.0) * 0.0025 * burst;

    float x = uv.x + (soff + wave) * amt;

    /* RGB channel separation, exploding during bursts. */
    float ca = params.rgb_split * params.SourceSize.z * (1.0 + burst * 6.0) * amt;
    vec2  b  = vec2(x, uv.y);
    vec3 col;
    col.r = texture(Source, clamp(b + vec2(ca, 0.0), 0.0, 1.0)).r;
    col.g = texture(Source, clamp(b, 0.0, 1.0)).g;
    col.b = texture(Source, clamp(b - vec2(ca, 0.0), 0.0, 1.0)).b;

    /* Rectangular block corruption: invert colour on randomly-keyed blocks
     * while a burst is active. */
    vec2 bg  = vec2(36.0, 18.0);
    vec2 blk = floor(vec2(x, uv.y) * bg);
    float ig = step(1.0 - params.block_invert * burst,
                    hash1(dot(blk, vec2(1.0, 57.0)) + slot * 0.31 + params.seed));
    col = mix(col, vec3(1.0) - col, ig);

    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
