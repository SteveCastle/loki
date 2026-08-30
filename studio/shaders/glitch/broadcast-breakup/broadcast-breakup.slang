#version 450

// broadcast-breakup — digital TV (DVB/MPEG) transport-error emulation.
//
// A weak digital broadcast doesn't degrade gracefully like analog: the
// decoder drops whole transport packets, so rows of macroblocks freeze on
// stale content and drift out of position, mis-decoded blocks fill with the
// wrong chroma (the classic green/magenta garbage) and leftover DCT basis
// patterns, quantization collapses to chunky flat tiles, and in a deep fade
// the entire frame just sticks. Errors persist until a clean refresh
// arrives — which is exactly what PassFeedback0 (this pass's previous
// output) gives us: stale slices re-read their own corrupted history, so
// damage compounds over the length of a dropout.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float error_rate;
    float dropout;
    float hold;
    float garbage;
    float block_px;
    float chroma_error;
    float slice_drift;
    float crush;
    float speed;
    float seed;
} params;

#pragma parameter amount       "Breakup: amount"            1.0  0.0  1.0  0.01
#pragma parameter error_rate   "Breakup: error events/sec"  4.0  0.5 20.0  0.5
#pragma parameter dropout      "Breakup: dropout chance"    0.55 0.0  1.0  0.01
#pragma parameter hold         "Breakup: full-freeze chance" 0.15 0.0 1.0  0.01
#pragma parameter garbage      "Breakup: garbage blocks"    0.35 0.0  1.0  0.01
#pragma parameter block_px     "Breakup: macroblock px"     16.0 4.0 64.0  1.0
#pragma parameter chroma_error "Breakup: chroma corruption" 0.75 0.0  1.0  0.01
#pragma parameter slice_drift  "Breakup: slice drift"       1.5  0.0  8.0  0.1
#pragma parameter crush        "Breakup: block crush"       0.6  0.0  1.0  0.01
#pragma parameter speed        "Breakup: speed"             1.0  0.0  4.0  0.05
#pragma parameter seed         "Breakup: seed"              23.0 0.0 64.0  1.0

/* Colour-only: chroma corruption would dim the white coverage matte. */
#pragma matte_ignore chroma_error

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
layout(set = 0, binding = 3) uniform sampler2D PassFeedback0;

float hash1(float n) { return fract(sin(n * 17.3217) * 43758.5453); }
float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

/* BT.601-ish luma/chroma split — garbage blocks corrupt Cb/Cr, not RGB,
 * because that's what a real decoder loses (chroma planes go missing and
 * default to constants → saturated green/magenta tiles). */
vec3 rgb2ycc(vec3 c)
{
    float y = dot(c, vec3(0.299, 0.587, 0.114));
    return vec3(y, (c.b - y) * 0.565 + 0.5, (c.r - y) * 0.713 + 0.5);
}
vec3 ycc2rgb(vec3 c)
{
    float cb = c.y - 0.5, cr = c.z - 0.5;
    return vec3(c.x + 1.403 * cr, c.x - 0.344 * cb - 0.714 * cr, c.x + 1.773 * cb);
}

void main()
{
    vec2  uv  = vTexCoord;
    float amt = params.amount;
    float t   = float(params.FrameCount) * (1.0 / 60.0) * params.speed;

    /* Error events land in coarse time slots; each slot rolls signal
     * quality. sev = 0 means this slot decoded fine. */
    float slot = floor(t * max(params.error_rate, 0.001));
    float sev  = step(hash1(slot * 2.7 + params.seed), params.dropout)
               * (0.35 + 0.65 * hash1(slot * 4.3 + params.seed));

    /* A really bad fade sticks the entire frame. */
    float freezeAll = step(0.001, sev)
                    * step(hash1(slot * 6.1 + params.seed * 1.3), params.hold);

    /* Macroblock grid; a slice = one row of macroblocks (how MPEG packs
     * them into packets, and the unit real errors land on). */
    vec2  res   = params.SourceSize.xy;
    float bpx   = max(params.block_px, 4.0);
    vec2  grid  = res / bpx;
    vec2  blk   = floor(uv * grid);
    float slice = blk.y;

    /* Stale slices: frozen on history, nudged sideways out of position. */
    float rs    = hash21(vec2(slice * 3.7, slot * 11.0) + params.seed);
    float stale = max(step(rs, sev), freezeAll);
    float drift = (hash21(vec2(slice * 7.3, slot) + params.seed) - 0.5) * 2.0
                * params.slice_drift * bpx * params.SourceSize.z * stale;
    vec3 cur  = texture(Source, uv).rgb;
    vec3 prev = texture(PassFeedback0, clamp(uv + vec2(drift, 0.0), 0.0, 1.0)).rgb;

    /* Garbage macroblocks: content pulled from the wrong place, pixelated
     * to half-block resolution, chroma replaced by a saturated constant,
     * plus a leftover DCT basis-function checkerboard. */
    float rb   = hash21(blk + slot * 17.0 + params.seed * 2.0);
    float garb = step(rb, params.garbage * sev);

    vec2 wrong = fract(uv + vec2(hash21(blk + 3.1 + slot), hash21(blk + 9.7 + slot)) - 0.5);
    vec2 puv   = (floor(wrong * grid * 2.0) + 0.5) / (grid * 2.0);
    vec3 ycc   = rgb2ycc(texture(PassFeedback0, puv).rgb);

    float corner = step(0.5, hash21(blk * 1.3 + slot));       /* green or magenta */
    float cvar   = hash21(blk * 2.9 + slot) * 0.25 - 0.125;
    ycc.y = mix(ycc.y, corner + cvar, params.chroma_error);
    ycc.z = mix(ycc.z, corner - cvar, params.chroma_error);

    vec2  inb = fract(uv * grid);
    float bu  = floor(hash21(blk + slot * 3.3) * 4.0) + 1.0;
    float bv  = floor(hash21(blk + slot * 5.9) * 4.0) + 1.0;
    ycc.x = clamp(ycc.x + cos(inb.x * 3.14159 * bu) * cos(inb.y * 3.14159 * bv) * 0.22, 0.0, 1.0);
    vec3 gcol = clamp(ycc2rgb(ycc), 0.0, 1.0);

    /* Compose the error content, then crush it: hard posterize sells the
     * "quantizer ran out of bits" look. */
    vec3  errc = mix(prev, gcol, garb);
    float lev  = mix(64.0, 5.0, params.crush);
    errc = floor(errc * lev + 0.5) / lev;

    float region = max(stale, garb) * amt;
    vec3  outc   = mix(cur, errc, region);

    /* Fill the cold feedback buffer with clean frames on startup. */
    float warm = 1.0 - step(4.0, float(params.FrameCount));
    outc = mix(outc, cur, warm);

    FragColor = vec4(clamp(outc, 0.0, 1.0), 1.0);
}
