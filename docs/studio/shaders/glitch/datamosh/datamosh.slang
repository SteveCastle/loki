#version 450

// datamosh — digital compression-glitch emulation (mosh accumulator pass)
//
// Real datamosh happens when a video decoder loses an I-frame (keyframe) but
// keeps receiving P-frames: the motion vectors from the new scene get applied
// to the *stale* pixels of the old one, so the image melts and drags along
// motion, and corrupted macroblocks tear, freeze, and bleed.
//
// We have no real motion vectors here, so we synthesize them, and we get the
// "stale pixels" for free via PassFeedback0 (this pass's own previous-frame
// output — the only working history mechanism in slangfx).
//
// This pass is the heart of the effect: it persists and warps the previous
// frame, only letting a little of the current frame back in. A second
// "present" pass adds the cosmetic compression artifacts (chroma bleed, DCT
// crush) that should NOT compound inside this feedback loop.
//
// mosh_amount = 0 collapses this to a clean passthrough of the source.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float mosh_amount;
    float bleed;
    float flow_strength;
    float flow_scale;
    float block_size;
    float streak;
    float fall_speed;
    float block_glitch;
    float block_drift;
    float freeze_amount;
    float trail_decay;
    float speed;
    float seed;
} params;

#pragma parameter mosh_amount   "Datamosh: amount"            0.55 0.0  1.0  0.01
#pragma parameter bleed         "Datamosh: source bleed"      0.08 0.0  1.0  0.01
#pragma parameter flow_strength "Datamosh: smear strength"    0.80 0.0  2.0  0.01
#pragma parameter flow_scale    "Datamosh: flow detail"       3.0  0.5 12.0  0.1
#pragma parameter block_size    "Datamosh: macroblock px"     16.0 4.0 64.0  1.0
#pragma parameter streak        "Datamosh: vertical streak"   4.0  1.0 12.0  0.25
#pragma parameter fall_speed    "Datamosh: downward fall"     0.6  0.0  3.0  0.05
#pragma parameter block_glitch  "Datamosh: block tear amount" 0.25 0.0  1.0  0.01
#pragma parameter block_drift   "Datamosh: block tear dist"   0.5  0.0  2.0  0.01
#pragma parameter freeze_amount "Datamosh: block freeze"      0.18 0.0  1.0  0.01
#pragma parameter trail_decay   "Datamosh: trail persistence" 0.975 0.80 1.0 0.005
#pragma parameter speed         "Datamosh: motion speed"      1.0  0.0  4.0  0.05
#pragma parameter seed          "Datamosh: seed"              7.0  0.0 64.0  1.0

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

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}
float vnoise2(vec2 p)
{
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm2(vec2 p)
{
    float v = 0.0, a = 0.5;
    for (int k = 0; k < 4; ++k) { v += a * vnoise2(p); p *= 2.0; a *= 0.5; }
    return v;
}

void main()
{
    vec2  uv  = vTexCoord;
    float amt = params.mosh_amount;
    float t   = float(params.FrameCount) * (1.0 / 60.0) * params.speed;

    /* Macroblock grid — datamosh moves and freezes whole blocks as units.
     * Blocks are stretched vertically (`streak`) so torn/frozen regions read
     * as tall streaks down columns rather than a square mosaic. */
    vec2 res     = params.SourceSize.xy;
    vec2 bpx     = vec2(max(params.block_size, 1.0)) * vec2(1.0, params.streak);
    vec2 grid    = res / bpx;
    vec2 block   = floor(uv * grid);
    vec2 bcenter = (block + 0.5) / grid;

    /* Synthesized motion vectors: a low-frequency directional flow field,
     * sampled at block centers so the whole macroblock shares one vector.
     * Horizontal drift is kept small while the vertical component dominates
     * and carries a constant downward bias — so content streaks downward
     * (content is fetched from above: positive mv.y pulls it down). */
    float fx = fbm2(bcenter * params.flow_scale + vec2(0.0,  t * 0.30) + params.seed);
    float fy = fbm2(bcenter * params.flow_scale + vec2(5.2, -t * 0.27) + params.seed * 1.7);
    float horiz = (fx - 0.5) * 2.0 * 0.20;
    float vert  = (fy - 0.5) * 2.0 * 1.00 + params.fall_speed;
    vec2  mv = vec2(horiz, vert) * params.flow_strength * 0.03 * amt;

    /* Base smear: drag the previous frame along the motion field (the melt). */
    vec3 smear = texture(PassFeedback0, uv - mv).rgb;

    /* Per-block behaviour, re-rolled on a coarse time slot (~6 GOPs/sec). */
    float slot = floor(t * 6.0);
    float rb   = hash21(block * 0.7 + slot * 3.3 + params.seed * 2.0);

    float displace_thresh = params.block_glitch  * amt;
    float freeze_thresh   = displace_thresh + params.freeze_amount * amt;
    float do_displace = step(rb, displace_thresh);
    float do_freeze   = step(displace_thresh, rb) * step(rb, freeze_thresh);

    /* Displaced block: grab a stale region from elsewhere in the feedback
     * buffer — the classic "wrong block" tear. */
    float jx = (hash21(block + slot + params.seed) - 0.5) * 2.0 * 0.25;
    float jy = (hash21(block * 2.7 + slot + params.seed) - 0.5) * 2.0 - params.fall_speed;
    vec2  jump = vec2(jx, jy) * params.block_drift * 0.12 * amt;
    vec3 displaced = texture(PassFeedback0, uv + jump).rgb;

    /* Frozen block: hold previous content exactly (no motion, no refresh). */
    vec3 frozen = texture(PassFeedback0, uv).rgb;

    vec3 prev = smear;
    prev = mix(prev, displaced, do_displace);
    prev = mix(prev, frozen,    do_freeze);

    /* Trail persistence: stale content fades a touch each frame, giving the
     * melt a finite life and the characteristic darkening of long smears. */
    prev *= mix(1.0, params.trail_decay, amt);

    /* Source re-injection. Normal blocks let a little of the current frame
     * back in so the loop never diverges to mush; torn/frozen blocks don't.
     * Force a full refresh for the first few frames so the cold (black)
     * feedback buffer fills in immediately. */
    float stuck   = max(do_displace, do_freeze);
    float warmup  = 1.0 - step(4.0, float(params.FrameCount));
    float inject  = mix(1.0, params.bleed, amt) * (1.0 - stuck);
    inject        = max(inject, warmup);

    vec3 outc = mix(prev, texture(Source, uv).rgb, inject);
    FragColor = vec4(clamp(outc, 0.0, 1.0), 1.0);
}
