#version 450

// datamosh-skin — skin-tone-masked digital glitch (mosh accumulator pass)
//
// A variant of datamosh that concentrates the effect on regions near human
// skin colour and leaves the rest of the frame mostly clean. Skin is detected
// in YCbCr chroma space (a Gaussian around a skin-chroma centre that holds up
// across skin tones, since luma is factored out), gated to exclude near-black
// and blown-white pixels.
//
// The skin mask scales the local mosh amount per pixel (for the smear / source
// re-injection) and per block centre (for the tear / freeze decisions, so tall
// streak columns stay coherent). Off skin, `skin_floor` sets how much effect
// remains (0 = perfectly clean elsewhere).
//
// Everything else matches datamosh.slang: feedback smear, vertical streaks,
// downward fall, block tear/freeze. mosh_amount = 0 collapses to clean source.

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
    float skin_floor;
    float skin_tolerance;
    float skin_gamma;
} params;

#pragma parameter mosh_amount    "Datamosh: amount"            0.65 0.0  1.0  0.01
#pragma parameter bleed          "Datamosh: source bleed"      0.08 0.0  1.0  0.01
#pragma parameter flow_strength  "Datamosh: smear strength"    0.80 0.0  2.0  0.01
#pragma parameter flow_scale     "Datamosh: flow detail"       3.0  0.5 12.0  0.1
#pragma parameter block_size     "Datamosh: macroblock px"     7.0  4.0 64.0  1.0
#pragma parameter streak         "Datamosh: vertical streak"   5.0  1.0 12.0  0.25
#pragma parameter fall_speed     "Datamosh: downward fall"     0.6  0.0  3.0  0.05
#pragma parameter block_glitch   "Datamosh: block tear amount" 0.40 0.0  1.0  0.01
#pragma parameter block_drift    "Datamosh: block tear dist"   0.5  0.0  2.0  0.01
#pragma parameter freeze_amount  "Datamosh: block freeze"      0.18 0.0  1.0  0.01
#pragma parameter trail_decay    "Datamosh: trail persistence" 0.975 0.80 1.0 0.005
#pragma parameter speed          "Datamosh: motion speed"      1.0  0.0  4.0  0.05
#pragma parameter seed           "Datamosh: seed"              7.0  0.0 64.0  1.0
#pragma parameter skin_floor     "Skin: off-skin effect"       0.05 0.0  1.0  0.01
#pragma parameter skin_tolerance "Skin: colour tolerance"      0.055 0.01 0.20 0.005
#pragma parameter skin_gamma     "Skin: mask sharpness"        1.6  0.5  4.0  0.05

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

// Skin likelihood in YCbCr chroma space: a Gaussian around a skin-chroma
// centre (luma removed so it spans light→dark skin), gated to drop near-black
// and blown-white pixels. Returns 0..1.
float skin_mask(vec3 c)
{
    float Y  = dot(c, vec3(0.299, 0.587, 0.114));
    float Cb = (c.b - Y) * 0.564 + 0.5;
    float Cr = (c.r - Y) * 0.713 + 0.5;
    const vec2 center = vec2(0.43, 0.59);   // (Cb, Cr) skin centre
    vec2  d = vec2(Cb, Cr) - center;
    float s = exp(-dot(d, d) / (2.0 * params.skin_tolerance * params.skin_tolerance));
    s *= smoothstep(0.08, 0.22, Y);           // not too dark
    s *= 1.0 - smoothstep(0.92, 1.0, Y);      // not blown white
    return clamp(pow(s, params.skin_gamma), 0.0, 1.0);
}

void main()
{
    vec2  uv  = vTexCoord;
    float amt = params.mosh_amount;
    float t   = float(params.FrameCount) * (1.0 / 60.0) * params.speed;

    /* Vertically-stretched macroblock grid (thin tall streak columns). */
    vec2 res     = params.SourceSize.xy;
    vec2 bpx     = vec2(max(params.block_size, 1.0)) * vec2(1.0, params.streak);
    vec2 grid    = res / bpx;
    vec2 block   = floor(uv * grid);
    vec2 bcenter = (block + 0.5) / grid;

    /* Skin masks: per-pixel (smear / inject) and per-block-centre (tear /
     * freeze, so a whole column shares the decision). Detected on the clean
     * current frame. */
    float skin_px = skin_mask(texture(Source, uv).rgb);
    float skin_bk = skin_mask(texture(Source, bcenter).rgb);
    float amt_px  = amt * mix(params.skin_floor, 1.0, skin_px);
    float amt_bk  = amt * mix(params.skin_floor, 1.0, skin_bk);

    /* Synthesized motion: small horizontal drift, dominant vertical with a
     * constant downward bias (positive mv.y fetches from above → falls down). */
    float fx = fbm2(bcenter * params.flow_scale + vec2(0.0,  t * 0.30) + params.seed);
    float fy = fbm2(bcenter * params.flow_scale + vec2(5.2, -t * 0.27) + params.seed * 1.7);
    float horiz = (fx - 0.5) * 2.0 * 0.20;
    float vert  = (fy - 0.5) * 2.0 * 1.00 + params.fall_speed;
    vec2  mv = vec2(horiz, vert) * params.flow_strength * 0.03 * amt_px;

    /* Base smear: drag the previous frame along the motion field. */
    vec3 smear = texture(PassFeedback0, uv - mv).rgb;

    /* Per-block behaviour, re-rolled on a coarse time slot, gated by skin. */
    float slot = floor(t * 6.0);
    float rb   = hash21(block * 0.7 + slot * 3.3 + params.seed * 2.0);

    float displace_thresh = params.block_glitch  * amt_bk;
    float freeze_thresh   = displace_thresh + params.freeze_amount * amt_bk;
    float do_displace = step(rb, displace_thresh);
    float do_freeze   = step(displace_thresh, rb) * step(rb, freeze_thresh);

    /* Displaced block: grab a stale region from above (downward-biased tear). */
    float jx = (hash21(block + slot + params.seed) - 0.5) * 2.0 * 0.25;
    float jy = (hash21(block * 2.7 + slot + params.seed) - 0.5) * 2.0 - params.fall_speed;
    vec2  jump = vec2(jx, jy) * params.block_drift * 0.12 * amt_bk;
    vec3 displaced = texture(PassFeedback0, uv + jump).rgb;

    /* Frozen block: hold previous content exactly. */
    vec3 frozen = texture(PassFeedback0, uv).rgb;

    vec3 prev = smear;
    prev = mix(prev, displaced, do_displace);
    prev = mix(prev, frozen,    do_freeze);

    /* Trail persistence (gated by per-pixel skin amount). */
    prev *= mix(1.0, params.trail_decay, amt_px);

    /* Source re-injection. Off-skin pixels (low amt_px) re-inject the clean
     * frame fully, so they stay clean; torn/frozen blocks hold. First few
     * frames force a full refresh to fill the cold feedback buffer. */
    float stuck   = max(do_displace, do_freeze);
    float warmup  = 1.0 - step(4.0, float(params.FrameCount));
    float inject  = mix(1.0, params.bleed, amt_px) * (1.0 - stuck);
    inject        = max(inject, warmup);

    vec3 outc = mix(prev, texture(Source, uv).rgb, inject);
    FragColor = vec4(clamp(outc, 0.0, 1.0), 1.0);
}
