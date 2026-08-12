#version 450

// aurora-eq — the spectrum analyser, if it were weather instead of bars.
//
// The three bands are read as three points of a curve across the frame:
// bass at the left, mids in the middle, air at the right, each one a soft
// hill rather than a bar. Fractal noise breaks the curve into finer
// structure, so what you see has the shape of a spectrum — dozens of bins
// moving together — out of only three numbers. That curve is the height of
// a curtain of light that hangs over the picture, lit from within, drifting
// upward like an aurora does.
//
// Bars would need a real FFT per column. A curtain does not: an aurora has
// no bins to count, so the eye reads the shape and never audits it, and the
// three envelopes the driver already computes are enough to make the shape
// honest — the left of the frame really is the low end.
//
//   bass   raises the left of the curtain
//   mids   raises the middle
//   air    raises the right, and adds the fine shiver along the top edge
//
// Automate all three with audio drivers (∿ → Audio → band → **Level**;
// release ≈ 0.15 s keeps the curtain lively without strobing). Beat works
// too and gives three separate flames that leap on their own hits.
//
// `mirror` hangs a second curtain from the top; at 1.0 with a low height
// you get a horizon band across the middle of the frame.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float Time;
    float amount;
    float bass;
    float mids;
    float air;
    float height;
    float spread;
    float detail;
    float flow;
    float edge;
    float hue;
    float glow;
    float mirror;
} params;

#pragma parameter amount "AuroraEQ: mix"                     1.0  0.0 1.0  0.01
#pragma parameter bass   "AuroraEQ: bass level — automate"   0.0  0.0 1.0  0.01
#pragma parameter mids   "AuroraEQ: mid level — automate"    0.0  0.0 1.0  0.01
#pragma parameter air    "AuroraEQ: air level — automate"    0.0  0.0 1.0  0.01
#pragma parameter height "AuroraEQ: curtain height"          0.55 0.0 1.5  0.01
#pragma parameter spread "AuroraEQ: band spread"             1.0  0.3 2.5  0.02
#pragma parameter detail "AuroraEQ: spectrum detail"         6.0  1.0 24.0 0.5
#pragma parameter flow   "AuroraEQ: rise speed"              0.35 0.0 2.0  0.01
#pragma parameter edge   "AuroraEQ: crest brightness"        1.0  0.0 4.0  0.05
#pragma parameter hue    "AuroraEQ: palette"                 0.5  0.0 1.0  0.01
#pragma parameter glow   "AuroraEQ: brightness"              1.0  0.0 4.0  0.05
#pragma parameter mirror "AuroraEQ: hang from the top too"   0.0  0.0 1.0  0.01

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

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = p - i;
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p)
{
    return 0.55 * vnoise(p)
         + 0.30 * vnoise(p * 2.03 + 11.7)
         + 0.15 * vnoise(p * 4.11 + 31.3);
}

/* Frequency → colour, low to high. Deep magenta through green to a cold
 * white-blue: a spectrum reads left-to-right for everyone who has ever
 * looked at a meter, and keeping the same order here means the picture
 * agrees with what the ear is doing. */
vec3 palette(float x, float k)
{
    vec3 lo = mix(vec3(1.00, 0.20, 0.45), vec3(1.00, 0.45, 0.10), k);
    vec3 md = mix(vec3(0.35, 1.00, 0.55), vec3(0.95, 0.90, 0.25), k);
    vec3 hi = mix(vec3(0.45, 0.75, 1.00), vec3(0.85, 0.95, 1.00), k);
    return x < 0.5 ? mix(lo, md, smoothstep(0.0, 0.5, x))
                   : mix(md, hi, smoothstep(0.5, 1.0, x));
}

/* The spectrum curve: three hills, one per band, roughened by noise so it
 * reads as many bins rather than as three lumps. */
float curve(float x)
{
    float w = params.spread;
    float gb = exp(-pow((x - 0.12) / (0.30 * w), 2.0));
    float gm = exp(-pow((x - 0.50) / (0.34 * w), 2.0));
    float ga = exp(-pow((x - 0.88) / (0.28 * w), 2.0));

    float h = clamp(params.bass, 0.0, 1.0) * gb
            + clamp(params.mids, 0.0, 1.0) * gm
            + clamp(params.air,  0.0, 1.0) * ga;

    /* Fine structure. Two scales: a slow one that makes the curtain lean,
     * and a fast one that gives it the busy edge of a real analyser. */
    float rough = 0.72
                + 0.20 * fbm(vec2(x * params.detail, params.Time * 0.35))
                + 0.16 * fbm(vec2(x * params.detail * 3.1, params.Time * 0.9 + 17.0));

    return h * rough * params.height;
}

/* One curtain, measured from whichever edge it hangs off. `yb` is the
 * distance from that edge, 0..1. */
vec3 curtain(float x, float yb, float h, float asp)
{
    if (h <= 0.001) return vec3(0.0);

    /* Body. An aurora is a sheet seen edge-on: brightest along its lower
     * boundary and thinning upward. Hanging over a picture it also has to
     * stay a veil rather than a floor, so the foot is deliberately let go
     * translucent — a solid block at the bottom of frame is a bar chart. */
    float t    = clamp(yb / max(h, 1e-4), 0.0, 1.0);
    float body = smoothstep(h, h - 0.30 * h - 0.02, yb)
               * mix(0.35, 1.0, smoothstep(0.0, 0.85, t));

    /* Vertical streaks, scrolling upward. Aurorae are field lines: without
     * this the curtain is a gradient, and a gradient reads as fog. */
    float streak = clamp(0.30 + 1.35 * fbm(vec2(x * params.detail * 5.0,
                                               yb * 1.2 - params.Time * params.flow)), 0.0, 2.0);

    /* The crest: a bright rim exactly at the curve, which is the part that
     * actually looks like a meter. */
    float crest = exp(-pow((yb - h) / (0.020 + 0.05 * h), 2.0)) * params.edge;

    vec3 col = palette(clamp(x, 0.0, 1.0), params.hue);
    return col * (body * streak) + mix(col, vec3(1.0), 0.55) * crest;
}

void main()
{
    vec2  uv  = vTexCoord;
    float asp = params.OutputSize.x * params.OutputSize.w;

    float h = curve(uv.x);

    /* uv.y = 0 is the top of the frame, so the curtain that stands on the
     * floor is measured from 1. */
    vec3 sky = curtain(uv.x, 1.0 - uv.y, h, asp);

    /* The mirrored sheet hangs from the ceiling and is read backwards, so
     * the two curtains do not look like one image copied twice. */
    if (params.mirror > 0.0)
        sky += curtain(1.0 - uv.x, uv.y, curve(1.0 - uv.x) * params.mirror, asp) * params.mirror;

    sky *= params.glow;

    vec3 src = texture(Source, uv).rgb;
    /* Screen — the curtain is light hanging in front of the picture. */
    vec3 lit = 1.0 - (1.0 - clamp(src, 0.0, 1.0)) * (1.0 - clamp(sky, 0.0, 1.0));

    FragColor = vec4(clamp(mix(src, lit, clamp(params.amount, 0.0, 1.0)), 0.0, 1.0), 1.0);
}
