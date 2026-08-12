#version 450

// beam-burst — the highlights in the picture throw beams of light on the
// hit, like a lens flaring at a stage lamp.
//
// Nothing is drawn on top of the frame here: the beams are made OUT of the
// frame. Every bright pixel is dragged outward along the ray from a chosen
// centre and accumulated, so a lamp, a rim light or a specular on skin
// grows spokes of its own colour — which is why the effect always looks
// like it belongs to the shot instead of like an overlay dropped on it.
//
// Along the ray the three channels are weighted differently, red reaching
// furthest and blue falling off first, so the far end of a beam disperses
// into colour the way it does through real glass.
//
//   bass   the burst: each hit throws the beams out to their full reach and
//          they retract as the pulse decays
//   mids   the spokes — how sharply the light breaks into separate blades,
//          and a kick to their rotation
//   air    a fine star filter: short bright cross-flares on the very
//          brightest points, the thing a real four-point filter does to
//          hi-hats in a music video
//
// Automate with audio drivers (∿ → Audio → band → **Beat** for a flare per
// hit, **Level** for beams that live with the track). Set the centre on the
// light in your shot, not on the middle of the frame — reach 0.5 with a
// high threshold is a subtle glint, reach 1.5 with a low one is a cathedral.

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
    float reach;
    float burst;
    float rays;
    float spin;
    float thresh;
    float disperse;
    float glow;
    float cx;
    float cy;
} params;

#pragma parameter amount   "BeamBurst: mix"                     1.0  0.0  1.0  0.01
#pragma parameter bass     "BeamBurst: bass hit — automate"     0.0  0.0  1.0  0.01
#pragma parameter mids     "BeamBurst: mid hit — automate"      0.0  0.0  1.0  0.01
#pragma parameter air      "BeamBurst: air hit — automate"      0.0  0.0  1.0  0.01
#pragma parameter reach    "BeamBurst: beam reach"              0.45 0.0  1.5  0.01
#pragma parameter burst    "BeamBurst: reach kick on bass"      0.55 0.0  1.5  0.01
#pragma parameter rays     "BeamBurst: spokes"                  0.0  0.0  24.0 1.0
#pragma parameter spin     "BeamBurst: spoke spin (deg)"        30.0 -180.0 180.0 1.0
#pragma parameter thresh   "BeamBurst: highlight threshold"     0.62 0.0  1.0  0.01
#pragma parameter disperse "BeamBurst: colour dispersion"       0.45 0.0  1.0  0.01
#pragma parameter glow     "BeamBurst: brightness"              1.0  0.0  4.0  0.05
#pragma parameter cx       "BeamBurst: centre X"                0.5  0.0  1.0  0.01
#pragma parameter cy       "BeamBurst: centre Y"                0.5  0.0  1.0  0.01

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

#define TAPS 24

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

/* Only the highlights throw beams. Everything under the threshold is the
 * subject and must stay exactly where it is — that separation is the whole
 * difference between a flare and a smear. */
vec3 bright(vec2 s)
{
    vec3  c = texture(Source, clamp(s, 0.0, 1.0)).rgb;
    float l = luma(c);
    float k = smoothstep(params.thresh, min(params.thresh + 0.25, 1.0), l);
    /* Squared: a soft threshold lets half-lit mid-tones smear too, and the
     * result reads as radial blur instead of as light leaving a source. */
    return c * k * k;
}

void main()
{
    vec2  uv  = vTexCoord;
    float asp = params.OutputSize.x * params.OutputSize.w;
    vec3  src = texture(Source, uv).rgb;

    float b = clamp(params.bass, 0.0, 1.0);
    float m = clamp(params.mids, 0.0, 1.0);
    float a = clamp(params.air,  0.0, 1.0);

    vec2  ctr = vec2(params.cx, params.cy);
    /* The burst: a hit throws the beams out to their full length and they
     * pull back in as the envelope falls. */
    float len = clamp(params.reach + params.burst * b, 0.0, 2.0);

    /* Jitter where each pixel starts walking. Twenty-four evenly spaced taps
     * over a textured area line up between neighbouring pixels and weave the
     * ghosts into a visible plaid; offsetting the whole walk per pixel turns
     * that structure into grain, which nobody sees. */
    float jit = hash21(uv * params.OutputSize.xy);

    vec3 acc = vec3(0.0);
    vec3 wsum = vec3(0.0);
    for (int i = 0; i < TAPS; ++i)
    {
        float t = (float(i) + jit) / float(TAPS);
        /* Walk from this pixel toward the centre: what lands here is what
         * lies between here and the light, dragged out along the ray. */
        vec2  s = mix(uv, ctr, t * len);

        /* Per-channel falloff — red carries furthest, blue dies first, so
         * the tip of every beam breaks into colour. One fetch does all
         * three: the dispersion lives in the weights, not in the taps. */
        float f  = 1.0 - t;
        vec3  kd = mix(vec3(1.0), vec3(1.55, 1.0, 0.62), params.disperse);
        vec3  w  = pow(vec3(f), 2.2 / kd);

        acc  += bright(s) * w;
        wsum += w;
    }
    acc /= max(wsum, vec3(1e-4));

    /* Blades. `rays` = 0 is an even wash of light; turn it up and the wash
     * breaks into spokes, sharpened and rotated by the mid band. */
    vec2  p   = (uv - ctr) * vec2(asp, 1.0);
    float ang = atan(p.y, p.x);
    if (params.rays > 0.5)
    {
        float k = 0.5 + 0.5 * cos(ang * params.rays + radians(params.spin) * (0.4 + m));
        acc *= mix(0.35, 1.0, pow(k, mix(1.5, 6.0, m)));
    }

    /* Star filter: two short hard blades that only the very brightest
     * points earn, driven by the top end. This is the glint on a cymbal. */
    if (a > 0.001)
    {
        vec3 star = vec3(0.0);
        for (int i = 1; i <= 8; ++i)
        {
            float d = float(i) / 8.0 * 0.10 * (0.5 + a);
            float wq = (1.0 - float(i) / 9.0);
            star += (bright(uv + vec2(d / asp, 0.0)) + bright(uv - vec2(d / asp, 0.0))
                   + bright(uv + vec2(0.0, d))       + bright(uv - vec2(0.0, d))) * wq;
        }
        acc += star * (0.03 * a);
    }

    /* The flare is brightest at the moment of the hit — a beam that is
     * fully out but no brighter reads as a zoom, not as a burst. */
    float flash = 1.0 + 1.2 * b + 0.5 * m;
    acc *= params.glow * flash;

    /* Screen: beams are light leaving the scene, they only ever add. */
    vec3 lit = 1.0 - (1.0 - clamp(src, 0.0, 1.0)) * (1.0 - clamp(acc, 0.0, 1.0));

    FragColor = vec4(clamp(mix(src, lit, clamp(params.amount, 0.0, 1.0)), 0.0, 1.0), 1.0);
}
