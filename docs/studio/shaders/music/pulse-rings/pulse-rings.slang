#version 450

// pulse-rings — every hit throws a ring of light out through the picture.
//
// The ring is an EVENT, not a level: it is born at the origin the instant a
// note lands and travels outward as the pulse behind it decays, so what you
// automate is when a band is struck, not where the ring is. Radius comes
// from the falling edge of the envelope (1 → 0 becomes centre → rim) and
// brightness from the envelope itself, which is why a beat driver's decay
// time reads directly as how far the ring gets before it dies.
//
// Three rings run at once because three bands do different things to a
// room: `bass` is a slow wide swell that bends the picture as it passes,
// `mids` is the body of the hit, and `air` is a thin fast rim that outruns
// both and glitters. The geometry differences are baked in (widths and
// speeds are fixed multiples) so the three stay recognisable however hard
// you drive them.
//
// Where a ring passes, the frame is displaced outward like a wave crossing
// water, the crest picks up a tint, and R/B are sampled a hair apart so the
// edge splits into colour — the same dispersion a real lens shows on a
// bright moving edge.
//
// Automate `bass` / `mids` / `air` with audio drivers: ∿ on the row → Audio
// → pick the band → follow **Beat** (impulses; decay ≈ 0.25–0.5 s sets the
// ring's reach) or **Level** for a continuous swell that never quite dies.
//
// Subtle: warp 2, glow 0.3, chroma 0.2. Music video: warp 8, glow 1.2,
// chroma 0.6, spread 1.4.

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
    float spread;
    float width;
    float warp;
    float glow;
    float chroma;
    float cx;
    float cy;
    float fade;
} params;

#pragma parameter amount "PulseRings: mix"                        1.0  0.0 1.0  0.01
#pragma parameter bass   "PulseRings: bass hit — automate"        0.0  0.0 1.0  0.01
#pragma parameter mids   "PulseRings: mid hit — automate"         0.0  0.0 1.0  0.01
#pragma parameter air    "PulseRings: air hit — automate"         0.0  0.0 1.0  0.01
#pragma parameter spread "PulseRings: reach (frame heights)"      1.0  0.2 2.5  0.05
#pragma parameter width  "PulseRings: ring thickness"             0.12 0.01 0.5 0.005
#pragma parameter warp   "PulseRings: displacement (% of h)"      4.0  0.0 20.0 0.1
#pragma parameter glow   "PulseRings: crest glow"                 1.0  0.0 3.0  0.05
#pragma parameter chroma "PulseRings: colour split"               0.35 0.0 1.0  0.01
#pragma parameter cx     "PulseRings: origin X"                   0.5  0.0 1.0  0.01
#pragma parameter cy     "PulseRings: origin Y"                   0.5  0.0 1.0  0.01
#pragma parameter fade   "PulseRings: fade with distance"         0.6  0.0 1.0  0.01

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

float hash11(float p)
{
    float q = fract(p * 0.1031);
    q *= q + 33.33;
    q *= q + q;
    return fract(q);
}

/* Smooth 1D value noise in [0,1]. The air ring glitters with this rather
 * than with a sine: an even string of beads reads as a dotted line, and
 * light catching on something never lands evenly. */
float vnoise1(float x)
{
    float i = floor(x);
    float f = x - i;
    f = f * f * (3.0 - 2.0 * f);
    return mix(hash11(i), hash11(i + 1.0), f);
}

/* One ring's contribution at distance d (height units).
 *
 *   env    the band's envelope, 1 at the strike and falling
 *   reach  how far this ring travels at full decay
 *   thick  crest width
 *
 * Returns the crest profile scaled by how much life the pulse has left;
 * `dist` reports the ring's current radius so the caller can fade the far
 * end of the travel out. */
float ring(float env, float d, float reach, float thick, out float radius)
{
    float e = clamp(env, 0.0, 1.0);
    /* Radius rides the FALLING edge: full envelope = at the origin, spent
     * envelope = at the rim. A ring is therefore exactly as long as the
     * driver's decay, in seconds, with no rate of its own to keep in sync. */
    radius = reach * (1.0 - e);
    float x = (d - radius) / max(thick, 1e-4);
    /* Gaussian crest — no hard shoulder, so the wave has no visible width
     * at low amplitudes, it just gets fainter. */
    float crest = exp(-x * x);
    /* Amplitude decays a little more slowly than the envelope so the ring
     * is still visible when it reaches the corners. */
    return crest * pow(e, 0.75);
}

void main()
{
    vec2  uv  = vTexCoord;
    float asp = params.OutputSize.x * params.OutputSize.w;

    vec2  ctr = vec2(params.cx, params.cy);
    vec2  p   = (uv - ctr) * vec2(asp, 1.0);
    float d   = length(p);
    vec2  dir = d > 1e-5 ? p / d : vec2(0.0, 0.0);

    /* Bass swells wide and slow, air is a thin rim that outruns everything.
     * Fixed ratios: the three rings have to stay telling apart at any
     * setting, which they don't if each gets its own thickness slider. */
    float rb, rm, ra;
    float wb = ring(params.bass, d, params.spread * 1.00, params.width * 2.0, rb);
    float wm = ring(params.mids, d, params.spread * 0.85, params.width * 1.0, rm);
    float wa = ring(params.air,  d, params.spread * 1.35, params.width * 0.4, ra);

    /* A wave loses height as it spreads; `fade` is how much of that loss to
     * model (0 = a ring that stays as strong at the rim as at the origin). */
    float fb = mix(1.0, 1.0 / (1.0 + 1.6 * rb), params.fade);
    float fm = mix(1.0, 1.0 / (1.0 + 1.6 * rm), params.fade);
    float fa = mix(1.0, 1.0 / (1.0 + 1.6 * ra), params.fade);
    wb *= fb; wm *= fm; wa *= fa;

    /* Air shimmers along its own rim rather than glowing evenly, so the
     * thin ring reads as sparkle and not as a wire. Two noise rates beat
     * against each other — one alone still finds a rhythm the eye can
     * follow. */
    float ang = atan(p.y, p.x);
    /* Varying with angle ALONE gives every glint the full thickness of the
     * ring and the rim turns into a sunburst of spikes; mixing in the
     * distance breaks those into separate points of light. */
    float tw  = vnoise1(ang * 11.0 + params.Time * 6.0)
              * vnoise1(ang * 27.0 + d * 40.0 - params.Time * 9.0 + 31.7);
    wa *= 0.55 + 1.5 * tw;

    /* Push the picture out along the crest: bass carries most of the water,
     * air barely bends anything. */
    float push = (wb * 1.0 + wm * 0.55 + wa * 0.15) * params.warp * 0.01;
    vec2  off  = dir * push / vec2(asp, 1.0);

    /* Sample R and B either side of the crest so the moving edge splits into
     * colour the way a bright edge does through glass. */
    float split = (wb + wm + wa) * params.chroma * 0.01 * max(params.warp, 1.0) * 0.35;
    vec2  soff  = dir * split / vec2(asp, 1.0);

    vec3 col;
    col.r = texture(Source, clamp(uv + off + soff, 0.0, 1.0)).r;
    col.g = texture(Source, clamp(uv + off,        0.0, 1.0)).g;
    col.b = texture(Source, clamp(uv + off - soff, 0.0, 1.0)).b;

    /* Crest light. The tints run low-warm → mid → high-cool, the way a
     * spectrum reads left to right on any meter anyone has ever looked at. */
    vec3 lit = clamp((vec3(1.00, 0.42, 0.22) * wb * 1.0
                    + vec3(0.45, 0.95, 0.60) * wm * 0.8
                    + vec3(0.70, 0.90, 1.00) * wa * 1.2) * params.glow, 0.0, 6.0);

    /* Screen rather than add: adding drives all three channels to 1 the
     * moment a crest is bright, and a white ring is a ring with no band
     * left in it. Screening keeps the tint until the very last stop, and
     * only what is left over past white is allowed to bloom out. */
    col = 1.0 - (1.0 - clamp(col, 0.0, 1.0)) * (1.0 - clamp(lit, 0.0, 1.0));
    col += clamp(lit - 1.0, 0.0, 5.0) * 0.12;

    vec3 src = texture(Source, uv).rgb;
    FragColor = vec4(clamp(mix(src, col, clamp(params.amount, 0.0, 1.0)), 0.0, 1.0), 1.0);
}
