#version 450

// mote-swarm — a cloud of glowing motes hangs in front of the picture,
// scatters when it is hit, and drifts back together.
//
// The swarm is a lattice of particles, each wandering on its own little
// orbit so the cloud is never still. The hit does not move the motes one by
// one — it moves the SPACE they live in: the field is blown outward from
// the centre and given a twist, then relaxes as the pulse decays. That is
// why the whole cloud recoils together like something startled, and why it
// costs the same to draw a hundred motes as three.
//
// Two depths are drawn. The far layer is smaller, dimmer and slower, and it
// is deliberately softer — a shallow lens has one plane sharp and the rest
// is bokeh, and the eye reads that as volume rather than as wallpaper.
//
// The bands do three different things to the same cloud:
//   bass   the scatter — each hit blows the swarm outward, and it falls
//          back in as the pulse decays
//   mids   the flare — motes swell and brighten on the body of the track
//   air    twinkle, and a fine jitter that makes the cloud shiver
//
// Automate `bass` / `mids` / `air` with audio drivers (∿ → Audio → band →
// **Beat** to recoil per hit, **Level** to breathe with the mix). A slow
// drift with everything else at zero is a perfectly good bed of fireflies.

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
    float density;
    float size;
    float drift;
    float scatter;
    float swirl;
    float hue;
    float glow;
} params;

#pragma parameter amount  "MoteSwarm: mix"                     1.0  0.0  1.0  0.01
#pragma parameter bass    "MoteSwarm: bass hit — automate"     0.0  0.0  1.0  0.01
#pragma parameter mids    "MoteSwarm: mid hit — automate"      0.0  0.0  1.0  0.01
#pragma parameter air     "MoteSwarm: air hit — automate"      0.0  0.0  1.0  0.01
#pragma parameter density "MoteSwarm: mote density"            6.0  2.0  18.0 0.5
#pragma parameter size    "MoteSwarm: mote size (% of h)"      0.9  0.1  4.0  0.02
#pragma parameter drift   "MoteSwarm: drift speed"             0.35 0.0  2.0  0.01
#pragma parameter scatter "MoteSwarm: scatter on bass"         0.30 0.0  1.5  0.01
#pragma parameter swirl   "MoteSwarm: swirl on scatter (deg)"  25.0 -180.0 180.0 1.0
#pragma parameter hue     "MoteSwarm: palette"                 0.55 0.0  1.0  0.01
#pragma parameter glow    "MoteSwarm: brightness"              1.0  0.0  4.0  0.05

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

/* Two cool-to-warm ramps that cross at the palette midpoint: one knob walks
 * the swarm from deep blue through magenta to gold without ever landing on
 * the muddy middle you get from a straight lerp between opposites. */
vec3 palette(float t, float k)
{
    vec3 a = mix(vec3(0.25, 0.55, 1.00), vec3(1.00, 0.35, 0.75), clamp(k * 2.0, 0.0, 1.0));
    vec3 b = mix(vec3(0.40, 1.00, 0.85), vec3(1.00, 0.80, 0.35), clamp(k * 2.0 - 1.0, 0.0, 1.0));
    return mix(a, b, t);
}

/* One depth of the cloud. `scale` sets how big its cells are on screen,
 * `soft` how far from the focal plane it sits (bigger = more bokeh). */
vec3 cloud(vec2 q, float scale, float soft, float dim, float seedOff, float flare)
{
    float dens = params.density / scale;
    vec2  gq   = q * dens;
    vec2  base = floor(gq);
    float rad  = params.size * 0.01 * scale * (1.0 + 0.6 * flare);

    vec3 acc = vec3(0.0);
    for (int j = -1; j <= 1; ++j)
    for (int i = -1; i <= 1; ++i)
    {
        vec2  cell = base + vec2(float(i), float(j));
        vec2  seed = cell + seedOff;
        float h1 = hash21(seed);
        float h2 = hash21(seed + 11.3);
        float h3 = hash21(seed + 29.7);
        float h4 = hash21(seed + 53.1);

        /* Thin the lattice out — a mote in every cell is a grid, and the
         * eye finds a grid instantly. */
        if (h1 > 0.62) continue;

        /* Each mote wanders on its own slow ellipse. The amplitude stays
         * under half a cell so a 3×3 search still finds every mote that can
         * reach this pixel. */
        float t  = params.Time * params.drift;
        vec2  wob = vec2(sin(t * (0.6 + h2) + h3 * 31.0),
                         cos(t * (0.5 + h3) + h2 * 27.0)) * 0.30;
        /* Air shivers the whole cloud at a much higher rate. */
        wob += vec2(sin(params.Time * 21.0 + h4 * 51.0),
                    cos(params.Time * 19.0 + h1 * 47.0)) * 0.10 * params.air;

        vec2  sp = (cell + vec2(0.5) + wob) / dens;
        float d  = length(q - sp);

        /* Out-of-focus motes are wider and flatter — a defocused point is a
         * disc, not a brighter dot. */
        float core = exp(-pow(d / (rad * (0.45 + soft)), 2.0));
        float halo = exp(-pow(d / (rad * (2.0 + soft * 3.0)), 2.0)) * 0.28;

        float tw = 1.0 + params.air * 1.3 * sin(params.Time * 9.0 + h4 * 61.0);

        acc += palette(h2, params.hue) * (core + halo) * mix(0.3, 1.0, h4) * max(tw, 0.0);
    }
    return acc * dim;
}

void main()
{
    vec2  uv  = vTexCoord;
    float asp = params.OutputSize.x * params.OutputSize.w;
    vec2  p   = (uv - 0.5) * vec2(asp, 1.0);

    float b = clamp(params.bass, 0.0, 1.0);
    float m = clamp(params.mids, 0.0, 1.0);

    /* The hit moves the SPACE, not the motes: push the sampling point back
     * along the blast and the whole cloud appears to have been thrown
     * outward. Inverting the map instead of integrating per-particle
     * velocities is what keeps this stateless — and it means the swarm
     * lands exactly back where it started when the pulse reaches zero. */
    float s = params.scatter * b;
    vec2  q = p / (1.0 + s);

    /* Twist while it flies apart, harder near the middle, so the recoil
     * spirals instead of merely expanding. */
    float r  = length(q);
    float aw = radians(params.swirl) * s / (1.0 + 2.2 * r);
    float ca = cos(aw), sa = sin(aw);
    q = vec2(q.x * ca + q.y * sa, -q.x * sa + q.y * ca);

    /* Near layer sharp and bright, far layer small, soft and dim. */
    vec3 motes = cloud(q, 1.00, 0.00, 1.00, 0.0,  m)
               + cloud(q, 0.55, 0.35, 0.45, 71.0, m * 0.6);

    motes *= params.glow * (1.0 + 0.9 * m);

    vec3 src = texture(Source, uv).rgb;
    vec3 lit = 1.0 - (1.0 - clamp(src, 0.0, 1.0)) * (1.0 - clamp(motes, 0.0, 1.0));

    FragColor = vec4(clamp(mix(src, lit, clamp(params.amount, 0.0, 1.0)), 0.0, 1.0), 1.0);
}
