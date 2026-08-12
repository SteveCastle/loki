#version 450

// starfield — the frame becomes a window on a field of stars you are
// travelling through, and the ship rolls with the music.
//
// The stars are a lattice sampled in a space that scales with travel: as
// the scale grows, every star walks outward from the vanishing point and
// finally off the edge, which is exactly what perspective does to a point
// you fly past. Three layers run at once at different depths, so near stars
// sweep past while far ones barely move — the parallax is what sells it as
// distance rather than as a zoom.
//
// Each star is drawn between where it was a shutter ago and where it is
// now, so the faster you travel the longer it streaks. Stand still and they
// are points; kick the speed and the whole field pulls into warp lines.
//
// The three bands fly the ship:
//   bass   a surge of travel — each hit shoves you forward, and the field
//          stretches into streaks until the pulse decays
//   mids   roll: every hit twists the field about the vanishing point and
//          it unwinds as the pulse falls
//   air    the twinkle — stars scintillate on the top end, the way they
//          actually do through moving air
//
// Automate `bass` / `mids` / `air` with audio drivers (∿ → Audio → band →
// **Beat** for kicks you can count, **Level** for a field that surges with
// the track). `speed` alone, with nothing automated, is a perfectly good
// slow drift for a title card.

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
    float speed;
    float surge;
    float spin;
    float twist;
    float density;
    float size;
    float streak;
    float glow;
} params;

#pragma parameter amount  "Starfield: mix"                        1.0  0.0  1.0   0.01
#pragma parameter bass    "Starfield: bass hit — automate"        0.0  0.0  1.0   0.01
#pragma parameter mids    "Starfield: mid hit — automate"         0.0  0.0  1.0   0.01
#pragma parameter air     "Starfield: air hit — automate"         0.0  0.0  1.0   0.01
#pragma parameter speed   "Starfield: travel speed"               0.12 0.0  1.5   0.005
#pragma parameter surge   "Starfield: surge on bass"              0.35 0.0  2.0   0.01
#pragma parameter spin    "Starfield: roll (turns/min)"           1.0  -20.0 20.0 0.1
#pragma parameter twist   "Starfield: twist on mids (deg)"        12.0 -90.0 90.0 0.5
#pragma parameter density "Starfield: star density"               6.0  1.0  16.0  0.5
#pragma parameter size    "Starfield: star size (% of h)"         0.45 0.05 2.0   0.01
#pragma parameter streak  "Starfield: warp streaks"               1.0  0.0  4.0   0.05
#pragma parameter glow    "Starfield: brightness"                 1.0  0.0  4.0   0.05

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

#define LAYERS 3
#define LOG_RANGE 1.9     /* one layer covers a ~7× change of scale */
#define NEAR 0.30

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float seg_dist(vec2 p, vec2 a, vec2 b)
{
    vec2  pa = p - a;
    vec2  ba = b - a;
    float h  = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-7), 0.0, 1.0);
    return length(pa - ba * h);
}

/* One depth layer.
 *
 * `fz` is where this layer sits in its own scale cycle (0 = just born far
 * away, 1 = about to sweep past the camera). Sampling the lattice at
 * p / scale means a growing scale walks every star outward — perspective,
 * without a perspective divide. */
vec3 layer(vec2 p, float fz, float cyc, float dtz, float asp)
{
    float scale  = NEAR * exp(fz * LOG_RANGE);
    /* Where the same star sat one shutter ago. Both positions are divided
     * by the CURRENT scale, so the streak is measured in the same lattice
     * space the search runs in. */
    float ratio  = exp(-dtz * LOG_RANGE);

    /* Fade in from the far distance, and out as the star reaches the near
     * plane, so the cycle wrap is never visible. */
    float fade = smoothstep(0.06, 0.40, fz) * (1.0 - smoothstep(0.78, 1.0, fz));
    if (fade <= 0.001) return vec3(0.0);

    vec2  q    = p / scale;
    vec2  gq   = q * params.density;
    vec2  base = floor(gq);
    float rad  = (params.size * 0.01) / scale;

    vec3 acc = vec3(0.0);
    for (int j = -1; j <= 1; ++j)
    for (int i = -1; i <= 1; ++i)
    {
        vec2  cell = base + vec2(float(i), float(j));
        /* The cycle index is part of the seed: without it every lap through
         * the layer would deal the identical hand of stars. */
        vec2  seed = cell + vec2(cyc * 37.0, cyc * 91.0);
        float h1 = hash21(seed);
        float h2 = hash21(seed + 17.7);
        float h3 = hash21(seed + 41.3);
        float h4 = hash21(seed + 83.1);

        /* Not every cell holds a star — an even grid of them reads as
         * wallpaper. Density controls the lattice; this thins it out. */
        if (h1 > 0.34) continue;

        /* Star position inside its cell, kept off the edges so a 3×3 search
         * really does cover every star that can touch this pixel. */
        vec2 sp = (cell + vec2(0.2 + 0.6 * h2, 0.2 + 0.6 * h3)) / params.density;

        float d = seg_dist(q, sp * ratio, sp);

        /* Core plus a soft halo. Both are measured in lattice units, so a
         * star keeps its screen size as its layer scales. */
        float core = exp(-pow(d / (rad * 0.5), 2.0));
        float halo = exp(-pow(d / (rad * 2.6), 2.0)) * 0.30;

        /* Scintillation: stars are points, so the atmosphere makes them
         * flicker — air drives how hard. Each star gets its own phase. */
        float tw = 1.0 + params.air * 1.6 * sin(params.Time * 11.0 + h4 * 43.0);

        /* A quiet spread of colour temperature — a field of pure white dots
         * looks like dust on the lens, not like a sky. */
        vec3 tint = mix(vec3(0.75, 0.85, 1.00), vec3(1.00, 0.88, 0.72), h4);

        acc += tint * (core + halo) * mix(0.35, 1.0, h2) * max(tw, 0.0);
    }
    return acc * fade;
}

void main()
{
    vec2  uv  = vTexCoord;
    float asp = params.OutputSize.x * params.OutputSize.w;
    vec2  p   = (uv - 0.5) * vec2(asp, 1.0);

    /* Roll. The steady turn is the ship's own drift; the mid band adds a
     * twist that unwinds as the hit decays, so the field lurches on the
     * beat instead of spinning at a constant, lifeless rate. */
    float ang = radians(params.spin * 6.0) * params.Time
              + radians(params.twist) * clamp(params.mids, 0.0, 1.0);
    float ca = cos(ang), sa = sin(ang);
    p = vec2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);

    /* Travel. Bass shoves the whole field forward — the offset (not the
     * rate) is driven, so a hit is a jolt of distance covered and the field
     * settles back to its cruise as the pulse falls. */
    float z = params.Time * params.speed + params.surge * clamp(params.bass, 0.0, 1.0);

    /* Shutter, in cycle units: how much of the travel happens while the
     * frame is open. This is what turns points into warp lines. */
    float dtz = (params.speed + params.surge * clamp(params.bass, 0.0, 1.0) * 2.0)
              * params.streak * 0.03;
    dtz = clamp(dtz, 0.0, 0.35);

    vec3 stars = vec3(0.0);
    for (int k = 0; k < LAYERS; ++k)
    {
        float zk  = z + float(k) / float(LAYERS);
        float cyc = floor(zk);
        stars += layer(p, fract(zk), cyc + float(k) * 13.0, dtz, asp);
    }

    stars *= params.glow;

    vec3 src = texture(Source, uv).rgb;
    /* Screen: starlight adds to the picture, it never takes anything away. */
    vec3 lit = 1.0 - (1.0 - clamp(src, 0.0, 1.0)) * (1.0 - clamp(stars, 0.0, 1.0));

    FragColor = vec4(clamp(mix(src, lit, clamp(params.amount, 0.0, 1.0)), 0.0, 1.0), 1.0);
}
