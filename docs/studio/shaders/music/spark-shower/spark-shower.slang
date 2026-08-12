#version 450

// spark-shower — a hit throws a fistful of embers into the air.
//
// A real spark is a particle, so this draws particles: each hit launches a
// burst of them out of one point, every one with its own direction, its own
// speed and its own lifetime, all of them falling under the same gravity
// and cooling from white through yellow and orange to a dying red. They
// streak, because a spark moving that fast across a shutter draws a line
// rather than a dot.
//
// The burst is an EVENT driven by the envelope, not a level: the strike is
// envelope = 1 and the shower ages as it decays, so a beat driver's decay
// time is literally how long the sparks hang in the air before they burn
// out. Three showers run at once, one per band, and because their envelopes
// are independent they overlap the way a drum kit does:
//
//   bass   heavy slow embers, thrown low and wide, falling hard — the ones
//          that arc out and die on the floor
//   mids   the body of the shower, mid weight and mid speed
//   air    a fine fast crackle, barely any weight to it, gone almost at
//          once — cymbals and hats
//
// Automate `bass` / `mids` / `air` with audio drivers: ∿ on the row → Audio
// → band → **Beat** (decay 0.4–0.8 s reads as a real shower; 0.15 s is a
// spit of sparks). Level works too, and turns the shower into a fountain
// that never stops.
//
// Put the emitter at the bottom edge (origin Y = 1) for a firework or a
// welding torch; put it in the middle of a face for something less sane.

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
    float count;
    float speed;
    float gravity;
    float cone;
    float size;
    float trail;
    float glow;
    float cx;
    float cy;
} params;

#pragma parameter amount  "SparkShower: mix"                     1.0  0.0  1.0  0.01
#pragma parameter bass    "SparkShower: bass hit — automate"     0.0  0.0  1.0  0.01
#pragma parameter mids    "SparkShower: mid hit — automate"      0.0  0.0  1.0  0.01
#pragma parameter air     "SparkShower: air hit — automate"      0.0  0.0  1.0  0.01
#pragma parameter count   "SparkShower: sparks per hit"          20.0 1.0  32.0 1.0
#pragma parameter speed   "SparkShower: launch speed"            0.9  0.05 2.5  0.01
#pragma parameter gravity "SparkShower: gravity"                 0.6  -2.0 4.0  0.05
#pragma parameter cone    "SparkShower: spread (0 up, 1 all)"    0.55 0.0  1.0  0.01
#pragma parameter size    "SparkShower: spark size (% of h)"     1.1  0.05 4.0  0.01
#pragma parameter trail   "SparkShower: streak length"           1.0  0.0  2.0  0.02
#pragma parameter glow    "SparkShower: brightness"              1.0  0.0  4.0  0.05
#pragma parameter cx      "SparkShower: origin X"                0.5  0.0  1.0  0.01
#pragma parameter cy      "SparkShower: origin Y"                0.9  0.0  1.0  0.01

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

#define MAX_SPARKS 32
#define PI 3.14159265

float hash11(float p)
{
    float q = fract(p * 0.1031);
    q *= q + 33.33;
    q *= q + q;
    return fract(q);
}

/* Distance from p to the segment a→b. The streak is drawn as a segment
 * rather than a smeared dot because that is what a fast particle leaves on
 * a real shutter: a line with the head at one end. */
float seg_dist(vec2 p, vec2 a, vec2 b)
{
    vec2  pa = p - a;
    vec2  ba = b - a;
    float h  = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-7), 0.0, 1.0);
    return length(pa - ba * h);
}

/* An ember cools as it flies: white-hot at the strike, yellow, orange, then
 * the deep red just before it goes out. `tint` biases the whole ramp so the
 * three bands stay distinguishable in the same shower. */
vec3 ember(float age, vec3 tint)
{
    vec3 c = mix(vec3(1.0, 0.96, 0.88), vec3(1.0, 0.55, 0.14), smoothstep(0.0, 0.60, age));
    c = mix(c, vec3(0.90, 0.14, 0.03), smoothstep(0.62, 1.0, age));
    return c * tint;
}

/* One burst.
 *
 *   env     the band's envelope — 1 at the strike, falling to 0
 *   seed    keeps the three bursts from firing identical sparks
 *   pw      particle weight: scales gravity and drag (bass embers are
 *           heavy, air sparks are not)
 *   vs      launch-speed multiplier
 *   sz      size multiplier
 *   n       how many sparks this burst gets
 *   tint    colour bias
 */
vec3 burst(vec2 p, vec2 emit, float env, float seed, float pw, float vs,
           float sz, float n, vec3 tint, float asp)
{
    float e = clamp(env, 0.0, 1.0);
    if (e <= 0.001) return vec3(0.0);

    /* Age of the whole burst rides the falling edge of the envelope: full
     * envelope = just struck, spent envelope = burnt out. */
    float age = 1.0 - e;
    float dt  = params.trail * 0.09 + 0.004;   // shutter, in units of age

    vec3  acc = vec3(0.0);
    float rad = params.size * 0.01 * sz;

    for (int i = 0; i < MAX_SPARKS; ++i)
    {
        if (float(i) >= n) break;

        float fi = float(i) + seed;
        float h1 = hash11(fi * 7.13);          // direction
        float h2 = hash11(fi * 3.71 + 19.3);   // speed
        float h3 = hash11(fi * 11.7 + 47.1);   // brightness / flicker
        float h4 = hash11(fi * 5.29 + 71.9);   // lifetime spread

        /* Sparks do not all die together — the short-lived ones are what
         * make a shower look like a shower and not like a firework prop. */
        float life = mix(0.55, 1.0, h4);
        float t    = age / life;
        if (t >= 1.0) continue;

        /* Direction: a cone about straight up. cone = 0 is a jet, 1 is a
         * full sphere of sparks. */
        float ang = (h1 * 2.0 - 1.0) * PI * params.cone;
        vec2  dir = vec2(sin(ang), -cos(ang));
        float sp  = params.speed * vs * mix(0.45, 1.15, h2);

        /* Ballistics with a little drag, in height units. Drag matters:
         * without it every spark travels in a clean parabola and the
         * shower reads as geometry rather than as burning metal. */
        float td = t * (1.0 - 0.28 * t * pw);
        vec2  pos = emit + dir * sp * td + vec2(0.0, 0.5 * params.gravity * pw * t * t);
        float tb  = max(t - dt, 0.0);
        float tdb = tb * (1.0 - 0.28 * tb * pw);
        vec2  prev = emit + dir * sp * tdb + vec2(0.0, 0.5 * params.gravity * pw * tb * tb);

        /* Fade: bright at the strike, then the long tail of a cooling
         * ember, with a flicker that is per-spark so they do not blink
         * together. */
        float fade = pow(1.0 - t, 1.2) * mix(0.6, 1.0, h3);
        fade *= 0.75 + 0.25 * sin(t * 34.0 + h3 * 31.4);

        float d    = seg_dist(p, prev, pos);
        /* A spark is a hard little point wrapped in a soft glow. Widen the
         * core and it stops being burning metal and starts being a firefly. */
        float core = exp(-pow(d / (rad * 0.30), 2.0)) * 2.4;
        float halo = exp(-pow(d / (rad * 3.0), 2.0)) * 0.30;

        acc += ember(t, tint) * (core + halo) * fade * 1.4;
    }
    return acc;
}

void main()
{
    vec2  uv  = vTexCoord;
    float asp = params.OutputSize.x * params.OutputSize.w;

    /* Height-normalised space, so a spark's arc is the same shape and the
     * same size on any aspect ratio. */
    vec2 p    = (uv - 0.5) * vec2(asp, 1.0);
    vec2 emit = (vec2(params.cx, params.cy) - 0.5) * vec2(asp, 1.0);

    float n = max(params.count, 1.0);

    vec3 sparks = vec3(0.0);
    /* Heavy → light. The weights and speeds are fixed multiples so the
     * three showers stay tellable apart however hard you drive them. */
    sparks += burst(p, emit, params.bass, 0.0,  1.35, 0.75, 1.45, n * 0.75,
                    vec3(1.00, 0.72, 0.45), asp);
    sparks += burst(p, emit, params.mids, 13.7, 1.00, 1.00, 1.00, n,
                    vec3(1.00, 0.88, 0.70), asp);
    sparks += burst(p, emit, params.air,  57.3, 0.45, 1.55, 0.55, n * 1.25,
                    vec3(0.85, 0.95, 1.00), asp);

    sparks *= params.glow;

    /* The emitter itself flares on every hit — a spark shower always has a
     * hot point where the metal is actually being cut. */
    float flare = clamp(params.bass + params.mids * 0.7 + params.air * 0.4, 0.0, 2.0);
    float de    = length(p - emit);
    sparks += vec3(1.0, 0.85, 0.6) * exp(-pow(de / (params.size * 0.01 * 1.2 + 1e-4), 2.0)) * flare * params.glow * 0.8;

    vec3 src = texture(Source, uv).rgb;

    /* Sparks are light: they add, they never darken what they cross. */
    vec3 lit = 1.0 - (1.0 - clamp(src, 0.0, 1.0)) * (1.0 - clamp(sparks, 0.0, 1.0));
    lit += clamp(sparks - 1.0, 0.0, 4.0) * 0.25;   // let the hottest cores bloom past white

    FragColor = vec4(clamp(mix(src, lit, clamp(params.amount, 0.0, 1.0)), 0.0, 1.0), 1.0);
}
