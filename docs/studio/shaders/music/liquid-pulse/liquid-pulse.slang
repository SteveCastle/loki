#version 450

// liquid-pulse — the picture floats on a pool, and the music is what is
// hitting the water.
//
// The frame is read through a fractal flow field: where the field points,
// the image is dragged. On its own that is a slow, glassy drift. What the
// bands do is change the STATE of the liquid rather than paint anything on
// it, which is why this never reads as an overlay:
//
//   bass   the swell — a hit deepens the whole field, so the picture heaves
//          and settles as the pulse decays
//   mids   the vortex — each hit twists the pool about its centre, hardest
//          in the middle, unwinding as it falls
//   air    capillary ripples: the fine, fast chop that runs across the
//          surface of real water when something touches it
//
// The surface is lit as well as displaced: the gradient of the flow field
// is used as a normal, so the crests catch a specular sheen and the picture
// looks like it is under something rather than merely wobbling. `chroma`
// splits the channels across the steepest slopes, which is what thick
// liquid does to what you see through it.
//
// Automate `bass` / `mids` / `air` with audio drivers (∿ → Audio → band →
// **Level** for a pool that swells with the track, **Beat** for a stone
// dropped on every kick). Keep warp under about 3 % for something that
// still reads as a face; past 8 % it is abstract, which is also fine.

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
    float scale;
    float speed;
    float warp;
    float swirl;
    float ripple;
    float sheen;
    float hue;
    float chroma;
} params;

#pragma parameter amount "LiquidPulse: mix"                    1.0  0.0  1.0  0.01
#pragma parameter bass   "LiquidPulse: bass hit — automate"    0.0  0.0  1.0  0.01
#pragma parameter mids   "LiquidPulse: mid hit — automate"     0.0  0.0  1.0  0.01
#pragma parameter air    "LiquidPulse: air hit — automate"     0.0  0.0  1.0  0.01
#pragma parameter scale  "LiquidPulse: swell size"             2.4  0.3  12.0 0.1
#pragma parameter speed  "LiquidPulse: flow speed"             0.30 0.0  2.0  0.01
#pragma parameter warp   "LiquidPulse: displacement (% of h)"  2.5  0.0  15.0 0.05
#pragma parameter swirl  "LiquidPulse: vortex on mids (deg)"   35.0 -180.0 180.0 1.0
#pragma parameter ripple "LiquidPulse: chop on air"            1.0  0.0  4.0  0.02
#pragma parameter sheen  "LiquidPulse: surface sheen"          0.7  0.0  3.0  0.02
#pragma parameter hue    "LiquidPulse: sheen colour"           0.5  0.0  1.0  0.01
#pragma parameter chroma "LiquidPulse: colour split"           0.35 0.0  1.0  0.01

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
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y) * 2.0 - 1.0;
}

/* Three octaves is the fewest that still reads as water: one for the swell,
 * one for the wave, one for the texture on top of the wave. */
float fbm(vec2 p)
{
    return 0.58 * vnoise(p)
         + 0.28 * vnoise(p * 2.07 + 13.1)
         + 0.14 * vnoise(p * 4.13 + 41.7);
}

/* The height field of the pool at q. Time moves two octave sets against
 * each other so the surface never repeats visibly, and `air` lays a fast,
 * fine chop over the slow swell. */
float surface(vec2 q, float t, float chop)
{
    float h = fbm(q + vec2(t * 0.6, t * 0.37))
            + 0.6 * fbm(q * 1.7 - vec2(t * 0.41, t * 0.83) + 27.3);
    /* Chop: fine, fast, and NOISE — a product of two sines lays a lattice
     * of identical dashes over the frame, which reads as a screen door
     * rather than as water. */
    if (chop > 0.0)
        h += chop * (0.55 * vnoise(q * 8.1 + vec2(t * 3.1, -t * 2.7))
                   + 0.40 * vnoise(q * 17.3 - vec2(t * 4.3, t * 3.9) + 53.0));
    return h;
}

void main()
{
    vec2  uv  = vTexCoord;
    float asp = params.OutputSize.x * params.OutputSize.w;
    vec2  p   = (uv - 0.5) * vec2(asp, 1.0);

    float b = clamp(params.bass, 0.0, 1.0);
    float m = clamp(params.mids, 0.0, 1.0);
    float a = clamp(params.air,  0.0, 1.0);

    /* The vortex: a rotation that is strongest at the middle and dies off
     * outward, so the pool twists rather than the frame spinning. */
    float r  = length(p);
    float aw = radians(params.swirl) * m / (1.0 + 3.0 * r);
    float ca = cos(aw), sa = sin(aw);
    vec2  pr = vec2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);

    vec2  q  = pr * params.scale;
    float t  = params.Time * params.speed;

    /* Gradient of the height field = the surface normal, and the direction
     * the liquid drags what is underneath it. One field does both, so the
     * highlights always sit on the slopes that are actually bending the
     * picture. */
    float e  = 0.012 * params.scale;
    float chop = params.ripple * 1.4 * a;
    float h  = surface(q, t, chop);
    float hx = surface(q + vec2(e, 0.0), t, chop) - surface(q - vec2(e, 0.0), t, chop);
    float hy = surface(q + vec2(0.0, e), t, chop) - surface(q - vec2(0.0, e), t, chop);
    vec2  grad = vec2(hx, hy);

    /* Bass deepens the whole pool: the same field, further pushed. */
    float amp = params.warp * 0.01 * (0.30 + 1.4 * b);
    vec2  off = grad * amp;

    /* The chop is already in the height field, so it displaces and lights
     * through the same gradient the swell does — one surface, not a ripple
     * pattern pasted on top of one. */

    vec2 duv = off / vec2(asp, 1.0);

    /* Split the channels across the steepest slopes: thick liquid disperses
     * what you see through it, and the effect is strongest where the
     * surface is most tilted. */
    float sp = length(grad) * params.chroma * amp * 0.9;
    vec2  sd = (length(grad) > 1e-5 ? grad / length(grad) : vec2(0.0)) * sp / vec2(asp, 1.0);

    vec3 col;
    col.r = texture(Source, clamp(uv + duv + sd, 0.0, 1.0)).r;
    col.g = texture(Source, clamp(uv + duv,      0.0, 1.0)).g;
    col.b = texture(Source, clamp(uv + duv - sd, 0.0, 1.0)).b;

    /* Sheen. A fixed key light over the left shoulder, a fake normal from
     * the gradient, and a tight specular — enough to say "there is a
     * surface here" without pretending to be a renderer. */
    vec3  n   = normalize(vec3(-grad * 3.0, 1.0));
    vec3  l   = normalize(vec3(-0.45, -0.65, 0.75));
    float spec = pow(clamp(dot(n, l), 0.0, 1.0), 28.0);
    float rim  = pow(1.0 - clamp(n.z, 0.0, 1.0), 2.0);
    vec3  tint = mix(vec3(0.55, 0.80, 1.00), vec3(1.00, 0.80, 0.55), params.hue);

    col += tint * (spec * 1.4 + rim * 0.25) * params.sheen * (0.6 + 0.8 * b);

    vec3 src = texture(Source, uv).rgb;
    FragColor = vec4(clamp(mix(src, col, clamp(params.amount, 0.0, 1.0)), 0.0, 1.0), 1.0);
}
