#version 450

// magnet-warp — magnet held against a CRT, single pass.
//
// A magnet near the tube bends the electron beams before they land: the
// geometry pinches and twists toward the magnet, and because the three
// guns take different paths, the beams land on the *wrong* phosphors —
// the picture blooms into rainbow purity error that follows the
// distortion. Waving the magnet makes the blob swim, and hitting the
// degauss button fires the classic "thunk": a decaying shudder that
// shakes the whole raster while the colors swing wildly before settling.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float pos_x;
    float pos_y;
    float radius;
    float pinch;
    float swirl;
    float purity;
    float beam_split;
    float wobble;
    float degauss;
    float speed;
    float seed;
} params;

#pragma parameter amount     "Magnet: amount"          1.0  0.0  1.0  0.01
#pragma parameter pos_x      "Magnet: x"               0.35 0.0  1.0  0.01
#pragma parameter pos_y      "Magnet: y"               0.40 0.0  1.0  0.01
#pragma parameter radius     "Magnet: field radius"    0.35 0.05 1.0  0.01
#pragma parameter pinch      "Magnet: pinch"           0.35 -1.0 1.0  0.01
#pragma parameter swirl      "Magnet: swirl"           1.6  -4.0 4.0  0.05
#pragma parameter purity     "Magnet: rainbow purity"  0.7  0.0  1.0  0.01
#pragma parameter beam_split "Magnet: beam separation" 0.5  0.0  1.0  0.01
#pragma parameter wobble     "Magnet: hand wobble"     0.4  0.0  1.0  0.01
#pragma parameter degauss    "Magnet: degauss thunk"   0.5  0.0  1.0  0.01
#pragma parameter speed      "Magnet: speed"           1.0  0.0  4.0  0.05
#pragma parameter seed       "Magnet: seed"            3.0  0.0 64.0  1.0

/* Colour-only: the rainbow tint bands would ripple the coverage matte. */
#pragma matte_ignore purity

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

/* Warp one beam: pinch toward the magnet + tangential swirl, gaussian
 * falloff over the field radius. `k` scales the whole deflection — each
 * gun gets a slightly different k, which is where the rainbow comes from. */
vec2 beamWarp(vec2 uv, vec2 center, float aspect, float radius,
              float pinch, float swirl, float k)
{
    vec2  p = (uv - center) * vec2(aspect, 1.0);
    float r = length(p);
    float g = exp(-(r * r) / (radius * radius * 0.5)) * k;

    /* swirl: rotate around the magnet, strongest at the core */
    float a  = swirl * g;
    float ca = cos(a), sa = sin(a);
    p = vec2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);

    /* pinch: pull toward (or push away from) the magnet */
    p *= 1.0 - pinch * g;

    return center + p / vec2(aspect, 1.0);
}

void main()
{
    vec2  uv  = vTexCoord;
    float amt = params.amount;
    float t   = float(params.FrameCount) * (1.0 / 60.0) * params.speed;

    float aspect = params.SourceSize.x / params.SourceSize.y;

    /* Hand wobble: the magnet position and strength swim around. */
    vec2 center = vec2(params.pos_x, params.pos_y)
                + params.wobble * 0.06 * vec2(sin(t * 1.3 + params.seed),
                                              cos(t * 0.9 + params.seed * 2.0));
    float str = amt * (1.0 - params.wobble * 0.35 * (0.5 + 0.5 * sin(t * 2.1 + params.seed)));

    /* Degauss thunk every few seconds: a sharp decaying ring that both
     * shakes the raster and momentarily slams the field strength. */
    float period = 3.5;
    float tau    = mod(t + params.seed, period);
    float ring   = exp(-tau * 5.0) * sin(tau * 42.0) * params.degauss;
    uv  += ring * 0.012 * vec2(sin(uv.y * 21.0 + t * 50.0), 0.6) * amt;
    str *= 1.0 + abs(ring) * 3.0;

    /* Three guns, three slightly different deflections → color fringing
     * that grows with the field, exactly like real purity error. */
    float sep = params.beam_split * 0.18;
    vec2 ur = beamWarp(uv, center, aspect, params.radius, params.pinch, params.swirl, str * (1.0 + sep));
    vec2 ug = beamWarp(uv, center, aspect, params.radius, params.pinch, params.swirl, str);
    vec2 ub = beamWarp(uv, center, aspect, params.radius, params.pinch, params.swirl, str * (1.0 - sep));

    vec3 col;
    col.r = texture(Source, clamp(ur, 0.0, 1.0)).r;
    col.g = texture(Source, clamp(ug, 0.0, 1.0)).g;
    col.b = texture(Source, clamp(ub, 0.0, 1.0)).b;

    /* Rainbow purity blotch: hue bands ride the field falloff, slowly
     * rotating; multiplicative so black stays black (phosphors that were
     * never lit can't glow). */
    vec2  pp = (uv - center) * vec2(aspect, 1.0);
    float r  = length(pp);
    float g  = exp(-(r * r) / (params.radius * params.radius * 0.5)) * str;
    float hue = r * 9.0 - t * 0.4 + atan(pp.y, pp.x) * 0.5;
    vec3  tint = 0.5 + 0.5 * cos(6.2832 * (hue * 0.15915 + vec3(0.0, 0.33, 0.67)));
    col *= 1.0 + (tint - 0.5) * 2.0 * params.purity * g;

    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
