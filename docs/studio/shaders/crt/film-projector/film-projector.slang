#version 450

// film-projector — small-gauge film projection (Super 8 / 16mm), single pass.
//
// The whole chain from reel to wall: the film weaves in the gate and jumps
// when a splice rattles through, the two-blade shutter and an aging lamp
// beat against each other as flicker, grain boils in the midtones, vertical
// scratches wander across the print, dust and fibers land on individual
// frames, and the lamp hotspot falls off into a warm vignette. Dirt and
// grain re-roll at film cadence (`fps`), not display rate — that 18/24fps
// chatter is most of what reads as "projected".
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float fps;
    float weave;
    float flicker;
    float grain;
    float scratches;
    float dust;
    float splice;
    float warmth;
    float vignette;
    float speed;
    float seed;
} params;

#pragma parameter amount    "Film: amount"          1.0  0.0  1.0  0.01
#pragma parameter fps       "Film: cadence fps"     18.0 8.0  30.0 1.0
#pragma parameter weave     "Film: gate weave"      0.5  0.0  1.0  0.01
#pragma parameter flicker   "Film: shutter flicker" 0.4  0.0  1.0  0.01
#pragma parameter grain     "Film: grain"           0.4  0.0  1.0  0.01
#pragma parameter scratches "Film: scratches"       0.5  0.0  1.0  0.01
#pragma parameter dust      "Film: dust & fibers"   0.5  0.0  1.0  0.01
#pragma parameter splice    "Film: splice jumps"    0.35 0.0  1.0  0.01
#pragma parameter warmth    "Film: lamp warmth"     0.5  0.0  1.0  0.01
#pragma parameter vignette  "Film: vignette"        0.5  0.0  1.0  0.01
#pragma parameter speed     "Film: speed"           1.0  0.0  4.0  0.05
#pragma parameter seed      "Film: seed"            13.0 0.0 64.0  1.0

/* Dirt, grain, flicker and vignette are all paint/dimming — none of it
 * should thin the layer's coverage. */
#pragma matte_ignore amount

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

float hash1(float n) { return fract(sin(n * 17.3217) * 43758.5453); }
float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}
float vnoise1(float x)
{
    float i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(hash1(i), hash1(i + 1.0), f);
}

void main()
{
    vec2  uv  = vTexCoord;
    float amt = params.amount;
    float t   = float(params.FrameCount) * (1.0 / 60.0) * params.speed;
    float sd  = params.seed;

    /* Film-frame index: dirt, grain and jumps change at print cadence. */
    float fq = floor(t * max(params.fps, 1.0));

    /* Gate weave: the frame never sits quite still — slow wander plus a
     * tiny per-frame registration error. */
    vec2 weave = (vec2(vnoise1(t * 2.3 + sd), vnoise1(t * 1.7 + sd + 9.0)) - 0.5)
               * params.weave * 0.006
               + (vec2(hash1(fq + sd), hash1(fq + sd + 4.0)) - 0.5)
               * params.weave * 0.002;

    /* Splice: a repaired cut rattles through the gate — the frame slips
     * vertically for an instant and the frame line flashes into view. */
    float sp    = step(hash1(fq * 0.731 + sd), params.splice * 0.04);
    float slip  = sp * (hash1(fq * 1.37 + sd) - 0.3) * 0.35;
    vec2  suv   = clamp(uv + weave * amt + vec2(0.0, slip * amt), 0.0, 1.0);

    vec3 col = texture(Source, suv).rgb;

    /* Frame line: the black bar between frames rides the splice slip. */
    float lineY = fract(uv.y + slip * 2.0);
    float fline = sp * (step(lineY, 0.05) + step(0.95, lineY));
    col *= 1.0 - clamp(fline, 0.0, 1.0) * amt;

    /* Shutter + lamp flicker: a fast blade beat over a slow lamp breath,
     * plus per-frame exposure wobble from the print itself. */
    float fl = 1.0
             - params.flicker * amt * (0.05 + 0.05 * sin(t * 150.8 + sd)
                                       + 0.06 * (hash1(fq * 3.1 + sd) - 0.5)
                                       + 0.04 * vnoise1(t * 0.7 + sd));
    col *= fl;

    /* Print grain: boils per film frame, strongest in the midtones. */
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    float g = (hash21(floor(uv * params.SourceSize.xy * 0.6) + fq * 7.3 + sd) - 0.5)
            * params.grain * 0.35 * (0.25 + luma * (1.0 - luma) * 3.0);
    col += g * amt;

    /* Vertical scratches: a few lines that drift, vanish, and return.
     * Emulsion-side scratches print dark, base-side print light. */
    for (int i = 0; i < 4; i++) {
        float fi   = float(i);
        float on   = step(hash21(vec2(fi, floor(t * 0.4) + sd)), params.scratches * 0.8);
        float x0   = hash1(fi * 9.7 + sd + floor(t * 0.4) * 3.0)
                   + (vnoise1(t * 3.1 + fi * 11.0) - 0.5) * 0.01;
        float m    = on * (1.0 - smoothstep(0.0, params.SourceSize.z * 1.8, abs(uv.x - x0)));
        float tone = hash1(fi * 5.3 + sd) < 0.5 ? -0.35 : 0.3;
        col += tone * m * amt * (0.5 + 0.5 * hash21(vec2(uv.y * 40.0, fq)));
    }

    /* Dust: specks and short fibers that live for exactly one frame. */
    float aspect = params.SourceSize.x / params.SourceSize.y;
    for (int i = 0; i < 6; i++) {
        float fi = float(i);
        float on = step(hash21(vec2(fi * 3.3, fq) + sd), params.dust * 0.6);
        vec2  dp = vec2(hash21(vec2(fq, fi) + sd), hash21(vec2(fi, fq) + sd * 2.0));
        vec2  dv = (uv - dp) * vec2(aspect, 1.0);
        /* squash randomly into a sliver → reads as a fiber, not a dot */
        dv.x /= 0.2 + 0.8 * hash21(vec2(fi, fq) + sd + 7.0);
        float r  = 0.002 + 0.004 * hash21(vec2(fq, fi * 7.7) + sd);
        float m  = on * (1.0 - smoothstep(r * 0.5, r, length(dv)));
        col = mix(col, vec3(0.05), m * amt * 0.9);
    }

    /* Lamp: warm tint, lifted blacks, hotspot falling into vignette. */
    vec3 warm = mix(vec3(1.0), vec3(1.08, 1.0, 0.85), params.warmth * amt);
    col = col * warm + vec3(0.03, 0.025, 0.02) * params.warmth * amt;
    vec2  c = (uv - 0.5) * vec2(aspect, 1.0);
    float r2 = dot(c, c);
    col *= 1.0 + 0.08 * params.vignette * amt * exp(-r2 * 6.0);   /* hotspot */
    col *= 1.0 - params.vignette * amt * smoothstep(0.15, 0.75, r2);

    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
