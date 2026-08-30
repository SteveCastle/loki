#version 450

// contrast — pivot contrast with an optional filmic S and local contrast,
// single pass.
//
// Three tools that usually get conflated, on separate sliders: `contrast`
// scales linearly around a movable pivot (what "contrast" sliders do),
// `scurve` blends in a smooth film-print S that rolls off toward black and
// white instead of clipping, and `clarity` is wide-radius local contrast —
// the midtone punch move. `sat` compensates the saturation swing that
// contrast changes drag along. Neutral by default; amount = 0 is a clean
// passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float contrast;
    float pivot;
    float scurve;
    float clarity;
    float sat;
} params;

#pragma parameter amount   "Contrast: mix"      1.0 0.0  1.0 0.01
#pragma parameter contrast "Contrast"           1.0 0.0  2.5 0.01
#pragma parameter pivot    "Pivot"              0.5 0.1  0.9 0.01
#pragma parameter scurve   "Filmic S-curve"     0.0 0.0  1.0 0.01
#pragma parameter clarity  "Clarity (local)"    0.0 -1.0 1.0 0.01
#pragma parameter sat      "Saturation"         1.0 0.0  2.0 0.01

/* Colour-only: crushed blacks must not thin the coverage matte. */
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

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main()
{
    vec2 uv   = vTexCoord;
    vec3 base = texture(Source, uv).rgb;
    vec3 c    = base;

    /* Clarity: unsharp mask on luma at a wide radius — an 8-tap ring is
     * plenty for a low-frequency estimate. Scaling rgb by the luma ratio
     * keeps hue and saturation where they were. */
    if (abs(params.clarity) > 1e-4) {
        vec2  r  = params.SourceSize.zw * 14.0;
        float lb = 0.0;
        lb += luma(texture(Source, clamp(uv + vec2( r.x,  0.0), 0.0, 1.0)).rgb);
        lb += luma(texture(Source, clamp(uv + vec2(-r.x,  0.0), 0.0, 1.0)).rgb);
        lb += luma(texture(Source, clamp(uv + vec2( 0.0,  r.y), 0.0, 1.0)).rgb);
        lb += luma(texture(Source, clamp(uv + vec2( 0.0, -r.y), 0.0, 1.0)).rgb);
        lb += luma(texture(Source, clamp(uv + r * 0.707,        0.0, 1.0)).rgb);
        lb += luma(texture(Source, clamp(uv - r * 0.707,        0.0, 1.0)).rgb);
        lb += luma(texture(Source, clamp(uv + r * vec2(0.707, -0.707), 0.0, 1.0)).rgb);
        lb += luma(texture(Source, clamp(uv - r * vec2(0.707, -0.707), 0.0, 1.0)).rgb);
        lb *= 0.125;
        float l  = luma(c);
        float ln = clamp(l + params.clarity * (l - lb) * 0.9, 0.0, 1.0);
        c *= ln / max(l, 1e-4);
    }

    /* Linear contrast about the pivot. */
    c = (c - params.pivot) * params.contrast + params.pivot;

    /* Filmic S: smooth roll-off toward the ends instead of a hard clip. */
    vec3 cc = clamp(c, 0.0, 1.0);
    c = mix(c, cc * cc * (3.0 - 2.0 * cc), params.scurve);

    /* Saturation compensation. */
    c = mix(vec3(luma(c)), c, params.sat);

    FragColor = vec4(clamp(mix(base, c, params.amount), 0.0, 1.0), 1.0);
}
