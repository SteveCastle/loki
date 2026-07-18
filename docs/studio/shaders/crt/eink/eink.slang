#version 450

// eink — e-paper display, single self-feedback pass.
//
// The anti-CRT: no glow, no scanlines — warm paper, dark ink, a few gray
// levels pushed through a 4×4 ordered dither, static paper grain, and the
// tell-tale refresh ghosting where the previous image never quite clears.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float ink_levels;
    float dither_e;
    float paper_e;
    float ghost_e;
    float grain_e;
    float contrast_e;
} params;

#pragma parameter amount     "E-ink: mix"         1.0  0.0 1.0  0.01
#pragma parameter ink_levels "E-ink: gray levels" 4.0  2.0 16.0 1.0
#pragma parameter dither_e   "E-ink: dither"      0.7  0.0 1.0  0.05
#pragma parameter paper_e    "E-ink: paper tone"  0.85 0.5 1.0  0.01
#pragma parameter ghost_e    "E-ink: ghosting"    0.35 0.0 0.9  0.01
#pragma parameter grain_e    "E-ink: paper grain" 0.05 0.0 0.3  0.01
#pragma parameter contrast_e "E-ink: contrast"    1.15 0.5 2.0  0.05

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
layout(set = 0, binding = 3) uniform sampler2D PassFeedback0;

float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// 4×4 Bayer threshold, 0..1.
float bayer4(vec2 px)
{
    vec2 p = mod(floor(px), 4.0);
    float x = p.x;
    float y = p.y;
    float v = mod(x + y * 4.0 + x * 2.0 * mod(y, 2.0) + mod(x, 2.0) * 8.0, 16.0);
    // A compact permutation that covers 0..15 well enough for paper.
    return (v + 0.5) / 16.0 - 0.5;
}

void main()
{
    vec2 px = vTexCoord * params.OutputSize.xy;

    float l = dot(texture(Source, vTexCoord).rgb, vec3(0.299, 0.587, 0.114));
    l = clamp((l - 0.5) * params.contrast_e + 0.5, 0.0, 1.0);
    l += bayer4(px) * params.dither_e / max(params.ink_levels, 2.0);
    float n = max(params.ink_levels, 2.0) - 1.0;
    l = clamp(floor(l * n + 0.5) / n, 0.0, 1.0);

    // Warm paper + slightly blue-black ink; static (non-animated) grain.
    vec3 paper = vec3(0.94, 0.92, 0.87) * params.paper_e;
    vec3 ink = vec3(0.10, 0.10, 0.12);
    float grain = (hash(floor(px * 0.5)) - 0.5) * params.grain_e;
    vec3 col = mix(ink, paper, clamp(l + grain, 0.0, 1.0));

    // Refresh ghosting: the darker of this frame and the fading last one.
    vec3 prev = texture(PassFeedback0, vTexCoord).rgb;
    col = min(col, mix(vec3(1.0), prev, params.ghost_e) + (1.0 - params.ghost_e));

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
