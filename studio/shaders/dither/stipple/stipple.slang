#version 450

// stipple — etching / mezzotint stipple dither, single pass.
//
// Random-threshold dithering rendered as ink dots: each micro-cell holds a
// dot whose presence follows the local tone against a static hash
// threshold, with jittered dot centers so the field reads as hand
// stippling rather than a grid.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float cell_st;
    float dot_st;
    float jitter_st;
    float gamma_st;
    float ink4_r;
    float ink4_g;
    float ink4_b;
    float paper4_r;
    float paper4_g;
    float paper4_b;
} params;

#pragma parameter amount    "Stipple: mix"        1.0  0.0  1.0  0.01
#pragma parameter cell_st   "Stipple: cell (px)"  4.0  2.0  16.0 1.0
#pragma parameter dot_st    "Stipple: dot size"   0.38 0.1  0.6  0.01
#pragma parameter jitter_st "Stipple: jitter"     0.8  0.0  1.0  0.05
#pragma parameter gamma_st  "Stipple: tone gamma" 0.9  0.4  2.5  0.05
#pragma parameter ink4_r    "Ink R"               0.15 0.0  1.0  0.01
#pragma parameter ink4_g    "Ink G"               0.12 0.0  1.0  0.01
#pragma parameter ink4_b    "Ink B"               0.10 0.0  1.0  0.01
#pragma parameter paper4_r  "Paper R"             0.95 0.0  1.0  0.01
#pragma parameter paper4_g  "Paper G"             0.93 0.0  1.0  0.01
#pragma parameter paper4_b  "Paper B"             0.87 0.0  1.0  0.01

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

float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main()
{
    float cell = max(params.cell_st, 2.0);
    vec2 gp = vTexCoord * params.OutputSize.xy / cell;
    vec2 id = floor(gp);

    vec3 s = texture(Source, (id + 0.5) * cell * params.OutputSize.zw).rgb;
    float l = pow(clamp(dot(s, vec3(0.299, 0.587, 0.114)), 0.0, 1.0), params.gamma_st);

    // A dot exists in this cell when the static threshold beats the tone
    // (darker → more dots), and grows slightly with darkness.
    float thr = hash(id);
    float have = step(l, thr);
    vec2 center = vec2(0.5) + (vec2(hash(id + 11.0), hash(id + 29.0)) - 0.5) * params.jitter_st * 0.7;
    float d = length(fract(gp) - center);
    float radius = params.dot_st * (0.7 + 0.6 * (thr - l));
    float ink = have * (1.0 - smoothstep(radius - 0.1, radius + 0.06, d));

    vec3 paper = vec3(params.paper4_r, params.paper4_g, params.paper4_b)
               * (1.0 - (hash(floor(gp * 2.3)) - 0.5) * 0.05);
    vec3 col = mix(paper, vec3(params.ink4_r, params.ink4_g, params.ink4_b), ink);

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
