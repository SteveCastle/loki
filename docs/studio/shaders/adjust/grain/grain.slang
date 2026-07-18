#version 450

// grain — animated film grain: true static (each frame is a fresh,
// uncorrelated noise field), optionally per-channel colour grain, and
// weighted toward shadows the way film reads.
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float strength;
    float g_size;
    float shadow_bias;
    float colored;
} params;

#pragma parameter amount       "Grain: mix" 1.0 0.0 1.0 0.01
#pragma parameter strength     "Grain strength" 0.08 0.0 0.5 0.005
#pragma parameter g_size       "Grain size (px)" 1.5 1.0 8.0 0.25
#pragma parameter shadow_bias  "Weight toward shadows" 0.5 0.0 1.0 0.02
#pragma parameter colored      "Colour grain" 0.0 0.0 1.0 1.0

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

// 3D hash: the frame index is a separate dimension, NOT added to the cell
// coords — hash(cell + frame) translates the field diagonally each frame and
// the grain reads as directional drift instead of static.
float hash3(vec3 p)
{
    p = fract(p * vec3(123.34, 345.45, 678.56));
    p += dot(p, p.yzx + 34.345);
    return fract((p.x + p.y) * p.z);
}

void main()
{
    vec2 uv = vTexCoord;
    vec3 base = texture(Source, uv).rgb;
    vec2 cell = floor(uv * params.OutputSize.xy / params.g_size);
    float fr = float(params.FrameCount % 1024u);
    vec3 g;
    if (params.colored > 0.5)
        g = vec3(hash3(vec3(cell, fr)),
                 hash3(vec3(cell + 101.0, fr)),
                 hash3(vec3(cell + 733.0, fr))) - 0.5;
    else
        g = vec3(hash3(vec3(cell, fr)) - 0.5);
    float wt = mix(1.0, 1.0 - luma(base), params.shadow_bias);
    vec3 c = base + g * params.strength * wt;
    FragColor = vec4(mix(base, clamp(c, 0.0, 1.0), params.amount), 1.0);
}
