#version 450

// motion-bloom pass 4 — composite the blurred motion-glow (Source) additively
// over the clean Original, so moving regions bleed coloured light while static
// areas stay sharp. amount = 0 is a clean passthrough of the Original.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float intensity;
    float sat;
} params;

#pragma parameter amount    "MotionBloom: mix"        1.0  0.0  1.0  0.01
#pragma parameter intensity "MotionBloom: intensity"  2.0  0.0  6.0  0.05
#pragma parameter sat       "MotionBloom: glow sat"   1.6  0.0  4.0  0.05

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;     // blurred motion glow
layout(set = 0, binding = 3) uniform sampler2D Original;   // clean input

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main()
{
    vec2 uv = vTexCoord;
    vec3 base = texture(Original, uv).rgb;
    vec3 glow = texture(Source, uv).rgb * params.intensity;

    float l = luma(glow);
    glow = max(vec3(0.0), mix(vec3(l), glow, params.sat));

    vec3 outc = 1.0 - (1.0 - base) * (1.0 - clamp(glow, 0.0, 1.0));  // screen
    FragColor = vec4(mix(base, clamp(outc, 0.0, 1.0), params.amount), 1.0);
}
