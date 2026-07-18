#version 450

// motion-bloom pass 2 — horizontal 9-tap Gaussian.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float radius;
} params;

#pragma parameter radius "MotionBloom: radius" 3.0 0.5 8.0 0.25

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

void main()
{
    vec2 uv = vTexCoord;
    vec2 o = vec2(params.radius * params.SourceSize.z, 0.0);
    vec4 sum = texture(Source, uv) * 0.2270270270;
    sum += texture(Source, uv - 4.0 * o) * 0.0162162162;
    sum += texture(Source, uv - 3.0 * o) * 0.0540540541;
    sum += texture(Source, uv - 2.0 * o) * 0.1216216216;
    sum += texture(Source, uv - 1.0 * o) * 0.1945945946;
    sum += texture(Source, uv + 1.0 * o) * 0.1945945946;
    sum += texture(Source, uv + 2.0 * o) * 0.1216216216;
    sum += texture(Source, uv + 3.0 * o) * 0.0540540541;
    sum += texture(Source, uv + 4.0 * o) * 0.0162162162;
    FragColor = sum;
}
