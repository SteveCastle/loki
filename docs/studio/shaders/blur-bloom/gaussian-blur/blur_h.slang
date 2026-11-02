#version 450

// gaussian-blur pass 1 — horizontal Gaussian, true sigma = `radius` px
// (scaled down by a positive `aspect`). Taps sit one mip texel apart, the
// level picked so a texel is at most sigma/6 wide: 43 taps then always span
// +-3.5 sigma, and because each tap reads a prefiltered texel the kernel
// stays continuous at any radius. The output is piecewise linear between
// those texels, and the vertical pass's sigma (6x wider) erases it.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float radius;
    float aspect;
} params;

#pragma parameter radius "Blur: radius (px)" 3.0 0.0 250.0 0.25
#pragma parameter aspect "Blur: vertical <-> horizontal" 0.0 -1.0 1.0 0.01

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;   // prep, mipmapped

void main()
{
    float sigma = max(params.radius * clamp(1.0 + params.aspect, 0.0, 1.0), 0.05);
    float texel = max(sigma / 6.0, 1.0);                  // px between taps
    float lod = log2(texel);
    float k = -0.5 / (sigma * sigma);

    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    for (int i = -21; i <= 21; ++i) {
        float x = float(i) * texel;
        float w = exp(x * x * k);
        sum += textureLod(Source, vTexCoord + vec2(x * params.SourceSize.z, 0.0), lod) * w;
        wsum += w;
    }
    FragColor = sum / wsum;
}
