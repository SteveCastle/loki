#version 450

// gaussian-blur pass 0 — horizontal Gaussian, true sigma = `radius` px.
// A 25-tap loop with spacing that scales as sigma/4 stays well-sampled at
// any radius (the old sparse 9-tap kernel undersampled big radii and left
// stepping artifacts). Rendered into a float framebuffer (see .slangp) so
// the intermediate never quantizes to 8 bits — that quantization is where
// gradient banding is born.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float radius;
} params;

#pragma parameter radius "Blur: radius (px)" 3.0 0.25 32.0 0.25

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
    float sigma = max(params.radius, 0.05);
    float spacing = max(sigma * 0.25, 1.0);          // px between taps
    vec2 step_uv = vec2(spacing * params.SourceSize.z, 0.0);

    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    for (int i = -12; i <= 12; ++i) {
        float x = float(i) * spacing;
        float w = exp(-0.5 * (x * x) / (sigma * sigma));
        sum += texture(Source, vTexCoord + float(i) * step_uv) * w;
        wsum += w;
    }
    FragColor = sum / wsum;
}
