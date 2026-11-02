#version 450

// gaussian-blur pass 0 — copies the input into a float framebuffer whose
// full mip chain feeds the blur passes (mipmap_input1 in the .slangp), and
// optionally decodes to linear light first. The mips are the point: a big
// sigma samples a level whose texels are ~sigma/6 wide, so every tap
// averages the detail between it and its neighbour instead of skipping it —
// skipped detail is what made large radii blocky.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float linlight;
} params;

#pragma parameter linlight "Blur: linear light" 0.0 0.0 1.0 1.0
// Gamma-expanding a white silhouette widens it: keep coverage linear.
#pragma matte_ignore linlight

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
    vec3 c = textureLod(Source, vTexCoord, 0.0).rgb;
    FragColor = vec4(mix(c, pow(max(c, 0.0), vec3(2.2)), params.linlight), 1.0);
}
