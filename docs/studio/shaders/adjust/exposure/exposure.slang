#version 450

// exposure — basic tone controls: exposure (photographic stops),
// brightness offset, contrast around mid-grey, and gamma. All defaults
// are neutral, so adding the layer changes nothing until you dial it.
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float exposure;
    float brightness;
    float contrast;
    float gamma_adj;
} params;

#pragma parameter amount       "Exposure: mix" 1.0 0.0 1.0 0.01
#pragma parameter exposure     "Exposure (stops)" 0.0 -3.0 3.0 0.05
#pragma parameter brightness   "Brightness" 0.0 -0.5 0.5 0.01
#pragma parameter contrast     "Contrast" 1.0 0.25 2.5 0.05
#pragma parameter gamma_adj    "Gamma" 1.0 0.25 3.0 0.05

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
    vec3 base = texture(Source, vTexCoord).rgb;
    vec3 c = base * exp2(params.exposure) + params.brightness;
    c = (c - 0.5) * params.contrast + 0.5;
    c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / params.gamma_adj));
    FragColor = vec4(mix(base, c, params.amount), 1.0);
}
