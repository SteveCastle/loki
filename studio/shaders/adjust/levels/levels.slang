#version 450

// levels — classic input/output level remap with midtone gamma, the
// Photoshop Levels dialog as a layer. Neutral by default.
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float in_black;
    float in_white;
    float mid_gamma;
    float out_black;
    float out_white;
} params;

#pragma parameter amount       "Levels: mix" 1.0 0.0 1.0 0.01
#pragma parameter in_black     "Input black point" 0.0 0.0 0.5 0.005
#pragma parameter in_white     "Input white point" 1.0 0.5 1.0 0.005
#pragma parameter mid_gamma    "Midtone gamma" 1.0 0.25 3.0 0.05
#pragma parameter out_black    "Output black point" 0.0 0.0 0.5 0.005
#pragma parameter out_white    "Output white point" 1.0 0.5 1.0 0.005

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
    vec3 c = (base - params.in_black)
           / max(params.in_white - params.in_black, 1e-4);
    c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / params.mid_gamma));
    c = c * (params.out_white - params.out_black) + params.out_black;
    FragColor = vec4(mix(base, clamp(c, 0.0, 1.0), params.amount), 1.0);
}
