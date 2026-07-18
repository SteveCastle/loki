#version 450

// pixelate — mosaic resample to `px`-sized blocks (sampled at block
// centres). amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float px;
} params;

#pragma parameter amount       "Pixelate: mix" 1.0 0.0 1.0 0.01
#pragma parameter px           "Pixel size (px)" 8.0 1.0 64.0 1.0

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
    vec3 base = texture(Source, uv).rgb;
    vec2 grid = params.OutputSize.xy / params.px;
    vec2 puv = (floor(uv * grid) + 0.5) / grid;
    vec3 c = texture(Source, puv).rgb;
    FragColor = vec4(mix(base, c, params.amount), 1.0);
}
