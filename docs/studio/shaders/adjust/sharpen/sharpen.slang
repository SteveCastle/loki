#version 450

// sharpen — unsharp mask: subtract a small cross blur and add the
// difference back scaled by `strength`. amount = 0 is a clean
// passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float strength;
    float s_radius;
} params;

#pragma parameter amount       "Sharpen: mix" 1.0 0.0 1.0 0.01
#pragma parameter strength     "Sharpen strength" 0.5 0.0 3.0 0.05
#pragma parameter s_radius     "Sharpen radius (px)" 1.0 0.25 3.0 0.25

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
    vec2 t = params.s_radius * params.SourceSize.zw;
    vec3 base = texture(Source, uv).rgb;
    vec3 blur = base * 0.4
              + texture(Source, uv + vec2( t.x, 0.0)).rgb * 0.15
              + texture(Source, uv + vec2(-t.x, 0.0)).rgb * 0.15
              + texture(Source, uv + vec2(0.0,  t.y)).rgb * 0.15
              + texture(Source, uv + vec2(0.0, -t.y)).rgb * 0.15;
    vec3 c = base + (base - blur) * params.strength;
    FragColor = vec4(mix(base, clamp(c, 0.0, 1.0), params.amount), 1.0);
}
