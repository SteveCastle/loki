#version 450

// vignette — radial corner darkening with size, softness, and shape
// (roundness 1 = circular in aspect-corrected space, 0 = hugs the
// frame). amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float strength;
    float v_size;
    float softness;
    float roundness;
} params;

#pragma parameter amount       "Vignette: mix" 1.0 0.0 1.0 0.01
#pragma parameter strength     "Vignette strength" 0.35 0.0 1.0 0.01
#pragma parameter v_size       "Vignette size" 0.75 0.0 1.5 0.01
#pragma parameter softness     "Vignette softness" 0.45 0.05 1.5 0.01
#pragma parameter roundness    "Roundness (frame-circular)" 1.0 0.0 1.0 0.02

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
    float aspect = params.OutputSize.x * params.OutputSize.w;
    vec2 vp = (uv - 0.5) * mix(vec2(2.0), vec2(aspect, 1.0) * 1.4142,
                               params.roundness);
    float d = length(vp);
    float mask = 1.0 - params.strength
               * smoothstep(params.v_size, params.v_size + params.softness, d);
    FragColor = vec4(mix(base, base * mask, params.amount), 1.0);
}
