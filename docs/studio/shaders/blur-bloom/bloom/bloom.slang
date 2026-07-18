#version 450

// bloom pass 3 — composite: screen-blend the blurred bright-pass (Source) back
// over the clean Original. `intensity` scales the glow; `sat` punches its
// colour so highlights bleed warm/coloured light. amount = 0 is a clean
// passthrough of the Original.

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

#pragma parameter amount    "Bloom: mix"        1.0  0.0  1.0  0.01
#pragma parameter intensity "Bloom: intensity"  1.3  0.0  4.0  0.05
#pragma parameter sat       "Bloom: glow sat"   1.4  0.0  3.0  0.05

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;     // blurred bright-pass
layout(set = 0, binding = 3) uniform sampler2D Original;   // clean input

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main()
{
    vec2 uv = vTexCoord;
    vec3 base = texture(Original, uv).rgb;
    vec3 glow = texture(Source, uv).rgb * params.intensity;

    float l = luma(glow);
    glow = max(vec3(0.0), mix(vec3(l), glow, params.sat));

    // Screen blend keeps highlights from clipping to flat white.
    vec3 outc = 1.0 - (1.0 - base) * (1.0 - clamp(glow, 0.0, 1.0));

    FragColor = vec4(mix(base, clamp(outc, 0.0, 1.0), params.amount), 1.0);
}
