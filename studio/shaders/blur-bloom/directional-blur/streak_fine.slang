#version 450

// directional-blur pass 0 — the fine half of a two-pass box streak. This
// pass smears each pixel over a 1/24 slice of `dist`; the final pass then
// takes 24 taps a whole slice apart over this output. Together that is 576
// evenly spaced samples (~1 px apart at 600 px), so long streaks are a
// smooth smear instead of stacked ghost copies.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float angle;
    float dist;
} params;

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

const int N = 24;

void main()
{
    float a = radians(params.angle);
    vec2 slice = vec2(cos(a), sin(a)) * (params.dist / float(N)) * params.SourceSize.zw;

    vec3 sum = vec3(0.0);
    for (int i = 0; i < N; ++i) {
        float t = (float(i) + 0.5) / float(N) - 0.5;     // -0.5..0.5 of a slice
        sum += textureLod(Source, vTexCoord + slice * t, 0.0).rgb;
    }
    FragColor = vec4(sum / float(N), 1.0);
}
