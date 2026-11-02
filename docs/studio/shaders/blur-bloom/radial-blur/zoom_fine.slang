#version 450

// radial-blur pass 0 — the fine half of a two-pass zoom smear. Zooming is
// multiplicative, so the smear is uniform in log-scale: this pass covers
// the first 1/24 of the log range, and the final pass's 24 taps (each a
// further 1/24 in) compose with it into 576 evenly spaced zoom samples.
// A single 24-tap pass left visible stacked copies at high strength.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float strength;
    float center_x;
    float center_y;
    float falloff;
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
    vec2 uv = vTexCoord;
    vec2 center = vec2(params.center_x, params.center_y);
    float local = params.strength * mix(1.0, smoothstep(0.0, 0.5, length(center - uv)), params.falloff);
    float span = -log(1.0 - 0.5 * local);                 // log-scale length of the smear

    vec3 sum = vec3(0.0);
    for (int i = 0; i < N; ++i) {
        float t = (float(i) + 0.5) / float(N * N);        // 0..1/N of the smear
        sum += textureLod(Source, center + (uv - center) * exp(-span * t), 0.0).rgb;
    }
    FragColor = vec4(sum / float(N), 1.0);
}
