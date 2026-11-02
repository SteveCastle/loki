#version 450

// radial-blur pass 1 — zoom streak toward a movable focal point. Each
// pixel averages along the ray to the center; strength sets how far along
// that ray the smear reaches. 24 taps a whole 1/24 of the (log-scale)
// smear apart over pass 0's fine smear (see zoom_fine.slang). amount = 0
// is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float strength;
    float center_x;
    float center_y;
    float falloff;
} params;

#pragma parameter amount   "Radial: mix" 1.0 0.0 1.0 0.01
#pragma parameter strength "Radial: strength" 0.25 0.0 1.0 0.01
#pragma parameter center_x "Radial: center x" 0.5 0.0 1.0 0.01
#pragma parameter center_y "Radial: center y" 0.5 0.0 1.0 0.01
#pragma parameter falloff  "Radial: protect center" 0.0 0.0 1.0 0.01

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;     // fine smear (float)
layout(set = 0, binding = 3) uniform sampler2D Original;   // clean input

float hash3(vec3 p)
{
    p = fract(p * vec3(123.34, 345.45, 678.56));
    p += dot(p, p.yzx + 34.345);
    return fract((p.x + p.y) * p.z);
}

const int N = 24;

void main()
{
    vec2 uv = vTexCoord;
    vec2 center = vec2(params.center_x, params.center_y);
    // Optionally keep a sharp core near the focal point.
    float local = params.strength * mix(1.0, smoothstep(0.0, 0.5, length(center - uv)), params.falloff);
    float span = -log(1.0 - 0.5 * local);

    vec3 sum = vec3(0.0);
    for (int i = 0; i < N; ++i) {
        float t = float(i) / float(N);
        sum += textureLod(Source, center + (uv - center) * exp(-span * t), 0.0).rgb;
    }
    vec3 c = sum / float(N);

    // TPDF dither (see gaussian-blur): long smears are smooth ramps.
    vec2 px = uv * params.OutputSize.xy;
    float fr = float(params.FrameCount % 1024u);
    c += (hash3(vec3(px, fr)) + hash3(vec3(px + 917.0, fr)) - 1.0) * (1.0 / 255.0);

    vec3 base = textureLod(Original, uv, 0.0).rgb;
    FragColor = vec4(mix(base, clamp(c, 0.0, 1.0), params.amount), 1.0);
}
