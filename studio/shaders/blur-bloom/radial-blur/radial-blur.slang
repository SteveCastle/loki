#version 450

// radial-blur — zoom streak toward/away from a movable focal point. Each
// pixel averages samples along the ray to the center; strength sets how
// far along that ray the smear reaches. amount = 0 is a clean passthrough.

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
layout(set = 0, binding = 2) uniform sampler2D Source;

void main()
{
    vec2 uv = vTexCoord;
    vec3 base = texture(Source, uv).rgb;

    vec2 center = vec2(params.center_x, params.center_y);
    vec2 toward = center - uv;

    // Optionally keep a sharp core near the focal point.
    float dist = length(toward);
    float local = params.strength * mix(1.0, smoothstep(0.0, 0.5, dist), params.falloff);

    vec3 sum = vec3(0.0);
    for (int i = 0; i < 24; ++i) {
        float t = (float(i) + 0.5) / 24.0;          // 0..1 along the ray
        sum += texture(Source, uv + toward * (local * 0.5) * t).rgb;
    }
    vec3 c = sum / 24.0;

    FragColor = vec4(mix(base, c, params.amount), 1.0);
}
