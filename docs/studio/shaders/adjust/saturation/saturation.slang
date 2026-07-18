#version 450

// saturation — saturation, vibrance (boosts muted colours more than
// already-vivid ones), and hue rotation. Neutral by default.
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float saturation;
    float vibrance;
    float hue_deg;
} params;

#pragma parameter amount       "Saturation: mix" 1.0 0.0 1.0 0.01
#pragma parameter saturation   "Saturation" 1.0 0.0 3.0 0.05
#pragma parameter vibrance     "Vibrance" 0.0 -1.0 1.0 0.02
#pragma parameter hue_deg      "Hue rotate (deg)" 0.0 -180.0 180.0 1.0

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

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// Rotate hue around the grey axis (Rodrigues rotation, no HSV round trip).
vec3 hue_rotate(vec3 c, float rad)
{
    const vec3 k = vec3(0.57735026919);
    float ca = cos(rad), sa = sin(rad);
    return c * ca + cross(k, c) * sa + k * dot(k, c) * (1.0 - ca);
}

void main()
{
    vec3 base = texture(Source, vTexCoord).rgb;
    vec3 c = base;
    float l = luma(c);
    float sat_now = max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b);
    float k = params.saturation + params.vibrance * (1.0 - sat_now);
    c = mix(vec3(l), c, max(k, 0.0));
    c = hue_rotate(c, radians(params.hue_deg));
    FragColor = vec4(mix(base, clamp(c, 0.0, 1.0), params.amount), 1.0);
}
