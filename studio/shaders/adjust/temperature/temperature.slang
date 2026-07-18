#version 450

// temperature — photographic white balance: warm/cool along the
// orange-blue axis and green/magenta tint, optionally renormalised so
// overall brightness stays put. Neutral by default.
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float temperature;
    float tint;
    float luma_lock;
} params;

#pragma parameter amount       "White balance: mix" 1.0 0.0 1.0 0.01
#pragma parameter temperature  "Temperature (cool-warm)" 0.0 -1.0 1.0 0.01
#pragma parameter tint         "Tint (green-magenta)" 0.0 -1.0 1.0 0.01
#pragma parameter luma_lock    "Preserve brightness" 1.0 0.0 1.0 1.0

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

void main()
{
    vec3 base = texture(Source, vTexCoord).rgb;
    float l0 = luma(base);
    vec3 c = base * vec3(1.0 + 0.25 * params.temperature,
                         1.0 + 0.20 * params.tint,
                         1.0 - 0.25 * params.temperature);
    if (params.luma_lock > 0.5) {
        float l1 = luma(c);
        c *= (l1 > 1e-4) ? l0 / l1 : 1.0;
    }
    FragColor = vec4(mix(base, clamp(c, 0.0, 1.0), params.amount), 1.0);
}
