#version 450

// spectrum-smear — prismatic rainbow streak, single pass.
//
// Light through cheap glass: the frame is smeared along a direction (or
// radially out from centre) with each tap tinted by its position in the
// spectrum, so highlights drag little rainbows. `vivid` blends between
// physically-mannered fringing and full candy prism.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float length_sp;
    float angle_sp;
    float radial_sp;
    float vivid_sp;
} params;

#pragma parameter amount    "Prism: mix"          1.0  0.0    1.0   0.01
#pragma parameter length_sp "Prism: length (px)"  24.0 0.0    120.0 1.0
#pragma parameter angle_sp  "Prism: angle (deg)"  90.0 -180.0 180.0 1.0
#pragma parameter radial_sp "Prism: radial"       0.0  0.0    1.0   1.0
#pragma parameter vivid_sp  "Prism: vividness"    0.6  0.0    1.0   0.01

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

// Rainbow across t in [0,1]: red → green → blue.
vec3 spectral(float t)
{
    vec3 c = vec3(max(0.0, 1.0 - abs(t - 0.15) * 3.2),
                  max(0.0, 1.0 - abs(t - 0.5) * 3.2),
                  max(0.0, 1.0 - abs(t - 0.85) * 3.2));
    return mix(vec3(1.0), c / max(max(c.r, max(c.g, c.b)), 1e-4), params.vivid_sp);
}

void main()
{
    vec2 dirLinear = vec2(cos(radians(params.angle_sp)), sin(radians(params.angle_sp)));
    vec2 v = vTexCoord - 0.5;
    vec2 dirRadial = length(v) > 1e-4 ? normalize(v) * length(v) * 2.0 : vec2(0.0);
    vec2 dir = mix(dirLinear, dirRadial, step(0.5, params.radial_sp))
             * params.length_sp * params.SourceSize.zw;

    vec3 acc = vec3(0.0);
    vec3 wsum = vec3(1e-4);
    for (int i = 0; i < 12; i++) {
        float t = float(i) / 11.0;
        vec3 w = spectral(t);
        acc += texture(Source, vTexCoord - dir * (t - 0.5)).rgb * w;
        wsum += w;
    }
    vec3 col = acc / wsum;

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
