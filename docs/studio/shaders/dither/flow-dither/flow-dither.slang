#version 450

// flow-dither — animated flowing dither field, single pass.
//
// The artistic one: a two-tone dither whose threshold field is domain-
// warped by drifting sine currents, so the dither boundary between light
// and dark constantly flows like ink in water. Duotone-tinted; crank the
// warp for full psychedelic marbling.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float Time;
    float amount;
    float scale_fd;
    float warp_fd;
    float speed_fd;
    float levels_fd;
    float dark_r;
    float dark_g;
    float dark_b;
    float light_r;
    float light_g;
    float light_b;
} params;

#pragma parameter amount    "Flow: mix"          1.0  0.0 1.0  0.01
#pragma parameter scale_fd  "Flow: grain scale"  3.0  1.0 12.0 0.5
#pragma parameter warp_fd   "Flow: warp"         6.0  0.0 30.0 0.5
#pragma parameter speed_fd  "Flow: speed"        0.8  0.0 4.0  0.05
#pragma parameter levels_fd "Flow: levels"       2.0  2.0 6.0  1.0
#pragma parameter dark_r    "Dark R"             0.05 0.0 1.0  0.01
#pragma parameter dark_g    "Dark G"             0.06 0.0 1.0  0.01
#pragma parameter dark_b    "Dark B"             0.16 0.0 1.0  0.01
#pragma parameter light_r   "Light R"            0.98 0.0 1.5  0.01
#pragma parameter light_g   "Light G"            0.88 0.0 1.5  0.01
#pragma parameter light_b   "Light B"            0.70 0.0 1.5  0.01

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

float ign(vec2 p)
{
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

void main()
{
    float t = params.Time * params.speed_fd;
    vec2 p = vTexCoord * params.OutputSize.xy / max(params.scale_fd, 1.0);

    // Two octaves of drifting sine currents warp the threshold domain.
    vec2 w = vec2(sin(p.y * 0.11 + t * 0.9) + 0.5 * sin(p.y * 0.031 - t * 0.4),
                  cos(p.x * 0.09 - t * 0.7) + 0.5 * cos(p.x * 0.027 + t * 0.5));
    float thr = ign(floor(p + w * params.warp_fd));

    vec3 s = texture(Source, vTexCoord).rgb;
    float l = dot(s, vec3(0.299, 0.587, 0.114));

    float n = max(params.levels_fd, 2.0) - 1.0;
    float q = clamp(floor(l * n + thr) / n, 0.0, 1.0);

    vec3 col = mix(vec3(params.dark_r, params.dark_g, params.dark_b),
                   vec3(params.light_r, params.light_g, params.light_b), q);

    FragColor = vec4(mix(s, col, params.amount), 1.0);
}
