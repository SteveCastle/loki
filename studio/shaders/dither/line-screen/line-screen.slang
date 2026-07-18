#version 450

// line-screen — line halftone dither, single pass.
//
// Gravure / engraving screen: ink lines at a fixed angle whose thickness
// carries the tone. A touch of line wobble keeps it hand-printed instead
// of vector-perfect.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float period_ls;
    float angle_ls;
    float gamma_ls;
    float wobble_ls;
    float ink2_r;
    float ink2_g;
    float ink2_b;
    float paper2_r;
    float paper2_g;
    float paper2_b;
} params;

#pragma parameter amount    "Line screen: mix"          1.0  0.0   1.0  0.01
#pragma parameter period_ls "Line screen: spacing (px)" 5.0  2.0   24.0 0.5
#pragma parameter angle_ls  "Line screen: angle (deg)"  22.5 -90.0 90.0 2.5
#pragma parameter gamma_ls  "Line screen: tone gamma"   1.0  0.4   2.5  0.05
#pragma parameter wobble_ls "Line screen: wobble"       0.3  0.0   2.0  0.05
#pragma parameter ink2_r    "Ink R"                     0.10 0.0   1.0  0.01
#pragma parameter ink2_g    "Ink G"                     0.09 0.0   1.0  0.01
#pragma parameter ink2_b    "Ink B"                     0.16 0.0   1.0  0.01
#pragma parameter paper2_r  "Paper R"                   0.96 0.0   1.0  0.01
#pragma parameter paper2_g  "Paper G"                   0.93 0.0   1.0  0.01
#pragma parameter paper2_b  "Paper B"                   0.86 0.0   1.0  0.01

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
    vec2 p = vTexCoord * params.OutputSize.xy;
    float a = radians(params.angle_ls);
    float across = -p.x * sin(a) + p.y * cos(a);   // distance across the lines
    float along  =  p.x * cos(a) + p.y * sin(a);

    // Hand-printed wobble: the line path drifts slowly along its length.
    across += sin(along * 0.05) * params.wobble_ls
            + sin(along * 0.013 + 1.7) * params.wobble_ls * 2.0;

    vec3 s = texture(Source, vTexCoord).rgb;
    float l = pow(clamp(dot(s, vec3(0.299, 0.587, 0.114)), 0.0, 1.0), params.gamma_ls);

    // Triangle profile across the line period; ink where tone is darker
    // than the profile, with a half-pixel soft edge.
    float phase = abs(fract(across / max(params.period_ls, 1.0)) - 0.5) * 2.0;
    float ink = 1.0 - smoothstep(l - 0.08, l + 0.08, phase);

    vec3 col = mix(vec3(params.paper2_r, params.paper2_g, params.paper2_b),
                   vec3(params.ink2_r, params.ink2_g, params.ink2_b), ink);

    FragColor = vec4(mix(s, col, params.amount), 1.0);
}
