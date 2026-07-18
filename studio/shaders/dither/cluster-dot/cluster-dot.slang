#version 450

// cluster-dot — newspaper clustered-dot ordered dither, single pass.
//
// Unlike Bayer (which disperses thresholds), a clustered-dot matrix grows
// ink dots outward from cell centers — the classic newsprint photo screen.
// Mono with tintable ink and paper for that Sunday-edition look.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float cell_cd;
    float contrast_cd;
    float ink_r;
    float ink_g;
    float ink_b;
    float paper_r;
    float paper_g;
    float paper_b;
} params;

#pragma parameter amount      "Newsprint: mix"       1.0  0.0 1.0  0.01
#pragma parameter cell_cd     "Newsprint: cell (px)" 2.0  1.0 8.0  1.0
#pragma parameter contrast_cd "Newsprint: contrast"  1.15 0.5 2.0  0.05
#pragma parameter ink_r       "Ink R"                0.12 0.0 1.0  0.01
#pragma parameter ink_g       "Ink G"                0.11 0.0 1.0  0.01
#pragma parameter ink_b       "Ink B"                0.13 0.0 1.0  0.01
#pragma parameter paper_r     "Paper R"              0.94 0.0 1.0  0.01
#pragma parameter paper_g     "Paper G"              0.91 0.0 1.0  0.01
#pragma parameter paper_b     "Paper B"              0.84 0.0 1.0  0.01

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

// 4×4 clustered-dot threshold matrix (dot grows from the cell centre).
float clusterT(vec2 ip)
{
    vec2 q = mod(floor(ip), 4.0);
    int i = int(q.x) + int(q.y) * 4;
    float m =
        i ==  0 ? 12.0 : i ==  1 ?  5.0 : i ==  2 ?  6.0 : i ==  3 ? 13.0 :
        i ==  4 ?  4.0 : i ==  5 ?  0.0 : i ==  6 ?  1.0 : i ==  7 ?  7.0 :
        i ==  8 ? 11.0 : i ==  9 ?  3.0 : i == 10 ?  2.0 : i == 11 ?  8.0 :
        i == 12 ? 15.0 : i == 13 ? 10.0 : i == 14 ?  9.0 : 14.0;
    return (m + 0.5) / 16.0;
}

void main()
{
    float cell = max(params.cell_cd, 1.0);
    vec2 px = floor(vTexCoord * params.OutputSize.xy / cell);
    vec3 s = texture(Source, (px + 0.5) * cell * params.OutputSize.zw).rgb;
    float l = dot(s, vec3(0.299, 0.587, 0.114));
    l = clamp((l - 0.5) * params.contrast_cd + 0.5, 0.0, 1.0);

    float ink = step(l, clusterT(px));   // dark → dot cluster fills in
    vec3 col = mix(vec3(params.paper_r, params.paper_g, params.paper_b),
                   vec3(params.ink_r, params.ink_g, params.ink_b), ink);

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
