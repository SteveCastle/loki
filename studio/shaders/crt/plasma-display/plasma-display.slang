#version 450

// plasma-display — orange gas-plasma panel (GRiD Compass / Toshiba T3200),
// single pass.
//
// Coarse square cells in neon orange on near-black glass: few brightness
// levels (early plasma had almost no grayscale), visible cell gaps, a warm
// gas glow around lit cells and a faint 60 Hz shimmer.
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
    float cell_px;
    float levels_p;
    float gap_p;
    float glow_p;
    float flicker_p;
    float tint_r;
    float tint_g;
    float tint_b;
} params;

#pragma parameter amount    "Plasma: mix"        1.0  0.0 1.0  0.01
#pragma parameter cell_px   "Plasma: cell (px)"  5.0  2.0 16.0 1.0
#pragma parameter levels_p  "Plasma: levels"     6.0  2.0 16.0 1.0
#pragma parameter gap_p     "Plasma: cell gap"   0.30 0.0 0.8  0.01
#pragma parameter glow_p    "Plasma: gas glow"   0.55 0.0 1.5  0.05
#pragma parameter flicker_p "Plasma: shimmer"    0.05 0.0 0.3  0.01
#pragma parameter tint_r    "Plasma R"           1.05 0.0 1.5  0.01
#pragma parameter tint_g    "Plasma G"           0.42 0.0 1.5  0.01
#pragma parameter tint_b    "Plasma B"           0.05 0.0 1.5  0.01

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

float cell_luma(vec2 cell, vec2 cells)
{
    vec3 s = texture(Source, (cell + 0.5) / cells).rgb;
    float l = dot(s, vec3(0.299, 0.587, 0.114));
    // Quantize to the panel's few brightness levels.
    float n = max(params.levels_p, 2.0);
    return floor(l * n + 0.5) / n;
}

void main()
{
    vec2 cells = params.OutputSize.xy / max(params.cell_px, 1.0);
    vec2 cell = floor(vTexCoord * cells);
    float l = cell_luma(cell, cells);

    // Cell body with gap gutters.
    vec2 f = fract(vTexCoord * cells);
    float edge = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
    float body = smoothstep(0.0, params.gap_p * 0.5 + 0.02, edge);

    // Neighbor gas glow bleeds across gaps.
    float glow = cell_luma(cell + vec2(1.0, 0.0), cells) + cell_luma(cell - vec2(1.0, 0.0), cells)
               + cell_luma(cell + vec2(0.0, 1.0), cells) + cell_luma(cell - vec2(0.0, 1.0), cells);
    float lit = l * body + (l + glow * 0.25) * params.glow_p * 0.35;

    // 60 Hz-ish shimmer, offset per row so it rolls.
    lit *= 1.0 - params.flicker_p * (0.5 + 0.5 * sin(params.Time * 120.0 + cell.y * 0.7));

    vec3 col = vec3(params.tint_r, params.tint_g, params.tint_b) * lit
             + vec3(0.02, 0.008, 0.0);   // unlit glass is faintly warm

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
