#version 450

// led-wall — stadium / storefront RGB LED billboard, single pass.
//
// Rounded-square RGB modules with dark seams, per-module color banding,
// bloom bleeding across the gaps and a slow per-module brightness shimmer
// — the look of pointing a camera at a big LED wall.
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
    float cell_w;
    float gap_w;
    float round_w;
    float bloom_w;
    float banding_w;
    float shimmer_w;
} params;

#pragma parameter amount    "LED wall: mix"         1.0  0.0  1.0  0.01
#pragma parameter cell_w    "LED wall: module (px)" 7.0  3.0  24.0 1.0
#pragma parameter gap_w     "LED wall: seam"        0.22 0.0  0.6  0.01
#pragma parameter round_w   "LED wall: roundness"   0.35 0.05 1.0  0.01
#pragma parameter bloom_w   "LED wall: bloom"       0.45 0.0  1.5  0.05
#pragma parameter banding_w "LED wall: banding"     6.0  2.0  16.0 1.0
#pragma parameter shimmer_w "LED wall: shimmer"     0.08 0.0  0.4  0.01

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

float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec3 module_color(vec2 cell, vec2 cells)
{
    vec3 c = texture(Source, (cell + 0.5) / cells).rgb;
    float n = max(params.banding_w, 2.0);
    return floor(c * n + 0.5) / n;   // driver banding
}

void main()
{
    vec2 cells = params.OutputSize.xy / max(params.cell_w, 1.0);
    vec2 cell = floor(vTexCoord * cells);
    vec3 col = module_color(cell, cells);

    // Rounded-square module footprint with dark seams.
    vec2 f = abs(fract(vTexCoord * cells) - 0.5) * 2.0;      // 0 centre → 1 edge
    float sq = pow(pow(f.x, 2.0 / params.round_w) + pow(f.y, 2.0 / params.round_w),
                   params.round_w * 0.5);
    float body = 1.0 - smoothstep(1.0 - params.gap_w - 0.1, 1.0 - params.gap_w * 0.4, sq);

    // Bloom: neighbors bleed light over the seams.
    vec3 bleed = module_color(cell + vec2(1.0, 0.0), cells) + module_color(cell - vec2(1.0, 0.0), cells)
               + module_color(cell + vec2(0.0, 1.0), cells) + module_color(cell - vec2(0.0, 1.0), cells);
    vec3 lit = col * body + (col * 0.6 + bleed * 0.1) * params.bloom_w;

    // Per-module refresh shimmer, slow and slightly random.
    float sh = hash(cell) * 6.2831853;
    lit *= 1.0 - params.shimmer_w * 0.5 * (1.0 + sin(params.Time * 3.0 + sh));

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, lit, params.amount), 1.0);
}
