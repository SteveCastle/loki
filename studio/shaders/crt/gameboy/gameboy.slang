#version 450

// gameboy — original DMG-01 pea-soup LCD, single self-feedback pass.
//
// Downsamples to a coarse dot grid, quantizes luma to the four classic
// green-on-green shades (with optional 2×2 ordered dither), draws the LCD
// cell grid, and smears motion with the DMG's famously slow LCD response
// (PassFeedback ghosting).
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float res_px;
    float ghost;
    float grid_gb;
    float dither_gb;
    float contrast_gb;
} params;

#pragma parameter amount      "Game Boy: mix"          1.0   0.0  1.0  0.01
#pragma parameter res_px      "Game Boy: dots across"  160.0 64.0 480.0 8.0
#pragma parameter ghost       "Game Boy: LCD ghosting" 0.55  0.0  0.95 0.01
#pragma parameter grid_gb     "Game Boy: dot grid"     0.25  0.0  1.0  0.01
#pragma parameter dither_gb   "Game Boy: dither"       0.5   0.0  1.0  0.05
#pragma parameter contrast_gb "Game Boy: contrast"     1.1   0.5  2.0  0.05

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
layout(set = 0, binding = 3) uniform sampler2D PassFeedback0;

void main()
{
    // Coarse LCD grid, square dots (vertical count follows the aspect).
    float cells_x = max(params.res_px, 8.0);
    float cells_y = cells_x * params.OutputSize.y * params.OutputSize.z;
    vec2 cells = vec2(cells_x, cells_y);
    vec2 cell = floor(vTexCoord * cells);
    vec2 cuv = (cell + 0.5) / cells;

    // Luma with contrast, then 2×2 Bayer dither into 4 shades.
    vec3 s = texture(Source, cuv).rgb;
    float l = dot(s, vec3(0.299, 0.587, 0.114));
    l = clamp((l - 0.5) * params.contrast_gb + 0.5, 0.0, 1.0);
    vec2 par = mod(cell, 2.0);
    float bayer = (par.x + 2.0 * par.y) / 4.0 - 0.375;   // -0.375..0.375
    l += bayer * params.dither_gb * 0.25;
    float shade = clamp(floor(l * 4.0), 0.0, 3.0);

    // DMG palette, darkest to lightest.
    vec3 pal0 = vec3(0.059, 0.220, 0.059);
    vec3 pal1 = vec3(0.188, 0.384, 0.188);
    vec3 pal2 = vec3(0.545, 0.675, 0.059);
    vec3 pal3 = vec3(0.608, 0.737, 0.059);
    vec3 col = shade < 0.5 ? pal0 : shade < 1.5 ? pal1 : shade < 2.5 ? pal2 : pal3;

    // LCD cell grid: darken the gutters between dots.
    vec2 f = fract(vTexCoord * cells);
    float edge = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
    col *= 1.0 - params.grid_gb * (1.0 - smoothstep(0.0, 0.14, edge));

    // Slow LCD response: drag toward the previous frame.
    vec3 prev = texture(PassFeedback0, vTexCoord).rgb;
    col = mix(col, prev, params.ghost * params.amount);

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
