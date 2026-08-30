#version 450

// teletext — broadcast teletext / Ceefax mosaic graphics, single pass.
//
// The real system drew "graphics" as character cells of 2×3 mosaic blocks,
// two colors per cell, from a palette of eight (black + RGB primaries and
// their mixes + white). The picture is quantized onto that grid: each cell
// picks a foreground color from the 3-bit palette and lights whichever of
// its six blocks are brighter than the cell's own midpoint. Weak reception
// corrupted single characters — cells flip to the wrong color or pattern —
// and cells with the FLASH attribute blink. Rendered with the double-height
// scanline look of a TV set on a teletext page.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float cols;
    float rows;
    float level;
    float bg_level;
    float reception;
    float flash;
    float scanline;
    float speed;
    float seed;
} params;

#pragma parameter amount    "Teletext: amount"          1.0  0.0  1.0   0.01
#pragma parameter cols      "Teletext: columns"         40.0 10.0 120.0 1.0
#pragma parameter rows      "Teletext: rows"            24.0 6.0  72.0  1.0
#pragma parameter level     "Teletext: color threshold" 0.35 0.05 0.95  0.01
#pragma parameter bg_level  "Teletext: bg threshold"    0.75 0.2  1.0   0.01
#pragma parameter reception "Teletext: bad reception"   0.15 0.0  1.0   0.01
#pragma parameter flash     "Teletext: flashing cells"  0.1  0.0  1.0   0.01
#pragma parameter scanline  "Teletext: scanlines"       0.4  0.0  1.0   0.01
#pragma parameter speed     "Teletext: speed"           1.0  0.0  4.0   0.05
#pragma parameter seed      "Teletext: seed"            17.0 0.0 64.0   1.0

/* Mosaic cells recolour to the palette (often black) — that must not
 * punch holes in the layer, so coverage keeps the clean silhouette. */
#pragma matte_ignore amount

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

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

/* Snap a color to the 3-bit teletext palette: each channel fully on or off. */
vec3 pal3(vec3 c, float thr) { return step(vec3(thr), c); }

void main()
{
    vec2  uv  = vTexCoord;
    float amt = params.amount;
    float t   = float(params.FrameCount) * (1.0 / 60.0) * params.speed;

    vec2 grid = vec2(max(params.cols, 1.0), max(params.rows, 1.0));
    vec2 cell = floor(uv * grid);

    /* Sample the six 2×3 mosaic block centers of this cell. */
    vec3  blocks[6];
    float lmin = 1.0, lmax = 0.0;
    vec3  bright = vec3(0.0), dark = vec3(0.0);
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 2; i++) {
        vec2 sub = (cell + (vec2(float(i), float(j)) + 0.5) / vec2(2.0, 3.0)) / grid;
        vec3 c   = texture(Source, sub).rgb;
        blocks[j * 2 + i] = c;
        float l = luma(c);
        if (l <= lmin) { lmin = l; dark = c; }
        if (l >= lmax) { lmax = l; bright = c; }
    }

    /* Two colors per cell: fg from the brightest block; bg stays black
     * unless even the darkest block is bold enough to earn a color. */
    vec3 fg = pal3(bright, params.level);
    vec3 bg = pal3(dark, params.bg_level) * step(params.bg_level * 0.5, lmin);

    /* Which block is this pixel in? Adaptive midpoint keeps cell contrast. */
    vec2  inCell = fract(uv * grid);
    int   bi     = int(floor(inCell.x * 2.0)) + int(floor(inCell.y * 3.0)) * 2;
    float mid    = (lmin + lmax) * 0.5;
    float on     = step(mid, luma(blocks[bi]));
    /* flat cells (no contrast) render solid fg if bright, bg if dark */
    on = mix(step(params.level, lmax), on, step(0.06, lmax - lmin));

    /* Bad reception: whole characters decode wrong for a moment — the
     * cell swaps to a random palette color and its pattern inverts. */
    float slot = floor(t * 6.0);
    float bad  = step(hash21(cell + slot * 7.0 + params.seed), params.reception * 0.25);
    vec3  badc = pal3(vec3(hash21(cell + slot + 1.0),
                           hash21(cell + slot + 2.0),
                           hash21(cell + slot + 3.0)), 0.5);
    fg = mix(fg, badc, bad);
    on = mix(on, 1.0 - on, bad * step(0.5, hash21(cell + slot + 4.0)));

    /* FLASH attribute: some cells blink at ~1Hz. */
    float fcell = step(hash21(cell * 1.7 + params.seed), params.flash * 0.3);
    on *= 1.0 - fcell * step(0.5, fract(t * 0.75));

    vec3 col = mix(bg, fg, on);

    /* TV presentation: paired scanlines (teletext rows were double height). */
    float rowpx = params.rows * 6.0;
    float sl    = 0.5 + 0.5 * sin(uv.y * rowpx * 3.14159 * 2.0);
    col *= 1.0 - params.scanline * 0.35 * sl;

    vec3 clean = texture(Source, uv).rgb;
    FragColor = vec4(clamp(mix(clean, col, amt), 0.0, 1.0), 1.0);
}
