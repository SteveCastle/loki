#version 450

// pc98 — the NEC PC-9801 pixel-art look. Models the real hardware
// constraints of the platform's classic graphics mode:
//
//   · 640×400 pixel grid (the standard PC-98 hi-res mode)
//   · 16 on-screen colors chosen from a 4096-color master palette
//     (12-bit RGB — 4 bits per channel)
//   · the ordered tile dithering artists hand-placed to fake gradients
//     with so few colors (2×2 checkerboards and finer Bayer tones, which
//     the CRT's dot pitch then blended together)
//
// `levels` stands in for the 16-color scarcity: it sets how few steps
// each color ramp gets before dithering has to carry the gradient — the
// lower it is, the more of the image dissolves into the classic
// checkerboard tones. The final output is always snapped to the 4-bit
// master palette. amount = 0 passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float res_x;
    float res_y;
    float levels;
    float dither;
    float pattern;
} params;

#pragma parameter amount  "PC-98: mix" 1.0 0.0 1.0 0.01
#pragma parameter res_x   "PC-98: grid width (px)" 640.0 64.0 1280.0 16.0
#pragma parameter res_y   "PC-98: grid height (px)" 400.0 50.0 800.0 10.0
#pragma parameter levels  "PC-98: ramp steps" 6.0 2.0 16.0 1.0
#pragma parameter dither  "PC-98: dither strength" 1.0 0.0 2.0 0.05
#pragma parameter pattern "PC-98: dither tile (2/4/8 px)" 1.0 0.0 2.0 1.0

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

// Recursive Bayer thresholds — bayer2 is exactly the 2×2 checkerboard
// tile the PC-98 art tools stamped; 4/8 are its finer "tone" variants.
float bayer2(vec2 a)
{
    a = floor(mod(a, 2.0));
    return fract(a.x * 0.5 + a.y * a.y * 0.75);
}
float bayer4(vec2 a) { return bayer2(0.5 * a) * 0.25 + bayer2(a); }
float bayer8(vec2 a) { return bayer4(0.5 * a) * 0.25 + bayer4(a); }

void main()
{
    vec3 base = texture(Source, vTexCoord).rgb;

    // Snap to the virtual PC-98 pixel grid and sample each cell once.
    vec2 grid = vec2(params.res_x, params.res_y);
    vec2 cell = floor(vTexCoord * grid);
    vec3 c = texture(Source, (cell + 0.5) / grid).rgb;

    // Zero-mean ordered-dither offset, one threshold per grid cell so the
    // tile pattern lives on PC-98 pixels regardless of output size.
    float t;
    if      (params.pattern < 0.5) t = bayer2(cell);
    else if (params.pattern < 1.5) t = bayer4(cell);
    else                           t = bayer8(cell);

    // Few ramp steps per channel (the 16-on-screen-color scarcity); the
    // dither offset decides which side of a step each pixel falls on —
    // that boundary band is where the checkerboards appear.
    float steps = max(params.levels - 1.0, 1.0);
    c += (t - 0.5) * params.dither / steps;
    c = floor(clamp(c, 0.0, 1.0) * steps + 0.5) / steps;

    // Master palette: every displayable color was 4 bits per channel.
    c = floor(c * 15.0 + 0.5) / 15.0;

    FragColor = vec4(mix(base, c, params.amount), 1.0);
}
