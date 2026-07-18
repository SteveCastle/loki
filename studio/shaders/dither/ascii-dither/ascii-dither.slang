#version 450

// ascii-dither — terminal glyph dithering, single pass.
//
// Each cell becomes a character from a density ramp ( .:*o&8@#), drawn
// procedurally from 5×5 bitmaps packed into floats — tone carried by glyph
// coverage, terminal-green-on-black by default with tintable ink and an
// optional colored mode that inherits the cell's own hue.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float cell_ac;
    float contrast_ac;
    float colored_ac;
    float bg_ac;
    float fg_r;
    float fg_g;
    float fg_b;
} params;

#pragma parameter amount      "ASCII: mix"        1.0  0.0 1.0  0.01
#pragma parameter cell_ac     "ASCII: cell (px)"  10.0 6.0 32.0 1.0
#pragma parameter contrast_ac "ASCII: contrast"   1.1  0.5 2.0  0.05
#pragma parameter colored_ac  "ASCII: use colors" 0.0  0.0 1.0  1.0
#pragma parameter bg_ac       "ASCII: background" 0.06 0.0 0.4  0.01
#pragma parameter fg_r        "Glyph R"           0.35 0.0 1.5  0.01
#pragma parameter fg_g        "Glyph G"           1.05 0.0 1.5  0.01
#pragma parameter fg_b        "Glyph B"           0.45 0.0 1.5  0.01

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

// 5×5 glyph bitmap packed into a float (bit i = pixel x + 5*y, y up).
// p in [0,1]² inside the cell.
float glyph(float n, vec2 p)
{
    vec2 q = floor(p * 5.0);
    if (q.x < 0.0 || q.x > 4.0 || q.y < 0.0 || q.y > 4.0) return 0.0;
    return mod(floor(n / exp2(q.x + 5.0 * (4.0 - q.y))), 2.0);
}

void main()
{
    float cell = max(params.cell_ac, 4.0);
    vec2 gp = vTexCoord * params.OutputSize.xy / cell;
    vec2 id = floor(gp);
    vec3 s = texture(Source, (id + 0.5) * cell * params.OutputSize.zw).rgb;
    float l = dot(s, vec3(0.299, 0.587, 0.114));
    l = clamp((l - 0.5) * params.contrast_ac + 0.5, 0.0, 1.0);

    // Density ramp: (space) . : * o & 8 @# — brighter cell, denser glyph.
    float n =
        l < 0.125 ? 0.0 :
        l < 0.25  ? 4096.0 :        // .
        l < 0.375 ? 131208.0 :      // :
        l < 0.5   ? 332772.0 :      // *
        l < 0.625 ? 15255086.0 :    // o
        l < 0.75  ? 23385164.0 :    // &
        l < 0.875 ? 15252014.0 :    // 8
                    11512810.0;     // @
    // Glyph area with a small margin inside the cell.
    vec2 p = (fract(gp) - 0.1) / 0.8;
    float on = glyph(n, p);

    vec3 fg = params.colored_ac > 0.5
        ? s * 1.4 + 0.08
        : vec3(params.fg_r, params.fg_g, params.fg_b) * (0.35 + 0.75 * l);
    vec3 col = mix(vec3(params.bg_ac) * vec3(0.9, 1.0, 0.95), fg, on);

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
