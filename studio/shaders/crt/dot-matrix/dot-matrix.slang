#version 450

// dot-matrix — LED marquee / flip-dot sign, single pass.
//
// The frame becomes a board of round LEDs: each cell lights by its luma,
// unlit dots stay as faint dark husks, and everything sits on a matte
// black board. Amber by default — dial the LED color for red bodega-sign
// or green station-board moods.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float dot_px;
    float dot_size_dm;
    float softness_dm;
    float unlit_dm;
    float levels_dm;
    float led_r;
    float led_g;
    float led_b;
} params;

#pragma parameter amount      "Dots: mix"        1.0  0.0  1.0  0.01
#pragma parameter dot_px      "Dots: pitch (px)" 8.0  3.0  32.0 1.0
#pragma parameter dot_size_dm "Dots: dot size"   0.42 0.15 0.5  0.01
#pragma parameter softness_dm "Dots: softness"   0.15 0.01 0.5  0.01
#pragma parameter unlit_dm    "Dots: unlit dot"  0.10 0.0  0.4  0.01
#pragma parameter levels_dm   "Dots: levels"     8.0  2.0  16.0 1.0
#pragma parameter led_r       "LED R"            1.05 0.0  1.5  0.01
#pragma parameter led_g       "LED G"            0.55 0.0  1.5  0.01
#pragma parameter led_b       "LED B"            0.08 0.0  1.5  0.01

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
    // Correct the cell grid for output aspect so dots stay round.
    vec2 cells = params.OutputSize.xy / max(params.dot_px, 1.0);
    vec2 cell = floor(vTexCoord * cells);
    vec2 cuv = (cell + 0.5) / cells;

    float l = dot(texture(Source, cuv).rgb, vec3(0.299, 0.587, 0.114));
    float n = max(params.levels_dm, 2.0);
    l = floor(l * n + 0.5) / n;

    // Round LED footprint inside the cell.
    vec2 f = fract(vTexCoord * cells) - 0.5;
    float d = length(f);
    float dot_mask = 1.0 - smoothstep(params.dot_size_dm - params.softness_dm,
                                      params.dot_size_dm, d);

    vec3 led = vec3(params.led_r, params.led_g, params.led_b);
    // Lit LED glows past its footprint a little; unlit dots stay as husks.
    float lit = l * (dot_mask + (1.0 - smoothstep(params.dot_size_dm, 0.75, d)) * 0.25 * l);
    vec3 col = led * lit + led * params.unlit_dm * dot_mask * (1.0 - l);

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
