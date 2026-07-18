#version 450

// mosaic — tiled cells that AVERAGE the pixels they cover (unlike
// pixelate's point sample), with optional grout lines between tiles and a
// per-tile brightness wobble for a hand-set look. amount = 0 passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float cell;
    float grout;
    float grout_dark;
    float variance;
} params;

#pragma parameter amount     "Mosaic: mix" 1.0 0.0 1.0 0.01
#pragma parameter cell       "Mosaic: tile size (px)" 16.0 2.0 128.0 1.0
#pragma parameter grout      "Mosaic: grout width" 0.06 0.0 0.4 0.01
#pragma parameter grout_dark "Mosaic: grout darkness" 0.6 0.0 1.0 0.01
#pragma parameter variance   "Mosaic: tile variance" 0.08 0.0 0.5 0.01

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

float hash2(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 34.45);
    return fract(p.x * p.y);
}

void main()
{
    vec2 uv = vTexCoord;
    vec3 base = texture(Source, uv).rgb;

    vec2 grid = params.OutputSize.xy / max(params.cell, 2.0);
    vec2 id = floor(uv * grid);

    // Average a 4x4 grid of taps inside the tile — a true area average, so
    // detail melts into the tile color instead of aliasing like a point
    // sample would.
    vec3 avg = vec3(0.0);
    for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) {
        vec2 o = (vec2(float(x), float(y)) + 0.5) / 4.0;
        avg += texture(Source, (id + o) / grid).rgb;
    }
    avg /= 16.0;

    // Per-tile brightness wobble.
    avg *= 1.0 + (hash2(id) - 0.5) * 2.0 * params.variance;

    // Grout: darken near tile edges.
    vec2 f = fract(uv * grid);
    vec2 edge = min(f, 1.0 - f);                     // 0 at edges, 0.5 center
    float g = smoothstep(params.grout * 0.5, params.grout * 0.5 + 0.02,
                         min(edge.x, edge.y));
    avg *= mix(1.0 - params.grout_dark, 1.0, g);

    FragColor = vec4(mix(base, avg, params.amount), 1.0);
}
