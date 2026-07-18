#version 450

// bayer-dither — classic ordered dithering, single pass.
//
// The reference dither: a Bayer threshold matrix (2×2, 4×4 or 8×8,
// selectable) quantizes each channel to a few levels, with chunky-pixel
// scaling and an optional mono mode. The foundation the rest of the
// dither family riffs on.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float chunk_bd;
    float matrix_bd;
    float levels_bd;
    float mono_bd;
    float contrast_bd;
} params;

#pragma parameter amount      "Bayer: mix"           1.0 0.0 1.0 0.01
#pragma parameter chunk_bd    "Bayer: pixel size"    2.0 1.0 12.0 1.0
#pragma parameter matrix_bd   "Bayer: matrix (2/4/8)" 3.0 1.0 3.0 1.0
#pragma parameter levels_bd   "Bayer: levels"        2.0 2.0 8.0 1.0
#pragma parameter mono_bd     "Bayer: monochrome"    0.0 0.0 1.0 1.0
#pragma parameter contrast_bd "Bayer: contrast"      1.1 0.5 2.0 0.05

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

// Base 2×2 Bayer cell value for q in {0,1}²: 0,2 / 3,1.
float b2i(vec2 q) { return q.x * 2.0 + q.y * 3.0 - 4.0 * q.x * q.y; }

// Normalized Bayer threshold for 2^K × 2^K (K = 1..3).
float bayerT(vec2 ip, int K)
{
    float v = 0.0;
    float w = pow(4.0, float(K - 1));
    for (int k = 0; k < 3; k++) {
        if (k >= K) break;
        v += b2i(mod(floor(ip / pow(2.0, float(k))), 2.0)) * w;
        w /= 4.0;
    }
    return (v + 0.5) / pow(4.0, float(K));
}

void main()
{
    float chunk = max(params.chunk_bd, 1.0);
    vec2 px = floor(vTexCoord * params.OutputSize.xy / chunk);
    vec3 c = texture(Source, (px + 0.5) * chunk * params.OutputSize.zw).rgb;
    c = clamp((c - 0.5) * params.contrast_bd + 0.5, 0.0, 1.0);
    if (params.mono_bd > 0.5) c = vec3(dot(c, vec3(0.299, 0.587, 0.114)));

    float t = bayerT(px, int(clamp(params.matrix_bd, 1.0, 3.0)));
    float n = max(params.levels_bd, 2.0) - 1.0;
    vec3 d = floor(c * n + t) / n;

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, clamp(d, 0.0, 1.0), params.amount), 1.0);
}
