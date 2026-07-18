#version 450

// blue-noise-dither — interleaved gradient noise dithering, single pass.
//
// IGN is famously blue-noise-like: high-frequency, pattern-free thresholds
// that read as smooth organic grain instead of Bayer's crosshatch weave.
// Optionally re-rolls per frame for living, film-like dither.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float chunk_bn;
    float levels_bn;
    float mono_bn;
    float animate_bn;
    float softness_bn;
} params;

#pragma parameter amount      "Blue noise: mix"        1.0 0.0 1.0 0.01
#pragma parameter chunk_bn    "Blue noise: pixel size" 2.0 1.0 12.0 1.0
#pragma parameter levels_bn   "Blue noise: levels"     3.0 2.0 8.0 1.0
#pragma parameter mono_bn     "Blue noise: monochrome" 0.0 0.0 1.0 1.0
#pragma parameter animate_bn  "Blue noise: animate"    1.0 0.0 1.0 1.0
#pragma parameter softness_bn "Blue noise: softness"   0.0 0.0 1.0 0.05

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

// Interleaved gradient noise (Jimenez); the golden-ratio scroll per frame
// keeps the field temporally decorrelated without directional drift.
float ign(vec2 p)
{
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

void main()
{
    float chunk = max(params.chunk_bn, 1.0);
    vec2 px = floor(vTexCoord * params.OutputSize.xy / chunk);
    vec3 c = texture(Source, (px + 0.5) * chunk * params.OutputSize.zw).rgb;
    if (params.mono_bn > 0.5) c = vec3(dot(c, vec3(0.299, 0.587, 0.114)));

    vec2 shift = params.animate_bn > 0.5
        ? vec2(float(params.FrameCount % 64u)) * vec2(5.588238, 5.588238)
        : vec2(0.0);
    float t = ign(px + shift);

    float n = max(params.levels_bn, 2.0) - 1.0;
    // softness > 0 blends between hard dithered steps and the smooth input.
    vec3 hard = floor(c * n + t) / n;
    vec3 d = mix(hard, c, params.softness_bn * 0.5);

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, clamp(d, 0.0, 1.0), params.amount), 1.0);
}
