#version 450

// gaussian-blur pass 1 — vertical Gaussian over the (float-framebuffer)
// h-blurred frame, mixed against the clean Original by `amount`. A hair of
// temporal dither is folded into the blurred result before it quantizes to
// 8-bit output — that is what keeps smooth blurred gradients from banding.
// amount = 0 is a clean (byte-exact) passthrough: the dither rides on the
// blurred color, inside the mix.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float radius;
} params;

#pragma parameter amount "Blur: mix" 1.0 0.0 1.0 0.01

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;     // h-blurred (float)
layout(set = 0, binding = 3) uniform sampler2D Original;   // clean input

// 3D hash: the frame index is a separate dimension so the dither pattern is
// fresh every frame (temporal dither averages away on video).
float hash3(vec3 p)
{
    p = fract(p * vec3(123.34, 345.45, 678.56));
    p += dot(p, p.yzx + 34.345);
    return fract((p.x + p.y) * p.z);
}

void main()
{
    vec2 uv = vTexCoord;
    float sigma = max(params.radius, 0.05);
    float spacing = max(sigma * 0.25, 1.0);
    vec2 step_uv = vec2(0.0, spacing * params.SourceSize.w);

    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    for (int i = -16; i <= 16; ++i) {
        float y = float(i) * spacing;
        float w = exp(-0.5 * (y * y) / (sigma * sigma));
        sum += texture(Source, uv + float(i) * step_uv) * w;
        wsum += w;
    }
    vec3 blurred = sum.rgb / wsum;

    // Triangular (TPDF) dither, +-1 LSB of 8-bit, refreshed per frame:
    // rectangular +-0.5 LSB only perturbs pixels near a quantization
    // boundary and leaves plateau centres banded; triangular spans the full
    // level so every pixel decorrelates.
    vec2 px = uv * params.OutputSize.xy;
    float fr = float(params.FrameCount % 1024u);
    float d = hash3(vec3(px, fr)) + hash3(vec3(px + 917.0, fr)) - 1.0;
    blurred += d * (1.0 / 255.0);

    vec3 base = texture(Original, uv).rgb;
    FragColor = vec4(mix(base, clamp(blurred, 0.0, 1.0), params.amount), 1.0);
}
