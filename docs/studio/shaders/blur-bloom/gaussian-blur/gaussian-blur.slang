#version 450

// gaussian-blur pass 2 — vertical Gaussian over the (float, mipmapped)
// h-blurred frame, same tap scheme as blur_h, then back out of linear light
// and mixed against the clean Original by `amount`. A hair of temporal
// dither is folded into the blurred result before it quantizes to 8-bit
// output — that is what keeps smooth blurred gradients from banding.
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
    float aspect;
    float linlight;
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
layout(set = 0, binding = 2) uniform sampler2D Source;     // h-blurred (float, mipmapped)
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
    float sigma = max(params.radius * clamp(1.0 - params.aspect, 0.0, 1.0), 0.05);
    float texel = max(sigma / 6.0, 1.0);
    float lod = log2(texel);
    float k = -0.5 / (sigma * sigma);

    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    for (int i = -21; i <= 21; ++i) {
        float y = float(i) * texel;
        float w = exp(y * y * k);
        sum += textureLod(Source, uv + vec2(0.0, y * params.SourceSize.w), lod) * w;
        wsum += w;
    }
    vec3 blurred = max(sum.rgb / wsum, 0.0);
    blurred = mix(blurred, pow(blurred, vec3(1.0 / 2.2)), params.linlight);

    // Triangular (TPDF) dither, +-1 LSB of 8-bit, refreshed per frame:
    // rectangular +-0.5 LSB only perturbs pixels near a quantization
    // boundary and leaves plateau centres banded; triangular spans the full
    // level so every pixel decorrelates.
    vec2 px = uv * params.OutputSize.xy;
    float fr = float(params.FrameCount % 1024u);
    float d = hash3(vec3(px, fr)) + hash3(vec3(px + 917.0, fr)) - 1.0;
    blurred += d * (1.0 / 255.0);

    vec3 base = textureLod(Original, uv, 0.0).rgb;
    FragColor = vec4(mix(base, clamp(blurred, 0.0, 1.0), params.amount), 1.0);
}
