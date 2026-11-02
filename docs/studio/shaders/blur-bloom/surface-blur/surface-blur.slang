#version 450

// surface-blur pass 1 — vertical half of the bilateral filter over the
// (float) h-pass, then mixed against the clean Original by `amount` with
// the same TPDF dither as gaussian-blur so smoothed gradients don't band.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float radius;
    float threshold;
} params;

#pragma parameter amount "Surface: mix" 1.0 0.0 1.0 0.01

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;     // h-filtered (float)
layout(set = 0, binding = 3) uniform sampler2D Original;   // clean input

float hash3(vec3 p)
{
    p = fract(p * vec3(123.34, 345.45, 678.56));
    p += dot(p, p.yzx + 34.345);
    return fract((p.x + p.y) * p.z);
}

void main()
{
    vec2 uv = vTexCoord;
    vec3 c0 = textureLod(Source, uv, 0.0).rgb;
    float R = max(params.radius, 0.01);
    float ks = -0.5 / (0.16 * R * R);
    float kr = -1.0 / (2.0 * 3.0 * params.threshold * params.threshold);
    float stepy = R / 24.0 * params.SourceSize.w;

    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int i = -24; i <= 24; ++i) {
        float y = float(i) * R / 24.0;
        vec3 c = textureLod(Source, uv + vec2(0.0, float(i) * stepy), 0.0).rgb;
        vec3 d = c - c0;
        float w = exp(y * y * ks + dot(d, d) * kr);
        sum += c * w;
        wsum += w;
    }
    vec3 smoothed = sum / wsum;

    vec2 px = uv * params.OutputSize.xy;
    float fr = float(params.FrameCount % 1024u);
    float dn = hash3(vec3(px, fr)) + hash3(vec3(px + 917.0, fr)) - 1.0;
    smoothed += dn * (1.0 / 255.0);

    vec3 base = textureLod(Original, uv, 0.0).rgb;
    FragColor = vec4(mix(base, clamp(smoothed, 0.0, 1.0), params.amount), 1.0);
}
