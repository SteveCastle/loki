#version 450

// depth-blur pass 2 — per-pixel post-filter, un-premultiply, back out of
// linear light, mix against the clean Original. The post-filter disc
// scales with each pixel's own radius (see lens-blur pass 2), and pixels
// whose circle is under half a pixel pass the Original through untouched,
// so the in-focus region stays bit-exact.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float radius;
    float linlight;
    float map_invert;
    float map_gamma;
} params;

#pragma parameter amount   "Depth blur: mix" 1.0 0.0 1.0 0.01

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;     // gathered, premultiplied (float)
layout(set = 0, binding = 3) uniform sampler2D Original;   // clean input
layout(set = 0, binding = 4) uniform sampler2D Mask;       // blur map

float hash3(vec3 p)
{
    p = fract(p * vec3(123.34, 345.45, 678.56));
    p += dot(p, p.yzx + 34.345);
    return fract((p.x + p.y) * p.z);
}

const int TAPS = 16;

void main()
{
    vec2 uv = vTexCoord;
    vec3 base = textureLod(Original, uv, 0.0).rgb;
    float m = textureLod(Mask, uv, 0.0).r;
    m = mix(m, 1.0 - m, params.map_invert);
    float R = params.radius * pow(clamp(m, 0.0, 1.0), params.map_gamma);
    if (R < 0.5) { FragColor = vec4(base, 1.0); return; }

    float spacing = R * sqrt(3.14159265 / 128.0);
    vec2 shape = spacing * params.SourceSize.zw;
    vec4 acc = vec4(0.0);
    for (int i = 0; i < TAPS; ++i) {
        float r = sqrt((float(i) + 0.5) / float(TAPS));
        float th = float(i) * 2.39996323;
        acc += textureLod(Source, uv + vec2(cos(th), sin(th)) * r * shape, 0.0);
    }
    vec3 blurred = max(acc.rgb / max(acc.a, 1e-6), 0.0);
    blurred = mix(blurred, pow(blurred, vec3(1.0 / 2.2)), params.linlight);

    // TPDF dither (see gaussian-blur): smooth defocus otherwise bands.
    vec2 px = uv * params.OutputSize.xy;
    float fr = float(params.FrameCount % 1024u);
    float d = hash3(vec3(px, fr)) + hash3(vec3(px + 917.0, fr)) - 1.0;
    blurred += d * (1.0 / 255.0);

    // Fade in over the first pixel of radius so the sharp/blurred seam
    // has no step.
    float k = params.amount * smoothstep(0.5, 1.5, R);
    FragColor = vec4(mix(base, clamp(blurred, 0.0, 1.0), k), 1.0);
}
