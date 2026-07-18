#version 450

// chromatic-aberration — lens-style transverse CA, single pass.
//
// Channels are resampled at slightly different radial magnifications and
// smeared across several sub-taps, so the fringing is soft and spectral
// the way real glass fails — not a hard single-tap offset. Strength grows
// toward the frame edge with a tunable falloff.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float strength_ca;
    float falloff_ca;
    float soft_ca;
} params;

#pragma parameter amount      "CA: mix"           1.0  0.0 1.0 0.01
#pragma parameter strength_ca "CA: strength"      1.2  0.0 4.0 0.05
#pragma parameter falloff_ca  "CA: edge falloff"  2.0  0.5 4.0 0.1
#pragma parameter soft_ca     "CA: fringe soften" 0.6  0.0 1.0 0.05

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
    vec2 v = vTexCoord - 0.5;
    float r2 = dot(v, v) * 4.0;                       // 0 centre → ~1 corner-ish
    float mag = params.strength_ca * 0.01 * pow(r2, params.falloff_ca * 0.5);

    // Spectral accumulation: 6 taps sweep from red-side to blue-side
    // magnification with triangular RGB weights.
    vec3 acc = vec3(0.0);
    vec3 wsum = vec3(1e-4);
    for (int i = 0; i < 6; i++) {
        float s = (float(i) / 5.0) * 2.0 - 1.0;       // -1 .. 1
        float spread = mix(1.0, abs(s), 1.0 - params.soft_ca);
        vec2 uv = 0.5 + v * (1.0 + mag * s);
        vec3 tap = texture(Source, uv).rgb;
        vec3 w = vec3(max(0.0, 1.0 - abs(s + 0.8) * 1.4),
                      max(0.0, 1.0 - abs(s) * 1.4),
                      max(0.0, 1.0 - abs(s - 0.8) * 1.4)) * spread;
        acc += tap * w;
        wsum += w;
    }
    vec3 col = acc / wsum;

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
