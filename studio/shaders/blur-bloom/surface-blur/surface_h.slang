#version 450

// surface-blur pass 0 — horizontal half of a separable bilateral filter:
// a Gaussian in space whose taps are also weighted by how close their
// colour is to the centre pixel's. Flat areas (skin, sky, walls) smooth
// out while edges whose contrast exceeds `threshold` stay sharp — the
// "surface blur" / skin-smoothing look. 49 taps always span +-radius.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float radius;
    float threshold;
} params;

#pragma parameter radius    "Surface: radius (px)" 8.0 0.0 60.0 0.25
#pragma parameter threshold "Surface: edge threshold" 0.12 0.01 1.0 0.01

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
    vec2 uv = vTexCoord;
    vec3 c0 = textureLod(Source, uv, 0.0).rgb;
    float R = max(params.radius, 0.01);
    float ks = -0.5 / (0.16 * R * R);                     // sigma = 0.4 * radius
    float kr = -1.0 / (2.0 * 3.0 * params.threshold * params.threshold);
    float stepx = R / 24.0 * params.SourceSize.z;

    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int i = -24; i <= 24; ++i) {
        float x = float(i) * R / 24.0;
        vec3 c = textureLod(Source, uv + vec2(float(i) * stepx, 0.0), 0.0).rgb;
        vec3 d = c - c0;
        float w = exp(x * x * ks + dot(d, d) * kr);
        sum += c * w;
        wsum += w;
    }
    FragColor = vec4(sum / wsum, 1.0);
}
