#version 450

// tilt-shift pass 1 — horizontal Gaussian whose sigma grows with distance from
// the focus band (0 inside it, radius * 1.7 px — the old fixed kernel's
// sigma — once past the falloff). A real progressive blur rather than a
// crossfade between sharp and one fixed blur, so the transition zone
// actually softens gradually. Taps read the mip level matched to sigma, as
// in gaussian-blur, so large radii stay smooth.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float radius;
    float focus;
    float band;
    float softness;
} params;

#pragma parameter radius "TiltShift: blur radius" 4.0 0.5 40.0 0.25

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;   // float, mipmapped

void main()
{
    vec2 uv = vTexCoord;
    float m = smoothstep(params.band, params.band + params.softness, abs(uv.y - params.focus));
    float sigma = max(params.radius * 1.7 * m, 0.05);
    float texel = max(sigma / 6.0, 1.0);
    float lod = log2(texel);
    float k = -0.5 / (sigma * sigma);

    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    for (int i = -21; i <= 21; ++i) {
        float x = float(i) * texel;
        float w = exp(x * x * k);
        sum += textureLod(Source, uv + vec2(x * params.SourceSize.z, 0.0), lod) * w;
        wsum += w;
    }
    FragColor = sum / wsum;
}
