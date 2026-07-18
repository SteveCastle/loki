#version 450

// posterize-pop — comic / cel-shaded look: quantise colour into a few bands,
// boost saturation so it reads as flat poster ink, and stamp a black Sobel
// outline around shapes. Single pass. amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float levels;
    float sat;
    float outline;
    float thickness;
} params;

#pragma parameter amount    "PosterizePop: mix"          1.0  0.0  1.0   0.01
#pragma parameter levels    "PosterizePop: colour bands" 5.0  2.0  16.0  1.0
#pragma parameter sat       "PosterizePop: saturation"   1.5  0.0  3.0   0.05
#pragma parameter outline   "PosterizePop: outline"      1.0  0.0  2.0   0.05
#pragma parameter thickness "PosterizePop: line weight"  1.0  0.5  3.0   0.1

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

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float L(vec2 uv) { return luma(texture(Source, uv).rgb); }

void main()
{
    vec2 uv = vTexCoord;
    vec2 px = params.SourceSize.zw * params.thickness;

    vec3 src = texture(Source, uv).rgb;

    // Posterise.
    float n = max(2.0, floor(params.levels));
    vec3 q = floor(src * n + 0.5) / n;

    // Saturation lift around luma.
    float l = luma(q);
    q = clamp(mix(vec3(l), q, params.sat), 0.0, 1.0);

    // Sobel outline on luma.
    float tl = L(uv + px * vec2(-1, -1)), tc = L(uv + px * vec2(0, -1)), tr = L(uv + px * vec2(1, -1));
    float ml = L(uv + px * vec2(-1,  0)),                                mr = L(uv + px * vec2(1,  0));
    float bl = L(uv + px * vec2(-1,  1)), bc = L(uv + px * vec2(0,  1)), br = L(uv + px * vec2(1,  1));
    float gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    float gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
    float e = smoothstep(0.4, 1.2, length(vec2(gx, gy)) * params.outline);

    vec3 col = mix(q, vec3(0.0), e);             // ink the edges black

    FragColor = vec4(mix(src, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
