#version 450

// chroma-shift — radial chromatic aberration with a soft vignette. The R/G/B
// channels are sampled at increasing distances from frame centre, so the image
// stays crisp in the middle and fringes into colour toward the edges, like a
// cheap lens wide open. A little saturation lift makes the fringes sing. Single
// pass. amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float aberration;
    float falloff;
    float vignette;
    float sat;
} params;

#pragma parameter amount     "ChromaShift: mix"            1.0  0.0  1.0  0.01
#pragma parameter aberration "ChromaShift: aberration"     0.40 0.0  2.0  0.02
#pragma parameter falloff    "ChromaShift: edge falloff"   2.2  1.0  5.0  0.1
#pragma parameter vignette   "ChromaShift: vignette"       0.55 0.0  1.5  0.05
#pragma parameter sat        "ChromaShift: saturation"     1.3  0.0  3.0  0.05

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

void main()
{
    vec2 uv = vTexCoord;
    vec2 c = uv - 0.5;
    float r = length(c) * 1.41421356;            // 0 centre -> ~1 corner
    vec2 dir = c * pow(r, params.falloff);       // stronger toward edges

    float a = params.aberration * 0.03;
    vec3 col;
    col.r = texture(Source, uv - dir * a).r;
    col.g = texture(Source, uv).g;
    col.b = texture(Source, uv + dir * a).b;

    // Saturation lift around luma.
    float l = luma(col);
    col = clamp(mix(vec3(l), col, params.sat), 0.0, 1.0);

    // Soft vignette.
    float vig = 1.0 - params.vignette * r * r;
    col *= clamp(vig, 0.0, 1.0);

    vec3 src = texture(Source, uv).rgb;
    FragColor = vec4(mix(src, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
