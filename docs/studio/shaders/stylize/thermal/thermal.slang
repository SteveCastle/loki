#version 450

// thermal — false-colour "thermal camera" look. Luminance is remapped through
// a black -> magenta -> red -> orange -> yellow -> white heat ramp, with an
// optional Sobel outline so warm shapes get a bright rim. Contrast control
// pushes the palette. Single pass. amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float contrast;
    float bias;
    float outline;
} params;

#pragma parameter amount   "Thermal: mix"          1.0  0.0  1.0  0.01
#pragma parameter contrast "Thermal: contrast"     1.3  0.3  3.0  0.05
#pragma parameter bias     "Thermal: warmth bias"  0.0 -0.5  0.5  0.01
#pragma parameter outline  "Thermal: edge outline" 0.6  0.0  2.0  0.05

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

// 5-stop heat ramp.
vec3 heat(float t)
{
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.0, 0.0, 0.0);
    const vec3 c1 = vec3(0.30, 0.0, 0.50);   // magenta
    const vec3 c2 = vec3(0.90, 0.10, 0.10);  // red
    const vec3 c3 = vec3(1.0, 0.55, 0.0);    // orange
    const vec3 c4 = vec3(1.0, 0.95, 0.30);   // yellow
    const vec3 c5 = vec3(1.0, 1.0, 1.0);     // white
    if (t < 0.2) return mix(c0, c1, t / 0.2);
    if (t < 0.4) return mix(c1, c2, (t - 0.2) / 0.2);
    if (t < 0.6) return mix(c2, c3, (t - 0.4) / 0.2);
    if (t < 0.8) return mix(c3, c4, (t - 0.6) / 0.2);
    return mix(c4, c5, (t - 0.8) / 0.2);
}

void main()
{
    vec2 uv = vTexCoord;
    vec2 px = params.SourceSize.zw;

    float l = L(uv);
    l = clamp((l - 0.5) * params.contrast + 0.5 + params.bias, 0.0, 1.0);
    vec3 col = heat(l);

    // Sobel outline brightens warm rims.
    float gx = L(uv + vec2(px.x, 0)) - L(uv - vec2(px.x, 0));
    float gy = L(uv + vec2(0, px.y)) - L(uv - vec2(0, px.y));
    float e = clamp(length(vec2(gx, gy)) * 4.0, 0.0, 1.0);
    col += params.outline * e;

    vec3 src = texture(Source, uv).rgb;
    FragColor = vec4(mix(src, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
