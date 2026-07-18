#version 450

// mono — black & white via a channel mixer (weights are renormalised,
// so only their ratio matters — crank red for infrared-style skies),
// with an optional colour tint (sepia at the defaults).
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float red_w;
    float green_w;
    float blue_w;
    float tint_hue;
    float tint_amt;
} params;

#pragma parameter amount       "Mono: mix" 1.0 0.0 1.0 0.01
#pragma parameter red_w        "Red weight" 0.30 0.0 2.0 0.02
#pragma parameter green_w      "Green weight" 0.59 0.0 2.0 0.02
#pragma parameter blue_w       "Blue weight" 0.11 0.0 2.0 0.02
#pragma parameter tint_hue     "Tint hue" 0.10 0.0 1.0 0.01
#pragma parameter tint_amt     "Tint strength" 0.0 0.0 1.0 0.01

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

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main()
{
    vec3 base = texture(Source, vTexCoord).rgb;
    vec3 w = vec3(params.red_w, params.green_w, params.blue_w);
    w /= max(w.r + w.g + w.b, 1e-4);
    float l = dot(base, w);
    vec3 c = vec3(l);
    c *= mix(vec3(1.0), hsv2rgb(vec3(params.tint_hue, 0.45, 1.0)),
             params.tint_amt);
    FragColor = vec4(mix(base, clamp(c, 0.0, 1.0), params.amount), 1.0);
}
