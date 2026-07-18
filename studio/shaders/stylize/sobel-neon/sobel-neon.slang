#version 450

// sobel-neon — Sobel edge detection rendered as glowing neon lines whose hue
// follows the edge's gradient direction. The flat interior is darkened (and
// optionally desaturated) so the traced edges pop like a light drawing. Single
// pass, purely spatial. amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float scale;
    float threshold;
    float glow;
    float hue_shift;
    float bg;
} params;

#pragma parameter amount    "SobelNeon: mix"             1.0  0.0  1.0  0.01
#pragma parameter scale     "SobelNeon: edge gain"       2.0  0.2  8.0  0.1
#pragma parameter threshold "SobelNeon: edge threshold"  0.10 0.0  1.0  0.01
#pragma parameter glow      "SobelNeon: glow"            1.4  0.0  3.0  0.05
#pragma parameter hue_shift "SobelNeon: hue offset"      0.0  0.0  1.0  0.01
#pragma parameter bg        "SobelNeon: background"      0.18 0.0  1.0  0.01

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

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float L(vec2 uv) { return luma(texture(Source, uv).rgb); }

void main()
{
    vec2 uv = vTexCoord;
    vec2 px = params.SourceSize.zw;

    // Sobel kernels over the 3x3 luma neighbourhood.
    float tl = L(uv + px * vec2(-1, -1)), tc = L(uv + px * vec2(0, -1)), tr = L(uv + px * vec2(1, -1));
    float ml = L(uv + px * vec2(-1,  0)),                                mr = L(uv + px * vec2(1,  0));
    float bl = L(uv + px * vec2(-1,  1)), bc = L(uv + px * vec2(0,  1)), br = L(uv + px * vec2(1,  1));

    float gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    float gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

    float mag = clamp(length(vec2(gx, gy)) * params.scale, 0.0, 1.0);
    float edge = smoothstep(params.threshold, params.threshold + 0.25, mag);

    // Hue from gradient direction; neon edge colour.
    float hue = fract(atan(gy, gx) / 6.28318 + 0.5 + params.hue_shift);
    vec3 neon = hsv2rgb(vec3(hue, 0.9, 1.0)) * params.glow;

    vec3 src = texture(Source, uv).rgb;
    vec3 base = src * params.bg;                  // darkened interior
    vec3 outc = base + neon * edge;               // additive glow on edges

    outc = mix(src, clamp(outc, 0.0, 1.0), params.amount);
    FragColor = vec4(outc, 1.0);
}
