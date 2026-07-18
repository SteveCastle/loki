#version 450

// split-tone — tint shadows one hue and highlights another (teal &
// orange lives here). The casts are chroma-only, so luminance is
// untouched; `balance` slides the shadow/highlight crossover.
// Strengths default to 0 (neutral). amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float shadow_hue;
    float shadow_amt;
    float high_hue;
    float high_amt;
    float balance;
} params;

#pragma parameter amount       "Split tone: mix" 1.0 0.0 1.0 0.01
#pragma parameter shadow_hue   "Shadow hue" 0.6 0.0 1.0 0.01
#pragma parameter shadow_amt   "Shadow strength" 0.0 0.0 1.0 0.01
#pragma parameter high_hue     "Highlight hue" 0.08 0.0 1.0 0.01
#pragma parameter high_amt     "Highlight strength" 0.0 0.0 1.0 0.01
#pragma parameter balance      "Balance (dark-bright)" 0.0 -1.0 1.0 0.02

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

void main()
{
    vec3 base = texture(Source, vTexCoord).rgb;
    float l = luma(base);
    float w = smoothstep(0.0, 1.0, l + 0.25 * params.balance);
    vec3 sc = hsv2rgb(vec3(params.shadow_hue, 1.0, 1.0));
    vec3 hc = hsv2rgb(vec3(params.high_hue,   1.0, 1.0));
    sc -= luma(sc);   // chroma-only casts: luminance stays put
    hc -= luma(hc);
    vec3 c = base + sc * params.shadow_amt * (1.0 - w) * 0.5
                  + hc * params.high_amt   * w         * 0.5;
    FragColor = vec4(mix(base, clamp(c, 0.0, 1.0), params.amount), 1.0);
}
