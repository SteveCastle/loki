#version 450

// duotone — maps luminance onto a gradient between two tunable colours (set by
// hue), the way a duotone print does: shadows take the first colour, highlights
// the second, midtones blend. Adds optional film grain. Great for a graphic,
// poster-like recolour. Single pass. amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float shadow_hue;
    float light_hue;
    float contrast;
    float grain;
} params;

#pragma parameter amount     "Duotone: mix"           1.0  0.0  1.0  0.01
#pragma parameter shadow_hue "Duotone: shadow hue"    0.62 0.0  1.0  0.01
#pragma parameter light_hue  "Duotone: highlight hue" 0.08 0.0  1.0  0.01
#pragma parameter contrast   "Duotone: contrast"      1.2  0.3  3.0  0.05
#pragma parameter grain      "Duotone: film grain"    0.06 0.0  0.4  0.01

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

// 3D hash: the frame index is a separate dimension, NOT added to the pixel
// coords — hash(px + frame) translates the whole field diagonally each
// frame and the grain reads as directional drift instead of static.
float hash3(vec3 p)
{
    p = fract(p * vec3(123.34, 345.45, 678.56));
    p += dot(p, p.yzx + 34.345);
    return fract((p.x + p.y) * p.z);
}

void main()
{
    vec2 uv = vTexCoord;
    float l = luma(texture(Source, uv).rgb);
    l = clamp((l - 0.5) * params.contrast + 0.5, 0.0, 1.0);

    vec3 shadow = hsv2rgb(vec3(params.shadow_hue, 0.85, 0.35));
    vec3 light  = hsv2rgb(vec3(params.light_hue, 0.75, 1.0));
    float t = smoothstep(0.0, 1.0, l);
    vec3 col = mix(shadow, light, t);

    // Per-pixel animated grain, re-rolled (not shifted) each frame.
    float g = hash3(vec3(floor(uv * params.SourceSize.xy),
                         float(params.FrameCount % 1024u))) - 0.5;
    col += g * params.grain;

    vec3 src = texture(Source, uv).rgb;
    FragColor = vec4(mix(src, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
