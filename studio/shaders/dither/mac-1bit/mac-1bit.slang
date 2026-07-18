#version 450

// mac-1bit — classic Macintosh one-bit dither, single pass.
//
// The System 7 / HyperCard look: 8×8 Bayer to pure 1-bit, chunky square
// pixels, warm paper white and soft black ink (real compact Macs never hit
// pure #000/#fff), with an invert toggle for MacPaint negative moods.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float chunk_mb;
    float contrast_mb;
    float bright_mb;
    float invert_mb;
} params;

#pragma parameter amount      "Mac 1-bit: mix"        1.0  0.0 1.0  0.01
#pragma parameter chunk_mb    "Mac 1-bit: pixel size" 2.0  1.0 10.0 1.0
#pragma parameter contrast_mb "Mac 1-bit: contrast"   1.2  0.5 2.5  0.05
#pragma parameter bright_mb   "Mac 1-bit: brightness" 1.05 0.4 2.0  0.05
#pragma parameter invert_mb   "Mac 1-bit: invert"     0.0  0.0 1.0  1.0

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

float b2i(vec2 q) { return q.x * 2.0 + q.y * 3.0 - 4.0 * q.x * q.y; }

float bayer8T(vec2 ip)
{
    float v = b2i(mod(ip, 2.0)) * 16.0
            + b2i(mod(floor(ip / 2.0), 2.0)) * 4.0
            + b2i(mod(floor(ip / 4.0), 2.0));
    return (v + 0.5) / 64.0;
}

void main()
{
    float chunk = max(params.chunk_mb, 1.0);
    vec2 px = floor(vTexCoord * params.OutputSize.xy / chunk);
    vec3 s = texture(Source, (px + 0.5) * chunk * params.OutputSize.zw).rgb;
    float l = dot(s, vec3(0.299, 0.587, 0.114)) * params.bright_mb;
    l = clamp((l - 0.5) * params.contrast_mb + 0.5, 0.0, 1.0);

    float on = step(bayer8T(px), l);
    if (params.invert_mb > 0.5) on = 1.0 - on;

    // Platinum-ish paper, CRT-soft ink.
    vec3 col = mix(vec3(0.09, 0.09, 0.11), vec3(0.93, 0.94, 0.91), on);

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
