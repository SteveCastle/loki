#version 450

// vhs-bootleg — pass 0: the signal itself.
//
// VHS records colour at a fraction of the luma bandwidth and a little
// behind it, so colour smears sideways out of every hard edge and lands
// off-register. Each dub of a dub loses more of it: chroma washes out,
// blacks lift, the luma gets crunchier and the recorder's sharpening
// leaves a ringing halo on every vertical edge.
//
// This pass only degrades the signal — the tape transport damage (jitter,
// tracking, dropouts, head switch) is pass 1.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float Time;
    float chroma_bw;
    float chroma_lag;
    float gen_loss;
    float ring;
    float smear;
    float chroma_noise;
} params;

#pragma parameter chroma_bw    "VHS: chroma smear"   4.0  0.0 16.0 0.25
#pragma parameter chroma_lag   "VHS: chroma lag"     3.0 -8.0 16.0 0.25
#pragma parameter gen_loss     "VHS: generation loss" 0.55 0.0 1.0 0.01
#pragma parameter ring         "VHS: edge ringing"   0.6  0.0  2.0 0.01
#pragma parameter smear        "VHS: luma softness"  2.4  0.0  8.0 0.1
#pragma parameter chroma_noise "VHS: colour noise"   0.25 0.0  1.0 0.01

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

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

void main()
{
    vec2  uv = vTexCoord;
    float tx = params.SourceSize.z;

    // Chroma: a wide one-sided-ish box along the line, offset by the lag,
    // so colour trails to the right of whatever produced it.
    vec3  chroma = vec3(0.0);
    float wsum = 0.0;
    for (int i = -8; i <= 8; ++i) {
        float w = 1.0 - abs(float(i)) / 9.0;
        float o = (float(i) * params.chroma_bw - params.chroma_lag) * tx;
        vec3  s = texture(Source, vec2(clamp(uv.x + o, 0.0, 1.0), uv.y)).rgb;
        chroma += (s - vec3(luma(s))) * w;
        wsum   += w;
    }
    chroma /= max(wsum, 1e-4);

    // Luma: softened along the line, then over-sharpened by the deck's
    // enhancer — the overshoot is what leaves the bright edge halo.
    float sx = params.smear * tx;
    float y0 = luma(texture(Source, uv).rgb);
    float yl = luma(texture(Source, vec2(clamp(uv.x - sx, 0.0, 1.0), uv.y)).rgb);
    float yr = luma(texture(Source, vec2(clamp(uv.x + sx, 0.0, 1.0), uv.y)).rgb);
    float y  = (y0 * 2.0 + yl + yr) * 0.25;

    float ex = luma(texture(Source, vec2(clamp(uv.x - 2.0 * tx, 0.0, 1.0), uv.y)).rgb);
    float exr = luma(texture(Source, vec2(clamp(uv.x + 2.0 * tx, 0.0, 1.0), uv.y)).rgb);
    y += (y0 * 2.0 - ex - exr) * 0.5 * params.ring;

    // Generation loss: colour drains away and the black floor lifts toward
    // the grey a worn tape settles on.
    chroma *= mix(1.0, 0.35, params.gen_loss);
    y = mix(y, y * 0.88 + 0.07, params.gen_loss);

    // Colour noise: coarse blotches that crawl, never fine grain — chroma
    // is the low-bandwidth channel, so its noise is big and slow.
    vec2 cell = floor(uv * vec2(30.0, 90.0) + vec2(0.0, floor(params.Time * 14.0)));
    vec3 blotch = vec3(hash21(cell), hash21(cell + 17.0), hash21(cell + 41.0)) - 0.5;
    chroma += blotch * params.chroma_noise * 0.25;

    FragColor = vec4(clamp(vec3(y) + chroma, 0.0, 1.0), 1.0);
}
