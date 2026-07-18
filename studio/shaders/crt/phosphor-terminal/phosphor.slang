#version 450

// phosphor — monochrome phosphor terminal engine, single self-feedback pass.
//
// One source drives three presets (green / amber / paper-white): the frame
// is collapsed to luma, shaped by brightness + contrast, and re-emitted
// through a colored phosphor with scanlines, barrel curvature, soft glow,
// long-persistence trails (PassFeedback), mains flicker, tube noise and a
// corner vignette. tint_r/g/b are overridden per preset.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float Time;
    float amount;
    float brightness;
    float contrast_t;
    float scan_strength;
    float scan_px;
    float curvature;
    float glow;
    float persistence;
    float flicker;
    float noise_t;
    float vignette_t;
    float tint_r;
    float tint_g;
    float tint_b;
} params;

#pragma parameter amount        "Terminal: mix"           1.0  0.0 1.0  0.01
#pragma parameter brightness    "Terminal: brightness"    1.15 0.2 2.5  0.05
#pragma parameter contrast_t    "Terminal: contrast"      1.25 0.5 2.5  0.05
#pragma parameter scan_strength "Terminal: scanlines"     0.35 0.0 0.9  0.01
#pragma parameter scan_px       "Terminal: scan spacing"  3.0  2.0 10.0 0.5
#pragma parameter curvature     "Terminal: curvature"     0.12 0.0 0.5  0.01
#pragma parameter glow          "Terminal: glow"          0.55 0.0 1.5  0.05
#pragma parameter persistence   "Terminal: persistence"   0.45 0.0 0.95 0.01
#pragma parameter flicker       "Terminal: flicker"       0.06 0.0 0.3  0.01
#pragma parameter noise_t       "Terminal: noise"         0.06 0.0 0.4  0.01
#pragma parameter vignette_t    "Terminal: vignette"      0.35 0.0 1.0  0.01
#pragma parameter tint_r        "Phosphor R"              0.25 0.0 1.5  0.01
#pragma parameter tint_g        "Phosphor G"              1.05 0.0 1.5  0.01
#pragma parameter tint_b        "Phosphor B"              0.30 0.0 1.5  0.01

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
layout(set = 0, binding = 3) uniform sampler2D PassFeedback0;

float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// 3D hash: the frame index is a separate dimension, NOT added to the pixel
// coords — hash(px + frame) translates the whole field diagonally each
// frame and the noise reads as directional drift instead of static.
float hash3(vec3 p)
{
    p = fract(p * vec3(123.34, 345.45, 678.56));
    p += dot(p, p.yzx + 34.345);
    return fract((p.x + p.y) * p.z);
}

float luma_at(vec2 uv)
{
    vec3 c = texture(Source, uv).rgb;
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main()
{
    // Barrel curvature; outside the tube face is black.
    vec2 v = vTexCoord * 2.0 - 1.0;
    v *= 1.0 + params.curvature * 0.22 * vec2(dot(v, v));
    vec2 uv = v * 0.5 + 0.5;
    float on_tube = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);

    // Beam response: luma -> shaped intensity, plus a cheap 5-tap glow halo.
    float l = luma_at(uv);
    l = clamp((l - 0.5) * params.contrast_t + 0.5, 0.0, 1.0) * params.brightness;
    vec2 px = params.SourceSize.zw * 1.6;
    float halo = luma_at(uv + vec2(px.x, 0.0)) + luma_at(uv - vec2(px.x, 0.0))
               + luma_at(uv + vec2(0.0, px.y)) + luma_at(uv - vec2(0.0, px.y));
    l += halo * 0.25 * params.glow;

    // Scanlines (raster rows in output space).
    float row = uv.y * params.OutputSize.y / max(params.scan_px, 1.0);
    float scan = 1.0 - params.scan_strength * 0.5 * (1.0 + cos(6.2831853 * row));
    l *= mix(1.0, scan, on_tube);

    // Mains flicker + tube noise.
    float t = params.Time;
    l *= 1.0 - params.flicker * (0.5 + 0.5 * sin(t * 100.0)) * (0.6 + 0.4 * hash(vec2(floor(t * 60.0))));
    l += (hash3(vec3(floor(vTexCoord * params.OutputSize.xy),
                     float(params.FrameCount % 1024u))) - 0.5) * params.noise_t * (0.25 + l);

    // Phosphor emission + vignette.
    vec3 tint = vec3(params.tint_r, params.tint_g, params.tint_b);
    float vig = 1.0 - params.vignette_t * dot(v * 0.75, v * 0.75);
    vec3 term = tint * max(l, 0.0) * max(vig, 0.0) * on_tube;

    // Long-persistence trails: previous frame's face decays underneath.
    vec3 prev = texture(PassFeedback0, vTexCoord).rgb;
    term = max(term, prev * params.persistence);

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, term, params.amount), 1.0);
}
