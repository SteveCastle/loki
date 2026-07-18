#version 450

// crt-tv — living-room consumer television, single pass.
//
// The whole late-night broadcast package: barrel curvature, a staggered
// slot-mask phosphor triad, chroma bleed (luma stays sharp while color
// smears sideways, like composite video), bright-line beam scanlines, a
// drifting mains hum bar, antenna static and corner shading.
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
    float curvature;
    float mask_strength;
    float mask_px;
    float scan_strength;
    float bleed;
    float glow_tv;
    float static_tv;
    float hum;
    float vignette_tv;
} params;

#pragma parameter amount        "TV: mix"            1.0  0.0 1.0  0.01
#pragma parameter curvature     "TV: curvature"      0.18 0.0 0.5  0.01
#pragma parameter mask_strength "TV: phosphor mask"  0.45 0.0 1.0  0.01
#pragma parameter mask_px       "TV: mask size (px)" 3.0  1.0 8.0  0.5
#pragma parameter scan_strength "TV: scanlines"      0.30 0.0 0.9  0.01
#pragma parameter bleed         "TV: chroma bleed"   2.5  0.0 8.0  0.1
#pragma parameter glow_tv       "TV: glow"           0.35 0.0 1.5  0.05
#pragma parameter static_tv     "TV: static"         0.05 0.0 0.5  0.01
#pragma parameter hum           "TV: hum bar"        0.08 0.0 0.5  0.01
#pragma parameter vignette_tv   "TV: corner shade"   0.30 0.0 1.0  0.01

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

// 3D hash: the frame index is a separate dimension, NOT added to the pixel
// coords — hash(px + frame) translates the whole field diagonally each
// frame and the static reads as directional drift instead of true snow.
float hash3(vec3 p)
{
    p = fract(p * vec3(123.34, 345.45, 678.56));
    p += dot(p, p.yzx + 34.345);
    return fract((p.x + p.y) * p.z);
}

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main()
{
    vec2 v = vTexCoord * 2.0 - 1.0;
    v *= 1.0 + params.curvature * 0.22 * vec2(dot(v, v));
    vec2 uv = v * 0.5 + 0.5;
    float on_tube = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);

    // Composite-style chroma bleed: sharp luma, sideways-smeared chroma.
    vec3 sharp = texture(Source, uv).rgb;
    vec2 bpx = vec2(params.SourceSize.z * params.bleed, 0.0);
    vec3 smear = (texture(Source, uv - bpx).rgb + sharp + texture(Source, uv + bpx).rgb) / 3.0;
    vec3 col = vec3(luma(sharp)) + (smear - vec3(luma(smear)));

    // Glow from a wide cross tap.
    vec2 gpx = params.SourceSize.zw * 2.2;
    vec3 halo = texture(Source, uv + gpx).rgb + texture(Source, uv - gpx).rgb
              + texture(Source, uv + vec2(gpx.x, -gpx.y)).rgb
              + texture(Source, uv + vec2(-gpx.x, gpx.y)).rgb;
    col += halo * 0.25 * params.glow_tv * params.glow_tv;

    // Staggered slot-mask triad in output pixels.
    vec2 opx = uv * params.OutputSize.xy / max(params.mask_px, 1.0);
    float tri = mod(floor(opx.x), 3.0);
    vec3 mask = vec3(tri == 0.0 ? 1.0 : 0.35, tri == 1.0 ? 1.0 : 0.35, tri == 2.0 ? 1.0 : 0.35);
    float slot = mod(floor(opx.x / 3.0), 2.0);
    float slot_row = step(0.5, fract(opx.y * 0.5 + slot * 0.5));
    mask = mix(mask, mask * 0.7, params.scan_strength * 0.0 + (1.0 - slot_row) * 0.25);
    col *= mix(vec3(1.0), mask * 1.25, params.mask_strength);

    // Beam scanlines — bright content widens the beam.
    float row = uv.y * params.OutputSize.y / max(params.mask_px, 1.0);
    float beam = 0.5 + 0.5 * cos(6.2831853 * row);
    col *= 1.0 - params.scan_strength * beam * (1.0 - clamp(luma(col), 0.0, 1.0) * 0.6);

    // Drifting hum bar + antenna static.
    float bar = fract(uv.y - params.Time * 0.11);
    col *= 1.0 - params.hum * smoothstep(0.0, 0.18, bar) * smoothstep(0.35, 0.18, bar);
    float n = hash3(vec3(floor(vTexCoord * params.OutputSize.xy),
                         float(params.FrameCount % 1024u)));
    col += (n - 0.5) * params.static_tv;

    // Corner shading, tube cutoff.
    col *= max(1.0 - params.vignette_tv * dot(v * 0.8, v * 0.8), 0.0) * on_tube;

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
