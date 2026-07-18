#version 450

// soft-crt pass 2 — composite: a subtle analog-CRT finishing grade built from
// five independently tunable parts, applied over the clean Original:
//
//   1. soften   — mix the Gaussian-blurred frame over the sharp one.
//   2. bloom    — soft-threshold the blurred frame and screen-blend it back,
//                 so highlights bleed a gentle glow.
//   3. exposure — overall gain, applied before the contrast curve.
//   4. contrast — linear contrast around mid-grey.
//   5. scanlines — fine sin² darkening with tunable spacing (output px).
//   6. vignette — aspect-corrected radial darkening toward the corners.
//   7. noise    — animated film grain, slightly heavier in shadows.
//
// Every part has its own strength slider and turning one to 0 removes just
// that part; amount = 0 is a clean passthrough of the Original.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float soften;
    float bloom_threshold;
    float bloom_intensity;
    float exposure;
    float contrast;
    float scan_strength;
    float scan_period;
    float vignette;
    float noise_strength;
    float noise_size;
} params;

#pragma parameter amount          "Soft CRT: mix"            1.0  0.0  1.0  0.01
#pragma parameter soften          "Blur: soften"             0.25 0.0  1.0  0.01
#pragma parameter bloom_threshold "Bloom: threshold"         0.6  0.0  1.0  0.01
#pragma parameter bloom_intensity "Bloom: intensity"         0.7  0.0  3.0  0.05
#pragma parameter exposure        "Exposure"                 1.0  0.25 2.5  0.05
#pragma parameter contrast        "Contrast"                 1.15 0.5  2.5  0.05
#pragma parameter scan_strength   "Scanlines: strength"      0.12 0.0  1.0  0.01
#pragma parameter scan_period     "Scanlines: spacing (px)"  3.0  2.0 12.0  0.5
#pragma parameter vignette        "Vignette"                 0.25 0.0  1.0  0.01
#pragma parameter noise_strength  "Noise: strength"          0.05 0.0  0.4  0.005
#pragma parameter noise_size      "Noise: grain size (px)"   1.5  1.0  6.0  0.25

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;     // blurred frame
layout(set = 0, binding = 3) uniform sampler2D Original;   // clean input

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// 3D hash: the frame index is a separate dimension, NOT added to the cell
// coords — hash(cell + frame) translates the whole field diagonally each
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
    vec3 orig    = texture(Original, uv).rgb;
    vec3 blurred = texture(Source, uv).rgb;

    // 1. soften — blend the blurred frame over the sharp one.
    vec3 c = mix(orig, blurred, params.soften);

    // 2. bloom — the blur passes already exist, so the glow is just a
    // soft-threshold of the blurred frame, screen-blended to avoid clipping.
    vec3 glow = max(blurred - vec3(params.bloom_threshold), 0.0)
              / max(1.0 - params.bloom_threshold, 1e-3)
              * params.bloom_intensity;
    c = 1.0 - (1.0 - c) * (1.0 - clamp(glow, 0.0, 1.0));

    // 3. exposure — plain gain before the contrast pivot so pushing it
    // brightens mids without inverting the curve.
    c *= params.exposure;

    // 4. contrast around mid-grey.
    c = clamp((c - 0.5) * params.contrast + 0.5, 0.0, 1.0);

    // 5. scanlines — sin² profile so lines stay soft at small spacings.
    float s = sin(3.14159265 * uv.y * params.OutputSize.y / params.scan_period);
    c *= 1.0 - params.scan_strength * (1.0 - s * s);

    // 6. vignette — radial falloff in aspect-corrected coords so the shading
    // is round on any frame shape.
    vec2 vp = (uv - 0.5) * vec2(params.OutputSize.x * params.OutputSize.w, 1.0);
    c *= 1.0 - params.vignette * smoothstep(0.35, 0.85, length(vp));

    // 7. grain — animated hash noise in `noise_size`-px cells, weighted
    // toward shadows/mids the way film grain reads.
    vec2 cell = floor(uv * params.OutputSize.xy / params.noise_size);
    float g = hash3(vec3(cell, float(params.FrameCount % 1024u))) - 0.5;
    c += g * params.noise_strength * (1.0 - 0.6 * luma(c));

    FragColor = vec4(mix(orig, clamp(c, 0.0, 1.0), params.amount), 1.0);
}
