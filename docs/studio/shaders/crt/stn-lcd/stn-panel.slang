#version 450

// stn-lcd — pass 1: the panel.
//
// Everything a mid-90s passive-matrix laptop screen did to the picture it
// was fed by pass 0: washed-out contrast under a blue-gray cast, vertical
// crosstalk shadows streaking above and below bright content (one shared
// column driver per column), the dual-scan seam across the middle where
// the two half-screens met, contrast rolling off with vertical viewing
// angle, backlight bleeding in from the edges, and the cell grid itself.
//
// Mixes against Original, so amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float res_x;
    float crosstalk;
    float seam;
    float washout;
    float tint;
    float angle_shade;
    float bleed;
    float grid;
} params;

#pragma parameter amount      "STN: amount"           1.0  0.0  1.0  0.01
#pragma parameter crosstalk   "STN: column crosstalk" 0.5  0.0  1.0  0.01
#pragma parameter seam        "STN: dual-scan seam"   0.5  0.0  1.0  0.01
#pragma parameter washout     "STN: contrast washout" 0.55 0.0  1.0  0.01
#pragma parameter tint        "STN: blue-gray cast"   0.5  0.0  1.0  0.01
#pragma parameter angle_shade "STN: viewing angle"    0.4  0.0  1.0  0.01
#pragma parameter bleed       "STN: backlight bleed"  0.4  0.0  1.0  0.01
#pragma parameter grid        "STN: cell grid"        0.5  0.0  1.0  0.01

/* The panel only recolours and dims — coverage keeps the clean silhouette. */
#pragma matte_ignore amount

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
layout(set = 0, binding = 3) uniform sampler2D Original;

void main()
{
    vec2  uv  = vTexCoord;
    float amt = params.amount;

    /* The dual-scan seam: the lower half was driven by its own controller,
     * landing a hair off in both brightness and horizontal phase. */
    float lower = step(0.5, uv.y);
    vec2  suv   = uv + lower * vec2(params.seam * 0.0015, 0.0);

    vec3 col = texture(Source, suv).rgb;

    /* Column crosstalk: sparse taps up and down the same column — bright
     * content prints faint ghost streaks along its whole column. */
    float xt = 0.0;
    for (int k = 1; k <= 4; k++) {
        float dy = float(k) * 14.0 * params.SourceSize.w;
        xt += dot(texture(Source, clamp(suv + vec2(0.0, dy), 0.0, 1.0)).rgb, vec3(0.333));
        xt += dot(texture(Source, clamp(suv - vec2(0.0, dy), 0.0, 1.0)).rgb, vec3(0.333));
    }
    col -= (xt * 0.125 - 0.5) * params.crosstalk * 0.16 * amt;

    /* Washed-out contrast under a blue-gray cast: lifted blacks, dim
     * whites, muted saturation. */
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, vec3(luma), params.washout * 0.45 * amt);
    col = mix(col, col * 0.72 + 0.16, params.washout * amt);
    col = mix(col, col * vec3(0.93, 0.97, 1.06) + vec3(0.0, 0.01, 0.03), params.tint * amt);

    /* Vertical viewing angle: contrast inverts toward the top, washes out
     * toward the bottom — the STN sweet spot was a few degrees tall. */
    float va = (uv.y - 0.5) * 2.0;
    col = mix(col, mix(vec3(0.35), 1.0 - col * 0.6, 0.4),
              max(-va, 0.0) * params.angle_shade * 0.35 * amt);
    col = mix(col, col * 0.8 + 0.18, max(va, 0.0) * params.angle_shade * 0.3 * amt);

    /* Seam brightness step + a one-pixel dark rule across the middle. */
    col *= 1.0 - lower * params.seam * 0.05 * amt;
    float rule = 1.0 - smoothstep(0.0, params.SourceSize.w * 1.5, abs(uv.y - 0.5));
    col *= 1.0 - rule * params.seam * 0.5 * amt;

    /* Backlight bleed: the CCFL tubes glow in from top and bottom edges. */
    float eb = exp(-uv.y * 14.0) + exp(-(1.0 - uv.y) * 14.0);
    col += vec3(0.9, 0.95, 1.0) * eb * params.bleed * 0.12 * amt;

    /* Cell grid: dark gaps between cells on the same grid as pass 0. */
    vec2 panel = vec2(params.res_x,
                      max(params.res_x * params.SourceSize.y / params.SourceSize.x, 1.0));
    vec2  cf   = fract(uv * panel);
    float gap  = (1.0 - smoothstep(0.0, 0.18, cf.x))
               + (1.0 - smoothstep(0.0, 0.18, cf.y));
    col *= 1.0 - clamp(gap, 0.0, 1.0) * params.grid * 0.25 * amt;

    vec3 clean = texture(Original, vTexCoord).rgb;
    FragColor = vec4(clamp(mix(clean, col, amt), 0.0, 1.0), 1.0);
}
