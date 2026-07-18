#version 450

// pvm-grille — broadcast production monitor (Sony PVM / BVM), single pass.
//
// The retro-gaming grail look: a razor-sharp aperture grille (vertical RGB
// stripes), faint scanlines, tight bloom around highlights, a small
// saturation lift and no curvature — professional glass, not a home TV.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float grille_strength;
    float grille_px;
    float scan_strength;
    float bloom_p;
    float bloom_threshold;
    float sharp_p;
    float sat_p;
} params;

#pragma parameter amount          "PVM: mix"             1.0  0.0 1.0 0.01
#pragma parameter grille_strength "PVM: aperture grille" 0.40 0.0 1.0 0.01
#pragma parameter grille_px       "PVM: grille size"     2.0  1.0 6.0 0.5
#pragma parameter scan_strength   "PVM: scanlines"       0.18 0.0 0.8 0.01
#pragma parameter bloom_p         "PVM: bloom"           0.50 0.0 1.5 0.05
#pragma parameter bloom_threshold "PVM: bloom threshold" 0.65 0.0 1.0 0.01
#pragma parameter sharp_p         "PVM: sharpness"       0.35 0.0 1.5 0.05
#pragma parameter sat_p           "PVM: saturation"      1.12 0.5 2.0 0.02

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

void main()
{
    vec2 uv = vTexCoord;
    vec3 col = texture(Source, uv).rgb;

    // Unsharp sharpening against a small cross blur.
    vec2 px = params.SourceSize.zw;
    vec3 blur = (texture(Source, uv + vec2(px.x, 0.0)).rgb +
                 texture(Source, uv - vec2(px.x, 0.0)).rgb +
                 texture(Source, uv + vec2(0.0, px.y)).rgb +
                 texture(Source, uv - vec2(0.0, px.y)).rgb) * 0.25;
    col = clamp(col + (col - blur) * params.sharp_p, 0.0, 1.0);

    // Tight highlight bloom from a wider cross.
    vec2 bpx = px * 2.0;
    vec3 wide = (texture(Source, uv + bpx).rgb + texture(Source, uv - bpx).rgb +
                 texture(Source, uv + vec2(bpx.x, -bpx.y)).rgb +
                 texture(Source, uv + vec2(-bpx.x, bpx.y)).rgb) * 0.25;
    vec3 bright = max(wide - vec3(params.bloom_threshold), vec3(0.0)) / max(1.0 - params.bloom_threshold, 1e-3);
    col += bright * params.bloom_p * 0.6;

    // Aperture grille: vertical RGB stripes, no vertical stagger.
    float gx = floor(uv.x * params.OutputSize.x / max(params.grille_px, 1.0));
    float tri = mod(gx, 3.0);
    vec3 grille = vec3(tri == 0.0 ? 1.0 : 0.45, tri == 1.0 ? 1.0 : 0.45, tri == 2.0 ? 1.0 : 0.45);
    col *= mix(vec3(1.0), grille * 1.2, params.grille_strength);

    // Faint scanlines.
    float row = uv.y * params.OutputSize.y / max(params.grille_px, 1.0);
    col *= 1.0 - params.scan_strength * 0.5 * (1.0 + cos(6.2831853 * row));

    // Broadcast color: small saturation lift.
    float l = luma(col);
    col = clamp(mix(vec3(l), col, params.sat_p), 0.0, 1.0);

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
