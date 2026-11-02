#version 450

// depth-blur pass 1 — variable-radius lens blur. The layer's MASK is the
// blur map: each pixel's circle of confusion is radius * mask, so a depth
// layer (Focus mode: white = far from the focal plane) wired in as the mask
// gives real depth of field — every pixel blurred by its own distance, not
// a mix between one sharp and one blurred copy.
//
// Gather-as-scatter: a pixel collects samples over its OWN disc, but a
// sample only counts if that sample's circle reaches back to here. A sharp
// subject therefore never smears into the blurry background around it
// (its circles are tiny), and the blurry background never gathers over the
// subject (the subject's own disc is ~0). Same golden-angle spiral and
// mip-matched taps as lens-blur, so big radii have no holes.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float radius;
    float blades;
    float rotation;
    float protect;
    float map_invert;
    float map_gamma;
} params;

#pragma parameter radius     "Depth blur: max radius (px)" 16.0 0.0 150.0 0.25
#pragma parameter blades     "Depth blur: blades (0 = round)" 0.0 0.0 9.0 1.0
#pragma parameter rotation   "Depth blur: blade rotation (deg)" 0.0 0.0 360.0 1.0
#pragma parameter protect    "Depth blur: edge protection" 1.0 0.0 1.0 0.01
#pragma parameter map_invert "Depth blur: invert map (white = sharp)" 0.0 0.0 1.0 1.0
#pragma parameter map_gamma  "Depth blur: map curve" 1.0 0.2 5.0 0.01
// The mask is read here as the per-pixel radius; the group must not ALSO
// mix the result over the input through it.
#pragma mask_control

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;     // weighted linear (float, mipmapped)
layout(set = 0, binding = 3) uniform sampler2D Mask;       // blur map

const int TAPS = 128;
const float GOLDEN = 2.39996323;
const float TAU = 6.28318531;

float coc(vec2 uv)
{
    float m = textureLod(Mask, uv, 0.0).r;
    m = mix(m, 1.0 - m, params.map_invert);
    return params.radius * pow(clamp(m, 0.0, 1.0), params.map_gamma);
}

void main()
{
    vec2 uv = vTexCoord;
    float R = coc(uv);
    vec4 center = textureLod(Source, uv, 0.0);
    if (R < 0.5) { FragColor = center; return; }

    float spacing = R * sqrt(3.14159265 / float(TAPS));
    float lod = log2(max(spacing * 0.8, 1.0));
    float n = floor(params.blades + 0.5);
    bool poly = n >= 3.0;
    float seg = TAU / max(n, 3.0);
    float rot = radians(params.rotation);
    vec2 px = params.SourceSize.zw;

    vec4 acc = center;
    float norm = 1.0;
    for (int i = 0; i < TAPS; ++i) {
        float r = sqrt((float(i) + 0.5) / float(TAPS));
        float th = float(i) * GOLDEN;
        float s = 1.0;
        if (poly) {
            float a = mod(th - rot, seg) - 0.5 * seg;
            s = cos(0.5 * seg) / cos(a);
        }
        vec2 off = vec2(cos(th), sin(th)) * (r * s * R);
        vec2 suv = uv + off * px;
        float d = length(off);
        // Does this sample's own circle reach us? Sharp neighbours don't.
        float reach = smoothstep(d - 1.0, d + 1.0, coc(suv));
        float w = mix(1.0, reach, params.protect) * (s * s);
        acc += textureLod(Source, suv, lod) * w;
        norm += w;
    }
    FragColor = acc / norm;
}
