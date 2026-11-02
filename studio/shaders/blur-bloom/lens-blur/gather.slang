#version 450

// lens-blur pass 1 — out-of-focus lens (bokeh) blur: a flat disc kernel,
// or an n-bladed aperture polygon, instead of a Gaussian's soft falloff.
// Gathers 160 samples on a golden-angle (Vogel) spiral, which covers the
// disc evenly without the rings a polar grid shows. Each sample reads the
// mip level whose texel matches the spacing between spiral samples, so the
// samples tile the aperture with no holes at any radius. Blades remap the
// spiral radially onto the polygon; samples pushed toward a corner cover
// more area, so they are weighted by the stretch squared. Output stays
// premultiplied (rgb*w, w) for the post-filter pass.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float radius;
    float blades;
    float rotation;
    float squeeze;
} params;

#pragma parameter radius   "Lens: aperture radius (px)" 12.0 0.0 150.0 0.25
#pragma parameter blades   "Lens: blades (0 = round)" 0.0 0.0 9.0 1.0
#pragma parameter rotation "Lens: blade rotation (deg)" 0.0 0.0 360.0 1.0
#pragma parameter squeeze  "Lens: anamorphic squeeze" 0.0 0.0 1.0 0.01

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

const int TAPS = 160;
const float GOLDEN = 2.39996323;
const float TAU = 6.28318531;

void main()
{
    vec2 uv = vTexCoord;
    float R = params.radius;
    // Mean distance between neighbouring spiral samples; prefilter a bit
    // under it so the disc edge stays crisp while the samples still overlap.
    float spacing = R * sqrt(3.14159265 / float(TAPS));
    float lod = log2(max(spacing * 0.8, 1.0));

    float n = floor(params.blades + 0.5);
    bool poly = n >= 3.0;
    float seg = TAU / max(n, 3.0);
    float rot = radians(params.rotation);
    // Anamorphic lenses squeeze the aperture into a vertical oval.
    vec2 shape = vec2(1.0 - 0.5 * params.squeeze, 1.0) * R * params.SourceSize.zw;

    vec4 acc = vec4(0.0);
    float norm = 0.0;
    for (int i = 0; i < TAPS; ++i) {
        float r = sqrt((float(i) + 0.5) / float(TAPS));
        float th = float(i) * GOLDEN;
        float s = 1.0;
        if (poly) {
            float a = mod(th - rot, seg) - 0.5 * seg;
            s = cos(0.5 * seg) / cos(a);
        }
        vec2 off = vec2(cos(th), sin(th)) * (r * s);
        acc += textureLod(Source, uv + off * shape, lod) * (s * s);
        norm += s * s;
    }
    FragColor = acc / norm;
}
