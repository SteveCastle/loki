#version 450

// datamosh-skin — present pass (cosmetic artifacts, skin-masked)
//
// Same cosmetic stage as datamosh_present.slang (per-block chroma fringe + DCT
// crush), but the contribution is scaled by the skin mask so off-skin areas
// stay clean. Skin is detected on the clean `Original` input (not the moshed
// Source) so the mask is reliable.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float mosh_amount;
    float block_size;
    float streak;
    float chroma_shift;
    float quantize;
    float seed;
    float skin_floor;
    float skin_tolerance;
    float skin_gamma;
} params;

#pragma parameter mosh_amount    "Datamosh: amount"          0.65 0.0  1.0  0.01
#pragma parameter block_size     "Datamosh: macroblock px"   7.0  4.0 64.0  1.0
#pragma parameter streak         "Datamosh: vertical streak" 5.0  1.0 12.0  0.25
#pragma parameter chroma_shift   "Datamosh: chroma bleed"    0.4  0.0  2.0  0.01
#pragma parameter quantize       "Datamosh: DCT crush"       0.5  0.0  1.0  0.01
#pragma parameter seed           "Datamosh: seed"            7.0  0.0 64.0  1.0
#pragma parameter skin_floor     "Skin: off-skin effect"     0.05 0.0  1.0  0.01
#pragma parameter skin_tolerance "Skin: colour tolerance"    0.055 0.01 0.20 0.005
#pragma parameter skin_gamma     "Skin: mask sharpness"      1.6  0.5  4.0  0.05

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;    // moshed image (pass 0)
layout(set = 0, binding = 3) uniform sampler2D Original;  // clean input, for skin mask

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

float skin_mask(vec3 c)
{
    float Y  = dot(c, vec3(0.299, 0.587, 0.114));
    float Cb = (c.b - Y) * 0.564 + 0.5;
    float Cr = (c.r - Y) * 0.713 + 0.5;
    const vec2 center = vec2(0.43, 0.59);
    vec2  d = vec2(Cb, Cr) - center;
    float s = exp(-dot(d, d) / (2.0 * params.skin_tolerance * params.skin_tolerance));
    s *= smoothstep(0.08, 0.22, Y);
    s *= 1.0 - smoothstep(0.92, 1.0, Y);
    return clamp(pow(s, params.skin_gamma), 0.0, 1.0);
}

void main()
{
    vec2  uv  = vTexCoord;
    float skin = skin_mask(texture(Original, uv).rgb);
    float amt  = params.mosh_amount * mix(params.skin_floor, 1.0, skin);

    vec2 res   = params.SourceSize.xy;
    vec2 bpx   = vec2(max(params.block_size, 1.0)) * vec2(1.0, params.streak);
    vec2 grid  = res / bpx;
    vec2 block = floor(uv * grid);

    /* Tall-block-aligned, vertically-biased chroma fringe. */
    float cr   = hash21(block + params.seed);
    vec2  coff = vec2((cr - 0.5) * 0.4, (fract(cr * 7.13) - 0.5))
               * params.chroma_shift * bpx * params.SourceSize.zw * amt;

    vec3 col;
    col.r = texture(Source, uv + coff).r;
    col.g = texture(Source, uv).g;
    col.b = texture(Source, uv - coff).b;

    /* DCT-ish crush, scaled by skin amount. */
    float levels = mix(255.0, mix(255.0, 5.0, params.quantize), amt);
    col = floor(col * levels + 0.5) / levels;

    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
