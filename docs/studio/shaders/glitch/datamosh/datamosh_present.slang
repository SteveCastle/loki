#version 450

// datamosh — present pass (cosmetic compression artifacts)
//
// Runs once on the moshed image from datamosh.slang. These effects are kept
// OUT of the feedback loop on purpose: chroma bleed and DCT crush would
// compound into garbage if they fed back into themselves every frame. Here
// they're applied a single time at output resolution.
//
// All contributions scale with mosh_amount, so mosh_amount = 0 leaves the
// (already clean) image untouched.

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
} params;

#pragma parameter mosh_amount  "Datamosh: amount"          0.55 0.0  1.0  0.01
#pragma parameter block_size   "Datamosh: macroblock px"   16.0 4.0 64.0  1.0
#pragma parameter streak       "Datamosh: vertical streak" 4.0  1.0 12.0  0.25
#pragma parameter chroma_shift "Datamosh: chroma bleed"    0.4  0.0  2.0  0.01
#pragma parameter quantize     "Datamosh: DCT crush"       0.5  0.0  1.0  0.01
#pragma parameter seed         "Datamosh: seed"            7.0  0.0 64.0  1.0

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

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

void main()
{
    vec2  uv  = vTexCoord;
    float amt = params.mosh_amount;

    vec2 res   = params.SourceSize.xy;
    vec2 bpx   = vec2(max(params.block_size, 1.0)) * vec2(1.0, params.streak);
    vec2 grid  = res / bpx;
    vec2 block = floor(uv * grid);

    /* Per-block chroma displacement: tall-block-aligned colour fringing, biased
     * vertical so it reads as a streak rather than a square fringe. */
    float cr   = hash21(block + params.seed);
    vec2  coff = vec2((cr - 0.5) * 0.4, (fract(cr * 7.13) - 0.5))
               * params.chroma_shift * bpx * params.SourceSize.zw * amt;

    vec3 col;
    col.r = texture(Source, uv + coff).r;
    col.g = texture(Source, uv).g;
    col.b = texture(Source, uv - coff).b;

    /* DCT-ish crush: posterise toward a few levels per channel for the blocky,
     * low-bit compressed look. */
    float levels = mix(255.0, mix(255.0, 5.0, params.quantize), amt);
    col = floor(col * levels + 0.5) / levels;

    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
