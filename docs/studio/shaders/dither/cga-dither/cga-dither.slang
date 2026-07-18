#version 450

// cga-dither — palette-constrained ordered dither, single pass.
//
// Bayer-dithers the frame down to a 4-color early-PC palette: CGA mode 4
// (black/cyan/magenta/white), CGA mode 4 palette 0 (black/green/red/brown),
// or straight 1-bit black & white. Chunky pixels included. The authentic
// "we only have four colors and we're going to like it" energy.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float chunk_cg;
    float palette_cg;
    float spread_cg;
    float bright_cg;
} params;

#pragma parameter amount     "CGA: mix"                 1.0  0.0 1.0 0.01
#pragma parameter chunk_cg   "CGA: pixel size"          3.0  1.0 12.0 1.0
#pragma parameter palette_cg "CGA: palette (0/1/2=1bit)" 0.0 0.0 2.0 1.0
#pragma parameter spread_cg  "CGA: dither spread"       0.55 0.0 1.0 0.01
#pragma parameter bright_cg  "CGA: brightness"          1.1  0.5 2.0 0.05

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

float b2i(vec2 q) { return q.x * 2.0 + q.y * 3.0 - 4.0 * q.x * q.y; }

float bayer4T(vec2 ip)
{
    float v = b2i(mod(ip, 2.0)) * 4.0 + b2i(mod(floor(ip / 2.0), 2.0));
    return (v + 0.5) / 16.0;
}

vec3 pal(int mode, int i)
{
    if (mode == 0) {   // CGA mode 4 palette 1 (high)
        return i == 0 ? vec3(0.0) :
               i == 1 ? vec3(0.33, 1.0, 1.0) :
               i == 2 ? vec3(1.0, 0.33, 1.0) : vec3(1.0);
    }
    if (mode == 1) {   // CGA mode 4 palette 0
        return i == 0 ? vec3(0.0) :
               i == 1 ? vec3(0.33, 1.0, 0.33) :
               i == 2 ? vec3(1.0, 0.33, 0.33) : vec3(0.67, 0.33, 0.0);
    }
    return i < 2 ? vec3(0.0) : vec3(1.0);   // 1-bit
}

void main()
{
    float chunk = max(params.chunk_cg, 1.0);
    vec2 px = floor(vTexCoord * params.OutputSize.xy / chunk);
    vec3 c = texture(Source, (px + 0.5) * chunk * params.OutputSize.zw).rgb * params.bright_cg;

    // Push the color around by the ordered threshold, then snap to the
    // nearest palette entry — the classic ordered palette dither.
    int mode = int(clamp(params.palette_cg, 0.0, 2.0));
    c += (bayer4T(px) - 0.5) * params.spread_cg;

    vec3 best = pal(mode, 0);
    float bestD = 1e9;
    for (int i = 0; i < 4; i++) {
        vec3 p = pal(mode, i);
        float d = dot(c - p, c - p);
        if (d < bestD) { bestD = d; best = p; }
    }

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, best, params.amount), 1.0);
}
