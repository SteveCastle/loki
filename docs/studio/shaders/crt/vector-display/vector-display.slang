#version 450

// vector-display — XY vector monitor (Asteroids / Vectrex / radar scope),
// single self-feedback pass.
//
// Nothing but beam strokes: Sobel edges become glowing green traces on
// black glass, with a soft halo, long phosphor trails (PassFeedback), a
// nervous beam jitter and an optional faint graticule grid.
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
    float beam;
    float threshold_v;
    float halo_v;
    float trail_v;
    float jitter_v;
    float grid_v;
    float tint_r;
    float tint_g;
    float tint_b;
} params;

#pragma parameter amount      "Vector: mix"         1.0  0.0 1.0  0.01
#pragma parameter beam        "Vector: beam gain"   2.2  0.2 8.0  0.1
#pragma parameter threshold_v "Vector: threshold"   0.10 0.0 0.6  0.01
#pragma parameter halo_v      "Vector: halo"        0.60 0.0 1.5  0.05
#pragma parameter trail_v     "Vector: phosphor"    0.62 0.0 0.95 0.01
#pragma parameter jitter_v    "Vector: beam jitter" 0.5  0.0 3.0  0.05
#pragma parameter grid_v      "Vector: graticule"   0.05 0.0 0.4  0.01
#pragma parameter tint_r      "Beam R"              0.25 0.0 1.5  0.01
#pragma parameter tint_g      "Beam G"              1.10 0.0 1.5  0.01
#pragma parameter tint_b      "Beam B"              0.45 0.0 1.5  0.01

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

float luma_at(vec2 uv)
{
    return dot(texture(Source, uv).rgb, vec3(0.299, 0.587, 0.114));
}

// Sobel gradient magnitude at uv with the given pixel step.
float edges(vec2 uv, vec2 px)
{
    float tl = luma_at(uv + vec2(-px.x, -px.y)), t = luma_at(uv + vec2(0.0, -px.y)), tr = luma_at(uv + vec2(px.x, -px.y));
    float l  = luma_at(uv + vec2(-px.x,  0.0)),                                      r  = luma_at(uv + vec2(px.x,  0.0));
    float bl = luma_at(uv + vec2(-px.x,  px.y)), b = luma_at(uv + vec2(0.0,  px.y)), br = luma_at(uv + vec2(px.x,  px.y));
    float gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    float gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
    return length(vec2(gx, gy));
}

void main()
{
    // Beam jitter: the whole trace wanders a fraction of a pixel.
    float t = params.Time;
    vec2 jit = (vec2(hash(vec2(floor(t * 90.0), 1.0)), hash(vec2(floor(t * 90.0), 7.0))) - 0.5)
             * params.jitter_v * params.SourceSize.zw;
    vec2 uv = vTexCoord + jit;

    vec2 px = params.SourceSize.zw;
    float core = edges(uv, px);
    core = smoothstep(params.threshold_v, params.threshold_v + 0.25, core * params.beam);

    // Halo: the same trace, wider and dimmer.
    float halo = edges(uv, px * 2.5);
    halo = smoothstep(params.threshold_v, params.threshold_v + 0.5, halo * params.beam) * 0.5;

    float trace = clamp(core + halo * params.halo_v, 0.0, 1.4);

    // Faint radar graticule.
    vec2 g = fract(uv * 10.0);
    float grat = (1.0 - smoothstep(0.0, 0.04, min(min(g.x, 1.0 - g.x), min(g.y, 1.0 - g.y)))) * params.grid_v;

    vec3 tint = vec3(params.tint_r, params.tint_g, params.tint_b);
    vec3 col = tint * (trace + grat * 0.5);

    // Phosphor trails.
    vec3 prev = texture(PassFeedback0, vTexCoord).rgb;
    col = max(col, prev * params.trail_v);

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
