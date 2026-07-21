#version 450

// underwater — sunken-footage look: the frame wobbles behind gentle moving
// refraction, a teal water column tints and darkens with depth, and animated
// caustic light ripples play across everything. Single pass; amount = 0 is a
// clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float wobble;
    float speed;
    float tint;
    float caustics;
    float depth;
} params;

#pragma parameter amount   "Underwater: mix"        1.0  0.0 1.0 0.01
#pragma parameter wobble   "Underwater: wobble"     0.6  0.0 2.0 0.02
#pragma parameter speed    "Underwater: speed"      1.0  0.0 4.0 0.05
#pragma parameter tint     "Underwater: teal tint"  0.6  0.0 1.0 0.02
#pragma parameter caustics "Underwater: caustics"   0.5  0.0 1.5 0.02
#pragma parameter depth    "Underwater: depth fade" 0.45 0.0 1.0 0.02

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

// Two crossing interference bands, cubed for sharp bright ridges — the
// classic pool-floor caustic pattern.
float caustic(vec2 p, float t)
{
    float c = sin(p.x * 13.0 + t * 1.7 + sin(p.y *  9.0 + t * 1.3))
            + sin(p.y * 11.0 - t * 1.1 + sin(p.x *  7.0 - t * 1.9));
    c = clamp(0.5 + 0.25 * c, 0.0, 1.0);
    return c * c * c;
}

void main()
{
    vec2 uv = vTexCoord;
    float t = float(params.FrameCount) * (1.0 / 60.0) * params.speed;
    float aspect = params.SourceSize.x / params.SourceSize.y;
    vec2 p = uv * vec2(aspect, 1.0);

    // Layered surface refraction: two frequencies so it never visibly loops.
    vec2 sway = vec2(
        sin(p.y * 9.0 + t * 1.6) + 0.6 * sin(p.y * 23.0 - t * 2.3),
        sin(p.x * 8.0 - t * 1.2) + 0.6 * sin(p.x * 19.0 + t * 1.9));
    vec2 wuv = uv + sway * params.wobble * 0.006;

    vec3 col = texture(Source, wuv).rgb;

    // Water column: teal absorption that deepens down the frame (row 0 = top).
    vec3 sea = col * vec3(0.45, 0.78, 0.85) + vec3(0.0, 0.045, 0.075);
    col = mix(col, sea, params.tint);
    col *= 1.0 - params.depth * uv.y * 0.75;

    // Dancing light.
    col += vec3(0.85, 1.0, 0.95) * caustic(p * 1.4, t) * params.caustics * 0.35;

    vec3 src = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(src, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
