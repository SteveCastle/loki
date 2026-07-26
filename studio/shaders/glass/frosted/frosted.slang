#version 450

// frosted — bathroom-window glass: the image is jittered through a smooth
// noise field (the frost texture) and softened with a small ring blur, with a
// cool brightness lift so it reads as glass rather than plain blur. detail
// sets the grain size of the frost. Single pass; amount = 0 is a clean
// passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float frost;
    float blur;
    float detail;
    float chill;
} params;

#pragma parameter amount "FrostedGlass: mix"        1.0  0.0  1.0  0.01
#pragma parameter frost  "FrostedGlass: frost"      0.6  0.0  2.0  0.02
#pragma parameter blur   "FrostedGlass: softness"   0.5  0.0  2.0  0.02
#pragma parameter detail "FrostedGlass: grain size" 60.0 10.0 240.0 1.0
#pragma parameter chill  "FrostedGlass: cool sheen" 0.35 0.0  1.0  0.02

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
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main()
{
    vec2 uv = vTexCoord;
    float aspect = params.SourceSize.x / params.SourceSize.y;
    vec2 p = uv * vec2(aspect, 1.0) * params.detail;

    // The frost: a smooth random refraction per grain of the glass.
    vec2 jit = vec2(vnoise(p) - 0.5, vnoise(p + 31.7) - 0.5)
             * params.frost * 0.02;
    vec2 fuv = uv + jit;

    // Small ring blur on top so edges frost out instead of just wobbling.
    vec2 px = params.blur * 2.5 / params.SourceSize.xy;
    vec3 col = texture(Source, fuv).rgb * 0.4;
    col += texture(Source, fuv + vec2( px.x,  px.y * 0.5)).rgb * 0.15;
    col += texture(Source, fuv + vec2(-px.x,  px.y * 0.5)).rgb * 0.15;
    col += texture(Source, fuv + vec2( px.x * 0.5, -px.y)).rgb * 0.15;
    col += texture(Source, fuv + vec2(-px.x * 0.5, -px.y)).rgb * 0.15;

    // Cold glass sheen: a touch brighter, a touch blue.
    col = mix(col, col * vec3(0.94, 0.99, 1.06) + vec3(0.05, 0.06, 0.08),
              params.chill);

    vec3 src = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(src, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
