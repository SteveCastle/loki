#version 450

// fog — a drifting veil of layered value-noise rolls over the frame. Density
// sets how thick the bank is, height biases it toward the bottom (ground fog)
// or top of the frame, and everything under the veil softens toward a cool
// grey. Single pass; amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float density;
    float fscale;
    float speed;
    float height;
} params;

#pragma parameter amount  "Fog: mix"          1.0  0.0  1.0 0.01
#pragma parameter density "Fog: density"      0.7  0.0  2.0 0.02
#pragma parameter fscale  "Fog: patch size"   3.0  0.5 12.0 0.1
#pragma parameter speed   "Fog: drift"        1.0  0.0  4.0 0.05
#pragma parameter height  "Fog: ground bias"  0.4 -1.0  1.0 0.02

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

float fbm(vec2 p)
{
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * vnoise(p);
        p = p * 2.03 + 17.1;
        a *= 0.5;
    }
    return v;
}

void main()
{
    vec2 uv = vTexCoord;
    float t = float(params.FrameCount) * (1.0 / 60.0) * params.speed;
    float aspect = params.SourceSize.x / params.SourceSize.y;
    vec2 p = uv * vec2(aspect, 1.0) * params.fscale;

    // Two banks drifting at different speeds read as slow tumbling depth.
    float n = fbm(p + vec2(t * 0.10, t * 0.02))
            + fbm(p * 1.7 - vec2(t * 0.06, -t * 0.015));
    n *= 0.5;

    // Ground bias: positive = thicker at the bottom (row 0 = top).
    float bias = clamp(1.0 + params.height * (uv.y * 2.0 - 1.0), 0.0, 2.0);
    float veil = clamp((0.35 + 0.85 * n) * params.density * bias, 0.0, 1.0);

    vec3 col = texture(Source, uv).rgb;
    vec3 fogc = vec3(0.72, 0.76, 0.82);
    // Fog flattens contrast before it whites out — mix toward the veil
    // colour, easing so thin fog is barely-there haze.
    vec3 hazed = mix(col, fogc, veil * veil * (3.0 - 2.0 * veil));

    FragColor = vec4(mix(col, hazed, params.amount), 1.0);
}
