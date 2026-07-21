#version 450

// smoke — rising wisps of fractal smoke curl over the frame, and the hot air
// under them shimmers (the image refracts slightly where the smoke is dense).
// shade runs white steam -> black soot. Single pass; amount = 0 is a clean
// passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float density;
    float swirl;
    float sscale;
    float speed;
    float shade;
} params;

#pragma parameter amount  "Smoke: mix"         1.0  0.0 1.0  0.01
#pragma parameter density "Smoke: density"     0.65 0.0 1.5  0.02
#pragma parameter swirl   "Smoke: heat haze"   0.5  0.0 2.0  0.02
#pragma parameter sscale  "Smoke: wisp size"   2.5  0.5 10.0 0.1
#pragma parameter speed   "Smoke: rise speed"  1.0  0.0 4.0  0.05
#pragma parameter shade   "Smoke: soot"        0.35 0.0 1.0  0.02

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
        p = p * 2.11 + 13.7;
        a *= 0.5;
    }
    return v;
}

void main()
{
    vec2 uv = vTexCoord;
    float t = float(params.FrameCount) * (1.0 / 60.0) * params.speed;
    float aspect = params.SourceSize.x / params.SourceSize.y;
    vec2 p = uv * vec2(aspect, 1.0) * params.sscale;

    // Rising domain-warped smoke: the warp field itself climbs (row 0 = top,
    // so wisps rise by adding to y over time) and stirs the wisp lookup.
    vec2 rise = vec2(0.0, t * 0.35);
    float w1 = fbm(p + rise + vec2(t * 0.05, 0.0));
    float w2 = fbm(p * 1.4 - rise * 0.7 + 5.2);
    float wisp = fbm(p + rise + 1.8 * vec2(w1, w2));
    float smoke = smoothstep(0.38, 0.85, wisp) * params.density;
    smoke = clamp(smoke, 0.0, 1.0);

    // Heat haze: refract the image under the smoke by the warp gradient.
    vec2 shim = (vec2(w1, w2) - 0.5) * params.swirl * 0.02 * (0.3 + smoke);
    vec3 col = texture(Source, uv + shim).rgb;

    // Soot: white steam -> black smoke; dense smoke also mutes saturation.
    vec3 smokec = mix(vec3(0.92), vec3(0.06), params.shade);
    float l = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, vec3(l), smoke * 0.5);
    col = mix(col, smokec, smoke);

    vec3 src = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(src, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
