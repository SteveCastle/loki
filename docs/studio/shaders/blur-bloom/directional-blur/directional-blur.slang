#version 450

// directional-blur pass 1 — motion-blur streak along an angle, centred on
// the pixel and `dist` pixels long; keyframe the angle or distance for
// whip-pan looks. 24 taps one slice apart over pass 0's fine smear (see
// streak_fine.slang), then mixed against the clean Original by `amount`.
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float angle;
    float dist;
} params;

#pragma parameter amount "Directional: mix" 1.0 0.0 1.0 0.01
#pragma parameter angle  "Directional: angle (deg)" 0.0 0.0 360.0 1.0
#pragma parameter dist   "Directional: distance (px)" 24.0 0.0 600.0 1.0

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;     // fine smear (float)
layout(set = 0, binding = 3) uniform sampler2D Original;   // clean input

float hash3(vec3 p)
{
    p = fract(p * vec3(123.34, 345.45, 678.56));
    p += dot(p, p.yzx + 34.345);
    return fract((p.x + p.y) * p.z);
}

const int N = 24;

void main()
{
    vec2 uv = vTexCoord;
    float a = radians(params.angle);
    vec2 dir = vec2(cos(a), sin(a)) * params.dist * params.SourceSize.zw;

    vec3 sum = vec3(0.0);
    for (int i = 0; i < N; ++i) {
        float t = (float(i) + 0.5) / float(N) - 0.5;     // -0.5..0.5 across the streak
        sum += textureLod(Source, uv + dir * t, 0.0).rgb;
    }
    vec3 c = sum / float(N);

    // TPDF dither (see gaussian-blur): long smears are smooth ramps.
    vec2 px = uv * params.OutputSize.xy;
    float fr = float(params.FrameCount % 1024u);
    c += (hash3(vec3(px, fr)) + hash3(vec3(px + 917.0, fr)) - 1.0) * (1.0 / 255.0);

    vec3 base = textureLod(Original, uv, 0.0).rgb;
    FragColor = vec4(mix(base, clamp(c, 0.0, 1.0), params.amount), 1.0);
}
