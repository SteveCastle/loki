#version 450

// directional-blur — motion-blur streak along an angle. 33 taps spread
// symmetrically over `dist` pixels in the chosen direction; keyframe the
// angle or distance for whip-pan looks. amount = 0 is a clean passthrough.

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
#pragma parameter dist   "Directional: distance (px)" 24.0 0.0 300.0 1.0

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

void main()
{
    vec2 uv = vTexCoord;
    vec3 base = texture(Source, uv).rgb;

    float a = radians(params.angle);
    vec2 dir = vec2(cos(a), sin(a)) * params.dist * params.SourceSize.zw;

    vec3 sum = vec3(0.0);
    for (int i = -16; i <= 16; ++i) {
        float t = float(i) / 16.0;                  // -1..1 across the streak
        sum += texture(Source, uv + dir * (0.5 * t)).rgb;
    }
    vec3 c = sum / 33.0;

    FragColor = vec4(mix(base, c, params.amount), 1.0);
}
