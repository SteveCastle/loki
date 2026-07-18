#version 450

// channel-drift — each channel wanders its own slow path, single pass.
//
// Dreamy misregistration: R, G and B drift independently on offset
// Lissajous orbits, like a projector whose color plates never quite sit
// still. Slow speeds read as woozy VHS; fast reads as nervous glitch.
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
    float drift_cd;
    float speed_cd;
    float sway_cd;
} params;

#pragma parameter amount   "Drift: mix"          1.0 0.0 1.0  0.01
#pragma parameter drift_cd "Drift: distance (px)" 5.0 0.0 30.0 0.5
#pragma parameter speed_cd "Drift: speed"        0.6 0.0 4.0  0.05
#pragma parameter sway_cd  "Drift: sway shape"   1.0 0.2 3.0  0.05

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
    float t = params.Time * params.speed_cd;
    float k = params.sway_cd;
    vec2 px = params.drift_cd * params.SourceSize.zw;

    vec2 rOff = vec2(sin(t * 1.00 * k), cos(t * 1.31)) * px;
    vec2 gOff = vec2(sin(t * 0.73 + 2.1), cos(t * 1.11 * k + 1.3)) * px * 0.6;
    vec2 bOff = vec2(sin(t * 1.27 * k + 4.2), cos(t * 0.87 + 3.7)) * px;

    vec3 col = vec3(texture(Source, vTexCoord + rOff).r,
                    texture(Source, vTexCoord + gOff).g,
                    texture(Source, vTexCoord + bOff).b);

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
