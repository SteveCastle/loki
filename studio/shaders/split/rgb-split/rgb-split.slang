#version 450

// rgb-split — directional RGB channel separation, single pass.
//
// The classic: red and blue pull apart along an angle while green holds
// the frame. `spread` biases how far red moves versus blue, and `wiggle`
// makes the split breathe and swing over time for a live-glitch feel.
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
    float dist_rs;
    float angle_rs;
    float spread_rs;
    float wiggle_rs;
    float wspeed_rs;
} params;

#pragma parameter amount    "RGB split: mix"          1.0 0.0    1.0   0.01
#pragma parameter dist_rs   "RGB split: distance (px)" 6.0 0.0   40.0  0.5
#pragma parameter angle_rs  "RGB split: angle (deg)"  0.0 -180.0 180.0 1.0
#pragma parameter spread_rs "RGB split: R/B balance"  0.0 -1.0   1.0   0.05
#pragma parameter wiggle_rs "RGB split: wiggle"       0.0 0.0    1.0   0.01
#pragma parameter wspeed_rs "RGB split: wiggle speed" 1.5 0.0    8.0   0.05

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
    float t = params.Time * params.wspeed_rs;
    float dist = params.dist_rs * (1.0 + params.wiggle_rs * 0.6 * sin(t * 2.3));
    float ang = radians(params.angle_rs) + params.wiggle_rs * 0.6 * sin(t * 1.7);
    vec2 dir = vec2(cos(ang), sin(ang)) * dist * params.SourceSize.zw;

    float r = texture(Source, vTexCoord + dir * (1.0 + params.spread_rs)).r;
    float g = texture(Source, vTexCoord).g;
    float b = texture(Source, vTexCoord - dir * (1.0 - params.spread_rs)).b;

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, vec3(r, g, b), params.amount), 1.0);
}
