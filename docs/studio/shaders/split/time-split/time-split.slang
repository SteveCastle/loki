#version 450

// time-split — temporal RGB split, single self-feedback pass.
//
// The channels live at different points in time: red is the live frame,
// green and blue lag behind through the feedback buffer at different
// rates. Static shots stay perfectly registered; anything that moves
// pulls apart into rainbow motion fringes with red leading.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float lag_g;
    float lag_b;
} params;

#pragma parameter amount "Time split: mix"        1.0  0.0 1.0  0.01
#pragma parameter lag_g  "Time split: green lag"  0.45 0.0 0.95 0.01
#pragma parameter lag_b  "Time split: blue lag"   0.75 0.0 0.95 0.01

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

void main()
{
    vec3 cur = texture(Source, vTexCoord).rgb;
    vec3 prev = texture(PassFeedback0, vTexCoord).rgb;

    // Each lagging channel is an exponential moving average of its own
    // history (the feedback holds this pass's previous OUTPUT, so the lag
    // compounds frame over frame).
    vec3 split = vec3(cur.r,
                      mix(cur.g, prev.g, params.lag_g),
                      mix(cur.b, prev.b, params.lag_b));

    FragColor = vec4(mix(cur, split, params.amount), 1.0);
}
