#version 450

// stn-lcd — pass 0: pixel response.
//
// Passive-matrix STN cells took 150–300ms to settle, so anything moving
// dragged a translucent smear behind it (the "mouse submarine" era). The
// panel is quantized to its native cell grid here, then each cell chases
// the incoming value through PassFeedback0 — ghosting lives at panel
// resolution, exactly like the hardware. Pass 1 draws the panel itself.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float res_x;
    float persistence;
} params;

#pragma parameter res_x       "STN: panel columns"    320.0 80.0 1024.0 16.0
#pragma parameter persistence "STN: response smear"   0.65  0.0  0.95   0.01

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
    vec2 uv = vTexCoord;

    /* Native cell grid: res_x columns, rows follow the frame aspect. */
    vec2 panel = vec2(params.res_x,
                      max(params.res_x * params.SourceSize.y / params.SourceSize.x, 1.0));
    vec2 cuv = (floor(uv * panel) + 0.5) / panel;

    vec3 cur  = texture(Source, cuv).rgb;
    vec3 prev = texture(PassFeedback0, cuv).rgb;

    /* Slow liquid crystal: each frame only moves part way to the target.
     * Rising and falling at slightly different rates, like real cells. */
    float k = params.persistence;
    vec3  rate = mix(vec3(1.0 - k), vec3(1.0 - k) * 0.7, step(cur, prev));
    vec3  outc = mix(prev, cur, clamp(rate, 0.02, 1.0));

    /* Cold start: snap to the source so the first frames aren't black. */
    outc = mix(outc, cur, 1.0 - step(4.0, float(params.FrameCount)));

    FragColor = vec4(outc, 1.0);
}
