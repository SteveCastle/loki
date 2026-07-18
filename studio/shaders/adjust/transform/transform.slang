#version 450

// transform — zoom / pan / rotate / flip for composing layers (crop a
// source, punch in, mirror). Rotation is aspect-corrected; out-of-frame
// areas go black. Neutral by default. amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float zoom;
    float pan_x;
    float pan_y;
    float rot_deg;
    float flip_h;
    float flip_v;
} params;

#pragma parameter amount       "Transform: mix" 1.0 0.0 1.0 0.01
#pragma parameter zoom         "Zoom" 1.0 0.25 4.0 0.01
#pragma parameter pan_x        "Pan X" 0.0 -1.0 1.0 0.005
#pragma parameter pan_y        "Pan Y" 0.0 -1.0 1.0 0.005
#pragma parameter rot_deg      "Rotate (deg)" 0.0 -180.0 180.0 0.5
#pragma parameter flip_h       "Flip horizontal" 0.0 0.0 1.0 1.0
#pragma parameter flip_v       "Flip vertical" 0.0 0.0 1.0 1.0

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
    float aspect = params.OutputSize.x * params.OutputSize.w;
    vec2 pp = (uv - 0.5) * vec2(aspect, 1.0);
    float a = radians(-params.rot_deg);
    float ca = cos(a), sa = sin(a);
    pp = vec2(pp.x * ca - pp.y * sa, pp.x * sa + pp.y * ca) / params.zoom;
    vec2 suv = pp / vec2(aspect, 1.0) + 0.5
             - vec2(params.pan_x, params.pan_y);
    if (params.flip_h > 0.5) suv.x = 1.0 - suv.x;
    if (params.flip_v > 0.5) suv.y = 1.0 - suv.y;
    vec3 c = texture(Source, suv).rgb;
    if (any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0))))
        c = vec3(0.0);
    FragColor = vec4(mix(base, c, params.amount), 1.0);
}
