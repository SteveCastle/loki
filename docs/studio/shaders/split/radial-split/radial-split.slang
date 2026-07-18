#version 450

// radial-split — rotational chromatic split, single pass.
//
// Instead of pushing channels apart linearly, red and blue are rotated in
// opposite directions around the frame centre, growing with radius — the
// centre stays registered while edges twist into RGB pinwheels.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float rot_rp;
    float falloff_rp;
} params;

#pragma parameter amount     "Radial split: mix"           1.0 0.0 1.0  0.01
#pragma parameter rot_rp     "Radial split: rotate (deg)"  2.5 0.0 15.0 0.1
#pragma parameter falloff_rp "Radial split: edge falloff"  1.5 0.5 4.0  0.1

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

vec2 rot_about_center(vec2 uv, float a, vec2 aspect)
{
    vec2 v = (uv - 0.5) * aspect;
    float c = cos(a), s = sin(a);
    v = vec2(v.x * c - v.y * s, v.x * s + v.y * c);
    return v / aspect + 0.5;
}

void main()
{
    // Aspect-corrected radius so the twist is circular on any frame.
    vec2 aspect = vec2(params.OutputSize.x * params.OutputSize.w, 1.0);
    vec2 v = (vTexCoord - 0.5) * aspect;
    float r = clamp(length(v) * 2.0, 0.0, 1.0);
    float a = radians(params.rot_rp) * pow(r, params.falloff_rp);

    float cr = texture(Source, rot_about_center(vTexCoord, a, aspect)).r;
    float cg = texture(Source, vTexCoord).g;
    float cb = texture(Source, rot_about_center(vTexCoord, -a, aspect)).b;

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, vec3(cr, cg, cb), params.amount), 1.0);
}
