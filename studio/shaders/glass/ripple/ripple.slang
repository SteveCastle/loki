#version 450

// ripple — reeded ("fluted") architectural glass: parallel cylindrical reeds
// refract the image in bands, with a specular sheen running along each reed's
// crest. angle turns the flutes (0 = vertical reeds). Single pass; amount = 0
// is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float reeds;
    float refract_;
    float angle;
    float sheen;
} params;

#pragma parameter amount   "RippleGlass: mix"        1.0  0.0  1.0   0.01
#pragma parameter reeds    "RippleGlass: reeds"      28.0 4.0  120.0 1.0
#pragma parameter refract_ "RippleGlass: refraction" 0.7  0.0  2.0   0.02
#pragma parameter angle    "RippleGlass: angle"      0.0  0.0  180.0 1.0
#pragma parameter sheen    "RippleGlass: sheen"      0.4  0.0  1.0   0.02

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
    float aspect = params.SourceSize.x / params.SourceSize.y;
    float a = radians(params.angle);
    vec2 dir = vec2(cos(a), sin(a));   // across the reeds

    // Position across the flutes, aspect-corrected so reeds stay parallel.
    float s = dot(uv * vec2(aspect, 1.0), dir) * params.reeds * 6.2831853;

    // A cylinder bends light toward its centreline: offset along `dir` by
    // the reed profile's slope (cos), strongest at the flanks.
    float slope = cos(s);
    vec2 off = dir * slope * params.refract_ * (0.35 / max(params.reeds, 1.0));
    vec3 col = texture(Source, uv + off * vec2(1.0 / aspect, 1.0)).rgb;

    // Specular streak on each crest + a faint shadow in each valley.
    float crest = pow(max(sin(s), 0.0), 24.0);
    float valley = pow(max(-sin(s), 0.0), 6.0);
    col += vec3(1.0) * crest * params.sheen * 0.35;
    col *= 1.0 - valley * params.sheen * 0.18;

    vec3 src = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(src, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
