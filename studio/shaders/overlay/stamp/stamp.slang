#version 450

// stamp — composite an external texture over the frame, single pass.
//
// Places the `Stamp` texture (see stamp.slangp; the web tuner lets you
// swap it for any image or rendered text) at a position with size,
// rotation, opacity and a blend mode. The stamp keeps its own aspect
// ratio; size is its width as a fraction of the frame.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float pos_x;
    float pos_y;
    float size_st;
    float rot_st;
    float opacity_st;
    float blend_st;
} params;

#pragma parameter amount     "Stamp: mix"        1.0  0.0    1.0   0.01
#pragma parameter pos_x      "Stamp: x"          0.5  0.0    1.0   0.005
#pragma parameter pos_y      "Stamp: y"          0.5  0.0    1.0   0.005
#pragma parameter size_st    "Stamp: size"       0.25 0.02   1.5   0.005
#pragma parameter rot_st     "Stamp: rotate (deg)" 0.0 -180.0 180.0 1.0
#pragma parameter opacity_st "Stamp: opacity"    1.0  0.0    1.0   0.01
#pragma parameter blend_st   "Stamp: blend (0 norm 1 screen 2 mult)" 0.0 0.0 2.0 1.0

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
layout(set = 0, binding = 3) uniform sampler2D Stamp;

void main()
{
    vec3 base = texture(Source, vTexCoord).rgb;

    // Frame-space px, offset from the stamp centre, un-rotated.
    vec2 fpx = vTexCoord * params.OutputSize.xy;
    vec2 cpx = vec2(params.pos_x, params.pos_y) * params.OutputSize.xy;
    float a = radians(params.rot_st);
    vec2 d = fpx - cpx;
    d = vec2(d.x * cos(a) + d.y * sin(a), -d.x * sin(a) + d.y * cos(a));

    // Stamp extent in frame px: width from size_st, height from its aspect.
    vec2 tdim = vec2(textureSize(Stamp, 0));
    float w = params.size_st * params.OutputSize.x;
    vec2 half_ext = vec2(w, w * tdim.y / max(tdim.x, 1.0)) * 0.5;

    vec2 suv = d / (half_ext * 2.0) + 0.5;
    float inside = step(0.0, suv.x) * step(suv.x, 1.0) * step(0.0, suv.y) * step(suv.y, 1.0);
    vec4 s = texture(Stamp, clamp(suv, 0.0, 1.0));
    float alpha = s.a * inside * params.opacity_st;

    int blend = int(clamp(params.blend_st, 0.0, 2.0));
    vec3 blended =
        blend == 0 ? s.rgb :
        blend == 1 ? 1.0 - (1.0 - base) * (1.0 - s.rgb) :
                     base * s.rgb;
    vec3 col = mix(base, blended, alpha);

    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
