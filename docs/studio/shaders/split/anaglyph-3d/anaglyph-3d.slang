#version 450

// anaglyph-3d — pseudo 3D-glasses encoding, single pass.
//
// Fakes a stereo pair by shifting two "eyes" horizontally and encoding
// them for paper glasses: red/cyan, green/magenta or amber/blue. Depth is
// pseudo — parallax is scaled by local luma (`depth_3d`) so bright things
// pop off the screen; at 0 the whole frame floats at one depth.
// `rivalry_3d` desaturates before encoding to tame retinal rivalry.
//
// Works as a look on its own, and actually pops with the matching glasses.
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float sep_3d;
    float depth_3d;
    float mode_3d;
    float rivalry_3d;
} params;

#pragma parameter amount     "3D: mix"                  1.0 0.0 1.0  0.01
#pragma parameter sep_3d     "3D: eye separation (px)"  8.0 0.0 40.0 0.5
#pragma parameter depth_3d   "3D: luma depth"           0.5 0.0 1.0  0.01
#pragma parameter mode_3d    "3D: glasses (0 R/C 1 G/M 2 A/B)" 0.0 0.0 2.0 1.0
#pragma parameter rivalry_3d "3D: rivalry tame"         0.35 0.0 1.0 0.01

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

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

vec3 eye(vec2 uv, float dir)
{
    // Per-pixel parallax: base separation, scaled toward local brightness.
    float l = luma(texture(Source, uv).rgb);
    float px = params.sep_3d * mix(1.0, l, params.depth_3d) * 0.5;
    vec3 c = texture(Source, uv + vec2(dir * px * params.SourceSize.z, 0.0)).rgb;
    return mix(c, vec3(luma(c)), params.rivalry_3d);
}

void main()
{
    vec3 L = eye(vTexCoord, 1.0);    // left eye looks right
    vec3 R = eye(vTexCoord, -1.0);

    int mode = int(clamp(params.mode_3d, 0.0, 2.0));
    vec3 col =
        mode == 0 ? vec3(L.r, R.g, R.b) :             // red / cyan
        mode == 1 ? vec3(R.r, L.g, R.b) :             // green / magenta
                    vec3(L.r, L.g * 0.8 + L.r * 0.2, R.b);  // amber / blue

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, col, params.amount), 1.0);
}
