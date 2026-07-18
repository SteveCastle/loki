#version 450

// halftone — CMY newsprint. Each subtractive ink (cyan, magenta, yellow) is
// printed as a grid of dots on its own rotated screen; dot size grows with ink
// coverage, and the offset angles form the classic rosette. The result is a
// comic / newspaper print. Single pass. amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float dot_size;
    float angle;
    float sharp;
} params;

#pragma parameter amount   "Halftone: mix"          1.0  0.0  1.0   0.01
#pragma parameter dot_size "Halftone: dot size px"  10.0 2.0  24.0  0.5
#pragma parameter angle    "Halftone: screen angle" 15.0 0.0  90.0  1.0
#pragma parameter sharp    "Halftone: dot edge"     0.12 0.02 0.40  0.01

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

// One rotated halftone screen: 1.0 inside the ink dot, 0.0 outside.
float screen(vec2 px, float ang, float ink, float cell)
{
    float s = sin(ang), c = cos(ang);
    vec2 p = mat2(c, -s, s, c) * px;
    vec2 g = mod(p, cell) - 0.5 * cell;
    float d = length(g) / (0.5 * cell);
    float radius = sqrt(clamp(ink, 0.0, 1.0));   // dot area ~ ink coverage
    return 1.0 - smoothstep(radius - params.sharp, radius + params.sharp, d);
}

void main()
{
    vec2 uv = vTexCoord;
    vec2 px = uv * params.SourceSize.xy;         // pixel coords
    float cell = params.dot_size;
    float base = radians(params.angle);

    vec3 src = texture(Source, uv).rgb;
    float cI = screen(px, base + 0.2618, 1.0 - src.r, cell);   // cyan   (+15)
    float mI = screen(px, base + 1.3090, 1.0 - src.g, cell);   // magenta(+75)
    float yI = screen(px, base + 0.0,    1.0 - src.b, cell);   // yellow ( 0)

    vec3 col = vec3(1.0 - cI, 1.0 - mI, 1.0 - yI);

    FragColor = vec4(mix(src, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
