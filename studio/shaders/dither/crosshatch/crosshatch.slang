#version 450

// crosshatch — pen-and-ink hatching, single pass.
//
// Tone builds the way an illustrator does it: one diagonal hatch for the
// midtones, the opposite diagonal layered in for shadows, then horizontal
// and vertical passes for the deepest blacks. Stroke jitter and a paper
// tint sell the hand-drawn look.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float amount;
    float period_ch;
    float width_ch;
    float jitter_ch;
    float contrast_ch;
    float ink3_r;
    float ink3_g;
    float ink3_b;
    float paper3_r;
    float paper3_g;
    float paper3_b;
} params;

#pragma parameter amount      "Crosshatch: mix"          1.0  0.0 1.0  0.01
#pragma parameter period_ch   "Crosshatch: spacing (px)" 7.0  3.0 24.0 0.5
#pragma parameter width_ch    "Crosshatch: stroke width" 0.35 0.1 0.8  0.01
#pragma parameter jitter_ch   "Crosshatch: jitter"       0.6  0.0 2.0  0.05
#pragma parameter contrast_ch "Crosshatch: contrast"     1.1  0.5 2.0  0.05
#pragma parameter ink3_r      "Ink R"                    0.13 0.0 1.0  0.01
#pragma parameter ink3_g      "Ink G"                    0.11 0.0 1.0  0.01
#pragma parameter ink3_b      "Ink B"                    0.10 0.0 1.0  0.01
#pragma parameter paper3_r    "Paper R"                  0.97 0.0 1.0  0.01
#pragma parameter paper3_g    "Paper G"                  0.95 0.0 1.0  0.01
#pragma parameter paper3_b    "Paper B"                  0.88 0.0 1.0  0.01

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

float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Ink coverage of one hatch direction: stroke bands across `coord` with
// per-line jitter so strokes wander like a nib, softened at the edges.
float hatch(float coord, float along, float period)
{
    float line_id = floor(coord / period);
    float wander = (hash(vec2(line_id, 3.7)) - 0.5) * params.jitter_ch * period * 0.3
                 + sin(along * 0.02 + line_id * 17.0) * params.jitter_ch;
    float d = abs(fract((coord + wander) / period) - 0.5) * 2.0;
    return 1.0 - smoothstep(params.width_ch - 0.12, params.width_ch + 0.12, d);
}

void main()
{
    vec2 p = vTexCoord * params.OutputSize.xy;
    vec3 s = texture(Source, vTexCoord).rgb;
    float l = dot(s, vec3(0.299, 0.587, 0.114));
    l = clamp((l - 0.5) * params.contrast_ch + 0.5, 0.0, 1.0);

    float period = max(params.period_ch, 2.0);
    float ink = 0.0;
    // Layer hatches in as the tone darkens, each with a soft ramp.
    ink = max(ink, hatch(p.x + p.y, p.x - p.y, period) * smoothstep(0.85, 0.65, l));
    ink = max(ink, hatch(p.x - p.y, p.x + p.y, period) * smoothstep(0.60, 0.40, l));
    ink = max(ink, hatch(p.y, p.x, period)             * smoothstep(0.38, 0.20, l));
    ink = max(ink, hatch(p.x, p.y, period)             * smoothstep(0.18, 0.04, l));

    // Paper grain (static).
    vec3 paper = vec3(params.paper3_r, params.paper3_g, params.paper3_b)
               * (1.0 - (hash(floor(p * 0.5)) - 0.5) * 0.06);
    vec3 col = mix(paper, vec3(params.ink3_r, params.ink3_g, params.ink3_b), ink);

    FragColor = vec4(mix(s, col, params.amount), 1.0);
}
