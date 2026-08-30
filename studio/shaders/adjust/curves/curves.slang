#version 450

// curves — parametric tone curve, single pass.
//
// The Lightroom-style region curve rather than free control points: five
// overlapping regions (blacks, shadows, midtones, highlights, whites) each
// push their part of the range up or down with smooth falloff, so the
// result is always a well-behaved curve. Per-channel gamma tilts underneath
// give the classic "curves on the red channel" colour cast moves.
// Neutral by default; amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float blacks;
    float shadows;
    float mids;
    float highlights;
    float whites;
    float r_gamma;
    float g_gamma;
    float b_gamma;
} params;

#pragma parameter amount     "Curves: mix"       1.0 0.0  1.0 0.01
#pragma parameter blacks     "Blacks"            0.0 -1.0 1.0 0.01
#pragma parameter shadows    "Shadows"           0.0 -1.0 1.0 0.01
#pragma parameter mids       "Midtones"          0.0 -1.0 1.0 0.01
#pragma parameter highlights "Highlights"        0.0 -1.0 1.0 0.01
#pragma parameter whites     "Whites"            0.0 -1.0 1.0 0.01
#pragma parameter r_gamma    "Red gamma"         1.0 0.25 3.0 0.01
#pragma parameter g_gamma    "Green gamma"       1.0 0.25 3.0 0.01
#pragma parameter b_gamma    "Blue gamma"        1.0 0.25 3.0 0.01

/* Colour-only: a curve that darkens must not thin the coverage matte. */
#pragma matte_ignore amount

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

float bump(float x, float c, float w)
{
    float d = (x - c) / w;
    return exp(-d * d);
}

/* The five-region curve: each control nudges its band, gaussian-weighted
 * so neighbouring regions blend; blacks/whites use hard-ended falloffs so
 * the extremes actually reach 0 and 1. */
float tone(float x)
{
    float y = x;
    y += params.blacks     * 0.30 * pow(1.0 - x, 4.0);
    y += params.shadows    * 0.25 * bump(x, 0.22, 0.18);
    y += params.mids       * 0.25 * bump(x, 0.50, 0.22);
    y += params.highlights * 0.25 * bump(x, 0.78, 0.18);
    y += params.whites     * 0.30 * pow(x, 4.0);
    return clamp(y, 0.0, 1.0);
}

void main()
{
    vec3 base = texture(Source, vTexCoord).rgb;

    vec3 c = pow(clamp(base, 0.0, 1.0),
                 1.0 / vec3(params.r_gamma, params.g_gamma, params.b_gamma));
    c = vec3(tone(c.r), tone(c.g), tone(c.b));

    FragColor = vec4(mix(base, c, params.amount), 1.0);
}
