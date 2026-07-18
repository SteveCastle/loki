#version 450

// feedback-echo — analog video-feedback art (the "camera pointed at its own
// monitor" look), single self-feedback pass.
//
// Each frame we take the previous output, zoom + rotate it slightly about the
// centre, rotate its hue, and dim it, then max-blend the live frame on top.
// Bright/moving content seeds echoes that spiral inward (or outward) and cycle
// through the spectrum — infinite rainbow tunnels that trail motion. Because
// the hue rotation compounds through the feedback loop, the trails rainbow on
// their own.
//
// zoom > 1 pulls echoes inward (tunnel); < 1 pushes them outward (bloom).
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float persistence;
    float zoom;
    float rotate;
    float decay;
    float hue_rate;
    float speed;
} params;

#pragma parameter amount      "Echo: amount"             0.95 0.0  1.0   0.01
#pragma parameter persistence "Echo: trail persistence"  0.72 0.0  0.97  0.01
#pragma parameter zoom        "Echo: zoom per frame"     1.02 0.95 1.06  0.001
#pragma parameter rotate      "Echo: rotate (deg/frame)" 0.6  -5.0 5.0   0.05
#pragma parameter decay       "Echo: trail decay"        0.98 0.80 0.999 0.005
#pragma parameter hue_rate    "Echo: hue cycle"          4.0  0.0 20.0   0.1
#pragma parameter speed       "Echo: speed"              1.0  0.0  4.0   0.05

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

// Hue rotation about the RGB grey axis (Rodrigues rotation).
vec3 hue_shift(vec3 col, float a)
{
    const vec3 k = vec3(0.57735027);
    float c = cos(a), s = sin(a);
    return col * c + cross(k, col) * s + k * dot(k, col) * (1.0 - c);
}

void main()
{
    vec2  uv = vTexCoord;
    float sp = params.speed;

    /* Zoom + rotate the sampling coordinate about the centre. */
    float ang = radians(params.rotate) * sp;
    float cs = cos(ang), sn = sin(ang);
    mat2  R = mat2(cs, -sn, sn, cs);
    vec2  c = (uv - 0.5);
    c = R * (c * mix(1.0, params.zoom, sp));
    vec2 puv = c + 0.5;

    /* Previous frame, hue-rotated and dimmed → the echo. The hue rotation
     * compounds through the loop, so trails cycle the spectrum on their own. */
    vec3 echo = texture(PassFeedback0, puv).rgb;
    echo = hue_shift(echo, radians(params.hue_rate) * sp);
    echo *= params.decay;

    vec3 cur = texture(Source, uv).rgb;

    /* Persistence smear: blend the live frame toward the spiralling echo, so
     * the image dissolves into hue-cycled feedback tunnels regardless of how
     * bright the source is. A max with the live frame keeps moving subjects
     * legible on top of their own trails. */
    vec3 trail = mix(cur, echo, params.persistence);
    vec3 outc  = mix(cur, max(trail, cur * 0.85), params.amount);
    FragColor = vec4(clamp(outc, 0.0, 1.0), 1.0);
}
