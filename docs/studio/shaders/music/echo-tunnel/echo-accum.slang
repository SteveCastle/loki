#version 450

// echo-tunnel, pass 0 of 2 — the corridor itself.
//
// This pass holds the tunnel and nothing else: it reads its own previous
// output (PassFeedback0), moves it one step outward, dims it, and prints
// the picture's highlights into it again. Keeping the corridor in its own
// buffer is what makes the effect stable — if the composite were fed back,
// the picture would be re-printed every frame and the whole thing would
// white out within a second.
//
// Original = the picture entering the effect (the layer's input, before
// this stack ran), so the print is always of the clean frame.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float Time;
    float bass;
    float mids;
    float air;
    float flow;
    float kick;
    float spin;
    float decay;
    float print;
    float thresh;
    float hue;
} params;

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;         // the clean picture (pass 0 in = layer in)
layout(set = 0, binding = 3) uniform sampler2D PassFeedback0;  // the corridor, last frame

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

/* Hue rotation about the luma axis: the tail changes colour without
 * changing exposure, so a long corridor rainbows instead of flickering. */
vec3 hue_rot(vec3 c, float deg)
{
    float a = radians(deg);
    float s = sin(a), k = cos(a);
    mat3 m = mat3(
        0.299 + 0.701 * k + 0.168 * s, 0.587 - 0.587 * k + 0.330 * s, 0.114 - 0.114 * k - 0.497 * s,
        0.299 - 0.299 * k - 0.328 * s, 0.587 + 0.413 * k + 0.035 * s, 0.114 - 0.114 * k + 0.292 * s,
        0.299 - 0.300 * k + 1.250 * s, 0.587 - 0.588 * k - 1.050 * s, 0.114 + 0.886 * k - 0.203 * s);
    return max(c * m, vec3(0.0));
}

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main()
{
    vec2  uv  = vTexCoord;
    float asp = params.OutputSize.x * params.OutputSize.w;

    float b = clamp(params.bass, 0.0, 1.0);
    float m = clamp(params.mids, 0.0, 1.0);
    float a = clamp(params.air,  0.0, 1.0);

    /* Where this pixel reads the corridor from. Reading INSIDE of where we
     * are is what sends the content outward: each copy is found a little
     * nearer the middle than it was drawn, so every frame it walks another
     * step toward the rim and finally off the edge. */
    float step_z = (params.flow + params.kick * b) * 0.01;
    float ang    = radians(params.spin * m);

    vec2 q = (uv - 0.5) * vec2(asp, 1.0);
    q *= 1.0 - step_z;
    float ca = cos(ang), sa = sin(ang);
    q = vec2(q.x * ca - q.y * sa, q.x * sa + q.y * ca);
    vec2 puv = q / vec2(asp, 1.0) + 0.5;

    /* Past the edge there is no corridor yet — read black rather than
     * dragging the border pixel down the whole hall. */
    float inside = step(0.0, puv.x) * step(puv.x, 1.0) * step(0.0, puv.y) * step(puv.y, 1.0);
    vec3  prev   = texture(PassFeedback0, clamp(puv, 0.0, 1.0)).rgb * inside;

    /* decay (< 1) is the only loop gain in the effect — it is what keeps a
     * feedback buffer from running away. */
    prev *= clamp(params.decay, 0.0, 0.999);

    /* Every lap rotates the tail's hue a few more degrees, so distance down
     * the corridor reads as colour. Air drives it: the rainbow lands on the
     * cymbals, not on the kick. */
    if (params.hue > 0.0 && a > 0.0) prev = hue_rot(prev, params.hue * a);

    /* (The glitter is applied when the corridor is composited, not here: a
     * multiply inside the loop compounds every frame and beads the trails.) */

    /* What gets printed: the picture's highlights only, so the corridor is
     * built out of light rather than out of grey. A bass hit prints hard —
     * that flash is the head of a new copy. */
    vec3  src   = texture(Source, uv).rgb;
    float lv    = luma(src);
    float mask  = smoothstep(params.thresh, min(params.thresh + 0.35, 1.0), lv);

    /* Printing the same thing every frame into a buffer that only loses
     * `1 - decay` of itself per frame settles at print/(1 - decay) — which
     * is 14× at a trail life of 0.93, i.e. white. Scaling the print by that
     * same factor makes `print` mean what it says (how bright the corridor
     * gets) and makes it independent of how long the trail lives. */
    float loss  = 1.0 - clamp(params.decay, 0.0, 0.999);
    vec3  stamp = src * mask * (params.print * (0.35 + 1.65 * b) * loss);

    FragColor = vec4(clamp(prev + stamp, 0.0, 8.0), 1.0);
}
