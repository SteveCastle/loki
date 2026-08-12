#version 450

// echo-tunnel — the frame's own highlights, printed on the beat and flying
// out of the screen.
//
// Pass 1 of 2: the composite. Source = the corridor built in pass 0,
// Original = the clean picture. The corridor is screened back over the
// picture, because it is light in the room rather than paint on the lens —
// it can only ever add.
//
// Every frame the corridor is read back a step magnified, dimmed, and the
// bright parts of the picture are printed into it again; sixty times a
// second that turns one print into a hall of copies receding behind the
// newest one. Each `bass` hit prints a bright one, so what you end up
// looking at is a record of the last few beats, spaced by the tempo you
// actually played.
//
// This is a feedback effect, so its motion is per FRAME rather than per
// second: flow, spin and trail life all scale with the comp's frame rate.
//
// The three bands drive three axes of the same corridor:
//   bass   print brightness — the flash that starts each copy — plus a
//          shove to the flow, so the hall lurches outward on the hit
//   mids   spin: the corridor twists while the mids carry the track
//   air    hue rotation and glitter on the older copies, so the tail
//          rainbows away instead of merely dimming
//
// Automate them with audio drivers (∿ on the row → Audio → band →
// **Beat** for prints you can count, **Level** for a hall that breathes).
// Trail life 0.97 with a low print gives long ghostly corridors; 0.85 with
// a hard bass kick gives exactly one sharp copy per beat.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float Time;
    float amount;
    float air;
    float glow;
} params;

#pragma parameter amount "EchoTunnel: mix"                      1.0   0.0  1.0   0.01
#pragma parameter bass   "EchoTunnel: bass hit — automate"      0.0   0.0  1.0   0.01
#pragma parameter mids   "EchoTunnel: mid hit — automate"       0.0   0.0  1.0   0.01
#pragma parameter air    "EchoTunnel: air hit — automate"       0.0   0.0  1.0   0.01
#pragma parameter flow   "EchoTunnel: flow (% per frame)"       1.6   -6.0 6.0   0.05
#pragma parameter kick   "EchoTunnel: flow kick on bass (%)"    1.2   0.0  8.0   0.05
#pragma parameter spin   "EchoTunnel: spin on mids (deg/frame)" 0.6   -6.0 6.0   0.05
#pragma parameter decay  "EchoTunnel: trail life"               0.93  0.70 0.995 0.005
#pragma parameter print  "EchoTunnel: corridor brightness"      0.90  0.0  2.0   0.01
#pragma parameter thresh "EchoTunnel: print threshold"          0.45  0.0  1.0   0.01
#pragma parameter hue    "EchoTunnel: hue drift on air (deg)"   4.0   0.0  30.0  0.1
#pragma parameter glow   "EchoTunnel: tunnel brightness"        1.0   0.0  3.0   0.05

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;    // the corridor
layout(set = 0, binding = 3) uniform sampler2D Original;  // the clean picture

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main()
{
    vec2 uv     = vTexCoord;
    vec3 src    = texture(Original, uv).rgb;
    vec3 tunnel = texture(Source, uv).rgb * params.glow;

    /* Glitter on the corridor, applied HERE rather than in the accumulator:
     * inside the feedback loop a multiply compounds every frame and beads
     * the trails into dotted lines. Applied on the way out it is what it
     * should be — light catching on the older copies. */
    if (params.air > 0.001)
    {
        float tw = hash21(floor(uv * params.OutputSize.xy / 3.0) + floor(vec2(params.Time * 24.0)));
        tunnel *= 1.0 + params.air * 1.4 * smoothstep(0.80, 1.0, tw);
    }
    tunnel = clamp(tunnel, 0.0, 1.0);

    /* Screen: light already in the frame stays put, the corridor can only
     * add to it, and nothing the corridor crosses gets darker. */
    vec3 lit = 1.0 - (1.0 - clamp(src, 0.0, 1.0)) * (1.0 - tunnel);

    FragColor = vec4(clamp(mix(src, lit, clamp(params.amount, 0.0, 1.0)), 0.0, 1.0), 1.0);
}
