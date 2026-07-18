#version 450

// perlin-flow — a motion-reactive liquid-Perlin melt.
//
// Pass 1 of 2. Source = current frame (from passthrough); PassFeedback0 =
// previous frame (motion delta); PassFeedback1 = this pass's own previous
// output (the flowing accumulator).
//
// The image is dissolved through an animated Perlin DOMAIN-WARP field (fbm of
// fbm — the Inigo-Quilez "warp the warp" trick) so it ripples like heat-haze
// over oil. Motion (frame difference) does three things at once: it deepens the
// warp so moving regions churn harder, it drags the feedback buffer ALONG the
// flow's curl so the melt streaks instead of sitting still, and it cross-fades
// in a psychedelic cosine-palette tint keyed to the noise field. Still areas
// stay readable; the more it moves, the more it boils. amount = 0 is a clean
// passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float scale;
    float warp;
    float motion_warp;
    float speed;
    float decay;
    float flow;
    float trip;
    float sensitivity;
} params;

#pragma parameter amount       "PerlinFlow: mix"               1.0   0.0  1.0   0.01
#pragma parameter scale        "PerlinFlow: noise scale"       3.0   0.5  12.0  0.1
#pragma parameter warp         "PerlinFlow: base warp"         0.06  0.0  0.30  0.005
#pragma parameter motion_warp  "PerlinFlow: motion warp"       0.14  0.0  0.50  0.005
#pragma parameter speed        "PerlinFlow: flow speed"        0.30  0.0  2.0   0.01
#pragma parameter decay        "PerlinFlow: melt trail life"   0.90  0.70 0.99  0.005
#pragma parameter flow         "PerlinFlow: streak length"     0.05  0.0  0.20  0.005
#pragma parameter trip         "PerlinFlow: palette tint"      0.55  0.0  1.5   0.01
#pragma parameter sensitivity  "PerlinFlow: motion gain"       9.0   0.5  40.0  0.5

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;        // current frame
layout(set = 0, binding = 3) uniform sampler2D PassFeedback0; // previous frame
layout(set = 0, binding = 4) uniform sampler2D PassFeedback1; // previous output

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// --- Perlin gradient noise (quintic-smoothed) -------------------------------
vec2 grad(vec2 p)
{
    // pseudo-random unit-ish gradient per lattice point
    float n = sin(dot(p, vec2(127.1, 311.7))) * 43758.5453;
    return fract(vec2(n, n * 1.7)) * 2.0 - 1.0;
}

float pnoise(vec2 p)
{
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);   // quintic fade
    float a = dot(grad(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0));
    float b = dot(grad(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
    float c = dot(grad(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
    float d = dot(grad(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 0.5 + 0.5;
}

float fbm(vec2 p)
{
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { s += a * pnoise(p); p = p * 2.02 + 11.3; a *= 0.5; }
    return s;
}

// IQ cosine palette — cheap, smooth, fully saturated psychedelia.
vec3 palette(float t)
{
    return 0.5 + 0.5 * cos(6.28318 * (vec3(1.0, 1.0, 1.0) * t
                          + vec3(0.0, 0.33, 0.67)));
}

void main()
{
    vec2 uv  = vTexCoord;
    float t  = float(params.FrameCount) * params.speed * 0.1;
    float asp = params.SourceSize.x / max(1.0, params.SourceSize.y);

    // Motion magnitude in [0,1] with a cold-start guard.
    vec3 cur  = texture(Source, uv).rgb;
    vec3 prev = texture(PassFeedback0, uv).rgb;
    float warm = step(2.0, float(params.FrameCount));
    float m = clamp(length(cur - prev) * params.sensitivity, 0.0, 1.0) * warm;

    // --- Domain warp: fbm of fbm, animated, aspect-corrected ---------------
    vec2 np = vec2(uv.x * asp, uv.y) * params.scale;
    vec2 q = vec2(fbm(np + vec2(0.0, 0.0) + t),
                  fbm(np + vec2(5.2, 1.3) - t));
    vec2 r = vec2(fbm(np + 4.0 * q + vec2(1.7, 9.2) + 0.7 * t),
                  fbm(np + 4.0 * q + vec2(8.3, 2.8) - 0.5 * t));
    float field = fbm(np + 4.0 * r);          // scalar field for colour/hue

    // Warp amount grows where the frame is moving.
    float wamt = params.warp + params.motion_warp * m;
    vec2 disp = (r - 0.5) * wamt;             // screen-space displacement

    // Curl-ish flow direction (perpendicular to the field gradient) to drag
    // the feedback buffer so the melt streaks along the swirl.
    vec2 fdir = normalize(vec2(q.y - r.y, r.x - q.x) + 1e-4);

    // --- Chromatic melt: split RGB across the warp so edges prism --------
    vec3 melt;
    melt.r = texture(Source, uv + disp * 1.10).r;
    melt.g = texture(Source, uv + disp).g;
    melt.b = texture(Source, uv + disp * 0.90).b;

    // Psychedelic palette tint keyed to the noise field + time + motion.
    vec3 pal = palette(field + 0.15 * t + 0.4 * m);
    float tint = params.trip * (0.35 + 0.65 * m);
    vec3 col = mix(melt, melt * (0.5 + 1.5 * pal), clamp(tint, 0.0, 1.0));

    // --- Flowing feedback melt -------------------------------------------
    // Drag last frame's output along the curl; decay (<1) bounds the loop.
    vec2 sv = fdir * params.flow * (0.4 + m);
    vec3 ghost = texture(PassFeedback1, uv - sv).rgb * params.decay;
    ghost = mix(ghost, ghost.gbr, 0.25 * m);  // slow hue rotation of the trail
    col = max(col, ghost);                     // bright melt wins, stays clean

    // Push saturation where it moves so the warp reads as molten neon.
    float l = luma(col);
    col = clamp(mix(vec3(l), col, 1.0 + 1.2 * m), 0.0, 1.0);

    col = mix(cur, col, params.amount);
    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
