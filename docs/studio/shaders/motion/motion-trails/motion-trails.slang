#version 450

// motion-trails — WILD motion-reactive smear with RGB tearing.
//
// Pass 1 of 2. Source = current frame (from passthrough); PassFeedback0 =
// previous frame (motion delta); PassFeedback1 = this pass's own previous
// output (the trail accumulator).
//
// Where the frame moves, we (1) estimate a flow direction from the temporal-
// difference gradient, (2) drag the trail buffer ALONG that direction so trails
// streak instead of sitting still, (3) tear the R/G/B channels apart along the
// flow so fast motion rips into chromatic ghosts, and (4) hue-cycle the
// accumulator so the streaks rainbow. Still areas stay sharp; the more it
// moves, the more it shreds. amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float sensitivity;
    float gamma;
    float decay;
    float smear_len;
    float tear;
    float hue_drift;
    float sat;
} params;

#pragma parameter amount      "MotionTrails: amount"          1.0  0.0  1.0   0.01
#pragma parameter sensitivity "MotionTrails: sensitivity"     10.0 0.5 40.0   0.5
#pragma parameter gamma       "MotionTrails: response curve"  0.6  0.2  3.0   0.05
#pragma parameter decay       "MotionTrails: trail life"      0.94 0.70 0.995 0.005
#pragma parameter smear_len   "MotionTrails: streak length"   0.07 0.0  0.25  0.005
#pragma parameter tear        "MotionTrails: RGB tear"        0.03 0.0  0.12  0.002
#pragma parameter hue_drift   "MotionTrails: rainbow speed"   6.0  0.0  30.0  0.1
#pragma parameter sat         "MotionTrails: tear saturation" 1.6  0.0  4.0   0.05

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

// Signed temporal luminance change at p (how much brighter than last frame).
float sdiff(vec2 p)
{
    return luma(texture(Source, p).rgb) - luma(texture(PassFeedback0, p).rgb);
}

vec3 hue_shift(vec3 col, float a)
{
    const vec3 k = vec3(0.57735027);
    float c = cos(a), s = sin(a);
    return col * c + cross(k, col) * s + k * dot(k, col) * (1.0 - c);
}

void main()
{
    vec2 uv = vTexCoord;
    vec2 px = params.SourceSize.zw;

    vec3 cur  = texture(Source, uv).rgb;
    vec3 prev = texture(PassFeedback0, uv).rgb;

    /* Motion magnitude → [0,1], with a cold-start guard. */
    float warm = step(2.0, float(params.FrameCount));
    float m = pow(clamp(length(cur - prev) * params.sensitivity, 0.0, 1.0), params.gamma) * warm;

    /* Cheap flow direction = gradient of the temporal-difference field. */
    float gx = sdiff(uv + vec2(px.x, 0.0)) - sdiff(uv - vec2(px.x, 0.0));
    float gy = sdiff(uv + vec2(0.0, px.y)) - sdiff(uv - vec2(0.0, px.y));
    vec2  g = vec2(gx, gy);
    float gl = length(g);
    vec2  dir = (gl > 1e-4) ? g / gl : vec2(0.0, -1.0);

    /* Drag the trail buffer along the flow so trails streak and accumulate. */
    vec2 smear = dir * params.smear_len * m;
    vec2 tv    = dir * params.tear * m;          /* channel-tear offset */

    vec3 trail;
    trail.r = texture(PassFeedback1, uv - smear + tv).r;
    trail.g = texture(PassFeedback1, uv - smear).g;
    trail.b = texture(PassFeedback1, uv - smear - tv).b;
    /* decay (< 1) is the ONLY loop gain — keeps the feedback bounded. */
    trail = clamp(hue_shift(trail, radians(params.hue_drift) * m) * params.decay, 0.0, 1.0);

    /* Live frame also tears its channels apart during motion. */
    vec3 live;
    live.r = texture(Source, clamp(uv + tv * 1.6, 0.0, 1.0)).r;
    live.g = cur.g;
    live.b = texture(Source, clamp(uv - tv * 1.6, 0.0, 1.0)).b;
    vec3 base = mix(cur, live, m);

    /* Max-blend the ghost trails so bright areas (white sheets, sky) stay
     * intact instead of washing out — trails only show where they out-shine
     * the live frame. */
    vec3 outc = max(base, trail);

    /* Crank saturation where there's motion so the chromatic tearing reads as
     * vivid neon streaks rather than pale haze. Pivoting around luma keeps it
     * bounded — colours push to the gamut edge, never to white. */
    float l = luma(outc);
    outc = clamp(mix(vec3(l), outc, 1.0 + params.sat * m), 0.0, 1.0);

    outc = mix(cur, outc, params.amount);
    FragColor = vec4(clamp(outc, 0.0, 1.0), 1.0);
}
