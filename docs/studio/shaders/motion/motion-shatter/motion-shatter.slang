#version 450

// motion-shatter — glitch that triggers only where the frame is moving.
//
// Pass 1 of 2. Source = current frame (from passthrough); PassFeedback0 =
// previous frame. Local motion (per macroblock) shatters that block — it jumps
// to a random offset and the RGB channels split — while still areas stay clean.
// The prevailing (global) motion adds a baseline so the whole frame reacts to
// big movements. Like datamosh, but driven by real inter-frame motion instead
// of a random timer; the editor controls when it fires by cutting to movement.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float sensitivity;
    float gamma;
    float block_size;
    float displace;
    float rgb_split;
    float global_boost;
    float seed;
} params;

#pragma parameter amount       "MotionShatter: amount"        1.0  0.0  1.0   0.01
#pragma parameter sensitivity  "MotionShatter: sensitivity"   10.0 0.5 40.0   0.5
#pragma parameter gamma        "MotionShatter: response curve" 0.8 0.2  3.0   0.05
#pragma parameter block_size   "MotionShatter: block px"      14.0 4.0 64.0   1.0
#pragma parameter displace     "MotionShatter: shatter dist"  0.06 0.0  0.3   0.005
#pragma parameter rgb_split    "MotionShatter: RGB split"     2.0  0.0  8.0   0.1
#pragma parameter global_boost "MotionShatter: global react"  0.4  0.0  2.0   0.05
#pragma parameter seed         "MotionShatter: seed"          17.0 0.0 64.0   1.0

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

float hash1(float n) { return fract(sin(n * 17.3217) * 43758.5453); }

#define TAPS 6
float prevailing_motion()
{
    float m = 0.0;
    for (int i = 0; i < TAPS; ++i)
    for (int j = 0; j < TAPS; ++j)
    {
        vec2 tp = (vec2(float(i), float(j)) + 0.5) / float(TAPS);
        m += length(texture(Source, tp).rgb - texture(PassFeedback0, tp).rgb);
    }
    return m / float(TAPS * TAPS);
}

void main()
{
    vec2 uv = vTexCoord;
    float warm = step(2.0, float(params.FrameCount));

    /* Macroblock grid. */
    vec2 res     = params.SourceSize.xy;
    vec2 bpx     = vec2(max(params.block_size, 1.0));
    vec2 grid    = res / bpx;
    vec2 block   = floor(uv * grid);
    vec2 bcenter = (block + 0.5) / grid;

    /* Per-block local motion + global baseline → shatter energy. */
    float mLoc = length(texture(Source, bcenter).rgb - texture(PassFeedback0, bcenter).rgb);
    mLoc = clamp(mLoc * params.sensitivity, 0.0, 1.0);
    float gm = prevailing_motion() * params.sensitivity * params.global_boost;
    float energy = clamp(pow(clamp(mLoc + gm, 0.0, 1.0), params.gamma), 0.0, 1.0) * warm * params.amount;

    /* Shatter: block jumps to a random offset, scaled by motion energy. */
    vec2 jump = (vec2(hash1(dot(block, vec2(1.0, 57.0)) + params.seed),
                      hash1(dot(block, vec2(31.0, 7.0)) + params.seed)) - 0.5)
                * 2.0 * params.displace * energy;
    vec2 suv = uv + jump;

    /* RGB split scaled by energy. */
    float ca = params.rgb_split * params.SourceSize.z * (1.0 + energy * 6.0) * energy;
    vec3 col;
    col.r = texture(Source, clamp(suv + vec2(ca, 0.0), 0.0, 1.0)).r;
    col.g = texture(Source, clamp(suv, 0.0, 1.0)).g;
    col.b = texture(Source, clamp(suv - vec2(ca, 0.0), 0.0, 1.0)).b;

    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
