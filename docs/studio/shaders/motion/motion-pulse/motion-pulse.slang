#version 450

// motion-pulse — full-frame pulse driven by prevailing inter-frame motion.
//
// Pass 1 of 2. Source = the current frame (from the passthrough pass);
// PassFeedback0 = the previous frame. The average frame-to-frame difference
// over a tap grid is the "prevailing motion"; it punches the image inward,
// flashes it brighter, and snaps the RGB channels apart — so the look pulses
// with how much the footage is moving, with no clock. The editor cuts to taste.
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
    float zoom_punch;
    float flash;
    float rgb_kick;
} params;

#pragma parameter amount      "MotionPulse: amount"        1.0  0.0  1.0  0.01
#pragma parameter sensitivity "MotionPulse: sensitivity"   6.0  0.5 30.0  0.5
#pragma parameter gamma       "MotionPulse: response curve" 0.7 0.2  3.0  0.05
#pragma parameter zoom_punch  "MotionPulse: zoom punch"    0.10 0.0  0.6  0.01
#pragma parameter flash       "MotionPulse: flash"         0.5  0.0  3.0  0.05
#pragma parameter rgb_kick    "MotionPulse: RGB snap"      0.5  0.0  3.0  0.05

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

#define TAPS 6

// Average frame-to-frame difference over a coarse grid = prevailing motion.
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

    /* Prevailing motion → pulse in [0,1]. Zeroed for the first frames while the
     * feedback buffer is cold. */
    float warm  = step(2.0, float(params.FrameCount));
    float gm    = prevailing_motion() * params.sensitivity * warm;
    float pulse = clamp(pow(clamp(gm, 0.0, 1.0), params.gamma), 0.0, 1.0);

    /* Zoom punch toward centre. */
    vec2 uvz = (uv - 0.5) / (1.0 + params.zoom_punch * pulse) + 0.5;

    /* RGB snap. */
    float ca = params.rgb_kick * params.SourceSize.z * 10.0 * pulse * params.amount;
    vec3 col;
    col.r = texture(Source, clamp(uvz + vec2(ca, 0.0), 0.0, 1.0)).r;
    col.g = texture(Source, clamp(uvz, 0.0, 1.0)).g;
    col.b = texture(Source, clamp(uvz - vec2(ca, 0.0), 0.0, 1.0)).b;

    /* Flash. */
    col *= 1.0 + params.flash * pulse * params.amount;

    vec3 clean = texture(Source, uv).rgb;
    col = mix(clean, col, params.amount);
    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
