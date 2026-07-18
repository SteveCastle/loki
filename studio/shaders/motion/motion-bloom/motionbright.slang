#version 450

// motion-bloom pass 1 — motion bright-pass. Emits the current colour only where
// the frame is MOVING (temporal difference vs PassFeedback0), so the bloom that
// follows lights up around motion. Half resolution for a cheap wide glow.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float sensitivity;
    float gamma;
} params;

#pragma parameter sensitivity "MotionBloom: sensitivity"   12.0 0.5 40.0 0.5
#pragma parameter gamma       "MotionBloom: response"      0.7  0.2 3.0  0.05

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

void main()
{
    vec2 uv = vTexCoord;
    vec3 cur  = texture(Source, uv).rgb;
    vec3 prev = texture(PassFeedback0, uv).rgb;
    float warm = step(2.0, float(params.FrameCount));
    float m = pow(clamp(length(cur - prev) * params.sensitivity, 0.0, 1.0),
                  params.gamma) * warm;
    FragColor = vec4(cur * m, 1.0);
}
