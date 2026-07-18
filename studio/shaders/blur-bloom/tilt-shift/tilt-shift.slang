#version 450

// tilt-shift pass 2 — selective focus. Keeps a sharp horizontal band (the
// clean Original) around `focus`, fading into the blurred Source above and
// below it, like a tilt-shift lens that miniaturises a scene. A small extra
// saturation in the focus band sells the toy look. amount = 0 is a clean
// passthrough of the Original.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float focus;
    float band;
    float softness;
    float pop;
} params;

#pragma parameter amount   "TiltShift: mix"          1.0  0.0  1.0  0.01
#pragma parameter focus    "TiltShift: focus y"      0.5  0.0  1.0  0.01
#pragma parameter band     "TiltShift: focus height" 0.16 0.02 0.6  0.01
#pragma parameter softness "TiltShift: falloff"      0.18 0.02 0.6  0.01
#pragma parameter pop      "TiltShift: focus pop"    0.4  0.0  1.5  0.05

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;     // blurred
layout(set = 0, binding = 3) uniform sampler2D Original;   // sharp

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main()
{
    vec2 uv = vTexCoord;
    vec3 sharp = texture(Original, uv).rgb;
    vec3 blur  = texture(Source, uv).rgb;

    // 0 inside the band, ramping to 1 (full blur) away from focus.
    float d = abs(uv.y - params.focus);
    float m = smoothstep(params.band, params.band + params.softness, d);

    vec3 col = mix(sharp, blur, m);

    // Saturation pop weighted toward the in-focus band.
    float l = luma(col);
    col = clamp(mix(vec3(l), col, 1.0 + params.pop * (1.0 - m)), 0.0, 1.0);

    FragColor = vec4(mix(sharp, col, params.amount), 1.0);
}
