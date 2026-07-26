#version 450

// security-cam — pass 0: monitor persistence.
//
// A cheap surveillance monitor has a slow tube and the camera a slow
// sensor, so movement drags a fading smear behind it. The previous frame
// decays and whichever is brighter — decayed history or the new frame —
// survives, which trails highlights without muddying the whole picture.
//
// persist = 0 is a clean passthrough into the display pass.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float persist;
} params;

#pragma parameter persist "Cam: image lag" 0.35 0.0 0.95 0.01

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

void main()
{
    vec3 src  = texture(Source, vTexCoord).rgb;
    vec3 prev = texture(PassFeedback0, vTexCoord).rgb * params.persist;
    FragColor = vec4(max(src, prev), 1.0);
}
