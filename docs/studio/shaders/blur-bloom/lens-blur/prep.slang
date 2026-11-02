#version 450

// lens-blur pass 0 — linear light + highlight weighting, into a float
// mipmapped buffer. Each pixel is stored premultiplied by its weight
// (rgb*w, w): the gather pass divides by the summed alpha, so bright
// pixels dominate any disc they fall in — that is what turns specular
// highlights into bokeh balls. Premultiplying BEFORE the mip chain is
// generated keeps the weighting correct at every level, so a tiny
// highlight still blooms after it has been averaged down.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float highlights;
    float threshold;
    float linlight;
} params;

#pragma parameter highlights "Lens: highlight boost" 0.25 0.0 1.0 0.01
#pragma parameter threshold  "Lens: highlight threshold" 0.75 0.0 1.0 0.01
#pragma parameter linlight   "Lens: linear light" 1.0 0.0 1.0 1.0
// A white silhouette is all "highlight", and gamma-expanding it widens it:
// both would grow the coverage, so the matte chain blurs it plainly.
#pragma matte_ignore highlights linlight

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

void main()
{
    vec3 c = textureLod(Source, vTexCoord, 0.0).rgb;
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));         // display-referred
    float h = smoothstep(params.threshold, 1.0, l);
    float w = 1.0 + params.highlights * 24.0 * h * h;
    vec3 lin = mix(c, pow(max(c, 0.0), vec3(2.2)), params.linlight);
    FragColor = vec4(lin * w, w);
}
