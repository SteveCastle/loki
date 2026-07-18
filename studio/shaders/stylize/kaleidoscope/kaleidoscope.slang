#version 450

// kaleidoscope — fold the frame into N mirrored wedges around the centre,
// slowly rotate the mirror, and cycle the hue. Mirror-tiled sampling keeps the
// seams seamless. Single pass. amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float segments;
    float zoom;
    float spin;
    float hue_speed;
} params;

#pragma parameter amount    "Kaleido: mix"          1.0  0.0  1.0   0.01
#pragma parameter segments  "Kaleido: segments"     8.0  2.0  24.0  1.0
#pragma parameter zoom      "Kaleido: zoom"         1.2  0.3  4.0   0.05
#pragma parameter spin      "Kaleido: spin speed"   0.20 0.0  2.0   0.01
#pragma parameter hue_speed "Kaleido: hue cycle"    0.30 0.0  2.0   0.01

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

vec3 hue_rotate(vec3 col, float a)
{
    const vec3 k = vec3(0.57735027);
    float c = cos(a), s = sin(a);
    return col * c + cross(k, col) * s + k * dot(k, col) * (1.0 - c);
}

// Triangle-wave reflect so out-of-range UVs mirror-tile seamlessly.
vec2 mirror_uv(vec2 v) { return abs(fract(v * 0.5) * 2.0 - 1.0); }

void main()
{
    vec2 uv = vTexCoord;
    float t = float(params.FrameCount) * 0.016;
    float asp = params.SourceSize.x / max(1.0, params.SourceSize.y);

    vec2 c = uv - 0.5;
    c.x *= asp;
    float r = length(c);
    float a = atan(c.y, c.x) + t * params.spin;

    float seg = 6.28318 / max(2.0, floor(params.segments));
    a = mod(a, seg);
    a = abs(a - seg * 0.5);                      // mirror inside the wedge

    vec2 p = vec2(cos(a), sin(a)) * (r / params.zoom);
    p.x /= asp;
    vec3 col = texture(Source, mirror_uv(p + 0.5)).rgb;

    col = hue_rotate(col, t * params.hue_speed);

    vec3 src = texture(Source, uv).rgb;
    FragColor = vec4(mix(src, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
