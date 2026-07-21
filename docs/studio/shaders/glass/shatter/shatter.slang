#version 450

// shatter — the frame seen through a broken pane: voronoi shards each refract
// the image with their own small offset and brightness, with bright crack
// lines between them. seed re-rolls the break pattern (keyframe it for a
// progressive smash). Single pass; amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float shards;
    float refract_;
    float cracks;
    float width;
    float seed;
} params;

#pragma parameter amount   "ShatterGlass: mix"         1.0   0.0   1.0  0.01
#pragma parameter shards   "ShatterGlass: shards"      7.0   2.0   30.0 1.0
#pragma parameter refract_ "ShatterGlass: refraction"  0.6   0.0   2.0  0.02
#pragma parameter cracks   "ShatterGlass: crack glow"  0.7   0.0   1.5  0.02
#pragma parameter width    "ShatterGlass: crack width" 0.035 0.005 0.15 0.005
#pragma parameter seed     "ShatterGlass: seed"        0.0   0.0   50.0 1.0

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

vec2 hash22(vec2 p)
{
    vec3 a = fract(p.xyx * vec3(123.34, 234.34, 345.65));
    a += dot(a, a + 34.45);
    return fract(vec2(a.x * a.y, a.y * a.z));
}

void main()
{
    vec2 uv = vTexCoord;
    float aspect = params.SourceSize.x / params.SourceSize.y;
    vec2 p = (uv * vec2(aspect, 1.0) + params.seed * 7.31) * params.shards;
    vec2 ip = floor(p);
    vec2 fp = fract(p);

    // Voronoi: nearest shard centre (F1 + its id) and second nearest (F2).
    // F2 - F1 is the distance to the crack between them.
    float f1 = 8.0;
    float f2 = 8.0;
    vec2 id = vec2(0.0);
    for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++) {
            vec2 n = vec2(float(x), float(y));
            vec2 c = n + hash22(ip + n) - fp;
            float d = dot(c, c);
            if (d < f1) { f2 = f1; f1 = d; id = ip + n; }
            else if (d < f2) { f2 = d; }
        }
    float edge = sqrt(f2) - sqrt(f1);

    // Each shard refracts with its own tilt and catches its own light.
    vec2 tilt = (hash22(id + 3.7) - 0.5) * params.refract_ * 0.04;
    vec3 col = texture(Source, uv + tilt).rgb;
    col *= 0.92 + 0.16 * hash22(id + 9.1).x;

    // Cracks: a bright core line with a soft glassy glow around it.
    float line = 1.0 - smoothstep(0.0, params.width, edge);
    float glow = 1.0 - smoothstep(0.0, params.width * 3.0, edge);
    col += vec3(0.9, 0.95, 1.0) * (line * 0.9 + glow * 0.25) * params.cracks;

    vec3 src = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(src, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
