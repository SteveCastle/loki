#version 450

// voronoi-shatter — a motion-reactive crystalline shatter built on an
// animated Voronoi (cellular) diagram.
//
// Pass 1 of 2. Source = current frame (from passthrough); PassFeedback0 =
// previous frame (motion delta); PassFeedback1 = this pass's own previous
// output (glow accumulator).
//
// The frame is fractured into Voronoi cells whose feature points orbit on
// sine paths, so the whole mosaic crawls. Each cell SAMPLES the image at its
// own moving centre — turning the picture into a raft of drifting glass shards
// rather than a flat tile grid. The F2-F1 border distance (gap to the second-
// nearest cell) is rendered as glowing leaded cracks, hue-cycled per cell via
// a golden-ratio hash. Motion (frame difference) drives the chaos: it pushes
// the cell jitter so shards scatter, blows each shard outward from its centre
// (the "shatter"), and ignites the cracks. A radius-dependent swirl twists the
// lattice for extra flow. amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float cells;
    float jitter;
    float shatter;
    float crack;
    float swirl;
    float speed;
    float decay;
    float sensitivity;
} params;

#pragma parameter amount       "Voronoi: mix"                 1.0   0.0   1.0   0.01
#pragma parameter cells        "Voronoi: cell count"          14.0  3.0   48.0  0.5
#pragma parameter jitter       "Voronoi: cell jitter"         0.85  0.0   1.0   0.01
#pragma parameter shatter      "Voronoi: shard explode"       0.30  0.0   1.0   0.01
#pragma parameter crack        "Voronoi: crack glow"          1.2   0.0   3.0   0.05
#pragma parameter swirl        "Voronoi: swirl twist"         1.0   0.0   4.0   0.05
#pragma parameter speed        "Voronoi: drift speed"         0.40  0.0   2.0   0.01
#pragma parameter decay        "Voronoi: glow trail life"     0.85  0.70  0.98  0.005
#pragma parameter sensitivity  "Voronoi: motion gain"         9.0   0.5   40.0  0.5

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

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}
vec2 hash22(vec2 p)
{
    float n = sin(dot(p, vec2(41.0, 289.0))) * 43758.5453;
    return fract(vec2(n, n * 1.37)) ;
}

// IQ cosine palette for per-cell stained-glass colour.
vec3 palette(float t)
{
    return 0.5 + 0.5 * cos(6.28318 * (vec3(1.0) * t + vec3(0.0, 0.33, 0.67)));
}

// Voronoi over lattice coords x. Animated feature points (orbit on sine
// paths). Returns nearest dist f1, second f2, the vector to the winning
// centre (cell space), and a stable per-cell id.
void voronoi(vec2 x, float t, float jit,
             out float f1, out float f2, out vec2 toCenter, out float id)
{
    vec2 n = floor(x);
    vec2 f = fract(x);
    f1 = 8.0; f2 = 8.0; toCenter = vec2(0.0); id = 0.0;
    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++)
    {
        vec2 g = vec2(float(i), float(j));
        vec2 o = hash22(n + g);
        // orbit each feature point so the lattice crawls
        o = 0.5 + 0.5 * sin(t + 6.28318 * o);
        vec2 r = g + mix(vec2(0.5), o, jit) - f;
        float d = dot(r, r);
        if (d < f1) { f2 = f1; f1 = d; toCenter = r; id = hash21(n + g); }
        else if (d < f2) { f2 = d; }
    }
    f1 = sqrt(f1); f2 = sqrt(f2);
}

void main()
{
    vec2 uv  = vTexCoord;
    float t  = float(params.FrameCount) * params.speed * 0.1;
    float asp = params.SourceSize.x / max(1.0, params.SourceSize.y);

    // Motion magnitude [0,1] with cold-start guard.
    vec3 cur  = texture(Source, uv).rgb;
    vec3 prev = texture(PassFeedback0, uv).rgb;
    float warm = step(2.0, float(params.FrameCount));
    float m = clamp(length(cur - prev) * params.sensitivity, 0.0, 1.0) * warm;

    // --- Radius-dependent swirl twists the whole lattice -----------------
    vec2 c = uv - 0.5;
    c.x *= asp;
    float ang = params.swirl * (0.25 + 0.75 * m) * exp(-2.5 * length(c))
              + 0.2 * t;
    float cs = cos(ang), sn = sin(ang);
    vec2 sc = mat2(cs, -sn, sn, cs) * c;

    // Lattice coordinates for the Voronoi.
    vec2 lx = sc * params.cells;

    float f1, f2, id; vec2 toC;
    float jit = clamp(params.jitter * (0.6 + 0.6 * m), 0.0, 1.0);
    voronoi(lx, t, jit, f1, f2, toC, id);

    // --- Shatter: blow each shard outward from its own centre ------------
    // toC points from the cell centre to the pixel; pushing UV along it (more
    // with motion) makes shards fly apart. Convert cell-space back to uv.
    vec2 shardDir = toC / params.cells;
    shardDir.x /= asp;
    float explode = params.shatter * (0.15 + 0.85 * m);
    vec2 suv = uv + shardDir * explode;

    // Per-cell chromatic split along the shard direction.
    vec2 ca = shardDir * (0.02 + 0.10 * m);
    vec3 shard;
    shard.r = texture(Source, suv + ca).r;
    shard.g = texture(Source, suv).g;
    shard.b = texture(Source, suv - ca).b;

    // --- Stained-glass tint, golden-ratio hue per cell -------------------
    vec3 glass = palette(fract(id * 1.61803398 + 0.1 * t));
    float tintAmt = 0.25 + 0.35 * m;
    vec3 col = mix(shard, shard * (0.4 + 1.6 * glass), tintAmt);

    // --- Glowing leaded cracks from the F2-F1 border distance ------------
    float border = f2 - f1;
    float w = 0.04 + 0.06 * (1.0 - m);                 // sharper cracks on motion
    float edge = smoothstep(0.0, w, border);           // 0 on the seam
    float crackGlow = (1.0 - edge);
    vec3 crackCol = palette(fract(id * 1.61803398 + 0.5)) * (1.0 + 2.0 * m);
    col = mix(col, col * 0.15, 0.85 * crackGlow);       // darken the lead line
    col += crackCol * crackGlow * params.crack * (0.4 + 0.6 * m); // neon glow

    // --- Feedback glow trails (cracks smear, decay bounds the loop) ------
    vec3 ghost = texture(PassFeedback1, uv - shardDir * 0.02 * m).rgb * params.decay;
    col = max(col, ghost * (0.6 + 0.4 * m));

    // Saturate where it moves so the glass reads vivid.
    float l = luma(col);
    col = clamp(mix(vec3(l), col, 1.0 + 1.0 * m), 0.0, 1.0);

    col = mix(cur, col, params.amount);
    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
