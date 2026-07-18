#version 450

// pixelsort — glitch-art pixel sorting (Kim Asendorf style), single pass.
//
// True pixel sorting reorders contiguous spans of pixels by brightness. We
// approximate the look cheaply per fragment: from each pixel we march along
// the sort axis through the contiguous "sortable" run (pixels whose key is
// above a threshold) and pull the run's extreme-key pixel back to here. The
// result is the signature long vertical streaks where bright (or dark) regions
// melt and smear into sorted bands.
//
// `invert` switches between sorting bright runs vs dark runs; `invert_dir`
// flips the smear direction; `wobble` animates the threshold so the sorted
// edges crawl. amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float threshold;
    float sort_length;
    float invert;
    float invert_dir;
    float wobble;
    float speed;
} params;

#pragma parameter amount      "Pixelsort: amount"          1.0  0.0  1.0   0.01
#pragma parameter threshold   "Pixelsort: threshold"       0.50 0.02 0.98  0.01
#pragma parameter sort_length "Pixelsort: streak length"   180.0 4.0 512.0 1.0
#pragma parameter invert      "Pixelsort: sort dark"       0.0  0.0  1.0   1.0
#pragma parameter invert_dir  "Pixelsort: sort upward"     0.0  0.0  1.0   1.0
#pragma parameter wobble      "Pixelsort: edge wobble"     0.15 0.0  0.5   0.01
#pragma parameter speed       "Pixelsort: wobble speed"    1.0  0.0  4.0   0.05

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

#define MARCH 128

float hash11(float n) { return fract(sin(n * 17.3217) * 43758.5453); }
float vnoise1(float x)
{
    float i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(hash11(i), hash11(i + 1.0), f);
}

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main()
{
    vec2  uv = vTexCoord;
    float t  = float(params.FrameCount) * (1.0 / 60.0) * params.speed;

    /* Key: brightness, or its inverse when sorting dark runs. */
    vec3  cur  = texture(Source, uv).rgb;
    float kcur = mix(luma(cur), 1.0 - luma(cur), params.invert);

    /* Threshold crawls with a noise wobble keyed on x and time so the sorted
     * span edges aren't static horizontal lines. */
    float th = clamp(params.threshold
                     + params.wobble * (vnoise1(uv.x * 40.0 + t) - 0.5) * 2.0,
                     0.02, 0.98);

    /* March direction along the sort axis (vertical). */
    float ddir   = (params.invert_dir < 0.5) ? -1.0 : 1.0;
    float pxstep = max(params.sort_length / float(MARCH), 0.5);
    vec2  stepuv = vec2(0.0, ddir) * params.SourceSize.zw * pxstep;

    /* Walk the contiguous sortable run, tracking the extreme-key pixel. */
    vec3  best = cur;
    float bestk = kcur;
    for (int i = 1; i < MARCH; ++i)
    {
        vec2 p = clamp(uv + stepuv * float(i), vec2(0.0), vec2(1.0));
        vec3 c = texture(Source, p).rgb;
        float k = mix(luma(c), 1.0 - luma(c), params.invert);
        if (k <= th) break;          /* run ended */
        if (k > bestk) { bestk = k; best = c; }
    }

    /* Only pixels that are themselves inside a run get sorted. */
    float in_run = step(th, kcur);
    vec3 outc = mix(cur, best, params.amount * in_run);
    FragColor = vec4(outc, 1.0);
}
