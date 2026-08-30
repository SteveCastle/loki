#version 450

// rear-projection — 90s big-screen rear-projection TV, single pass.
//
// Three CRT guns fire through mirrors onto the back of a lenticular
// screen, and none of it quite lines up: red and blue converge worse the
// farther you get from center (the colored halos around everything near
// the edges), focus falls off with them, highlights bloom, the screen
// gain puts a bright hotspot in the middle that falls away in the
// corners, and the vertical lenticular ridges stripe the whole picture.
// Purely optical — nothing here animates.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float converge;
    float softness;
    float bloom;
    float hotspot;
    float lenticular;
    float scanline;
    float warmth;
} params;

#pragma parameter amount     "RPTV: amount"           1.0  0.0  1.0  0.01
#pragma parameter converge   "RPTV: misconvergence"   5.0  0.0 20.0  0.25
#pragma parameter softness   "RPTV: edge softness"    0.6  0.0  1.0  0.01
#pragma parameter bloom      "RPTV: highlight bloom"  0.5  0.0  1.0  0.01
#pragma parameter hotspot    "RPTV: screen hotspot"   0.6  0.0  1.0  0.01
#pragma parameter lenticular "RPTV: lenticular lines" 0.4  0.0  1.0  0.01
#pragma parameter scanline   "RPTV: scanlines"        0.25 0.0  1.0  0.01
#pragma parameter warmth     "RPTV: tube warmth"      0.35 0.0  1.0  0.01

/* Blur/convergence legitimately spread coverage; the dimming layers
 * (hotspot falloff, lenticular stripes, scanlines) must not thin it. */
#pragma matte_ignore hotspot lenticular scanline

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

/* Sample one gun: small cross blur whose radius grows off-center. */
vec3 gunTap(vec2 uv, vec2 blurStep)
{
    vec3 c = texture(Source, clamp(uv, 0.0, 1.0)).rgb * 0.4;
    c += texture(Source, clamp(uv + blurStep, 0.0, 1.0)).rgb * 0.15;
    c += texture(Source, clamp(uv - blurStep, 0.0, 1.0)).rgb * 0.15;
    c += texture(Source, clamp(uv + vec2(blurStep.y, -blurStep.x), 0.0, 1.0)).rgb * 0.15;
    c += texture(Source, clamp(uv - vec2(blurStep.y, -blurStep.x), 0.0, 1.0)).rgb * 0.15;
    return c;
}

void main()
{
    vec2  uv  = vTexCoord;
    float amt = params.amount;

    float aspect = params.SourceSize.x / params.SourceSize.y;
    vec2  d  = (uv - 0.5) * vec2(aspect, 1.0);
    float r2 = dot(d, d);

    /* Red and blue guns sit either side of green, so their error is mostly
     * horizontal and grows quadratically toward the edges. A touch of
     * vertical error keeps the corners honest. */
    vec2 off = vec2(d.x * abs(d.x) * 1.6 + d.x * 0.2, d.y * abs(d.y) * 0.5)
             * params.converge * params.SourceSize.z * amt / max(aspect, 1e-3);

    /* Focus falls off with convergence: blur radius grows off-center. */
    vec2 blurStep = normalize(d + vec2(1e-4)) * sqrt(r2)
                  * params.softness * 2.4 * params.SourceSize.zw * vec2(aspect, 1.0);

    vec3 col;
    col.r = gunTap(uv + off, blurStep).r;
    col.g = gunTap(uv, blurStep * 0.7).g;
    col.b = gunTap(uv - off, blurStep).b;

    /* Highlight bloom: the phosphor + screen scatter around bright areas. */
    vec3 wide = gunTap(uv, params.SourceSize.zw * 5.0);
    col += max(wide - 0.65, 0.0) * params.bloom * 0.8 * amt;

    /* Screen gain: hotspot in the middle, dim corners. */
    float gain = mix(1.0, 0.62 + 0.55 * exp(-r2 * 2.6), params.hotspot * amt);
    col *= gain;

    /* Lenticular ridges: fine vertical stripes across the whole screen. */
    float lp = sin(uv.x * params.SourceSize.x * 2.1);
    col *= 1.0 - params.lenticular * 0.10 * (0.5 + 0.5 * lp) * amt;

    /* Faint scanlines from the source tubes, softened by the screen. */
    float sl = 0.5 + 0.5 * sin(uv.y * params.SourceSize.y * 3.14159);
    col *= 1.0 - params.scanline * 0.22 * sl * amt;

    /* Warm tubes: gentle red push, slightly crushed gamma. */
    col = mix(col, pow(col, vec3(1.1)) * vec3(1.05, 1.0, 0.93), params.warmth * amt);

    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
