#version 450

// rf-ghost — analog RF reception trouble, single pass.
//
// The bad-antenna package, straight from the pre-cable era:
//
//   * Multipath ghosting — the signal also arrives bounced off a building,
//     a few microseconds late, so a faint displaced copy (or several)
//     sits to the right of the picture. Strong reflections arrive phase-
//     inverted and print as an eerie negative.
//   * Hum bar — mains hum beats against the frame rate as a wide dark
//     band rolling slowly up the screen, dragging the picture sideways
//     a touch as it passes.
//   * Herringbone — a co-channel carrier prints fine drifting diagonal
//     ripples over everything.
//   * Snow — thermal noise, worse whenever the signal fades.
//   * Fading — the channel breathes: every few seconds the picture washes
//     out, jitters, and snows over, then locks back in.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float ghosts;
    float ghost_sep;
    float ghost_strength;
    float ghost_invert;
    float hum;
    float hum_speed;
    float herring;
    float herring_freq;
    float snow;
    float fade;
    float jitter;
    float speed;
    float seed;
} params;

#pragma parameter amount         "RF: amount"            1.0  0.0  1.0   0.01
#pragma parameter ghosts         "RF: ghost count"       2.0  0.0  4.0   1.0
#pragma parameter ghost_sep      "RF: ghost distance"    0.045 0.0 0.25  0.001
#pragma parameter ghost_strength "RF: ghost strength"    0.45 0.0  1.0   0.01
#pragma parameter ghost_invert   "RF: ghost inversion"   0.35 0.0  1.0   0.01
#pragma parameter hum            "RF: hum bar"           0.5  0.0  1.0   0.01
#pragma parameter hum_speed      "RF: hum roll speed"    0.12 -1.0 1.0   0.01
#pragma parameter herring        "RF: herringbone"       0.35 0.0  1.0   0.01
#pragma parameter herring_freq   "RF: herringbone freq"  180.0 40.0 600.0 5.0
#pragma parameter snow           "RF: snow"              0.25 0.0  1.0   0.01
#pragma parameter fade           "RF: signal fading"     0.5  0.0  1.0   0.01
#pragma parameter jitter         "RF: line jitter"       0.3  0.0  1.0   0.01
#pragma parameter speed          "RF: speed"             1.0  0.0  4.0   0.05
#pragma parameter seed           "RF: seed"              9.0  0.0 64.0   1.0

/* Colour-only: inverted ghosts, the hum bar's darkening, snow and the
 * fade washout would all read as transparency on the coverage matte. */
#pragma matte_ignore ghost_invert hum snow fade

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

float hash1(float n) { return fract(sin(n * 17.3217) * 43758.5453); }
float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

void main()
{
    vec2  uv  = vTexCoord;
    float amt = params.amount;
    float t   = float(params.FrameCount) * (1.0 / 60.0) * params.speed;

    /* Channel fading: two incommensurate sines make an irregular breath.
     * weak = 0 → clean lock, weak = 1 → deep fade. */
    float weak = params.fade
               * (0.5 + 0.5 * sin(t * 0.31 + params.seed + 2.0 * sin(t * 0.117 + params.seed)));
    weak = weak * weak;    /* mostly fine, occasionally awful */

    /* Hum bar position (computed first: it also bends geometry). */
    float hb  = fract(uv.y - t * params.hum_speed);
    float bar = smoothstep(0.0, 0.22, hb) * (1.0 - smoothstep(0.28, 0.5, hb));

    /* Horizontal sync jitter, per scanline, worse in a fade / under the bar. */
    float line = floor(uv.y * params.SourceSize.y);
    float jit  = (hash21(vec2(line, floor(t * 60.0))) - 0.5)
               * params.jitter * (0.5 + 2.0 * weak) * 2.5 * params.SourceSize.z;
    uv.x += (jit + params.hum * bar * 0.004 * sin(t * 3.1)) * amt;

    vec3 col = texture(Source, clamp(uv, 0.0, 1.0)).rgb;

    /* Multipath ghosts: delayed copies to the right; each successive
     * reflection is weaker, drifts slowly, and may print inverted. */
    float total = 1.0;
    for (int i = 1; i <= 4; i++) {
        float fi = float(i);
        if (fi > params.ghosts + 0.5) break;
        float dx = params.ghost_sep * fi * (1.0 + 0.15 * sin(t * 0.4 + fi * 2.1 + params.seed));
        float a  = params.ghost_strength * pow(0.55, fi - 1.0) * (0.6 + 0.8 * weak);
        vec3  gc = texture(Source, clamp(uv - vec2(dx, 0.0), 0.0, 1.0)).rgb;
        float inv = params.ghost_invert * step(0.4, hash1(fi * 5.3 + params.seed));
        gc = mix(gc, 1.0 - gc, inv);
        col   += gc * a;
        total += a;
    }
    col /= total;

    /* Herringbone: fine diagonal carrier beat, phase crawling fast. */
    float hpat = sin(dot(uv, vec2(1.0, 0.35)) * params.herring_freq + t * 32.0)
               * sin(uv.y * params.herring_freq * 0.13 - t * 5.0);
    col += hpat * params.herring * 0.08 * (0.5 + weak) * amt;

    /* Hum bar darkens (and slightly crushes) the band it covers. */
    col *= 1.0 - params.hum * bar * 0.45 * amt;

    /* Snow: per-pixel per-frame sparkle, scaled by fade depth. */
    float gr = hash21(uv * params.SourceSize.xy + fract(t * 60.0) * 61.7);
    float sn = params.snow * (0.25 + 0.75 * weak);
    col = mix(col, vec3(gr), sn * amt);

    /* A deep fade washes out contrast and saturation before it snows over. */
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, vec3(luma), weak * 0.45 * amt);
    col = mix(col, vec3(0.5), weak * 0.3 * amt);

    /* amount also blends against the untouched frame for a clean bypass. */
    vec3 clean = texture(Source, vTexCoord).rgb;
    FragColor = vec4(clamp(mix(clean, col, amt), 0.0, 1.0), 1.0);
}
