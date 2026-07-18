#version 450

// scrambled-crt — newpixie-crt with analog cable-scramble emulation
//
// Scrambling model (sinewave / "gated sync" cable scrambling, the iconic
// 80s/90s premium-channel look):
//
//   1. Per-scanline horizontal sinewave displacement, frequency drifts.
//   2. Per-channel chroma phase desync (color goes wrong / streaks).
//   3. Sync tears at pseudo-random Y bands (horizontal jumps).
//   4. Hum bars: slow rolling vertical brightness modulation.
//   5. Static bursts: occasional snow blends.
//
// All scramble effects modulate around the existing newpixie-crt CRT model
// (curvature, scanlines, shadow mask, ghosting). Disable scrambling by
// setting scramble_amount = 0; the shader collapses to vanilla newpixie.
//
// Original newpixie-crt: Mattias Gustavsson (MIT / public domain), slang
// adaptation by hunterk. Scramble overlay: this file.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OutputSize;
    uint FrameCount;
    float use_frame;
    float curvature;
    float wiggle_toggle;
    float scanroll;
    float vignette;
    float ghosting;
    float scramble_amount;
    float scramble_freq;
    float scramble_speed;
    float chroma_break;
    float hum_strength;
    float static_strength;
    float tear_freq;
    float instability;
    float jitter;
    float vroll_freq;
    float agc_pump;
    float detune_walk;
    float detune_range;
    float lock_freq;
    float lock_chance;
    float lock_depth;
    float lock_jitter;
} params;

#pragma parameter use_frame      "Use Frame Image"            0.0  0.0  1.0  1.0
#pragma parameter curvature      "Curvature"                  2.0  0.0001 4.0 0.25
#pragma parameter wiggle_toggle  "Interference"               0.0  0.0  1.0  1.0
#pragma parameter scanroll       "Rolling Scanlines"          1.0  0.0  1.0  1.0
#pragma parameter vignette       "Vignette"                   1.0  0.0  1.0  0.05
#pragma parameter ghosting       "Ghosting"                   1.0  0.0  2.0  0.10
#pragma parameter scramble_amount "Scramble: amount"            0.45 0.0  1.0  0.01
#pragma parameter scramble_freq   "Scramble: lines/screen"      32.0 4.0 160.0 1.0
#pragma parameter scramble_speed  "Scramble: phase drift"        2.0 0.0 10.0  0.1
#pragma parameter chroma_break    "Scramble: chroma desync"      0.9 0.0  2.5  0.05
#pragma parameter hum_strength    "Scramble: hum bars"           0.22 0.0  1.0  0.01
#pragma parameter static_strength "Scramble: static along wave"  0.85 0.0  1.5  0.05
#pragma parameter tear_freq       "Scramble: tears/sec"          2.5 0.0 12.0  0.1
#pragma parameter instability     "Scramble: intensity swing"    1.0 0.0  2.0  0.05
#pragma parameter jitter          "Scramble: per-line jitter"    0.7 0.0  2.0  0.05
#pragma parameter vroll_freq      "Scramble: V-sync drops/sec"   0.4 0.0  4.0  0.05
#pragma parameter agc_pump        "Scramble: AGC pumping"        0.5 0.0  1.5  0.05
#pragma parameter detune_walk     "Tune: drift speed"               0.25 0.0  3.0  0.01
#pragma parameter detune_range    "Tune: drift range (cap)"         0.35 0.0  1.0  0.01
#pragma parameter lock_freq       "Tune: lock events/sec"           1.5  0.0 10.0  0.1
#pragma parameter lock_chance     "Tune: lock probability"          0.10 0.0  1.0  0.01
#pragma parameter lock_depth      "Tune: focus depth toward clean"  0.85 0.0  1.0  0.01
#pragma parameter lock_jitter     "Tune: focus stutter / hunt"      0.6  0.0  1.0  0.01

#define use_frame    params.use_frame
#define gl_FragCoord (vTexCoord.xy * params.OutputSize.xy)
#define backbuffer accum1
#define blurbuffer blur2
#define frametexture frametexture

layout(std140, set = 0, binding = 0) uniform UBO
{
    mat4 MVP;
} global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;

void main()
{
    gl_Position = global.MVP * Position;
    vTexCoord = vec2(TexCoord.x, 1.0 - TexCoord.y);
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;
layout(set = 0, binding = 3) uniform sampler2D accum1;
layout(set = 0, binding = 4) uniform sampler2D blur2;
layout(set = 0, binding = 5) uniform sampler2D frametexture;

vec3 tsample(sampler2D samp, vec2 tc, float offs, vec2 resolution)
{
    tc = tc * vec2(1.025, 0.92) + vec2(-0.0125, 0.04);
    vec3 s = pow(abs(texture(samp, vec2(tc.x, 1.0 - tc.y)).rgb), vec3(2.2));
    return s * vec3(1.25);
}

vec3 filmic(vec3 LinearColor)
{
    vec3 x = max(vec3(0.0), LinearColor - vec3(0.004));
    return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
}

vec2 curve(vec2 uv)
{
    uv = (uv - 0.5);
    uv *= vec2(0.925, 1.095);
    uv *= params.curvature;
    uv.x *= 1.0 + pow((abs(uv.y) / 4.0), 2.0);
    uv.y *= 1.0 + pow((abs(uv.x) / 3.0), 2.0);
    uv /= params.curvature;
    uv += 0.5;
    uv = uv * 0.92 + 0.04;
    return uv;
}

float rand(vec2 co)
{
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// 1D hash for time-stepped events (tear/burst seeds).
float hash1(float n) { return fract(sin(n * 17.3217) * 43758.5453); }

// Smooth value noise + 4-octave fbm. Used for the macro instability
// envelope so the destruction level wanders organically over seconds
// instead of being a constant.
float vnoise(float x)
{
    float i = floor(x);
    float f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(hash1(i), hash1(i + 1.0), f);
}
float fbm(float x)
{
    float v = 0.0, a = 0.5;
    for (int k = 0; k < 4; ++k) { v += a * vnoise(x); x *= 2.0; a *= 0.5; }
    return v;
}

#define resolution params.OutputSize.xy

void main()
{
    /* Time / scramble phase */
    float time = mod(params.FrameCount, 849.0) * 36.0;
    float st   = float(params.FrameCount) * params.scramble_speed * 0.05;

    vec2 uv = vTexCoord.xy;

    /* Curve */
    vec2 curved_uv = mix(curve(uv), uv, 0.4);
    float scale = -0.101;
    vec2 scuv = curved_uv * (1.0 - scale) + scale / 2.0 + vec2(0.003, -0.001);

    uv = scuv;

    /* Existing wiggle (signal interference) */
    float x = params.wiggle_toggle *
              sin(0.1 * time + curved_uv.y * 13.0) *
              sin(0.23 * time + curved_uv.y * 19.0) *
              sin(0.3 + 0.11 * time + curved_uv.y * 23.0) * 0.0012;
    float o = sin(gl_FragCoord.y * 1.5) / resolution.x;
    x += o * 0.25;

    /* ═══════════════════════════════════════════════════════════════════════
     * SCRAMBLE CORE — physically-motivated analog cable scrambling
     * ═══════════════════════════════════════════════════════════════════════
     *
     * A real Jerrold/Hamlin scrambler suppressed the horizontal sync pulse
     * sinusoidally at audio rates, so the receiver's flywheel oscillator
     * would lose-and-regain horizontal lock continuously. Visually that
     * meant: per-scanline horizontal drift, jagged because adjacent lines
     * see different sync conditions; the drift amplitude itself wandered
     * over seconds as RF noise levels drifted; static blasts coincided
     * with the worst lock failures (not random); brightness pumped because
     * AGC tried to compensate; and chroma went rainbow because the 3.58MHz
     * subcarrier phase reference was lost.
     */

    /* Frame-rate-relative time, used by all sub-effects so behavior is
     * identical regardless of ffmpeg framerate. */
    float t_sec = float(params.FrameCount) * (1.0 / 60.0);

    /* ───────────────── Tuning walk: variable signal quality ───────────────
     * A constrained random-walk-like "how well are we locked" signal. It
     * wanders a little in and out of tune at all times (banded fbm drift),
     * and occasionally fires a sporadic event that either FOCUSES briefly
     * toward a clean lock (the only path that nears perfect — hence rare) or
     * HUNTS with fast jitter (AFC searching: agitated, not clean). The result
     * `clarity` multiplies the whole scramble intensity (1 = full scramble,
     * →0 = momentarily clean). `detune_range` caps the drift so the picture
     * seldom approaches perfect on its own; `lock_depth` sets how clean a
     * focus event gets. Synthesized from time — fragment shaders carry no
     * cross-frame state — so it reads as a random walk yet is reproducible. */
    float tw_drift  = fbm(t_sec * params.detune_walk + 3.7) * params.detune_range;

    float tw_slot   = floor(t_sec * params.lock_freq);
    float tw_fire   = step(1.0 - params.lock_chance, hash1(tw_slot * 5.17));
    float tw_kind   = hash1(tw_slot * 2.71);            /* <0.5 focus, else hunt */
    float tw_pos    = hash1(tw_slot * 9.43);
    float tw_phase  = fract(t_sec * params.lock_freq);

    /* Focus: short fast pulse toward clean, stuttered so the lock slips. */
    float tw_foc_env = 1.0 - smoothstep(0.0, 0.30, abs(tw_phase - tw_pos));
    tw_foc_env *= mix(1.0 - params.lock_jitter, 1.0, hash1(floor(t_sec * 60.0) * 3.7));
    float tw_focus   = tw_fire * step(tw_kind, 0.5) * tw_foc_env * params.lock_depth;

    /* Hunt: fast per-frame jitter burst, active mid-slot, kept within band. */
    float tw_hunt_env = 1.0 - smoothstep(0.0, 0.60, abs(tw_phase - 0.5));
    float tw_hunt_jit = (hash1(floor(t_sec * 30.0) * 4.3) - 0.5) * 2.0;
    float tw_hunt     = tw_fire * step(0.5, tw_kind) * tw_hunt_env * tw_hunt_jit
                      * params.lock_jitter * params.detune_range;

    /* Combine: drift+hunt stay banded; rare focus can push toward perfect. */
    float tune    = clamp(tw_drift + tw_hunt, 0.0, params.detune_range);
    tune          = clamp(max(tune, tw_focus), 0.0, 1.0);
    float clarity = 1.0 - tune;

    /* Macro instability envelope: 4-octave fbm in [0..1.5] that drifts on
     * a ~few-second timescale. Multiplied into every scramble sub-effect
     * so the whole destruction level swells and collapses naturally. */
    float macro = 0.35 + fbm(t_sec * 0.7) * 1.4;
    macro = mix(1.0, macro, params.instability);
    float intensity = params.scramble_amount * macro * clarity;

    /* Base sinewave shear: two harmonics with a wandering primary
     * frequency. This is the recognizable "sinewave scrambling" envelope. */
    float freq_drift = params.scramble_freq + 12.0 * sin(st * 0.13)
                                            + 8.0 * vnoise(t_sec * 0.4) - 4.0;
    float base = sin(scuv.y * freq_drift + st) * 0.13
               + sin(scuv.y * freq_drift * 1.83 + st * 1.27) * 0.05;

    /* Per-scanline jitter: hash keyed on (scanline, frame) so adjacent
     * lines kick by different amounts. This is what makes the picture
     * jumpy rather than smoothly waving — sync recovery timing varies
     * line-to-line. Quantized to ~480 logical scanlines so single
     * rasterized output rows on the same source line agree. */
    float line_id   = floor(scuv.y * 480.0);
    float frame_id  = floor(t_sec * 30.0);
    float line_seed = line_id + frame_id * 1024.0;
    float kick      = (hash1(line_seed * 0.137) - 0.5);
    /* Weight the kick by base-shear magnitude: where the sinewave is
     * already pushing, sync is most likely to fail, so kicks pile on
     * there. Coupling makes it look like one effect, not two. */
    float kick_weight = 0.4 + abs(base) * 5.0;
    kick *= kick_weight * 0.05 * params.jitter;

    /* Multi-band tearing: several sync-loss events per second, each at
     * its own random Y, with a smooth narrow band envelope. */
    float tear_x = 0.0;
    float tear_static = 0.0;     /* used later to drive coupled static */
    for (int b = 0; b < 4; ++b) {
        float ts   = floor(t_sec * (params.tear_freq + float(b) * 0.7));
        float ty   = hash1(ts * (1.7 + float(b) * 0.31));
        float tamp = (hash1(ts * 2.3 + float(b)) - 0.5) * 0.7;
        float tband = smoothstep(0.045, 0.0, abs(scuv.y - ty));
        tear_x      += tamp * tband;
        tear_static += tband;    /* unsigned, for static coupling */
    }

    /* Vertical sync drops: occasionally the picture rolls vertically by a
     * random fraction for a few frames. Not every frame — gated. */
    float vr_seed = floor(t_sec * (params.vroll_freq + 0.001));
    float vr_gate = step(0.62, hash1(vr_seed * 7.31));
    float vr_amount = vr_gate * (hash1(vr_seed * 13.7) - 0.5) * 0.4;
    scuv.y = fract(scuv.y + vr_amount * params.scramble_amount * macro * clarity);

    /* Combine into final horizontal displacement. */
    float scramble_x = (base + kick + tear_x) * intensity;
    x += scramble_x;

    /* Main color, RGB bleed. Sample all three channels at the SAME shifted
     * position — in real sinewave scrambling the entire scanline shifts
     * as a unit (the chroma 3.58 MHz subcarrier rides with the luma).
     * The original newpixie shader's tiny per-channel offsets (±0.0009)
     * are kept for the CRT phosphor-bleed look but no per-channel scramble
     * phase shift is applied. */
    vec3 col;
    col.r = tsample(backbuffer, vec2(x + scuv.x + 0.0009, scuv.y + 0.0009),
                    resolution.y / 800.0, resolution).x + 0.02;
    col.g = tsample(backbuffer, vec2(x + scuv.x + 0.0000, scuv.y - 0.0011),
                    resolution.y / 800.0, resolution).y + 0.02;
    col.b = tsample(backbuffer, vec2(x + scuv.x - 0.0015, scuv.y + 0.0000),
                    resolution.y / 800.0, resolution).z + 0.02;

    /* ───────────────── Scramble: chroma I/Q artifacts ─────────────────────
     * Real cable scrambling didn't smear color sideways like a continuous
     * phase shift; it broke the 3.58 MHz subcarrier reference, producing
     * either monochrome dropouts (color killer engages) or discrete I/Q
     * axis swaps on alternate fields (gated-sync chroma inversion). Both
     * behaviors keyed per-scanline, both gated by `chroma_break * intensity`.
     *
     * 1. Color killer: in disturbed lines, briefly desaturate to luma.
     *    Threshold-driven on (kick magnitude + tear envelope).
     * 2. I/Q axis flip: roughly equivalent to swapping R↔B with a small
     *    luminance correction. Less common, gated harder. */
    float cb = params.chroma_break * intensity;
    float disturb = abs(kick) * 14.0 + tear_static * 1.2;

    /* Color-killer dropouts: luma-only on the most disturbed lines. */
    float ck_thresh   = mix(1.6, 0.6, cb);   /* lower threshold = more frequent */
    float ck_strength = smoothstep(ck_thresh, ck_thresh + 0.4, disturb);
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, vec3(luma), ck_strength * 0.85);

    /* I/Q flip (R↔B swap on randomly-keyed scanlines). Discrete: a line
     * either flips or it doesn't, no smooth phase. Less likely than killer. */
    float iq_gate = step(mix(0.985, 0.86, cb), hash1(line_seed * 0.241));
    if (iq_gate > 0.5) {
        col.rb = col.br;
    }

    float i = clamp(col.r * 0.299 + col.g * 0.587 + col.b * 0.114, 0.0, 1.0);
    i = pow(1.0 - pow(i, 2.0), 1.0);
    i = (1.0 - i) * 0.85 + 0.15;

    /* Reset time for ghosting feedback math (matches original). */
    time = float(mod(params.FrameCount, 640) * 1);

    /* Ghosting */
    float ghs = 0.15 * params.ghosting;
    vec3 r = tsample(blurbuffer,
        vec2(x - 0.014, -0.027) * 0.85 +
        0.007 * vec2(0.35 * sin(1.0 / 7.0 + 15.0 * curved_uv.y + 0.9 * time),
                     0.35 * sin(2.0 / 7.0 + 10.0 * curved_uv.y + 1.37 * time)) +
        vec2(scuv.x + 0.001, scuv.y + 0.001),
        5.5 + 1.3 * sin(3.0 / 9.0 + 31.0 * curved_uv.x + 1.70 * time), resolution).xyz *
        vec3(0.5, 0.25, 0.25);
    vec3 g = tsample(blurbuffer,
        vec2(x - 0.019, -0.020) * 0.85 +
        0.007 * vec2(0.35 * cos(1.0 / 9.0 + 15.0 * curved_uv.y + 0.5 * time),
                     0.35 * sin(2.0 / 9.0 + 10.0 * curved_uv.y + 1.50 * time)) +
        vec2(scuv.x, scuv.y - 0.002),
        5.4 + 1.3 * sin(3.0 / 3.0 + 71.0 * curved_uv.x + 1.90 * time), resolution).xyz *
        vec3(0.25, 0.5, 0.25);
    vec3 b = tsample(blurbuffer,
        vec2(x - 0.017, -0.003) * 0.85 +
        0.007 * vec2(0.35 * sin(2.0 / 3.0 + 15.0 * curved_uv.y + 0.7 * time),
                     0.35 * cos(2.0 / 3.0 + 10.0 * curved_uv.y + 1.63 * time)) +
        vec2(scuv.x - 0.002, scuv.y),
        5.3 + 1.3 * sin(3.0 / 7.0 + 91.0 * curved_uv.x + 1.65 * time), resolution).xyz *
        vec3(0.25, 0.25, 0.5);

    col += vec3(ghs * (1.0 - 0.299)) * pow(clamp(vec3(3.0) * r, vec3(0.0), vec3(1.0)), vec3(2.0)) * vec3(i);
    col += vec3(ghs * (1.0 - 0.587)) * pow(clamp(vec3(3.0) * g, vec3(0.0), vec3(1.0)), vec3(2.0)) * vec3(i);
    col += vec3(ghs * (1.0 - 0.114)) * pow(clamp(vec3(3.0) * b, vec3(0.0), vec3(1.0)), vec3(2.0)) * vec3(i);

    /* Levels */
    col *= vec3(0.95, 1.05, 0.95);
    col = clamp(col * 1.3 + 0.75 * col * col + 1.25 * col * col * col * col * col,
                vec3(0.0), vec3(10.0));

    /* Vignette */
    float vig = ((1.0 - 0.99 * params.vignette) +
                 1.0 * 16.0 * curved_uv.x * curved_uv.y *
                 (1.0 - curved_uv.x) * (1.0 - curved_uv.y));
    vig = 1.3 * pow(vig, 0.5);
    col *= vig;

    time *= params.scanroll;

    /* Scanlines */
    float scans = clamp(0.35 + 0.18 * sin(6.0 * time - curved_uv.y * resolution.y * 1.5),
                        0.0, 1.0);
    float s = pow(scans, 0.9);
    col = col * vec3(s);

    /* Shadow mask */
    col *= 1.0 - 0.23 * (clamp((mod(gl_FragCoord.xy.x, 3.0)) / 2.0, 0.0, 1.0));

    /* ───────────────── Scramble: hum bars + AGC pumping ───────────────────
     * Hum: 60 Hz AC mains beating against the 60 Hz NTSC field rate
     * produces 1–2 broad bars that crawl slowly. The visible cycle count
     * per frame height is determined by the small frequency offset between
     * mains and field rate, not by an audio-rate harmonic. We model that
     * with ~1 cycle/screen plus a slow drift from a 2nd-harmonic beat.
     *
     * AGC: the receiver's gain control reacts within tens of milliseconds —
     * within a single frame, not over seconds. We pump brightness using a
     * fast hash that turns over every ~2 frames, modulated by macro for
     * "the AGC pumps harder when the signal is more disturbed." */
    float hum_drift = 0.05 * sin(t_sec * 0.31);   /* phase wanders ~5%/cycle */
    float hum = sin((scuv.y + hum_drift) * 6.283  + t_sec * 0.4) * 0.65
              + sin((scuv.y + hum_drift) * 12.566 - t_sec * 0.31) * 0.18;
    col *= 1.0 + hum * params.hum_strength * intensity;

    /* Fast AGC: ~30 Hz pump plus per-frame jitter. Both gated by macro
     * so AGC is calm when signal is good, frantic when it's bad. */
    float agc_seed = floor(t_sec * 30.0);
    float agc_fast = (hash1(agc_seed * 1.71) - 0.5) * 0.4
                   + (hash1(agc_seed * 3.13) - 0.5) * 0.2;
    float agc = 1.0 + agc_fast * params.agc_pump * (0.4 + macro * 0.6)
                    + (macro - 1.0) * params.agc_pump * 0.15;
    agc = mix(1.0, agc, clarity);   /* AGC settles toward neutral as we lock in */
    col *= agc;

    /* Tone map */
    col = filmic(col);

    /* Existing fine RGB noise */
    vec2 seed = curved_uv * resolution.xy;
    col -= 0.015 * pow(vec3(rand(seed + time),
                            rand(seed + time * 2.0),
                            rand(seed + time * 3.0)),
                       vec3(1.5));

    /* ───────────────── Scramble: static coupled to the wave ───────────────
     * Real scrambled-cable static rode WITH the sinewave displacement,
     * not as separate bursts: it was densest where the displacement was
     * largest (sync most failed) and along the tear bands. Three terms:
     *  • shear-coupled: |base+kick| → bigger shear, more snow
     *  • tear-coupled: tear_static envelope from the multi-band loop
     *  • macro-coupled: brief flashes when the instability envelope spikes
     */
    float shear_static = abs(base + kick) * 4.0;
    float macro_flash  = step(1.25, macro) * (macro - 1.25) * 2.0;
    float static_density = (shear_static + tear_static * 0.8 + macro_flash * 0.5)
                         * params.static_strength * intensity;
    static_density = clamp(static_density, 0.0, 0.92);
    vec3 snow = vec3(rand(seed + time + 1.0),
                     rand(seed + time + 2.0),
                     rand(seed + time + 3.0));
    col = mix(col, snow, static_density);

    /* Flicker */
    col *= (1.0 - 0.004 * (sin(50.0 * time + curved_uv.y * 2.0) * 0.5 + 0.5));

    uv = curved_uv;

    /* Frame */
    uv = vec2(uv.x, 1.0 - uv.y);
    vec4 f = texture(frametexture, vTexCoord.xy);
    f.xyz = mix(f.xyz, vec3(0.5, 0.5, 0.5), 0.5);
    float fvig = clamp(-0.00 + 512.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.2, 0.8);
    col = mix(col, mix(max(col, 0.0), pow(abs(f.xyz), vec3(1.4)) * fvig, f.w * f.w),
              vec3(use_frame));

    FragColor = vec4(col, 1.0);
}
