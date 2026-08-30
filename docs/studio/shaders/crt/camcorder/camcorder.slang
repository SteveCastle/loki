#version 450

// camcorder — handheld 90s consumer camcorder, single pass.
//
// The found-footage package, tuned for the backrooms look: a camera
// nobody is holding steady, in a room lit by humming fluorescents,
// recording to a tape that has been re-used too many times.
//
//   * Handheld sway — slow breathing drift plus a tiny roll, with the
//     occasional bump that rings for a moment before it settles. The
//     frame is cropped in slightly so the wander never shows an edge.
//   * Autofocus hunt — every few seconds the AF loses the wall and
//     breathes through a blur cycle before it locks again.
//   * Auto exposure — the iris pumps slowly, always a beat behind.
//   * Fluorescent cast — sickly yellow-green, muted colour, blacks
//     lifted into murk; hot lights haze over.
//   * Tape — luma softness, chroma smeared and delayed to the right,
//     shadow grain, white dropout streaks, a head-switching band
//     tearing at the very bottom, interlace field flicker.
//   * OSD — blinking ●REC, SP speed, a burned-in date and a running
//     timecode, drawn from a 3x5 bitmap font in stable screen space.
//
// amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float Time;
    float amount;
    float sway;
    float zoom;
    float focus;
    float soft;
    float pump;
    float bleed;
    float fluor;
    float grain;
    float dropout;
    float head;
    float interlace;
    float vig;
    float osd;
    float tc_start;
    float date_m;
    float date_d;
    float date_y;
    float speed;
    float seed;
} params;

#pragma parameter amount    "Cam: mix"             1.0    0.0    1.0    0.01
#pragma parameter sway      "Cam: handheld sway"   0.5    0.0    1.0    0.01
#pragma parameter zoom      "Cam: crop-in"         0.06   0.0    0.25   0.005
#pragma parameter focus     "Cam: autofocus hunt"  0.5    0.0    1.0    0.01
#pragma parameter soft      "Cam: lens softness"   0.35   0.0    1.0    0.01
#pragma parameter pump      "Cam: exposure pump"   0.5    0.0    1.0    0.01
#pragma parameter bleed     "Cam: chroma bleed"    0.55   0.0    1.0    0.01
#pragma parameter fluor     "Cam: fluorescent"     0.65   0.0    1.0    0.01
#pragma parameter grain     "Cam: tape grain"      0.35   0.0    1.0    0.01
#pragma parameter dropout   "Cam: dropouts"        0.4    0.0    1.0    0.01
#pragma parameter head      "Cam: head switching"  0.6    0.0    1.0    0.01
#pragma parameter interlace "Cam: field flicker"   0.4    0.0    1.0    0.01
#pragma parameter vig       "Cam: vignette"        0.35   0.0    1.2    0.01
#pragma parameter osd       "Cam: OSD"             1.0    0.0    1.0    0.01
#pragma parameter tc_start  "Cam: counter start"   0.0    0.0   10.0    0.01
#pragma parameter date_m    "Cam: date month"      6.0    1.0   12.0    1.0
#pragma parameter date_d    "Cam: date day"       14.0    1.0   31.0    1.0
#pragma parameter date_y    "Cam: date year"    1994.0 1980.0 1999.0    1.0
#pragma parameter speed     "Cam: speed"           1.0    0.0    4.0    0.05
#pragma parameter seed      "Cam: seed"            7.0    0.0   64.0    1.0

/* Colour-only: exposure pumping, the murky cast, grain, white dropout
 * streaks, the head-band noise, field flicker, the vignette and the
 * burned-in OSD would all read as transparency on the coverage matte.
 * Geometry (sway, crop, focus blur) stays so the silhouette matches. */
#pragma matte_ignore pump fluor grain dropout head interlace vig osd

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

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

// 3D hash: the frame index is its own dimension so the grain re-rolls in
// place instead of sliding diagonally across the picture.
float hash3(vec3 p)
{
    p = fract(p * vec3(123.34, 345.45, 678.56));
    p += dot(p, p.yzx + 34.345);
    return fract((p.x + p.y) * p.z);
}

/* ---- 3x5 bitmap font -------------------------------------------------
 * One row per argument, top row first; within a row the left-hand column
 * is the low bit (so "1.1" = 5, "11." = 3). */
#define GLYPH(a, b, c, d, e) \
    (uint(a) | (uint(b) << 3) | (uint(c) << 6) | (uint(d) << 9) | (uint(e) << 12))

#define CH_R     10
#define CH_E     11
#define CH_C     12
#define CH_S     13
#define CH_P     14
#define CH_COLON 15
#define CH_DOT   16
#define CH_BLANK 17

uint glyphMask(int c)
{
    if (c ==  0) return GLYPH(7, 5, 5, 5, 7);
    if (c ==  1) return GLYPH(2, 3, 2, 2, 7);
    if (c ==  2) return GLYPH(7, 4, 7, 1, 7);
    if (c ==  3) return GLYPH(7, 4, 7, 4, 7);
    if (c ==  4) return GLYPH(5, 5, 7, 4, 4);
    if (c ==  5) return GLYPH(7, 1, 7, 4, 7);
    if (c ==  6) return GLYPH(7, 1, 7, 5, 7);
    if (c ==  7) return GLYPH(7, 4, 4, 4, 4);
    if (c ==  8) return GLYPH(7, 5, 7, 5, 7);
    if (c ==  9) return GLYPH(7, 5, 7, 4, 7);
    if (c == CH_R) return GLYPH(7, 5, 7, 3, 5);
    if (c == CH_E) return GLYPH(7, 1, 7, 1, 7);
    if (c == CH_C) return GLYPH(7, 1, 1, 1, 7);
    if (c == CH_S) return GLYPH(7, 1, 7, 4, 7);
    if (c == CH_P) return GLYPH(7, 5, 7, 1, 1);
    if (c == CH_COLON) return GLYPH(0, 2, 0, 2, 0);
    if (c == CH_DOT) return GLYPH(0, 0, 0, 0, 2);
    return 0u;
}

// One font pixel of glyph `code`; g = (column, row) with row 0 at the top.
float glyphPixel(int code, ivec2 g)
{
    if (g.x < 0 || g.x > 2 || g.y < 0 || g.y > 4) return 0.0;
    return float((glyphMask(code) >> uint(g.y * 3 + g.x)) & 1u);
}

// Character slot under font-pixel coordinate d (cells are 4 px apart:
// 3 px of glyph, 1 px of gap). Returns -1 when d is off the text line.
int slotAt(vec2 d, int count)
{
    if (d.x < 0.0 || d.y < 0.0 || d.y >= 5.0) return -1;
    int i = int(floor(d.x * 0.25));
    return i < count ? i : -1;
}

float slotInk(vec2 d, int i, int code)
{
    return glyphPixel(code, ivec2(int(floor(d.x)) - i * 4, int(floor(d.y))));
}

float recLabel(vec2 d)                  // "REC"
{
    int i = slotAt(d, 3);
    if (i < 0) return 0.0;
    int code = i == 0 ? CH_R : (i == 1 ? CH_E : CH_C);
    return slotInk(d, i, code);
}

float spLabel(vec2 d)                   // "SP"
{
    int i = slotAt(d, 2);
    if (i < 0) return 0.0;
    return slotInk(d, i, i == 0 ? CH_S : CH_P);
}

float tcLabel(vec2 d, float secs)       // "H:MM:SS" tape counter
{
    int i = slotAt(d, 7);
    if (i < 0) return 0.0;
    int hh = int(mod(floor(secs / 3600.0), 10.0));
    int mm = int(mod(floor(secs / 60.0), 60.0));
    int ss = int(mod(floor(secs), 60.0));
    int code = CH_COLON;
    if (i == 0) code = hh;
    else if (i == 2) code = mm / 10;
    else if (i == 3) code = mm - (mm / 10) * 10;
    else if (i == 5) code = ss / 10;
    else if (i == 6) code = ss - (ss / 10) * 10;
    return slotInk(d, i, code);
}

float dateLabel(vec2 d, int m, int dd, int yy)   // "MM.DD.YYYY"
{
    int i = slotAt(d, 10);
    if (i < 0) return 0.0;
    int code = CH_DOT;
    if (i == 0) code = m / 10;
    else if (i == 1) code = m - (m / 10) * 10;
    else if (i == 3) code = dd / 10;
    else if (i == 4) code = dd - (dd / 10) * 10;
    else if (i == 6) code = (yy / 1000);
    else if (i == 7) code = (yy / 100) - (yy / 1000) * 10;
    else if (i == 8) code = (yy / 10) - (yy / 100) * 10;
    else if (i == 9) code = yy - (yy / 10) * 10;
    return slotInk(d, i, code);
}

// Every OSD field at once, in output pixels. Called twice — once for the
// text, once offset by a font pixel for its drop shadow.
float osdInk(vec2 px, float sc, float pad, float secs, float blink)
{
    float ink = 0.0;
    if (blink > 0.5)
        ink = recLabel((px - vec2(pad + 5.0 * sc, pad)) / sc);
    ink = max(ink, spLabel((px - vec2(params.OutputSize.x - pad - 7.0 * sc, pad)) / sc));
    ink = max(ink, dateLabel((px - vec2(pad, params.OutputSize.y - pad - 5.0 * sc)) / sc,
                             int(clamp(params.date_m, 1.0, 12.0)),
                             int(clamp(params.date_d, 1.0, 31.0)),
                             int(clamp(params.date_y, 0.0, 9999.0))));
    ink = max(ink, tcLabel((px - vec2(params.OutputSize.x - pad - 27.0 * sc,
                                      params.OutputSize.y - pad - 5.0 * sc)) / sc, secs));
    return ink;
}

/* Signed dropout amplitude for one source line at horizontal position x:
 * positive = bright oxide hit, negative = clogged-head dark hit, 0 = none.
 * Hits are short comets — a soft head, an exponential tail that crumbles
 * away in ragged chunks — and cluster around a bad zone drifting through
 * the picture, with rare rough patches raising the rate everywhere. */
float dropStreak(float line, float lines, float x, float t, float s)
{
    float dwin  = floor(t * 30.0);                     /* re-roll at 30 Hz */
    float bandC = hash21(vec2(floor(t * 0.7), s));     /* bad zone drifts  */
    float band  = exp(-abs((line + 0.5) / lines - bandC) * 9.0);
    float burst = step(0.94, hash21(vec2(floor(t * 2.3), s + 4.0)));
    float p     = params.dropout * 0.015 * (0.3 + 1.4 * band + 2.0 * burst);
    if (hash21(vec2(line, dwin)) > p) return 0.0;

    float id  = line * 7.13 + dwin * 3.71;             /* per-hit seeds */
    float x0  = hash21(vec2(id, 13.0));
    float len = 0.01 + 0.05 * hash21(vec2(id, 27.0));
    float u   = (x - x0) / len;
    if (u < 0.0 || u > 1.2) return 0.0;

    float prof = smoothstep(0.0, 0.15, u) * exp(-u * 2.5)
               * (1.0 - smoothstep(0.9, 1.2, u));
    prof *= 0.6 + 0.4 * hash21(vec2(floor(u * 24.0), id));   /* ragged tail */
    float amp  = 0.35 + 0.65 * hash21(vec2(id, 41.0));
    float dark = step(0.85, hash21(vec2(id, 55.0)));
    return prof * amp * mix(1.0, -0.8, dark);
}

void main()
{
    float amt = params.amount;
    float t   = params.Time * params.speed;
    float s   = params.seed;
    float aspect = params.OutputSize.x / max(params.OutputSize.y, 1.0);

    /* ---- Handheld: drift + roll + the occasional bump ---------------- */
    vec2 drift = vec2(
        sin(t * 0.31 + s)        * 0.45 + sin(t * 0.93 + 2.1) * 0.35 + sin(t * 1.71 + s * 1.7) * 0.20,
        sin(t * 0.27 + s + 0.7)  * 0.45 + sin(t * 1.13 + 1.3) * 0.30 + sin(t * 2.31)           * 0.15
    ) * params.sway * 0.012;
    float roll = (sin(t * 0.43 + s) * 0.6 + sin(t * 1.07 + 1.9) * 0.4) * params.sway * 0.012;

    // Bumps: some four-second windows contain a knock that rings down.
    float bwin = floor(t * 0.25);
    float bph  = fract(t * 0.25);
    float bump = step(0.55, hash21(vec2(bwin, s + 11.0)));
    vec2  bdir = vec2(hash21(vec2(bwin, 1.3)), hash21(vec2(bwin, 5.7))) - 0.5;
    drift += bdir * bump * exp(-bph * 9.0) * sin(bph * 55.0) * params.sway * 0.02;
    roll  += (hash21(vec2(bwin, 8.4)) - 0.5) * bump * exp(-bph * 9.0) * sin(bph * 47.0)
           * params.sway * 0.02;

    // Crop in so the wander never pulls an edge into frame, then rotate
    // about the centre (in aspect-true space so the roll doesn't skew).
    vec2 c = vTexCoord - 0.5;
    c *= 1.0 - params.zoom - params.sway * 0.03;
    c.x *= aspect;
    float cs = cos(roll), sn = sin(roll);
    c = vec2(c.x * cs - c.y * sn, c.x * sn + c.y * cs);
    c.x /= aspect;
    vec2 uv = c + 0.5 + drift;

    /* ---- Tracking wiggle + head-switching tear at the bottom --------- */
    float lineId = floor(vTexCoord.y * params.SourceSize.y);
    float fr     = floor(t * 60.0);
    uv.x += (hash21(vec2(lineId, fr)) - 0.5) * params.grain * 0.4 * params.SourceSize.z;
    float hband = smoothstep(0.955, 0.99, vTexCoord.y) * params.head;
    uv.x += hband * (hash21(vec2(fr, 2.2)) - 0.35) * 0.06;

    uv = clamp(uv, 0.001, 0.999);

    /* ---- Autofocus hunt: blur breathes in, wobbles, re-locks --------- */
    float fwin = floor(t * 0.2 + 0.37);
    float fph  = fract(t * 0.2 + 0.37);
    float hunt = step(0.5, hash21(vec2(fwin, s + 3.0)))
               * smoothstep(0.0, 0.2, fph) * (1.0 - smoothstep(0.5, 0.85, fph))
               * (0.6 + 0.4 * sin(fph * 40.0 + s));
    float radius = (params.soft * 1.4 + params.focus * 4.0 * hunt) * params.SourceSize.w
                 * (params.SourceSize.y / 480.0);   /* scale blur with resolution */

    vec3 col = texture(Source, uv).rgb * 0.25;
    for (int i = 0; i < 6; i++) {
        float a = float(i) * 1.0471976 + 0.5;      /* 6 taps around a ring */
        col += texture(Source, clamp(uv + vec2(cos(a) / aspect, sin(a)) * radius,
                                     0.0, 1.0)).rgb * 0.125;
    }

    /* ---- VHS chroma: smeared wide and delayed to the right ----------- */
    vec3 cb = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        vec2 off = vec2(-(float(i) + 0.5) * params.bleed * 5.0 * params.SourceSize.z, 0.0);
        cb += texture(Source, clamp(uv + off, 0.0, 1.0)).rgb * 0.25;
    }
    float Y = luma(col);
    col = mix(col, vec3(Y) + (cb - vec3(luma(cb))), params.bleed);

    /* ---- Auto exposure: the iris pumps, always chasing --------------- */
    float ex = 1.0 + params.pump * (0.30 * sin(t * 0.53 + s) * sin(t * 0.21 + 1.1)
                                  + 0.12 * sin(t * 1.31 + s * 0.7));
    col *= ex;
    col += max(col - 0.75, vec3(0.0)) * 0.6;       /* hot lights haze over */
    col = col / (1.0 + 0.18 * max(col - vec3(1.0), vec3(0.0)));

    /* ---- Fluorescent murk: green-yellow, muted, lifted blacks -------- */
    Y = luma(col);
    col = mix(col, vec3(Y), 0.30 * params.fluor);
    col *= mix(vec3(1.0), vec3(1.06, 1.05, 0.78), params.fluor);
    col = mix(col, col * 0.86 + vec3(0.055, 0.065, 0.045), params.fluor * 0.55);

    /* ---- Tape grain: horizontally smeared, living in the shadows ----- */
    float n = hash3(vec3(floor(vTexCoord * params.OutputSize.xy * vec2(0.5, 1.0)),
                         float(params.FrameCount % 512u)));
    col += (n - 0.5) * params.grain * 0.28 * (1.1 - clamp(Y, 0.0, 1.0) * 0.7);

    /* ---- Dropouts: brief comet streaks where the oxide let go. The
     * damage isn't uniform — hits cluster around a drifting bad zone on
     * the tape, with the odd rough patch peppering the whole frame. Each
     * streak blooms softly across neighbouring lines, its tail crumbles
     * away raggedly, and a few hits are dark (a clogged head) instead
     * of bright. ------------------------------------------------------- */
    if (params.dropout > 0.001) {
        float lines = params.SourceSize.y;
        float fy = vTexCoord.y * lines;
        float drop = 0.0;
        for (int k = -1; k <= 1; k++) {
            float ln = floor(fy) + float(k);
            float a  = dropStreak(ln, lines, vTexCoord.x, t, s);
            float d  = fy - (ln + 0.5);
            drop += a * exp(-d * d * 1.8);      /* ~1.5-line vertical bloom */
        }
        /* Bright hits land off-white and keep a whisper of the scene's
         * luma; dark hits sink toward tape black. */
        float w = clamp(drop, 0.0, 1.0);
        float b = clamp(-drop, 0.0, 1.0);
        col = mix(col, vec3(0.90, 0.90, 0.86) * (0.78 + 0.22 * Y), w);
        col = mix(col, vec3(0.05, 0.05, 0.06), b);
    }

    /* ---- Head-switching band: noise chews the last few lines --------- */
    float hn = hash3(vec3(floor(vTexCoord * params.OutputSize.xy), float(params.FrameCount % 256u)));
    col = mix(col, vec3(hn * 0.7), hband * 0.6);

    /* ---- Burned-in OSD, in stable screen space (the camera writes it
     * after the sensor, so it never sways or blurs) -------------------- */
    if (params.osd > 0.001) {
        vec2 px = vTexCoord * params.OutputSize.xy;
        float sc  = max(floor(params.OutputSize.y / 150.0), 2.0);
        float pad = sc * 6.0;
        float secs = params.tc_start * 3600.0 + t;
        float blink = step(0.4, fract(t * 0.55));

        /* The character generator itself is rock steady, but its output
         * rides the same wobbly tape signal as the picture: every output
         * line lands with a small horizontal error, and the whole overlay
         * breathes gently. */
        float oline = floor(px.y);
        px.x += (hash21(vec2(oline, fr)) - 0.5) * sc * 0.5
              + sin(t * 1.7 + oline * 0.045) * sc * 0.22;

        /* Soft ink: staggered taps melt the glyph edges into the picture,
         * and a faint tail bleeds off to the right the way bright OSD
         * smears on tape. */
        float ink = osdInk(px, sc, pad, secs, blink) * 0.5
                  + osdInk(px + vec2(0.55 * sc, 0.0), sc, pad, secs, blink) * 0.25
                  + osdInk(px - vec2(0.55 * sc, 0.0), sc, pad, secs, blink) * 0.25;
        float tail  = osdInk(px - vec2(1.7 * sc, 0.0), sc, pad, secs, blink);
        float shade = osdInk(px - vec2(sc), sc, pad, secs, blink);
        col = mix(col, vec3(0.0), shade * 0.45 * params.osd);
        col = mix(col, vec3(0.93, 0.92, 0.86), ink * 0.85 * params.osd);
        col += vec3(0.9, 0.9, 0.85) * tail * 0.10 * params.osd;

        // Record dot, left of the REC legend (wobbles with the text).
        vec2 dot_c = vec2(pad + 2.0 * sc, pad + 2.0 * sc);
        float rec_dot = smoothstep(2.2 * sc, 0.9 * sc, length(px - dot_c)) * blink;
        col = mix(col, vec3(0.92, 0.12, 0.10), rec_dot * 0.88 * params.osd);
    }

    /* ---- Interlace field flicker + vignette -------------------------- */
    float field = mod(floor(vTexCoord.y * params.OutputSize.y) + float(params.FrameCount % 2u), 2.0);
    col *= 1.0 - params.interlace * 0.16 * field;

    vec2 vc = vTexCoord * 2.0 - 1.0;
    col *= max(1.0 - params.vig * 0.55 * dot(vc, vc), 0.0);

    /* amount also blends against the untouched frame for a clean bypass. */
    vec3 clean = texture(Source, vTexCoord).rgb;
    FragColor = vec4(clamp(mix(clean, col, amt), 0.0, 1.0), 1.0);
}
