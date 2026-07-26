#version 450

// security-cam — pass 1: the monitor in the back office.
//
// Not a simulation of any particular camera — an artistic read of the
// look: a wide, slightly bulging lens, a coarse low-line sensor with the
// gain wound up (crushed blacks, blown highlights, grain that lives in
// the shadows), a near-monochrome cold picture, mains hum crawling up the
// tube, the sync lock stuttering every so often, and the burned-in OSD —
// camera number, running clock, blinking record dot — sitting in the
// signal so the scanlines run right through it.
//
// The OSD glyphs are drawn procedurally from a 3x5 bitmap font, so they
// stay crisp at any comp size and cost no texture.
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
    float lens;
    float tube;
    float cam_lines;
    float gain;
    float wash;
    float cool;
    float scan_cam;
    float interlace;
    float noise_cam;
    float hum_cam;
    float desync;
    float vignette_cam;
    float osd;
    float cam_id;
    float clock_start;
} params;

#pragma parameter amount       "Cam: mix"              1.0   0.0   1.0   0.01
#pragma parameter lens         "Cam: wide lens"        0.45  0.0   1.0   0.01
#pragma parameter tube         "Cam: tube curve"       0.14  0.0   0.5   0.01
#pragma parameter cam_lines    "Cam: sensor lines"     360.0 64.0 1080.0 4.0
#pragma parameter gain         "Cam: auto gain"        0.55  0.0   1.5   0.01
#pragma parameter wash         "Cam: desaturate"       0.85  0.0   1.0   0.01
#pragma parameter cool         "Cam: cold cast"        0.6   0.0   1.0   0.01
#pragma parameter scan_cam     "Cam: scanlines"        0.28  0.0   0.9   0.01
#pragma parameter interlace    "Cam: field flicker"    0.35  0.0   1.0   0.01
#pragma parameter noise_cam    "Cam: sensor noise"     0.14  0.0   0.6   0.01
#pragma parameter hum_cam      "Cam: hum bar"          0.10  0.0   0.5   0.01
#pragma parameter desync       "Cam: sync slip"        0.6   0.0   1.0   0.01
#pragma parameter vignette_cam "Cam: vignette"         0.45  0.0   1.2   0.01
#pragma parameter osd          "Cam: OSD"              0.9   0.0   1.0   0.01
#pragma parameter cam_id       "Cam: camera number"    4.0   0.0  99.0   1.0
#pragma parameter clock_start  "Cam: clock start (h)"  2.37  0.0  24.0   0.01

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
layout(set = 0, binding = 3) uniform sampler2D Original;

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

#define CH_C     10
#define CH_A     11
#define CH_M     12
#define CH_R     13
#define CH_E     14
#define CH_COLON 15
#define CH_BLANK 16

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
    if (c == CH_C) return GLYPH(7, 1, 1, 1, 7);
    if (c == CH_A) return GLYPH(7, 5, 7, 5, 5);
    if (c == CH_M) return GLYPH(5, 7, 7, 5, 5);
    if (c == CH_R) return GLYPH(7, 5, 7, 3, 5);
    if (c == CH_E) return GLYPH(7, 1, 7, 1, 7);
    if (c == CH_COLON) return GLYPH(0, 2, 0, 2, 0);
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

float camLabel(vec2 d, int id)          // "CAM NN"
{
    int i = slotAt(d, 6);
    if (i < 0) return 0.0;
    int code = CH_BLANK;
    if (i == 0) code = CH_C;
    else if (i == 1) code = CH_A;
    else if (i == 2) code = CH_M;
    else if (i == 4) code = (id / 10) - (id / 100) * 10;
    else if (i == 5) code = id - (id / 10) * 10;
    return slotInk(d, i, code);
}

float clockLabel(vec2 d, float secs)    // "HH:MM:SS"
{
    int i = slotAt(d, 8);
    if (i < 0) return 0.0;
    int hh = int(mod(floor(secs / 3600.0), 24.0));
    int mm = int(mod(floor(secs / 60.0), 60.0));
    int ss = int(mod(floor(secs), 60.0));
    int code = CH_COLON;
    if (i == 0) code = hh / 10;
    else if (i == 1) code = hh - (hh / 10) * 10;
    else if (i == 3) code = mm / 10;
    else if (i == 4) code = mm - (mm / 10) * 10;
    else if (i == 6) code = ss / 10;
    else if (i == 7) code = ss - (ss / 10) * 10;
    return slotInk(d, i, code);
}

float recLabel(vec2 d)                  // "REC"
{
    int i = slotAt(d, 3);
    if (i < 0) return 0.0;
    int code = i == 0 ? CH_R : (i == 1 ? CH_E : CH_C);
    return slotInk(d, i, code);
}

// Every OSD field at once, in output pixels. Called twice — once for the
// text, once offset by a font pixel for its drop shadow.
float osdInk(vec2 px, float sc, float pad, int id, float secs, float blink)
{
    float ink = camLabel((px - vec2(pad, pad)) / sc, id);
    vec2 clk = vec2(params.OutputSize.x - pad - 31.0 * sc,
                    params.OutputSize.y - pad - 5.0 * sc);
    ink = max(ink, clockLabel((px - clk) / sc, secs));
    if (blink > 0.5)
        ink = max(ink, recLabel((px - vec2(params.OutputSize.x - pad - 11.0 * sc, pad)) / sc));
    return ink;
}

void main()
{
    vec3 base = texture(Original, vTexCoord).rgb;
    float t = params.Time;

    // The tube first: everything downstream — picture and OSD alike —
    // lives on the curved glass, so the burn-in bends with it.
    vec2 c = vTexCoord * 2.0 - 1.0;
    vec2 cu = c * (1.0 + params.tube * 0.22 * dot(c, c));
    vec2 scr = cu * 0.5 + 0.5;
    float on_tube = step(0.0, scr.x) * step(scr.x, 1.0) * step(0.0, scr.y) * step(scr.y, 1.0);

    // Wide lens: the edges are pulled inward so the centre bulges. Pulling
    // in (rather than pushing out) keeps every tap inside the frame.
    vec2 lv = cu * (1.0 - params.lens * 0.24 * dot(cu, cu));
    vec2 uv = lv * 0.5 + 0.5;

    // Sync slip: a couple of times a minute the line lock stutters — a few
    // lines tear sideways and the picture jumps before it settles.
    float win = floor(t * 0.5);
    float slip = step(0.85, hash21(vec2(win, 3.7))) * params.desync;
    float ph = fract(t * 0.5);
    float settle = exp(-ph * 5.0);
    uv.y = fract(uv.y + slip * settle * 0.05 * sin(ph * 34.0));

    // Coarse sensor: snap to a fixed line count, square-ish cells.
    float lines = max(params.cam_lines, 32.0);
    float aspect = params.OutputSize.x / max(params.OutputSize.y, 1.0);
    vec2 grid = vec2(floor(lines * aspect), lines);
    float line_id = floor(uv.y * lines);
    float tear = step(0.93, hash21(vec2(line_id, floor(t * 30.0))));
    uv.x += slip * settle * tear * (hash21(vec2(line_id, 9.1)) - 0.5) * 0.12;
    vec2 quv = clamp((floor(uv * grid) + 0.5) / grid, 0.002, 0.998);

    vec3 col = texture(Source, quv).rgb;

    // Wound-up gain: highlights blow out and haze, blacks crush.
    vec2 gpx = params.SourceSize.zw * 3.0;
    vec3 halo = texture(Source, quv + gpx).rgb + texture(Source, quv - gpx).rgb
              + texture(Source, quv + vec2(gpx.x, -gpx.y)).rgb
              + texture(Source, quv + vec2(-gpx.x, gpx.y)).rgb;
    col += halo * 0.25 * params.gain * 0.5;
    col = max((col - 0.05) * (1.0 + params.gain), vec3(0.0));

    // Cheap near-monochrome sensor with a cold cast.
    float y = luma(col);
    col = mix(col, vec3(y), params.wash);
    col *= mix(vec3(1.0), vec3(0.80, 0.93, 1.10), params.cool);

    // Grain lives in the shadows, where the gain is doing the most work.
    float n = hash3(vec3(floor(vTexCoord * params.OutputSize.xy),
                         float(params.FrameCount % 512u)));
    col += (n - 0.5) * params.noise_cam * (1.3 - clamp(y, 0.0, 1.0) * 0.9);

    // Burned-in OSD, drawn before the scanlines so they run through it.
    if (params.osd > 0.001) {
        vec2 px = scr * params.OutputSize.xy;
        float sc = max(floor(params.OutputSize.y / 150.0), 2.0);
        float pad = sc * 5.0;
        float secs = params.clock_start * 3600.0 + t;
        float blink = step(0.35, fract(t * 0.5));
        int id = int(clamp(params.cam_id, 0.0, 99.0));

        float ink = osdInk(px, sc, pad, id, secs, blink);
        float shade = osdInk(px - vec2(sc), sc, pad, id, secs, blink);
        col = mix(col, vec3(0.0), shade * 0.75 * params.osd);
        col = mix(col, vec3(0.94, 0.96, 0.92), ink * params.osd);

        // Record dot, left of the REC legend.
        vec2 dot_c = vec2(params.OutputSize.x - pad - 15.0 * sc, pad + 2.0 * sc);
        float rec_dot = smoothstep(1.9 * sc, 1.1 * sc, length(px - dot_c)) * blink;
        col = mix(col, vec3(0.90, 0.13, 0.10), rec_dot * params.osd);
    }

    // Beam scanlines at sensor-line pitch, plus the odd/even field
    // flicker that makes cheap interlaced footage shimmer.
    float row = scr.y * lines;
    float beam = 0.5 + 0.5 * cos(6.2831853 * row);
    col *= 1.0 - params.scan_cam * beam;
    float field = mod(floor(row) + float(params.FrameCount % 2u), 2.0);
    col *= 1.0 - params.interlace * 0.22 * field;

    // Mains hum crawling up the tube.
    float bar = fract(scr.y - t * 0.08);
    col *= 1.0 - params.hum_cam * smoothstep(0.0, 0.22, bar) * smoothstep(0.5, 0.22, bar);

    col *= max(1.0 - params.vignette_cam * dot(c * 0.85, c * 0.85), 0.0) * on_tube;

    FragColor = vec4(mix(base, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
