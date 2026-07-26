#version 450

// vhs-bootleg — pass 1: the tape and the deck.
//
// Everything that happens between the oxide and the picture on a tape
// that has been copied too many times and played too often: the timebase
// wanders line to line, the whole picture sways where the head first
// meets the tape, a ragged band of mistracked lines crawls up the frame,
// oxide dropouts flash as bright dashes, the head-switching point tears
// the bottom of the picture, the vertical hold gives out every so often,
// and the deck's AGC breathes on the whole thing.
//
// Runs on the degraded signal from vhs-chroma.slang and mixes back
// against the untouched Original, so amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float Time;
    float amount;
    float overscan;
    float tape_lines;
    float jitter;
    float wobble;
    float tracking;
    float dropout;
    float snow;
    float headswitch;
    float roll;
    float comb;
    float pump;
    float edge;
} params;

#pragma parameter amount     "VHS: mix"            1.0   0.0   1.0  0.01
#pragma parameter overscan   "VHS: overscan"       1.03  1.0   1.2  0.005
#pragma parameter tape_lines "VHS: tape lines"     240.0 80.0 720.0 4.0
#pragma parameter jitter     "VHS: line jitter"    2.5   0.0  16.0  0.1
#pragma parameter wobble     "VHS: tape wobble"    1.8   0.0  12.0  0.1
#pragma parameter tracking   "VHS: tracking error" 0.5   0.0   1.0  0.01
#pragma parameter dropout    "VHS: dropouts"       0.45  0.0   1.0  0.01
#pragma parameter snow       "VHS: snow"           0.07  0.0   0.5  0.01
#pragma parameter headswitch "VHS: head switch"    0.6   0.0   1.0  0.01
#pragma parameter roll       "VHS: vertical hold"  0.5   0.0   1.0  0.01
#pragma parameter comb       "VHS: interlace comb" 0.5   0.0   1.0  0.01
#pragma parameter pump       "VHS: AGC pump"       0.5   0.0   1.0  0.01
#pragma parameter edge       "VHS: worn edges"     0.6   0.0   1.0  0.01

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

float hash3(vec3 p)
{
    p = fract(p * vec3(123.34, 345.45, 678.56));
    p += dot(p, p.yzx + 34.345);
    return fract((p.x + p.y) * p.z);
}

void main()
{
    vec3  base = texture(Original, vTexCoord).rgb;
    float t = params.Time;

    // A dub taped off a screen is always a little zoomed and cropped.
    vec2 uv = (vTexCoord - 0.5) / max(params.overscan, 0.5) + 0.5;

    // Vertical hold giving out: every few seconds the picture climbs, then
    // the deck catches it again.
    float win = floor(t / 6.0);
    float ph  = fract(t / 6.0);
    float go  = step(0.68, hash21(vec2(win, 5.3))) * params.roll;
    float env = smoothstep(0.0, 0.05, ph) * smoothstep(0.55, 0.25, ph);
    uv.y = fract(uv.y + go * env * fract(t * 0.9));

    // Tape line pitch — VHS resolves a few hundred lines, not the comp's.
    float lines = max(params.tape_lines, 60.0);
    float line_id = floor(uv.y * lines);

    // Timebase error: every line starts a hair early or late.
    float offx = (hash21(vec2(line_id, floor(t * 24.0))) - 0.5) * params.jitter;

    // Tape sway, worst at the top of the picture where the head enters.
    float wob = sin(uv.y * 8.0 + t * 2.1) * 0.6 + sin(uv.y * 21.0 - t * 1.3) * 0.4;
    offx += wob * params.wobble * (1.0 + (1.0 - uv.y) * 0.8);

    // Tracking: a ragged band of badly-read lines crawling up the frame.
    float band_y = fract(0.25 - t * 0.045);
    float band_d = abs(fract(uv.y - band_y + 0.5) - 0.5);
    float in_band = smoothstep(0.075, 0.0, band_d) * params.tracking;
    offx += in_band * (hash21(vec2(line_id, floor(t * 60.0))) - 0.5) * 42.0;

    // Head-switching point: the bottom few lines of the *displayed* frame
    // are torn sideways — it sits below the picture the overscan crops to,
    // so it keys off screen space, not the zoomed source.
    float hs_h = max(params.headswitch, 0.001) * 0.06;
    float hs = smoothstep(1.0 - hs_h, 1.0 - hs_h * 0.35, vTexCoord.y);
    offx += hs * 30.0 * (0.6 + 0.4 * sin(t * 47.0));

    // Sample on the tape's line grid, with the accumulated line offset.
    vec2 suv = vec2(clamp(uv.x + offx * params.SourceSize.z, 0.002, 0.998),
                    clamp((line_id + 0.5) / lines, 0.002, 0.998));
    vec3 col = texture(Source, suv).rgb;

    float n = hash3(vec3(floor(vTexCoord * params.OutputSize.xy * vec2(0.5, 1.0)),
                         float(params.FrameCount % 1024u)));

    // Dropouts: oxide missing from the tape, read as short dashes — mostly
    // bright, occasionally a dead black one. Gated twice on purpose: a
    // handful of lines are hit, and only part of each of those lines. One
    // flat probability per cell carpets the whole frame in confetti.
    float dt = floor(t * 15.0);
    float hit_line = step(0.97, hash21(vec2(line_id, dt + 61.0)));
    float seg = floor(suv.x * 26.0) + line_id * 13.0;
    float drop = hit_line * step(0.72, hash21(vec2(seg, dt))) * params.dropout;
    float dark = step(0.78, hash21(vec2(seg, 3.0)));
    col = mix(col, mix(vec3(0.88, 0.90, 0.93), vec3(0.02), dark), drop * 0.9);

    // Inside the tracking band the deck is guessing: noise, no colour.
    col = mix(col, vec3(luma(col) * 0.55 + n * 0.45), in_band * 0.85);

    // Snow, then the noise strip between the heads along the very bottom.
    col += (n - 0.5) * params.snow;
    float hs_noise = smoothstep(1.0 - hs_h * 0.45, 1.0 - hs_h * 0.1, vTexCoord.y)
                   * params.headswitch;
    col = mix(col, vec3(n * 0.8 + 0.06), hs_noise * 0.85);

    // While the picture is rolling, the bar that normally hides above the
    // top of the screen rides through it — the wrap seam is exactly where
    // uv.y just came back around.
    float rolling = step(0.0005, go * env * fract(t * 0.9));
    col = mix(col, vec3(n * 0.22), rolling * smoothstep(0.02, 0.0, uv.y) * 0.9);

    // Interlace comb + the AGC breathing on the whole picture.
    float even = mod(floor(vTexCoord.y * params.OutputSize.y), 2.0);
    col *= 1.0 - params.comb * 0.12 * even;
    col *= 1.0 + params.pump * 0.09 * sin(t * 1.9 + sin(t * 0.63) * 2.4);

    // Worn edges: the sides of the tape are the first thing to go.
    float e = smoothstep(0.0, 0.035, uv.x) * smoothstep(1.0, 0.965, uv.x)
            * smoothstep(0.0, 0.02, uv.y);
    col *= mix(1.0, e, params.edge);

    FragColor = vec4(mix(base, clamp(col, 0.0, 1.0), params.amount), 1.0);
}
