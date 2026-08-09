#version 450

// camera-shake — a virtual camera rig with a low note going through it.
//
// The primary axis is Z, not X/Y: bass reaches a camera as a vibration
// along the lens, so the frame surges forward and back (`thrust`) and the
// lateral travel (`shake`) is the smaller, secondary component. `punch` is
// the one-way shove into the shot that a real impact adds on top of the
// oscillation, and `roll` is the rig twisting on its mount. Z runs a third
// faster than the lateral axes, the way a stiffer mode does on real gear.
//
// The movement itself is generated here, from multi-octave value noise on
// Time, so the thing you automate is how HARD the camera is hit, not where
// it is. Drive `hit` from a bass envelope and the camera vibrates through
// the note instead of sliding with it — which is the difference between a
// shake and a pan. `idle` is the always-there handheld float underneath.
//
// Two things keep the frame edge clean, which matters most when the shake
// is meant to go unnoticed:
//
//   cover   pre-zooms by exactly enough to hold the widest move the current
//           settings can make — measured at a FULL hit, not the current
//           one, so the framing never breathes with the music.
//   soften  blurs anything that still reaches past the edge (cover < 1, or
//           the exact peak of a move) by fanning those samples into a disc
//           instead of letting them all clamp to the same pixel. A hard
//           smeared band is obvious; defocus is not.
//
// Realistic: thrust around 2 %, little lateral travel, low roughness, rate
// 4–8 Hz, response ≥ 1.5 so only the real hits move it. Over-emphasised:
// thrust 8 %+, roughness 1, rate 12–20 Hz, response < 1 so everything
// shoves the camera.

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OriginalSize;
    vec4  OutputSize;
    uint  FrameCount;
    float Time;
    float hit;
    float idle;
    float thrust;
    float shake;
    float roll;
    float punch;
    float rate;
    float rough;
    float bias;
    float response;
    float blur;
    float cover;
    float soften;
} params;

#pragma parameter hit      "CameraShake: hit — automate with bass" 0.0  0.0  1.0  0.01
#pragma parameter idle     "CameraShake: idle handheld"            0.15 0.0  1.0  0.01
#pragma parameter thrust   "CameraShake: push/pull, Z (%)"         2.0  0.0  25.0 0.1
#pragma parameter shake    "CameraShake: lateral travel (% of h)"  1.0  0.0  12.0 0.1
#pragma parameter roll     "CameraShake: roll (deg)"               0.4  0.0  8.0  0.05
#pragma parameter punch    "CameraShake: punch-in on hits (%)"     1.0  0.0  15.0 0.1
#pragma parameter rate     "CameraShake: rate (Hz)"                6.0  0.5  30.0 0.1
#pragma parameter rough    "CameraShake: roughness"                0.45 0.0  1.0  0.01
#pragma parameter bias     "CameraShake: vertical bias"            0.65 0.0  1.0  0.01
#pragma parameter response "CameraShake: hit response curve"       1.0  0.25 4.0  0.05
#pragma parameter blur     "CameraShake: motion blur"              0.6  0.0  3.0  0.05
#pragma parameter cover    "CameraShake: edge cover"               1.0  0.0  1.0  0.01
#pragma parameter soften   "CameraShake: edge softening (%)"       1.5  0.0  10.0 0.1

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

#define TAPS 9
#define NOISE_WRAP 4096.0

float hash11(float p)
{
    float q = fract(p * 0.1031);
    q *= q + 33.33;
    q *= q + q;
    return fract(q);
}

/* Value noise in [-1,1], smooth between whole cycles. The phase is wrapped
 * so the hash keeps its float precision however long the comp runs (the
 * pattern only repeats after 4096 cycles — minutes, at any usable rate). */
float vnoise1(float x)
{
    float w = mod(x, NOISE_WRAP);
    float i = floor(w);
    float f = w - i;
    f = f * f * (3.0 - 2.0 * f);
    return mix(hash11(i), hash11(i + 1.0), f) * 2.0 - 1.0;
}

/* Higher octaves fade in with `rough`: 0 is a heavy rig swaying on its
 * mount, 1 is a light one rattling. Normalised so the peak stays near 1
 * either way, which is what keeps `cover` honest. */
float fbm1(float x)
{
    float w1 = params.rough * 0.55;
    float w2 = params.rough * params.rough * 0.30;
    float a  = vnoise1(x)
             + w1 * vnoise1(x * 2.17 + 19.37)
             + w2 * vnoise1(x * 4.41 + 47.11);
    return a / (1.0 + w1 + w2);
}

/* Past the picture there is nothing to read, and a clamp would drag one
 * row of pixels out into a band you cannot miss. Fold the coordinate back
 * instead: mirrored content continues the frame plausibly, and the disc
 * blur in main() smooths the fold away. */
vec2 mirror_uv(vec2 s)
{
    vec2 m = mod(abs(s), 2.0);
    return min(m, 2.0 - m);
}

/* Where this pixel reads the source at time tt. The whole camera move lives
 * in here so the motion blur can simply walk the line between two of these
 * — roll and punch-in smear correctly, not just the translation. */
vec2 rig_uv(float tt, vec2 uv, float env, float zoom0, float asp)
{
    float ph = tt * params.rate;
    float nx = fbm1(ph);
    float ny = fbm1(ph + 137.4);
    float nr = fbm1(ph * 0.63 + 311.2);
    /* Z runs faster than the lateral sway — a longitudinal mode is the
     * stiff one, and it is what makes this read as a note travelling
     * through the camera rather than an operator wobbling. */
    float nz = fbm1(ph * 1.3 + 573.9);

    /* A kick shoves a camera up and down more than side to side; `bias`
     * leans the travel one way without changing how far it goes. */
    float trav = params.shake * 0.01;
    float ax = trav * (1.0 - 0.9 * max(0.0, 2.0 * params.bias - 1.0));
    float ay = trav * (1.0 - 0.9 * max(0.0, 1.0 - 2.0 * params.bias));

    vec2  off  = vec2(nx * ax, ny * ay) * env;
    float ang  = radians(nr * params.roll) * env;
    /* Surge along the lens: thrust swings both ways around the resting
     * frame, punch only ever shoves in. */
    float zoom = zoom0 * (1.0 + (params.punch + params.thrust * nz) * 0.01 * env);

    /* Height-normalised space: a roll stays round and the travel means the
     * same thing on any aspect ratio. */
    vec2 q = (uv - 0.5) * vec2(asp, 1.0) / zoom;
    float ca = cos(ang), sa = sin(ang);
    q = vec2(q.x * ca - q.y * sa, q.x * sa + q.y * ca) + off;
    return q / vec2(asp, 1.0) + 0.5;
}

void main()
{
    vec2  uv  = vTexCoord;
    float asp = params.OutputSize.x * params.OutputSize.w;

    /* The idle float is always there; the automated hit rides on top of it,
     * shaped by the response curve (>1 = only the big hits land). */
    float env = params.idle + pow(clamp(params.hit, 0.0, 1.0), params.response);

    /* Pre-zoom for the widest move these settings can make — at a FULL hit,
     * not the current one, so the framing holds still between hits. Solved
     * from "rotated half-frame over zoom, plus travel, still fits", then
     * divided by the deepest pull-back so the surge has room to go the
     * other way too. */
    float envMax = params.idle + 1.0;
    float trav   = params.shake * 0.01 * envMax;
    float amax   = radians(params.roll) * envMax;
    float pull   = min(params.thrust * 0.01 * envMax, 0.85);
    float zy     = (1.0 + amax * asp) / max(1.0 - 2.0 * trav,       0.15);
    float zx     = (1.0 + amax / asp) / max(1.0 - 2.0 * trav / asp, 0.15);
    float zoom0  = mix(1.0, max(max(zy, zx), 1.0) / (1.0 - pull), params.cover);

    /* Half a shutter either side of now: 1.0 ≈ a 1/50s exposure. */
    float hs  = params.blur * 0.01;
    vec2  uvA = rig_uv(params.Time - hs, uv, env, zoom0, asp);
    vec2  uvB = rig_uv(params.Time + hs, uv, env, zoom0, asp);

    /* Width of the softened band, in height units. The ramp starts a hair
     * inside the edge so the softening has no onset of its own. */
    float feather = max(params.soften * 0.01, 1e-5);

    vec3 acc = vec3(0.0);
    for (int k = 0; k < TAPS; ++k)
    {
        float f = (float(k) + 0.5) / float(TAPS);
        vec2  s = mix(uvA, uvB, f);

        /* Signed distance from this sample to the nearest source edge, in
         * height units — negative once it has run off the picture. */
        vec2  din = vec2(min(s.x, 1.0 - s.x) * asp, min(s.y, 1.0 - s.y));
        float over = clamp((feather * 0.25 - min(din.x, din.y)) / feather, 0.0, 1.5);

        /* Off the edge, the taps fan out over a disc (golden-angle points)
         * rather than all landing on the same place: mirrored content plus
         * that defocus reads as picture, not as a border. Inside the frame
         * the disc is zero-radius and the taps stay strung along the
         * motion path, so nothing here softens the shot itself. */
        float ang = 2.39996 * float(k);
        vec2  dir = vec2(cos(ang), sin(ang)) * sqrt((float(k) + 0.5) / float(TAPS));
        float rad = over * feather * 0.8;
        s += dir * vec2(rad / asp, rad);

        acc += texture(Source, mirror_uv(s)).rgb;
    }
    FragColor = vec4(clamp(acc / float(TAPS), 0.0, 1.0), 1.0);
}
