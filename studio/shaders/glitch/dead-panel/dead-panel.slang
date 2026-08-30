#version 450

// dead-panel — physically cracked LCD emulation, single pass.
//
// What a dropped monitor actually looks like: a web of fracture lines
// radiating from the impact point, a black blotch where the liquid crystal
// pooled, iridescent pressure fringes around the damage (Newton's rings in
// the cell gap), stuck single-pixel driver lines running from the wound to
// the panel edge, scattered dead pixels, and — when the crack severed a
// driver flex — the far side of the panel showing the image torn sideways
// with column-driver banding.
//
// Mostly static, as physical damage is; only the failing driver lines
// flicker. amount = 0 is a clean passthrough.

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float amount;
    float pos_x;
    float pos_y;
    float cracks;
    float crack_len;
    float jag;
    float blotch;
    float fringe;
    float dead_lines;
    float flicker;
    float tear;
    float seed;
} params;

#pragma parameter amount     "Panel: amount"          1.0  0.0  1.0  0.01
#pragma parameter pos_x      "Panel: impact x"        0.68 0.0  1.0  0.01
#pragma parameter pos_y      "Panel: impact y"        0.30 0.0  1.0  0.01
#pragma parameter cracks     "Panel: crack count"     7.0  0.0 12.0  1.0
#pragma parameter crack_len  "Panel: crack length"    0.55 0.05 1.5  0.01
#pragma parameter jag        "Panel: crack jaggedness" 0.5 0.0  1.0  0.01
#pragma parameter blotch     "Panel: ink blotch size" 0.10 0.0  0.5  0.005
#pragma parameter fringe     "Panel: pressure fringe" 0.6  0.0  1.0  0.01
#pragma parameter dead_lines "Panel: stuck lines"     0.5  0.0  1.0  0.01
#pragma parameter flicker    "Panel: line flicker"    0.4  0.0  1.0  0.01
#pragma parameter tear       "Panel: sub-panel tear"  0.4  0.0  1.0  0.01
#pragma parameter seed       "Panel: seed"            5.0  0.0 64.0  1.0

/* Paint, not geometry: the ink blotch / cracks / dead pixels must stay
 * opaque black, not become holes — coverage keeps the clean silhouette. */
#pragma matte_ignore amount

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
float vnoise1(float x)
{
    float i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(hash1(i), hash1(i + 1.0), f);
}
float fbm1(float x)
{
    return vnoise1(x) * 0.6 + vnoise1(x * 2.7 + 13.0) * 0.3 + vnoise1(x * 6.1 + 31.0) * 0.1;
}

void main()
{
    vec2  uv  = vTexCoord;
    float amt = params.amount;
    float t   = float(params.FrameCount) * (1.0 / 60.0);
    float sd  = params.seed;

    float aspect = params.SourceSize.x / params.SourceSize.y;
    vec2  impact = vec2(params.pos_x, params.pos_y);
    vec2  p   = (uv - impact) * vec2(aspect, 1.0);
    float r   = length(p);
    float ang = atan(p.y, p.x);

    /* ------ sub-panel tear: the half-plane past the main fracture axis is
     * driven wrong — image shifted sideways plus column-driver banding. */
    float mainAng = hash1(sd * 3.7) * 6.2832;
    vec2  axisN   = vec2(cos(mainAng), sin(mainAng));
    float side    = smoothstep(0.0, 0.02, dot(p, axisN)) * step(0.001, params.tear);
    vec2  suv     = uv + side * vec2(params.tear * 0.02, params.tear * 0.006) * amt;
    vec3  col     = texture(Source, clamp(suv, 0.0, 1.0)).rgb;
    float colBand = step(0.5, fract(uv.x * params.SourceSize.x * 0.5));
    col *= 1.0 - side * params.tear * 0.25 * colBand * amt;

    /* ------ crack web: jagged spokes from the impact + one partial ring. */
    float crackD = 1e5;    /* distance-ish field to the nearest crack line */
    float n = clamp(params.cracks, 0.0, 12.0);
    for (int i = 0; i < 12; i++) {
        if (float(i) >= n) break;
        float fi   = float(i);
        float base = (fi + 0.3 + hash1(fi * 9.1 + sd) * 0.6) / max(n, 1.0) * 6.2832;
        /* spoke wanders as it travels: angle jitters with radius */
        float wob  = (fbm1(r * 14.0 + fi * 7.0 + sd * 2.0) - 0.5) * params.jag * 0.9;
        float da   = ang - (base + wob);
        da = atan(sin(da), cos(da));               /* wrap to [-pi, pi] */
        float len  = params.crack_len * (0.4 + 0.6 * hash1(fi * 3.3 + sd));
        float d    = abs(da) * r;                  /* arc distance to the spoke */
        d += step(len, r) * 1.0;                   /* kill beyond spoke length */
        crackD = min(crackD, d);
    }
    /* partial concentric ring crack around the impact */
    float ringR = params.blotch * 2.6 + 0.03;
    float ringD = abs(r - ringR)
                + step(0.25 + hash1(sd * 5.1) * 0.5, fract(ang * 0.15915 + hash1(sd * 7.7)));
    crackD = min(crackD, ringD);

    float px = params.SourceSize.z * aspect;       /* ~1px in p-space */
    float crackCore = 1.0 - smoothstep(0.0, px * 1.4, crackD);
    float crackGlow = 1.0 - smoothstep(0.0, px * 5.0, crackD);

    /* ------ ink blotch: pooled liquid crystal, black with a ragged edge. */
    float wobble = 0.7 + 0.6 * fbm1(ang * 1.4323 + sd * 11.0);   /* organic outline */
    float pulse  = 1.0 + 0.03 * sin(t * 0.7 + sd);               /* faint pressure breathing */
    float br     = params.blotch * wobble * pulse;
    float blob   = 1.0 - smoothstep(br * 0.75, br, r);

    /* ------ iridescent pressure fringes: Newton's rings hugging the blotch
     * and bleeding along the cracks. */
    float fzone  = (1.0 - smoothstep(br, br * 2.2 + 0.001, r)) * (1.0 - blob);
    fzone = max(fzone, crackGlow * 0.6);
    float phase  = r * 90.0 + fbm1(ang * 2.0 + sd) * 3.0;
    vec3  rainbow = 0.5 + 0.5 * cos(6.2832 * (phase * 0.15915 + vec3(0.0, 0.33, 0.67)));
    col = mix(col, rainbow, fzone * params.fringe * 0.55 * amt);

    /* crack lines: bright bleed edge around a dark core */
    col = mix(col, vec3(0.9, 0.95, 1.0), crackGlow * 0.35 * amt);
    col = mix(col, vec3(0.02), crackCore * amt);

    /* blotch on top of everything local */
    col = mix(col, vec3(0.0), blob * amt);

    /* ------ stuck driver lines: single-pixel columns (and a few rows) of
     * saturated color running from the damage to the panel edge. */
    float colId  = floor(uv.x * params.SourceSize.x);
    float rowId  = floor(uv.y * params.SourceSize.y);
    float near   = exp(-abs(uv.x - impact.x) * 6.0);   /* denser near the wound */
    float lgate  = step(1.0 - params.dead_lines * 0.02 * (0.3 + near), hash1(colId * 1.7 + sd));
    /* the line only runs on the side of the damage row it was torn toward */
    float lspan  = hash1(colId * 4.9 + sd) < 0.5
                 ? step(uv.y, impact.y) : step(impact.y, uv.y);
    float rgate  = step(1.0 - params.dead_lines * 0.008, hash1(rowId * 2.3 + sd * 1.7));

    float fl     = mix(1.0, 0.55 + 0.45 * sin(t * (30.0 + hash1(colId) * 40.0) + colId),
                       params.flicker);
    float sel    = hash1(colId * 7.7 + sd * 3.0);
    vec3  lineC  = sel < 0.25 ? vec3(1.0, 0.1, 0.1)
                 : sel < 0.50 ? vec3(0.1, 1.0, 0.1)
                 : sel < 0.75 ? vec3(0.2, 0.3, 1.0) : vec3(1.0);
    col = mix(col, lineC * fl, lgate * lspan * amt);
    col = mix(col, vec3(1.0) * fl, rgate * amt * 0.9);

    /* ------ dead / stuck pixels sprinkled around the damage. */
    vec2  pxc  = floor(uv * params.SourceSize.xy);
    float dpx  = hash21(pxc + sd);
    float dens = 0.0015 + 0.01 * exp(-r * 4.0);
    float dead = step(1.0 - dens, dpx);
    vec3  dcol = hash21(pxc + sd + 1.0) < 0.6 ? vec3(0.0) : vec3(1.0);
    col = mix(col, dcol, dead * amt);

    FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
