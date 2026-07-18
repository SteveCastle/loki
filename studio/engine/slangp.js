/*
 * slangfx-web — .slangp preset parser.
 *
 * Port of src/slangp.c. Parses libretro-format INI-style preset files into a
 * plain object describing the multi-pass chain: shader paths, scale rules,
 * sampling/wrap modes, external textures, and parameter overrides.
 */

export const SCALE_SOURCE = 'source';
export const SCALE_VIEWPORT = 'viewport';
export const SCALE_ABSOLUTE = 'absolute';

const WRAP_MODES = new Set(['clamp_to_border', 'clamp_to_edge', 'repeat', 'mirrored_repeat']);

const FORMATS = new Set([
  'R8_UNORM', 'R8G8_UNORM', 'R8G8B8A8_UNORM', 'R8G8B8A8_SRGB',
  'R10G10B10A2_UNORM', 'R16_UNORM', 'R16_SFLOAT', 'R16G16B16A16_UNORM',
  'R16G16B16A16_SFLOAT', 'R32_SFLOAT', 'R32G32B32A32_SFLOAT',
]);

function parseBool(s) {
  if (s == null) return null;
  const t = s.toLowerCase();
  if (t === 'true' || t === 'yes' || t === 'on' || t === '1') return true;
  if (t === 'false' || t === 'no' || t === 'off' || t === '0') return false;
  return null;
}

function parseFloatStrict(s) {
  if (s == null || s === '') return null;
  const v = parseFloat(s);
  return Number.isNaN(v) ? null : v;
}

function unquote(s) {
  if (s.length >= 2 && ((s[0] === '"' && s.endsWith('"')) || (s[0] === "'" && s.endsWith("'"))))
    return s.slice(1, -1);
  return s;
}

/* Tokenize an INI-style buffer into an ordered key=value map (last wins for
 * lookups, like kv_find scanning finds first — the C version finds the FIRST
 * occurrence; preserve that). */
function lexPairs(src) {
  const pairs = new Map(); // first occurrence wins, matching kv_find semantics
  for (let rawLine of src.split(/\r?\n/)) {
    const t = rawLine.trim();
    if (!t || t.startsWith('#')) continue;
    const eq = t.indexOf('=');
    if (eq < 0) { if (!pairs.has(t)) pairs.set(t, ''); continue; }
    const key = t.slice(0, eq).trim();
    const val = unquote(t.slice(eq + 1).trim());
    if (key && !pairs.has(key)) pairs.set(key, val);
  }
  return pairs;
}

function splitSemi(src) {
  if (!src) return [];
  return src.split(';').map((s) => s.trim()).filter(Boolean);
}

/* Split "prefix<digits>" -> [prefix, index] or null. */
function splitIndexed(key) {
  const m = key.match(/^(.*?)(\d+)$/);
  if (!m) return null;
  return [m[1], parseInt(m[2], 10)];
}

function makePass() {
  return {
    path: null,
    alias: null,
    filterLinear: false,
    mipmapInput: false,
    wrapMode: 'clamp_to_border',
    scaleTypeX: SCALE_SOURCE,
    scaleTypeY: SCALE_SOURCE,
    scaleX: 1.0,
    scaleY: 1.0,
    srgbFramebuffer: false,
    floatFramebuffer: false,
    fboFormat: null,          // one of FORMATS or null for default
    frameCountMod: 0,
  };
}

const PASS_PREFIXES_WITH_UNDERSCORE = new Set([
  'scale_', 'filter_linear_', 'mipmap_input_', 'wrap_mode_', 'frame_count_mod_',
  'fbo_format_', 'shader_', 'alias_', 'srgb_framebuffer_', 'float_framebuffer_',
]);

function applyPassField(passes, key, value) {
  const split = splitIndexed(key);
  if (!split) return false;
  let [prefix, idx] = split;
  if (idx < 0 || idx >= passes.length) return false;

  // Tolerate the `scale_<i>` style: strip one trailing underscore when the
  // prefix isn't a per-axis variant.
  if (prefix.endsWith('_') && PASS_PREFIXES_WITH_UNDERSCORE.has(prefix))
    prefix = prefix.slice(0, -1);

  const ps = passes[idx];
  switch (prefix) {
    case 'shader': ps.path = value; return true;
    case 'alias': ps.alias = value; return true;
    case 'filter_linear': { const b = parseBool(value); if (b !== null) ps.filterLinear = b; return true; }
    case 'mipmap_input': { const b = parseBool(value); if (b !== null) ps.mipmapInput = b; return true; }
    case 'wrap_mode': if (WRAP_MODES.has(value.toLowerCase())) ps.wrapMode = value.toLowerCase(); return true;
    case 'frame_count_mod': { const v = parseInt(value, 10); if (!Number.isNaN(v)) ps.frameCountMod = v; return true; }
    case 'srgb_framebuffer': { const b = parseBool(value); if (b !== null) ps.srgbFramebuffer = b; return true; }
    case 'float_framebuffer': { const b = parseBool(value); if (b !== null) ps.floatFramebuffer = b; return true; }
    case 'fbo_format': { const up = value.toUpperCase(); if (FORMATS.has(up)) ps.fboFormat = up; return true; }
    case 'scale': { const v = parseFloatStrict(value); if (v !== null) { ps.scaleX = v; ps.scaleY = v; } return true; }
    case 'scale_x': { const v = parseFloatStrict(value); if (v !== null) ps.scaleX = v; return true; }
    case 'scale_y': { const v = parseFloatStrict(value); if (v !== null) ps.scaleY = v; return true; }
    case 'scale_type': {
      const t = value.toLowerCase();
      if (t === SCALE_SOURCE || t === SCALE_VIEWPORT || t === SCALE_ABSOLUTE) { ps.scaleTypeX = t; ps.scaleTypeY = t; }
      return true;
    }
    case 'scale_type_x': {
      const t = value.toLowerCase();
      if (t === SCALE_SOURCE || t === SCALE_VIEWPORT || t === SCALE_ABSOLUTE) ps.scaleTypeX = t;
      return true;
    }
    case 'scale_type_y': {
      const t = value.toLowerCase();
      if (t === SCALE_SOURCE || t === SCALE_VIEWPORT || t === SCALE_ABSOLUTE) ps.scaleTypeY = t;
      return true;
    }
    default: return false;
  }
}

/* Resolve `p` against `baseDir` unless absolute. URL-friendly: collapses
 * "a/b/../c" so fetch() paths stay canonical. */
export function resolvePath(baseDir, p) {
  if (p == null) return null;
  const abs = /^([a-zA-Z]:[\\/]|[\\/])/.test(p) || /^[a-z]+:\/\//.test(p);
  let joined = abs || !baseDir ? p : `${baseDir.replace(/[\\/]+$/, '')}/${p}`;
  joined = joined.replace(/\\/g, '/');
  // Normalize "." and ".." segments (keep leading root / drive / scheme).
  const m = joined.match(/^([a-z]+:\/\/[^/]*\/|[a-zA-Z]:\/|\/)?(.*)$/);
  const root = m[1] || '';
  const out = [];
  for (const seg of m[2].split('/')) {
    if (seg === '' || seg === '.') continue;
    if (seg === '..' && out.length && out[out.length - 1] !== '..') out.pop();
    else out.push(seg);
  }
  return root + out.join('/');
}

export function dirnameOf(p) {
  const norm = p.replace(/\\/g, '/');
  const i = norm.lastIndexOf('/');
  return i < 0 ? '.' : norm.slice(0, i);
}

/**
 * Parse a .slangp from a string.
 * @param {string} src preset text
 * @param {string|null} baseDir directory for resolving relative paths
 * @returns {{baseDir, passes, textures, params}}
 */
export function parsePreset(src, baseDir = null) {
  const pairs = lexPairs(src);

  const shadersStr = pairs.get('shaders');
  const nPasses = shadersStr != null ? parseInt(shadersStr, 10) : NaN;
  if (Number.isNaN(nPasses)) throw new Error("missing or invalid 'shaders' count");

  const passes = Array.from({ length: nPasses }, makePass);
  for (const [key, value] of pairs) applyPassField(passes, key, value);
  for (const ps of passes) if (ps.path && baseDir) ps.path = resolvePath(baseDir, ps.path);

  const textures = [];
  for (const name of splitSemi(pairs.get('textures'))) {
    const tex = {
      name,
      path: resolvePath(baseDir, pairs.get(name) ?? ''),
      filterLinear: false,
      mipmap: false,
      wrapMode: 'clamp_to_border',
    };
    const lin = parseBool(pairs.get(`${name}_linear`));
    if (lin !== null) tex.filterLinear = lin;
    const mip = parseBool(pairs.get(`${name}_mipmap`));
    if (mip !== null) tex.mipmap = mip;
    const wrap = pairs.get(`${name}_wrap_mode`);
    if (wrap && WRAP_MODES.has(wrap.toLowerCase())) tex.wrapMode = wrap.toLowerCase();
    textures.push(tex);
  }

  const params = [];
  for (const name of splitSemi(pairs.get('parameters'))) {
    const v = parseFloatStrict(pairs.get(name));
    if (v !== null) params.push({ name, value: v });
  }

  return { baseDir, passes, textures, params };
}
