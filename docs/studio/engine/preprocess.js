/*
 * slangfx-web — .slang source preprocessing.
 *
 * Port of the preprocessing half of src/slang_compile.c, plus the two
 * source rewrites WebGPU requires:
 *
 *   1. `layout(push_constant) uniform Push {...}` → std140 UBO at
 *      set=1, binding=0. WebGPU has no push constants; a dedicated bind
 *      group avoids binding collisions with the shader's own resources.
 *   2. Combined `sampler2D` split into `texture2D` + `sampler` (tint's
 *      SPIR-V reader rejects OpTypeSampledImage variables). The sampler
 *      objects live in bind group 2, numbered in declaration order. A
 *      #define keeps every use site (including function arguments)
 *      compiling unchanged.
 *
 * Pipeline: flatten #include → rewrite → collect/strip pragmas → split
 * on `#pragma stage`.
 */

import { dirnameOf, resolvePath } from './slangp.js';

/** Flatten #include "..." directives recursively (matches the C flattener:
 * runs BEFORE stage splitting so pragmas inside includes are seen).
 * @param {string} src
 * @param {string} fileDir directory of the including file
 * @param {(path: string) => Promise<string>} readFile
 */
export async function flattenIncludes(src, fileDir, readFile, depth = 0) {
  if (depth > 32) throw new Error('#include depth >32');
  const out = [];
  for (const line of src.split(/\r?\n/)) {
    const m = line.match(/^\s*#include\s+(["<])([^">]+)[">]/);
    if (m) {
      const resolved = resolvePath(fileDir, m[2]);
      let content;
      try {
        content = await readFile(resolved);
      } catch (e) {
        throw new Error(`could not open include '${resolved}': ${e.message}`);
      }
      out.push('');
      out.push(await flattenIncludes(content, dirnameOf(resolved), readFile, depth + 1));
      out.push('');
    } else {
      out.push(line);
    }
  }
  return out.join('\n');
}

/* Parse `#pragma parameter NAME "DESC" DEFAULT MIN MAX STEP` argument tail.
 * Tolerant of missing fields beyond NAME and DEFAULT. */
function parseParamLine(args) {
  let rest = args.trim();
  const nameMatch = rest.match(/^(\S+)\s*/);
  if (!nameMatch) return null;
  const name = nameMatch[1];
  rest = rest.slice(nameMatch[0].length);

  let desc = '';
  const descMatch = rest.match(/^"([^"]*)"\s*/);
  if (descMatch) { desc = descMatch[1]; rest = rest.slice(descMatch[0].length); }

  const nums = [];
  for (const tok of rest.trim().split(/\s+/)) {
    const v = parseFloat(tok);
    if (Number.isNaN(v)) break;
    nums.push(v);
    if (nums.length === 4) break;
  }
  return {
    name,
    desc,
    default: nums.length >= 1 ? nums[0] : 0.0,
    min: nums.length >= 2 ? nums[1] : 0.0,
    max: nums.length >= 3 ? nums[2] : 1.0,
    step: nums.length >= 4 ? nums[3] : 0.01,
  };
}

/* Identifiers that are legal in GLSL but keywords / reserved words in WGSL.
 * tint refuses to emit them (debug names are preserved verbatim), so we
 * rename `X` → `X_sfx` before compilation. The rename runs on the flattened
 * source before pragma collection, so #pragma parameter names, SPIR-V
 * reflection, and WGSL all agree. GLSL keywords and builtin function names
 * (e.g. `mod`, `filter` is GLSL-reserved already) must NOT be listed. */
const WGSL_RESERVED_GLSL_IDENTIFIERS = [
  'macro', 'module', 'mut', 'impl', 'trait', 'crate', 'self', 'super',
  'handle', 'premerge', 'regardless', 'unless', 'meta', 'final', 'override',
  'fallthrough', 'demote', 'require', 'resource', 'typeof', 'typeid',
  'yield', 'become', 'debugger', 'instanceof', 'null', 'nullptr', 'std',
  'wgsl', 'async', 'await', 'where', 'type', 'ref', 'ptr',
];
const RESERVED_RE = new RegExp(`\\b(${WGSL_RESERVED_GLSL_IDENTIFIERS.join('|')})\\b`, 'g');

/** Rename a single identifier the same way the source rewrite does. */
export function renameReserved(name) {
  return WGSL_RESERVED_GLSL_IDENTIFIERS.includes(name) ? `${name}_sfx` : name;
}

/* GLSL builtin texture functions whose first argument is a combined
 * sampler. At these call sites a `sampler2D(tex, smp)` constructor is legal
 * ("point of use"); everywhere else the pair must be passed as two args. */
const TEXTURE_BUILTINS =
  'texture|textureLod|textureGrad|textureOffset|textureLodOffset|textureProj|' +
  'texelFetch|texelFetchOffset|textureSize|textureGather|textureGatherOffset|' +
  'textureQueryLod|textureQueryLevels';

/* `//@param NAME "Label" DEFAULT MIN MAX STEP` — one-line sugar that
 * declares a tunable float and its UI slider together. Each line expands to
 * a `#pragma parameter` (→ slider metadata), a `float NAME;` member
 * injected into the shader's push_constant block (or an auto-generated one
 * when the shader has none), and a `#define NAME <block>.NAME` so the body
 * references it bare. Everything after the expansion is plain libretro
 * slang, so the sugar works in .slang files and hand-written editor
 * shaders alike. */
function expandAutoParams(src) {
  const names = [];
  let text = src.replace(/^[ \t]*\/\/@param[ \t]+(\w+)[ \t]*(.*?)[ \t]*\r?$/gm, (all, name, rest) => {
    names.push(name);
    return `#pragma parameter ${name} ${rest || `"${name}" 0.5 0.0 1.0 0.01`}`;
  });
  if (!names.length) return text;

  const members = names.map((n) => `    float ${n};`).join('\n');
  const pushBlock = /(layout\s*\(\s*push_constant\s*\)\s*uniform\s+\w+\s*\{)([^}]*)(\}\s*(\w+)\s*;)/;
  const m = text.match(pushBlock);
  if (m) {
    const defines = names.map((n) => `#define ${n} ${m[4]}.${n}`).join('\n');
    text = text.replace(pushBlock, (all, open, body, close) =>
      `${open}${body}\n${members}\n${close}\n${defines}\n`);
  } else {
    const block =
      `layout(push_constant) uniform SlangfxAutoParams\n{\n${members}\n} slangfx_auto_;\n` +
      names.map((n) => `#define ${n} slangfx_auto_.${n}`).join('\n') + '\n';
    text = text.replace(/^([ \t]*#version[^\n]*\n)/, `$1${block}`);
  }
  return text;
}

/* Apply the WebGPU rewrites to the full (flattened) source, BEFORE stage
 * splitting so sampler slot numbering is consistent across stages.
 *
 * tint's SPIR-V reader rejects combined image-samplers and glslang only
 * allows `sampler2D(t, s)` constructors directly inside builtin texture
 * calls, so the split has to be a real source transform:
 *   1. global `uniform sampler2D N;` → `texture2D N_tex; sampler N_smp;`
 *      (sampler objects collected into bind group 2, declaration order)
 *   2. `#define A B` alias macros where B is a sampler → A treated as a
 *      sampler rooted at B, macro line dropped
 *   3. function params `sampler2D p` → `texture2D p_tex, sampler p_smp`
 *   4. sampler names in builtin texture calls → `sampler2D(R_tex, R_smp)`
 *   5. remaining bare sampler names (user-function call args) → `R_tex, R_smp`
 *
 * Returns the rewritten text plus the discovered sampler table. */
function rewriteForWebGPU(src) {
  let text = expandAutoParams(src);
  text = text.replace(RESERVED_RE, '$1_sfx');
  text = text.replace(
    /layout\s*\(\s*push_constant\s*\)\s*uniform/g,
    'layout(std140, set = 1, binding = 0) uniform'
  );

  // 1. Split global declarations.
  const samplers = [];
  const roots = new Map(); // symbol -> root symbol whose _tex/_smp pair exists
  text = text.replace(
    /layout\s*\(([^)]*)\)\s*uniform\s+sampler2D\s+(\w+)\s*;/g,
    (all, layoutArgs, name) => {
      const setM = layoutArgs.match(/set\s*=\s*(\d+)/);
      const bindM = layoutArgs.match(/binding\s*=\s*(\d+)/);
      const smpIndex = samplers.length;
      samplers.push({
        name,
        texSet: setM ? parseInt(setM[1], 10) : 0,
        texBinding: bindM ? parseInt(bindM[1], 10) : 0,
        smpBinding: smpIndex, // bind group 2
      });
      roots.set(name, name);
      return (
        `layout(${layoutArgs}) uniform texture2D ${name}_tex;\n` +
        `layout(set = 2, binding = ${smpIndex}) uniform sampler ${name}_smp;`
      );
    }
  );

  // 2. Resolve `#define alias sampler` macros (chains too), drop the lines.
  let grew = true;
  while (grew) {
    grew = false;
    text = text.replace(/^[ \t]*#define[ \t]+(\w+)[ \t]+(\w+)[ \t]*\r?$/gm, (all, a, b) => {
      if (roots.has(b)) {
        if (!roots.has(a) || roots.get(a) !== roots.get(b)) {
          roots.set(a, roots.get(b));
          grew = true;
        }
        return '';
      }
      return all;
    });
  }

  // 3. Function parameters of type sampler2D.
  text = text.replace(
    /([(,]\s*)(?:in\s+)?sampler2D\s+(\w+)/g,
    (all, lead, name) => {
      roots.set(name, name);
      return `${lead}texture2D ${name}_tex, sampler ${name}_smp`;
    }
  );

  // 4 + 5. Rewrite uses. Skip remaining preprocessor lines so `#define
  // FOO texture(Source, ...)`-style macro bodies still get step 4 applied,
  // while `#undef`/includes/etc. are left alone structurally.
  if (roots.size > 0) {
    const symbol = [...roots.keys()].map((s) => s.replace(/[$]/g, '\\$&')).join('|');
    const builtinCall = new RegExp(`\\b(${TEXTURE_BUILTINS})\\s*\\(\\s*(${symbol})\\b`, 'g');
    const bare = new RegExp(`\\b(${symbol})\\b(?!_tex\\b|_smp\\b)`, 'g');
    text = text
      .replace(builtinCall, (all, fn, name) => `${fn}(sampler2D(${roots.get(name)}_tex, ${roots.get(name)}_smp)`)
      .replace(bare, (all, name) => `${roots.get(name)}_tex, ${roots.get(name)}_smp`);
  }

  return { text, samplers };
}

/**
 * Preprocess a flattened .slang source into per-stage GLSL.
 * @param {string} flatSrc source with includes already flattened
 * @returns {{vertexGlsl, fragmentGlsl, params, formatPragma, samplers}}
 */
export function preprocessSlang(flatSrc) {
  const { text, samplers } = rewriteForWebGPU(flatSrc);

  const header = [];
  const vert = [];
  const frag = [];
  let target = header;
  let sawStage = false;
  const params = [];
  let formatPragma = null;

  for (const line of text.split('\n')) {
    const m = line.match(/^\s*#pragma\s+(\S+)\s*(.*?)\s*$/);
    if (m) {
      const [, kind, args] = m;
      if (kind === 'stage') {
        if (!sawStage) {
          // Commit the shared header into both stage buffers.
          vert.push(...header);
          frag.push(...header);
          sawStage = true;
        }
        if (args === 'vertex') target = vert;
        else if (args === 'fragment') target = frag;
        else throw new Error(`unknown #pragma stage value '${args}'`);
        target.push('');
        continue;
      }
      if (kind === 'parameter') {
        const p = parseParamLine(args);
        if (p) params.push(p);
        target.push('');
        continue;
      }
      if (kind === 'format') {
        formatPragma = args || null;
        target.push('');
        continue;
      }
      if (kind === 'name') { target.push(''); continue; }
      // Unknown pragma: keep (GLSL ignores unknown pragmas).
    }
    target.push(line);
  }

  if (!sawStage || vert.length === 0 || frag.length === 0)
    throw new Error('slang source is missing #pragma stage vertex or fragment');

  return {
    vertexGlsl: vert.join('\n'),
    fragmentGlsl: frag.join('\n'),
    params,
    formatPragma,
    samplers,
  };
}
