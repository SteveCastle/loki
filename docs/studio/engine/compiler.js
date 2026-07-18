/*
 * slangfx-web — slang shader → WGSL compilation.
 *
 * Mirrors src/slang_compile.c but targets WebGPU: the wasm toolchain is
 * glslang (GLSL 450 → SPIR-V) followed by tint/twgsl (SPIR-V → WGSL).
 * Reflection runs on our own SPIR-V walk (spv-reflect.js) so uniform
 * offsets are authoritative for the buffers we fill each frame.
 */

import { flattenIncludes, preprocessSlang } from './preprocess.js';
import { reflectSpirv } from './spv-reflect.js';
import { dirnameOf } from './slangp.js';

/**
 * Compile one .slang source into WGSL + reflection metadata.
 *
 * @param {string} source raw .slang text
 * @param {object} opts
 * @param {string} opts.path source path (for #include resolution + errors)
 * @param {(path: string) => Promise<string>} opts.readFile text loader
 * @param {{compileGLSL: Function}} opts.glslang
 * @param {{convertSpirV2WGSL: Function}} opts.twgsl
 * @returns {Promise<SlangModule>}
 */
export async function compileSlang(source, { path = '<inline>', readFile, glslang, twgsl, textureLodWorkaround = false }) {
  const flat = await flattenIncludes(source, dirnameOf(path), readFile);
  const pre = preprocessSlang(flat);

  // WGSL forbids textureSample() in non-uniform control flow. When a shader
  // trips Chrome's uniformity analysis, callers retry with this flag: plain
  // texture() becomes an explicit base-level lookup, which is uniformity-
  // safe and identical for the single-mip textures these effects sample.
  if (textureLodWorkaround) {
    pre.fragmentGlsl = pre.fragmentGlsl.replace(
      /^(\s*#version[^\n]*\n)/,
      '$1#define texture(s, uv) textureLod(s, uv, 0.0)\n'
    );
  }

  // glslang.wasm reports its useful diagnostics (line-numbered GLSL errors)
  // through console.error/warn, not the thrown exception. Capture them so
  // editor UIs can show the real message.
  const compileStage = (glsl, stage) => {
    if (globalThis.__slangfxGlslangLog) globalThis.__slangfxGlslangLog.length = 0;
    try {
      return glslang.compileGLSL(glsl, stage, true);
    } catch (e) {
      const log = globalThis.__slangfxGlslangLog ?? [];
      const detail = log.filter((l) => /error/i.test(l)).join('\n') || log.join('\n');
      throw new Error(`${path}: ${stage} compile failed: ${detail || e.message || e}`);
    }
  };
  const vertSpv = compileStage(pre.vertexGlsl, 'vertex');
  const fragSpv = compileStage(pre.fragmentGlsl, 'fragment');

  // Reflect the fragment stage (slang contract: both stages share the same
  // block declarations; fragment is reliably present). Merge in the vertex
  // reflection for blocks the fragment optimized away entirely.
  const fragRefl = reflectSpirv(fragSpv);
  const vertRefl = reflectSpirv(vertSpv);
  const ubo = fragRefl.ubo ?? vertRefl.ubo;
  const push = fragRefl.push ?? vertRefl.push;

  let vertWgsl, fragWgsl;
  try {
    vertWgsl = twgsl.convertSpirV2WGSL(vertSpv);
    fragWgsl = twgsl.convertSpirV2WGSL(fragSpv);
  } catch (e) {
    throw new Error(`${path}: SPIR-V → WGSL failed: ${e.message || e}`);
  }
  // twgsl's wrapper reuses one output slot across calls; a failed conversion
  // can return the previous shader's text. Verify each stage marker.
  if (!vertWgsl || !vertWgsl.includes('@vertex'))
    throw new Error(`${path}: SPIR-V → WGSL produced no vertex entry point`);
  if (!fragWgsl || !fragWgsl.includes('@fragment'))
    throw new Error(`${path}: SPIR-V → WGSL produced no fragment entry point`);

  // Dedupe #pragma parameter by name (first declaration wins, matching the
  // live tuner's discover_params).
  const seen = new Set();
  const params = pre.params.filter((p) => (seen.has(p.name) ? false : (seen.add(p.name), true)));

  // Keep only samplers that survived into the fragment/vertex WGSL (unused
  // declarations still occupy their binding slots in the shader interface,
  // and tint keeps them, so the full table is correct as-is).
  return {
    path,
    vertWgsl,
    fragWgsl,
    ubo,          // {size, members:[{name, offset, size}]} | null — set(0) binding(0)
    push,         // same shape | null — set(1) binding(0), the former push_constant block
    samplers: pre.samplers, // [{name, texSet, texBinding, smpBinding}]
    params,       // [{name, desc, default, min, max, step}]
    formatPragma: pre.formatPragma,
  };
}

/** Convenience: which WGSL declarations actually made it into each stage.
 * WebGPU bind group layouts must only include bindings the stage uses if
 * visibility is limited; we declare everything FRAGMENT|VERTEX and rely on
 * tint keeping declarations, so this is primarily for diagnostics. */
export function wgslDeclaredBindings(wgsl) {
  const out = [];
  const re = /@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<([^>]+)>)?\s+(\w+)\s*:\s*([^;]+);/g;
  let m;
  while ((m = re.exec(wgsl))) {
    out.push({
      group: parseInt(m[1], 10),
      binding: parseInt(m[2], 10),
      addressSpace: m[3] ?? null,
      name: m[4],
      type: m[5].trim(),
    });
  }
  return out;
}
