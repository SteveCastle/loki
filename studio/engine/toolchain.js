/*
 * slangfx-web — browser loader for the wasm shader toolchain.
 *
 * glslang (GLSL 450 → SPIR-V) ships as an ES module that locates its .wasm
 * next to itself; twgsl (tint: SPIR-V → WGSL) ships as a classic UMD script
 * that registers `globalThis.twgsl`. Both are cached singletons.
 *
 * In Node (tests, offline tooling) load them directly instead:
 *   require('@webgpu/glslang/dist/node-devel/glslang.js')()
 *   require('web/vendor/twgsl.js')(pathToWasm)
 */

let cached = null;

export async function loadToolchain({
  glslangJsUrl = new URL('../vendor/glslang.js', import.meta.url).href,
  twgslJsUrl = new URL('../vendor/twgsl.js', import.meta.url).href,
  twgslWasmUrl = new URL('../vendor/twgsl.wasm', import.meta.url).href,
} = {}) {
  if (cached) return cached;
  cached = (async () => {
    const glslangFactory = (await import(/* webpackIgnore: true */ glslangJsUrl)).default;
    const glslang = await glslangFactory();

    if (!globalThis.twgsl) {
      await new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = twgslJsUrl;
        s.onload = resolve;
        s.onerror = () => reject(new Error(`failed to load ${twgslJsUrl}`));
        document.head.appendChild(s);
      });
    }
    const twgsl = await globalThis.twgsl(twgslWasmUrl);
    return { glslang, twgsl };
  })();
  return cached;
}
