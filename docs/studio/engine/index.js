/*
 * slangfx-web — libretro slang shader chains on WebGPU.
 *
 * Quick start (browser):
 *
 *   import { SlangFx, loadToolchain } from 'slangfx-web';
 *
 *   const toolchain = await loadToolchain();
 *   const fx = await SlangFx.create({
 *     canvas,
 *     toolchain,
 *     readFile: async (p) => (await fetch('/' + p)).text(),
 *   });
 *   await fx.setSourceSize(video.videoWidth, video.videoHeight);
 *   await fx.addLayer('shaders/crt/soft-crt/soft-crt.slangp');
 *   fx.setParam(0, 'scan_strength', 0.3);
 *   const tick = () => { fx.render(video, video.currentTime); requestAnimationFrame(tick); };
 *   tick();
 */

export { SlangFx } from './engine.js';
export { loadToolchain } from './toolchain.js';
export { parsePreset, resolvePath, dirnameOf } from './slangp.js';
export { compileSlang, wgslDeclaredBindings } from './compiler.js';
export { flattenIncludes, preprocessSlang } from './preprocess.js';
export { reflectSpirv } from './spv-reflect.js';
export { Blitter } from './blit.js';
