/*
 * slangfx-web — WebGPU multi-pass engine.
 *
 * Browser/WebGPU port of src/slang_pipeline.c. One SlangFx instance owns the
 * device-side chain: an input texture, a stack of preset layers (each a full
 * libretro-slang multi-pass pipeline), and a present blit to an optional
 * canvas. Layers chain GPU-side — layer i+1's bind groups reference layer
 * i's final texture directly, no copies, one command submit per frame.
 *
 * Semantics ported from the native engine:
 *   - per-pass framebuffers sized by slangp scale rules (final pass forced
 *     to output dims, rgba8unorm)
 *   - Source / Original / OriginalHistoryN / Pass<n> / alias /
 *     PassFeedback<n> / <alias>Feedback / external-texture sampler binding
 *   - standard uniforms written at reflected offsets each frame:
 *     MVP, SourceSize, OriginalSize, OutputSize, FinalViewportSize,
 *     FrameCount, FrameDirection, Rotation, Time, and any `<TexName>Size`
 *   - #pragma parameters live-updatable per frame without a rebuild
 *   - feedback snapshot copies after each frame; first-frame clear to black
 *   - full mip chains (blit cascade) for `mipmap_input` consumers
 *
 * WebGPU deltas (see web/README.md): push constants become a UBO in bind
 * group 1, combined samplers are split (samplers live in bind group 2),
 * clamp_to_border falls back to clamp_to_edge, and unfilterable/unsupported
 * formats fall back to rgba16float.
 */

import { parsePreset, dirnameOf } from './slangp.js';
import { compileSlang } from './compiler.js';
import { renameReserved } from './preprocess.js';
import { Blitter, MaskBlender, MaskComposer } from './blit.js';

/* Quad matching the native vertex contract (location 0 = vec4 Position,
 * location 1 = vec2 TexCoord), with v oriented so that every pass keeps
 * texture row 0 = image top under WebGPU's y-up NDC. */
const QUAD_VERTS = new Float32Array([
  //  x,    y,   z,   w,   u,   v
  -1, -1, 0, 1, 0, 1,
   1, -1, 0, 1, 1, 1,
   1,  1, 0, 1, 1, 0,
  -1,  1, 0, 1, 0, 0,
]);
const QUAD_INDICES = new Uint16Array([0, 1, 2, 0, 2, 3]);

const WRAP_TO_GPU = {
  clamp_to_border: 'clamp-to-edge', // WebGPU has no border addressing
  clamp_to_edge: 'clamp-to-edge',
  repeat: 'repeat',
  mirrored_repeat: 'mirror-repeat',
};

/* slangp format → WebGPU format. Unsupported or unfilterable formats fall
 * back to rgba16float (renderable + filterable everywhere). */
const FORMAT_TO_GPU = {
  R8_UNORM: 'r8unorm',
  R8G8_UNORM: 'rg8unorm',
  R8G8B8A8_UNORM: 'rgba8unorm',
  R8G8B8A8_SRGB: 'rgba8unorm-srgb',
  R10G10B10A2_UNORM: 'rgb10a2unorm',
  R16_UNORM: 'rgba16float',
  R16_SFLOAT: 'r16float',
  R16G16B16A16_UNORM: 'rgba16float',
  R16G16B16A16_SFLOAT: 'rgba16float',
  R32_SFLOAT: 'rgba16float',
  R32G32B32A32_SFLOAT: 'rgba16float',
};

function resolvePassFormat(pass, formatPragma) {
  if (pass.fboFormat) return FORMAT_TO_GPU[pass.fboFormat] ?? 'rgba8unorm';
  if (formatPragma && FORMAT_TO_GPU[formatPragma]) {
    // #pragma format applies when the preset doesn't override.
    if (!pass.floatFramebuffer && !pass.srgbFramebuffer) return FORMAT_TO_GPU[formatPragma];
  }
  if (pass.floatFramebuffer) return 'rgba16float';
  if (pass.srgbFramebuffer) return 'rgba8unorm-srgb';
  return 'rgba8unorm';
}

function resolvePassDims(pass, prevW, prevH, finalW, finalH) {
  const axis = (type, scale, prev, final) => {
    switch (type) {
      case 'viewport': return final * scale;
      case 'absolute': return scale;
      default: return prev * scale;
    }
  };
  const w = Math.max(1, Math.round(axis(pass.scaleTypeX, pass.scaleX, prevW, finalW)));
  const h = Math.max(1, Math.round(axis(pass.scaleTypeY, pass.scaleY, prevH, finalH)));
  return [w, h];
}

function mipLevelsFor(w, h) {
  return 1 + Math.floor(Math.log2(Math.max(w, h)));
}

/* ---------------------------------------------------------------------- */

class PassState {
  constructor() {
    this.mod = null;          // compiled SlangModule
    this.slangp = null;       // slangp pass entry
    this.outW = 0;
    this.outH = 0;
    this.format = 'rgba8unorm';
    this.mipLevels = 1;
    this.outTex = null;
    this.outView = null;      // all mips (sampled)
    this.fboView = null;      // mip 0 (render target)
    this.sampler = null;
    this.isFeedbackProducer = false;
    this.feedbackTex = null;
    this.feedbackView = null;
    this.pipeline = null;
    this.bindGroups = [];     // [group0, group1, group2]
    this.uboBuf = null;
    this.pushBuf = null;
    this.uboData = null;      // Uint8Array staging
    this.pushData = null;
  }
}

/** One .slangp preset instantiated as a WebGPU pass chain. */
class PresetRuntime {
  /**
   * @param {SlangFx} fx
   * @param {object} preset parsed .slangp
   * @param {object[]} modules compiled modules, one per pass
   * @param {GPUTextureView} inputView chain input for this layer
   * @param {number} inputW @param {number} inputH
   */
  constructor(fx, preset, modules, inputView, inputW, inputH) {
    this.fx = fx;
    this.preset = preset;
    this.inputView = inputView;
    this.inputW = inputW;
    this.inputH = inputH;
    this.outputW = inputW;
    this.outputH = inputH;
    this.frameCount = 0;
    this.feedbackCleared = false;
    this.paramValues = new Map(); // name -> current value
    this.paramMeta = [];          // [{name, desc, default, min, max, step}]
    this.passes = modules.map(() => new PassState());
    this.extTextures = new Map(); // name -> {view, sampler}
    modules.forEach((m, i) => {
      this.passes[i].mod = m;
      this.passes[i].slangp = preset.passes[i] ?? null;
    });
    this.aliases = new Map();
    preset.passes.forEach((p, i) => { if (p.alias) this.aliases.set(p.alias, i); });
  }

  collectParams() {
    const seen = new Map();
    for (const ps of this.passes) {
      for (const p of ps.mod.params) if (!seen.has(p.name)) seen.set(p.name, { ...p });
    }
    for (const ov of this.preset.params) {
      const meta = seen.get(renameReserved(ov.name));
      if (meta) meta.default = ov.value;
    }
    this.paramMeta = [...seen.values()];
    for (const meta of this.paramMeta)
      if (!this.paramValues.has(meta.name)) this.paramValues.set(meta.name, meta.default);
  }

  setParam(name, value) {
    const meta = this.paramMeta.find((p) => p.name === name);
    if (!meta) return 0;
    let v = value;
    if (meta.max > meta.min) v = Math.min(meta.max, Math.max(meta.min, v));
    this.paramValues.set(name, v);
    return 1;
  }

  /* Map a sampler name to {view, sampler}. Port of resolve_sampler_name,
   * plus `Mask`: the layer's painted mask (white when none is set), so
   * custom shaders can read the mask directly. */
  resolveSampler(name, passIdx, prevView, prevSampler) {
    const { fx } = this;
    if (name === 'Source') return { view: prevView, sampler: prevSampler };
    if (name === 'Mask') return { view: this.maskView ?? fx.whiteView, sampler: fx.inputSampler };
    if (name === 'Original' || name.startsWith('OriginalHistory'))
      return { view: this.inputView, sampler: fx.inputSampler };
    if (name.startsWith('PassFeedback')) {
      const n = parseInt(name.slice('PassFeedback'.length), 10);
      if (!Number.isNaN(n) && n >= 0 && n < this.passes.length)
        return { feedbackOf: n, sampler: this.passes[n].sampler };
      return null;
    }
    if (name.endsWith('Feedback') && name.length > 8) {
      const alias = name.slice(0, -8);
      if (this.aliases.has(alias)) {
        const n = this.aliases.get(alias);
        return { feedbackOf: n, sampler: this.passes[n].sampler };
      }
      return null;
    }
    if (name.startsWith('Pass')) {
      const n = parseInt(name.slice(4), 10);
      if (!Number.isNaN(n) && String(n) === name.slice(4) && n >= 0 && n < passIdx)
        return { view: this.passes[n].outView, sampler: this.passes[n].sampler };
    }
    if (this.aliases.has(name)) {
      const n = this.aliases.get(name);
      if (n < passIdx) return { view: this.passes[n].outView, sampler: this.passes[n].sampler };
    }
    if (this.extTextures.has(name)) return this.extTextures.get(name);
    return null;
  }

  /* `<TexName>Size` fields not covered by the standard writes. */
  lookupSizeField(fieldName) {
    if (!fieldName || !fieldName.endsWith('Size')) return null;
    const base = fieldName.slice(0, -4);
    if (['Source', 'Original', 'Output', 'FinalViewport'].includes(base)) return null;
    if (base.startsWith('OriginalHistory')) return [this.inputW, this.inputH];
    if (base.startsWith('PassFeedback')) {
      const n = parseInt(base.slice('PassFeedback'.length), 10);
      if (!Number.isNaN(n) && n >= 0 && n < this.passes.length)
        return [this.passes[n].outW, this.passes[n].outH];
      return null;
    }
    if (base.startsWith('Pass')) {
      const digits = base.slice(4);
      const n = parseInt(digits, 10);
      if (!Number.isNaN(n) && String(n) === digits && n >= 0 && n < this.passes.length)
        return [this.passes[n].outW, this.passes[n].outH];
    }
    if (base.endsWith('Feedback')) {
      const alias = base.slice(0, -8);
      if (this.aliases.has(alias)) {
        const n = this.aliases.get(alias);
        return [this.passes[n].outW, this.passes[n].outH];
      }
      return null;
    }
    if (this.aliases.has(base)) {
      const n = this.aliases.get(base);
      return [this.passes[n].outW, this.passes[n].outH];
    }
    if (this.extTextures.has(base)) {
      const t = this.extTextures.get(base);
      return [t.width ?? this.inputW, t.height ?? this.inputH];
    }
    return null;
  }

  async build() {
    const { device } = this.fx;
    const nPasses = this.passes.length;
    this.collectParams();

    // External textures (Phase 8 equivalent). A host-supplied override
    // (image / canvas / bitmap) replaces the preset's file per layer.
    for (const tex of this.preset.textures) {
      try {
        const bitmap = this.textureOverrides?.[tex.name] ?? await this.fx.readImage(tex.path);
        const gpuTex = device.createTexture({
          label: `slangfx ext ${tex.name}`,
          size: [bitmap.width, bitmap.height],
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: gpuTex }, [bitmap.width, bitmap.height]);
        this.extTextures.set(tex.name, {
          view: gpuTex.createView(),
          sampler: device.createSampler({
            magFilter: tex.filterLinear ? 'linear' : 'nearest',
            minFilter: tex.filterLinear ? 'linear' : 'nearest',
            addressModeU: WRAP_TO_GPU[tex.wrapMode],
            addressModeV: WRAP_TO_GPU[tex.wrapMode],
          }),
          width: bitmap.width,
          height: bitmap.height,
        });
      } catch (e) {
        console.warn(`slangfx: external texture '${tex.name}' failed to load:`, e);
      }
    }

    // Phase A: dims, formats, output textures, samplers.
    let prevW = this.inputW;
    let prevH = this.inputH;
    for (let i = 0; i < nPasses; i++) {
      const ps = this.passes[i];
      const sp = ps.slangp;
      let [w, h] = sp ? resolvePassDims(sp, prevW, prevH, this.outputW, this.outputH) : [this.outputW, this.outputH];
      if (i === nPasses - 1) { w = this.outputW; h = this.outputH; }
      ps.outW = w;
      ps.outH = h;
      ps.format = i === nPasses - 1 ? 'rgba8unorm' : resolvePassFormat(sp ?? {}, ps.mod.formatPragma);
      ps.mipLevels = 1;
      const next = this.preset.passes[i + 1];
      if (next && next.mipmapInput) ps.mipLevels = mipLevelsFor(w, h);

      ps.outTex = device.createTexture({
        label: `slangfx pass${i} out`,
        size: [w, h],
        format: ps.format,
        mipLevelCount: ps.mipLevels,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
      ps.outView = ps.outTex.createView();
      ps.fboView = ps.mipLevels > 1 ? ps.outTex.createView({ baseMipLevel: 0, mipLevelCount: 1 }) : ps.outView;
      ps.sampler = device.createSampler({
        magFilter: sp?.filterLinear ? 'linear' : 'nearest',
        minFilter: sp?.filterLinear ? 'linear' : 'nearest',
        mipmapFilter: 'linear',
        addressModeU: WRAP_TO_GPU[sp?.wrapMode ?? 'clamp_to_edge'],
        addressModeV: WRAP_TO_GPU[sp?.wrapMode ?? 'clamp_to_edge'],
        lodMaxClamp: ps.mipLevels,
      });
      prevW = w;
      prevH = h;
    }

    // Phase C: resolve sampler bindings, marking feedback producers.
    const resolved = []; // per pass: Map(name -> {view?, feedbackOf?, sampler})
    {
      let prevView = this.inputView;
      let prevSampler = this.fx.inputSampler;
      for (let i = 0; i < nPasses; i++) {
        const map = new Map();
        for (const s of this.passes[i].mod.samplers) {
          const r = this.resolveSampler(s.name, i, prevView, prevSampler);
          if (r && r.feedbackOf != null) this.passes[r.feedbackOf].isFeedbackProducer = true;
          map.set(s.name, r);
        }
        resolved.push(map);
        prevView = this.passes[i].outView;
        prevSampler = this.passes[i].sampler;
      }
    }

    // Phase D: feedback textures.
    for (const ps of this.passes) {
      if (!ps.isFeedbackProducer) continue;
      ps.feedbackTex = device.createTexture({
        label: 'slangfx feedback',
        size: [ps.outW, ps.outH],
        format: ps.format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      ps.feedbackView = ps.feedbackTex.createView();
    }

    // Phase F: pipelines, uniform buffers, bind groups.
    for (let i = 0; i < nPasses; i++) {
      const ps = this.passes[i];
      const mod = ps.mod;

      const vertModule = device.createShaderModule({ label: `${mod.path} vs`, code: mod.vertWgsl });
      const fragModule = device.createShaderModule({ label: `${mod.path} fs`, code: mod.fragWgsl });
      for (const m of [vertModule, fragModule]) {
        if (m.getCompilationInfo) {
          const info = await m.getCompilationInfo();
          const errs = info.messages.filter((x) => x.type === 'error');
          if (errs.length)
            throw new Error(`${mod.path}: WGSL rejected: ${errs.map((e) => e.message).join('; ')}`);
        }
      }

      const ALL = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
      const g0Entries = [];
      if (mod.ubo) g0Entries.push({ binding: 0, visibility: ALL, buffer: { type: 'uniform' } });
      for (const s of mod.samplers)
        g0Entries.push({ binding: s.texBinding, visibility: ALL, texture: { sampleType: 'float' } });
      const g1Entries = mod.push ? [{ binding: 0, visibility: ALL, buffer: { type: 'uniform' } }] : [];
      const g2Entries = mod.samplers.map((s) => ({ binding: s.smpBinding, visibility: ALL, sampler: {} }));

      const layouts = [
        device.createBindGroupLayout({ entries: g0Entries }),
        device.createBindGroupLayout({ entries: g1Entries }),
        device.createBindGroupLayout({ entries: g2Entries }),
      ];

      ps.pipeline = await device.createRenderPipelineAsync({
        label: `slangfx ${mod.path} pass${i}`,
        layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
        vertex: {
          module: vertModule,
          entryPoint: 'main',
          buffers: [{
            arrayStride: 24,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x4' },
              { shaderLocation: 1, offset: 16, format: 'float32x2' },
            ],
          }],
        },
        fragment: { module: fragModule, entryPoint: 'main', targets: [{ format: ps.format }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
      });

      if (mod.ubo) {
        ps.uboBuf = device.createBuffer({ size: mod.ubo.size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        ps.uboData = new Uint8Array(mod.ubo.size);
      }
      if (mod.push) {
        ps.pushBuf = device.createBuffer({ size: mod.push.size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        ps.pushData = new Uint8Array(mod.push.size);
      }

      const g0 = [];
      if (mod.ubo) g0.push({ binding: 0, resource: { buffer: ps.uboBuf } });
      for (const s of mod.samplers) {
        const r = resolved[i].get(s.name);
        let view = r?.view ?? null;
        if (r && r.feedbackOf != null) view = this.passes[r.feedbackOf].feedbackView;
        if (!view) {
          console.warn(`slangfx: pass ${i} sampler '${s.name}' unresolved; binding black`);
          view = this.fx.dummyView;
        }
        g0.push({ binding: s.texBinding, resource: view });
      }
      const g1 = mod.push ? [{ binding: 0, resource: { buffer: ps.pushBuf } }] : [];
      const g2 = mod.samplers.map((s) => {
        const r = resolved[i].get(s.name);
        return { binding: s.smpBinding, resource: r?.sampler ?? this.fx.inputSampler };
      });

      ps.bindGroups = [
        device.createBindGroup({ layout: layouts[0], entries: g0 }),
        device.createBindGroup({ layout: layouts[1], entries: g1 }),
        device.createBindGroup({ layout: layouts[2], entries: g2 }),
      ];
    }
  }

  get finalPass() { return this.passes[this.passes.length - 1]; }

  writeUniforms(timeSec) {
    const { queue } = this.fx.device;
    const frameCount = this.frameCount;
    let prevW = this.inputW;
    let prevH = this.inputH;

    for (const ps of this.passes) {
      const writeBlock = (data, block) => {
        if (!data || !block) return;
        data.fill(0);
        const dv = new DataView(data.buffer);
        const writeVec4 = (offset, x, y, z, w) => {
          dv.setFloat32(offset, x, true);
          dv.setFloat32(offset + 4, y, true);
          dv.setFloat32(offset + 8, z, true);
          dv.setFloat32(offset + 12, w, true);
        };
        for (const m of block.members) {
          if (!m.name) continue;
          switch (m.name) {
            case 'MVP':
              // identity mat4
              for (let c = 0; c < 4; c++) dv.setFloat32(m.offset + c * 20, 1, true);
              break;
            case 'SourceSize': writeVec4(m.offset, prevW, prevH, 1 / prevW, 1 / prevH); break;
            case 'OriginalSize': writeVec4(m.offset, this.inputW, this.inputH, 1 / this.inputW, 1 / this.inputH); break;
            case 'OutputSize':
            case 'FinalViewportSize': writeVec4(m.offset, ps.outW, ps.outH, 1 / ps.outW, 1 / ps.outH); break;
            case 'FrameCount': {
              let fc = frameCount;
              const mod = ps.slangp?.frameCountMod ?? 0;
              if (mod > 0) fc %= mod;
              dv.setUint32(m.offset, fc >>> 0, true);
              break;
            }
            case 'FrameDirection': dv.setInt32(m.offset, 1, true); break;
            case 'Rotation': dv.setUint32(m.offset, 0, true); break;
            case 'Time': dv.setFloat32(m.offset, timeSec, true); break;
            default: {
              const sz = this.lookupSizeField(m.name);
              if (sz) { writeVec4(m.offset, sz[0], sz[1], 1 / sz[0], 1 / sz[1]); break; }
              if (this.paramValues.has(m.name)) dv.setFloat32(m.offset, this.paramValues.get(m.name), true);
              break;
            }
          }
        }
      };

      writeBlock(ps.pushData, ps.mod.push);
      writeBlock(ps.uboData, ps.mod.ubo);
      if (ps.pushBuf) queue.writeBuffer(ps.pushBuf, 0, ps.pushData);
      if (ps.uboBuf) queue.writeBuffer(ps.uboBuf, 0, ps.uboData);

      prevW = ps.outW;
      prevH = ps.outH;
    }
  }

  encode(encoder) {
    const { blitter, quad } = this.fx;

    if (!this.feedbackCleared) {
      for (const ps of this.passes) {
        if (!ps.isFeedbackProducer) continue;
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: ps.feedbackView,
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          }],
        });
        pass.end();
      }
      this.feedbackCleared = true;
    }

    for (const ps of this.passes) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: ps.fboView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(ps.pipeline);
      pass.setBindGroup(0, ps.bindGroups[0]);
      pass.setBindGroup(1, ps.bindGroups[1]);
      pass.setBindGroup(2, ps.bindGroups[2]);
      pass.setVertexBuffer(0, quad.vbuf);
      pass.setIndexBuffer(quad.ibuf, 'uint16');
      pass.drawIndexed(6);
      pass.end();

      if (ps.mipLevels > 1) blitter.generateMips(encoder, ps.outTex, ps.format, ps.mipLevels);
    }

    for (const ps of this.passes) {
      if (!ps.isFeedbackProducer) continue;
      encoder.copyTextureToTexture(
        { texture: ps.outTex, mipLevel: 0 },
        { texture: ps.feedbackTex },
        [ps.outW, ps.outH]
      );
    }

    this.frameCount++;
  }

  destroy() {
    for (const ps of this.passes) {
      ps.outTex?.destroy();
      ps.feedbackTex?.destroy();
      ps.uboBuf?.destroy();
      ps.pushBuf?.destroy();
    }
  }
}

/* ---------------------------------------------------------------------- */

/**
 * Public engine. See web/README.md for the embedding guide.
 */
export class SlangFx {
  /**
   * @param {object} opts
   * @param {{glslang, twgsl}} opts.toolchain from loadToolchain()
   * @param {(path: string) => Promise<string>} opts.readFile shader/preset loader
   * @param {(path: string) => Promise<ImageBitmap>} [opts.readImage]
   * @param {HTMLCanvasElement} [opts.canvas] present target (optional — headless works)
   * @param {GPUDevice} [opts.device] reuse an existing device
   */
  static async create(opts) {
    const fx = new SlangFx();
    fx.toolchain = opts.toolchain;
    fx.readFile = opts.readFile;
    fx.readImage = opts.readImage ?? (async (path) => {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
      return createImageBitmap(await res.blob());
    });

    if (opts.device) {
      fx.device = opts.device;
    } else {
      if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('No WebGPU adapter found');
      fx.device = await adapter.requestDevice();
    }

    fx.blitter = new Blitter(fx.device);
    fx.quad = {
      vbuf: fx.device.createBuffer({ size: QUAD_VERTS.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST }),
      ibuf: fx.device.createBuffer({ size: QUAD_INDICES.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST }),
    };
    fx.device.queue.writeBuffer(fx.quad.vbuf, 0, QUAD_VERTS);
    fx.device.queue.writeBuffer(fx.quad.ibuf, 0, QUAD_INDICES);

    fx.inputSampler = fx.device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    });
    const dummy = fx.device.createTexture({
      size: [1, 1], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    fx.device.queue.writeTexture({ texture: dummy }, new Uint8Array([0, 0, 0, 255]), {}, [1, 1]);
    fx.dummyView = dummy.createView();
    const white = fx.device.createTexture({
      size: [1, 1], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    fx.device.queue.writeTexture({ texture: white }, new Uint8Array([255, 255, 255, 255]), {}, [1, 1]);
    fx.whiteView = white.createView();
    fx.maskBlender = new MaskBlender(fx.device);
    fx.maskComposer = new MaskComposer(fx.device);

    if (opts.canvas) {
      fx.canvas = opts.canvas;
      fx.ctx = opts.canvas.getContext('webgpu');
      fx.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
      fx.ctx.configure({ device: fx.device, format: fx.canvasFormat, alphaMode: 'opaque' });
    }

    fx.layers = [];          // [{path, enabled, runtime|null, error|null}]
    fx.moduleCache = new Map();
    fx.inputW = 0;
    fx.inputH = 0;
    fx.startTime = null;
    return fx;
  }

  /** Size (or resize) the chain input. Rebuilds all layers. */
  async setSourceSize(w, h) {
    if (w === this.inputW && h === this.inputH && this.inputTexture) return;
    this.inputW = w;
    this.inputH = h;
    this.inputTexture?.destroy();
    this.inputTexture = this.device.createTexture({
      label: 'slangfx input',
      size: [w, h],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST |
             GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.inputView = this.inputTexture.createView();
    await this.rebuild();
  }

  async compileModule(path, { textureLodWorkaround = false } = {}) {
    const key = `${path}|${textureLodWorkaround ? 'lod' : 'std'}`;
    if (!this.moduleCache.has(key)) {
      this.moduleCache.set(key, (async () => {
        const src = await this.readFile(path);
        return compileSlang(src, {
          path,
          readFile: this.readFile,
          glslang: this.toolchain.glslang,
          twgsl: this.toolchain.twgsl,
          textureLodWorkaround,
        });
      })());
    }
    return this.moduleCache.get(key);
  }

  /** Append a layer from a .slangp path. Returns its index.
   * `label` overrides the display name in getLayerInfo (useful for layers
   * backed by virtual/in-memory files, e.g. a live shader editor). */
  async addLayer(presetPath, { label = null } = {}) {
    this.layers.push({ path: presetPath, label, enabled: true, runtime: null, error: null, savedParams: null });
    await this.rebuild();
    return this.layers.length - 1;
  }

  /** Drop cached compiled modules whose source path starts with `prefix`.
   * Call before rebuild() when the backing source text has changed (live
   * shader editing) — the cache is keyed by path and would stay stale. */
  invalidateModules(prefix) {
    for (const key of [...this.moduleCache.keys()])
      if (key.startsWith(prefix)) this.moduleCache.delete(key);
  }

  async removeLayer(i) { this.layers.splice(i, 1); await this.rebuild(); }

  async moveLayer(i, dir) {
    const j = i + dir;
    if (j < 0 || j >= this.layers.length) return;
    [this.layers[i], this.layers[j]] = [this.layers[j], this.layers[i]];
    await this.rebuild();
  }

  async toggleLayer(i, enabled) { this.layers[i].enabled = enabled; await this.rebuild(); }

  async clearLayers() { this.layers = []; await this.rebuild(); }

  /* Create (or recreate) a layer's mask texture at input dims plus the GPU
   * state for every mask node in the stack. The stack is re-composited into
   * maskTex each frame by MaskComposer (see render()), so node sources are
   * free to change between frames — painted canvases re-upload, matte /
   * roto sources swap views.
   *
   * Node fields the engine consumes:
   *   kind        'tex' semantics via source/view + channel, or 'key'
   *   source      canvas/image backing (engine owns a texture for it)
   *   view        externally-owned GPUTextureView (matte scratch, roto frame)
   *   useInput    sample this layer's input (chroma key on own content)
   *   channel     'r' | 'luma' | 'alpha'   blend  'add'|'subtract'|'multiply'|'max'|'min'
   *   keyRGB, similarity, smoothness, strength, invert, enabled, active */
  _buildLayerMask(layer, layerInput) {
    layer.maskTex?.destroy();
    layer.maskTex = this.device.createTexture({
      label: 'slangfx layer mask',
      size: [this.inputW, this.inputH],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    layer.maskView = layer.maskTex.createView();
    for (const node of layer.maskState.nodes) {
      node._optsBuf = this.device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      node._tex = null;
      if (node.source) {
        node._tex = this.device.createTexture({
          label: 'slangfx mask node source',
          size: [this.inputW, this.inputH],
          format: 'rgba8unorm',
          // RENDER_ATTACHMENT: copyExternalImageToTexture's GPU-canvas blit
          // path requires it — without it uploads fail silently.
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
      }
      const view = node._tex?.createView()
        ?? node.view
        ?? (node.useInput ? layerInput : this.whiteView);
      node._bindGroup = this.maskComposer.bindGroup(view, this.inputSampler, node._optsBuf);
    }
    this.updateLayerMask(this.layers.indexOf(layer));
  }

  _destroyLayerMaskGpu(layer) {
    layer.maskTex?.destroy();
    layer.blendTex?.destroy();
    layer.maskOptsBuf?.destroy();
    layer.maskTex = layer.maskView = layer.blendTex = layer.blendView = null;
    layer.blendBindGroup = layer.maskOptsBuf = null;
    this.maskComposer.destroyPost(layer.maskState);
    for (const node of layer.maskState?.nodes ?? []) {
      node._optsBuf?.destroy();
      node._tex?.destroy();
      node._optsBuf = node._tex = node._bindGroup = null;
    }
  }

  /** Rebuild every layer's GPU state (structural changes + source resize). */
  async rebuild() {
    if (!this.inputTexture) return;
    for (const layer of this.layers) {
      if (layer.runtime) layer.savedParams = new Map(layer.runtime.paramValues);
      layer.runtime?.destroy();
      layer.runtime = null;
      layer.error = null;
      layer.effectiveView = null;
      this._destroyLayerMaskGpu(layer);
    }
    let inputView = this.inputView;
    for (const layer of this.layers) {
      if (!layer.enabled) continue;
      const layerInput = inputView;
      // Decide once: maskState can be assigned asynchronously (project
      // restore) while this rebuild awaits a compile — the blend section
      // below must not see a mask the top of the loop never built.
      const hasMask = !!layer.maskState?.nodes?.length;
      if (hasMask) this._buildLayerMask(layer, layerInput);
      const buildOnce = async (compileOpts) => {
        const presetText = await this.readFile(layer.path);
        const preset = parsePreset(presetText, dirnameOf(layer.path));
        const modules = [];
        for (const pass of preset.passes) modules.push(await this.compileModule(pass.path, compileOpts));
        const rt = new PresetRuntime(this, preset, modules, layerInput, this.inputW, this.inputH);
        rt.maskView = layer.maskView ?? null;   // `Mask` sampler for custom shaders
        rt.textureOverrides = layer.textureOverrides ?? null;
        if (layer.savedParams)
          for (const [k, v] of layer.savedParams) rt.paramValues.set(k, v);
        await rt.build();
        return rt;
      };
      try {
        let rt;
        try {
          rt = await buildOnce({});
        } catch (e) {
          if (!/uniform control flow/.test(String(e.message ?? e))) throw e;
          // Retry with the textureLod uniformity workaround.
          rt = await buildOnce({ textureLodWorkaround: true });
        }
        layer.runtime = rt;
        if (hasMask) {
          layer.blendTex = this.device.createTexture({
            label: 'slangfx masked out',
            size: [this.inputW, this.inputH],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
          });
          layer.blendView = layer.blendTex.createView();
          layer.maskOptsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this._writeMaskOpts(layer);
          layer.blendBindGroup = this.maskBlender.bindGroup(
            layerInput, rt.finalPass.outView, layer.maskView, this.inputSampler, layer.maskOptsBuf);
          layer.effectiveView = layer.blendView;
        } else {
          layer.effectiveView = rt.finalPass.outView;
        }
        inputView = layer.effectiveView;
      } catch (e) {
        layer.error = String(e.message ?? e);
        console.error(`slangfx: layer '${layer.path}' failed:`, e);
      }
    }
  }

  _writeMaskOpts(layer) {
    if (!layer.maskOptsBuf || !layer.maskState) return;
    this.device.queue.writeBuffer(layer.maskOptsBuf, 0,
      new Float32Array([layer.maskState.opacity ?? 1, layer.maskState.invert ? 1 : 0, 0, 0]));
  }

  /** Attach (or replace) a layer's mask stack: { opacity, invert, nodes }.
   * See _buildLayerMask for the node contract. */
  async setLayerMask(i, maskState) {
    const layer = this.layers[i];
    if (!layer) return;
    layer.maskState = maskState;
    await this.rebuild();
  }

  /** Remove a layer's mask entirely. */
  async clearLayerMask(i) {
    const layer = this.layers[i];
    if (!layer?.maskState) return;
    layer.maskState = null;
    await this.rebuild();
  }

  /** Re-upload canvas-backed mask node pixels (cheap — call during brush
   * strokes; no rebuild). The stack recomposites automatically next frame. */
  updateLayerMask(i) {
    const layer = this.layers[i];
    if (!layer?.maskTex) return;
    for (const node of layer.maskState?.nodes ?? []) {
      const src = node.source;
      if (!node._tex || !src) continue;
      if ((src.width ?? 0) !== this.inputW || (src.height ?? 0) !== this.inputH) continue;
      this.device.queue.copyExternalImageToTexture({ source: src }, { texture: node._tex }, [this.inputW, this.inputH]);
    }
  }

  /** Replace one of a layer's external textures (preset `textures = ...`)
   * with any copyExternalImageToTexture source — image, canvas, bitmap.
   * Used for user-supplied stamps / rendered titles. */
  async setLayerTexture(i, name, source) {
    const layer = this.layers[i];
    if (!layer) return;
    (layer.textureOverrides ??= {})[name] = source;
    await this.rebuild();
  }

  /** Update mask opacity / inversion without a rebuild. */
  setLayerMaskOptions(i, { opacity, invert } = {}) {
    const layer = this.layers[i];
    if (!layer?.maskState) return;
    if (opacity != null) layer.maskState.opacity = opacity;
    if (invert != null) layer.maskState.invert = invert;
    this._writeMaskOpts(layer);
  }

  /** Per-layer info for UIs: params with metadata + current values. */
  getLayerInfo() {
    return this.layers.map((layer, i) => ({
      index: i,
      path: layer.path,
      name: layer.label ?? layer.path.split('/').pop(),
      enabled: layer.enabled,
      error: layer.error,
      mask: layer.maskState
        ? { opacity: layer.maskState.opacity ?? 1, invert: !!layer.maskState.invert }
        : null,
      textures: layer.runtime ? layer.runtime.preset.textures.map((t) => t.name) : [],
      params: layer.runtime
        ? layer.runtime.paramMeta.map((m) => ({ ...m, value: layer.runtime.paramValues.get(m.name) }))
        : [],
    }));
  }

  setParam(layerIdx, name, value) {
    return this.layers[layerIdx]?.runtime?.setParam(name, value) ?? 0;
  }

  resetParams(layerIdx) {
    const rt = this.layers[layerIdx]?.runtime;
    if (!rt) return;
    for (const m of rt.paramMeta) rt.paramValues.set(m.name, m.default);
  }

  get activeLayers() {
    return this.layers.filter((l) => l.enabled && l.runtime);
  }

  get finalView() {
    const ls = this.activeLayers;
    return ls.length ? ls[ls.length - 1].effectiveView : this.inputView;
  }

  get finalTexture() {
    const ls = this.activeLayers;
    if (!ls.length) return this.inputTexture;
    const last = ls[ls.length - 1];
    return last.blendTex ?? last.runtime.finalPass.outTex;
  }

  /**
   * Render one frame.
   * @param {CanvasImageSource|VideoFrame|null} source new frame to upload
   *        (null = reuse the current input contents)
   * @param {number} [timeSec] shader Time; defaults to a wall clock
   */
  render(source = null, timeSec = undefined) {
    if (!this.inputTexture) return;
    if (timeSec == null) {
      if (this.startTime == null) this.startTime = performance.now();
      timeSec = (performance.now() - this.startTime) / 1000;
    }
    if (source) {
      try {
        this.device.queue.copyExternalImageToTexture(
          { source },
          { texture: this.inputTexture },
          [this.inputW, this.inputH]
        );
      } catch (e) {
        // e.g. video not ready yet
      }
    }

    const encoder = this.device.createCommandEncoder();
    for (const layer of this.activeLayers) {
      // Recomposite the mask node stack first — the layer's passes and the
      // blend below sample maskTex this same submission.
      if (layer.maskState?.nodes?.length && layer.maskView)
        this.maskComposer.encode(encoder, layer);
      layer.runtime.writeUniforms(timeSec);
      layer.runtime.encode(encoder);
      if (layer.blendBindGroup)
        this.maskBlender.encode(encoder, layer.blendBindGroup, layer.blendView);
      // Host hook: lets an embedder inject extra passes into a layer's
      // effective output before the next layer samples it (e.g. compositing
      // media that sits above this effect in a track stack).
      this.onAfterLayer?.(encoder, layer);
    }

    if (this.ctx) {
      const canvasView = this.ctx.getCurrentTexture().createView();
      const cw = this.canvas.width;
      const ch = this.canvas.height;
      const scale = Math.min(cw / this.inputW, ch / this.inputH);
      const vw = Math.max(1, Math.floor(this.inputW * scale));
      const vh = Math.max(1, Math.floor(this.inputH * scale));
      this.blitter.blit(encoder, this.finalView, canvasView, this.canvasFormat, {
        viewport: { x: Math.floor((cw - vw) / 2), y: Math.floor((ch - vh) / 2), w: vw, h: vh },
      });
    }

    this.device.queue.submit([encoder.finish()]);
  }

  /** Read back a frame as RGBA8 pixels — the processed output by default,
   * or any COPY_SRC-capable texture of input dims (e.g. inputTexture, used
   * by the mask eyedropper to sample pre-effect colors). */
  async readPixels(texture = null) {
    const tex = texture ?? this.finalTexture;
    const w = this.inputW;
    const h = this.inputH;
    const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
    const buf = this.device.createBuffer({
      size: bytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, [w, h]);
    this.device.queue.submit([encoder.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(buf.getMappedRange());
    const pixels = new Uint8ClampedArray(w * h * 4);
    for (let row = 0; row < h; row++)
      pixels.set(mapped.subarray(row * bytesPerRow, row * bytesPerRow + w * 4), row * w * 4);
    buf.unmap();
    buf.destroy();
    return { pixels, width: w, height: h };
  }

  /** Export the current processed frame as a PNG blob. */
  async exportPNG() {
    const { pixels, width, height } = await this.readPixels();
    const canvas = new OffscreenCanvas(width, height);
    const ctx2d = canvas.getContext('2d');
    ctx2d.putImageData(new ImageData(pixels, width, height), 0, 0);
    return canvas.convertToBlob({ type: 'image/png' });
  }

  destroy() {
    for (const layer of this.layers) layer.runtime?.destroy();
    this.inputTexture?.destroy();
  }
}
