/*
 * Lowkey Studio — per-layer motion blur.
 *
 * A camera's shutter stays open for part of each frame, so anything moving
 * smears along its path. A layer with motion blur on is drawn `samples`
 * times at instants spread across that open interval (centred on the frame
 * time, `shutter` degrees of a 360° frame), each draw carrying its own
 * transform — and for a splat layer its own camera — at that instant. The
 * draws are averaged in premultiplied float, then resolved to straight
 * alpha: a comp-sized picture the compositor places like any processed
 * layer (still wearing the layer's opacity, blend mode and mask, which are
 * applied once, after the blur).
 *
 * Only motion that comes from the timeline is blurred: keyframes and
 * drivers on the transform, or on a splat's camera. A video's own content
 * motion is already in its frames.
 */

const ACCUM_WGSL = /* wgsl */ `
struct Xform {
  sizes : vec4<f32>,   // comp W, comp H, media w, media h
  place : vec4<f32>,   // centre x, centre y (comp px, y-down), scale x, scale y
  misc  : vec4<f32>,   // weight, rotation (rad), -, -
};
@group(0) @binding(0) var<uniform> xf : Xform;
@group(0) @binding(1) var tex : texture_2d<f32>;
@group(0) @binding(2) var smp : sampler;

struct VSOut { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i : u32) -> VSOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
  let c = corners[i];
  var p = (c - vec2<f32>(0.5, 0.5)) * xf.sizes.zw * xf.place.zw;
  let r = xf.misc.y;
  p = mat2x2<f32>(cos(r), sin(r), -sin(r), cos(r)) * p + xf.place.xy;
  var o : VSOut;
  o.pos = vec4<f32>(p.x / xf.sizes.x * 2.0 - 1.0, 1.0 - p.y / xf.sizes.y * 2.0, 0.0, 1.0);
  o.uv = c;
  return o;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let c = textureSample(tex, smp, in.uv);
  return vec4<f32>(c.rgb * c.a, c.a) * xf.misc.x;
}
`;

const RESOLVE_WGSL = /* wgsl */ `
@group(0) @binding(0) var accum : texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) fc : vec4<f32>) -> @location(0) vec4<f32> {
  let c = textureLoad(accum, vec2<i32>(fc.xy), 0);
  if (c.a <= 1e-4) { return vec4<f32>(0.0); }
  return vec4<f32>(clamp(c.rgb / c.a, vec3<f32>(0.0), vec3<f32>(1.0)), min(c.a, 1.0));
}
`;

const UBO_SLOT = 256;        // dynamic-offset alignment
export const MAX_SAMPLES = 32;

/** Sub-frame instants for a frame at comp time t: `samples` points spread
 * evenly over the open shutter, centred on t. */
export function shutterTimes(t, fps, shutterDeg, samples) {
  const n = Math.max(2, Math.min(MAX_SAMPLES, Math.round(samples)));
  const open = (Math.max(1, Math.min(720, shutterDeg)) / 360) / fps;
  const out = [];
  for (let i = 0; i < n; i++) out.push(t + (i / (n - 1) - 0.5) * open);
  return out;
}

export class MotionBlur {
  constructor(device) {
    this.device = device;
    const am = device.createShaderModule({ label: 'motion blur accum', code: ACCUM_WGSL });
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    this.accumPipe = device.createRenderPipeline({
      label: 'motion blur accum',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module: am, entryPoint: 'vs' },
      fragment: {
        module: am, entryPoint: 'fs',
        targets: [{
          format: 'rgba16float',
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });
    const rm = device.createShaderModule({ label: 'motion blur resolve', code: RESOLVE_WGSL });
    this.resolvePipe = device.createRenderPipeline({
      label: 'motion blur resolve', layout: 'auto',
      vertex: { module: rm, entryPoint: 'vs' },
      fragment: { module: rm, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.ubo = device.createBuffer({
      label: 'motion blur xforms', size: UBO_SLOT * MAX_SAMPLES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.targets = new Map();   // clipId -> {w, h, accum, out, view, resolveBG}
    this.bindGroups = new WeakMap();   // texture view -> bind group
  }

  _target(clipId, w, h) {
    let t = this.targets.get(clipId);
    if (t && t.w === w && t.h === h) return t;
    if (t) { t.accum.destroy(); t.out.destroy(); }
    const accum = this.device.createTexture({
      label: 'motion blur accum', size: [w, h], format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const out = this.device.createTexture({
      label: 'motion blur out', size: [w, h], format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    t = {
      w, h, accum, out, view: out.createView(), accumView: accum.createView(),
    };
    t.resolveBG = this.device.createBindGroup({
      layout: this.resolvePipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: t.accumView }],
    });
    this.targets.set(clipId, t);
    return t;
  }

  _bindGroup(view) {
    let bg = this.bindGroups.get(view);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.ubo, size: 48 } },
          { binding: 1, resource: view },
          { binding: 2, resource: this.sampler },
        ],
      });
      this.bindGroups.set(view, bg);
    }
    return bg;
  }

  /**
   * Blur one layer. `sample(i)` prepares sub-frame i and returns its raw
   * draw ({view, w, h, x, y, scaleX, scaleY, rot}) — and may itself encode
   * GPU work into the encoder it is handed (a splat re-rendering its
   * camera). Every sample is submitted on its own so per-sample uniform
   * writes (ours and the splat's) land in order. Returns the result view.
   */
  render(clipId, compW, compH, n, sample) {
    const dev = this.device;
    const tgt = this._target(clipId, compW, compH);
    const w = 1 / n;
    for (let i = 0; i < n; i++) {
      const enc = dev.createCommandEncoder({ label: 'motion blur sample' });
      const d = sample(i, enc);
      if (d) {
        dev.queue.writeBuffer(this.ubo, i * UBO_SLOT, new Float32Array([
          compW, compH, d.w, d.h,
          d.x, d.y, d.scaleX, d.scaleY,
          w, d.rot * Math.PI / 180, 0, 0,
        ]));
      }
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: tgt.accumView, loadOp: i === 0 ? 'clear' : 'load', storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        }],
      });
      if (d) {
        pass.setPipeline(this.accumPipe);
        pass.setBindGroup(0, this._bindGroup(d.view), [i * UBO_SLOT]);
        pass.draw(6);
      }
      pass.end();
      dev.queue.submit([enc.finish()]);
    }
    const enc = dev.createCommandEncoder({ label: 'motion blur resolve' });
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: tgt.view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
    });
    rp.setPipeline(this.resolvePipe);
    rp.setBindGroup(0, tgt.resolveBG);
    rp.draw(3);
    rp.end();
    dev.queue.submit([enc.finish()]);
    return tgt.view;
  }

  release(clipId) {
    const t = this.targets.get(clipId);
    if (!t) return;
    t.accum.destroy();
    t.out.destroy();
    this.targets.delete(clipId);
  }
}
