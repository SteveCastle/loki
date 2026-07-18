/*
 * slangfx studio — media layer compositor.
 *
 * Draws each active media clip's texture into the comp frame (the engine's
 * input texture) with a 2D transform — position, uniform scale, rotation,
 * opacity — using ordinary source-over alpha blending. Clips are drawn
 * bottom track first, so higher tracks composite on top; the shader chain
 * then runs on the finished frame.
 *
 * One tiny pipeline; per-clip uniform buffer + bind group (cached by clip
 * id, invalidated when the source texture view changes).
 */

const COMPOSITE_WGSL = /* wgsl */ `
struct Xform {
  // sizes: comp W, comp H, media w, media h  (pixels)
  sizes : vec4<f32>,
  // place: center x, center y (comp px, y-down), scale x, scale y (1 = 100%)
  place : vec4<f32>,
  // misc: opacity 0..1, rotation (rad)
  misc  : vec4<f32>,
};

@group(0) @binding(0) var<uniform> xf : Xform;
@group(0) @binding(1) var tex : texture_2d<f32>;
@group(0) @binding(2) var smp : sampler;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i : u32) -> VSOut {
  // Two triangles over the unit quad; corner (0,0) = media top-left.
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
  );
  let c = corners[i];
  let media = xf.sizes.zw;
  let comp = xf.sizes.xy;
  // Local space: centered on the media, y-down comp pixels.
  var p = (c - vec2<f32>(0.5, 0.5)) * media * xf.place.zw;
  let r = xf.misc.y;
  let rot = mat2x2<f32>(cos(r), sin(r), -sin(r), cos(r));
  p = rot * p + xf.place.xy;
  // Comp pixels (y-down) -> NDC (y-up).
  let ndc = vec2<f32>(p.x / comp.x * 2.0 - 1.0, 1.0 - p.y / comp.y * 2.0);
  var out : VSOut;
  out.pos = vec4<f32>(ndc, 0.0, 1.0);
  out.uv = c;
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let color = textureSample(tex, smp, in.uv);
  return vec4<f32>(color.rgb, color.a * xf.misc.x);
}
`;

export class Compositor {
  constructor(device, format = 'rgba8unorm') {
    this.device = device;
    const module = device.createShaderModule({ label: 'slangfx compositor', code: COMPOSITE_WGSL });
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    this.pipeline = device.createRenderPipeline({
      label: 'slangfx compositor',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [{
          format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.items = new Map(); // clipId -> {ubo, bindGroup, view}
  }

  _item(clipId, view) {
    let item = this.items.get(clipId);
    if (!item || item.view !== view) {
      const ubo = item?.ubo ?? this.device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      item = {
        ubo,
        view,
        bindGroup: this.device.createBindGroup({
          layout: this.layout,
          entries: [
            { binding: 0, resource: { buffer: ubo } },
            { binding: 1, resource: view },
            { binding: 2, resource: this.sampler },
          ],
        }),
      };
      this.items.set(clipId, item);
    }
    return item;
  }

  release(clipId) {
    const item = this.items.get(clipId);
    if (item) { item.ubo.destroy(); this.items.delete(clipId); }
  }

  /**
   * Composite `draws` into `targetView`. By default the target is cleared
   * to opaque black first; pass `over: true` to draw on top of existing
   * contents (used to layer media above an effect's output).
   * @param {Array<{clipId, view, w, h, x, y, scaleX, scaleY, rot, opacity}>}
   *        draws bottom-most first; scale 1 = 100%, rot degrees, opacity 0..1
   */
  composite(encoder, targetView, compW, compH, draws, { over = false } = {}) {
    for (const d of draws) {
      const item = this._item(d.clipId, d.view);
      this.device.queue.writeBuffer(item.ubo, 0, new Float32Array([
        compW, compH, d.w, d.h,
        d.x, d.y, d.scaleX, d.scaleY,
        d.opacity, d.rot * Math.PI / 180, 0, 0,
      ]));
    }
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: targetView,
        loadOp: over ? 'load' : 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.pipeline);
    for (const d of draws) {
      pass.setBindGroup(0, this.items.get(d.clipId).bindGroup);
      pass.draw(6);
    }
    pass.end();
  }
}
