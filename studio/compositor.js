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
  // misc: opacity 0..1, rotation (rad), mask opacity, mask invert (0/1)
  misc  : vec4<f32>,
};

@group(0) @binding(0) var<uniform> xf : Xform;
@group(0) @binding(1) var tex : texture_2d<f32>;
@group(0) @binding(2) var smp : sampler;
// Comp-space mask multiplied into the clip's alpha (1x1 white = no mask).
@group(0) @binding(3) var maskTex : texture_2d<f32>;

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
  var m = textureSample(maskTex, smp, in.pos.xy / xf.sizes.xy).r;
  if (xf.misc.w > 0.5) { m = 1.0 - m; }
  let a = color.a * xf.misc.x * mix(1.0, m, xf.misc.z);
  return vec4<f32>(color.rgb, a);
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
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });
    const white = device.createTexture({
      size: [1, 1], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: white }, new Uint8Array([255, 255, 255, 255]), {}, [1, 1]);
    this.whiteView = white.createView();
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

  _item(clipId, view, maskView) {
    const mask = maskView ?? this.whiteView;
    let item = this.items.get(clipId);
    if (!item || item.view !== view || item.maskView !== mask) {
      const ubo = item?.ubo ?? this.device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      item = {
        ubo,
        view,
        maskView: mask,
        bindGroup: this.device.createBindGroup({
          layout: this.layout,
          entries: [
            { binding: 0, resource: { buffer: ubo } },
            { binding: 1, resource: view },
            { binding: 2, resource: this.sampler },
            { binding: 3, resource: mask },
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
  composite(encoder, targetView, compW, compH, draws, { over = false, transparent = false } = {}) {
    for (const d of draws) {
      const item = this._item(d.clipId, d.view, d.maskView);
      this.device.queue.writeBuffer(item.ubo, 0, new Float32Array([
        compW, compH, d.w, d.h,
        d.x, d.y, d.scaleX, d.scaleY,
        d.opacity, d.rot * Math.PI / 180,
        d.maskView ? (d.maskOpacity ?? 1) : 0, d.maskInvert ? 1 : 0,
      ]));
    }
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: targetView,
        loadOp: over ? 'load' : 'clear',
        storeOp: 'store',
        // transparent: matte targets need a=0 background so a clip's own
        // alpha can serve as the mask outside its quad.
        clearValue: { r: 0, g: 0, b: 0, a: transparent ? 0 : 1 },
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
