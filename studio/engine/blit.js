/*
 * slangfx-web — fullscreen blit helper.
 *
 * WebGPU has no vkCmdBlitImage; a trivial textured fullscreen triangle
 * stands in for it. Used for: presenting the final pass to the canvas
 * (aspect-fit), and generating mip chains for `mipmap_input` passes.
 *
 * Texture-space invariant used across the engine: row 0 of every texture is
 * the TOP of the image. NDC y=+1 maps to viewport row 0 in WebGPU, so the
 * fullscreen geometry samples v = (1 - ndc.y) / 2.
 */

const BLIT_WGSL = /* wgsl */ `
struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i : u32) -> VSOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
    vec2<f32>(-1.0,  1.0),
  );
  var out : VSOut;
  out.pos = vec4<f32>(p[i], 0.0, 1.0);
  out.uv = vec2<f32>((p[i].x + 1.0) * 0.5, (1.0 - p[i].y) * 0.5);
  return out;
}

@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var smp : sampler;

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  return textureSample(src, smp, in.uv);
}
`;

export class Blitter {
  constructor(device) {
    this.device = device;
    this.module = device.createShaderModule({ label: 'slangfx blit', code: BLIT_WGSL });
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.pipelines = new Map(); // format -> GPURenderPipeline
    this.linearSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.nearestSampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
  }

  pipelineFor(format) {
    let p = this.pipelines.get(format);
    if (!p) {
      p = this.device.createRenderPipeline({
        label: `slangfx blit ${format}`,
        layout: this.pipelineLayout,
        vertex: { module: this.module, entryPoint: 'vs' },
        fragment: { module: this.module, entryPoint: 'fs', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      });
      this.pipelines.set(format, p);
    }
    return p;
  }

  bindGroup(view, sampler) {
    return this.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: view },
        { binding: 1, resource: sampler ?? this.linearSampler },
      ],
    });
  }

  /** Draw `view` over the full target. */
  blit(encoder, view, targetView, targetFormat, { sampler, clear = true, viewport = null } = {}) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: targetView,
        loadOp: clear ? 'clear' : 'load',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.pipelineFor(targetFormat));
    pass.setBindGroup(0, this.bindGroup(view, sampler));
    if (viewport) pass.setViewport(viewport.x, viewport.y, viewport.w, viewport.h, 0, 1);
    pass.draw(3);
    pass.end();
  }

  /** Generate mips level 1..n by blitting each level from the previous. */
  generateMips(encoder, texture, format, mipLevelCount) {
    for (let level = 1; level < mipLevelCount; level++) {
      const srcView = texture.createView({ baseMipLevel: level - 1, mipLevelCount: 1 });
      const dstView = texture.createView({ baseMipLevel: level, mipLevelCount: 1 });
      this.blit(encoder, srcView, dstView, format, { sampler: this.linearSampler });
    }
  }
}

const MASK_BLEND_WGSL = /* wgsl */ `
struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i : u32) -> VSOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
    vec2<f32>(-1.0,  1.0),
  );
  var out : VSOut;
  out.pos = vec4<f32>(p[i], 0.0, 1.0);
  out.uv = vec2<f32>((p[i].x + 1.0) * 0.5, (1.0 - p[i].y) * 0.5);
  return out;
}

@group(0) @binding(0) var t_in : texture_2d<f32>;
@group(0) @binding(1) var t_fx : texture_2d<f32>;
@group(0) @binding(2) var t_mask : texture_2d<f32>;
@group(0) @binding(3) var smp : sampler;
// x = opacity, y = invert (0/1)
@group(0) @binding(4) var<uniform> opts : vec4<f32>;

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  var m = textureSample(t_mask, smp, in.uv).r;
  if (opts.y > 0.5) { m = 1.0 - m; }
  m = m * opts.x;
  let a = textureSample(t_in, smp, in.uv);
  let b = textureSample(t_fx, smp, in.uv);
  return mix(a, b, m);
}
`;

/** Composites a layer's effect over its input through a painted mask:
 * out = mix(input, effect, mask.r * opacity), optionally inverted. */
export class MaskBlender {
  constructor(device) {
    this.device = device;
    const module = device.createShaderModule({ label: 'slangfx mask blend', code: MASK_BLEND_WGSL });
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.pipeline = device.createRenderPipeline({
      label: 'slangfx mask blend',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  bindGroup(inputView, effectView, maskView, sampler, optsBuf) {
    return this.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: inputView },
        { binding: 1, resource: effectView },
        { binding: 2, resource: maskView },
        { binding: 3, resource: sampler },
        { binding: 4, resource: { buffer: optsBuf } },
      ],
    });
  }

  encode(encoder, bindGroup, targetView) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: targetView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }
}
