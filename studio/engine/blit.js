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

const MASK_NODE_WGSL = /* wgsl */ `
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
// p0: strength, invert, similarity, smoothness
// p1: key r, key g, key b, blend-identity (value an inactive node would write)
// p2: kind (0 = sample channel, 1 = chroma key), channel (0 r / 1 luma / 2 alpha)
struct NodeOpts {
  p0 : vec4<f32>,
  p1 : vec4<f32>,
  p2 : vec4<f32>,
};
@group(0) @binding(2) var<uniform> u : NodeOpts;

fn cbcr(c : vec3<f32>) -> vec2<f32> {
  return vec2<f32>(
    -0.168736 * c.r - 0.331264 * c.g + 0.5      * c.b,
     0.5      * c.r - 0.418688 * c.g - 0.081312 * c.b);
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let c = textureSample(src, smp, in.uv);
  var v : f32;
  if (u.p2.x < 0.5) {
    if (u.p2.y < 0.5)      { v = c.r; }
    else if (u.p2.y < 1.5) { v = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)); }
    else                   { v = c.a; }
  } else {
    let d = distance(cbcr(c.rgb), cbcr(u.p1.rgb));
    v = 1.0 - smoothstep(u.p0.z, u.p0.z + max(u.p0.w, 1e-4), d);
  }
  if (u.p0.y > 0.5) { v = 1.0 - v; }
  // Strength eases toward the blend mode's identity, so 0 = node has no
  // effect for every mode (including multiply, whose identity is 1).
  v = mix(u.p1.w, v, u.p0.x);
  return vec4<f32>(v, v, v, 1.0);
}
`;

/* value an inactive/zero-strength node contributes per blend mode */
export const MASK_BLEND_IDENTITY = { add: 0, subtract: 0, multiply: 1, max: 0, min: 1 };

const MASK_BLEND_STATES = {
  add:      { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } },
  subtract: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'reverse-subtract' } },
  multiply: { color: { srcFactor: 'dst', dstFactor: 'zero', operation: 'add' } },
  max:      { color: { srcFactor: 'one', dstFactor: 'one', operation: 'max' } },
  min:      { color: { srcFactor: 'one', dstFactor: 'one', operation: 'min' } },
};

/**
 * Renders a layer's mask node stack into its mask texture, one fullscreen
 * pass per frame: clear to black, then each node samples its source
 * texture (painted canvas, chroma-keyed input, another layer's matte, or —
 * later — an AI roto frame) and blends in with its mode. Node sources are
 * plain texture views resolved per rebuild, so time-varying sources only
 * need to swap the view (or re-upload the texture) between frames.
 */
export class MaskComposer {
  constructor(device) {
    this.device = device;
    const module = device.createShaderModule({ label: 'slangfx mask node', code: MASK_NODE_WGSL });
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.pipelines = {};
    for (const [mode, blendColor] of Object.entries(MASK_BLEND_STATES)) {
      this.pipelines[mode] = device.createRenderPipeline({
        label: `slangfx mask node ${mode}`,
        layout: pipelineLayout,
        vertex: { module, entryPoint: 'vs' },
        fragment: {
          module,
          entryPoint: 'fs',
          targets: [{
            format: 'rgba8unorm',
            blend: {
              color: blendColor.color,
              alpha: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' },
            },
          }],
        },
        primitive: { topology: 'triangle-list' },
      });
    }
  }

  bindGroup(srcView, sampler, optsBuf) {
    return this.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: srcView },
        { binding: 1, resource: sampler },
        { binding: 2, resource: { buffer: optsBuf } },
      ],
    });
  }

  writeNodeOpts(node) {
    const kind = node.kind === 'key' ? 1 : 0;
    const channel = node.channel === 'luma' ? 1 : node.channel === 'alpha' ? 2 : 0;
    const key = node.keyRGB ?? [0, 1, 0];
    this.device.queue.writeBuffer(node._optsBuf, 0, new Float32Array([
      node.strength ?? 1, node.invert ? 1 : 0, node.similarity ?? 0.18, node.smoothness ?? 0.1,
      key[0], key[1], key[2], MASK_BLEND_IDENTITY[node.blend] ?? 0,
      kind, channel, 0, 0,
    ]));
  }

  /** Encode the full stack into layer.maskView. Nodes with enabled=false
   * or active=false (source clip not under the playhead) are skipped. */
  encode(encoder, layer) {
    const nodes = layer.maskState?.nodes ?? [];
    for (const n of nodes)
      if (n._optsBuf && n.enabled !== false && n.active !== false) this.writeNodeOpts(n);
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: layer.maskView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    for (const n of nodes) {
      if (!n._bindGroup || n.enabled === false || n.active === false) continue;
      pass.setPipeline(this.pipelines[n.blend] ?? this.pipelines.add);
      pass.setBindGroup(0, n._bindGroup);
      pass.draw(3);
    }
    pass.end();
  }
}

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
