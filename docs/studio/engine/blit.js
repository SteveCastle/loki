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

/* Separable post-pass over the composed mask: grow/shrink (square-kernel
 * dilate/erode) and feather (gaussian blur). One pipeline, op selected per
 * pass; each effect runs as a horizontal then a vertical pass. */
const MASK_POST_WGSL = /* wgsl */ `
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
// x: radius (px), y: op (0 dilate / 1 erode / 2 blur), zw: uv step per px
@group(0) @binding(2) var<uniform> u : vec4<f32>;

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let radius = i32(u.x + 0.5);
  var v = textureSample(src, smp, in.uv).r;
  if (u.y > 1.5) {
    // Gaussian blur, sigma = radius/2 so the visible falloff spans ~radius.
    // Tap count is capped; beyond it the taps stride outward at fractional
    // offsets (the linear sampler interpolates), so large feathers cost the
    // same as radius-64 and stay smooth on already-soft mask edges.
    let taps = min(radius, 64);
    let stepPx = u.x / f32(taps);
    let sigma = max(u.x * 0.5, 0.5);
    var sum = v;
    var wsum = 1.0;
    for (var i = 1; i <= taps; i++) {
      let d = f32(i) * stepPx;
      let w = exp(-(d * d) / (2.0 * sigma * sigma));
      let o = u.zw * d;
      sum += (textureSample(src, smp, in.uv + o).r + textureSample(src, smp, in.uv - o).r) * w;
      wsum += 2.0 * w;
    }
    v = sum / wsum;
  } else if (u.y > 0.5) {
    for (var i = 1; i <= radius; i++) {
      let o = u.zw * f32(i);
      v = min(v, min(textureSample(src, smp, in.uv + o).r, textureSample(src, smp, in.uv - o).r));
    }
  } else {
    for (var i = 1; i <= radius; i++) {
      let o = u.zw * f32(i);
      v = max(v, max(textureSample(src, smp, in.uv + o).r, textureSample(src, smp, in.uv - o).r));
    }
  }
  return vec4<f32>(v, v, v, 1.0);
}
`;

/* value an inactive/zero-strength node contributes per blend mode */
export const MASK_BLEND_IDENTITY = { add: 0, subtract: 0, multiply: 1, max: 0, min: 1 };

/* Post-pass radius caps. Expand is exact (one tap per pixel — striding a
 * min/max would skip thin features), so it stays at the loop bound. Feather
 * strides its taps beyond the loop bound, so it can range much further. */
export const MASK_POST_MAX = 64;
export const MASK_FEATHER_MAX = 250;

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

    // Expand / feather post passes over the composed stack.
    const postModule = device.createShaderModule({ label: 'slangfx mask post', code: MASK_POST_WGSL });
    this.postLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.postPipeline = device.createRenderPipeline({
      label: 'slangfx mask post',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.postLayout] }),
      vertex: { module: postModule, entryPoint: 'vs' },
      fragment: { module: postModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    });
    // clamp-to-edge (WebGPU default) so grown masks don't wrap.
    this.postSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  }

  /* Ping-pong textures + per-pass uniforms for one maskState's expand /
   * feather chain, (re)created when the mask dims change. Up to 4 passes:
   * morph H, morph V, blur H, blur V — even passes always read texA, odd
   * passes read texB, so bind groups are fixed per slot. */
  _ensurePost(maskState, w, h) {
    let p = maskState._post;
    if (p && (p.w !== w || p.h !== h)) { this.destroyPost(maskState); p = null; }
    if (!p) {
      const mk = () => this.device.createTexture({
        label: 'slangfx mask post',
        size: [w, h],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      const texA = mk(), texB = mk();
      const viewA = texA.createView(), viewB = texB.createView();
      const ubos = [0, 1, 2, 3].map(() => this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }));
      const bgs = ubos.map((ubo, i) => this.device.createBindGroup({
        layout: this.postLayout,
        entries: [
          { binding: 0, resource: i % 2 === 0 ? viewA : viewB },
          { binding: 1, resource: this.postSampler },
          { binding: 2, resource: { buffer: ubo } },
        ],
      }));
      p = { w, h, texA, texB, viewA, viewB, ubos, bgs };
      maskState._post = p;
    }
    return p;
  }

  /** Release a maskState's post-pass GPU state (no-op when absent). */
  destroyPost(maskState) {
    const p = maskState?._post;
    if (!p) return;
    p.texA.destroy();
    p.texB.destroy();
    for (const b of p.ubos) b.destroy();
    delete maskState._post;
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
   * or active=false (source clip not under the playhead) are skipped.
   * maskState.expand (signed px) and .feather (px) run as separable post
   * passes on the composed result; when both are 0 the stack renders
   * straight into maskView with no extra cost. Dims come from
   * layer.maskW/maskH (media masks) or layer.maskTex (fx layers). */
  encode(encoder, layer) {
    const st = layer.maskState;
    const nodes = st?.nodes ?? [];
    for (const n of nodes)
      if (n._optsBuf && n.enabled !== false && n.active !== false) this.writeNodeOpts(n);

    const expand = Math.max(-MASK_POST_MAX, Math.min(MASK_POST_MAX, Math.round(st?.expand ?? 0)));
    const feather = Math.max(0, Math.min(MASK_FEATHER_MAX, Math.round(st?.feather ?? 0)));
    const w = layer.maskW ?? layer.maskTex?.width ?? 0;
    const h = layer.maskH ?? layer.maskTex?.height ?? 0;
    const post = (expand !== 0 || feather > 0) && w > 0 && h > 0
      ? this._ensurePost(st, w, h) : null;

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: post ? post.viewA : layer.maskView,
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
    if (!post) return;

    const steps = [];
    if (expand !== 0) {
      const op = expand > 0 ? 0 : 1;   // dilate grows white, erode shrinks it
      steps.push({ r: Math.abs(expand), op, dir: 'h' }, { r: Math.abs(expand), op, dir: 'v' });
    }
    if (feather > 0)
      steps.push({ r: feather, op: 2, dir: 'h' }, { r: feather, op: 2, dir: 'v' });
    steps.forEach((s, i) => {
      this.device.queue.writeBuffer(post.ubos[i], 0, new Float32Array([
        s.r, s.op, s.dir === 'h' ? 1 / w : 0, s.dir === 'h' ? 0 : 1 / h,
      ]));
      const last = i === steps.length - 1;
      const target = last ? layer.maskView : (i % 2 === 0 ? post.viewB : post.viewA);
      const p = encoder.beginRenderPass({
        colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      });
      p.setPipeline(this.postPipeline);
      p.setBindGroup(0, post.bgs[i]);
      p.draw(3);
      p.end();
    });
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
