# Lowkey Studio

A keyframe compositor for layering media and GPU shader effects, running
entirely in the browser on **WebGPU** — After-Effects-style timeline, clip
editing, per-property keyframes with easing, mask painting, and a live
shader editor. No install, no upload: media stays on your machine.

This directory is the **source of truth** for the studio app in the loki
monorepo. The published copy lives at `docs/studio/` (served by GitHub
Pages as the *Studio* tab of the docs site).

## Layout

```
studio/
├── index.html, app.js, comp.js, compositor.js,
│   timeline.js, shader-editor.js, style.css     the app
├── effects.json         generated shader-preset manifest
├── engine/              slangfx-web runtime (WebGPU multi-pass engine +
│                        wasm-toolchain glue) — vendored from the slangfx repo
├── vendor/              glslang.wasm (GLSL→SPIR-V) + twgsl.wasm (SPIR-V→WGSL)
├── shaders/             the bundled .slangp effect presets
├── publish.mjs          copy runtime files → ../docs/studio
└── serve.mjs            dev server for the docs tree (port 8790)
```

## Develop

```bash
node studio/publish.mjs      # sync studio/ → docs/studio/
node studio/serve.mjs        # http://localhost:8790/studio/
```

Open in Chrome/Edge 113+ (WebGPU required).

## Upstream

The shader **engine, compiler toolchain glue, presets, and CLI** live in
the [slangfx](../../beat-cut/slangfx) repo — that's where core library
changes happen. `engine/`, `vendor/`, and `shaders/` here are vendored
snapshots; re-copy from slangfx (`web/src`, `web/vendor`, `shaders/`) to
pick up upstream changes, then re-run `publish.mjs`.
