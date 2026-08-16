# Lowkey Studio

A keyframe compositor for layering media and GPU shader effects, running
entirely in the browser on **WebGPU** — After-Effects-style timeline, clip
editing, per-property keyframes with easing, mask painting, and a live
shader editor. No install, no upload: media stays on your machine.

This directory is the **source of truth** for the studio app in the loki
monorepo. The published copy lives at `docs/studio/` (served by GitHub
Pages as the *Studio* tab of the docs site).

**[ARCHITECTURE.md](ARCHITECTURE.md)** is the technical tour: what each
module owns, what happens in a frame, how the shader engine and the mask,
driver and export paths fit together.

## Effects, layers, and scope

Every clip — media or fx — owns an ordered **effect stack** (`clip.effects`).
Where the stack runs is the only difference between the two clip kinds:

| stack lives on   | what it processes                                    |
| ---------------- | ---------------------------------------------------- |
| a **media** clip | that clip alone, before it composites into the frame |
| an **fx** clip   | every layer below it — an *adjustment layer*         |

A clip's mask gates its whole stack as one group (on an fx clip) or cuts the
clip's alpha (on a media clip), so masking a group of effects means masking
one clip.

**Layers vs. effects in the UI.** `+ Layer` makes layers — an empty
adjustment layer, a title, or a shape layer you draw onto the preview. The
`＋` search box and the inspector's `+ Effect` only ever *stack onto a clip*:
the search box targets whichever layer the inspector has in focus (there is
always one — the inspector falls back to the topmost layer live at the
playhead rather than showing nothing), and `+ Effect` targets its own row's
clip.

**Shape layers** hold a *list* of shapes (`clip.shapes`), each with a preset,
a fill and a rect in the layer's unit texture space; array order is z-order.
`+ Shape` in the inspector draws another one into the same layer. The
texture square is the layer's footprint, so adding or removing a shape
shrink-wraps it onto the union of the shapes and applies the inverse change
to the clip transform (values *and* keyframes) — the box the gizmo drags
always hugs the shapes, and nothing moves on screen when it refits.

Implementation: an engine layer is one *effect*, and the layers a clip
contributes share that clip's id as their `groupId` — `SlangFx.rebuild()`
then builds the group's mask once on the head layer and blends the stack's
output back over the group's input at the tail. A media clip's stack can't
live in that shared chain (it must not see other layers), so those clips get
a private headless `SlangFx` fed a transparent frame holding the clip alone;
the processed result composites as a full-frame quad, gated back to the
clip's own alpha unless *let effects spill outside the media* is on.

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
the [slangfx](https://github.com/SteveCastle/slangfx) repo — that's where core library
changes happen. `engine/`, `vendor/`, and `shaders/` here are vendored
snapshots; re-copy from slangfx (`web/src`, `web/vendor`, `shaders/`) to
pick up upstream changes, then re-run `publish.mjs`.
