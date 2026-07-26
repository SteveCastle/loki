// Publish Lowkey Studio to the docs site: copies the runtime files from
// studio/ into docs/studio/, which GitHub Pages serves at /studio/.
// Usage: node studio/publish.mjs
import { cpSync, rmSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const src = path.dirname(fileURLToPath(import.meta.url));
const dest = path.resolve(src, '..', 'docs', 'studio');

const FILES = [
  'index.html', 'style.css', 'effects.json', 'studio-icon.png',
  'app.js', 'comp.js', 'compositor.js', 'timeline.js', 'shader-editor.js',
  'driver.js', 'audio-analysis.js',
];
const DIRS = ['engine', 'vendor', 'shaders', 'demo'];

rmSync(dest, { recursive: true, force: true });
mkdirSync(dest, { recursive: true });
for (const f of FILES) cpSync(path.join(src, f), path.join(dest, f));
for (const d of DIRS) cpSync(path.join(src, d), path.join(dest, d), { recursive: true });
console.log(`published studio → ${dest}`);
