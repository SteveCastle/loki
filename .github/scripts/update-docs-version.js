// Point the docs site's download buttons and "Version X" labels at a release.
//
//   node .github/scripts/update-docs-version.js 2.32.0 [docsDir]
//
// Run by the Release workflow once the release's binaries are published, so the
// buttons never link to assets that do not exist yet. Idempotent: running it
// again for the same version changes nothing. Exits 1 if the home page no
// longer contains any version reference (it was restructured: update the
// patterns below rather than silently shipping stale links).
const fs = require('fs');
const path = require('path');

const SEMVER = '\\d+\\.\\d+\\.\\d+';

// Each rule keeps the text before the version (capture group 1) and swaps only the version.
const RULES = [
  // <p class="version">Version 2.31.0</p>
  new RegExp(`(class="version">Version )${SEMVER}`, 'g'),
  // .../releases/download/v2.31.0/
  new RegExp(`(releases/download/v)${SEMVER}`, 'g'),
  // Lowkey.Media.Viewer.Setup.2.31.0.exe, ...Viewer-2.31.0-arm64.dmg, ...Viewer-2.31.0.AppImage
  new RegExp(`(Lowkey\\.Media\\.Viewer(?:\\.Setup\\.|-))${SEMVER}`, 'g'),
];

function updateText(text, version) {
  let count = 0;
  let out = text;
  RULES.forEach((re) => {
    out = out.replace(re, (_m, prefix) => {
      count += 1;
      return `${prefix}${version}`;
    });
  });
  return { text: out, count };
}

function htmlFiles(dir) {
  const found = [];
  fs.readdirSync(dir, { withFileTypes: true }).forEach((e) => {
    const p = path.join(dir, e.name);
    // v1/ is the archived old site; superpowers/ holds planning notes.
    if (e.isDirectory() && !['v1', 'superpowers', 'static', 'node_modules'].includes(e.name)) {
      found.push(...htmlFiles(p));
    } else if (e.isFile() && e.name.endsWith('.html')) {
      found.push(p);
    }
  });
  return found;
}

function main() {
  const version = process.argv[2];
  const docsDir = process.argv[3] || 'docs';
  if (!new RegExp(`^${SEMVER}$`).test(version || '')) {
    console.error('usage: update-docs-version.js <x.y.z> [docsDir]');
    process.exit(2);
  }
  htmlFiles(docsDir).forEach((file) => {
    const before = fs.readFileSync(file, 'utf8');
    const { text, count } = updateText(before, version);
    if (count === 0) return; // page has no version references: nothing to do
    if (text !== before) {
      fs.writeFileSync(file, text);
      console.log(`updated ${file} (${count} reference${count > 1 ? 's' : ''})`);
    } else {
      console.log(`current  ${file}`);
    }
  });
  const index = path.join(docsDir, 'index.html');
  if (!fs.existsSync(index) || updateText(fs.readFileSync(index, 'utf8'), version).count === 0) {
    console.error(`::error::${index} has no version references; update the patterns in update-docs-version.js`);
    process.exit(1);
  }
}

if (require.main === module) main();
module.exports = { updateText };
