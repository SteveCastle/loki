#!/usr/bin/env node
// Generate docs/server/api.html from the OpenAPI spec.
//
//   node scripts/build-api-docs.js          write docs/server/api.html
//   node scripts/build-api-docs.js --check  exit 1 if api.html is out of date
//
// Sources:
//   docs/server/openapi.yaml    every endpoint (the source of truth; the Go test
//                               media-server/openapi_coverage_test.go fails when
//                               a registered route is missing from it)
//   docs/server/api-intro.html  hand-written Conventions section
//   scripts/api-docs/head.html  page head, site header, hero
//   scripts/api-docs/foot.html  closing markup and the TOC script
//
// Descriptions in the spec use a markdown subset: paragraphs, `code`,
// **bold**, *italic*, [links](url) and "- " bullet lists.
'use strict';

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

const ROOT = path.resolve(__dirname, '..');
const DOCS = path.join(ROOT, 'docs', 'server');
const OUT = path.join(DOCS, 'api.html');

const spec = yaml.load(fs.readFileSync(path.join(DOCS, 'openapi.yaml'), 'utf8'));
const schemas = (spec.components && spec.components.schemas) || {};
const METHODS = ['get', 'post', 'put', 'patch', 'delete', 'head', 'options'];
const ACCESS = { public: 'Public', read: 'Read', admin: 'Admin' };

// ---------------------------------------------------------------- helpers

const esc = (s) =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function inline(text) {
  let s = esc(text);
  const codes = [];
  s = s.replace(/`([^`]+)`/g, (_, c) => {
    codes.push(`<code>${c}</code>`);
    return `\u0000${codes.length - 1}\u0000`;
  });
  s = s
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t, u) => `<a href="${u}">${t}</a>`);
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => codes[+i]);
}

function md(text) {
  if (!text) return '';
  return String(text)
    .trim()
    .split(/\n\s*\n/)
    .map((block) => {
      const lines = block.split('\n');
      if (lines.every((l) => /^\s*- /.test(l) || /^\s{2,}\S/.test(l))) {
        const items = [];
        lines.forEach((l) => {
          if (/^\s*- /.test(l)) items.push(l.replace(/^\s*- /, ''));
          else items[items.length - 1] += ' ' + l.trim();
        });
        return `<ul>${items.map((i) => `<li>${inline(i)}</li>`).join('')}</ul>`;
      }
      return `<p>${inline(lines.join(' '))}</p>`;
    })
    .join('\n');
}

function resolve(s) {
  if (s && s.$ref) {
    const name = s.$ref.split('/').pop();
    return { name, schema: schemas[name] || {} };
  }
  return { name: null, schema: s || {} };
}

function pathHTML(p) {
  return esc(p).replace(/(\{[^}]+\})/g, '<span class="c-ph">$1</span>');
}

// One-line type label, e.g. "string", "array of Job", "map of string", "string | null".
function typeLabel(s) {
  if (!s) return '';
  if (s.$ref) {
    const name = s.$ref.split('/').pop();
    return `<a class="t-ref" href="#schema-${slug(name)}">${esc(name)}</a>`;
  }
  const alts = s.oneOf || s.anyOf;
  if (alts) return alts.map(typeLabel).join(' <span class="c-punct">|</span> ');
  let t = s.type;
  if (Array.isArray(t)) return t.map((x) => esc(x)).join(' <span class="c-punct">|</span> ');
  if (t === 'array') return `array of ${typeLabel(s.items || {}) || 'any'}`;
  if (t === 'object' && s.additionalProperties && typeof s.additionalProperties === 'object' && !s.properties) {
    return `map of ${typeLabel(s.additionalProperties)}`;
  }
  if (!t && s.enum) t = 'string';
  let out = esc(t || (s.properties ? 'object' : 'any'));
  if (s.format) out += ` <span class="c-punct">(${esc(s.format)})</span>`;
  return out;
}

function valueList(vals) {
  return vals.map((v) => `<code data-colored="1"><span class="c-val">${esc(JSON.stringify(v))}</span></code>`).join(' ');
}

// Children to expand under a field: object properties, or the item object of an array.
function childrenOf(s, depth) {
  if (depth > 6) return null;
  const r = resolve(s);
  if (r.name && depth > 0) return null; // named types link instead of expanding
  let t = r.schema;
  if (t.type === 'array' && t.items) {
    const it = resolve(t.items);
    if (it.name) return null;
    t = it.schema;
  }
  if (t.properties) return t;
  if (t.type === 'object' && t.additionalProperties && typeof t.additionalProperties === 'object') {
    const ap = resolve(t.additionalProperties);
    if (!ap.name && ap.schema.properties) return ap.schema;
  }
  return null;
}

function fieldRows(obj, depth) {
  const required = new Set(obj.required || []);
  return Object.entries(obj.properties || {})
    .map(([name, prop]) => fieldRow(name, prop, required.has(name), depth))
    .join('');
}

function fieldRow(name, prop, required, depth) {
  const r = resolve(prop);
  const p = r.name ? prop : r.schema;
  const meta = [];
  const target = r.name ? schemas[r.name] || {} : p;
  if (target.enum) meta.push(`One of ${valueList(target.enum)}.`);
  if (p.default !== undefined) meta.push(`Default ${valueList([p.default])}.`);
  if (p.minimum !== undefined || p.maximum !== undefined) {
    meta.push(
      [p.minimum !== undefined ? `min ${p.minimum}` : '', p.maximum !== undefined ? `max ${p.maximum}` : '']
        .filter(Boolean)
        .join(', ') + '.'
    );
  }
  const desc = (p.description || (r.name ? '' : '')) + '';
  const kids = childrenOf(prop, depth + (r.name ? 1 : 0));
  return (
    `<div class="field" style="--depth:${depth}">` +
    `<div class="f-head"><code class="f-name" data-colored="1">${esc(name)}</code>` +
    `<span class="f-type">${typeLabel(prop)}</span>` +
    (required ? '<span class="f-req">required</span>' : '') +
    (p.deprecated ? '<span class="f-dep">deprecated</span>' : '') +
    `</div>` +
    (desc || meta.length ? `<div class="f-desc">${md(desc)}${meta.length ? `<p class="f-meta">${meta.join(' ')}</p>` : ''}</div>` : '') +
    `</div>` +
    (kids ? fieldRows(kids, depth + 1) : '')
  );
}

// Body for a request or response: type line, then the expanded shape.
function bodyHTML(schemaRef) {
  if (!schemaRef) return '';
  const r = resolve(schemaRef);
  let s = r.schema;
  let head = '';
  if (r.name) head = `<p class="shape-type">${typeLabel(schemaRef)}</p>`;
  if (s.type === 'array' && s.items) {
    head = `<p class="shape-type">${typeLabel(s.type === 'array' && !r.name ? s : schemaRef)}</p>`;
    const it = resolve(s.items);
    s = it.schema;
  } else if (!r.name && !s.properties) {
    head = `<p class="shape-type">${typeLabel(s)}</p>`;
  }
  const desc = !r.name && s.description ? md(s.description) : '';
  const rows = s.properties ? `<div class="fields">${fieldRows(s, 0)}</div>` : '';
  const alt = !s.properties && (s.oneOf || s.anyOf) ? '' : '';
  return head + desc + rows + alt;
}

function exampleHTML(media) {
  let ex = media.example;
  if (ex === undefined && media.examples) {
    const first = Object.values(media.examples)[0];
    ex = first && first.value;
  }
  if (ex === undefined) return '';
  const text = typeof ex === 'string' ? ex : JSON.stringify(ex, null, 2);
  return (
    `<details class="op-example"><summary>Example</summary><div class="code-block">` +
    text
      .split('\n')
      .map((l) => `<code>${esc(l)}</code>`)
      .join('<br>') +
    `</div></details>`
  );
}

function contentHTML(content) {
  if (!content) return '';
  return Object.entries(content)
    .map(
      ([type, media]) =>
        `<p class="ctype"><code data-colored="1">${esc(type)}</code></p>` + bodyHTML(media.schema) + exampleHTML(media)
    )
    .join('');
}

function paramsHTML(params, where, title) {
  const list = params.filter((p) => p.in === where);
  if (!list.length) return '';
  const rows = list
    .map((p) => {
      const s = p.schema || {};
      return fieldRow(p.name, { ...s, description: p.description || s.description }, !!p.required, 0);
    })
    .join('');
  return `<div class="op-block"><h4 class="op-label">${title}</h4><div class="fields">${rows}</div></div>`;
}

function statusClass(code) {
  const c = String(code)[0];
  return c === '2' ? 's-ok' : c === '3' ? 's-redirect' : 's-err';
}

function operationHTML(p, method, op, pathParams) {
  const id = op.operationId || slug(method + '-' + p);
  const params = [...(pathParams || []), ...(op.parameters || [])].map((x) => (x.$ref ? resolveParam(x) : x));
  const access = ACCESS[op['x-access']] || esc(op['x-access'] || '');
  const responses = Object.entries(op.responses || {}).map(([code, r]) => [
    code,
    r && r.$ref ? ((spec.components.responses || {})[r.$ref.split('/').pop()] || {}) : r,
  ]);
  const ok = responses.filter(([c]) => /^[23]/.test(c) || c === 'default');
  const errs = responses.filter(([c]) => /^[45]/.test(c));
  let h = `<article class="op" id="op-${esc(id)}">`;
  h +=
    `<div class="op-head"><span class="op-method m-${method}">${method.toUpperCase()}</span>` +
    `<code class="op-path" data-colored="1"><span class="c-path">${pathHTML(p)}</span></code>` +
    `<span class="op-access" data-access="${esc(op['x-access'] || '')}">${access}</span>` +
    `<a class="op-anchor" href="#op-${esc(id)}" aria-label="Link to this endpoint">#</a></div>`;
  if (op.summary) h += `<p class="op-summary">${inline(op.summary)}</p>`;
  if (op.deprecated) h += `<p class="op-deprecated">Deprecated.</p>`;
  if (op.description) h += `<div class="op-desc">${md(op.description)}</div>`;
  h += paramsHTML(params, 'path', 'Path parameters');
  h += paramsHTML(params, 'query', 'Query parameters');
  h += paramsHTML(params, 'header', 'Headers');
  if (op.requestBody) {
    const rb = op.requestBody.$ref ? {} : op.requestBody;
    h +=
      `<div class="op-block"><h4 class="op-label">Request body${rb.required === false ? ' <span class="f-opt">optional</span>' : ''}</h4>` +
      (rb.description ? md(rb.description) : '') +
      contentHTML(rb.content) +
      `</div>`;
  }
  ok.forEach(([code, r]) => {
    h +=
      `<div class="op-block"><h4 class="op-label">Response <span class="status ${statusClass(code)}">${esc(code)}</span></h4>` +
      (r.description ? md(r.description) : '') +
      contentHTML(r.content) +
      `</div>`;
  });
  if (errs.length) {
    h +=
      `<div class="op-block"><h4 class="op-label">Errors</h4><ul class="op-errors">` +
      errs
        .map(([code, r]) => {
          const types = r.content ? Object.keys(r.content) : [];
          const body = types.length
            ? ` <span class="err-body">${types
                .map((t) => {
                  const sch = r.content[t].schema;
                  return sch && sch.$ref ? typeLabel(sch) : esc(t);
                })
                .join(', ')}</span>`
            : '';
          return `<li><span class="status ${statusClass(code)}">${esc(code)}</span><span class="err-desc">${inline(
            (r.description || '').trim()
          )}</span>${body}</li>`;
        })
        .join('') +
      `</ul></div>`;
  }
  return h + `</article>`;
}

function resolveParam(ref) {
  const name = ref.$ref.split('/').pop();
  return (spec.components.parameters || {})[name] || {};
}

function schemaHTML(name, s) {
  const r = { ...s };
  let h = `<article class="op schema" id="schema-${slug(name)}"><div class="op-head">` +
    `<code class="op-path" data-colored="1"><span class="c-cmd">${esc(name)}</span></code>` +
    `<span class="op-access">${typeLabel({ ...r, $ref: undefined }) || 'object'}</span>` +
    `<a class="op-anchor" href="#schema-${slug(name)}" aria-label="Link to this schema">#</a></div>`;
  if (r.description) h += `<div class="op-desc">${md(r.description)}</div>`;
  if (r.enum) h += `<p class="f-meta">One of ${valueList(r.enum)}.</p>`;
  if (r.properties) h += `<div class="fields">${fieldRows(r, 0)}</div>`;
  if (r.type === 'array' && r.items) {
    const kids = childrenOf(r.items, 0);
    if (kids) h += `<div class="fields">${fieldRows(kids, 0)}</div>`;
  }
  if (r.oneOf || r.anyOf) h += `<p class="f-meta">One of ${(r.oneOf || r.anyOf).map(typeLabel).join(', ')}.</p>`;
  return h + `</article>`;
}

// ---------------------------------------------------------------- page

const tagOrder = (spec.tags || []).map((t) => t.name);
const byTag = new Map(tagOrder.map((t) => [t, []]));
for (const [p, item] of Object.entries(spec.paths || {})) {
  for (const m of METHODS) {
    const op = item[m];
    if (!op) continue;
    const tag = (op.tags && op.tags[0]) || 'Other';
    if (!byTag.has(tag)) byTag.set(tag, []);
    byTag.get(tag).push([p, m, op, item.parameters]);
  }
}

const counts = { ops: 0, paths: Object.keys(spec.paths || {}).length };
const sections = [];
const toc = [];
for (const [tag, ops] of byTag) {
  if (!ops.length) continue;
  counts.ops += ops.length;
  const t = (spec.tags || []).find((x) => x.name === tag) || {};
  const id = slug(tag);
  toc.push(`          <li><a href="#${id}">${esc(tag)}</a></li>`);
  sections.push(
    `      <section id="${id}">\n        <h2>${esc(tag)}</h2>\n` +
      (t.description ? `        <div class="tag-desc">${md(t.description)}</div>\n` : '') +
      ops.map(([p, m, op, pp]) => '        ' + operationHTML(p, m, op, pp)).join('\n') +
      `\n      </section>`
  );
}
const schemaNames = Object.keys(schemas).sort((a, b) => a.localeCompare(b));
toc.push(`          <li><a href="#schemas">Schemas</a></li>`);
sections.push(
  `      <section id="schemas">\n        <h2>Schemas</h2>\n` +
    `        <p>Shared object shapes referenced above. Every type name in an endpoint links here.</p>\n` +
    schemaNames.map((n) => '        ' + schemaHTML(n, schemas[n])).join('\n') +
    `\n      </section>`
);

let head = fs.readFileSync(path.join(__dirname, 'api-docs', 'head.html'), 'utf8');
const foot = fs.readFileSync(path.join(__dirname, 'api-docs', 'foot.html'), 'utf8');
const intro = fs.readFileSync(path.join(DOCS, 'api-intro.html'), 'utf8');

// The release workflow bumps "Version X.Y.Z" in the published page; keep it.
if (fs.existsSync(OUT)) {
  const cur = /class="version">Version (\d+\.\d+\.\d+)/.exec(fs.readFileSync(OUT, 'utf8'));
  if (cur) head = head.replace(/(class="version">Version )\d+\.\d+\.\d+/, `$1${cur[1]}`);
}

// TOC for the hand-written intro: one entry per <section id>, with its
// <h3 id> subsections nested under it.
const tocIntro = [...intro.matchAll(/<section id="([^"]+)">\s*<h2>([^<]+)<\/h2>([\s\S]*?)<\/section>/g)].map(
  ([, id, title, body]) => {
    const subs = [...body.matchAll(/<h3 id="([^"]+)">([^<]+)<\/h3>/g)];
    const nested = subs.length
      ? '\n            <ul>\n' +
        subs.map(([, sid, st]) => `              <li><a href="#${sid}">${esc(st)}</a></li>`).join('\n') +
        '\n            </ul>\n          '
      : '';
    return `          <li><a href="#${id}">${esc(title)}</a>${nested}</li>`;
  }
);

const html =
  head +
  `    <div class="content-wrapper">\n      <nav class="toc">\n        <h2>Contents</h2>\n        <ul>\n` +
  tocIntro.join('\n') + '\n' +
  toc.join('\n') +
  `\n        </ul>\n      </nav>\n\n      <main>\n` +
  `      <!-- GENERATED by scripts/build-api-docs.js from docs/server/openapi.yaml - edit those, not this file. -->\n` +
  intro.trimEnd() + '\n\n' +
  sections.join('\n\n') +
  `\n      </main>\n` +
  foot;

if (process.argv.includes('--check')) {
  const cur = fs.existsSync(OUT) ? fs.readFileSync(OUT, 'utf8') : '';
  if (cur !== html) {
    console.error('docs/server/api.html is out of date: run node scripts/build-api-docs.js');
    process.exit(1);
  }
  console.log('docs/server/api.html is up to date');
} else {
  fs.writeFileSync(OUT, html);
  console.log(`wrote ${path.relative(ROOT, OUT)}: ${counts.ops} operations on ${counts.paths} paths, ${schemaNames.length} schemas`);
}
