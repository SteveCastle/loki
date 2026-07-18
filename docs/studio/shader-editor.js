/*
 * slangfx studio — lightweight slang/GLSL code editor.
 *
 * Zero-dependency editor: a transparent-text <textarea> layered over a
 * syntax-highlighted <pre> with a line-number gutter, kept in scroll sync.
 * The textarea stays the real input (native selection, IME, undo), the
 * <pre> underneath provides the colors.
 */

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/* One pass over the (already HTML-escaped) source; earliest alternation
 * wins, so comments swallow keywords inside them. */
const MASTER = new RegExp([
  /\/\*[\s\S]*?\*\//.source,                       // 1 block comment
  /\/\/[^\n]*/.source,                             // 2 line comment (may be //@param)
  /#[A-Za-z][^\n]*/.source,                        // 3 preprocessor / pragma
  /\b\d+\.\d+|\b\d+\.?|\.\d+/.source,              // 4 number
  /\b(?:if|else|for|while|return|break|continue|const|struct|void|uniform|layout|push_constant|std140|set|binding|location|in|out|flat)\b/.source, // 5 keyword
  /\b(?:float|vec[234]|mat[234]|int|uint|bool|sampler2D)\b/.source, // 6 type
  /\b(?:texture|textureLod|mix|clamp|smoothstep|step|fract|floor|ceil|round|abs|min|max|pow|exp2?|log2?|sqrt|inversesqrt|sin|cos|tan|asin|acos|atan|length|distance|normalize|dot|cross|reflect|refract|mod|sign|dFdx|dFdy)\b/.source, // 7 builtin
  /\b(?:params|global|Source|Original|Mask|vTexCoord|FragColor|gl_Position|Position|TexCoord|main)\b/.source, // 8 special
].map((p) => `(${p})`).join('|'), 'g');

const CLASSES = ['tk-c', 'tk-c', 'tk-d', 'tk-n', 'tk-k', 'tk-t', 'tk-b', 'tk-i'];

export function highlightSlang(src) {
  return esc(src).replace(MASTER, (m, ...groups) => {
    for (let i = 0; i < CLASSES.length; i++) {
      if (groups[i] !== undefined) {
        const cls = (i === 1 && m.startsWith('//@param')) ? 'tk-p' : CLASSES[i];
        return `<span class="${cls}">${m}</span>`;
      }
    }
    return m;
  });
}

/**
 * @param {object} opts
 * @param {string} opts.value initial text
 * @param {(text: string) => void} [opts.onInput]
 * @returns {{el: HTMLElement, textarea: HTMLTextAreaElement,
 *            getValue: () => string, setValue: (t: string) => void}}
 */
export function makeShaderEditor({ value = '', onInput = null } = {}) {
  const el = document.createElement('div');
  el.className = 'sed';
  el.innerHTML = `
    <pre class="sed-gutter" aria-hidden="true"></pre>
    <div class="sed-body">
      <pre class="sed-hl" aria-hidden="true"><code></code></pre>
      <textarea class="sed-ta" spellcheck="false" wrap="off" autocomplete="off" autocapitalize="off"></textarea>
    </div>`;
  const gutter = el.querySelector('.sed-gutter');
  const hl = el.querySelector('.sed-hl');
  const code = hl.querySelector('code');
  const ta = el.querySelector('.sed-ta');
  ta.value = value;

  let lineCount = -1;
  const refresh = () => {
    // Trailing newline so the last (possibly empty) line keeps its height.
    code.innerHTML = highlightSlang(ta.value) + '\n';
    const n = ta.value.split('\n').length;
    if (n !== lineCount) {
      lineCount = n;
      gutter.textContent = Array.from({ length: n }, (_, i) => i + 1).join('\n');
    }
  };
  const syncScroll = () => {
    hl.scrollTop = ta.scrollTop;
    hl.scrollLeft = ta.scrollLeft;
    gutter.scrollTop = ta.scrollTop;
  };

  ta.addEventListener('input', () => { refresh(); onInput?.(ta.value); });
  ta.addEventListener('scroll', syncScroll);
  ta.addEventListener('keydown', (e) => {
    e.stopPropagation();          // keep Space/arrow shortcuts out of the app
    if (e.key === 'Tab') {
      e.preventDefault();
      const { selectionStart: s, selectionEnd: t } = ta;
      ta.value = ta.value.slice(0, s) + '    ' + ta.value.slice(t);
      ta.selectionStart = ta.selectionEnd = s + 4;
      refresh();
      onInput?.(ta.value);
    }
  });
  refresh();

  return {
    el,
    textarea: ta,
    getValue: () => ta.value,
    setValue: (t) => { ta.value = t; refresh(); syncScroll(); },
  };
}

/* ---- cheat sheet ----------------------------------------------------- */

export const CHEAT_HTML = `
<h4>Tunable parameters</h4>
<pre><span class="tk-p">//@param name "Label" default min max step</span></pre>
<p>One line declares the uniform <em>and</em> its inspector slider +
keyframable timeline lane. Reference it by bare <code>name</code>.</p>

<h4>Per-frame inputs</h4>
<pre><code>vTexCoord          <span class="tk-c">// 0..1 UV, (0,0) = top-left</span>
params.SourceSize  <span class="tk-c">// (w, h, 1/w, 1/h) of Source</span>
params.OutputSize  <span class="tk-c">// (w, h, 1/w, 1/h) of this pass</span>
params.FrameCount  <span class="tk-c">// uint frame counter</span>
params.Time        <span class="tk-c">// seconds (comp time)</span></code></pre>

<h4>Samplers</h4>
<pre><code><span class="tk-c">// everything below this clip:</span>
layout(set=0, binding=2) uniform sampler2D Source;
<span class="tk-c">// this clip's painted mask (white = on):</span>
layout(set=0, binding=4) uniform sampler2D Mask;</code></pre>

<h4>Required skeleton</h4>
<pre><code>#version 450
layout(push_constant) uniform Push {
  vec4 SourceSize; vec4 OutputSize;
  uint FrameCount; float Time;
} params;
layout(std140, set=0, binding=0)
  uniform UBO { mat4 MVP; } global;

#pragma stage vertex
<span class="tk-c">/* keep the standard vertex main */</span>

#pragma stage fragment
layout(location=0) in  vec2 vTexCoord;
layout(location=0) out vec4 FragColor;</code></pre>

<h4>Recipes</h4>
<pre><code><span class="tk-c">// sample the frame</span>
vec3 c = texture(Source, uv).rgb;
<span class="tk-c">// pixel-size offsets</span>
uv += vec2(3.0, 0.0) * params.SourceSize.zw;
<span class="tk-c">// animate with time</span>
float w = sin(params.Time * 6.2832);
<span class="tk-c">// vignette</span>
float d = distance(uv, vec2(0.5));
c *= smoothstep(0.85, 0.4, d);
<span class="tk-c">// blend original ↔ effect</span>
FragColor = vec4(mix(base, c, amount), 1.0);</code></pre>

<h4>Notes</h4>
<p><code>texture()</code> in branches may hit WGSL uniformity rules — the
engine auto-retries with <code>textureLod</code>. Custom clips are
single-pass; multi-pass chains come from bundled <code>.slangp</code>
presets. Save shaders to reuse them from the ＋ menu.</p>
`;
