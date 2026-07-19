/*
 * slangfx studio — timeline panel.
 *
 * After-Effects-style editing surface: a zoomable time ruler (down to
 * individual frames), stacked tracks holding media / fx clips, and
 * twirl-down property lanes with draggable keyframe diamonds.
 *
 * The timeline owns presentation + interaction state (zoom, scroll,
 * selection, expansion) and mutates the comp model directly, but defers
 * app-level policy — property lists, value setting semantics, playback —
 * to a `host` interface provided by app.js:
 *
 *   comp()                        the comp model
 *   time() / setTime(t)          playhead (comp seconds)
 *   playing()
 *   history                       comp.js History
 *   assetOf(id)                   media asset lookup ({duration} or null)
 *   propList(clip)                [{key,label,min,max,step,unit}] animatable props
 *   getProp(clip,key)             PropTrack | null (not yet created)
 *   valueAt(clip,key)             evaluated value at the playhead
 *   setPropValue(clip,key,v)      set honoring animation state
 *   toggleAnim(clip,key)          stopwatch
 *   toggleKey(clip,key)           add/remove key at playhead
 *   onModelChange({structural})   model mutated (app re-renders + saves)
 *   onSelect()                    selection changed
 *   addMediaAt(files,t,trackIdx)  drag-dropped files
 *   status(msg)
 */

import {
  clipEnd, splitClip, uid, ensureDur, removeEmptyTracks,
  quantize, clamp, trackOf, findClip, EASING_LABELS, sortKeys, upsertKey,
} from './comp.js';

const TRACK_H = 36;
const PROP_H = 24;
const RULER_H = 28;
const HEAD_W = 252;
const SNAP_PX = 11;
const MIN_CLIP_FRAMES = 1;
const END_PAD = 90;   // px of grab-space kept past the comp end

const fmtNum = (v) => {
  const n = +v;
  if (!Number.isFinite(n)) return '0';
  return Math.abs(n) >= 100 ? n.toFixed(1).replace(/\.0$/, '') : (+n.toFixed(3)).toString();
};

export function fmtTimecode(t, fps) {
  if (!Number.isFinite(t)) return '-:--:--';
  const totalFrames = Math.round(t * fps);
  const f = totalFrames % fps;
  const s = Math.floor(totalFrames / fps);
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${mm}:${String(ss).padStart(2, '0')}:${String(f).padStart(2, '0')}`;
}

function parseTimecode(text, fps) {
  const parts = text.trim().split(':').map((p) => parseFloat(p));
  if (parts.some((p) => Number.isNaN(p))) return null;
  if (parts.length === 3) return parts[0] * 60 + parts[1] + parts[2] / fps;
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  if (parts.length === 1) return parts[0];
  return null;
}

/* One shared floating context menu (also used by the viewport gizmo). */
let menuEl = null;
function closeMenu() { menuEl?.remove(); menuEl = null; }
export function showMenu(x, y, items) {
  closeMenu();
  menuEl = document.createElement('div');
  menuEl.className = 'ctx-menu';
  for (const it of items) {
    if (it === '-') {
      const hr = document.createElement('div');
      hr.className = 'ctx-sep';
      menuEl.appendChild(hr);
      continue;
    }
    const row = document.createElement('div');
    row.className = 'ctx-item' + (it.checked ? ' checked' : '') + (it.danger ? ' danger' : '');
    row.addEventListener('pointerdown', (e) => e.stopPropagation());
    row.addEventListener('click', () => { closeMenu(); it.action(); });
    if (it.trailing) {
      // Secondary action button at the row's right edge (e.g. delete).
      row.classList.add('has-trail');
      const lab = document.createElement('span');
      lab.className = 'ctx-label';
      lab.textContent = it.label;
      const btn = document.createElement('button');
      btn.className = 'ctx-trail' + (it.trailing.danger ? ' danger' : '');
      btn.textContent = it.trailing.label;
      if (it.trailing.title) btn.title = it.trailing.title;
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        closeMenu();
        it.trailing.action();
      });
      row.append(lab, btn);
    } else {
      row.textContent = it.label;
    }
    menuEl.appendChild(row);
  }
  document.body.appendChild(menuEl);
  const r = menuEl.getBoundingClientRect();
  menuEl.style.left = `${Math.min(x, innerWidth - r.width - 6)}px`;
  menuEl.style.top = `${Math.min(y, innerHeight - r.height - 6)}px`;
}
document.addEventListener('pointerdown', (e) => {
  if (menuEl && !menuEl.contains(e.target)) closeMenu();
});

export class Timeline {
  constructor(root, host) {
    this.host = host;
    this.pps = 60;                 // pixels per second
    this.snap = true;
    this.follow = true;            // auto-scroll during playback
    this.selClips = new Set();     // clip ids
    this.selKeys = new Set();      // key objects
    this.keyOwners = new Map();    // key object -> {clip, prop}
    this.expanded = new Set();     // clip ids with property lanes shown
    this.animInputs = [];          // [{clip, key, input}] refreshed per frame
    this._buildDom(root);
    this._bindGlobal();
  }

  /* ---- DOM scaffold ------------------------------------------------- */

  _buildDom(root) {
    root.classList.add('tl');
    root.innerHTML = `
      <div class="tl-toolbar">
        <button class="btn tl-tb" data-act="home" title="to start (Home)">⏮</button>
        <button class="btn tl-tb" data-act="prev" title="previous frame (←)">◀</button>
        <button class="btn tl-tb tl-play" data-act="play" title="play / pause (Space)">▶</button>
        <button class="btn tl-tb" data-act="next" title="next frame (→)">▶</button>
        <button class="btn tl-tb" data-act="end" title="to end (End)">⏭</button>
        <input class="tl-time" spellcheck="false" title="playhead — m:ss:ff, editable">
        <span class="tl-sep"></span>
        <button class="btn tl-tb" data-act="split" title="split clip(s) at playhead (S)">✂</button>
        <button class="btn tl-tb tl-toggle" data-act="snap" title="snapping (frames, edges, playhead)">🧲</button>
        <button class="btn tl-tb tl-toggle" data-act="loop" title="loop playback">🔁</button>
        <span class="tl-flex"></span>
        <span class="tl-zoom-label" title="timeline zoom"></span>
        <button class="btn tl-tb" data-act="zoom-out" title="zoom out (−)">−</button>
        <input type="range" class="tl-zoom" min="0" max="1" step="0.001" title="zoom — Ctrl+wheel on the timeline also zooms">
        <button class="btn tl-tb" data-act="zoom-in" title="zoom in (+)">＋</button>
        <button class="btn tl-tb" data-act="fit" title="fit whole comp">Fit</button>
      </div>
      <div class="tl-main">
        <div class="tl-heads">
          <div class="tl-corner"></div>
          <div class="tl-head-rows"></div>
        </div>
        <div class="tl-scroll">
          <div class="tl-content">
            <canvas class="tl-ruler"></canvas>
            <div class="tl-rows"></div>
            <div class="tl-playhead"><div class="tl-ph-cap"></div></div>
          </div>
        </div>
      </div>`;
    this.root = root;
    this.$ = (sel) => root.querySelector(sel);
    this.toolbar = this.$('.tl-toolbar');
    this.playBtn = this.$('.tl-play');
    this.nextBtn = root.querySelectorAll('[data-act=next]')[0];
    this.timeInput = this.$('.tl-time');
    this.zoomSlider = this.$('.tl-zoom');
    this.zoomLabel = this.$('.tl-zoom-label');
    this.headsEl = this.$('.tl-heads');
    this.headRows = this.$('.tl-head-rows');
    this.scrollEl = this.$('.tl-scroll');
    this.contentEl = this.$('.tl-content');
    this.ruler = this.$('.tl-ruler');
    this.rowsEl = this.$('.tl-rows');
    this.playheadEl = this.$('.tl-playhead');

    this.toolbar.addEventListener('click', (e) => {
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act) this._toolbarAction(act);
    });
    this.zoomSlider.addEventListener('input', () => {
      const [lo, hi] = this._zoomRange();
      const pps = lo * (hi / lo) ** parseFloat(this.zoomSlider.value);
      this._setZoom(pps, this.host.time());
    });
    this.timeInput.addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter') {
        const t = parseTimecode(this.timeInput.value, this.host.comp().fps);
        if (t != null) this.host.setTime(t);
        this.timeInput.blur();
      } else if (e.key === 'Escape') this.timeInput.blur();
    });

    this.scrollEl.addEventListener('scroll', () => {
      this.headRows.style.transform = `translateY(${-this.scrollEl.scrollTop}px)`;
      this._drawRuler();
    });
    new ResizeObserver(() => { this._drawRuler(); this._syncZoomSlider(); }).observe(this.scrollEl);

    // Ctrl+wheel zoom anchored under the cursor.
    this.scrollEl.addEventListener('wheel', (e) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      const t = this._timeAtClientX(e.clientX);
      this._setZoom(this.pps * (e.deltaY < 0 ? 1.25 : 0.8), t, e.clientX);
    }, { passive: false });

    // Ruler scrubbing.
    this.ruler.addEventListener('pointerdown', (e) => {
      this.ruler.setPointerCapture(e.pointerId);
      this._scrubbing = true;
      this.host.setTime(this._timeAtClientX(e.clientX));
    });
    this.ruler.addEventListener('pointermove', (e) => {
      if (this._scrubbing) this.host.setTime(this._timeAtClientX(e.clientX));
    });
    this.ruler.addEventListener('pointerup', () => { this._scrubbing = false; });

    // Drag-drop media straight onto the timeline at the drop position.
    this.scrollEl.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); });
    this.scrollEl.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!e.dataTransfer.files.length) return;
      const t = Math.max(0, this._timeAtClientX(e.clientX));
      this.host.addMediaAt([...e.dataTransfer.files], this._snapTime(t), this._trackIndexAtClientY(e.clientY));
    });

    // Click empty space clears selection.
    this.rowsEl.addEventListener('pointerdown', (e) => {
      if (e.target === this.rowsEl || e.target.classList.contains('tl-row'))
        this._select(null, e.shiftKey);
    });
  }

  _toolbarAction(act) {
    const h = this.host;
    const frame = 1 / h.comp().fps;
    switch (act) {
      case 'home': h.setTime(0); break;
      case 'end': h.setTime(h.comp().dur); break;
      case 'prev': h.setTime(h.time() - frame); break;
      case 'next': h.setTime(h.time() + frame); break;
      case 'play': h.togglePlay(); break;
      case 'split': this.splitAtPlayhead(); break;
      case 'snap': this.snap = !this.snap; this._syncToggles(); break;
      case 'loop': h.toggleLoop(); this._syncToggles(); break;
      case 'zoom-in': this._setZoom(this.pps * 1.5, h.time()); break;
      case 'zoom-out': this._setZoom(this.pps / 1.5, h.time()); break;
      case 'fit': this.zoomFit(); break;
    }
  }

  _syncToggles() {
    this.root.querySelector('[data-act=snap]').classList.toggle('active', this.snap);
    this.root.querySelector('[data-act=loop]').classList.toggle('active', this.host.looping());
  }

  _bindGlobal() {
    document.addEventListener('keydown', (e) => {
      if (e.target.matches?.('input, textarea, select, [contenteditable]')) return;
      const h = this.host;
      const frame = 1 / h.comp().fps;
      const step = e.shiftKey ? 10 * frame : frame;
      if (e.code === 'Space') { e.preventDefault(); h.togglePlay(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); h.setTime(h.time() - step); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); h.setTime(h.time() + step); }
      else if (e.key === 'Home') { e.preventDefault(); h.setTime(0); }
      else if (e.key === 'End') { e.preventDefault(); h.setTime(h.comp().dur); }
      else if (e.key === 's' || e.key === 'S') this.splitAtPlayhead();
      else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); this.deleteSelection(); }
      else if ((e.key === '=' || e.key === '+') && !e.ctrlKey) this._setZoom(this.pps * 1.5, h.time());
      else if (e.key === '-' && !e.ctrlKey) this._setZoom(this.pps / 1.5, h.time());
      else if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === 'z') { e.preventDefault(); h.undo(); }
      else if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'y' || (e.shiftKey && e.key.toLowerCase() === 'z'))) { e.preventDefault(); h.redo(); }
    });
  }

  /* ---- coordinates --------------------------------------------------- */

  _timeToX(t) { return t * this.pps; }
  _xToTime(x) { return x / this.pps; }

  _timeAtClientX(clientX) {
    const rect = this.scrollEl.getBoundingClientRect();
    return this._xToTime(clientX - rect.left + this.scrollEl.scrollLeft);
  }

  _trackIndexAtClientY(clientY) {
    const rows = [...this.rowsEl.querySelectorAll('.tl-row.track')];
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i].getBoundingClientRect();
      if (clientY < r.bottom) return i;
    }
    return rows.length ? rows.length - 1 : null;
  }

  /** Nearest magnet target (clip edges / playhead / comp bounds) within
   * snapping reach of `t`, or null. */
  _magnetTarget(t, { excludeClip = null, extraTargets = [] } = {}) {
    if (!this.snap) return null;
    const comp = this.host.comp();
    const targets = [0, comp.dur, this.host.time(), ...extraTargets];
    for (const track of comp.tracks)
      for (const c of track.clips) {
        if (c === excludeClip) continue;
        targets.push(c.start, clipEnd(c));
      }
    const thresh = SNAP_PX / this.pps;
    let best = null;
    let bestD = Infinity;
    for (const target of targets) {
      const d = Math.abs(t - target);
      if (d < thresh && d < bestD) { best = target; bestD = d; }
    }
    return best;
  }

  /** Magnet targets beat the frame grid; the grid is the fallback. */
  _snapTime(t, opts = {}) {
    const m = this._magnetTarget(t, opts);
    if (m != null) return Math.max(0, m);
    return Math.max(0, quantize(t, this.host.comp().fps));
  }

  /* ---- zoom ----------------------------------------------------------- */

  _zoomRange() {
    // Fit leaves END_PAD after the comp end so edge handles stay grabbable.
    const w = Math.max(this.scrollEl.clientWidth - END_PAD - 8, 100);
    const comp = this.host.comp();
    const lo = Math.max(1e-3, w / Math.max(comp.dur, 0.5));
    const hi = Math.max(lo * 1.001, comp.fps * 80);  // ≥ 80 px per frame
    return [lo, hi];
  }

  _setZoom(pps, anchorTime = null, anchorClientX = null) {
    const [lo, hi] = this._zoomRange();
    pps = clamp(pps, lo, hi);
    if (pps === this.pps) return;
    let anchorPx;
    if (anchorTime != null) {
      anchorPx = anchorClientX != null
        ? anchorClientX - this.scrollEl.getBoundingClientRect().left
        : clamp(this._timeToX(anchorTime) - this.scrollEl.scrollLeft, 0, this.scrollEl.clientWidth);
    }
    this.pps = pps;
    this.render();
    if (anchorTime != null)
      this.scrollEl.scrollLeft = Math.max(0, this._timeToX(anchorTime) - anchorPx);
    this._drawRuler();
  }

  _syncZoomSlider() {
    const [lo, hi] = this._zoomRange();
    this.pps = clamp(this.pps, lo, hi);
    this.zoomSlider.value = String(clamp(Math.log(this.pps / lo) / Math.log(hi / lo), 0, 1));
    const ppf = this.pps / this.host.comp().fps;
    this.zoomLabel.textContent = ppf >= 4 ? `${ppf.toFixed(0)} px/frame` : `${this.pps.toFixed(0)} px/s`;
  }

  zoomFit() {
    const [lo] = this._zoomRange();
    this.pps = lo;
    this.render();
    this.scrollEl.scrollLeft = 0;
    this._drawRuler();
  }

  /* ---- ruler ----------------------------------------------------------- */

  _drawRuler() {
    const canvas = this.ruler;
    const w = this.scrollEl.clientWidth;
    const dpr = devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(RULER_H * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(RULER_H * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${RULER_H}px`;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const comp = this.host.comp();
    const fps = comp.fps;
    const x0 = this.scrollEl.scrollLeft;
    const styles = getComputedStyle(this.root);
    ctx.fillStyle = styles.getPropertyValue('--tl-ruler-bg') || '#14161b';
    ctx.fillRect(0, 0, w, RULER_H);

    // Pick a major step giving labels ≥ 64 px apart; sub-second steps are
    // whole frame counts so frame boundaries stay honest.
    const steps = [1 / fps, 2 / fps, 5 / fps, 10 / fps, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    let step = steps[steps.length - 1];
    for (const s of steps) if (s * this.pps >= 64) { step = s; break; }
    const minor = step * this.pps >= 320 ? step / 10 : step / (step >= 1 ? 5 : 2);
    const frameW = this.pps / fps;

    // Frame grid when zoomed close.
    if (frameW >= 5) {
      ctx.fillStyle = 'rgba(255,255,255,0.06)';
      const first = Math.floor(x0 / frameW);
      for (let f = first; f * frameW < x0 + w; f++) {
        const x = f * frameW - x0;
        ctx.fillRect(x, RULER_H - 5, 1, 5);
      }
    }

    ctx.fillStyle = 'rgba(255,255,255,0.22)';
    const firstMinor = Math.floor(x0 / (minor * this.pps));
    for (let i = firstMinor; i * minor * this.pps < x0 + w; i++) {
      const x = i * minor * this.pps - x0;
      ctx.fillRect(x, RULER_H - 9, 1, 9);
    }

    ctx.font = '10px "Segoe UI", system-ui, sans-serif';
    ctx.textBaseline = 'top';
    const firstMajor = Math.floor(x0 / (step * this.pps));
    for (let i = firstMajor; i * step * this.pps < x0 + w; i++) {
      const t = i * step;
      const x = t * this.pps - x0;
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.fillRect(x, RULER_H - 13, 1, 13);
      ctx.fillStyle = t > comp.dur + 1e-9 ? 'rgba(200,205,215,0.3)' : 'rgba(200,205,215,0.85)';
      const label = step < 1 ? fmtTimecode(t, fps) : fmtTimecode(t, fps).replace(/:\d\d$/, '');
      ctx.fillText(label, x + 4, 3);
    }

    // Out-of-comp shade.
    const endX = comp.dur * this.pps - x0;
    if (endX < w) {
      ctx.fillStyle = 'rgba(0,0,0,0.35)';
      ctx.fillRect(Math.max(endX, 0), 0, w, RULER_H);
    }
  }

  /* ---- rendering -------------------------------------------------------- */

  _layout() {
    const comp = this.host.comp();
    this.contentEl.style.width = `${this._timeToX(comp.dur) + END_PAD}px`;
  }

  /** Full rebuild of tracks, clips, and property lanes. */
  render() {
    const comp = this.host.comp();
    this._layout();
    this._syncZoomSlider();
    this._syncToggles();
    this.headRows.replaceChildren();
    this.rowsEl.replaceChildren();
    this.animInputs = [];
    this.keyOwners = new Map();

    // Drop stale selection/expansion.
    const liveIds = new Set();
    for (const tr of comp.tracks) for (const c of tr.clips) liveIds.add(c.id);
    for (const id of [...this.selClips]) if (!liveIds.has(id)) this.selClips.delete(id);
    for (const id of [...this.expanded]) if (!liveIds.has(id)) this.expanded.delete(id);

    comp.tracks.forEach((track, trackIdx) => {
      this._renderTrackRow(track, trackIdx);
      const expandedClips = track.clips
        .filter((c) => this.expanded.has(c.id))
        .sort((a, b) => a.start - b.start);
      for (const clip of expandedClips)
        for (const propDef of this.host.propList(clip))
          this._renderPropRow(clip, propDef);
    });

    // Prune selKeys that no longer exist.
    for (const k of [...this.selKeys]) if (!this.keyOwners.has(k)) this.selKeys.delete(k);

    this._drawRuler();
    this.updatePlayhead(true);
  }

  _mkHeadRow(cls, h) {
    const el = document.createElement('div');
    el.className = `tl-head-row ${cls}`;
    el.style.height = `${h}px`;
    this.headRows.appendChild(el);
    return el;
  }

  _mkRow(cls, h) {
    const el = document.createElement('div');
    el.className = `tl-row ${cls}`;
    el.style.height = `${h}px`;
    this.rowsEl.appendChild(el);
    return el;
  }

  /* ---- track row ---- */

  _renderTrackRow(track, trackIdx) {
    const comp = this.host.comp();
    const head = this._mkHeadRow('track', TRACK_H);

    const flagBtn = (cls, glyphOn, glyphOff, flag, title) => {
      const b = document.createElement('button');
      b.className = `tl-mini tl-flag ${cls}` + (track[flag] ? ' off' : '');
      b.textContent = track[flag] ? glyphOff : glyphOn;
      b.title = title;
      b.addEventListener('click', () => {
        this.host.history.record(comp, () => { track[flag] = !track[flag]; });
        this.host.onModelChange({ structural: true });
      });
      return b;
    };
    const eye = flagBtn('tl-eye', '👁', '👁', 'hidden',
      'show / hide this track (hidden tracks render nothing; fx are bypassed)');
    const spk = flagBtn('tl-spk', '🔊', '🔇', 'muted',
      'mute / unmute this track’s audio');

    const grip = document.createElement('span');
    grip.className = 'tl-track-grip';
    grip.textContent = '≡';
    grip.title = 'drag to reorder — clips here composite over lower tracks';

    const name = document.createElement('span');
    name.className = 'tl-track-name';
    name.textContent = track.name;
    name.title = 'double-click to rename';
    name.addEventListener('dblclick', () => {
      const input = document.createElement('input');
      input.className = 'tl-rename';
      input.value = track.name;
      const commit = () => {
        this.host.history.record(comp, () => { track.name = input.value.trim() || track.name; });
        this.host.onModelChange({ structural: false });
      };
      input.addEventListener('keydown', (e) => {
        e.stopPropagation();
        if (e.key === 'Enter') input.blur();
        if (e.key === 'Escape') { input.value = track.name; input.blur(); }
      });
      input.addEventListener('blur', commit);
      name.replaceChildren(input);
      input.focus();
      input.select();
    });

    const del = this._miniBtn('✕', 'delete track (and its clips)', false, () => {
      this.host.history.record(comp, () => {
        comp.tracks.splice(trackIdx, 1);
        removeEmptyTracks(comp);
      });
      this.host.onModelChange({ structural: true });
    });
    head.append(eye, spk, grip, name, del);
    head.addEventListener('pointerdown', (e) => this._trackDragStart(e, trackIdx, head));

    const row = this._mkRow('track', TRACK_H);
    row.dataset.trackIdx = String(trackIdx);
    if (track.hidden) row.classList.add('hidden-track');
    for (const clip of track.clips) this._renderClip(row, track, clip, trackIdx);
  }

  _miniBtn(text, title, disabled, onClick) {
    const b = document.createElement('button');
    b.className = 'tl-mini';
    b.textContent = text;
    b.title = title;
    b.disabled = disabled;
    b.addEventListener('click', onClick);
    return b;
  }

  /* ---- track drag-reorder ---- */

  /** Head rows and body rows are appended in lockstep (one of each per
   * track, then per expanded prop lane), so the timeline groups into
   * per-track blocks by walking both lists and splitting on `.track`. */
  _trackBlocks() {
    const heads = [...this.headRows.children];
    const rows = [...this.rowsEl.children];
    const blocks = [];
    heads.forEach((headEl, i) => {
      if (headEl.classList.contains('track')) blocks.push({ els: [], h: 0 });
      const b = blocks[blocks.length - 1];
      b.els.push(headEl, rows[i]);
      b.h += headEl.offsetHeight;
    });
    for (const b of blocks) b.top = b.els[0].getBoundingClientRect().top;
    return blocks;
  }

  /** Drag a track head vertically to reorder tracks. Other blocks shift
   * out of the way live; the drop commits one splice to the model. */
  _trackDragStart(e, trackIdx, head) {
    if (e.button !== 0) return;
    if (e.target.closest('button, input')) return;
    const comp = this.host.comp();
    if (comp.tracks.length < 2) return;
    const y0 = e.clientY;
    const pointerId = e.pointerId;
    let blocks = null;         // built once the drag passes the threshold
    let target = trackIdx;

    const move = (ev) => {
      if (ev.pointerId !== pointerId) return;
      if (!blocks) {
        if (Math.abs(ev.clientY - y0) < 5) return;
        blocks = this._trackBlocks();
        this.root.classList.add('tl-reordering');
        for (const el of blocks[trackIdx].els) el.classList.add('tl-drag-src');
        try { head.setPointerCapture(pointerId); } catch {}
      }
      const src = blocks[trackIdx];
      const last = blocks[blocks.length - 1];
      const rawDy = ev.clientY - y0;
      const dy = clamp(rawDy,
        blocks[0].top - src.top,
        last.top + last.h - (src.top + src.h));
      // Judge the drop target from the unclamped intent — a hard fling
      // past either end must still land the track there, not tie on the
      // clamped midpoint.
      const center = src.top + rawDy + src.h / 2;
      target = 0;
      blocks.forEach((b, i) => {
        if (i === trackIdx) {
          for (const el of b.els) el.style.transform = `translateY(${dy}px)`;
          return;
        }
        const mid = b.top + b.h / 2;
        if (mid < center) target++;
        const shift = i > trackIdx
          ? (mid < center ? -src.h : 0)
          : (mid > center ? src.h : 0);
        for (const el of b.els) el.style.transform = shift ? `translateY(${shift}px)` : '';
      });
    };

    const finish = (commit) => {
      removeEventListener('pointermove', move);
      removeEventListener('pointerup', up);
      removeEventListener('pointercancel', cancel);
      if (!blocks) return;
      for (const b of blocks) {
        for (const el of b.els) {
          el.style.transform = '';
          el.classList.remove('tl-drag-src');
        }
      }
      this.root.classList.remove('tl-reordering');
      if (commit && target !== trackIdx) {
        this.host.history.record(comp, () => {
          const [tr] = comp.tracks.splice(trackIdx, 1);
          comp.tracks.splice(target, 0, tr);
        });
        this.host.onModelChange({ structural: true });
      }
    };
    const up = (ev) => { if (ev.pointerId === pointerId) finish(true); };
    const cancel = (ev) => { if (ev.pointerId === pointerId) finish(false); };
    addEventListener('pointermove', move);
    addEventListener('pointerup', up);
    addEventListener('pointercancel', cancel);
  }

  /* ---- clip ---- */

  _renderClip(row, track, clip, trackIdx) {
    const comp = this.host.comp();
    const el = document.createElement('div');
    el.className = `tl-clip ${clip.kind}` +
      (clip.kind === 'fx' && clip.fxKind === 'custom' ? ' custom' : '') +
      (this.selClips.has(clip.id) ? ' sel' : '') +
      (clip.kind === 'fx' && !clip.enabled ? ' off' : '');
    el.style.left = `${this._timeToX(clip.start)}px`;
    el.style.width = `${Math.max(this._timeToX(clip.dur), 4)}px`;
    el.dataset.clipId = clip.id;

    const twirl = document.createElement('span');
    twirl.className = 'tl-twirl';
    twirl.textContent = this.expanded.has(clip.id) ? '▾' : '▸';
    twirl.title = 'show animatable properties';
    twirl.addEventListener('pointerdown', (e) => e.stopPropagation());
    twirl.addEventListener('click', (e) => {
      e.stopPropagation();
      if (this.expanded.has(clip.id)) this.expanded.delete(clip.id);
      else this.expanded.add(clip.id);
      this.render();
    });

    const label = document.createElement('span');
    label.className = 'tl-clip-name';
    const badge = clip.kind === 'fx' ? (clip.fxKind === 'custom' ? '✎ ' : 'ƒx ') : '';
    label.textContent = badge + clip.name;

    const kfCount = this._keyframeCount(clip);
    if (kfCount) {
      const dot = document.createElement('span');
      dot.className = 'tl-clip-kf';
      dot.textContent = `◆${kfCount}`;
      dot.title = `${kfCount} keyframe${kfCount > 1 ? 's' : ''}`;
      el.append(twirl, label, dot);
    } else {
      el.append(twirl, label);
    }

    const hl = document.createElement('div');
    hl.className = 'tl-handle l';
    const hr = document.createElement('div');
    hr.className = 'tl-handle r';
    el.append(hl, hr);

    el.addEventListener('pointerdown', (e) => this._clipPointerDown(e, el, track, clip, trackIdx));
    el.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this._select(clip.id, false);
      this._clipMenu(e.clientX, e.clientY, clip);
    });
    el.addEventListener('dblclick', () => {
      if (this.expanded.has(clip.id)) this.expanded.delete(clip.id);
      else this.expanded.add(clip.id);
      this.render();
    });
    row.appendChild(el);
  }

  _keyframeCount(clip) {
    const bag = clip.kind === 'media' ? clip.props : clip.params;
    let n = 0;
    for (const p of Object.values(bag ?? {})) if (p.anim) n += p.keys.length;
    return n;
  }

  _clipMenu(x, y, clip) {
    const comp = this.host.comp();
    const items = [
      { label: 'Split at playhead', action: () => this.splitAtPlayhead() },
      {
        label: 'Duplicate',
        action: () => {
          this.host.history.record(comp, () => {
            const copy = structuredClone(clip);
            copy.id = uid('clip');
            copy.start = clipEnd(clip);
            trackOf(comp, clip)?.clips.push(copy);
            ensureDur(comp);
          });
          this.host.onModelChange({ structural: true });
        },
      },
      ...(clip.kind === 'media' ? [{
        label: 'Fit in frame',
        action: () => {
          const asset = this.host.assetOf(clip.assetId);
          const values = { x: comp.width / 2, y: comp.height / 2 };
          if (asset?.w) {
            const fit = Math.round(Math.min(comp.width / asset.w, comp.height / asset.h) * 10000) / 100;
            values.scaleX = fit;
            values.scaleY = fit;
          }
          this.host.history.record(comp, () => {
            for (const [k, v] of Object.entries(values)) {
              const prop = clip.props[k];
              if (prop.anim) {
                const t = clamp(quantize(this.host.time() - clip.start, comp.fps), 0, clip.dur);
                upsertKey(prop, t, v);
              } else {
                prop.v = v;
              }
            }
          });
          this.host.onModelChange({ structural: false });
          this.host.status(`${clip.name} fit to ${comp.width}×${comp.height}`);
        },
      }] : []),
      {
        label: 'Trim comp length to clip',
        action: () => {
          this.host.history.record(comp, () => {
            comp.dur = Math.max(1 / comp.fps, quantize(clipEnd(clip), comp.fps));
            ensureDur(comp);   // other clips may still reach further
          });
          this.host.setTime(Math.min(this.host.time(), comp.dur));
          this.host.onModelChange({ structural: false });
          this.zoomFit();
          this.host.status(`comp trimmed to ${fmtTimecode(comp.dur, comp.fps)}`);
        },
      },
      {
        label: 'Stretch clip to comp length',
        action: () => {
          this.host.history.record(comp, () => {
            clip.start = 0;
            clip.dur = Math.max(1 / comp.fps, comp.dur);
          });
          this.host.onModelChange({ structural: true });
          this.host.status(`${clip.name} spans the whole timeline` +
            (clip.kind === 'media' ? ' (video loops past its source length)' : ''));
        },
      },
    ];
    if (clip.kind === 'fx') {
      items.push({
        label: clip.enabled ? 'Disable (bypass)' : 'Enable',
        action: () => {
          this.host.history.record(comp, () => { clip.enabled = !clip.enabled; });
          this.host.onModelChange({ structural: true });
        },
      });
    }
    items.push('-', {
      label: 'Delete',
      danger: true,
      action: () => { this.selClips = new Set([clip.id]); this.deleteSelection(); },
    });
    showMenu(x, y, items);
  }

  _select(clipId, additive) {
    if (!clipId) {
      if (!additive) { this.selClips.clear(); this.selKeys.clear(); }
    } else if (additive) {
      if (this.selClips.has(clipId)) this.selClips.delete(clipId);
      else this.selClips.add(clipId);
    } else {
      this.selClips = new Set([clipId]);
      this.selKeys.clear();
    }
    this._refreshSelectionStyles();
    this.host.onSelect();
  }

  _refreshSelectionStyles() {
    for (const el of this.rowsEl.querySelectorAll('.tl-clip'))
      el.classList.toggle('sel', this.selClips.has(el.dataset.clipId));
    for (const el of this.rowsEl.querySelectorAll('.tl-kf'))
      el.classList.toggle('sel', this.selKeys.has(el._key));
  }

  get selectedClip() {
    const comp = this.host.comp();
    for (const id of [...this.selClips].reverse()) {
      const hit = findClip(comp, id);
      if (hit) return hit.clip;
    }
    return null;
  }

  selectClip(clipId) {
    this.selClips = clipId ? new Set([clipId]) : new Set();
    this.selKeys.clear();
    this._refreshSelectionStyles();
    this.host.onSelect();
  }

  /* Clip drag: move (body) or trim (edge handles). Live-updates the model
   * for immediate visual + preview feedback; history captures the state at
   * pointerdown and commits once at pointerup. */
  _clipPointerDown(e, el, track, clip, trackIdx) {
    if (e.button !== 0) return;
    e.preventDefault();
    const comp = this.host.comp();
    const wasSelected = this.selClips.has(clip.id);
    if (!wasSelected) this._select(clip.id, e.shiftKey);

    const mode = e.target.classList.contains('tl-handle')
      ? (e.target.classList.contains('l') ? 'trim-l' : 'trim-r')
      : 'move';
    const startX = e.clientX;
    const startY = e.clientY;
    const orig = { start: clip.start, dur: clip.dur, in: clip.in ?? 0, trackIdx };
    const minDur = MIN_CLIP_FRAMES / comp.fps;
    let moved = false;
    this.host.history.begin(comp);

    const onMove = (ev) => {
      const dxT = (ev.clientX - startX) / this.pps;
      if (!moved && Math.abs(ev.clientX - startX) < 3 && Math.abs(ev.clientY - startY) < 6) return;
      moved = true;

      if (mode === 'move') {
        // Either edge of the dragged clip can catch a magnet; a real magnet
        // hit on one edge must beat the other edge's frame-grid fallback.
        const raw = orig.start + dxT;
        const mStart = this._magnetTarget(raw, { excludeClip: clip });
        const mEnd = this._magnetTarget(raw + orig.dur, { excludeClip: clip });
        let cand;
        if (mStart != null || mEnd != null) {
          const dS = mStart != null ? Math.abs(mStart - raw) : Infinity;
          const dE = mEnd != null ? Math.abs(mEnd - orig.dur - raw) : Infinity;
          cand = dS <= dE ? mStart : mEnd - orig.dur;
        } else {
          cand = quantize(raw, comp.fps);
        }
        // The comp end is a hard wall — clips can't be dragged past it.
        clip.start = clamp(cand, 0, Math.max(0, comp.dur - clip.dur));
        // Vertical: retarget track.
        const targetIdx = this._trackIndexAtClientY(ev.clientY);
        const curTrack = trackOf(comp, clip);
        const curIdx = comp.tracks.indexOf(curTrack);
        if (targetIdx != null && targetIdx !== curIdx) {
          curTrack.clips.splice(curTrack.clips.indexOf(clip), 1);
          comp.tracks[targetIdx].clips.push(clip);
        }
      } else if (mode === 'trim-l') {
        let ns = this._snapTime(orig.start + dxT, { excludeClip: clip });
        ns = clamp(ns, orig.start - orig.in, orig.start + orig.dur - minDur);
        const d = ns - orig.start;
        clip.start = ns;
        clip.dur = orig.dur - d;
        if (clip.kind === 'media') clip.in = orig.in + d;
        // Keys stay put in comp time: shift clip-relative times by -d.
        this._shiftKeys(clip, -d, orig);
      } else {
        // Right trim stops at the comp end; videos longer than their
        // source loop, so no source-length clamp.
        const ne = this._snapTime(orig.start + orig.dur + dxT, { excludeClip: clip });
        clip.dur = clamp(ne - orig.start, minDur, Math.max(minDur, comp.dur - orig.start));
      }
      // While trimming, preview the frame at the cut point so you can see
      // exactly where the clip will start / end when released.
      if (mode === 'trim-l') {
        this.host.setTrimPreview?.(clip.start);
      } else if (mode === 'trim-r') {
        this.host.setTrimPreview?.(Math.max(clip.start, clipEnd(clip) - 1 / comp.fps));
      }
      ensureDur(comp);
      this.host.onModelChange({ structural: false, transient: true });
      this.render();
    };

    const onUp = (ev) => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      this.host.setTrimPreview?.(null);
      if (moved) {
        this.host.history.commit(comp);
        this.host.onModelChange({ structural: true });
        this.render();
      } else {
        this.host.history.pending = null;
        if (wasSelected && !ev.shiftKey) this._select(clip.id, false);
      }
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }

  /* On left trim the drag applies a running delta from `orig`; recompute key
   * times from their captured originals so repeated moves don't accumulate. */
  _shiftKeys(clip, delta, orig) {
    if (!orig._keySnapshot) {
      orig._keySnapshot = [];
      const bag = clip.kind === 'media' ? clip.props : clip.params;
      for (const p of Object.values(bag ?? {}))
        orig._keySnapshot.push({ p, times: p.keys.map((k) => k.t) });
    }
    for (const snap of orig._keySnapshot)
      snap.p.keys.forEach((k, i) => { k.t = snap.times[i] + delta; });
  }

  /* ---- property rows ---- */

  _renderPropRow(clip, def) {
    const comp = this.host.comp();
    const head = this._mkHeadRow('prop', PROP_H);
    const prop = this.host.getProp(clip, def.key);
    const anim = !!prop?.anim;

    const stopwatch = document.createElement('button');
    stopwatch.className = 'tl-stopwatch' + (anim ? ' on' : '');
    stopwatch.textContent = '⏱';
    stopwatch.title = anim
      ? 'animation ON — click to freeze at the current value (deletes keyframes)'
      : 'enable animation — sets a keyframe at the playhead';
    stopwatch.addEventListener('click', () => this.host.toggleAnim(clip, def.key));

    const label = document.createElement('span');
    label.className = 'tl-prop-name';
    label.textContent = def.label;
    label.title = `${clip.name} · ${def.key}${def.unit ? ` (${def.unit})` : ''}`;

    const val = document.createElement('input');
    val.className = 'tl-prop-val';
    val.type = 'number';
    if (def.step) val.step = String(def.step);
    val.value = fmtNum(this.host.valueAt(clip, def.key));
    val.title = anim ? 'value at playhead — typing adds/updates a keyframe' : 'value';
    val.addEventListener('keydown', (e) => e.stopPropagation());
    val.addEventListener('change', () => {
      const v = parseFloat(val.value);
      if (!Number.isNaN(v)) this.host.setPropValue(clip, def.key, v);
    });
    if (anim) this.animInputs.push({ clip, key: def.key, input: val });

    const nav = document.createElement('span');
    nav.className = 'tl-kf-nav';
    const tRel = () => this.host.time() - clip.start;
    const prev = this._miniBtn('◀', 'previous keyframe', false, () => {
      const keys = (this.host.getProp(clip, def.key)?.keys ?? []).filter((k) => k.t < tRel() - 1e-4);
      if (keys.length) this.host.setTime(clip.start + keys[keys.length - 1].t);
    });
    const diamond = this._miniBtn('◆', 'add / remove keyframe at playhead', false, () =>
      this.host.toggleKey(clip, def.key));
    diamond.classList.add('tl-kf-toggle');
    const next = this._miniBtn('▶', 'next keyframe', false, () => {
      const keys = (this.host.getProp(clip, def.key)?.keys ?? []).filter((k) => k.t > tRel() + 1e-4);
      if (keys.length) this.host.setTime(clip.start + keys[0].t);
    });
    nav.append(prev, diamond, next);
    head.append(stopwatch, label, val, nav);

    const row = this._mkRow('prop', PROP_H);

    // Lane shading over the clip's active range.
    const lane = document.createElement('div');
    lane.className = 'tl-prop-lane';
    lane.style.left = `${this._timeToX(clip.start)}px`;
    lane.style.width = `${Math.max(this._timeToX(clip.dur), 4)}px`;
    row.appendChild(lane);

    // Alt-click (or double-click) on the row adds a key at that time.
    row.addEventListener('pointerdown', (e) => {
      if (!e.altKey || e.target !== row && e.target !== lane) return;
      const t = this._snapTime(this._timeAtClientX(e.clientX));
      this.host.setTime(t);
      this.host.toggleKey(clip, def.key);
    });
    row.addEventListener('dblclick', (e) => {
      if (e.target !== row && e.target !== lane) return;
      const t = this._snapTime(this._timeAtClientX(e.clientX));
      this.host.setTime(t);
      this.host.toggleKey(clip, def.key);
    });

    // Connecting segments + diamonds.
    if (prop?.anim && prop.keys.length) {
      for (let i = 0; i < prop.keys.length - 1; i++) {
        const a = prop.keys[i], b = prop.keys[i + 1];
        const seg = document.createElement('div');
        seg.className = 'tl-kf-seg' + (a.e === 'hold' ? ' hold' : '');
        seg.style.left = `${this._timeToX(clip.start + a.t)}px`;
        seg.style.width = `${Math.max(this._timeToX(b.t - a.t), 0)}px`;
        row.appendChild(seg);
      }
      for (const key of prop.keys) this._renderKey(row, clip, prop, key, def);
    }
  }

  _renderKey(row, clip, prop, key, def) {
    const el = document.createElement('div');
    el.className = 'tl-kf' + (this.selKeys.has(key) ? ' sel' : '') + (key.e === 'hold' ? ' hold' : '');
    el.style.left = `${this._timeToX(clip.start + key.t)}px`;
    el.title = `${def.label} = ${fmtNum(key.v)} @ ${fmtTimecode(clip.start + key.t, this.host.comp().fps)} · ${key.e}`;
    el._key = key;
    this.keyOwners.set(key, { clip, prop });

    el.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();
      if (e.shiftKey) {
        if (this.selKeys.has(key)) this.selKeys.delete(key);
        else this.selKeys.add(key);
      } else if (!this.selKeys.has(key)) {
        this.selKeys = new Set([key]);
      }
      this._refreshSelectionStyles();

      const comp = this.host.comp();
      const startX = e.clientX;
      const origTimes = [...this.selKeys].map((k) => ({ k, t: k.t }));
      let moved = false;
      this.host.history.begin(comp);
      const onMove = (ev) => {
        const dt = (ev.clientX - startX) / this.pps;
        if (!moved && Math.abs(ev.clientX - startX) < 3) return;
        moved = true;
        for (const { k, t } of origTimes) {
          const owner = this.keyOwners.get(k) ?? { clip };
          const abs = this._snapTime(owner.clip.start + t + dt);
          k.t = clamp(abs - owner.clip.start, 0, owner.clip.dur);
        }
        for (const { k } of origTimes) {
          const owner = this.keyOwners.get(k);
          if (owner) sortKeys(owner.prop);
        }
        this.host.onModelChange({ structural: false, transient: true });
        this.render();
      };
      const onUp = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
        if (moved) {
          this.host.history.commit(comp);
          this.host.onModelChange({ structural: false });
          this.render();
        } else {
          this.host.history.pending = null;
        }
      };
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    });

    el.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!this.selKeys.has(key)) { this.selKeys = new Set([key]); this._refreshSelectionStyles(); }
      const comp = this.host.comp();
      // Easing lives on the key that STARTS a segment; applying a choice to
      // both segments touching the selected key matches what people mean by
      // "make this keyframe linear/bouncy" (and makes picking either end of
      // a two-key tween equivalent).
      const segmentsOf = (k) => {
        const owner = this.keyOwners.get(k);
        if (!owner) return [];
        const keys = owner.prop.keys;
        const i = keys.indexOf(k);
        const segs = [];
        if (i >= 0 && i < keys.length - 1) segs.push(k);      // outgoing
        if (i > 0) segs.push(keys[i - 1]);                     // incoming
        return segs;
      };
      const current = new Set(segmentsOf(key).map((s) => s.e));
      const items = EASING_LABELS.map(([id, name]) => ({
        label: name,
        checked: current.size === 1 && current.has(id),
        action: () => {
          this.host.history.record(comp, () => {
            for (const k of this.selKeys) for (const s of segmentsOf(k)) s.e = id;
          });
          this.host.onModelChange({ structural: false });
        },
      }));
      items.push('-', {
        label: 'Delete keyframe' + (this.selKeys.size > 1 ? 's' : ''),
        danger: true,
        action: () => this.deleteSelection(),
      });
      showMenu(e.clientX, e.clientY, items);
    });

    el.addEventListener('dblclick', (e) => {
      e.stopPropagation();
      this.host.setTime(clip.start + key.t);
    });

    row.appendChild(el);
  }

  /* ---- operations -------------------------------------------------------- */

  splitAtPlayhead() {
    const comp = this.host.comp();
    const t = quantize(this.host.time(), comp.fps);
    const targets = this.selClips.size
      ? [...this.selClips].map((id) => findClip(comp, id)).filter(Boolean)
      : comp.tracks.flatMap((track) =>
          track.clips.filter((c) => t > c.start && t < clipEnd(c)).map((clip) => ({ track, clip })));
    let n = 0;
    this.host.history.record(comp, () => {
      for (const { track, clip } of targets) {
        const right = splitClip(clip, t);
        if (right) { track.clips.push(right); n++; }
      }
    });
    if (n) {
      this.host.onModelChange({ structural: true });
      this.host.status(`split ${n} clip${n > 1 ? 's' : ''} at ${fmtTimecode(t, comp.fps)}`);
    }
  }

  deleteSelection() {
    const comp = this.host.comp();
    if (this.selKeys.size) {
      this.host.history.record(comp, () => {
        for (const key of this.selKeys) {
          const owner = this.keyOwners.get(key);
          if (!owner) continue;
          const i = owner.prop.keys.indexOf(key);
          if (i >= 0) owner.prop.keys.splice(i, 1);
          if (owner.prop.keys.length === 0) owner.prop.anim = false;
        }
      });
      this.selKeys.clear();
      this.host.onModelChange({ structural: false });
      return;
    }
    if (this.selClips.size) {
      this.host.history.record(comp, () => {
        for (const track of comp.tracks)
          track.clips = track.clips.filter((c) => !this.selClips.has(c.id));
        removeEmptyTracks(comp);
      });
      this.selClips.clear();
      this.host.onModelChange({ structural: true });
      this.host.onSelect();
    }
  }

  /* ---- per-frame ---------------------------------------------------------- */

  updatePlayhead(force = false) {
    const comp = this.host.comp();
    const t = this.host.time();
    const x = this._timeToX(t);
    this.playheadEl.style.transform = `translateX(${x}px)`;

    if (document.activeElement !== this.timeInput)
      this.timeInput.value = fmtTimecode(t, comp.fps);
    this.playBtn.textContent = this.host.playing() ? '⏸' : '▶';

    // Follow the playhead while playing.
    if (this.host.playing() && this.follow) {
      const viewL = this.scrollEl.scrollLeft;
      const viewW = this.scrollEl.clientWidth;
      if (x < viewL || x > viewL + viewW - 40)
        this.scrollEl.scrollLeft = Math.max(0, x - viewW * 0.15);
    }

    // Live value readouts for animated properties.
    for (const { clip, key, input } of this.animInputs)
      if (document.activeElement !== input)
        input.value = fmtNum(this.host.valueAt(clip, key));

    if (force) this._drawRuler();
  }
}
