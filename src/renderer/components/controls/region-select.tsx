import React, { useCallback, useContext, useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { rectFromDrag, runRegionSearch, Rect } from '../../region-search';
import { capabilities } from '../../platform';

const MIN_SIZE = 8;

export default function RegionSelect() {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);
  const [altHeld, setAltHeld] = useState(false);
  const [box, setBox] = useState<Rect | null>(null);
  // Shift during the drag switches the capture from whole-image similarity
  // ('clip') to face-identity search ('face'); the box recolors as a cue.
  const [faceMode, setFaceMode] = useState(false);
  const startRef = useRef<{ x: number; y: number } | null>(null);

  // Track Alt key globally.
  useEffect(() => {
    if (!capabilities.regionCapture) return; // desktop only
    const reset = () => {
      setAltHeld(false);
      startRef.current = null;
      setBox(null);
    };
    const down = (e: KeyboardEvent) => {
      if (e.key === 'Alt') setAltHeld(true);
    };
    const up = (e: KeyboardEvent) => {
      if (e.key === 'Alt') reset();
    };
    const esc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        startRef.current = null;
        setBox(null);
      }
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    window.addEventListener('keydown', esc);
    // Alt+Tab (or any focus loss) swallows the Alt keyup, which would leave
    // altHeld stuck true and the full-screen overlay blocking the whole UI.
    // Reset on blur so the overlay can never get wedged.
    window.addEventListener('blur', reset);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
      window.removeEventListener('keydown', esc);
      window.removeEventListener('blur', reset);
    };
  }, []);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!altHeld) return;
      e.preventDefault();
      startRef.current = { x: e.clientX, y: e.clientY };
      setBox({ x: e.clientX, y: e.clientY, width: 0, height: 0 });
    },
    [altHeld]
  );

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    // Self-heal: the keyup/blur listeners can miss the Alt release (e.g. the
    // OS or menu bar swallows it), which would leave this full-screen overlay
    // wedged over the UI. The mouse event carries the live modifier state, so
    // any movement without Alt actually held dismisses the overlay.
    if (!e.altKey && !startRef.current) {
      setAltHeld(false);
      setBox(null);
      return;
    }
    if (!startRef.current) return;
    setFaceMode(e.shiftKey);
    setBox(rectFromDrag(startRef.current, { x: e.clientX, y: e.clientY }));
  }, []);

  const onMouseUp = useCallback(
    async (e: React.MouseEvent) => {
      const start = startRef.current;
      startRef.current = null;
      if (!start) return;
      const mode = e.shiftKey ? 'face' : 'clip';
      const rect = rectFromDrag(start, { x: e.clientX, y: e.clientY });
      // Hide the box SYNCHRONOUSLY before capture so capturePage grabs the media
      // under it, not the outline. flushSync forces the DOM commit before the
      // requestAnimationFrame below (React 18 concurrent mode does not otherwise
      // guarantee the setState flushes before the rAF callback fires).
      flushSync(() => setBox(null));
      if (rect.width < MIN_SIZE || rect.height < MIN_SIZE) return;
      // Wait two frames: the box removal paints on the next frame, and we
      // capture on the one after — so the captured composite never contains the
      // outline even on a slow compositor.
      await new Promise((r) =>
        requestAnimationFrame(() => requestAnimationFrame(() => r(null)))
      );
      try {
        await runRegionSearch(
          rect,
          authToken,
          (ev) => libraryService.send(ev),
          mode
        );
      } catch (err: any) {
        libraryService.send({
          type: 'ADD_TOAST',
          data: { type: 'error', title: 'Region search', message: 'Region search failed: ' + (err?.message ?? 'unknown error') },
        });
      }
    },
    [authToken, libraryService]
  );

  if (!capabilities.regionCapture || !altHeld) return null;

  return (
    <div
      className="region-select-overlay"
      style={{ position: 'fixed', inset: 0, zIndex: 10000, cursor: 'crosshair' }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
    >
      {box && (
        <div
          className="region-select-box"
          style={{
            position: 'fixed',
            left: box.x,
            top: box.y,
            width: box.width,
            height: box.height,
            border: faceMode ? '2px solid #fa4' : '2px solid #4af',
            background: faceMode
              ? 'rgba(255,170,68,0.12)'
              : 'rgba(68,170,255,0.12)',
            pointerEvents: 'none',
          }}
        />
      )}
    </div>
  );
}
