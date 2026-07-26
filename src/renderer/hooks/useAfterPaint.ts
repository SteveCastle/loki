// useAfterPaint — flips to true in a task that runs AFTER the browser has
// painted the commit this hook mounted in.
//
// Used to keep expensive-but-not-urgent subtrees out of the commit that makes a
// surface visible. The command palette is the app's most-opened UI, so its
// shell must paint on the first frame; the search engine (tag index lookups,
// the categories fetch, the suggestion sections) and its tooltips mount on the
// following task instead.
//
// requestAnimationFrame alone is NOT enough — it fires *before* paint, so work
// scheduled there still lands in the same frame. We chain a zero-delay timeout
// inside it to land in the task after the frame. The standalone timeout is a
// safety net: in an occluded/background window rAF never fires at all, and the
// deferred subtree must still mount.
import { useEffect, useState } from 'react';

export function useAfterPaint(fallbackMs = 50): boolean {
  const [afterPaint, setAfterPaint] = useState(false);

  useEffect(() => {
    let inner = 0;
    const done = () => setAfterPaint(true);
    const raf = requestAnimationFrame(() => {
      inner = window.setTimeout(done, 0);
    });
    const fallback = window.setTimeout(done, fallbackMs);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(inner);
      clearTimeout(fallback);
    };
  }, [fallbackMs]);

  return afterPaint;
}

export default useAfterPaint;
