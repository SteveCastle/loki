// Beat analysis worker: runs beat-grid.js's analyzeTrack (resample, FFT,
// tempo search over a whole song) off the main thread. See
// analyzeTrackOffThread in app.js.
import { analyzeTrack } from './beat-grid.js';

self.onmessage = ({ data: { samples, sr, opts } }) => {
  try {
    self.postMessage({ result: analyzeTrack(samples, sr, opts) });
  } catch (e) {
    self.postMessage({ error: e?.message ?? String(e) });
  }
};
