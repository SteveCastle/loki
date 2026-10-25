// Build-time feature switches. webpack's EnvironmentPlugin bakes each
// `process.env.X` below into the bundle (defaults in
// .erb/configs/webpack.config.base.ts), so a disabled feature's UI is absent
// from both the Electron renderer and the SPA the media server embeds.

/**
 * loki-dream (Kromoss, a MOSS-Image fine-tune): Transform's Generate and
 * Reimagine, the Scene editor's "New scene from a prompt" and its Kromoss
 * engine. Off unless the app is built with LOKI_DREAM=1. Image and video
 * editing (loki-retouch / loki-reshoot) don't depend on it.
 */
export const DREAM_ENABLED = process.env.LOKI_DREAM === '1';
