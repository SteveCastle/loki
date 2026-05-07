/**
 * Converts a sidecar subtitle file's text into a WebVTT string suitable
 * for use with a `<track>` element.
 *
 * Why: HTML5 `<track>` only accepts WebVTT. SRT, the most common sidecar
 * format, differs in two trivial ways — no header, and timestamps use
 * `,` rather than `.` as the millisecond separator. Translating in the
 * renderer avoids a main-process round trip per video.
 */

export type SubtitleExt = 'srt' | 'vtt';

const TIMESTAMP_LINE = /^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}/;

export function srtToVtt(srt: string): string {
  const converted = srt
    .split('\n')
    .map((line) =>
      TIMESTAMP_LINE.test(line) ? line.replace(/,(\d{3})/g, '.$1') : line
    )
    .join('\n');
  return `WEBVTT\n\n${converted}`;
}

export function toVttString(content: string, ext: SubtitleExt): string {
  if (ext === 'srt') return srtToVtt(content);
  // VTT: ensure header is present.
  if (!/^WEBVTT/.test(content)) return `WEBVTT\n\n${content}`;
  return content;
}

/**
 * Builds a `blob:` URL for a VTT string. Returned URL must be revoked
 * via `URL.revokeObjectURL` once the consumer is done with it.
 */
export function vttBlobUrl(vtt: string): string {
  const blob = new Blob([vtt], { type: 'text/vtt' });
  return URL.createObjectURL(blob);
}
