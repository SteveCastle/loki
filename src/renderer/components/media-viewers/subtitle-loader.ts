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

// Strip a leading UTF-8 BOM and normalize CRLF / lone CR to LF. Sidecar
// subtitle files in the wild are routinely saved by Windows tools with
// both: a BOM ahead of the WEBVTT/SRT content corrupts the first cue
// identifier, and CRLF confuses strict WebVTT parsers when it appears on
// the header line. Without this normalization the browser silently drops
// every cue (track.cues.length === 0) even though the file loads.
function normalize(text: string): string {
  const stripped = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  return stripped.replace(/\r\n?/g, '\n');
}

export function srtToVtt(srt: string): string {
  const converted = normalize(srt)
    .split('\n')
    .map((line) =>
      TIMESTAMP_LINE.test(line) ? line.replace(/,(\d{3})/g, '.$1') : line
    )
    .join('\n');
  return `WEBVTT\n\n${converted}`;
}

export function toVttString(content: string, ext: SubtitleExt): string {
  if (ext === 'srt') return srtToVtt(content);
  // VTT: ensure header is present and the input is normalized so a stray
  // BOM / CRLF can't break parsing of an otherwise-valid file.
  const normalized = normalize(content);
  if (!/^WEBVTT/.test(normalized)) return `WEBVTT\n\n${normalized}`;
  return normalized;
}

/**
 * Builds a `blob:` URL for a VTT string. Returned URL must be revoked
 * via `URL.revokeObjectURL` once the consumer is done with it.
 */
export function vttBlobUrl(vtt: string): string {
  const blob = new Blob([vtt], { type: 'text/vtt' });
  return URL.createObjectURL(blob);
}
