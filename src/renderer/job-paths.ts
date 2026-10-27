// The media a job touched, as paths the viewer can open: the files it was
// given (its input) and the files it wrote (output_files). Used by the job
// queue panel to make both clickable.

const DRIVE_RE = /^[A-Za-z]:[\\/]/;
const MEDIA_EXT_RE = /\.(jpe?g|jfif|png|webp|avif|gif|bmp|tiff?|heic|heif|mp4|webm|mov|mkv|avi|m4v|mp3|wav|flac|ogg|m4a|aac|opus)$/i;

function looksLikePath(s: string): boolean {
  if (DRIVE_RE.test(s)) return true;
  if (/^(\/|\\|\.\/|\.\.\/|~\/)/.test(s)) return true;
  if (/[\\/]/.test(s) && !s.includes(' ')) return true;
  return MEDIA_EXT_RE.test(s);
}

/**
 * The paths in a job's input, or [] when it is not a plain path list (a
 * --query64 query, flags, prose). Mirrors the server's own reading of an
 * input (media-server/renderer/jobinput.go asPathList): newline-separated,
 * surrounding quotes dropped — commas are valid path characters.
 */
export function jobInputPaths(input: string | undefined | null): string[] {
  const s = (input || '').trim();
  if (!s || s.includes('--')) return [];
  const paths: string[] = [];
  for (const raw of s.split(/[\r\n]+/)) {
    const p = raw.trim().replace(/^["']+|["']+$/g, '');
    if (!p) continue;
    if (!looksLikePath(p)) return [];
    paths.push(p);
  }
  return paths;
}

/** The file name of a path (either separator). */
export function baseName(p: string): string {
  return p.split(/[\\/]/).pop() || p;
}

type LibraryServiceLike = {
  send: (event: string, data: Record<string, unknown>) => void;
  state?: { context?: { library?: Array<{ path?: string } | null | undefined> } };
};

/**
 * Show a media file: move the cursor to it when it is in the current library,
 * otherwise open its folder from disk with it selected (as clicking a job
 * toast's file path does). Either way a type filter that would hide the file
 * is cleared.
 */
export function navigateToMedia(libraryService: LibraryServiceLike, path: string): void {
  const library = libraryService.state?.context?.library;
  const inLibrary = Array.isArray(library) && library.some((item) => item?.path === path);
  if (inLibrary) {
    libraryService.send('RESET_CURSOR', { currentItem: { path }, reveal: true });
  }
  else libraryService.send('SET_FILE', { path });
}
