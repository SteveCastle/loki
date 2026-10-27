import { shell } from 'electron';

// Window-open requests from our windows go to the user's browser — but only
// web links. Anything else (a gsm:// media URL from a dragged thumbnail, a
// file:// path) would reach the OS, which has no handler and shows a "can't
// open this link" dialog.
export function openExternalLink(url: string) {
  try {
    const { protocol } = new URL(url);
    if (protocol === 'http:' || protocol === 'https:' || protocol === 'mailto:') {
      shell.openExternal(url);
    }
  } catch {
    // Not a URL: nothing to open.
  }
}
