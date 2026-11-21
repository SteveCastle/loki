import { mediaServerBase } from '../../platform';

// Translation targets for Text capture (mirrors appconfig.TextCaptureLanguages
// on the server, which accepts any language name). '' = the server's
// configured default; 'Original' = keep the text as written.
export const TEXT_CAPTURE_LANGUAGES = [
  'English',
  'Spanish',
  'French',
  'German',
  'Italian',
  'Portuguese',
  'Dutch',
  'Polish',
  'Russian',
  'Ukrainian',
  'Turkish',
  'Arabic',
  'Hebrew',
  'Hindi',
  'Indonesian',
  'Vietnamese',
  'Thai',
  'Chinese (Simplified)',
  'Chinese (Traditional)',
  'Japanese',
  'Korean',
];

/** Queue a capture-text job for one image; resolves to the new job's id. */
export async function createTextCaptureJob(
  path: string,
  authToken: string | null,
  language: string
): Promise<string | undefined> {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }
  const body: { input: string; fields?: { language: string } } = {
    // --overwrite: an explicit per-file request recaptures existing text.
    input: `capture-text --overwrite "${path}"`,
  };
  if (language) {
    body.fields = { language };
  }
  const response = await fetch(`${mediaServerBase}/create`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(10000),
  });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  try {
    return ((await response.json()) as { id?: string }).id;
  } catch {
    return undefined;
  }
}
