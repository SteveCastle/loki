import { mediaServerBase } from '../../platform';

// Creates a "generate description" job for `path` on the media server.
// Shared by the GenerateDescription button and the generateDescription
// hotkey so both send the identical command. Throws on failure — callers
// decide how to surface it (toast). Job lifecycle feedback itself comes
// from the ToastSystem via the SSE stream.
export async function createDescriptionJob(
  path: string,
  authToken: string | null,
  prompt?: string
): Promise<void> {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }

  const body: { input: string; fields?: { prompt: string } } = {
    input: `metadata --type description --apply all --overwrite "${path}"`,
  };
  const trimmed = prompt?.trim();
  if (trimmed) {
    body.fields = { prompt: trimmed };
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
}
