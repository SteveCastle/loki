// eslint-disable-next-line @typescript-eslint/no-var-requires
const { updateText } = require('../../.github/scripts/update-docs-version.js');

describe('docs download-link updater', () => {
  const page = `
    <p class="version">Version 2.31.0</p>
    const RELEASE_BASE = 'https://github.com/SteveCastle/loki/releases/download/v2.31.0/';
    href = RELEASE_BASE + 'Lowkey.Media.Viewer.Setup.2.31.0.exe';
    href = RELEASE_BASE + 'Lowkey.Media.Viewer-2.31.0-arm64.dmg';
    href = RELEASE_BASE + 'Lowkey.Media.Viewer-2.31.0.AppImage';
    href = RELEASE_BASE + 'lowkey-media-server-setup-windows-amd64.exe';
    <a href="https://example.com/v2.31.0/tool-1.2.3.zip">other</a>
  `;

  it('moves every version reference and nothing else', () => {
    const { text, count } = updateText(page, '2.32.0');
    expect(count).toBe(5);
    expect(text).toContain('Version 2.32.0');
    expect(text).toContain('releases/download/v2.32.0/');
    expect(text).toContain('Lowkey.Media.Viewer.Setup.2.32.0.exe');
    expect(text).toContain('Lowkey.Media.Viewer-2.32.0-arm64.dmg');
    expect(text).toContain('Lowkey.Media.Viewer-2.32.0.AppImage');
    expect(text).toContain('lowkey-media-server-setup-windows-amd64.exe'); // unversioned server assets stay
    expect(text).toContain('example.com/v2.31.0/tool-1.2.3.zip'); // unrelated versions stay
  });

  it('is idempotent', () => {
    const once = updateText(page, '2.32.0').text;
    expect(updateText(once, '2.32.0').text).toBe(once);
  });
});
