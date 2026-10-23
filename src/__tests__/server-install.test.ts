jest.mock('electron', () => ({
  app: { getVersion: () => '0.0.0', getPath: () => '' },
  ipcMain: { handle: jest.fn() },
  net: { fetch: jest.fn() },
  BrowserWindow: { getAllWindows: () => [] },
}));
jest.mock('../main/errorLog', () => ({ logEvent: jest.fn() }));

import { serverAssetName } from '../main/server-install';

// These names are what release.yml publishes; a rename there must land here.
describe('serverAssetName', () => {
  it('maps each platform to its release artifact', () => {
    expect(serverAssetName('win32', 'x64')).toBe(
      'lowkey-media-server-setup-windows-amd64.exe'
    );
    expect(serverAssetName('win32', 'arm64')).toBe(
      'lowkey-media-server-setup-windows-amd64.exe'
    );
    expect(serverAssetName('darwin', 'arm64')).toBe(
      'lowkey-media-server-darwin-arm64.dmg'
    );
    expect(serverAssetName('darwin', 'x64')).toBe(
      'lowkey-media-server-darwin-amd64.dmg'
    );
    expect(serverAssetName('linux', 'x64')).toBe(
      'lowkey-media-server-linux-amd64.tar.gz'
    );
  });

  it('has no artifact for unpublished targets', () => {
    expect(serverAssetName('linux', 'arm64')).toBeNull();
    expect(serverAssetName('freebsd', 'x64')).toBeNull();
  });
});
