jest.mock('electron', () => ({
  app: { getVersion: () => '0.0.0', getPath: () => '' },
  ipcMain: { handle: jest.fn() },
  net: { fetch: jest.fn() },
  BrowserWindow: { getAllWindows: () => [] },
}));
jest.mock('../main/errorLog', () => ({ logEvent: jest.fn() }));

import {
  autoStartServer,
  isOlderVersion,
  serverAssetName,
} from '../main/server-install';

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

describe('isOlderVersion', () => {
  it('compares dotted release versions numerically', () => {
    expect(isOlderVersion('2.35.0', '2.36.0')).toBe(true);
    expect(isOlderVersion('2.9.0', '2.10.0')).toBe(true);
    expect(isOlderVersion('v2.35.1', '2.36.0')).toBe(true);
    expect(isOlderVersion('2.36.0', '2.36.0')).toBe(false);
    expect(isOlderVersion('2.37.0', '2.36.0')).toBe(false);
    expect(isOlderVersion('2.36', '2.36.0')).toBe(false);
  });

  it('never offers an upgrade over a dev build or an unknown version', () => {
    expect(isOlderVersion('dev', '2.36.0')).toBe(false);
    expect(isOlderVersion(null, '2.36.0')).toBe(false);
    expect(isOlderVersion('2.35.0', '0.0.0-beta')).toBe(false);
  });
});

describe('autoStartServer', () => {
  const fs = jest.requireActual('fs') as typeof import('fs');
  const os = jest.requireActual('os') as typeof import('os');
  const path = jest.requireActual('path') as typeof import('path');
  const childProcess = jest.requireActual('child_process');
  const { net } = jest.requireMock('electron');

  const storeWith = (values: Record<string, unknown>) =>
    ({
      get: (k: string, d?: unknown) => (k in values ? values[k] : d),
      set: jest.fn(),
    } as any);

  let tmp: string;
  let exe: string;
  let spawnSpy: jest.SpyInstance;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lms-autostart-'));
    exe = path.join(tmp, 'lowkeymediaserver');
    fs.writeFileSync(exe, '');
    spawnSpy = jest
      .spyOn(childProcess, 'spawn')
      .mockReturnValue({ unref: jest.fn() } as any);
    net.fetch.mockReset();
  });

  afterEach(() => {
    spawnSpy.mockRestore();
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it('starts an installed, stopped server in the background by default', async () => {
    net.fetch.mockRejectedValue(new Error('ECONNREFUSED'));
    await autoStartServer(storeWith({ serverInstallPath: exe }));
    expect(spawnSpy).toHaveBeenCalledTimes(1);
    expect(spawnSpy.mock.calls[0][1]).toContain('--background');
  });

  it('leaves a running server alone', async () => {
    net.fetch.mockResolvedValue({ json: async () => ({ version: '1.0.0' }) });
    await autoStartServer(storeWith({ serverInstallPath: exe }));
    expect(spawnSpy).not.toHaveBeenCalled();
  });

  it('respects the setting being turned off', async () => {
    net.fetch.mockRejectedValue(new Error('ECONNREFUSED'));
    await autoStartServer(
      storeWith({ serverInstallPath: exe, autoStartServer: false })
    );
    expect(spawnSpy).not.toHaveBeenCalled();
  });
});
