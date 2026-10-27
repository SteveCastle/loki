import { isNetworkPath, _resetCacheForTesting } from '../main/network-drive-cache';
import child_process from 'child_process';

jest.mock('child_process');
const mockExecSync = child_process.execSync as jest.MockedFunction<typeof child_process.execSync>;

// Mock os.platform
jest.mock('os', () => ({ platform: () => 'win32' }));

beforeEach(() => {
  _resetCacheForTesting();
  mockExecSync.mockReset();
});

const WMIC_OUTPUT = `DeviceID  DriveType  \r\nC:        3          \r\nD:        3          \r\nZ:        4          \r\n`;

describe('isNetworkPath', () => {
  it('detects UNC paths as network', () => {
    expect(isNetworkPath('\\\\server\\share\\file.jpg')).toBe(true);
  });

  it('detects forward-slash UNC paths as network', () => {
    expect(isNetworkPath('//server/share/file.jpg')).toBe(true);
  });

  it('detects mapped network drive via wmic', () => {
    mockExecSync.mockReturnValue(Buffer.from(WMIC_OUTPUT));
    expect(isNetworkPath('Z:\\photos\\cat.jpg')).toBe(true);
  });

  it('detects local drive via wmic', () => {
    mockExecSync.mockReturnValue(Buffer.from(WMIC_OUTPUT));
    expect(isNetworkPath('C:\\Users\\me\\photo.jpg')).toBe(false);
  });

  it('caches wmic results across calls', () => {
    mockExecSync.mockReturnValue(Buffer.from(WMIC_OUTPUT));
    isNetworkPath('C:\\file1.jpg');
    isNetworkPath('C:\\file2.jpg');
    isNetworkPath('Z:\\file3.jpg');
    expect(mockExecSync).toHaveBeenCalledTimes(1);
  });

  it('re-queries wmic for unknown drive letters', () => {
    mockExecSync.mockReturnValue(Buffer.from(WMIC_OUTPUT));
    const now = jest.spyOn(Date, 'now').mockReturnValue(1_000_000);
    isNetworkPath('C:\\file.jpg'); // populates cache
    now.mockReturnValue(1_000_000 + 31_000);

    const UPDATED_OUTPUT = `DeviceID  DriveType  \r\nC:        3          \r\nD:        3          \r\nX:        4          \r\nZ:        4          \r\n`;
    mockExecSync.mockReturnValue(Buffer.from(UPDATED_OUTPUT));
    expect(isNetworkPath('X:\\new-nas\\file.jpg')).toBe(true);
    expect(mockExecSync).toHaveBeenCalledTimes(2);
    now.mockRestore();
  });

  it('treats all paths as network when wmic fails', () => {
    mockExecSync.mockImplementation(() => { throw new Error('wmic not found'); });
    expect(isNetworkPath('C:\\local\\file.jpg')).toBe(true);
  });

  const NET_USE_OUTPUT =
    'New connections will be remembered.\r\n\r\n' +
    'Status       Local     Remote                    Network\r\n' +
    '-------------------------------------------------------------------------------\r\n' +
    'OK           X:        \\\\192.168.50.42\\Balthasar Microsoft Windows Network\r\n' +
    'Unavailable  Z:        \\\\nas\\share          Microsoft Windows Network\r\n' +
    'The command completed successfully.\r\n';

  it('falls back to net use when wmic is missing (Windows 11 24H2+)', () => {
    mockExecSync.mockImplementation(((cmd: string) => {
      if (cmd.startsWith('wmic')) throw new Error('wmic not found');
      return Buffer.from(NET_USE_OUTPUT);
    }) as any);
    expect(isNetworkPath('C:\\local\\file.jpg')).toBe(false);
    expect(isNetworkPath('X:\\nas\\file.jpg')).toBe(true);
    expect(isNetworkPath('z:\\nas\\file.jpg')).toBe(true);
    expect(isNetworkPath('Q:\\later\\file.jpg')).toBe(false);
    // wmic + net use once; every letter is known after that
    expect(mockExecSync).toHaveBeenCalledTimes(2);
  });

  it('does not re-spawn on every call for an unknown letter', () => {
    mockExecSync.mockReturnValue(Buffer.from(WMIC_OUTPUT));
    for (let i = 0; i < 50; i++) isNetworkPath('Q:\\file.jpg');
    expect(mockExecSync).toHaveBeenCalledTimes(1);
  });

  it('handles lowercase drive letters', () => {
    mockExecSync.mockReturnValue(Buffer.from(WMIC_OUTPUT));
    expect(isNetworkPath('z:\\photos\\cat.jpg')).toBe(true);
    expect(isNetworkPath('c:\\Users\\me\\photo.jpg')).toBe(false);
  });
});
