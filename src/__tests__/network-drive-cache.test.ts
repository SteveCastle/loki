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
    isNetworkPath('C:\\file.jpg'); // populates cache

    const UPDATED_OUTPUT = `DeviceID  DriveType  \r\nC:        3          \r\nD:        3          \r\nX:        4          \r\nZ:        4          \r\n`;
    mockExecSync.mockReturnValue(Buffer.from(UPDATED_OUTPUT));
    expect(isNetworkPath('X:\\new-nas\\file.jpg')).toBe(true);
    expect(mockExecSync).toHaveBeenCalledTimes(2);
  });

  it('treats all paths as network when wmic fails', () => {
    mockExecSync.mockImplementation(() => { throw new Error('wmic not found'); });
    expect(isNetworkPath('C:\\local\\file.jpg')).toBe(true);
  });

  it('handles lowercase drive letters', () => {
    mockExecSync.mockReturnValue(Buffer.from(WMIC_OUTPUT));
    expect(isNetworkPath('z:\\photos\\cat.jpg')).toBe(true);
    expect(isNetworkPath('c:\\Users\\me\\photo.jpg')).toBe(false);
  });
});
