import '@testing-library/jest-dom';

// Mock electron APIs - using 'any' to avoid type conflicts with preload.d.ts
const mockElectron = {
  ipcRenderer: {
    invoke: jest.fn(),
    on: jest.fn(() => jest.fn()),
    sendMessage: jest.fn(),
  },
  store: {
    get: jest.fn((_key: string, defaultValue: any) => defaultValue),
    set: jest.fn(),
    getMany: jest.fn(() => ({})),
  },
  loadMediaFromDB: jest.fn(),
  loadMediaByDescriptionSearch: jest.fn(),
};

// Set up window.electron mock before tests
beforeAll(() => {
  (window as any).electron = mockElectron;
  (window as any).appArgs = {
    filePath: '',
    dbPath: '/test/db.sqlite',
    allArgs: [],
    appUserData: '/test/userData',
  };
});

describe('App', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Electron IPC mocking', () => {
    it('should have mocked electron object', () => {
      expect((window as any).electron).toBeDefined();
      expect((window as any).electron.ipcRenderer).toBeDefined();
      expect((window as any).electron.store).toBeDefined();
    });

    it('should mock ipcRenderer.invoke', async () => {
      mockElectron.ipcRenderer.invoke.mockResolvedValue({ success: true });

      const result = await (window as any).electron.ipcRenderer.invoke(
        'load-db',
        ['arg']
      );

      expect(result).toEqual({ success: true });
      expect(mockElectron.ipcRenderer.invoke).toHaveBeenCalledWith('load-db', [
        'arg',
      ]);
    });

    it('should mock ipcRenderer.on and return unsubscribe function', () => {
      const unsubscribe = jest.fn();
      mockElectron.ipcRenderer.on.mockReturnValue(unsubscribe);

      const handler = jest.fn();
      const cleanup = (window as any).electron.ipcRenderer.on(
        'test-event',
        handler
      );

      expect(typeof cleanup).toBe('function');
      expect(mockElectron.ipcRenderer.on).toHaveBeenCalledWith(
        'test-event',
        handler
      );
    });

    it('should mock store.get with default value', () => {
      const value = (window as any).electron.store.get('testKey', 'defaultVal');
      expect(value).toBe('defaultVal');
    });

    it('should mock store.set', () => {
      (window as any).electron.store.set('newKey', 'newValue');
      expect(mockElectron.store.set).toHaveBeenCalledWith('newKey', 'newValue');
    });

    it('should mock store.getMany', () => {
      mockElectron.store.getMany.mockReturnValue({ key1: 'value1' });

      const result = (window as any).electron.store.getMany([
        ['key1', 'default1'],
      ]);

      expect(result).toEqual({ key1: 'value1' });
    });
  });

  describe('appArgs', () => {
    it('should have default appArgs', () => {
      expect((window as any).appArgs).toBeDefined();
      expect((window as any).appArgs.filePath).toBe('');
      expect((window as any).appArgs.dbPath).toBe('/test/db.sqlite');
    });

    it('should have allArgs array', () => {
      expect(Array.isArray((window as any).appArgs.allArgs)).toBe(true);
    });

    it('should have appUserData path', () => {
      expect((window as any).appArgs.appUserData).toBe('/test/userData');
    });
  });

  describe('Media loading mocks', () => {
    it('should mock loadMediaFromDB', async () => {
      const mockMedia = [{ path: '/test.jpg', mtimeMs: 1000 }];
      mockElectron.loadMediaFromDB.mockResolvedValue(mockMedia);

      const result = await (window as any).electron.loadMediaFromDB(
        ['tag1'],
        'AND'
      );

      expect(result).toEqual(mockMedia);
      expect(mockElectron.loadMediaFromDB).toHaveBeenCalledWith(
        ['tag1'],
        'AND'
      );
    });

    it('should mock loadMediaByDescriptionSearch', async () => {
      const mockMedia = [{ path: '/search-result.jpg', mtimeMs: 2000 }];
      mockElectron.loadMediaByDescriptionSearch.mockResolvedValue(mockMedia);

      const result = await (window as any).electron.loadMediaByDescriptionSearch(
        'search term',
        ['tag1'],
        'OR'
      );

      expect(result).toEqual(mockMedia);
      expect(mockElectron.loadMediaByDescriptionSearch).toHaveBeenCalledWith(
        'search term',
        ['tag1'],
        'OR'
      );
    });
  });
});
