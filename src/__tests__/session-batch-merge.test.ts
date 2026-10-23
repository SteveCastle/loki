const setMany = jest.fn(async (_u: any) => undefined);
const set = jest.fn(async (_k: any, _v: any) => undefined);
jest.mock('../renderer/platform', () => ({
  sessionStore: {
    getAll: async () => null,
    set: (k: any, v: any) => set(k, v),
    setMany: (u: any) => setMany(u),
    flush: async () => undefined,
    clear: async () => undefined,
    clearKeys: async () => undefined,
  },
}));

import { setSessionValues, setSessionValue } from '../renderer/hooks/useSessionStore';

beforeEach(() => {
  jest.useFakeTimers();
  setMany.mockClear();
  set.mockClear();
});
afterEach(() => jest.useRealTimers());

const lib = { library: [{ path: '/a.jpg', mtimeMs: 1 }], initialFile: '/a.jpg' } as any;
const prev = { previousLibrary: [], previousCursor: 0 } as any;

test('a second batch within the debounce keeps the first batch', () => {
  setSessionValues({ library: lib });
  setSessionValues({ previous: prev });
  jest.runAllTimers();
  expect(setMany).toHaveBeenCalledTimes(1);
  expect(setMany.mock.calls[0][0]).toEqual({ library: lib, previous: prev });
});

test('a later single-key write is not overwritten by a stale batched copy', () => {
  setSessionValues({ cursor: { cursor: 1 } } as any);
  setSessionValue('cursor', { cursor: 9 } as any);
  jest.runAllTimers();
  const batched = setMany.mock.calls.map((c) => c[0]);
  expect(batched.every((b) => !('cursor' in b))).toBe(true);
  expect(set).toHaveBeenLastCalledWith('cursor', { cursor: 9 });
});
