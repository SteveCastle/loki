import { render, fireEvent, screen } from '@testing-library/react';

// The right-hand panel remembers the tab the user picked, across closing the
// panel and restarting the app. A tab that doesn't apply right now (Jobs while
// the job-server check is still pending, Transcript on an image, anything
// before the first item loads) falls back to the first available tab for
// display only — it must never overwrite the remembered choice, or the choice
// is lost on every mount.

const mockStore: Record<string, any> = {};
const mockContext: any = {};
let mockJobServer: boolean | null = null;

const IMAGE = 'C:/media/a.jpg';
const VIDEO = 'C:/media/b.mp4';

function setItem(path: string | null) {
  Object.assign(mockContext, {
    library: path ? [{ path }] : [],
    libraryLoadId: `load-${Math.random()}`,
    cursor: 0,
    textFilter: '',
    settings: { filters: 'all', sortBy: 'name', followTranscript: false },
    authToken: null,
  });
}

jest.mock('@xstate/react', () => ({
  useSelector: (_service: any, selector: any) =>
    selector({ context: mockContext }),
}));

jest.mock('../renderer/state', () => {
  const React = require('react');
  return {
    __esModule: true,
    GlobalStateContext: React.createContext({ libraryService: { send: jest.fn() } }),
  };
});

jest.mock('../renderer/platform', () => ({
  __esModule: true,
  store: {
    get: (key: string, def: any) => (key in mockStore ? mockStore[key] : def),
    set: (key: string, value: any) => {
      mockStore[key] = value;
    },
  },
}));

jest.mock('../renderer/hooks/useJobServerAvailable', () => ({
  __esModule: true,
  default: () => mockJobServer,
}));

jest.mock('../renderer/components/metadata/file-metadata', () => ({ __esModule: true, default: function Stub() { return <div>file-info-body</div>; } }));
jest.mock('../renderer/components/metadata/duplicates', () => ({ __esModule: true, default: function Stub() { return <div>duplicates-body</div>; } }));
jest.mock('../renderer/components/metadata/thumbnails', () => ({ __esModule: true, default: function Stub() { return <div>thumbnails-body</div>; } }));
jest.mock('../renderer/components/metadata/transcript', () => ({ __esModule: true, default: function Stub() { return <div>transcript-body</div>; } }));
jest.mock('../renderer/components/metadata/transformations', () => ({ __esModule: true, default: function Stub() { return <div>transformations-body</div>; } }));
jest.mock('../renderer/components/metadata/job-queue', () => ({ __esModule: true, default: function Stub() { return <div>jobs-body</div>; } }));
jest.mock('../renderer/components/metadata/metadata.css', () => ({}));

import Metadata from '../renderer/components/metadata/metadata';

beforeEach(() => {
  Object.keys(mockStore).forEach((k) => delete mockStore[k]);
  mockJobServer = null;
  setItem(VIDEO);
});

test('picking a tab persists it and a remount restores it', () => {
  const { unmount } = render(<Metadata />);
  fireEvent.click(screen.getByText('Duplicates'));
  expect(mockStore['metadata.activeTabKey']).toBe('duplicates');
  unmount();

  render(<Metadata />);
  expect(screen.getByText('duplicates-body')).toBeTruthy();
});

test('Jobs survives a remount while the job-server check is still pending', () => {
  mockJobServer = true;
  const { unmount } = render(<Metadata />);
  fireEvent.click(screen.getByText('Jobs'));
  unmount();

  mockJobServer = null; // fresh mount / app start: availability not known yet
  const { rerender } = render(<Metadata />);
  expect(screen.getByText('file-info-body')).toBeTruthy();
  expect(mockStore['metadata.activeTabKey']).toBe('jobs');

  mockJobServer = true;
  rerender(<Metadata />);
  expect(screen.getByText('jobs-body')).toBeTruthy();
});

test('a tab that does not fit the item comes back on the next item it fits', () => {
  const { rerender } = render(<Metadata />);
  fireEvent.click(screen.getByText('Transcript'));

  setItem(IMAGE);
  rerender(<Metadata />);
  expect(screen.queryByText('Transcript')).toBeNull();
  expect(mockStore['metadata.activeTabKey']).toBe('transcript');

  setItem(VIDEO);
  rerender(<Metadata />);
  expect(screen.getByText('transcript-body')).toBeTruthy();
});

test('no item loaded yet does not reset the choice', () => {
  mockStore['metadata.activeTabKey'] = 'thumbnails';
  setItem(null);
  const { rerender } = render(<Metadata />);
  expect(mockStore['metadata.activeTabKey']).toBe('thumbnails');

  setItem(IMAGE);
  rerender(<Metadata />);
  expect(screen.getByText('thumbnails-body')).toBeTruthy();
});
