import {
  getDirFromInitialFile,
  buildLibraryPathQuery,
} from '../renderer/components/controls/context-query';

describe('getDirFromInitialFile', () => {
  it('returns the parent directory of a file path', () => {
    expect(getDirFromInitialFile('/photos/cats/img.jpg')).toBe('/photos/cats');
    expect(getDirFromInitialFile('D:\\photos\\cats\\img.jpg')).toBe(
      'D:\\photos\\cats'
    );
  });

  it('returns a directory path unchanged', () => {
    expect(getDirFromInitialFile('/photos/cats')).toBe('/photos/cats');
  });

  it('keeps the trailing separator on a Windows drive root', () => {
    expect(getDirFromInitialFile('D:\\img.jpg')).toBe('D:\\');
  });
});

describe('buildLibraryPathQuery', () => {
  // Non-recursive: pathdir matches only the immediate directory (the server
  // expands it to LIKE 'dir/%' AND NOT LIKE 'dir/%/%').
  it('uses pathdir for the immediate directory when recursive is off', () => {
    expect(buildLibraryPathQuery('/photos/cats/img.jpg', false)).toBe(
      'pathdir:"/photos/cats"'
    );
    expect(buildLibraryPathQuery('D:\\photos\\cats\\img.jpg', false)).toBe(
      'pathdir:"D:\\photos\\cats"'
    );
  });

  // Recursive: the list view spans subdirectories, so match every path under
  // the directory with a trailing wildcard (server: path:"dir/*" → LIKE 'dir/%').
  it('uses a trailing-wildcard path query when recursive is on', () => {
    expect(buildLibraryPathQuery('/photos/cats/img.jpg', true)).toBe(
      'path:"/photos/cats/*"'
    );
    expect(buildLibraryPathQuery('D:\\photos\\cats\\img.jpg', true)).toBe(
      'path:"D:\\photos\\cats\\*"'
    );
  });

  it('handles a directory target (no file extension)', () => {
    expect(buildLibraryPathQuery('/photos/cats', false)).toBe(
      'pathdir:"/photos/cats"'
    );
    expect(buildLibraryPathQuery('/photos/cats', true)).toBe(
      'path:"/photos/cats/*"'
    );
  });

  it('handles a Windows drive root without doubling the separator', () => {
    expect(buildLibraryPathQuery('D:\\img.jpg', true)).toBe('path:"D:\\*"');
  });
});
