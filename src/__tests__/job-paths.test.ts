import { baseName, jobInputPaths, navigateToMedia } from '../renderer/job-paths';

describe('jobInputPaths', () => {
  test('a newline-separated path list, quotes dropped, commas kept', () => {
    expect(jobInputPaths('"C:\\media\\a, b.png\nD:/clips/c.mp4"')).toEqual(['C:\\media\\a, b.png', 'D:/clips/c.mp4']);
    expect(jobInputPaths('/home/me/x.jpg\r\n\r\n./rel/y.png')).toEqual(['/home/me/x.jpg', './rel/y.png']);
    expect(jobInputPaths('photo.webp')).toEqual(['photo.webp']);
  });

  test('queries, flags and prose are not paths', () => {
    expect(jobInputPaths('--query64=dGFnOmNhdA==')).toEqual([]);
    expect(jobInputPaths('C:/a.png --steps 4')).toEqual([]);
    expect(jobInputPaths('make it night')).toEqual([]);
    expect(jobInputPaths('C:/a.png\nnot a path at all')).toEqual([]);
    expect(jobInputPaths('')).toEqual([]);
    expect(jobInputPaths(undefined)).toEqual([]);
  });

  test('baseName takes either separator', () => {
    expect(baseName('C:\\m\\a.png')).toBe('a.png');
    expect(baseName('/m/b.mp4')).toBe('b.mp4');
  });
});

describe('navigateToMedia', () => {
  const service = (paths: string[]) => {
    const sent: Array<[string, unknown]> = [];
    return {
      sent,
      send: (event: string, data: unknown) => sent.push([event, data]),
      state: { context: { library: paths.map((path) => ({ path })) } },
    };
  };

  test('in the library: move the cursor to it', () => {
    const s = service(['C:/a.png', 'C:/b.png']);
    navigateToMedia(s, 'C:/b.png');
    expect(s.sent).toEqual([['RESET_CURSOR', { currentItem: { path: 'C:/b.png' } }]]);
  });

  test('elsewhere: open it from disk', () => {
    const s = service(['C:/a.png']);
    navigateToMedia(s, 'D:/out/new_edit.png');
    expect(s.sent).toEqual([['SET_FILE', { path: 'D:/out/new_edit.png' }]]);
  });
});
