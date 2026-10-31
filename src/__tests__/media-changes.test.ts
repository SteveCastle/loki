import {
  mergePaths,
  normalizePath,
  parseMediaChange,
  planMediaChange,
  samePath,
} from '../renderer/media-changes';

describe('samePath', () => {
  it('ignores separator style and Windows drive / UNC case', () => {
    expect(samePath('C:\\Media\\a.png', 'c:/media/a.png')).toBe(true);
    expect(samePath('\\\\NAS\\Share\\a.png', '//nas/share/a.png')).toBe(true);
    expect(samePath('C:/media/dir/', 'C:/media/dir')).toBe(true);
  });

  it('keeps POSIX paths case-sensitive', () => {
    expect(samePath('/media/A.png', '/media/a.png')).toBe(false);
    expect(normalizePath('/media/A.png')).toBe('/media/A.png');
  });

  it('is false for missing paths', () => {
    expect(samePath(null, 'a')).toBe(false);
    expect(samePath('', '')).toBe(false);
  });
});

describe('parseMediaChange', () => {
  it('reads the stream envelope, job id and paths', () => {
    const data = JSON.stringify({
      msg: JSON.stringify({ jobId: 'j1', paths: ['C:/a.png', 'c:\\a.png', 'C:/b.png'] }),
    });
    expect(parseMediaChange('created', data)).toEqual({
      kind: 'created',
      jobId: 'j1',
      paths: ['C:/a.png', 'C:/b.png'],
    });
  });

  it('accepts a bare payload without a job id (older servers)', () => {
    expect(parseMediaChange('updated', JSON.stringify({ paths: ['/x.mp4'] }))).toEqual({
      kind: 'updated',
      jobId: undefined,
      paths: ['/x.mp4'],
    });
  });

  it('returns null for malformed or empty payloads', () => {
    expect(parseMediaChange('created', 'not json')).toBeNull();
    expect(parseMediaChange('created', JSON.stringify({ paths: [] }))).toBeNull();
    expect(parseMediaChange('created', JSON.stringify({ paths: ['', 3] }))).toBeNull();
  });
});

describe('planMediaChange', () => {
  it('announces new files and never reloads or refreshes for them', () => {
    expect(
      planMediaChange({ kind: 'created', paths: ['C:/out_1.png', 'C:/out_2.png'] }, 'C:/in.png')
    ).toEqual({ refresh: [], reloadCurrent: false, announce: ['C:/out_1.png', 'C:/out_2.png'] });
  });

  it('reloads an overwritten file that is on screen instead of announcing it', () => {
    expect(
      planMediaChange({ kind: 'updated', paths: ['C:\\in.png'] }, 'c:/in.png')
    ).toEqual({ refresh: ['C:\\in.png'], reloadCurrent: true, announce: [] });
  });

  it('announces overwritten files that are not on screen', () => {
    expect(
      planMediaChange({ kind: 'updated', paths: ['C:/in.png', 'C:/other.png'] }, 'C:/in.png')
    ).toEqual({
      refresh: ['C:/in.png', 'C:/other.png'],
      reloadCurrent: true,
      announce: ['C:/other.png'],
    });
    expect(planMediaChange({ kind: 'updated', paths: ['/a.png'] }, null).announce).toEqual(['/a.png']);
  });
});

describe('mergePaths', () => {
  it('appends only paths not already listed', () => {
    expect(mergePaths(['C:/a.png'], ['c:\\a.png', 'C:/b.png'])).toEqual(['C:/a.png', 'C:/b.png']);
  });
});
