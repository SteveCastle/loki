// Context-palette multi-selection: shift+right-click while the palette is
// open adds the visual range from the opening item (the anchor) to the
// clicked item; ctrl+click toggles a single item. Order follows the list's
// display order, duplicates never accumulate, and tasks receive the paths as
// a newline-separated list.
import { extendSelection } from '../renderer/context-selection';

const items = ['a', 'b', 'c', 'd', 'e'].map((p) => ({ path: `C:\\m\\${p}.jpg` }));
const p = (i: number) => items[i].path;

describe('extendSelection', () => {
  it('adds the inclusive range from the anchor down to the clicked item', () => {
    expect(extendSelection([p(1)], 'range', 1, 3, p(3), items)).toEqual([
      p(1),
      p(2),
      p(3),
    ]);
  });

  it('supports ranges above the anchor', () => {
    expect(extendSelection([p(3)], 'range', 3, 1, p(1), items)).toEqual([
      p(3),
      p(1),
      p(2),
    ]);
  });

  it('never duplicates already-selected paths', () => {
    const twice = extendSelection(
      extendSelection([p(0)], 'range', 0, 2, p(2), items),
      'range',
      0,
      3,
      p(3),
      items
    );
    expect(twice).toEqual([p(0), p(1), p(2), p(3)]);
  });

  it('single mode toggles one item', () => {
    const added = extendSelection([p(0)], 'single', 0, 4, p(4), items);
    expect(added).toEqual([p(0), p(4)]);
    expect(extendSelection(added, 'single', 0, 4, p(4), items)).toEqual([p(0)]);
  });

  it('range without a usable anchor degrades to a single add', () => {
    expect(extendSelection([], 'range', null, 2, p(2), items)).toEqual([p(2)]);
    expect(extendSelection([p(0)], 'range', 0, undefined, p(2), items)).toEqual([
      p(0),
      p(2),
    ]);
  });

  it('clamps out-of-bounds indices to the item list', () => {
    expect(extendSelection([], 'range', 3, 99, p(4), items)).toEqual([
      p(3),
      p(4),
    ]);
  });

  it('paths with commas survive as single entries (newline is the separator)', () => {
    const comma = [{ path: 'C:\\m\\foo, bar.jpg' }, { path: 'C:\\m\\z.jpg' }];
    const sel = extendSelection([], 'range', 0, 1, comma[1].path, comma);
    expect(sel).toEqual(['C:\\m\\foo, bar.jpg', 'C:\\m\\z.jpg']);
    expect(sel.join('\n')).toBe('C:\\m\\foo, bar.jpg\nC:\\m\\z.jpg');
  });
});
