// Cursor bookkeeping behind "Remove from Library" (media-error panel): after
// the database rows for an unloadable item are erased, the viewer must land on
// the NEXT item — not skip one, not fall off the end.
import {
  cursorAfterRemoval,
  libraryWithout,
} from '../renderer/library-cursor';

const lib = (...paths: string[]) => paths.map((path) => ({ path }));

describe('libraryWithout', () => {
  it('drops the matching entry and leaves the original array alone', () => {
    const original = lib('a', 'b', 'c');
    const next = libraryWithout(original, 'b');
    expect(next.map((i) => i.path)).toEqual(['a', 'c']);
    expect(original.map((i) => i.path)).toEqual(['a', 'b', 'c']);
  });

  it('returns the library unchanged when the path is not present', () => {
    const original = lib('a', 'b');
    expect(libraryWithout(original, 'zzz')).toBe(original);
  });
});

describe('cursorAfterRemoval', () => {
  it('holds the index when removing the item at the cursor — that IS the next item', () => {
    // ['a','b','c'], cursor on 'b'. After removal: ['a','c'] and index 1 is
    // 'c' — the item that came after the one just removed.
    const library = lib('a', 'b', 'c');
    expect(cursorAfterRemoval(library, 1, 'b')).toBe(1);
    expect(libraryWithout(library, 'b')[1].path).toBe('c');
  });

  it('steps back to the new last item when the removed item was last', () => {
    const library = lib('a', 'b', 'c');
    expect(cursorAfterRemoval(library, 2, 'c')).toBe(1);
    expect(libraryWithout(library, 'c')[1].path).toBe('b');
  });

  it('shifts back one when the removed item was before the cursor', () => {
    // The user is looking at 'c'; removing 'a' must keep them on 'c'.
    const library = lib('a', 'b', 'c');
    const cursor = cursorAfterRemoval(library, 2, 'a');
    expect(cursor).toBe(1);
    expect(libraryWithout(library, 'a')[cursor].path).toBe('c');
  });

  it('leaves the cursor alone when the removed item was after it', () => {
    const library = lib('a', 'b', 'c');
    const cursor = cursorAfterRemoval(library, 0, 'c');
    expect(cursor).toBe(0);
    expect(libraryWithout(library, 'c')[cursor].path).toBe('a');
  });

  it('never goes negative when the last remaining item is removed', () => {
    expect(cursorAfterRemoval(lib('only'), 0, 'only')).toBe(0);
  });

  it('leaves the cursor untouched for a path that is not in the library', () => {
    expect(cursorAfterRemoval(lib('a', 'b'), 1, 'ghost')).toBe(1);
  });
});
