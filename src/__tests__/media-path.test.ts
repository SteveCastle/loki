// movedPath must pick exactly the paths the database move picked — the
// renderer uses it to rewrite the library it is already showing, and a looser
// rule would rename an on-screen item whose DB row never moved. These cases
// mirror media-server/media/move_test.go and move-media.test.ts.
import { movedPath, moveRange, parentDir } from '../renderer/media-path';

describe('movedPath', () => {
  it('rewrites the exact path', () => {
    expect(movedPath('/photos/a.jpg', '/photos/a.jpg', '/archive/a.jpg')).toBe(
      '/archive/a.jpg'
    );
  });

  it('rewrites paths under the moved folder, keeping the sub-path', () => {
    expect(movedPath('/photos/2023/sub/b.jpg', '/photos/2023', '/archive/2023')).toBe(
      '/archive/2023/sub/b.jpg'
    );
  });

  it('is segment-aligned — a shared name prefix is not a match', () => {
    expect(movedPath('/photos/2023extra/c.jpg', '/photos/2023', '/archive/2023')).toBe(
      '/photos/2023extra/c.jpg'
    );
  });

  it('preserves the stored separator style', () => {
    expect(movedPath('C:\\media\\shoot\\a.jpg', 'C:\\media\\shoot', 'D:\\archive\\shoot')).toBe(
      'D:\\archive\\shoot\\a.jpg'
    );
  });

  it('leaves unrelated paths alone', () => {
    expect(movedPath('/photos/2024/d.jpg', '/photos/2023', '/archive/2023')).toBe(
      '/photos/2024/d.jpg'
    );
    expect(movedPath('/elsewhere/a.jpg', '/photos/a.jpg', '/archive/a.jpg')).toBe(
      '/elsewhere/a.jpg'
    );
  });
});

describe('parentDir', () => {
  it('handles both separator styles regardless of the client OS', () => {
    expect(parentDir('/photos/2023/a.jpg')).toBe('/photos/2023');
    expect(parentDir('C:\\media\\shoot\\a.jpg')).toBe('C:\\media\\shoot');
  });

  it('returns the path unchanged when there is no parent', () => {
    expect(parentDir('a.jpg')).toBe('a.jpg');
    expect(parentDir('/a.jpg')).toBe('/a.jpg');
  });
});

describe('moveRange', () => {
  it('moves just the file by default', () => {
    expect(moveRange('/photos/a.jpg', '/archive/a.jpg', false)).toEqual({
      from: '/photos/a.jpg',
      to: '/archive/a.jpg',
    });
  });

  it('widens to the parent folders when the whole folder moved', () => {
    // "Find All With Same Base Path": the user points at one file in the
    // folder's new home, and every sibling under it follows.
    expect(
      moveRange('C:\\pics\\2023\\a.jpg', 'D:\\archive\\2023\\a.jpg', true)
    ).toEqual({
      from: 'C:\\pics\\2023',
      to: 'D:\\archive\\2023',
    });
  });

  it('composes with movedPath to relocate a sibling', () => {
    const { from, to } = moveRange('/pics/2023/a.jpg', '/archive/2023/a.jpg', true);
    expect(movedPath('/pics/2023/sub/b.jpg', from, to)).toBe('/archive/2023/sub/b.jpg');
  });
});
