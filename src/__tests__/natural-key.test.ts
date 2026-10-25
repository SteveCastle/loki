// natural-key must order EXACTLY like natural-compare (which it replaced in the
// name sort for speed), including its quirks: a leading '0' is a character,
// not a number start, and '-' sorts before digits.
import naturalCompare from 'natural-compare';
import { naturalKey, sortByNaturalKey } from '../renderer/natural-key';

const cmpKeys = (a: string, b: string) => {
  const ka = naturalKey(a);
  const kb = naturalKey(b);
  return ka < kb ? -1 : ka > kb ? 1 : 0;
};

describe('natural-key', () => {
  it('agrees with naturalCompare on random strings', () => {
    let seed = 42;
    const rnd = () => {
      seed = (seed * 16807) % 2147483647;
      return seed / 2147483647;
    };
    const alphabet = 'ab-_.0123456789 /\\:é~[]Z';
    const make = () => {
      let s = '';
      const len = Math.floor(rnd() * 9);
      for (let i = 0; i < len; i += 1) s += alphabet[Math.floor(rnd() * alphabet.length)];
      return s;
    };
    for (let i = 0; i < 20000; i += 1) {
      const a = make();
      const b = make();
      expect([a, b, cmpKeys(a, b)]).toEqual([a, b, Math.sign(naturalCompare(a, b))]);
    }
  });

  it('handles the known quirks', () => {
    for (const [a, b] of [
      ['img2', 'img10'],
      ['a007', 'a7'],
      ['x-1', 'x1'],
      ['abc', 'abcd'],
      ['9', '10'],
    ]) {
      expect(cmpKeys(a, b)).toBe(Math.sign(naturalCompare(a, b)));
    }
  });

  it('sorts stably', () => {
    const items = ['b', 'a', 'b', 'a'].map((k, i) => ({ k, i }));
    expect(sortByNaturalKey(items, (x) => x.k).map((x) => x.i)).toEqual([1, 3, 0, 2]);
  });
});
