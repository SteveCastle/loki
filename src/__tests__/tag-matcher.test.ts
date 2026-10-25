// The ranked tag matcher that replaced Fuse.js for tag type-ahead. These pin
// the RANKING contract (exact > whole-word prefix > prefix > word-start >
// substring > fuzzy > category fallback) and the scale contract: the full
// ~190K-tag table must answer well inside a frame, where Fuse took 0.25–1s.
import {
  buildIndex,
  matchTags,
  normalize,
  packTags,
} from '../renderer/search/tag-matcher';

type T = { label: string; category: string };

function search(tags: T[], q: string, limit = 50): string[] {
  const idx = buildIndex(packTags(tags));
  return matchTags(idx, q, limit).map((i) => tags[i].label);
}

const TAGS: T[] = [
  { label: 'Classical', category: 'Genres' },
  { label: 'alba', category: 'Suggested' },
  { label: 'Al Pacino', category: 'Actors' },
  { label: 'Tupac', category: 'Artists' },
  { label: 'blonde_hair', category: 'Suggested' },
  { label: 'hair_blonde_long', category: 'Suggested' },
  { label: 'cat', category: 'Suggested' },
  { label: 'cat_ears', category: 'Suggested' },
  { label: 'bobcat', category: 'Suggested' },
  { label: 'Beyoncé', category: 'People' },
];

describe('tag-matcher', () => {
  it('normalizes case, accents and autotagger separators', () => {
    expect(normalize('Blonde_Hair')).toBe('blonde hair');
    expect(normalize('Beyoncé')).toBe('beyonce');
    expect(normalize('  a--b..c  ')).toBe('a b c');
  });

  it('ranks a whole-word prefix above a partial one and a substring', () => {
    const r = search(TAGS, 'al');
    expect(r[0]).toBe('Al Pacino');
    expect(r.indexOf('alba')).toBeLessThan(r.indexOf('Classical'));
  });

  it('ranks exact > prefix > substring', () => {
    expect(search(TAGS, 'cat')).toEqual(['cat', 'cat_ears', 'bobcat']);
  });

  it('matches multi-word queries in any order across separators', () => {
    const r = search(TAGS, 'blonde hair');
    expect(r[0]).toBe('blonde_hair');
    expect(search(TAGS, 'hair blonde')).toEqual(
      expect.arrayContaining(['blonde_hair', 'hair_blonde_long'])
    );
  });

  it('tolerates typos in longer words', () => {
    expect(search(TAGS, 'pacnio')).toContain('Al Pacino');
    expect(search(TAGS, 'blond hiar')).toContain('blonde_hair');
  });

  it('does not fuzz short words', () => {
    expect(search(TAGS, 'cot')).toEqual([]);
  });

  it('folds accents in queries', () => {
    expect(search(TAGS, 'beyonce')).toEqual(['Beyoncé']);
  });

  it('falls back to category matches only for leftover slots', () => {
    const r = search(TAGS, 'actors');
    expect(r).toEqual(['Al Pacino']);
  });

  it('honours the limit', () => {
    expect(search(TAGS, 'a', 2)).toHaveLength(2);
  });

  it('answers a 190K-tag table quickly', () => {
    const words = ['blonde', 'hair', 'long', 'smile', 'outdoor', 'red', 'dress', 'portrait', 'beach', 'night'];
    const tags: T[] = [];
    for (let i = 0; i < 190000; i += 1) {
      tags.push({
        label: `${words[i % 10]}_${words[(i * 7) % 10]}_${i}`,
        category: 'Suggested',
      });
    }
    const idx = buildIndex(packTags(tags));
    const t0 = Date.now();
    for (const q of ['bl', 'blonde hair', 'portriat', 'zzzzzzzz']) matchTags(idx, q, 200);
    // Generous for slow CI; Fuse took >2s for these four.
    expect(Date.now() - t0).toBeLessThan(800);
  });
});
