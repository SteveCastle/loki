// The progress UI's reading of a search in flight: which chips are pending and
// how the search is described (see src/renderer/search-activity.ts).
import {
  describeSearch,
  pendingPredicateKeys,
} from '../renderer/search-activity';
import type { Predicate, Query } from '../renderer/query/types';

const p = (type: Predicate['type'], value: string, extra = {}): Predicate => ({
  type,
  value,
  exclude: false,
  ...extra,
});
const q = (...predicates: Predicate[]): Query => ({ predicates });

describe('pendingPredicateKeys', () => {
  it('marks only the chips the settled results lack', () => {
    const keys = pendingPredicateKeys(
      q(p('tag', 'cats'), p('path', 'zaush')),
      q(p('tag', 'cats'))
    );
    expect([...keys]).toEqual(['path:zaush']);
  });

  it('treats an exclude flip or a blend edit as pending', () => {
    expect(
      pendingPredicateKeys(
        q(p('tag', 'cats', { exclude: true })),
        q(p('tag', 'cats'))
      ).size
    ).toBe(1);
    expect(
      pendingPredicateKeys(
        q(p('similar', '/a.jpg', { nodes: [{ kind: 'text', value: 'night' }] })),
        q(p('similar', '/a.jpg'))
      ).size
    ).toBe(1);
  });

  it('is empty when nothing new was asked for (e.g. a chip removed)', () => {
    expect(
      pendingPredicateKeys(q(p('tag', 'cats')), q(p('tag', 'cats'), p('tag', 'dogs')))
        .size
    ).toBe(0);
  });
});

describe('describeSearch', () => {
  it('describes a path search as a full scan with a hint', () => {
    const d = describeSearch(q(p('path', 'zaush')), q());
    expect(d.kind).toBe('scan');
    expect(d.subject).toBe('zaush');
    expect(d.slowHint).toBeTruthy();
  });

  it('describes embedding searches as vector searches', () => {
    expect(describeSearch(q(p('visual', 'a red car')), q())).toMatchObject({
      kind: 'vector',
      subject: 'a red car',
    });
    expect(describeSearch(q(p('similar', '/a.jpg')), q()).kind).toBe('vector');
    expect(describeSearch(q(p('face', '/a.jpg')), q()).kind).toBe('vector');
  });

  it('names the newest pending chip but takes the heaviest kind', () => {
    // Adding a tag to a running path search still waits on the path scan.
    const d = describeSearch(q(p('path', 'zaush'), p('tag', 'cats')), q());
    expect(d.subject).toBe('cats');
    expect(d.kind).toBe('scan');
    expect(d.slowHint).toBeTruthy();
  });

  it('reads a removal as an update', () => {
    expect(
      describeSearch(q(p('tag', 'cats')), q(p('tag', 'cats'), p('tag', 'dogs')))
    ).toMatchObject({ kind: 'filter', verb: 'Updating results' });
  });
});
