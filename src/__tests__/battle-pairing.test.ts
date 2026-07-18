import {
  orderForBattle,
  recordBattleResult,
  resetBattlePairingState,
  sortConfidence,
  NEIGHBOR_WINDOW,
  PLACEMENT_ROUNDS,
} from '../renderer/battle-pairing';
import type { Item } from '../renderer/state';

const item = (path: string, elo?: number, battles?: number): Item => ({
  path,
  mtimeMs: 0,
  elo,
  battles,
});

// A spread of seeds: selection is deterministic per seed, so property
// assertions loop over many seeds instead of asserting exact picks.
const SEEDS = Array.from({ length: 40 }, (_, i) => `seed-${i}`);

beforeEach(() => resetBattlePairingState());

describe('orderForBattle', () => {
  it('returns a permutation with the pair up front', () => {
    const items = Array.from({ length: 20 }, (_, i) =>
      item(`p${i}`, 1400 + i * 10, i)
    );
    for (const seed of SEEDS) {
      const out = orderForBattle(items, seed);
      expect(out).toHaveLength(items.length);
      expect(new Set(out.map((i) => i.path)).size).toBe(items.length);
      expect(out[0].path).not.toBe(out[1].path);
    }
  });

  it('is deterministic for a given seed', () => {
    const items = Array.from({ length: 30 }, (_, i) =>
      item(`p${i}`, 1400 + i * 7, (i % 5) + 1)
    );
    const a = orderForBattle(items, 'stable-seed');
    const b = orderForBattle(items, 'stable-seed');
    expect(a.map((i) => i.path)).toEqual(b.map((i) => i.path));
  });

  it('anchors on the item with the fewest battles', () => {
    const items = [
      item('veteran1', 1600, 40),
      item('veteran2', 1500, 25),
      item('rookie', 1450, 2),
      item('veteran3', 1400, 30),
    ];
    for (const seed of SEEDS) {
      const [anchor] = orderForBattle(items, seed);
      expect(anchor.path).toBe('rookie');
    }
  });

  it('prefers unranked items inside the fewest-battles pool', () => {
    const items = [
      item('ranked-fresh', 1500, 0),
      item('unranked', undefined, 0),
      item('veteran', 1600, 12),
    ];
    for (const seed of SEEDS) {
      const [anchor] = orderForBattle(items, seed);
      expect(anchor.path).toBe('unranked');
    }
  });

  it('usually picks a close-rated opponent, sometimes a distant one', () => {
    // Anchor is pinned via battles=0; everyone else is spread 100 Elo apart,
    // so "inside the window" and "outside the window" are unambiguous. The
    // anchor is ranked (elo set) so binary-search placement doesn't kick in.
    const anchor = item('anchor', 1500, 0);
    const others = Array.from({ length: 200 }, (_, i) =>
      item(`o${i}`, i * 100 + 1, 5)
    );
    const items = [anchor, ...others];
    let close = 0;
    let far = 0;
    for (const seed of SEEDS) {
      const out = orderForBattle(items, seed);
      expect(out[0].path).toBe('anchor');
      const sortedByDist = others
        .map((o) => Math.abs((o.elo as number) - 1500))
        .sort((a, b) => a - b);
      const windowMax = sortedByDist[NEIGHBOR_WINDOW - 1];
      const d = Math.abs((out[1].elo as number) - 1500);
      if (d <= windowMax) close += 1;
      else far += 1;
    }
    // ~85% close / ~15% far; with 40 seeds both behaviors must appear and
    // close battles must dominate.
    expect(close).toBeGreaterThan(far);
    expect(far).toBeGreaterThan(0);
  });

  it('suppresses recent rematches', () => {
    const anchor = item('anchor', 1500, 1);
    const rival = item('rival', 1501, 5); // by far the closest rating
    const others = Array.from({ length: 5 }, (_, i) =>
      item(`o${i}`, 2000 + i * 400, 5)
    );
    const items = [anchor, rival, ...others];
    recordBattleResult('anchor', 1500, 'rival', 1501);
    for (const seed of SEEDS) {
      const out = orderForBattle(items, seed);
      expect(out[0].path).toBe('anchor');
      expect(out[1].path).not.toBe('rival');
    }
  });

  it('falls back to a recent pair when nothing else is eligible', () => {
    const items = [item('a', 1500, 1), item('b', 1500, 1)];
    recordBattleResult('a', 1512, 'b', 1488);
    const out = orderForBattle(items, 'any-seed');
    expect(out).toHaveLength(2);
    expect(new Set(out.map((i) => i.path))).toEqual(new Set(['a', 'b']));
  });

  it('handles tiny inputs', () => {
    expect(orderForBattle([], 'seed')).toEqual([]);
    const single = [item('only', 1500, 0)];
    expect(orderForBattle(single, 'seed')).toEqual(single);
  });
});

describe('binary-search placement', () => {
  // 100 ranked items rated 1000, 1010, ... 1990.
  const ranked = () =>
    Array.from({ length: 100 }, (_, i) => item(`r${i}`, 1000 + i * 10, 10));

  it('starts an unranked item against the median-rated item', () => {
    const items = [item('new', undefined, 0), ...ranked()];
    for (const seed of SEEDS.slice(0, 10)) {
      const out = orderForBattle(items, seed);
      expect(out[0].path).toBe('new');
      // Median of 1000..1990 — allow slack for rematch-skips.
      expect(out[1].elo as number).toBeGreaterThan(1400);
      expect(out[1].elo as number).toBeLessThan(1600);
    }
  });

  it('bisects upward after a win and downward after a loss', () => {
    const items = [item('new', undefined, 0), ...ranked()];
    const first = orderForBattle(items, 'seed-a');
    const firstOpp = first[1];

    // "new" beats the median → next opponent from the upper half.
    recordBattleResult('new', 1524, firstOpp.path, (firstOpp.elo as number) - 5);
    const afterWin = orderForBattle(
      [item('new', 1524, 1), ...ranked()],
      'seed-b'
    );
    expect(afterWin[0].path).toBe('new');
    expect(afterWin[1].elo as number).toBeGreaterThan(firstOpp.elo as number);
    // ~75th percentile of the remaining upper interval.
    expect(afterWin[1].elo as number).toBeGreaterThan(1650);
    expect(afterWin[1].elo as number).toBeLessThan(1850);

    // Then it loses → next opponent between the two previous ones.
    const secondOpp = afterWin[1];
    recordBattleResult(secondOpp.path, secondOpp.elo as number, 'new', 1512);
    const afterLoss = orderForBattle(
      [item('new', 1512, 2), ...ranked()],
      'seed-c'
    );
    expect(afterLoss[0].path).toBe('new');
    expect(afterLoss[1].elo as number).toBeGreaterThan(firstOpp.elo as number);
    expect(afterLoss[1].elo as number).toBeLessThan(secondOpp.elo as number);
  });

  it('exits placement after PLACEMENT_ROUNDS battles', () => {
    const pool = ranked();
    let items = [item('new', undefined, 0), ...pool];
    let current = orderForBattle(items, 'seed-0');
    for (let round = 0; round < PLACEMENT_ROUNDS; round++) {
      const opp = current[1];
      recordBattleResult('new', 1600, opp.path, opp.elo as number);
      items = [item('new', 1600, round + 1), ...pool];
      current = orderForBattle(items, `seed-${round + 1}`);
    }
    // Placement is over: with anchor at 1600 the opponent comes from the
    // close-rating window (or an explore pick), not a forced bisection of
    // the shrunken upper interval. Just assert the pair is still valid and
    // placement state didn't leak an empty-interval crash.
    expect(current[0].path).toBe('new');
    expect(current[1].path).not.toBe('new');
  });

  it('skips placement for small ranked pools', () => {
    const items = [
      item('new', undefined, 0),
      item('a', 1400, 3),
      item('b', 1500, 3),
      item('c', 1600, 3),
    ];
    const out = orderForBattle(items, 'seed-x');
    expect(out[0].path).toBe('new');
    expect(out).toHaveLength(4);
  });
});

describe('sortConfidence', () => {
  it('is 0 for a completely unbattled set and 1 for a saturated one', () => {
    const fresh = Array.from({ length: 50 }, (_, i) => item(`p${i}`));
    expect(sortConfidence(fresh).score).toBe(0);
    expect(sortConfidence(fresh).unranked).toBe(50);

    const seasoned = Array.from({ length: 50 }, (_, i) =>
      item(`p${i}`, 1500 + i, 30)
    );
    expect(sortConfidence(seasoned).score).toBe(1);
    expect(sortConfidence(seasoned).unranked).toBe(0);
  });

  it('grows monotonically with battles and scales the target with set size', () => {
    const at = (battles: number, n = 100) =>
      sortConfidence(
        Array.from({ length: n }, (_, i) => item(`p${i}`, 1500, battles))
      );
    expect(at(2).score).toBeLessThan(at(5).score);
    expect(at(5).score).toBeLessThan(at(10).score);

    // 100 items → target ceil(2·log2(100)) = 14; 16 items → 8.
    expect(at(0, 100).target).toBe(14);
    expect(at(0, 16).target).toBe(8);
    // Clamped: tiny sets still need 5, huge sets never need more than 20.
    expect(at(0, 4).target).toBe(5);
    expect(at(0, 2_000_000).target).toBe(20);
  });

  it('treats trivial sets as fully sorted', () => {
    expect(sortConfidence([]).score).toBe(1);
    expect(sortConfidence([item('only')]).score).toBe(1);
  });

  it('reports average battles for the tooltip', () => {
    const items = [item('a', 1500, 4), item('b', 1500, 8), item('c')];
    expect(sortConfidence(items).avgBattles).toBeCloseTo(4);
    expect(sortConfidence(items).unranked).toBe(1);
  });
});
