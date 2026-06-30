jest.mock('../renderer/platform', () => ({ __esModule: true }));

import { rectFromDrag } from '../renderer/region-search';

test('normalizes a top-left → bottom-right drag', () => {
  expect(rectFromDrag({ x: 10, y: 20 }, { x: 110, y: 220 })).toEqual({
    x: 10, y: 20, width: 100, height: 200,
  });
});

test('normalizes a bottom-right → top-left drag (any direction)', () => {
  expect(rectFromDrag({ x: 110, y: 220 }, { x: 10, y: 20 })).toEqual({
    x: 10, y: 20, width: 100, height: 200,
  });
});
