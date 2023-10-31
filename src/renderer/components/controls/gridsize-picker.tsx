import { useState, useContext } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';

import './gridsize-picker.css';

function getCoordinates(n: number, cells: number): [number, number] {
  if (n < 1 || n > cells) {
    throw new Error('Input must be between 1 and 256.');
  }

  // Decrement the number by 1 because grid coordinates start from 0,0.
  n = n - 1;
  const size = Math.sqrt(cells);
  const x = n % size;
  const y = Math.floor(n / size);

  return [x + 1, y + 1];
}

function isInside(
  n: number,
  gridSize: [number, number],
  cells: number
): boolean {
  const [x, y] = getCoordinates(n, cells);
  const [gridX, gridY] = gridSize;

  if (x <= gridX && y <= gridY) {
    return true;
  }

  return false;
}

export default function GridSizePicker() {
  const [hoveredSize, setHoveredSize] = useState<[number, number] | false>(
    false
  );

  const { libraryService } = useContext(GlobalStateContext);
  const gridSize = useSelector(
    libraryService,
    (state) => state.context.settings.gridSize
  );

  const filter = useSelector(
    libraryService,
    (state) => state.context.settings.filters
  );

  const isStatic = filter === 'static';

  return (
    <div
      className="GridSizePicker"
      style={{
        gridTemplateRows: `repeat(${isStatic ? 16 : 8}, 12px)`,
        gridTemplateColumns: `repeat(${isStatic ? 16 : 8}, 12px)`,
      }}
      onMouseLeave={() => {
        setHoveredSize(false);
      }}
    >
      {Array.from({ length: isStatic ? 256 : 64 }, (_, i) => i + 1).map((i) => {
        return (
          <div
            key={i}
            className={`grid-item ${
              hoveredSize && isInside(i, hoveredSize, isStatic ? 256 : 64)
                ? 'hovered'
                : ' '
            } ${isInside(i, gridSize, isStatic ? 256 : 64) ? 'selected' : ''}
            ${
              isInside(i, gridSize, isStatic ? 256 : 64) &&
              hoveredSize &&
              !isInside(i, hoveredSize, isStatic ? 256 : 64)
                ? 'deselected'
                : ''
            }
            `}
            onMouseEnter={() => {
              setHoveredSize(getCoordinates(i, isStatic ? 256 : 64));
            }}
            onClick={() => {
              libraryService.send('CHANGE_SETTING', {
                data: {
                  gridSize: getCoordinates(i, isStatic ? 256 : 64),
                },
              });
            }}
          />
        );
      })}
    </div>
  );
}
