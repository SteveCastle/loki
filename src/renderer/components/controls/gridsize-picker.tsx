import { useState, useContext } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';

import './gridsize-picker.css';

function getCoordinates(n: number): [number, number] {
  if (n < 1 || n > 256) {
    throw new Error('Input must be between 1 and 256.');
  }

  // Decrement the number by 1 because grid coordinates start from 0,0.
  n = n - 1;

  const x = n % 16;
  const y = Math.floor(n / 16);

  return [x + 1, y + 1];
}

function isInside(n: number, gridSize: [number, number]): boolean {
  const [x, y] = getCoordinates(n);
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

  return (
    <div
      className="GridSizePicker"
      onMouseLeave={() => {
        setHoveredSize(false);
      }}
    >
      {Array.from({ length: 256 }, (_, i) => i + 1).map((i) => {
        return (
          <div
            key={i}
            className={`grid-item ${
              hoveredSize && isInside(i, hoveredSize) ? 'hovered' : ' '
            } ${isInside(i, gridSize) ? 'selected' : ''}
            ${
              isInside(i, gridSize) && hoveredSize && !isInside(i, hoveredSize)
                ? 'deselected'
                : ''
            }
            `}
            onMouseEnter={() => {
              setHoveredSize(getCoordinates(i));
            }}
            onClick={() => {
              libraryService.send('CHANGE_SETTING', {
                data: {
                  gridSize: getCoordinates(i),
                },
              });
            }}
          />
        );
      })}
    </div>
  );
}
