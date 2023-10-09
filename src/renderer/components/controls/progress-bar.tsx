import React from 'react';
import { useCallback, useState } from 'react';
import { useEventListener, useIsomorphicLayoutEffect } from 'usehooks-ts';
import './progress-bar.css';

type Props = {
  total: number;
  value: number;
  isLoading: boolean;
  setCursor: (cursor: number) => void;
};

function mapRange(
  value: number,
  in_min: number,
  in_max: number,
  out_min: number,
  out_max: number
) {
  return ((value - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min;
}

interface Size {
  width: number;
  height: number;
}

function useElementSize<T extends HTMLElement = HTMLDivElement>(): [
  (node: T | null) => void,
  Size
] {
  // Mutable values like 'ref.current' aren't valid dependencies
  // because mutating them doesn't re-render the component.
  // Instead, we use a state as a ref to be reactive.
  const [ref, setRef] = useState<T | null>(null);
  const [size, setSize] = useState<Size>({
    width: 0,
    height: 0,
  });

  // Prevent too many rendering using useCallback
  const handleSize = useCallback(() => {
    setSize({
      width: ref?.offsetWidth || 0,
      height: ref?.offsetHeight || 0,
    });
  }, [ref?.offsetHeight, ref?.offsetWidth]);

  useEventListener('resize', handleSize);

  useIsomorphicLayoutEffect(() => {
    handleSize();
  }, [ref?.offsetHeight, ref?.offsetWidth]);

  return [setRef, size];
}

function getLabel(value: number, total: number) {
  return total > 0 ? `${value + 1} of ${total}` : 'No files';
}

export default function ProgressBar({
  total,
  value,
  setCursor,
  isLoading,
}: Props) {
  //height of element ref in pixels
  const [setRef, { width }] = useElementSize<HTMLDivElement>();
  const [isDragging, setIsDragging] = useState(false);

  function handleMouseDown(e: React.MouseEvent<HTMLDivElement, MouseEvent>) {
    setIsDragging(true);
    updateCursor(e);
  }

  function handleMouseMove(e: React.MouseEvent<HTMLDivElement, MouseEvent>) {
    if (isDragging) {
      updateCursor(e);
    }
  }

  function handleMouseUp() {
    setIsDragging(false);
  }

  function handleMouseLeave() {
    setIsDragging(false);
  }

  function updateCursor(e: React.MouseEvent<HTMLDivElement, MouseEvent>) {
    const newCursor = Math.round(
      mapRange(e.nativeEvent.offsetX, 0, width, 0, total - 1)
    );
    setCursor(newCursor);
  }
  return (
    <div
      className="ProgressBar"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      ref={setRef}
    >
      <div className="label">
        {isLoading ? (
          <span className="value">Loading Images</span>
        ) : (
          <span className="value">{getLabel(value, total)}</span>
        )}
      </div>
      <div
        style={{ width: `${Math.floor(((value + 1) / total) * 100)}%` }}
        className="progress"
      ></div>
    </div>
  );
}
