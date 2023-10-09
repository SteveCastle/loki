import React, { useContext, useCallback, useState, useEffect } from 'react';
import { useSelector } from '@xstate/react';
import { useEventListener, useIsomorphicLayoutEffect } from 'usehooks-ts';
import repeat from '../../../../assets/repeat.svg';

import { uniqueId } from 'xstate/lib/utils';
import { GlobalStateContext, Item } from '../../state';
import './video-controls.css';

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

function getLabel(currentVideoTimeStamp: number) {
  // Returns a string in the format of 00:00:00
  const hours = Math.floor(currentVideoTimeStamp / 3600);
  const minutes = Math.floor((currentVideoTimeStamp - hours * 3600) / 60);
  const seconds = Math.floor(
    currentVideoTimeStamp - hours * 3600 - minutes * 60
  );

  const hoursString = hours < 10 ? `0${hours}` : `${hours}`;
  const minutesString = minutes < 10 ? `0${minutes}` : `${minutes}`;
  const secondsString = seconds < 10 ? `0${seconds}` : `${seconds}`;

  return `${hoursString}:${minutesString}:${secondsString}`;
}

export default function VideoControls() {
  const { libraryService } = useContext(GlobalStateContext);
  const { actualVideoTime, videoLength, loopLength, playing } = useSelector(
    libraryService,
    (state) => state.context.videoPlayer
  );

  const [setRef, { width }] = useElementSize<HTMLDivElement>();
  const [isDragging, setIsDragging] = useState(false);

  function handleMouseDown(e: React.MouseEvent<HTMLDivElement, MouseEvent>) {
    setIsDragging(true);
    updateVideoTimeStamp(e);
  }

  function handleMouseMove(e: React.MouseEvent<HTMLDivElement, MouseEvent>) {
    if (isDragging) {
      updateVideoTimeStamp(e);
    }
  }

  function handleMouseUp() {
    setIsDragging(false);
  }

  function handleMouseLeave() {
    setIsDragging(false);
  }

  function updateVideoTimeStamp(
    e: React.MouseEvent<HTMLDivElement, MouseEvent>
  ) {
    const newTimeStamp = Math.round(
      mapRange(e.nativeEvent.offsetX, 0, width, 0, videoLength)
    );
    libraryService.send('SET_VIDEO_TIME', {
      timeStamp: newTimeStamp,
      eventId: uniqueId(),
    });
    libraryService.send('LOOP_VIDEO', {
      loopStartTime: newTimeStamp,
      loopLength,
    });
  }
  return (
    <div className="VideoControls">
      <div
        className="progressBar"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        ref={setRef}
      >
        <div className="label">
          <span className="value">{getLabel(actualVideoTime)}</span>
        </div>
        <div
          style={{
            width: `${Math.floor((actualVideoTime / videoLength) * 100)}%`,
          }}
          className="progress"
        ></div>
      </div>
      <div className="playerButtons">
        <button
          onClick={() => {
            libraryService.send('SET_PLAYING_STATE', {
              playing: !playing,
            });
          }}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
      </div>
      <div className="loopButtons">
        <div className="icon">
          <img src={repeat} alt="Repeat Icon" />
        </div>
        {[1, 2, 5, 10].map((length) => (
          <button
            key={length}
            className={[
              'loopButton',
              loopLength === length ? 'selected' : '',
            ].join(' ')}
            onClick={() => {
              if (loopLength && loopLength === length) {
                libraryService.send('LOOP_VIDEO', {
                  loopStartTime: 0,
                  loopLength: 0,
                });
                return;
              }
              libraryService.send('LOOP_VIDEO', {
                loopStartTime: actualVideoTime,
                loopLength: length,
              });
            }}
          >
            <span>{`${length}`}</span>
            <span className="units">sec</span>
          </button>
        ))}
      </div>
    </div>
  );
}
