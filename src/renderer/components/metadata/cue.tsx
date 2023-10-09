import { useContext, useEffect, useRef, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { VttCue } from 'main/parse-vtt';
import { uniqueId } from 'xstate/lib/utils';

type Props = {
  cue: VttCue;
  setScrollTop: (scrollTop: number) => void;
  followVideoTime?: boolean;
};

function convertVTTTimestampToSeconds(timestamp: string) {
  const [minutes, seconds] = timestamp.split(':');
  return parseInt(minutes) * 60 + parseFloat(seconds);
}

function usePrevious<T>(value: T): T {
  // The ref object is a generic container whose current property is mutable ...
  // ... and can hold any value, similar to an instance property on a class
  const ref: any = useRef<T>();
  // Store current value in ref
  useEffect(() => {
    ref.current = value;
  }, [value]); // Only re-run if value changes
  // Return previous value (happens before update in useEffect above)
  return ref.current;
}

export function Cue({ cue, setScrollTop, followVideoTime = false }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const { actualVideoTime } = useSelector(
    libraryService,
    (state) => state.context.videoPlayer
  );

  const startTimeInSeconds = convertVTTTimestampToSeconds(cue.startTime);
  const endTimeInSeconds = convertVTTTimestampToSeconds(cue.endTime);
  const isActive =
    actualVideoTime >= startTimeInSeconds && actualVideoTime < endTimeInSeconds;
  const previousIsActive = usePrevious(isActive);
  const ref = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (isActive && !previousIsActive && ref.current && followVideoTime) {
      setScrollTop(ref.current.offsetTop);
    }
  }, [actualVideoTime, followVideoTime]);

  return (
    <li
      ref={ref}
      key={cue.startTime}
      onClick={() => {
        const timeStamp = convertVTTTimestampToSeconds(cue.startTime);
        libraryService.send('SET_VIDEO_TIME', {
          timeStamp,
          eventId: uniqueId(),
        });
      }}
    >
      <div className={['cue', isActive ? 'active' : ''].join(' ')}>
        <div className="row">
          <span className="start">{cue.startTime}</span>
          <span className="end">{cue.endTime}</span>
        </div>
        <div className="row">
          <span className="text">{cue.text}</span>
        </div>
      </div>
    </li>
  );
}
