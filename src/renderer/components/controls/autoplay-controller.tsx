import { useSelector } from '@xstate/react';
import { useContext, useEffect, useRef } from 'react';
import { GlobalStateContext } from 'renderer/state';

export default function AutoPlayController() {
  const { libraryService } = useContext(GlobalStateContext);
  const settings = useSelector(
    libraryService,
    (state) => state.context.settings
  );
  const loopCount = useSelector(
    libraryService,
    (state) => state.context.videoPlayer.loopCount
  );

  const incrementCursor = () => {
    libraryService.send('INCREMENT_CURSOR');
  };

  // Time-based autoplay
  useAutoPlay(incrementCursor, settings.autoPlayTime || false);

  // Loop-based autoplay for videos and animated GIFs
  useAutoPlayLoops(incrementCursor, settings.autoPlayVideoLoops || false, loopCount);

  return null;
}

export function useAutoPlay(
  callback: () => void,
  autoPlayTime: number | false
) {
  const savedCallback = useRef(callback);

  // Keep latest callback in ref
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  // Set up interval
  useEffect(() => {
    if (typeof autoPlayTime !== 'number' || autoPlayTime <= 0) return;

    const interval = setInterval(() => {
      savedCallback.current();
    }, autoPlayTime * 1000);

    return () => clearInterval(interval);
  }, [autoPlayTime]);
}

export function useAutoPlayLoops(
  callback: () => void,
  autoPlayVideoLoops: number | false,
  loopCount: number
) {
  const savedCallback = useRef(callback);

  // Keep latest callback in ref
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  // Trigger when loop count reaches threshold
  useEffect(() => {
    if (typeof autoPlayVideoLoops !== 'number' || autoPlayVideoLoops <= 0) return;

    if (loopCount >= autoPlayVideoLoops) {
      savedCallback.current();
    }
  }, [loopCount, autoPlayVideoLoops]);
}
