import { useSelector } from '@xstate/react';
import { useContext, useEffect, useRef } from 'react';
import { GlobalStateContext } from 'renderer/state';

export default function AutoPlayController() {
  const { libraryService } = useContext(GlobalStateContext);
  const settings = useSelector(
    libraryService,
    (state) => state.context.settings
  );

  useAutoPlay(() => {
    libraryService.send('INCREMENT_CURSOR');
  }, settings.autoPlayTime || false);

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
