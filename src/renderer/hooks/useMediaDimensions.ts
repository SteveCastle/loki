import { useState, useEffect } from 'react';

type MediaDimensions = {
  width: number;
  height: number;
  orientation: 'portrait' | 'landscape' | 'unknown';
};

const defaultDimensions: MediaDimensions = {
  width: 0,
  height: 0,
  orientation: 'unknown',
};

type VideoRef = React.RefObject<HTMLVideoElement>;

function isVideoRef(ref: any): ref is VideoRef {
  return (
    typeof ref === 'object' &&
    ref !== null &&
    ref.current instanceof HTMLVideoElement
  );
}

function useMediaDimensions(
  ref: React.RefObject<HTMLImageElement | HTMLVideoElement>
): MediaDimensions {
  const [dimensions, setDimensions] =
    useState<MediaDimensions>(defaultDimensions);

  useEffect(() => {
    if (!ref.current) return;

    const updateDimensions = () => {
      if (!ref.current) return;
      let width = 1;
      let height = 1;
      if (ref && ref.current && isVideoRef(ref)) {
        width = ref.current.videoWidth;
        height = ref.current.videoHeight;
      } else {
        width = ref.current.getBoundingClientRect().width;
        height = ref.current.getBoundingClientRect().height;
      }
      const orientation = width > height ? 'landscape' : 'portrait';

      setDimensions({ width, height, orientation });
    };

    const loadHandler = () => {
      updateDimensions();
      ref.current?.addEventListener('resize', updateDimensions);
    };

    ref.current.addEventListener('load', loadHandler);
    ref.current.addEventListener('loadeddata', loadHandler);
    return () => {
      ref.current?.removeEventListener('load', loadHandler);
      ref.current?.removeEventListener('loadeddata', loadHandler);
      ref.current?.removeEventListener('resize', updateDimensions);
    };
  }, [ref]);

  return dimensions;
}

export default useMediaDimensions;
