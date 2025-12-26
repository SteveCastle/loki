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

function useMediaDimensions(
  ref: React.RefObject<HTMLImageElement | HTMLVideoElement>
): MediaDimensions {
  const [dimensions, setDimensions] =
    useState<MediaDimensions>(defaultDimensions);

  useEffect(() => {
    if (!ref.current) return;

    const updateDimensions = () => {
      if (!ref.current) return;
      let width = 0;
      let height = 0;

      if (ref.current instanceof HTMLVideoElement) {
        // For video, use videoWidth/videoHeight (intrinsic dimensions)
        width = ref.current.videoWidth;
        height = ref.current.videoHeight;
      } else if (ref.current instanceof HTMLImageElement) {
        // For images, use naturalWidth/naturalHeight (intrinsic dimensions)
        width = ref.current.naturalWidth;
        height = ref.current.naturalHeight;
      }

      // Only update if we have valid dimensions
      if (width > 0 && height > 0) {
        const orientation = width > height ? 'landscape' : 'portrait';
        setDimensions({ width, height, orientation });
      }
    };

    const loadHandler = () => {
      updateDimensions();
      ref.current?.addEventListener('resize', updateDimensions);
    };

    // Check if already loaded (for cached images)
    if (ref.current instanceof HTMLImageElement && ref.current.complete) {
      updateDimensions();
    } else if (
      ref.current instanceof HTMLVideoElement &&
      ref.current.readyState >= 1
    ) {
      updateDimensions();
    }

    ref.current.addEventListener('load', loadHandler);
    ref.current.addEventListener('loadeddata', loadHandler);
    ref.current.addEventListener('loadedmetadata', loadHandler);
    return () => {
      ref.current?.removeEventListener('load', loadHandler);
      ref.current?.removeEventListener('loadeddata', loadHandler);
      ref.current?.removeEventListener('loadedmetadata', loadHandler);
      ref.current?.removeEventListener('resize', updateDimensions);
    };
  }, [ref]);

  return dimensions;
}

export default useMediaDimensions;
