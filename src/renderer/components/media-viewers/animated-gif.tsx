import React, { useState, useEffect, useMemo, useContext, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';
import { GlobalStateContext } from '../../state';

import type { ScaleModeOption } from '../../../settings';
import './image.css';
import './sizing.css';
import MediaErrorMsg from './media-error';

type Props = {
  path: string;
  scaleMode?: ScaleModeOption;
  coverSize?: { width: number; height: number };
  mediaRef?: React.RefObject<HTMLImageElement>;
  handleLoad?: React.ReactEventHandler<HTMLImageElement>;
  orientation: 'portrait' | 'landscape' | 'unknown';
  cache?: 'thumbnail_path_1200' | 'thumbnail_path_600' | false;
  version?: number;
};

const fetchMediaPreview =
  (item: string, cache: 'thumbnail_path_1200' | 'thumbnail_path_600' | false) =>
  async (): Promise<string> => {
    const path = await window.electron.fetchMediaPreview(item, cache);
    return path;
  };

const fetchGifMetadata = (filePath: string) => async () => {
  const result = await window.electron.getGifMetadata(filePath);
  return result;
};

function AnimatedGifComponent({
  path,
  scaleMode = 'cover',
  mediaRef,
  coverSize = { width: 0, height: 0 },
  handleLoad,
  cache = false,
  orientation = 'unknown',
  version = 0,
}: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const [error, setError] = useState<boolean>(false);
  const loopStartTimeRef = useRef<number>(0);

  const { data: previewData } = useQuery<string, Error>(
    ['media', 'preview', path, cache, version],
    fetchMediaPreview(path, cache)
  );

  // Fetch GIF metadata (frame count and duration)
  const { data: gifMetadata } = useQuery(
    ['gif', 'metadata', path],
    fetchGifMetadata(path),
    {
      staleTime: Infinity, // GIF metadata won't change
      cacheTime: 1000 * 60 * 60, // Cache for 1 hour
    }
  );

  // Reset error state and loop tracking if path changes
  useEffect(() => {
    setError(false);
    loopStartTimeRef.current = Date.now();
  }, [path]);

  // Track GIF loops based on duration
  useEffect(() => {
    // Only track loops for multi-frame GIFs with valid duration
    if (!gifMetadata || gifMetadata.frameCount <= 1 || gifMetadata.duration <= 0) {
      return;
    }

    const loopDuration = gifMetadata.duration;
    loopStartTimeRef.current = Date.now();

    const intervalId = setInterval(() => {
      const elapsed = Date.now() - loopStartTimeRef.current;
      if (elapsed >= loopDuration) {
        libraryService.send('VIDEO_LOOPED');
        loopStartTimeRef.current = Date.now();
      }
    }, Math.min(loopDuration / 2, 500)); // Check at half the loop duration or every 500ms

    return () => clearInterval(intervalId);
  }, [gifMetadata, libraryService, path]);

  const imgStyle = useMemo(() => {
    if (scaleMode === 'cover' && coverSize.height && coverSize.width) {
      return { height: coverSize.height, width: coverSize.width };
    }
    if (typeof scaleMode === 'number') {
      return { height: `${scaleMode}%` };
    }
    return {};
  }, [scaleMode, coverSize]);

  const imgSrc = useMemo(() => {
    if (cache) {
      return window.electron.url.format({
        protocol: 'gsm',
        pathname: previewData,
        search: version ? `?v=${version}` : undefined,
      });
    }
    return window.electron.url.format({
      protocol: 'gsm',
      pathname: path,
      search: version ? `?v=${version}` : undefined,
    });
  }, [cache, previewData, path, version]);

  if (error) {
    return <MediaErrorMsg path={path} />;
  }

  return previewData || !cache ? (
    <img
      className={`Image ${scaleMode} ${orientation}`}
      style={imgStyle}
      ref={mediaRef}
      onLoad={(e) => {
        handleLoad && handleLoad(e);
      }}
      onError={() => {
        setError(true);
        console.log('failed to load animated gif');
      }}
      src={imgSrc}
      alt="detail"
    />
  ) : (
    <div className="ThumnailLoader">
      <div className="loading-bar">
        <SkeletonTheme baseColor="#202020" highlightColor="#444">
          <Skeleton />
        </SkeletonTheme>
      </div>
    </div>
  );
}

export const AnimatedGif = React.memo(AnimatedGifComponent, (prevProps, nextProps) => {
  return (
    prevProps.path === nextProps.path &&
    prevProps.scaleMode === nextProps.scaleMode &&
    prevProps.cache === nextProps.cache &&
    prevProps.orientation === nextProps.orientation &&
    prevProps.coverSize?.width === nextProps.coverSize?.width &&
    prevProps.coverSize?.height === nextProps.coverSize?.height &&
    prevProps.version === nextProps.version
  );
});

AnimatedGif.displayName = 'AnimatedGif';
