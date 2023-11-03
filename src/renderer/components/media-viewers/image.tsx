import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';

import type { ScaleModeOption } from '../../../settings';
import './image.css';
import './sizing.css';
import MediaErrorMsg from './media-error';

type Props = {
  path: string;
  useCache?: boolean;
  scaleMode?: ScaleModeOption;
  coverSize?: { width: number; height: number };
  mediaRef?: React.RefObject<HTMLImageElement>;
  handleLoad?: React.ReactEventHandler<HTMLImageElement>;
  orientation: 'portrait' | 'landscape' | 'unknown';
  cache?: 'thumbnail_path_1200' | 'thumbnail_path_600' | false;
};

const fetchMediaPreview =
  (item: string, cache: 'thumbnail_path_1200' | 'thumbnail_path_600' | false) =>
  async (): Promise<string> => {
    const path = await window.electron.fetchMediaPreview(item, cache);
    return path;
  };

export function Image({
  path,
  scaleMode = 'cover',
  mediaRef,
  coverSize = { width: 0, height: 0 },
  handleLoad,
  cache = false,
  orientation = 'unknown',
}: Props) {
  const [error, setError] = useState<boolean>(false);

  const { data, isError, isLoading } = useQuery<string, Error>(
    ['media', 'preview', path, cache],
    fetchMediaPreview(path, cache)
  );

  // Reset error state if path changes.
  useEffect(() => {
    setError(false);
  }, [path]);
  console.log('error', error);
  if (error) {
    return <MediaErrorMsg path={path} />;
  }

  return data || !cache ? (
    <img
      className={`Image ${scaleMode} ${orientation}`}
      loading="lazy"
      style={
        scaleMode === 'cover' && coverSize.height && coverSize.width
          ? { height: coverSize.height, width: coverSize.width }
          : typeof scaleMode === 'number'
          ? { height: `${scaleMode}%` }
          : {}
      }
      ref={mediaRef}
      onLoad={(e) => {
        handleLoad && handleLoad(e);
      }}
      onError={() => {
        setError(true);
        console.log('failed to load image');
      }}
      src={
        cache
          ? window.electron.url.format({ protocol: 'gsm', pathname: data })
          : window.electron.url.format({ protocol: 'gsm', pathname: path })
      }
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
