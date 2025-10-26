import React, { useState, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';

import type { ScaleModeOption } from '../../../settings';
import './image.css';
import './sizing.css';
import MediaErrorMsg from './media-error';
import { over } from 'lodash';

type Props = {
  path: string;
  useCache?: boolean;
  scaleMode?: ScaleModeOption;
  coverSize?: { width: number; height: number };
  mediaRef?: React.RefObject<HTMLImageElement>;
  handleLoad?: React.ReactEventHandler<HTMLImageElement>;
  orientation: 'portrait' | 'landscape' | 'unknown';
  cache?: 'thumbnail_path_1200' | 'thumbnail_path_600' | false;
  overRideCache?: boolean;
  version?: number;
};

const fetchMediaPreview =
  (item: string, cache: 'thumbnail_path_1200' | 'thumbnail_path_600' | false) =>
  async (): Promise<string> => {
    const path = await window.electron.fetchMediaPreview(item, cache);
    return path;
  };

function ImageComponent({
  path,
  scaleMode = 'cover',
  mediaRef,
  coverSize = { width: 0, height: 0 },
  handleLoad,
  cache = false,
  orientation = 'unknown',
  overRideCache = false,
  version = 0,
}: Props) {
  const [error, setError] = useState<boolean>(false);
  const { data } = useQuery<string, Error>(
    ['media', 'preview', path, cache, version],
    fetchMediaPreview(path, cache)
  );

  // Reset error state if path changes.
  useEffect(() => {
    setError(false);
  }, [path]);

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
    if (cache && !overRideCache) {
      return window.electron.url.format({
        protocol: 'gsm',
        pathname: data,
        search: version ? `?v=${version}` : undefined,
      });
    }
    return window.electron.url.format({
      protocol: 'gsm',
      pathname: path,
      search: version ? `?v=${version}` : undefined,
    });
  }, [cache, overRideCache, data, path, version]);

  if (error) {
    return <MediaErrorMsg path={path} />;
  }

  return data || !cache ? (
    <img
      className={`Image ${scaleMode} ${orientation}`}
      style={imgStyle}
      ref={mediaRef}
      onLoad={(e) => {
        handleLoad && handleLoad(e);
      }}
      onError={() => {
        setError(true);
        console.log('failed to load image');
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

export const Image = React.memo(ImageComponent, (prevProps, nextProps) => {
  return (
    prevProps.path === nextProps.path &&
    prevProps.scaleMode === nextProps.scaleMode &&
    prevProps.cache === nextProps.cache &&
    prevProps.orientation === nextProps.orientation &&
    prevProps.overRideCache === nextProps.overRideCache &&
    prevProps.coverSize?.width === nextProps.coverSize?.width &&
    prevProps.coverSize?.height === nextProps.coverSize?.height
  );
});

Image.displayName = 'Image';
