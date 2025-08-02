import React, { useEffect, useContext, useState } from 'react';

import { useSelector } from '@xstate/react';
import { useQuery } from '@tanstack/react-query';
import { GlobalStateContext } from '../../state';
import { ScaleModeOption, clampVolume } from 'settings';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';

import './video.css';
import './sizing.css';
import { Image } from './image';

type Props = {
  path: string;
  scaleMode?: ScaleModeOption;
  settable?: boolean;
  coverSize?: { width: number; height: number };
  playSound?: boolean;
  volume?: number;
  handleLoad?: React.ReactEventHandler<HTMLImageElement | HTMLVideoElement>;
  showControls?: boolean;
  mediaRef?: React.RefObject<HTMLVideoElement>;
  initialTimestamp?: number;
  startTime?: number;
  orientation: 'portrait' | 'landscape' | 'unknown';
  onTimestampChange?: (timestamp: number) => void;
  cache?: 'thumbnail_path_1200' | 'thumbnail_path_600' | false;
};

const fetchMediaPreview =
  (
    item: string,
    cache: 'thumbnail_path_1200' | 'thumbnail_path_600' | false,
    timeStamp: number
  ) =>
  async (): Promise<string> => {
    const path = await window.electron.fetchMediaPreview(
      item,
      cache,
      timeStamp
    );
    return path;
  };

// return true if the input is a valid value for currentTime
function validateCurrentTime(currentTime: number) {
  return !isNaN(currentTime) && currentTime >= 0 && currentTime !== Infinity;
}

export function Video({
  path,
  settable = false,
  playSound = false,
  volume = 1.0,
  scaleMode,
  handleLoad,
  mediaRef,
  showControls = false,
  initialTimestamp = 0,
  startTime = 0,
  coverSize = { width: 0, height: 0 },
  orientation = 'unknown',
  onTimestampChange = () => {
    return;
  },
  cache = false,
}: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const { timeStamp, loopLength, loopStartTime, playing } = useSelector(
    libraryService,
    (state) => state.context.videoPlayer
  );

  const eventId = useSelector(
    libraryService,
    (state) => state.context.videoPlayer.eventId
  );

  const { data, isLoading } = useQuery<string, Error>(
    ['media', 'preview', path, cache, startTime],
    fetchMediaPreview(path, cache, startTime)
  );

  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    if (mediaRef && mediaRef.current && settable) {
      // if currentTime is greater than loopLength plus the loopStartTime, then we need to reset the currentTime to the loopStartTime
      if (
        loopLength > 0 &&
        mediaRef.current.currentTime > loopLength + loopStartTime
      ) {
        mediaRef.current.currentTime = loopStartTime;
      }
    }
  }, [mediaRef?.current?.currentTime, loopLength, loopStartTime]);

  useEffect(() => {
    setError(false);
  }, [path]);
  useEffect(() => {
    if (mediaRef && mediaRef.current && initialTimestamp) {
      mediaRef.current.currentTime = initialTimestamp;
    }
    if (mediaRef && mediaRef.current && startTime) {
      mediaRef.current.currentTime = startTime;
    }
  }, [initialTimestamp, startTime]);

  useEffect(() => {
    if (mediaRef && mediaRef.current && settable) {
      mediaRef.current.currentTime = timeStamp;
    }
  }, [timeStamp, eventId, settable]);

  // if playing is true, play the video, if its false, pause the video
  useEffect(() => {
    console.log('Video playing state changed:', playing, 'settable:', settable);
    if (mediaRef && mediaRef.current && settable) {
      if (playing) {
        console.log('Playing video...');
        mediaRef.current.play().catch((e) => {
          console.log('playback error', e);
        });
      } else {
        console.log('Pausing video...');
        mediaRef.current.pause();
      }
    }
  }, [playing, settable]);

  useEffect(() => {
    if (!mediaRef || !mediaRef.current) return;
    const video = mediaRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      if (mediaRef && mediaRef.current && settable) {
        const currentTime = mediaRef.current.currentTime;
        libraryService.send('SET_ACTUAL_VIDEO_TIME', {
          timeStamp: currentTime,
          eventId,
        });
        if (onTimestampChange) {
          onTimestampChange(currentTime);
        }
      }
    };

    video.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [onTimestampChange]);

  // Apply volume setting to video element
  useEffect(() => {
    if (mediaRef && mediaRef.current) {
      mediaRef.current.volume = clampVolume(volume);
    }
  }, [volume]);

  if (error) {
    console.log('video error:', error);
    return (
      <Image
        path={path}
        scaleMode={scaleMode}
        coverSize={coverSize}
        handleLoad={handleLoad}
        orientation={orientation}
        cache={cache}
        overRideCache={true}
      />
    );
  }

  if (!cache) {
    return (
      <>
        <video
          style={
            scaleMode === 'cover' && coverSize.height && coverSize.width
              ? { height: coverSize.height, width: coverSize.width }
              : typeof scaleMode === 'number'
              ? { height: `${scaleMode}%` }
              : {}
          }
          ref={mediaRef}
          className={`Video ${scaleMode} ${orientation}`}
          onLoadedData={(e) => {
            if (
              mediaRef &&
              mediaRef.current &&
              initialTimestamp &&
              !startTime
            ) {
              mediaRef.current.currentTime = validateCurrentTime(
                e.currentTarget.duration * initialTimestamp
              )
                ? e.currentTarget.duration * initialTimestamp
                : 0;
            }
            if (mediaRef && mediaRef.current && startTime) {
              mediaRef.current.currentTime = startTime;
            }

            handleLoad && handleLoad(e);
          }}
          onError={() => {
            setError(true);
          }}
          onDoubleClick={(e) => {
            e.preventDefault();
          }}
          muted={!playSound}
          src={window.electron.url.format({ protocol: 'gsm', pathname: path })}
          controls={showControls}
          controlsList={'nodownload nofullscreen'}
          autoPlay
          loop
        />
      </>
    );
  }

  if (isLoading || !data) {
    return (
      <div className="ThumnailLoader">
        <div className="loading-bar">
          <SkeletonTheme baseColor="#202020" highlightColor="#444">
            <Skeleton />
          </SkeletonTheme>
        </div>
      </div>
    );
  }

  return (
    <video
      key={`${path}-${initialTimestamp}-${startTime}`}
      style={
        scaleMode === 'cover' && coverSize.height && coverSize.width
          ? { height: coverSize.height, width: coverSize.width }
          : typeof scaleMode === 'number'
          ? { height: `${scaleMode}%` }
          : {}
      }
      ref={mediaRef}
      className={`Video ${scaleMode} ${orientation}`}
      onError={() => {
        setError(true);
      }}
      onDoubleClick={(e) => {
        e.preventDefault();
      }}
      muted={!playSound}
      src={window.electron.url.format({ protocol: 'gsm', pathname: data })}
      controls={false}
      controlsList={'nodownload nofullscreen'}
      autoPlay
      loop
    />
  );
}
