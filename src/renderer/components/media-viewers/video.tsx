import React, { useEffect, useContext, useState, useRef } from 'react';

import { useSelector } from '@xstate/react';
import { useQuery } from '@tanstack/react-query';
import { GlobalStateContext } from '../../state';
import { ScaleModeOption, clampVolume } from 'settings';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';
import Hls from 'hls.js';
import { mediaUrl, hlsUrl, fetchMediaPreview as platformFetchMediaPreview } from '../../platform';
import { useVisibilityLoader } from '../../hooks/useVisibilityLoader';

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
  version?: number;
  /** Delay loading by ms to prevent loading during fast scroll (0 = no delay) */
  loadDelay?: number;
  useHLS?: boolean;
};

const fetchMediaPreview =
  (
    item: string,
    cache: 'thumbnail_path_1200' | 'thumbnail_path_600' | false,
    timeStamp: number
  ) =>
  async (): Promise<string> => {
    const path = await platformFetchMediaPreview(
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
  version = 0,
  loadDelay = 0,
  useHLS = false,
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

  // Delay loading to prevent loading videos that are quickly scrolled past (list mode)
  // When loadDelay is 0, load immediately (detail view)
  const shouldLoad = useVisibilityLoader(loadDelay);

  const { data, isLoading, isFetched } = useQuery<string, Error>(
    ['media', 'preview', path, cache, startTime, version],
    fetchMediaPreview(path, cache, startTime),
    { enabled: shouldLoad && !!cache }
  );

  const [error, setError] = useState<boolean>(false);
  const prevTimeRef = useRef<number>(0);

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
    prevTimeRef.current = 0;
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
    if (mediaRef && mediaRef.current && settable) {
      if (playing) {
        mediaRef.current.play().catch((e) => {
          console.log('playback error', e);
        });
      } else {
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
        const duration = mediaRef.current.duration;
        const prevTime = prevTimeRef.current;

        // Detect loop: previous time was near the end, current time is near the start
        // Use a threshold of 1 second to account for timeupdate event timing
        if (duration > 0 && prevTime > duration - 1 && currentTime < 1) {
          libraryService.send('VIDEO_LOOPED');
        }

        prevTimeRef.current = currentTime;

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

  const hlsRef = useRef<Hls | null>(null);
  const [hlsFailed, setHlsFailed] = useState(false);
  const [hlsReady, setHlsReady] = useState(false);
  const [hlsProgress, setHlsProgress] = useState<{ status: string; progress: number } | null>(null);

  // Reset HLS state when path changes
  useEffect(() => {
    setHlsFailed(false);
    setHlsReady(false);
    setHlsProgress(null);
  }, [path]);

  // Poll server for HLS generation status.
  // This must NOT depend on mediaRef.current because the progress UI
  // replaces the <video> element, so the ref won't be set until ready.
  const hlsManifestUrl = useRef<string | null>(null);

  useEffect(() => {
    if (!useHLS || hlsFailed || cache || !hlsUrl) return;

    let cancelled = false;
    let pollTimer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      if (cancelled) return;
      try {
        const res = await fetch(hlsUrl(path));
        if (!res.ok || cancelled) return;
        const data = await res.json();

        if (cancelled) return;

        if (data.status === 'ready') {
          hlsManifestUrl.current = data.url;
          setHlsProgress(null);
          setHlsReady(true);
        } else if (data.status === 'processing' || data.status === 'queued') {
          setHlsProgress({ status: data.status, progress: data.progress || 0 });
          pollTimer = setTimeout(poll, 1000);
        } else if (data.status === 'error') {
          console.log('HLS generation error:', data.error);
          setHlsFailed(true);
        }
      } catch {
        if (!cancelled) pollTimer = setTimeout(poll, 2000);
      }
    };

    poll();

    return () => {
      cancelled = true;
      clearTimeout(pollTimer);
    };
  }, [useHLS, hlsFailed, path, cache]);

  // Once HLS is ready and the <video> element is mounted, attach hls.js.
  useEffect(() => {
    if (!hlsReady || !mediaRef?.current || !hlsManifestUrl.current) return;

    const video = mediaRef.current;
    const url = hlsManifestUrl.current;

    if (Hls.isSupported()) {
      const hls = new Hls({ enableWorker: true });
      hlsRef.current = hls;
      hls.loadSource(url);
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch(() => {});
      });
      hls.on(Hls.Events.ERROR, (_event, errData) => {
        if (errData.fatal) {
          console.log('hls.js fatal error:', errData.type, errData.details);
          hls.destroy();
          hlsRef.current = null;
          setHlsFailed(true);
        }
      });
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = url;
    }

    return () => {
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
    };
  }, [hlsReady, mediaRef]);

  // Show HLS processing/loading status.
  // Block the normal <video> render until HLS is ready or has failed.
  const hlsActive = useHLS && !hlsFailed && hlsUrl && !cache;
  if (hlsActive && !hlsReady) {
    const pct = hlsProgress ? Math.round(hlsProgress.progress * 100) : 0;
    const label = !hlsProgress
      ? 'Preparing stream...'
      : hlsProgress.status === 'queued'
      ? 'Queued...'
      : `Processing${pct > 0 ? ` ${pct}%` : '...'}`;
    return (
      <div className="Video" style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        gap: '8px',
        color: '#aaa',
        fontSize: '14px',
      }}>
        <div>{label}</div>
        {hlsProgress && hlsProgress.status === 'processing' && (
          <div style={{
            width: '200px',
            height: '4px',
            background: '#333',
            borderRadius: '2px',
            overflow: 'hidden',
          }}>
            <div style={{
              width: `${pct}%`,
              height: '100%',
              background: '#888',
              transition: 'width 0.5s ease',
            }} />
          </div>
        )}
      </div>
    );
  }

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
    // Wait for visibility delay before loading video
    if (!shouldLoad) {
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
          onError={(err) => {
            console.log('video error:', err.currentTarget?.error?.code);
            setError(true);
          }}
          onDoubleClick={(e) => {
            e.preventDefault();
          }}
          muted={!playSound}
          src={hlsActive && hlsReady ? undefined : mediaUrl(path)}
          controls={showControls}
          controlsList={'nodownload nofullscreen'}
          autoPlay
          loop={!hlsActive}
        />
      </>
    );
  }

  if (!shouldLoad || (isLoading && !isFetched)) {
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
      onLoadedData={(e) => {
        handleLoad && handleLoad(e);
      }}
      onError={(err) => {
        console.log('video error:', err.currentTarget?.error?.code, data);

        setError(true);
      }}
      onDoubleClick={(e) => {
        e.preventDefault();
      }}
      muted={!playSound}
      src={mediaUrl(data || path, version)}
      controls={false}
      controlsList={'nodownload nofullscreen'}
      autoPlay
      loop
    />
  );
}
