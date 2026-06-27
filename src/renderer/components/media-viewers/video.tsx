import React, { useEffect, useContext, useState, useRef } from 'react';

import { useSelector } from '@xstate/react';
import { useQuery } from '@tanstack/react-query';
import { GlobalStateContext } from '../../state';
import { ScaleModeOption, clampVolume } from 'settings';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';
import Hls from 'hls.js';
import { mediaUrl, hlsUrl, fetchMediaPreview as platformFetchMediaPreview, findSubtitle } from '../../platform';
import { toVttString, vttBlobUrl } from './subtitle-loader';
import { useVisibilityLoader } from '../../hooks/useVisibilityLoader';
import { estimateFrameRate } from '../../video-frame';

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
  async (): Promise<string | null> => {
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
  const { timeStamp, loopLength, loopStartTime, playing, playbackRate } =
    useSelector(libraryService, (state) => state.context.videoPlayer);

  const eventId = useSelector(
    libraryService,
    (state) => state.context.videoPlayer.eventId
  );

  const availableSubtitle = useSelector(
    libraryService,
    (state) => state.context.availableSubtitle
  );
  const subtitlesEnabled = useSelector(
    libraryService,
    (state) => state.context.settings.subtitlesEnabled
  );
  const selectedAudioTrackIndex = useSelector(
    libraryService,
    (state) => state.context.selectedAudioTrackIndex
  );

  // Delay loading to prevent loading videos that are quickly scrolled past (list mode)
  // When loadDelay is 0, load immediately (detail view)
  const shouldLoad = useVisibilityLoader(loadDelay);

  const { data, isLoading, isFetched, isError } = useQuery<string | null, Error>(
    ['media', 'preview', path, cache, startTime, version],
    fetchMediaPreview(path, cache, startTime),
    {
      enabled: shouldLoad && !!cache,
      retry: 3,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 10000),
    }
  );

  const [error, setError] = useState<boolean>(false);
  const prevTimeRef = useRef<number>(0);
  const trackRef = useRef<HTMLTrackElement>(null);

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

  // Detect the video's frame rate from requestVideoFrameCallback cadence so the
  // controls can step a single frame. Runs only for the controlling (settable)
  // player. Resets to 0 (unknown) on path change, then dispatches once a stable
  // estimate emerges. If rVFC is unavailable or no stable estimate is reached,
  // frameRate stays 0 and the controls fall back to a 30fps step.
  useEffect(() => {
    const video = mediaRef?.current;
    if (!video || !settable) return;

    // Clear any stale frame rate carried over from the previous clip.
    libraryService.send('SET_VIDEO_FRAME_RATE', { frameRate: 0 });

    const rvfc = (video as any).requestVideoFrameCallback?.bind(video) as
      | ((cb: (now: number, metadata: any) => void) => number)
      | undefined;
    const cancelRvfc = (video as any).cancelVideoFrameCallback?.bind(video) as
      | ((handle: number) => void)
      | undefined;
    if (!rvfc) return; // Unsupported — leave frameRate 0.

    const samples: number[] = [];
    let handle: number | null = null;
    let done = false;

    const onFrame = (_now: number, metadata: any) => {
      if (done) return;
      if (typeof metadata?.mediaTime === 'number') {
        samples.push(metadata.mediaTime);
      }
      if (samples.length >= 8) {
        const fps = estimateFrameRate(samples);
        if (fps > 0) {
          libraryService.send('SET_VIDEO_FRAME_RATE', { frameRate: fps });
          done = true;
          return;
        }
        // Heavy looping/seeking produced only unusable deltas — keep a recent
        // window and keep sampling rather than growing unbounded.
        if (samples.length > 64) samples.splice(0, samples.length - 8);
      }
      handle = rvfc(onFrame);
    };
    handle = rvfc(onFrame);

    return () => {
      done = true;
      if (handle != null && cancelRvfc) cancelRvfc(handle);
    };
  }, [path, settable, libraryService]);

  useEffect(() => {
    const video = mediaRef?.current;
    if (!video) return;
    const handleMetadata = () => {
      const list = (video as any).audioTracks as
        | { length: number; [i: number]: any }
        | undefined;
      if (!list || typeof list.length !== 'number') {
        libraryService.send({ type: 'SET_AVAILABLE_AUDIO_TRACKS', tracks: [] });
        return;
      }
      const tracks: Array<{ id: string; label: string; language: string }> = [];
      for (let i = 0; i < list.length; i++) {
        const t = list[i];
        tracks.push({
          id: String(t.id ?? i),
          label: typeof t.label === 'string' && t.label ? t.label : `Track ${i + 1}`,
          language: typeof t.language === 'string' ? t.language : '',
        });
      }
      libraryService.send({ type: 'SET_AVAILABLE_AUDIO_TRACKS', tracks });
    };
    video.addEventListener('loadedmetadata', handleMetadata);
    // Some browsers fire only `loadeddata` reliably for already-cached files.
    video.addEventListener('loadeddata', handleMetadata);
    return () => {
      video.removeEventListener('loadedmetadata', handleMetadata);
      video.removeEventListener('loadeddata', handleMetadata);
    };
  }, [path, libraryService]);

  useEffect(() => {
    const video = mediaRef?.current;
    if (!video) return;
    const list = (video as any).audioTracks as
      | { length: number; [i: number]: { enabled: boolean } }
      | undefined;
    if (!list || list.length < 2) return;
    for (let i = 0; i < list.length; i++) {
      list[i].enabled = i === selectedAudioTrackIndex;
    }
  }, [selectedAudioTrackIndex]);

  useEffect(() => {
    let revoked: string | null = null;
    let cancelled = false;
    findSubtitle(path)
      .then((sidecar) => {
        if (cancelled) return;
        if (!sidecar) {
          libraryService.send({ type: 'SET_AVAILABLE_SUBTITLE', subtitle: null });
          return;
        }
        const vtt = toVttString(sidecar.content, sidecar.ext);
        const url = vttBlobUrl(vtt);
        revoked = url;
        libraryService.send({
          type: 'SET_AVAILABLE_SUBTITLE',
          subtitle: { blobUrl: url, label: sidecar.ext.toUpperCase() },
        });
      })
      .catch(() => {
        if (cancelled) return;
        libraryService.send({ type: 'SET_AVAILABLE_SUBTITLE', subtitle: null });
      });
    return () => {
      cancelled = true;
      if (revoked) URL.revokeObjectURL(revoked);
      libraryService.send({ type: 'SET_AVAILABLE_SUBTITLE', subtitle: null });
    };
  }, [path, libraryService]);

  useEffect(() => {
    const trackEl = trackRef.current;
    if (!trackEl || !trackEl.track) return;
    trackEl.track.mode = subtitlesEnabled ? 'showing' : 'hidden';
  }, [subtitlesEnabled, availableSubtitle]);

  // Constrain each cue's width via the WebVTT `size` property so long
  // captions wrap to a more readable line length. CSS can't reliably
  // constrain ::cue width — Chromium positions the per-cue display
  // elements absolutely and largely ignores parent padding for them.
  // Setting cue.size = 70 caps the cue at 70% of the video width
  // centered (default position=50, align=center), yielding ~35-45 chars
  // per line at typical caption font sizes.
  useEffect(() => {
    const trackEl = trackRef.current;
    if (!trackEl) return;
    const apply = () => {
      const cues = trackEl.track?.cues;
      if (!cues) return;
      for (let i = 0; i < cues.length; i++) {
        const cue = cues[i] as VTTCue;
        if (typeof cue.size === 'number') cue.size = 70;
      }
    };
    // Cues may already be loaded (track.readyState === 2) or still
    // loading — handle both.
    apply();
    trackEl.addEventListener('load', apply);
    return () => trackEl.removeEventListener('load', apply);
  }, [availableSubtitle]);

  // Apply volume setting to video element
  useEffect(() => {
    if (mediaRef && mediaRef.current) {
      mediaRef.current.volume = clampVolume(volume);
    }
  }, [volume]);

  // Apply the playback-speed setting. Keyed on `path` too so it re-applies when
  // a new clip mounts (a fresh element defaults back to 1x).
  useEffect(() => {
    if (mediaRef && mediaRef.current && settable) {
      mediaRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate, settable, path]);

  const hlsRef = useRef<Hls | null>(null);
  const [hlsFailed, setHlsFailed] = useState(false);
  const [hlsReady, setHlsReady] = useState(false);
  const [hlsGenerating, setHlsGenerating] = useState(false);
  const [hlsProgress, setHlsProgress] = useState<{ status: string; progress: number } | null>(null);
  const [hlsError, setHlsError] = useState<string | null>(null);
  const hlsManifestUrl = useRef<string | null>(null);
  const hlsPollCancel = useRef<(() => void) | null>(null);

  // Reset HLS state when path changes.
  useEffect(() => {
    hlsPollCancel.current?.();
    setHlsFailed(false);
    setHlsReady(false);
    setHlsGenerating(false);
    setHlsProgress(null);
    setHlsError(null);
    hlsManifestUrl.current = null;
  }, [path]);

  // Check if stream is already cached on mount (no generation triggered).
  const hlsActive = useHLS && !hlsFailed && hlsUrl && !cache;
  useEffect(() => {
    if (!hlsActive || hlsGenerating || hlsReady) return;

    let cancelled = false;
    // One-shot check: if server says "ready", skip the button.
    // Use check=true so the server does NOT start encoding — only reports cached/inflight status.
    fetch(hlsUrl!(path) + '&check=true', { method: 'GET' })
      .then((res) => res.ok ? res.json() : null)
      .then((data) => {
        if (cancelled || !data) return;
        if (data.status === 'ready') {
          hlsManifestUrl.current = data.url;
          setHlsReady(true);
        } else if (data.status === 'processing' || data.status === 'queued') {
          // Already generating (started by a previous session/request).
          setHlsGenerating(true);
          setHlsProgress({ status: data.status, progress: data.progress || 0 });
          startPolling();
        }
      })
      .catch(() => {});

    return () => { cancelled = true; };
  }, [hlsActive, path]);

  // Start polling for generation progress.
  const startPolling = () => {
    hlsPollCancel.current?.();
    let cancelled = false;
    let pollTimer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      if (cancelled) return;
      try {
        const res = await fetch(hlsUrl!(path));
        if (!res.ok || cancelled) return;
        const data = await res.json();
        if (cancelled) return;

        if (data.status === 'ready') {
          hlsManifestUrl.current = data.url;
          setHlsProgress(null);
          setHlsGenerating(false);
          setHlsReady(true);
        } else if (data.status === 'processing' || data.status === 'queued') {
          setHlsProgress({ status: data.status, progress: data.progress || 0 });
          pollTimer = setTimeout(poll, 1000);
        } else if (data.status === 'error') {
          setHlsGenerating(false);
          setHlsError(data.error || 'Generation failed');
        }
      } catch {
        if (!cancelled) pollTimer = setTimeout(poll, 2000);
      }
    };

    poll();
    hlsPollCancel.current = () => {
      cancelled = true;
      clearTimeout(pollTimer);
    };
  };

  // User clicks "Generate Stream" — fire the first request to kick off generation, then poll.
  const handleGenerateHLS = () => {
    if (!hlsUrl) return;
    setHlsGenerating(true);
    setHlsError(null);
    setHlsProgress(null);

    fetch(hlsUrl(path))
      .then((res) => res.ok ? res.json() : null)
      .then((data) => {
        if (!data) return;
        if (data.status === 'ready') {
          hlsManifestUrl.current = data.url;
          setHlsGenerating(false);
          setHlsReady(true);
        } else {
          setHlsProgress({ status: data.status, progress: data.progress || 0 });
          startPolling();
        }
      })
      .catch(() => {
        setHlsGenerating(false);
        setHlsError('Network error');
      });
  };

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

  // HLS UI states — shown instead of the <video> element until ready.
  if (hlsActive && !hlsReady) {
    // Generating: show progress bar.
    if (hlsGenerating) {
      const pct = hlsProgress ? Math.round(hlsProgress.progress * 100) : 0;
      const label = !hlsProgress
        ? 'Starting...'
        : hlsProgress.status === 'queued'
        ? 'Queued...'
        : `Processing${pct > 0 ? ` ${pct}%` : '...'}`;
      return (
        <div className="Video" style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexDirection: 'column', gap: '8px', color: '#aaa', fontSize: '14px',
        }}>
          <div>{label}</div>
          {hlsProgress && hlsProgress.status === 'processing' && (
            <div style={{
              width: '200px', height: '4px', background: '#333',
              borderRadius: '2px', overflow: 'hidden',
            }}>
              <div style={{
                width: `${pct}%`, height: '100%', background: '#888',
                transition: 'width 0.5s ease',
              }} />
            </div>
          )}
        </div>
      );
    }

    // Error: show message with retry button.
    if (hlsError) {
      return (
        <div className="Video" style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexDirection: 'column', gap: '8px', color: '#aaa', fontSize: '14px',
        }}>
          <div>Stream generation failed</div>
          <div style={{ fontSize: '12px', color: '#777' }}>{hlsError}</div>
          <button
            onClick={handleGenerateHLS}
            style={{
              marginTop: '4px', padding: '6px 16px', background: '#333',
              color: '#ccc', border: '1px solid #555', borderRadius: '4px',
              cursor: 'pointer', fontSize: '13px',
            }}
          >
            Retry
          </button>
        </div>
      );
    }

    // Idle: show generate button.
    return (
      <div className="Video" style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexDirection: 'column', gap: '8px', color: '#aaa', fontSize: '14px',
      }}>
        <button
          onClick={handleGenerateHLS}
          style={{
            padding: '8px 20px', background: '#333', color: '#ccc',
            border: '1px solid #555', borderRadius: '4px',
            cursor: 'pointer', fontSize: '13px',
          }}
        >
          Generate HLS Stream
        </button>
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
        overRideCache={!cache}
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
          src={hlsActive && hlsReady ? undefined : mediaUrl(path, version)}
          controls={showControls}
          controlsList={'nodownload nofullscreen'}
          autoPlay
          loop={!hlsActive}
        >
          {availableSubtitle && (
            <track
              ref={trackRef}
              kind="subtitles"
              src={availableSubtitle.blobUrl}
              label={availableSubtitle.label}
              default={subtitlesEnabled}
            />
          )}
        </video>
      </>
    );
  }

  // Thumbnail generation failed after retries — fall back to static image display
  if (cache && isError && isFetched) {
    return (
      <Image
        path={path}
        scaleMode={scaleMode}
        coverSize={coverSize}
        handleLoad={handleLoad}
        orientation={orientation}
        cache={false}
        overRideCache={true}
      />
    );
  }

  if (!shouldLoad || (isLoading && !isFetched) || !data) {
    // In cached mode: show skeleton while loading, while fetching,
    // or when thumbnail is not yet available — never fall back to
    // the original file path which may be on slow network storage.
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
      src={mediaUrl(data, version)}
      controls={false}
      controlsList={'nodownload nofullscreen'}
      autoPlay
      loop
    />
  );
}
