import React, { useEffect, useContext, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { ScaleModeOption, clampVolume } from 'settings';
import './audio.css';
import './sizing.css';
import MediaErrorMsg from './media-error';

type Props = {
  path: string;
  scaleMode?: ScaleModeOption;
  settable?: boolean;
  coverSize?: { width: number; height: number };
  playSound?: boolean;
  volume?: number;
  handleLoad?: React.ReactEventHandler<HTMLAudioElement>;
  showControls?: boolean;
  mediaRef?: React.RefObject<HTMLAudioElement>;
  initialTimestamp?: number;
  startTime?: number;
  orientation: 'portrait' | 'landscape' | 'unknown';
  onTimestampChange?: (timestamp: number) => void;
  cache?: 'thumbnail_path_1200' | 'thumbnail_path_600' | false;
};

// return true if the input is a valid value for currentTime
function validateCurrentTime(currentTime: number) {
  return !isNaN(currentTime) && currentTime >= 0 && currentTime !== Infinity;
}

export function Audio({
  path,
  settable = false,
  playSound = true,
  volume = 1.0,
  handleLoad,
  mediaRef,
  initialTimestamp = 0,
  startTime = 0,
  onTimestampChange = () => {
    return;
  },
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

  const [error, setError] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [showPlayButton, setShowPlayButton] = useState<boolean>(!settable);

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

  // if playing is true, play the audio, if its false, pause the audio
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
    const audio = mediaRef.current;
    if (!audio) return;

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

    const handlePlay = () => {
      setIsPlaying(true);
      // In list view, keep showing the button (it will show "Pause" when playing)
      if (!settable) {
        setShowPlayButton(true);
      } else {
        setShowPlayButton(false);
      }
    };

    const handlePause = () => {
      setIsPlaying(false);
      // In list view, always show the play button (it will show "Play" when paused)
      if (!settable) {
        setShowPlayButton(true);
      }
    };

    const handleEnded = () => {
      setIsPlaying(false);
      // In list view, always show the play button (it will show "Play" when ended)
      if (!settable) {
        setShowPlayButton(true);
      }
    };

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [onTimestampChange, settable]);

  // Apply volume setting to audio element
  useEffect(() => {
    if (mediaRef && mediaRef.current) {
      mediaRef.current.volume = clampVolume(volume);
    }
  }, [volume]);

  const handlePlayButtonClick = () => {
    if (mediaRef && mediaRef.current) {
      if (isPlaying) {
        mediaRef.current.pause();
      } else {
        mediaRef.current.play().catch((e) => {
          console.log('playback error', e);
        });
      }
    }
  };

  if (error) {
    console.log('audio error:', error);
    return <MediaErrorMsg path={path} />;
  }

  // Always show the audio player with icon - no thumbnail generation for audio files
  return (
    <div className="AudioPlayer">
      <div className="audio-info">
        <div className={`audio-icon ${isPlaying ? 'playing' : ''}`}>🎵</div>
        <div className="audio-title">
          {path.split('/').pop()?.split('\\').pop() || 'Audio File'}
        </div>
      </div>
      {showPlayButton && (
        <button className="audio-play-button" onClick={handlePlayButtonClick}>
          <div className="play-icon">{isPlaying ? '⏸️' : '▶️'}</div>
          <span>{isPlaying ? 'Pause' : 'Play'}</span>
        </button>
      )}
      <audio
        ref={mediaRef}
        className="Audio"
        onLoadedData={(e) => {
          if (mediaRef && mediaRef.current && initialTimestamp && !startTime) {
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
        muted={!playSound}
        src={window.electron.url.format({ protocol: 'gsm', pathname: path })}
        autoPlay={settable}
        loop={settable}
      />
    </div>
  );
}
