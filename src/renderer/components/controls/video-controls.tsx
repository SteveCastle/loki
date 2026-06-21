import React, {
  useContext,
  useCallback,
  useState,
  useRef,
  useEffect,
} from 'react';
import { useSelector } from '@xstate/react';
import { useEventListener, useIsomorphicLayoutEffect } from 'usehooks-ts';
import repeat from '../../../../assets/repeat.svg';
import play from '../../../../assets/play.svg';
import pause from '../../../../assets/pause.svg';
import soundHigh from '../../../../assets/sound-high.svg';
import soundOff from '../../../../assets/sound-off.svg';
import { uniqueId } from 'xstate/lib/utils';
import { GlobalStateContext } from '../../state';
import AudioTrackControls from './audio-track-controls';
import {
  frameStep,
  pixelToTime,
  selectDisplayTime,
  coalescedSeekTarget,
  seekBy,
} from '../../video-frame';
import './video-controls.css';

// --- Helper Functions (mapRange, getLabel, useElementSize - remain the same) ---
function mapRange(
  value: number,
  in_min: number,
  in_max: number,
  out_min: number,
  out_max: number
): number {
  const clampedValue = Math.max(in_min, Math.min(value, in_max));
  return (
    ((clampedValue - in_min) * (out_max - out_min)) / (in_max - in_min) +
    out_min
  );
}

interface Size {
  width: number;
  height: number;
}

function useElementSize<T extends HTMLElement = HTMLDivElement>(): [
  (node: T | null) => void,
  React.RefObject<T>,
  Size
] {
  const [node, setNode] = useState<T | null>(null);
  const ref = useRef<T>(null);

  const setRef = useCallback((newNode: T | null) => {
    setNode(newNode);
    (ref as React.MutableRefObject<T | null>).current = newNode;
  }, []);

  const [size, setSize] = useState<Size>({
    width: 0,
    height: 0,
  });

  const handleSize = useCallback(() => {
    setSize({
      width: node?.offsetWidth || 0,
      height: node?.offsetHeight || 0,
    });
  }, [node?.offsetHeight, node?.offsetWidth]);

  useEventListener('resize', handleSize);

  useIsomorphicLayoutEffect(() => {
    handleSize();
  }, [node?.offsetHeight, node?.offsetWidth]);

  return [setRef, ref, size];
}

function getLabel(currentVideoTimeStamp: number): string {
  const totalSeconds = Math.max(0, Math.floor(currentVideoTimeStamp));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  const hoursString = String(hours).padStart(2, '0');
  const minutesString = String(minutes).padStart(2, '0');
  const secondsString = String(seconds).padStart(2, '0');

  return `${hoursString}:${minutesString}:${secondsString}`;
}

// --- Component ---

interface VideoControlsProps {
  // The media element being controlled. When supplied, scrubbing seeks the
  // element directly (fast, coalesced, no per-frame XState churn). When absent,
  // the component falls back to the XState SET_VIDEO_TIME path.
  mediaRef?: React.RefObject<HTMLMediaElement>;
}

export default function VideoControls({ mediaRef }: VideoControlsProps = {}) {
  const { libraryService } = useContext(GlobalStateContext);
  const {
    actualVideoTime,
    videoLength,
    loopLength,
    playing,
    frameRate,
    playbackRate,
  } = useSelector(libraryService, (state) => state.context.videoPlayer);
  const { volume, playSound } = useSelector(
    libraryService,
    (state: any) => state.context.settings
  );

  const [setProgressBarRef, progressBarRef, { width: progressBarWidth }] =
    useElementSize<HTMLDivElement>();

  const [isDragging, setIsDragging] = useState(false);
  const [showVolumeControl, setShowVolumeControl] = useState(false);
  const volumeContainerRef = useRef<HTMLDivElement>(null);
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [hoverPosition, setHoverPosition] = useState(0);

  // Frame/transport popover: opens after hovering the play button for 3s (long
  // enough that it doesn't pop up during normal play/pause clicks), and stays
  // open while the cursor is over the button or the popover — same hover model
  // as the volume control.
  const [showFrameControls, setShowFrameControls] = useState(false);
  const playContainerRef = useRef<HTMLDivElement>(null);
  const playHoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Refs for smoother scrubbing logic
  const wasPlayingRef = useRef(false); // Store playing state before drag
  const rafRef = useRef<number | null>(null); // Store requestAnimationFrame ID
  const latestMouseXRef = useRef(0); // Store latest mouse position for rAF

  // Optimistic scrub position. While dragging, the thumb/label render from this
  // (cursor-locked, instant) instead of actualVideoTime (which only updates
  // after the seek decodes) — this is what makes scrubbing feel direct.
  const [dragTime, setDragTime] = useState<number | null>(null);
  // Latest requested seek time; the `seeked` handler converges to it so we keep
  // exactly one seek in flight (see coalescedSeekTarget).
  const pendingSeekRef = useRef<number | null>(null);
  // Live mirror of actualVideoTime read inside drag callbacks, so those
  // callbacks (and the global mouse-listener effect) don't re-create every time
  // actualVideoTime ticks mid-drag — which would re-subscribe the listeners.
  const actualVideoTimeRef = useRef(actualVideoTime);
  actualVideoTimeRef.current = actualVideoTime;

  const displayTime = selectDisplayTime(isDragging, dragTime, actualVideoTime);

  // Seek the media element directly with coalescing. Returns false when no
  // element is available so callers can fall back to the XState path.
  const seekElement = useCallback(
    (time: number): boolean => {
      const el = mediaRef?.current;
      if (!el) return false;
      pendingSeekRef.current = time;
      const target = coalescedSeekTarget(time, el.seeking, el.currentTime);
      if (target != null) el.currentTime = target;
      return true;
    },
    [mediaRef]
  );

  // When a coalesced seek completes, immediately seek to the most recent
  // pending target if the cursor has moved on. Active only during a drag.
  useEffect(() => {
    const el = mediaRef?.current;
    if (!el || !isDragging) return;
    const onSeeked = () => {
      const target = coalescedSeekTarget(
        pendingSeekRef.current,
        el.seeking,
        el.currentTime
      );
      if (target != null) el.currentTime = target;
    };
    el.addEventListener('seeked', onSeeked);
    return () => el.removeEventListener('seeked', onSeeked);
  }, [mediaRef, isDragging]);

  // --- Volume Logic ---
  const handleSettingChange = useCallback(
    (key: any, value: any) => {
      libraryService.send('CHANGE_SETTING', { data: { [key]: value } });
    },
    [libraryService]
  );

  const handleVolumeMouseEnter = useCallback(() => {
    if (playSound) {
      setShowVolumeControl(true);
    }
  }, [playSound]);

  const handleVolumeMouseLeave = useCallback(() => {
    // Small delay to allow moving to the volume control
    setTimeout(() => {
      if (!volumeContainerRef.current?.matches(':hover')) {
        setShowVolumeControl(false);
      }
    }, 100);
  }, []);

  const handleVolumeContainerMouseLeave = useCallback(() => {
    setShowVolumeControl(false);
  }, []);

  const handleSoundToggle = useCallback(() => {
    handleSettingChange('playSound', !playSound);
  }, [playSound, handleSettingChange]);

  // --- Frame/transport popover hover logic (3s open delay) ---
  const handlePlayMouseEnter = useCallback(() => {
    if (playHoverTimerRef.current) clearTimeout(playHoverTimerRef.current);
    playHoverTimerRef.current = setTimeout(() => {
      setShowFrameControls(true);
    }, 3000);
  }, []);

  const handlePlayMouseLeave = useCallback(() => {
    if (playHoverTimerRef.current) {
      clearTimeout(playHoverTimerRef.current);
      playHoverTimerRef.current = null;
    }
    // Brief grace period so the cursor can travel from the button into the
    // popover without it closing.
    setTimeout(() => {
      if (!playContainerRef.current?.matches(':hover')) {
        setShowFrameControls(false);
      }
    }, 120);
  }, []);

  const handlePlayContainerMouseLeave = useCallback(() => {
    if (playHoverTimerRef.current) {
      clearTimeout(playHoverTimerRef.current);
      playHoverTimerRef.current = null;
    }
    setShowFrameControls(false);
  }, []);

  // Clear any pending open timer on unmount.
  useEffect(() => {
    return () => {
      if (playHoverTimerRef.current) clearTimeout(playHoverTimerRef.current);
    };
  }, []);

  // --- Scrubbing Logic ---

  const handleProgressHover = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (
        !progressBarRef.current ||
        progressBarWidth <= 0 ||
        videoLength <= 0
      ) {
        setHoverTime(null);
        return;
      }
      const rect = progressBarRef.current.getBoundingClientRect();
      const offsetX = e.clientX - rect.left;

      // Clamp offsetX to be within progress bar bounds
      const clampedOffsetX = Math.max(0, Math.min(offsetX, progressBarWidth));

      const time = mapRange(
        clampedOffsetX,
        0,
        progressBarWidth,
        0,
        videoLength
      );
      setHoverTime(time);
      setHoverPosition(clampedOffsetX);
    },
    [progressBarRef, progressBarWidth, videoLength]
  );

  const handleProgressLeave = useCallback(() => {
    setHoverTime(null);
  }, []);

  // Step exactly one frame. Frame inspection implies a paused video, so pause
  // first if playing, then seek to the fractional frame time. frameStep falls
  // back to 30fps when the detected rate is unknown (see video-frame.ts).
  const stepFrame = useCallback(
    (direction: 1 | -1) => {
      if (videoLength <= 0) return;
      if (playing) {
        libraryService.send('SET_PLAYING_STATE', { playing: false });
      }
      const target = frameStep(
        actualVideoTime,
        frameRate,
        direction,
        videoLength
      );
      // Seek the element directly for an instant step, then sync the machine
      // (and every actualVideoTime consumer) to the new position so the thumb
      // and readout update without waiting on the seek round-trip.
      const el = mediaRef?.current;
      if (el) {
        el.pause();
        el.currentTime = target;
      }
      libraryService.send('SET_VIDEO_TIME', {
        timeStamp: target,
        eventId: uniqueId(),
      });
      libraryService.send('SET_ACTUAL_VIDEO_TIME', {
        timeStamp: target,
        eventId: uniqueId(),
      });
    },
    [videoLength, playing, actualVideoTime, frameRate, libraryService, mediaRef]
  );

  // Skip forward/back by a fixed number of seconds (rewind / fast-forward),
  // clamped to the clip. Seeks the element directly and syncs the machine.
  const SKIP_SECONDS = 10;
  const skip = useCallback(
    (deltaSeconds: number) => {
      if (videoLength <= 0) return;
      const target = seekBy(actualVideoTime, deltaSeconds, videoLength);
      const el = mediaRef?.current;
      if (el) el.currentTime = target;
      libraryService.send('SET_VIDEO_TIME', {
        timeStamp: target,
        eventId: uniqueId(),
      });
      libraryService.send('SET_ACTUAL_VIDEO_TIME', {
        timeStamp: target,
        eventId: uniqueId(),
      });
    },
    [videoLength, actualVideoTime, libraryService, mediaRef]
  );

  // Set the playback speed. Applies to the element immediately and persists in
  // the machine so it survives clip changes within the session.
  const PLAYBACK_SPEEDS = [0.25, 0.5, 1, 1.5, 2];
  const setSpeed = useCallback(
    (rate: number) => {
      const el = mediaRef?.current;
      if (el) el.playbackRate = rate;
      libraryService.send('SET_PLAYBACK_RATE', { playbackRate: rate });
    },
    [libraryService, mediaRef]
  );

  // Arrow keys step one frame while the slider is focused. Scoped to the slider
  // (rather than global) so it never hijacks arrow-key media navigation.
  const handleSliderKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        stepFrame(1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        stepFrame(-1);
      }
    },
    [stepFrame]
  );

  // Function to perform the actual time update (called within rAF)
  const performTimeUpdate = useCallback(() => {
    if (!progressBarRef.current || progressBarWidth <= 0 || videoLength <= 0) {
      rafRef.current = null; // Clear raf ID if prerequisites aren't met
      return;
    }

    const rect = progressBarRef.current.getBoundingClientRect();
    const offsetX = latestMouseXRef.current - rect.left;

    // Fractional seek target — NOT rounded to whole seconds.
    const newTimeStamp = pixelToTime(offsetX, progressBarWidth, videoLength);

    // 1) Optimistic, cursor-locked visual feedback — no seek round-trip.
    setDragTime(newTimeStamp);

    // 2) Move the picture. Prefer a direct, coalesced element seek (fast, no
    //    app-wide re-render). Only when no element ref is available do we fall
    //    back to the XState path (which is what makes scrubbing feel laggy).
    if (!seekElement(newTimeStamp)) {
      if (Math.abs(newTimeStamp - actualVideoTimeRef.current) > 0.001) {
        libraryService.send('SET_VIDEO_TIME', {
          timeStamp: newTimeStamp,
          eventId: uniqueId(),
        });
      }
    }
    rafRef.current = null; // Clear the raf ID after execution
  }, [progressBarRef, progressBarWidth, videoLength, seekElement, libraryService]);

  // Function called by the mousemove listener to request an update frame
  const requestUpdateFrame = useCallback(() => {
    // If there's a pending frame, cancel it
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
    }
    // Request a new frame to perform the update
    rafRef.current = requestAnimationFrame(performTimeUpdate);
  }, [performTimeUpdate]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      if (progressBarWidth <= 0 || videoLength <= 0) return; // Don't start drag if bar invalid

      wasPlayingRef.current = playing; // Store current playing state
      if (playing) {
        // Pause immediately (element + machine). The direct pause avoids the
        // video advancing during the first drag frames while the XState round
        // trip propagates.
        mediaRef?.current?.pause();
        libraryService.send('SET_PLAYING_STATE', { playing: false });
      }
      setIsDragging(true);
      latestMouseXRef.current = e.clientX; // Store initial position
      performTimeUpdate(); // Perform initial update immediately on click
    },
    [
      playing,
      libraryService,
      performTimeUpdate,
      progressBarWidth,
      videoLength,
      mediaRef,
    ]
  );

  // Effect to handle global mouse move and up listeners
  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => {
      // Store the latest mouse position
      latestMouseXRef.current = e.clientX;
      // Request an update frame (rAF will handle throttling)
      requestUpdateFrame();
    };

    const handleGlobalMouseUp = (e: MouseEvent) => {
      // If there's a pending frame, cancel it
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }

      // Final precise position from the release point.
      let finalTime = actualVideoTimeRef.current;
      if (progressBarRef.current && progressBarWidth > 0 && videoLength > 0) {
        const rect = progressBarRef.current.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        finalTime = pixelToTime(offsetX, progressBarWidth, videoLength);
      }

      // Land the element exactly on the release point (covers the case where
      // the last drag frame was coalesced away while a seek was in flight).
      const el = mediaRef?.current;
      if (el) el.currentTime = finalTime;

      // Sync the machine AND actualVideoTime (which the thumb falls back to)
      // before clearing the optimistic dragTime, so the bar doesn't flash back
      // to a stale position on release. SET_ACTUAL_VIDEO_TIME also keeps the
      // transcript / cue / tag-drop consumers in step.
      libraryService.send('SET_VIDEO_TIME', {
        timeStamp: finalTime,
        eventId: uniqueId(),
      });
      libraryService.send('SET_ACTUAL_VIDEO_TIME', {
        timeStamp: finalTime,
        eventId: uniqueId(),
      });

      pendingSeekRef.current = null;
      setIsDragging(false);
      setDragTime(null);

      // Resume playing only if it was playing before dragging started
      if (wasPlayingRef.current) {
        libraryService.send('SET_PLAYING_STATE', { playing: true });
      }

      // Set final loop position on mouseUp (not updated live during drag).
      if (loopLength > 0 && videoLength > 0) {
        libraryService.send('LOOP_VIDEO', {
          loopStartTime: finalTime,
          loopLength,
        });
      }
    };

    if (isDragging) {
      window.addEventListener('mousemove', handleGlobalMouseMove);
      window.addEventListener('mouseup', handleGlobalMouseUp);
      document.body.style.userSelect = 'none'; // Prevent text selection
      document.body.style.cursor = 'grabbing'; // Indicate dragging globally
    }

    // Cleanup function
    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      window.removeEventListener('mouseup', handleGlobalMouseUp);

      // Ensure any pending frame is cancelled on cleanup
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }

      // Restore styles
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
  }, [
    isDragging,
    requestUpdateFrame,
    libraryService,
    mediaRef,
    loopLength,
    progressBarRef,
    progressBarWidth,
    videoLength,
  ]);

  // --- Render (mostly same as before) ---
  return (
    <div className="VideoControls">
      <div className="controls-left">
        <div
          className="playButtonContainer"
          ref={playContainerRef}
          onMouseLeave={handlePlayContainerMouseLeave}
        >
          <button
            className="control-button"
            onClick={() => {
              libraryService.send('SET_PLAYING_STATE', {
                playing: !playing,
              });
            }}
            onMouseEnter={handlePlayMouseEnter}
            onMouseLeave={handlePlayMouseLeave}
            aria-label={playing ? 'Pause' : 'Play'}
          >
            {playing ? (
              <img src={pause} alt="Pause" />
            ) : (
              <img src={play} alt="Play" />
            )}
          </button>
          {showFrameControls && (
            // Stop click/double-click from reaching the detail panel, whose
            // double-click handler collapses panels (switches to list view) —
            // fast clicks here must not trigger it.
            <div
              className="frameControlHover"
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => e.stopPropagation()}
              onDoubleClick={(e) => e.stopPropagation()}
            >
              <div className="popover-row">
                <span className="popover-label">Speed</span>
                {PLAYBACK_SPEEDS.map((rate) => (
                  <button
                    key={rate}
                    className={`popover-button speed-button${
                      playbackRate === rate ? ' selected' : ''
                    }`}
                    onClick={() => setSpeed(rate)}
                    aria-pressed={playbackRate === rate}
                  >
                    {rate}×
                  </button>
                ))}
              </div>
              <div className="popover-row">
                <button
                  className="popover-button transport-button"
                  onClick={() => skip(-SKIP_SECONDS)}
                  disabled={videoLength <= 0}
                  aria-label={`Rewind ${SKIP_SECONDS} seconds`}
                  title={`Rewind ${SKIP_SECONDS}s`}
                >
                  «{SKIP_SECONDS}s
                </button>
                <button
                  className="popover-button transport-button"
                  onClick={() => stepFrame(-1)}
                  disabled={videoLength <= 0}
                  aria-label="Previous frame"
                  title="Previous frame (←)"
                >
                  −1f
                </button>
                <button
                  className="popover-button transport-button"
                  onClick={() => stepFrame(1)}
                  disabled={videoLength <= 0}
                  aria-label="Next frame"
                  title="Next frame (→)"
                >
                  +1f
                </button>
                <button
                  className="popover-button transport-button"
                  onClick={() => skip(SKIP_SECONDS)}
                  disabled={videoLength <= 0}
                  aria-label={`Fast-forward ${SKIP_SECONDS} seconds`}
                  title={`Fast-forward ${SKIP_SECONDS}s`}
                >
                  {SKIP_SECONDS}s»
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="controls-center">
        <div className="progress-container">
          <div
            className="progressBar"
            onMouseDown={handleMouseDown}
            ref={setProgressBarRef}
            style={{ cursor: isDragging ? 'grabbing' : 'pointer' }}
            role="slider" // Accessibility
            aria-label="Video progress"
            aria-valuemin={0}
            aria-valuemax={videoLength || 0}
            aria-valuenow={displayTime || 0}
            aria-valuetext={getLabel(displayTime || 0)}
            tabIndex={0} // Make focusable
            onKeyDown={handleSliderKeyDown}
            onMouseMove={handleProgressHover}
            onMouseLeave={handleProgressLeave}
          >
            {hoverTime !== null && !isDragging && (
              <div
                className="hover-timestamp"
                style={{
                  left: `${hoverPosition}px`,
                  transform: 'translateX(-50%)',
                }}
              >
                {getLabel(hoverTime)}
              </div>
            )}
            <div className="progress-track"></div>
            <div
              style={{
                width:
                  progressBarWidth > 0 && videoLength > 0
                    ? `${mapRange(
                        displayTime,
                        0,
                        videoLength,
                        0,
                        progressBarWidth
                      )}px`
                    : '0px',
                pointerEvents: 'none',
              }}
              className="progress"
            ></div>
            <div
              className="progressThumb"
              style={{
                left:
                  progressBarWidth > 0 && videoLength > 0
                    ? `${mapRange(
                        displayTime,
                        0,
                        videoLength,
                        0,
                        progressBarWidth
                      )}px`
                    : '0px',
                pointerEvents: 'none',
                opacity: isDragging ? 1 : 0.8,
              }}
            ></div>
          </div>
          <div className="timestamp-label">
            <span className="value">{getLabel(displayTime)}</span>
            <span className="total value"> / {getLabel(videoLength)}</span>
          </div>
        </div>
      </div>

      <div className="controls-right">
        <div className="loopButtons">
          <div className="icon">
            <img src={repeat} alt="Repeat Icon" />
          </div>
          {[1, 2, 5, 10].map((length) => (
            <button
              key={length}
              className={[
                'loopButton',
                'control-button',
                loopLength === length ? 'selected' : '',
              ].join(' ')}
              onClick={() => {
                const isCurrentlySelected = loopLength === length;
                const currentTimeForLoop = actualVideoTime;
                libraryService.send('LOOP_VIDEO', {
                  loopStartTime: isCurrentlySelected ? 0 : currentTimeForLoop,
                  loopLength: isCurrentlySelected ? 0 : length,
                });
              }}
              aria-pressed={loopLength === length}
            >
              <span>{`${length}s`}</span>
            </button>
          ))}
        </div>

        <div
          className="volumeButtonContainer"
          ref={volumeContainerRef}
          onMouseLeave={handleVolumeContainerMouseLeave}
        >
          <button
            className="control-button"
            onClick={handleSoundToggle}
            onMouseEnter={handleVolumeMouseEnter}
            onMouseLeave={handleVolumeMouseLeave}
            aria-label={playSound ? 'Mute' : 'Unmute'}
          >
            <img src={playSound ? soundHigh : soundOff} alt="Volume" />
          </button>
          {showVolumeControl && playSound && (
            <div className="volumeControlHover">
              <AudioTrackControls />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
