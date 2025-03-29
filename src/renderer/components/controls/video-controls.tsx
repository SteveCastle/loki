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
import { uniqueId } from 'xstate/lib/utils';
import { GlobalStateContext } from '../../state';
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
    // @ts-ignore
    ref.current = newNode;
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

  useEventListener(
    'resize',
    handleSize,
    typeof window !== 'undefined' ? window : undefined
  );

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

export default function VideoControls() {
  const { libraryService } = useContext(GlobalStateContext);
  const { actualVideoTime, videoLength, loopLength, playing } = useSelector(
    libraryService,
    (state) => state.context.videoPlayer
  );

  const [setProgressBarRef, progressBarRef, { width: progressBarWidth }] =
    useElementSize<HTMLDivElement>();

  const [isDragging, setIsDragging] = useState(false);
  // Refs for smoother scrubbing logic
  const wasPlayingRef = useRef(false); // Store playing state before drag
  const rafRef = useRef<number | null>(null); // Store requestAnimationFrame ID
  const latestMouseXRef = useRef(0); // Store latest mouse position for rAF

  // --- Scrubbing Logic ---

  // Function to perform the actual time update (called within rAF)
  const performTimeUpdate = useCallback(() => {
    if (!progressBarRef.current || progressBarWidth <= 0 || videoLength <= 0) {
      rafRef.current = null; // Clear raf ID if prerequisites aren't met
      return;
    }

    const rect = progressBarRef.current.getBoundingClientRect();
    const offsetX = latestMouseXRef.current - rect.left;

    const newTimeStamp = Math.round(
      mapRange(offsetX, 0, progressBarWidth, 0, videoLength)
    );

    // Only send update if time actually changed noticeably to prevent flooding
    // Allow small threshold for smoother seeking start/end
    if (Math.abs(newTimeStamp - actualVideoTime) > 0.05) {
      libraryService.send('SET_VIDEO_TIME', {
        timeStamp: newTimeStamp,
        eventId: uniqueId(), // Consider if uniqueId is needed here
      });
      // Don't update loop during drag for performance, maybe only on mouseup?
      // If live loop update is needed, uncomment below, but test performance.
      /*
            if (loopLength > 0) {
              libraryService.send('LOOP_VIDEO', {
                loopStartTime: newTimeStamp,
                loopLength,
              });
            }
            */
    }
    rafRef.current = null; // Clear the raf ID after execution
  }, [
    progressBarRef,
    progressBarWidth,
    videoLength,
    libraryService,
    actualVideoTime,
    // loopLength // Add if uncommenting loop update
  ]);

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
        // Pause the video only if it was playing
        libraryService.send('SET_PLAYING_STATE', { playing: false });
      }
      setIsDragging(true);
      latestMouseXRef.current = e.clientX; // Store initial position
      performTimeUpdate(); // Perform initial update immediately on click
    },
    [playing, libraryService, performTimeUpdate, progressBarWidth, videoLength] // Add deps
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

      setIsDragging(false);

      // Update one last time on mouseUp to ensure final position is set
      // Use performTimeUpdate directly to ensure it runs
      latestMouseXRef.current = e.clientX;
      performTimeUpdate();

      // Resume playing only if it was playing before dragging started
      if (wasPlayingRef.current) {
        libraryService.send('SET_PLAYING_STATE', { playing: true });
      }

      // Optional: Set final loop position on mouseUp if not done during drag
      if (
        loopLength > 0 &&
        progressBarRef.current &&
        progressBarWidth > 0 &&
        videoLength > 0
      ) {
        const rect = progressBarRef.current.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        const finalTimeStamp = Math.round(
          mapRange(offsetX, 0, progressBarWidth, 0, videoLength)
        );
        libraryService.send('LOOP_VIDEO', {
          loopStartTime: finalTimeStamp,
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
    performTimeUpdate,
    loopLength,
    progressBarRef,
    progressBarWidth,
    videoLength,
  ]); // Add performTimeUpdate and others to deps

  // --- Render (mostly same as before) ---
  return (
    <div className="VideoControls">
      <div
        className="progressBar"
        onMouseDown={handleMouseDown}
        ref={setProgressBarRef}
        style={{ cursor: isDragging ? 'grabbing' : 'pointer' }}
        role="slider" // Accessibility
        aria-label="Video progress"
        aria-valuemin={0}
        aria-valuemax={videoLength || 0}
        aria-valuenow={actualVideoTime || 0}
        aria-valuetext={getLabel(actualVideoTime || 0)}
        tabIndex={0} // Make focusable
        // Add keyboard controls maybe? (future enhancement)
      >
        <div className="label">
          <span className="value">{getLabel(actualVideoTime)}</span>
          <span className="total value"> / {getLabel(videoLength)}</span>
        </div>
        <div
          style={{
            width:
              progressBarWidth > 0 && videoLength > 0
                ? `${mapRange(
                    actualVideoTime,
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
                    actualVideoTime,
                    0,
                    videoLength,
                    0,
                    progressBarWidth
                  )}px`
                : '0px',
            pointerEvents: 'none',
            opacity: isDragging ? 1 : 0.8, // Make thumb more visible during drag
          }}
        ></div>
      </div>

      <div className="playerButtons">
        <button
          onClick={() => {
            libraryService.send('SET_PLAYING_STATE', {
              playing: !playing,
            });
          }}
          aria-label={playing ? 'Pause' : 'Play'}
        >
          {playing ? (
            <img src={pause} alt="Pause" />
          ) : (
            <img src={play} alt="Play" />
          )}
        </button>
      </div>

      <div className="loopButtons">
        <div className="icon">
          <img src={repeat} alt="Repeat Icon" />
        </div>
        {[1, 2, 5, 10].map((length) => (
          <button
            key={length}
            className={[
              'loopButton',
              loopLength === length ? 'selected' : '',
            ].join(' ')}
            onClick={() => {
              const isCurrentlySelected = loopLength === length;
              // Calculate the loop start time based on the *current* video time,
              // not potentially stale time during a drag.
              const currentTimeForLoop = actualVideoTime;
              libraryService.send('LOOP_VIDEO', {
                loopStartTime: isCurrentlySelected ? 0 : currentTimeForLoop,
                loopLength: isCurrentlySelected ? 0 : length,
              });
            }}
            aria-pressed={loopLength === length}
          >
            <span>{`${length}`}</span>
            <span className="units">sec</span>
          </button>
        ))}
      </div>
    </div>
  );
}
