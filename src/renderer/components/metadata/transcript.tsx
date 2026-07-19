import { useRef, useContext, useState, useMemo, useEffect } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { transcript as transcriptApi } from '../../platform';

import { GlobalStateContext } from '../../state';
import filter from '../../filter';
import { VttCue } from 'main/parse-vtt';
import { Cue, convertVTTTimestampToSeconds } from './cue';
import GenerateTranscript from './generate-transcript';
import './transcript.css';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';

/** Format a number of seconds as a WebVTT timestamp (HH:MM:SS.mmm). */
function secondsToVttTimestamp(seconds: number): string {
  const s = Math.max(0, seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s - h * 3600 - m * 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${sec
    .toFixed(3)
    .padStart(6, '0')}`;
}

const INSERT_LEAD_LAG_SECONDS = 2;

const loadTranscript = (path: string) => async () => {
  const transcript = await transcriptApi.loadTranscript(path);
  return transcript as VttCue[];
};

export default function Transcript() {
  const { libraryService } = useContext(GlobalStateContext);
  const queryClient = useQueryClient();
  const followTranscript = useSelector(
    libraryService,
    (state) => state.context.settings.followTranscript
  );
  const actualVideoTime = useSelector(
    libraryService,
    (state) => state.context.videoPlayer.actualVideoTime
  );
  const library = useSelector(libraryService, (state) =>
    filter(
      state.context.libraryLoadId,
      state.context.textFilter,
      state.context.library,
      state.context.settings.filters,
      state.context.settings.sortBy
    )
  );
  const cursor = useSelector(
    libraryService,
    (state) => state.context.cursor,
    (a, b) => a === b
  );

  const item = library[cursor];
  const path = item?.path;

  // Job checking removed - transcript jobs now handled by external job runner service
  const transcriptJobs: any[] = [];

  const { data: transcript } = useQuery({
    queryKey: ['transcript', path],
    queryFn: loadTranscript(path),
    enabled: !!path,
  });

  const scrollRef = useRef<HTMLDivElement>(null);

  // Search state. matches is the list of cue indices that contain the
  // (case-insensitive) search substring; matchIndex is the cursor into
  // that list driven by the prev/next arrows.
  const [searchQuery, setSearchQuery] = useState('');
  const [matchIndex, setMatchIndex] = useState(0);

  // After inserting a new cue we want it to mount in edit mode so the
  // user can immediately type. Track its startTime here; the matching
  // Cue picks autoEdit=true on mount and the local edit-state takes
  // over from there. Cleared in a follow-up tick so subsequent renders
  // don't keep pinning that cue into edit mode.
  const [autoEditStartTime, setAutoEditStartTime] = useState<string | null>(
    null
  );

  const matches = useMemo<number[]>(() => {
    if (!searchQuery || !transcript) return [];
    const q = searchQuery.toLowerCase();
    const out: number[] = [];
    for (let i = 0; i < transcript.length; i++) {
      if (transcript[i].text.toLowerCase().includes(q)) out.push(i);
    }
    return out;
  }, [searchQuery, transcript]);

  // Reset the cursor whenever the query changes so the next/prev arrows
  // start from the first hit.
  useEffect(() => {
    setMatchIndex(0);
  }, [searchQuery]);

  const currentMatchCueIndex = matches[matchIndex];

  // Scroll a cue element into the middle of the panel's viewport.
  const centerCueElement = (el: HTMLElement) => {
    const container = scrollRef.current;
    if (!container) return;
    const containerRect = container.getBoundingClientRect();
    const elRect = el.getBoundingClientRect();
    const target =
      container.scrollTop + (elRect.top - containerRect.top) - container.clientHeight / 2 + el.clientHeight / 2;
    container.scrollTo({ top: target, behavior: 'smooth' });
  };

  // Scroll the current match cue into the middle of the viewport. Uses a
  // data attribute lookup rather than per-cue refs so we don't have to
  // thread a ref array through the Cue component.
  //
  // Depends on `searchQuery` as well as `currentMatchCueIndex` so each
  // keystroke that lands on a match re-scrolls — that's instant feedback
  // when typing, even if the first match happens to stay in the same cue
  // as more characters are added.
  useEffect(() => {
    if (!searchQuery || currentMatchCueIndex == null) return;
    const el = scrollRef.current?.querySelector(
      `[data-cue-index="${currentMatchCueIndex}"]`
    ) as HTMLElement | null;
    if (el) centerCueElement(el);
  }, [currentMatchCueIndex, searchQuery]);

  // Jump back to the cue at the current playback time. The active cue
  // already carries the .active class (the Cue computes it), so target
  // that; when playback sits between cues, fall back to the last cue
  // starting at or before the current time.
  const scrollToActiveCue = () => {
    const container = scrollRef.current;
    if (!container || !transcript?.length) return;
    let el = container.querySelector(
      'li.cue-container.active'
    ) as HTMLElement | null;
    if (!el) {
      let index = 0;
      for (let i = 0; i < transcript.length; i++) {
        if (
          convertVTTTimestampToSeconds(transcript[i].startTime) <=
          actualVideoTime
        ) {
          index = i;
        } else {
          break;
        }
      }
      el = container.querySelector(
        `[data-cue-index="${index}"]`
      ) as HTMLElement | null;
    }
    if (el) centerCueElement(el);
  };

  const goNext = () => {
    if (matches.length === 0) return;
    setMatchIndex((i) => (i + 1) % matches.length);
  };
  const goPrev = () => {
    if (matches.length === 0) return;
    setMatchIndex((i) => (i - 1 + matches.length) % matches.length);
  };

  const handleInsertAtCurrentTime = async () => {
    if (!path) return;
    const startSeconds = Math.max(0, actualVideoTime - INSERT_LEAD_LAG_SECONDS);
    const endSeconds = actualVideoTime + INSERT_LEAD_LAG_SECONDS;
    const startTime = secondsToVttTimestamp(startSeconds);
    const endTime = secondsToVttTimestamp(endSeconds);
    try {
      await transcriptApi.insertTranscriptCue({
        mediaPath: path,
        startTime,
        endTime,
        text: '',
      });
      setAutoEditStartTime(startTime);
      await queryClient.invalidateQueries({ queryKey: ['transcript', path] });
    } catch (err) {
      console.error('Failed to insert transcript cue:', err);
    }
  };

  if (!path) {
    return null;
  }
  // function to setScroll top smoothly
  function setScrollTop(scrollTop: number) {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollTop,
        behavior: 'auto',
      });
    }
  }

  console.log('transcriptJobs', transcriptJobs);

  if (transcriptJobs.length > 0) {
    return (
      <div className="transcript-loader">
        <div className="transcript-loader-inner">
          <SkeletonTheme baseColor="#202020" highlightColor="#444">
            <Skeleton count={1} />
          </SkeletonTheme>
        </div>
      </div>
    );
  }
  if (!transcript) {
    return (
      <GenerateTranscript
        path={path}
        variant="centered"
        label="Generate Transcript"
      />
    );
  }

  return (
    <div className="Transcript" ref={scrollRef}>
      <div className="transcript-search">
        <input
          type="text"
          className="transcript-search-input"
          placeholder="Search transcript…"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => {
            // Stop hotkey system from intercepting typing.
            e.stopPropagation();
            if (e.key === 'Enter') {
              e.preventDefault();
              if (e.shiftKey) goPrev();
              else goNext();
            } else if (e.key === 'Escape') {
              e.preventDefault();
              setSearchQuery('');
            }
          }}
          onKeyUp={(e) => e.stopPropagation()}
        />
        <div className="transcript-search-counter">
          {searchQuery
            ? matches.length === 0
              ? '0 / 0'
              : `${matchIndex + 1} / ${matches.length}`
            : ''}
        </div>
        <button
          type="button"
          className="transcript-search-nav"
          onClick={goPrev}
          disabled={matches.length === 0}
          aria-label="Previous match"
          title="Previous match (Shift+Enter)"
        >
          ▲
        </button>
        <button
          type="button"
          className="transcript-search-nav"
          onClick={goNext}
          disabled={matches.length === 0}
          aria-label="Next match"
          title="Next match (Enter)"
        >
          ▼
        </button>
        <button
          type="button"
          className="transcript-search-nav transcript-locate-btn"
          onClick={scrollToActiveCue}
          aria-label="Scroll to current time"
          title="Scroll to the cue at the current playback time"
        >
          ◉
        </button>
      </div>
      {/* Top action bar with regenerate + insert buttons when transcript exists */}
      <div className="transcript-actions">
        <button
          type="button"
          className="transcript-insert-btn"
          onClick={handleInsertAtCurrentTime}
          title={`Insert a new cue at the current time (±${INSERT_LEAD_LAG_SECONDS}s)`}
        >
          + Insert at {secondsToVttTimestamp(actualVideoTime || 0)}
        </button>
        <GenerateTranscript
          path={path}
          label={'Regenerate Transcript'}
          variant="inline"
        />
      </div>
      <ul>
        {transcript?.map((cue, index) => (
          <Cue
            cue={cue}
            cueIndex={index}
            mediaPath={path}
            key={cue.startTime}
            setScrollTop={setScrollTop}
            followVideoTime={followTranscript}
            searchQuery={searchQuery}
            isCurrentMatch={index === currentMatchCueIndex}
            autoEdit={cue.startTime === autoEditStartTime}
          />
        ))}
      </ul>
    </div>
  );
}
