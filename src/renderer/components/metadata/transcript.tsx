import { useRef, useContext, useState, useMemo, useEffect } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery } from '@tanstack/react-query';
import { transcript as transcriptApi } from '../../platform';

import { GlobalStateContext } from '../../state';
import filter from '../../filter';
import { VttCue } from 'main/parse-vtt';
import { Cue } from './cue';
import GenerateTranscript from './generate-transcript';
import './transcript.css';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';

const loadTranscript = (path: string) => async () => {
  const transcript = await transcriptApi.loadTranscript(path);
  return transcript as VttCue[];
};

export default function Transcript() {
  const { libraryService } = useContext(GlobalStateContext);
  const followTranscript = useSelector(
    libraryService,
    (state) => state.context.settings.followTranscript
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

  // Scroll the current match cue into the middle of the viewport. Uses a
  // data attribute lookup rather than per-cue refs so we don't have to
  // thread a ref array through the Cue component.
  useEffect(() => {
    if (currentMatchCueIndex == null) return;
    const container = scrollRef.current;
    if (!container) return;
    const el = container.querySelector(
      `[data-cue-index="${currentMatchCueIndex}"]`
    ) as HTMLElement | null;
    if (!el) return;
    const containerRect = container.getBoundingClientRect();
    const elRect = el.getBoundingClientRect();
    const target =
      container.scrollTop + (elRect.top - containerRect.top) - container.clientHeight / 2 + el.clientHeight / 2;
    container.scrollTo({ top: target, behavior: 'smooth' });
  }, [currentMatchCueIndex]);

  const goNext = () => {
    if (matches.length === 0) return;
    setMatchIndex((i) => (i + 1) % matches.length);
  };
  const goPrev = () => {
    if (matches.length === 0) return;
    setMatchIndex((i) => (i - 1 + matches.length) % matches.length);
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
      </div>
      {/* Top action bar with regenerate button when transcript exists */}
      <div className="transcript-actions">
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
          />
        ))}
      </ul>
    </div>
  );
}
