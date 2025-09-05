import { useRef, useContext } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery } from '@tanstack/react-query';

import { GlobalStateContext } from '../../state';
import filter from '../../filter';
import { VttCue } from 'main/parse-vtt';
import { Cue } from './cue';
import GenerateTranscript from './generate-transcript';
import './transcript.css';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';

const loadTranscript = (path: string) => async () => {
  const transcript = await window.electron.transcript.loadTranscript(path);
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
    return <GenerateTranscript path={path} />;
  }

  return (
    <div className="Transcript" ref={scrollRef}>
      <ul>
        {transcript?.map((cue, index) => (
          <Cue
            cue={cue}
            cueIndex={index}
            mediaPath={path}
            key={cue.startTime}
            setScrollTop={setScrollTop}
            followVideoTime={followTranscript}
          />
        ))}
      </ul>
    </div>
  );
}
