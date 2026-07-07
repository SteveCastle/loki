import { useContext } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { useDepRequirement } from '../../onboarding/useDepRequirement';
import { fmtSize } from '../../onboarding/requirements';
import { mediaServerBase } from '../../platform';
import useJobServerAvailable from '../../hooks/useJobServerAvailable';
import './generate-tags.css';
import { SparkleIcon } from './section-action-icons';

type Props = {
  path: string;
};

export default function GenerateTags({ path }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );
  const jobServerAvailable = useJobServerAvailable(authToken);
  const tagger = useDepRequirement('wd-eva02-large-tagger-v3');

  const handleGenerateTags = async () => {
    try {
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      };

      if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
      }

      const response = await fetch(`${mediaServerBase}/create`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          // --overwrite: an explicit per-file request re-tags even if the
          // file already carries Suggested tags (the task's default is to
          // skip already-tagged items on bulk runs).
          input: `autotag --overwrite "${path}"`,
        }),
        signal: AbortSignal.timeout(10000),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Tag generation job created:', result.id);
      // No toast needed - job will appear in job toast list automatically
    } catch (error) {
      console.error('Failed to create tag generation job:', error);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to Create Job',
          message: 'Could not communicate with job service',
        },
      });
    }
  };

  // Don't show anything if server is not available
  if (jobServerAvailable !== true) {
    return null;
  }

  // Auto-tagging needs the tagger model; offer the one-time download in place.
  if (tagger.needsDownload) {
    return (
      <div className="GenerateTags section-action">
        <button
          className="section-action-pill"
          onClick={() => tagger.download().catch(() => {})}
          title={`One-time model download (${fmtSize(tagger.dep?.size_bytes)}); tagging runs locally`}
        >
          <SparkleIcon />
          <span>Get model ({fmtSize(tagger.dep?.size_bytes)})</span>
        </button>
      </div>
    );
  }
  if (tagger.downloading) {
    return (
      <div className="GenerateTags section-action">
        <button className="section-action-pill" disabled>
          <SparkleIcon />
          <span>Downloading… {tagger.pct}%</span>
        </button>
      </div>
    );
  }

  return (
    <div className="GenerateTags section-action">
      <button
        className="section-action-pill"
        onClick={handleGenerateTags}
        title="Generate tags for this file"
      >
        <SparkleIcon />
        <span>Generate</span>
      </button>
    </div>
  );
}
