import { useQuery } from '@tanstack/react-query';
import { Metadata } from 'main/metadata';
import { invoke } from '../../platform';
import { getFileType, FileTypes } from '../../../file-types';
import './description-overlay.css';

const loadFileMetadata = (path: string) => async (): Promise<Metadata> => {
  const metadata = await invoke('load-file-metadata', [path]);
  return (metadata || { path, tags: '' }) as Metadata;
};

// Closed-caption-style overlay across the bottom of the detail view. On
// images it captions the text captured from the picture (the transcript
// column, kept line by line so a comic script reads as dialogue); on video
// and audio it shows the description. Shares the ['file-metadata', path]
// query with the metadata panel, so edits and regenerated text appear here
// through the same cache invalidation with no extra requests.
export default function DescriptionOverlay({
  path,
  fontSize,
  sidePadding,
}: {
  path: string;
  fontSize: number;
  /** Horizontal padding as a percentage of the panel width per side. */
  sidePadding: number;
}) {
  const { data } = useQuery<Metadata, Error>(
    ['file-metadata', path],
    loadFileMetadata(path),
    { retry: true }
  );

  const isImage = getFileType(path) === FileTypes.Image;
  // Image text keeps its lines (blank runs collapsed); descriptions are
  // prose, so all whitespace collapses and the text flows edge-to-edge.
  const text = isImage
    ? data?.transcript
        ?.split(/\r?\n/)
        .map((l) => l.replace(/[ \t]+/g, ' ').trim())
        .filter(Boolean)
        .join('\n')
    : data?.description?.replace(/\s+/g, ' ').trim();
  if (!text) {
    return null;
  }

  return (
    <div
      className={`DescriptionOverlay${isImage ? ' captions' : ''}`}
      aria-live="polite"
      style={{
        fontSize: `${fontSize || 18}px`,
        paddingLeft: `${sidePadding ?? 4}%`,
        paddingRight: `${sidePadding ?? 4}%`,
      }}
    >
      {text}
    </div>
  );
}
