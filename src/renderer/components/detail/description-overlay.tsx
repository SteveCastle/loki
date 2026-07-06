import { useQuery } from '@tanstack/react-query';
import { Metadata } from 'main/metadata';
import { invoke } from '../../platform';
import './description-overlay.css';

const loadFileMetadata = (path: string) => async (): Promise<Metadata> => {
  const metadata = await invoke('load-file-metadata', [path]);
  return (metadata || { path, tags: '' }) as Metadata;
};

// Closed-caption-style overlay showing the media description across the
// bottom of the detail view. Shares the ['file-metadata', path] query with
// the metadata panel, so manual edits and regenerated descriptions appear
// here through the same cache invalidation with no extra requests.
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

  // Generators emit paragraphs and line breaks; collapse all whitespace so
  // the text flows edge-to-edge like a caption.
  const text = data?.description?.replace(/\s+/g, ' ').trim();
  if (!text) {
    return null;
  }

  return (
    <div
      className="DescriptionOverlay"
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
