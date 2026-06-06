import { useQuery } from '@tanstack/react-query';
import { invoke } from '../../platform';
import type { Predicate } from '../../query/types';
import './taxonomy.css';

interface CategoryLite {
  label: string;
}

interface SuggestionSectionsProps {
  text: string; // active search term (non-empty when shown)
  categories: CategoryLite[];
  onAdd: (predicate: Predicate) => void;
}

const SECTION_CAP = 8;

// Lazy media count for a single rendered category row. Only mounted for
// categories that are actually shown, so we never fetch counts for the
// filtered-out / capped categories.
function CategoryCount({ label }: { label: string }) {
  const { data: count } = useQuery<number, Error>(
    ['suggest', 'category-count', label],
    () => invoke('get-category-count', [label]) as Promise<number>,
    { refetchOnWindowFocus: false }
  );
  if (count === undefined) return null;
  return <span className="suggestion-meta">{count}</span>;
}

export default function SuggestionSections({
  text,
  categories,
  onAdd,
}: SuggestionSectionsProps) {
  const term = text.toLowerCase();

  // 1. Categories — substring match on label, capped.
  const matchedCategories = categories
    .filter((c) => c.label.toLowerCase().includes(term))
    .slice(0, SECTION_CAP);

  // 2. Paths — distinct directory fragments containing the term.
  const { data: pathResults } = useQuery<string[], Error>(
    ['suggest', 'paths', text],
    () => invoke('load-path-suggestions', [text]) as Promise<string[]>,
    { enabled: text.length > 0, refetchOnWindowFocus: false }
  );

  const distinctDirs: string[] = [];
  if (pathResults) {
    const seen = new Set<string>();
    for (const full of pathResults) {
      const segments = full.split(/[/\\]/);
      for (const seg of segments) {
        if (!seg) continue;
        if (!seg.toLowerCase().includes(term)) continue;
        if (seen.has(seg)) continue;
        seen.add(seg);
        distinctDirs.push(seg);
        if (distinctDirs.length >= SECTION_CAP) break;
      }
      if (distinctDirs.length >= SECTION_CAP) break;
    }
  }

  return (
    <div className="suggestion-sections">
      {matchedCategories.length > 0 && (
        <div className="suggestion-section">
          <div className="suggestion-section-label">Categories</div>
          {matchedCategories.map((c) => (
            <div
              key={c.label}
              className="suggestion-row suggestion-row-category"
              onClick={() =>
                onAdd({ type: 'category', value: c.label, exclude: false })
              }
            >
              <span className="suggestion-prefix">in:</span>
              <span className="suggestion-value">{c.label}</span>
              <CategoryCount label={c.label} />
            </div>
          ))}
        </div>
      )}

      <div className="suggestion-section">
        <div className="suggestion-section-label">Paths</div>
        <div
          className="suggestion-row suggestion-add-row"
          onClick={() =>
            onAdd({ type: 'path', value: text, exclude: false })
          }
        >
          <span className="suggestion-add-badge">+</span>
          <span className="suggestion-value">
            path contains &quot;{text}&quot;
          </span>
        </div>
        {distinctDirs.map((dir) => (
          <div
            key={dir}
            className="suggestion-row"
            onClick={() =>
              onAdd({ type: 'path', value: dir, exclude: false })
            }
          >
            <span className="suggestion-prefix">path:</span>
            <span className="suggestion-value">{dir}</span>
          </div>
        ))}
      </div>

      <div className="suggestion-section">
        <div className="suggestion-section-label">Description</div>
        <div
          className="suggestion-row suggestion-add-row"
          onClick={() =>
            onAdd({ type: 'description', value: text, exclude: false })
          }
        >
          <span className="suggestion-add-badge">+</span>
          <span className="suggestion-value">
            description contains &quot;{text}&quot;
          </span>
        </div>
      </div>

      <div className="suggestion-section">
        <div className="suggestion-section-label">Hash</div>
        <div
          className="suggestion-row suggestion-add-row"
          onClick={() =>
            onAdd({ type: 'hash', value: text, exclude: false })
          }
        >
          <span className="suggestion-add-badge">+</span>
          <span className="suggestion-value">
            hash contains &quot;{text}&quot;
          </span>
        </div>
      </div>
    </div>
  );
}
