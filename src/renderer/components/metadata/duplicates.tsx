import React, {
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import { GlobalStateContext, Item } from '../../state';
import { getFileType, FileTypes } from '../../../file-types';
import { Image } from '../media-viewers/image';
import { Video } from '../media-viewers/video';
import { Audio } from '../media-viewers/audio';

type Props = {
  basePath: string;
};

function MediaPreview({ path }: { path: string }) {
  const type = getFileType(path);
  if (type === FileTypes.Video) {
    return (
      <Video
        path={path}
        initialTimestamp={0.5}
        scaleMode="cover"
        orientation={'landscape'}
        cache={'thumbnail_path_600'}
        startTime={0}
      />
    );
  }
  if (type === FileTypes.Audio) {
    return (
      <Audio
        path={path}
        initialTimestamp={0}
        scaleMode="cover"
        orientation={'landscape'}
        cache={false}
        startTime={0}
      />
    );
  }
  return (
    <Image
      path={path}
      scaleMode="cover"
      orientation={'landscape'}
      cache={'thumbnail_path_600'}
    />
  );
}

export default function Duplicates({ basePath }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const [items, setItems] = useState<Item[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const results = await (
          window as unknown as {
            electron: {
              loadDuplicatesByPath: (
                path: string
              ) => Promise<{ library: { path: string }[]; cursor: number }>;
            };
          }
        ).electron.loadDuplicatesByPath(basePath);
        if (cancelled) return;
        const library = (results?.library || []) as { path: string }[];
        const mapped: Item[] = library.map((r) => ({
          path: r.path,
          mtimeMs: 0,
        }));
        const withSelf: Item[] = [
          { path: basePath, mtimeMs: 0 },
          ...mapped.filter((m) => m.path !== basePath),
        ];
        setItems(withSelf);
      } catch (e) {
        console.error(e);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    if (basePath) {
      load();
    }
    return () => {
      cancelled = true;
    };
  }, [basePath]);

  const handleDelete = useCallback(
    async (p: string) => {
      try {
        libraryService.send('DELETE_FILE', { data: { path: p } });
        setItems((prev) => prev.filter((i) => i.path !== p));
      } catch (e) {
        console.error(e);
      }
    },
    [libraryService]
  );

  const gridTemplate = useMemo(() => {
    const minSize = 220; // px, minimum card width
    return {
      display: 'grid',
      gridTemplateColumns: `repeat(auto-fit, minmax(${minSize}px, 1fr))`,
      justifyContent: 'stretch',
      justifyItems: 'stretch',
      alignContent: 'start',
      gap: '12px',
      width: '100%',
      overflow: 'auto',
      padding: '16px 24px',
      boxSizing: 'border-box' as const,
    };
  }, []);

  const renderCard = (it: Item, isCurrent: boolean) => (
    <div
      key={it.path + (isCurrent ? '-current' : '')}
      style={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        borderRadius: 12,
        overflow: 'hidden',
        background:
          'linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%)',
        border: '1px solid rgba(255,255,255,0.08)',
        boxShadow:
          '0 2px 6px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04)',
        transition: 'transform 120ms ease-out, box-shadow 120ms ease-out',
      }}
    >
      <div
        style={{
          position: 'relative',
          aspectRatio: '1 / 1',
          background: '#0f0f0f',
        }}
      >
        <div style={{ position: 'absolute', inset: 0 }}>
          <MediaPreview path={it.path} />
        </div>
        {isCurrent ? (
          <div
            style={{
              position: 'absolute',
              top: 8,
              left: 8,
              padding: '2px 8px',
              fontSize: 11,
              borderRadius: 999,
              background: 'rgba(0,0,0,0.5)',
              border: '1px solid rgba(255,255,255,0.25)',
            }}
          >
            Current
          </div>
        ) : null}
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 8,
          padding: '10px 12px',
          background: 'rgba(0,0,0,0.25)',
        }}
      >
        <div
          title={it.path}
          style={{
            width: '100%',
            textAlign: 'center' as const,
            fontSize: 12,
            lineHeight: 1.2,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {it.path}
        </div>
        <div
          style={{
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            width: '100%',
          }}
        >
          {isCurrent ? (
            <button
              onClick={async () => {
                try {
                  setLoading(true);
                  const res = await (
                    window as unknown as {
                      electron: {
                        mergeDuplicatesByPath: (path: string) => Promise<{
                          mergedInto: string;
                          deleted: string[];
                          copiedTags: number;
                        }>;
                      };
                    }
                  ).electron.mergeDuplicatesByPath(it.path);
                  // Remove deleted items from the list
                  setItems((prev) =>
                    prev.filter((p) => !res.deleted.includes(p.path))
                  );
                } catch (e) {
                  console.error(e);
                } finally {
                  setLoading(false);
                }
              }}
              style={{
                fontSize: 12,
                padding: '6px 12px',
                borderRadius: 999,
                border: '1px solid rgba(77,155,255,0.45)',
                background:
                  'linear-gradient(180deg, rgba(77,155,255,0.15) 0%, rgba(77,155,255,0.08) 100%)',
                color: '#b3d2ff',
                cursor: 'pointer',
                outline: 'none',
              }}
            >
              Merge
            </button>
          ) : null}
          <button
            onClick={() => handleDelete(it.path)}
            style={{
              fontSize: 12,
              padding: '6px 12px',
              borderRadius: 999,
              border: '1px solid rgba(255,77,77,0.45)',
              background:
                'linear-gradient(180deg, rgba(255,77,77,0.15) 0%, rgba(255,77,77,0.08) 100%)',
              color: '#ffb3b3',
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  );

  const renderPlaceholderCard = (idx: number) => (
    <div
      key={`ph-${idx}`}
      style={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        borderRadius: 12,
        overflow: 'hidden',
        background: 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(255,255,255,0.08)',
      }}
    >
      <div
        style={{ aspectRatio: '1 / 1', background: 'rgba(255,255,255,0.06)' }}
      />
      <div style={{ padding: '10px 12px', background: 'rgba(0,0,0,0.25)' }}>
        <div
          style={{
            height: 14,
            background: 'rgba(255,255,255,0.08)',
            borderRadius: 6,
          }}
        />
      </div>
    </div>
  );

  return (
    <div style={gridTemplate}>
      {loading
        ? Array.from({ length: 8 }).map((_, i) => renderPlaceholderCard(i))
        : items.map((it, i) => renderCard(it, i === 0))}
    </div>
  );
}
