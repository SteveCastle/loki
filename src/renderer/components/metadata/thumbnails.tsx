import React, { useEffect, useMemo, useState } from 'react';
import { getFileType, FileTypes } from '../../../file-types';
import { Image } from '../media-viewers/image';
import { Video } from '../media-viewers/video';

type ThumbInfo = {
  cache: 'thumbnail_path_100' | 'thumbnail_path_600' | 'thumbnail_path_1200';
  path: string;
  exists: boolean;
  size: number;
};

export default function Thumbnails({ path }: { path: string }) {
  const [thumbs, setThumbs] = useState<ThumbInfo[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [regening, setRegening] = useState<string | null>(null);

  const fileType = useMemo(() => getFileType(path), [path]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const results = await window.electron.listThumbnails(path);
        if (!cancelled) setThumbs(results);
      } catch (e) {
        console.error(e);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    if (path) load();
    return () => {
      cancelled = true;
    };
  }, [path]);

  const regenerate = async (cache: ThumbInfo['cache']) => {
    try {
      setRegening(cache);
      await window.electron.regenerateThumbnail(path, cache);
      const results = await window.electron.listThumbnails(path);
      setThumbs(results);
    } catch (e) {
      console.error(e);
    } finally {
      setRegening(null);
    }
  };

  const grid = useMemo(() => {
    return {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
      gap: '12px',
      padding: '16px 24px',
      width: '100%',
      height: '100%',
      overflow: 'auto',
      boxSizing: 'border-box' as const,
    };
  }, []);

  const card = {
    display: 'flex',
    flexDirection: 'column' as const,
    borderRadius: 12,
    overflow: 'hidden',
    background:
      'linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%)',
    border: '1px solid rgba(255,255,255,0.08)',
  };

  return (
    <div style={grid}>
      {loading || !thumbs
        ? Array.from({ length: 3 }).map((_, idx) => (
            <div key={`ph-${idx}`} style={card}>
              <div style={{ aspectRatio: '1 / 1', background: '#0f0f0f' }} />
              <div style={{ padding: '12px' }}>
                <div
                  style={{
                    height: 14,
                    background: 'rgba(255,255,255,0.08)',
                    borderRadius: 6,
                  }}
                />
              </div>
            </div>
          ))
        : thumbs.map((t) => (
            <div key={t.cache} style={card}>
              <div
                style={{
                  position: 'relative',
                  aspectRatio: '1 / 1',
                  background: '#0f0f0f',
                }}
              >
                <div style={{ position: 'absolute', inset: 8, right: 8 }}>
                  {fileType === FileTypes.Video ? (
                    <Video
                      path={path}
                      initialTimestamp={0}
                      scaleMode="cover"
                      orientation={'landscape'}
                      cache={
                        t.cache as 'thumbnail_path_1200' | 'thumbnail_path_600'
                      }
                      startTime={0}
                    />
                  ) : (
                    <Image
                      path={path}
                      cache={
                        t.cache as 'thumbnail_path_1200' | 'thumbnail_path_600'
                      }
                      orientation={'landscape'}
                    />
                  )}
                </div>
              </div>
              <div
                style={{
                  padding: '12px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: 12,
                  }}
                >
                  <div style={{ opacity: 0.9 }}>{t.cache}</div>
                  <div style={{ opacity: 0.7 }}>
                    {t.exists
                      ? `${Math.max(1, Math.round(t.size / 1024))} KB`
                      : 'missing'}
                  </div>
                </div>
                <div
                  title={t.path}
                  style={{
                    fontSize: 11,
                    opacity: 0.7,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {t.path}
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button
                    onClick={() => regenerate(t.cache)}
                    disabled={regening === t.cache}
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
                    {regening === t.cache ? 'Regenerating…' : 'Regenerate'}
                  </button>
                </div>
              </div>
            </div>
          ))}
    </div>
  );
}
