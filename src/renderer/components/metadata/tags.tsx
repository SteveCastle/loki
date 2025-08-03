import { useContext, memo, useRef, useEffect, useState } from 'react';
import { GlobalStateContext, Item } from '../../state';
import { uniqueId } from 'lodash';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import './tags.css';
import TimestampTooltip from './timestamp-tooltip';
import GenerateTags from './generate-tags';

type Tag = {
  tag_label: string;
  category_label?: string;
  weight?: number;
  time_stamp: number;
};

type Metadata = {
  path: string;
  tags: Tag[];
};
const loadTagsByMediaPath = (media: Item) => async (): Promise<Metadata> => {
  let metadata: any;
  metadata = await window.electron.ipcRenderer.invoke(
    'load-tags-by-media-path',
    [media]
  );

  metadata = metadata || { path: media.path, tags: [] };
  return metadata as Metadata;
};

const deleteTag = async ({ path, tag }: { path: string; tag: Tag }) => {
  await window.electron.ipcRenderer.invoke('delete-assignment', [path, tag]);
};

const updateTimestamp = async ({ 
  path, 
  tagLabel, 
  oldTimestamp, 
  newTimestamp 
}: { 
  path: string; 
  tagLabel: string; 
  oldTimestamp: number; 
  newTimestamp: number; 
}) => {
  await window.electron.ipcRenderer.invoke('update-timestamp', [path, tagLabel, oldTimestamp, newTimestamp]);
};

const removeTimestamp = async ({ 
  path, 
  tagLabel, 
  timestamp 
}: { 
  path: string; 
  tagLabel: string; 
  timestamp: number; 
}) => {
  console.log('Frontend: removing timestamp', { path, tagLabel, timestamp });
  await window.electron.ipcRenderer.invoke('remove-timestamp', [path, tagLabel, timestamp]);
};

interface Props {
  item: Item;
  enableTagGeneration?: boolean;
}


function Tags({ item, enableTagGeneration = false }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const queryClient = useQueryClient();
  const containerRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(enableTagGeneration); // Always visible for metadata panel
  
  // Reset visibility when item changes (for detail views)
  useEffect(() => {
    if (!enableTagGeneration) {
      setIsVisible(false);
    }
  }, [item.path, enableTagGeneration]);
  
  // For detail views, use intersection observer to only load when visible
  useEffect(() => {
    if (enableTagGeneration) return; // Skip for metadata panel
    
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect(); // Only load once when first visible
        }
      },
      { threshold: 0.1, rootMargin: '50px' } // Load slightly before visible
    );
    
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }
    
    return () => observer.disconnect();
  }, [enableTagGeneration, isVisible]); // Re-observe when visibility resets
  
  const { data, error, isLoading } = useQuery<Metadata, Error>({
    queryKey: ['tags-by-path', item.path],
    queryFn: loadTagsByMediaPath(item),
    retry: true,
    enabled: isVisible, // Only run query when visible
    staleTime: enableTagGeneration ? 0 : 5 * 60 * 1000, // 5 minutes for detail views
    cacheTime: enableTagGeneration ? 5 * 60 * 1000 : 10 * 60 * 1000, // Longer cache for detail views
  });

  const { mutate } = useMutation({
    mutationFn: deleteTag,
    onSuccess: () => {
      libraryService.send({ type: 'DELETED_ASSIGNMENT' });
      queryClient.invalidateQueries({
        queryKey: ['tags-by-path', item.path],
      });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
    },
  });

  const { mutate: mutateUpdateTimestamp } = useMutation({
    mutationFn: updateTimestamp,
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ['tags-by-path', item.path],
      });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
    },
  });

  const { mutate: mutateRemoveTimestamp } = useMutation({
    mutationFn: removeTimestamp,
    onSuccess: () => {
      console.log('Remove timestamp success, invalidating queries...');
      queryClient.invalidateQueries({
        queryKey: ['tags-by-path', item.path],
      });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
    },
    onError: (error) => {
      console.error('Remove timestamp error:', error);
    },
  });

  if (!isVisible) {
    return <div ref={containerRef} className={`Tags`} style={{ minHeight: '20px' }} />;
  }
  
  if (isLoading || !data) return <div ref={containerRef} className={`Tags`} />;
  if (error) return <div ref={containerRef} className={`Tags`}><p>{error.message}</p></div>;
  return (
    <div ref={containerRef} className={`Tags`}>
      <ul>
        {(data.tags || [])
          .map((tag, idx) => {
            return (
              <li
                key={`${tag.tag_label}-${tag.time_stamp || 0}-${idx}`}
                onClick={() => {
                  libraryService.send({
                    type: 'SET_QUERY_TAG',
                    data: { tag: tag.tag_label },
                  });
                }}
              >
                {tag.time_stamp ? (
                  <>
                    <span
                      data-tooltip-id={`tooltip-${tag.tag_label}-${tag.time_stamp}-${idx}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        libraryService.send('SET_VIDEO_TIME', {
                          timeStamp: tag.time_stamp,
                          eventId: uniqueId(),
                        });
                      }}
                    >
                      🕑
                    </span>
                    <TimestampTooltip
                      id={`tooltip-${tag.tag_label}-${tag.time_stamp}-${idx}`}
                      timestamp={tag.time_stamp}
                      onEdit={(newTimestamp) => {
                        console.log('Editing timestamp:', { 
                          path: item.path, 
                          tagLabel: tag.tag_label, 
                          oldTimestamp: tag.time_stamp, 
                          newTimestamp 
                        });
                        mutateUpdateTimestamp({
                          path: item.path,
                          tagLabel: tag.tag_label,
                          oldTimestamp: tag.time_stamp,
                          newTimestamp,
                        });
                      }}
                      onRemove={() => {
                        console.log('Removing timestamp:', { 
                          path: item.path, 
                          tagLabel: tag.tag_label, 
                          timestamp: tag.time_stamp 
                        });
                        mutateRemoveTimestamp({
                          path: item.path,
                          tagLabel: tag.tag_label,
                          timestamp: tag.time_stamp,
                        });
                      }}
                    />
                  </>
                ) : null}
                <span>{tag.tag_label}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    mutate({ path: item.path, tag });
                  }}
                >
                  ❌
                </button>
              </li>
            );
          })}
        {item.elo && <li>{item.elo.toFixed(0)}</li>}
      </ul>
      {enableTagGeneration && <GenerateTags path={item.path} />}
    </div>
  );
}

export default memo(Tags, (prevProps, nextProps) => {
  return prevProps.item.path === nextProps.item.path && 
         prevProps.enableTagGeneration === nextProps.enableTagGeneration;
});
