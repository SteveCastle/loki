import { useContext, memo } from 'react';
import { GlobalStateContext, Item } from '../../state';
import { uniqueId } from 'lodash';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Tooltip } from 'react-tooltip';
import './tags.css';
import { useSelector } from '@xstate/react';

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

interface Props {
  item: Item;
}

function getLabel(currentVideoTimeStamp: number) {
  // Returns a string in the format of 00:00:00
  const hours = Math.floor(currentVideoTimeStamp / 3600);
  const minutes = Math.floor((currentVideoTimeStamp - hours * 3600) / 60);
  const seconds = Math.floor(
    currentVideoTimeStamp - hours * 3600 - minutes * 60
  );

  const hoursString = hours < 10 ? `0${hours}` : `${hours}`;
  const minutesString = minutes < 10 ? `0${minutes}` : `${minutes}`;
  const secondsString = seconds < 10 ? `0${seconds}` : `${seconds}`;

  return `${hoursString}:${minutesString}:${secondsString}`;
}

function Tags({ item }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const { data, error, isLoading, refetch } = useQuery<Metadata, Error>(
    ['metadata', item],
    loadTagsByMediaPath(item),
    { retry: true }
  );
  const { mutate } = useMutation({
    mutationFn: deleteTag,
    onSuccess: () => {
      libraryService.send({ type: 'DELETED_ASSIGNMENT' });
      refetch();
    },
  });

  const { battleMode } = useSelector(libraryService, (state) => {
    return state.context.settings;
  });
  if (isLoading || !data) return null;
  if (error) return <p>{error.message}</p>;
  return (
    <div className={`Tags`}>
      <ul>
        {(data.tags || [])
          .filter(
            (tag) =>
              tag.time_stamp === item.timeStamp ||
              tag.tag_label !== item.tagLabel
          )
          .map((tag, idx) => {
            return (
              <li
                key={`${tag.tag_label}-${idx}`}
                onClick={() => {
                  libraryService.send({
                    type: 'SET_QUERY_TAG',
                    data: { tag: tag.tag_label },
                  });
                }}
              >
                {tag.time_stamp ? (
                  <span
                    data-tooltip-id={`tooltip-${tag.tag_label}-${idx}`}
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
                <Tooltip
                  id={`tooltip-${tag.tag_label}-${idx}`}
                  content={getLabel(tag.time_stamp)}
                />
              </li>
            );
          })}
        <li>{item.elo ? item.elo.toFixed(0) : 'Unranked'}</li>
      </ul>
    </div>
  );
}

export default memo(Tags);
