import React, { useContext, useEffect, useRef } from 'react';
import { useSelector } from '@xstate/react';
import { useDrag } from 'react-dnd';
import { GlobalStateContext } from '../../state';
import { Item } from '../../state';
import { Image } from '../media-viewers/image';
import { Video } from '../media-viewers/video';
import { Audio } from '../media-viewers/audio';
import { getFileType, FileTypes } from '../../../file-types';
import useMediaDimensions from 'renderer/hooks/useMediaDimensions';
import { ScaleModeOption } from 'settings';
import useTagDrop from 'renderer/hooks/useTagDrop';
import './list-item.css';
import Tags from '../metadata/tags';

type Props = {
  item: Item;
  idx: number;
  scaleMode: ScaleModeOption;
  height: number;
};

function getPlayer(
  path: string,
  mediaRef: React.RefObject<
    HTMLImageElement | HTMLVideoElement | HTMLAudioElement
  >,
  orientation: 'portrait' | 'landscape' | 'unknown',
  imageCache: 'thumbnail_path_1200' | 'thumbnail_path_600' | false,
  startTime = 0
) {
  if (getFileType(path, Boolean(imageCache)) === FileTypes.Video) {
    return (
      <Video
        path={path}
        initialTimestamp={0.5}
        scaleMode="cover"
        mediaRef={mediaRef as React.RefObject<HTMLVideoElement>}
        orientation={orientation}
        cache={imageCache}
        startTime={startTime}
      />
    );
  }
  if (getFileType(path) === FileTypes.Audio) {
    return (
      <Audio
        path={path}
        initialTimestamp={0}
        scaleMode="cover"
        mediaRef={mediaRef as React.RefObject<HTMLAudioElement>}
        orientation={orientation}
        cache={false}
        startTime={startTime}
      />
    );
  }
  if (getFileType(path) === FileTypes.Image) {
    return (
      <Image
        path={path}
        scaleMode="cover"
        mediaRef={mediaRef as React.RefObject<HTMLImageElement>}
        orientation={orientation}
        cache={imageCache}
      />
    );
  }
  return null;
}

export function ListItem({ item, idx, height }: Props) {
  const mediaRef = useRef<
    HTMLImageElement | HTMLVideoElement | HTMLAudioElement
  >(null);
  const { libraryService } = useContext(GlobalStateContext);
  const cursor = useSelector(libraryService, (state) => state.context.cursor);
  const { sortBy } = useSelector(libraryService, (state) => {
    return state.context.settings;
  });
  const state = useSelector(
    libraryService,
    (state) => state,
    (a, b) => {
      return a.matches(b);
    }
  );
  const canDrag =
    state.matches({ library: 'loadedFromDB' }) && sortBy === 'weight';
  const { showTags, showFileInfo } = useSelector(
    libraryService,
    (state) => state.context.settings
  );

  const imageCache = useSelector(libraryService, (state) => {
    return state.context.settings.listImageCache;
  });
  const { orientation } = useMediaDimensions(
    mediaRef as React.RefObject<HTMLImageElement | HTMLVideoElement>
  );
  const [{ isDragging, offset }, drag, dragPreview] = useDrag(
    () => ({
      type: 'MEDIA',
      item: item,
      canDrag,
      collect: (monitor) => ({
        isDragging: monitor.isDragging(),
        offset: monitor.getClientOffset(),
      }),
    }),
    [item, canDrag]
  );

  const { drop, collectedProps, containerRef } = useTagDrop(item, 'LIST');
  drag(drop(containerRef));
  return (
    <div
      ref={containerRef}
      style={{
        height,
      }}
      className={[
        'ListItem',
        cursor === idx ? 'selected' : '',
        collectedProps.isOver &&
        !collectedProps.isSelf &&
        collectedProps.itemType === 'MEDIA'
          ? 'hovered'
          : '',
        collectedProps.isOver &&
        !collectedProps.isSelf &&
        collectedProps.itemType === 'TAG'
          ? 'hovered-by-tag'
          : '',
        canDrag ? 'can-drag' : '',
        collectedProps.isLeft ? 'left' : 'right',
      ].join(' ')}
      onClick={() => libraryService.send('SET_CURSOR', { idx })}
      onContextMenu={(e) => {
        e.preventDefault();
        libraryService.send('SHOW_COMMAND_PALETTE', {
          position: { x: e.clientX, y: e.clientY },
        });
      }}
    >
      <div className="inner">
        {getPlayer(
          item.path,
          mediaRef,
          orientation,
          imageCache,
          item.timeStamp
        )}
      </div>
      {showTags === 'all' || showTags === 'list' ? (
        <div className="controls">
          <Tags item={item} />
        </div>
      ) : null}
      {showFileInfo === 'all' || showFileInfo === 'list' ? (
        <div className="item-info">
          <span
            className="file-path"
            onClick={() => libraryService.send('SET_FILE', { path: item.path })}
          >
            {item.path}
          </span>
        </div>
      ) : null}
    </div>
  );
}
