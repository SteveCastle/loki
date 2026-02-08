import React, {
  useContext,
  useRef,
  useCallback,
  useMemo,
  useEffect,
} from 'react';
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
  onDimensionsLoaded?: (itemKey: string, width: number, height: number) => void;
};

const GetPlayer = React.memo(
  (props: {
    path: string;
    mediaRef: React.RefObject<
      HTMLImageElement | HTMLVideoElement | HTMLAudioElement
    >;
    orientation: 'portrait' | 'landscape' | 'unknown';
    imageCache: 'thumbnail_path_1200' | 'thumbnail_path_600' | false;
    startTime?: number;
    onMediaLoad?: React.ReactEventHandler<HTMLImageElement | HTMLVideoElement>;
  }) => {
    const {
      path,
      mediaRef,
      orientation,
      imageCache,
      startTime = 0,
      onMediaLoad,
    } = props;

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
          handleLoad={onMediaLoad}
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
          handleLoad={onMediaLoad}
        />
      );
    }
    return null;
  }
);

GetPlayer.displayName = 'GetPlayer';

function ListItemComponent({ item, idx, height, onDimensionsLoaded }: Props) {
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
  const {
    orientation,
    width: naturalWidth,
    height: naturalHeight,
  } = useMediaDimensions(
    mediaRef as React.RefObject<HTMLImageElement | HTMLVideoElement>
  );

  // Track if we've already reported dimensions to avoid duplicates
  const hasReportedRef = useRef(false);

  // Report dimensions when media loads (for masonry layout)
  // This effect handles videos and fallback cases
  useEffect(() => {
    if (
      onDimensionsLoaded &&
      !hasReportedRef.current &&
      naturalWidth > 0 &&
      naturalHeight > 0
    ) {
      const itemKey =
        item.timeStamp != null ? `${item.path}::${item.timeStamp}` : item.path;
      onDimensionsLoaded(itemKey, naturalWidth, naturalHeight);
      hasReportedRef.current = true;
    }
  }, [
    onDimensionsLoaded,
    naturalWidth,
    naturalHeight,
    item.path,
    item.timeStamp,
  ]);

  // Direct callback for when media loads - more reliable than useMediaDimensions
  // because it fires exactly when the element triggers its load event.
  // Handles both <img> (onLoad) and <video> (onLoadedData) elements.
  const handleMediaLoad = useCallback(
    (e: React.SyntheticEvent<HTMLImageElement | HTMLVideoElement>) => {
      if (onDimensionsLoaded && !hasReportedRef.current) {
        const el = e.currentTarget;
        let w = 0;
        let h = 0;
        if (el instanceof HTMLImageElement) {
          w = el.naturalWidth;
          h = el.naturalHeight;
        } else if (el instanceof HTMLVideoElement) {
          w = el.videoWidth;
          h = el.videoHeight;
        }
        if (w > 0 && h > 0) {
          const itemKey =
            item.timeStamp != null
              ? `${item.path}::${item.timeStamp}`
              : item.path;
          onDimensionsLoaded(itemKey, w, h);
          hasReportedRef.current = true;
        }
      }
    },
    [onDimensionsLoaded, item.path, item.timeStamp]
  );
  const [, drag] = useDrag(
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

  const { drop, collectedProps, containerRef, isLeft } = useTagDrop(
    item,
    'LIST'
  );
  drag(drop(containerRef));

  const handleClick = useCallback(() => {
    libraryService.send('SET_CURSOR', { idx });
  }, [libraryService, idx]);

  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      libraryService.send('SHOW_COMMAND_PALETTE', {
        position: { x: e.clientX, y: e.clientY },
      });
    },
    [libraryService]
  );

  const handleFilePathClick = useCallback(() => {
    libraryService.send('SET_FILE', { path: item.path });
  }, [libraryService, item.path]);
  const classNames = useMemo(
    () =>
      [
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
        isLeft ? 'left' : 'right',
      ].join(' '),
    [
      cursor,
      idx,
      canDrag,
      collectedProps.isOver,
      collectedProps.isSelf,
      collectedProps.itemType,
      isLeft,
    ]
  );

  return (
    <div
      ref={containerRef}
      style={{
        height,
      }}
      className={classNames}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
    >
      <div className="inner">
        <GetPlayer
          path={item.path}
          mediaRef={mediaRef}
          orientation={orientation}
          imageCache={imageCache}
          startTime={item.timeStamp}
          onMediaLoad={onDimensionsLoaded ? handleMediaLoad : undefined}
        />
      </div>
      {showTags === 'all' || showTags === 'list' ? (
        <div className="controls">
          <Tags item={item} enableTagGeneration={false} />
        </div>
      ) : null}
      {showFileInfo === 'all' || showFileInfo === 'list' ? (
        <div className="item-info">
          <span className="file-path" onClick={handleFilePathClick}>
            {item.path}
          </span>
        </div>
      ) : null}
    </div>
  );
}

export const ListItem = React.memo(
  ListItemComponent,
  (prevProps, nextProps) => {
    return (
      prevProps.item.path === nextProps.item.path &&
      prevProps.idx === nextProps.idx &&
      prevProps.height === nextProps.height &&
      prevProps.item.timeStamp === nextProps.item.timeStamp &&
      prevProps.item.elo === nextProps.item.elo &&
      prevProps.item.weight === nextProps.item.weight
    );
  }
);

ListItem.displayName = 'ListItem';
