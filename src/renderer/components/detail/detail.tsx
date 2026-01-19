import { useContext, useEffect, useRef, useState } from 'react';
import useScrollOnDrag from 'react-scroll-ondrag';
import { useSelector } from '@xstate/react';
import { GlobalStateContext, Item } from '../../state';
import filter from 'renderer/filter';
import { Image } from '../media-viewers/image';
import { AnimatedGif } from '../media-viewers/animated-gif';
import VideoControls from '../controls/video-controls';
import { Video } from '../media-viewers/video';
import { Audio } from '../media-viewers/audio';
import { Settings } from 'settings';
import { getFileType, FileTypes } from '../../../file-types';
import useMediaDimensions from 'renderer/hooks/useMediaDimensions';
import './detail.css';
import useTagDrop from 'renderer/hooks/useTagDrop';
import Tags from '../metadata/tags';
import BattleMode from '../elo/BattleMode';

function resizeToCover(
  parentWidth: number,
  parentHeight: number,
  childWidth: number,
  childHeight: number
) {
  const parentAspectRatio = parentWidth / parentHeight;
  const childAspectRatio = childWidth / childHeight;

  let newChildWidth, newChildHeight;

  // If parent aspect ratio is greater, the child width should match the parent width
  if (parentAspectRatio > childAspectRatio) {
    newChildWidth = parentWidth;
    newChildHeight = parentWidth / childAspectRatio;
  }
  // If child aspect ratio is greater or equal, the child height should match the parent height
  else {
    newChildHeight = parentHeight;
    newChildWidth = parentHeight * childAspectRatio;
  }

  return {
    width: newChildWidth,
    height: newChildHeight,
  };
}

function isGifFile(filePath: string): boolean {
  const extension = filePath.split('.').pop()?.toLowerCase();
  return extension === 'gif';
}

function getPlayer(
  path: string,
  mediaRef: React.RefObject<
    HTMLImageElement | HTMLVideoElement | HTMLAudioElement
  >,
  handleLoad: React.ReactEventHandler<
    HTMLImageElement | HTMLVideoElement | HTMLAudioElement
  >,
  settings: Settings,
  orientation: 'portrait' | 'landscape' | 'unknown',
  coverSize: { width: number; height: number },
  imageCache: 'thumbnail_path_1200' | 'thumbnail_path_600' | false,
  startTime = 0
) {
  if (getFileType(path, Boolean(imageCache)) === FileTypes.Video) {
    return (
      <Video
        key={path}
        path={path}
        scaleMode={settings.scaleMode}
        settable
        coverSize={coverSize}
        handleLoad={handleLoad}
        mediaRef={mediaRef as React.RefObject<HTMLVideoElement>}
        playSound={settings.playSound}
        volume={settings.volume}
        showControls={settings.showControls}
        orientation={orientation}
        startTime={startTime}
      />
    );
  }
  if (getFileType(path) === FileTypes.Audio) {
    return (
      <Audio
        key={path}
        path={path}
        scaleMode={settings.scaleMode}
        settable
        coverSize={coverSize}
        handleLoad={handleLoad}
        mediaRef={mediaRef as React.RefObject<HTMLAudioElement>}
        playSound={settings.playSound}
        volume={settings.volume}
        showControls={settings.showControls}
        orientation={orientation}
        startTime={startTime}
      />
    );
  }
  if (getFileType(path) === FileTypes.Image) {
    // Use AnimatedGif component for GIF files to enable loop tracking
    // Single-frame GIFs will be handled like regular images by the component
    if (isGifFile(path)) {
      return (
        <AnimatedGif
          path={path}
          coverSize={coverSize}
          scaleMode={settings.scaleMode}
          mediaRef={mediaRef as React.RefObject<HTMLImageElement>}
          handleLoad={handleLoad}
          orientation={orientation}
          cache={imageCache}
        />
      );
    }
    return (
      <Image
        path={path}
        coverSize={coverSize}
        scaleMode={settings.scaleMode}
        mediaRef={mediaRef as React.RefObject<HTMLImageElement>}
        handleLoad={handleLoad}
        orientation={orientation}
        cache={imageCache}
      />
    );
  }
  return null;
}

export function Detail({ offset = 0 }: { offset?: number }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mediaRef = useRef<
    HTMLImageElement | HTMLVideoElement | HTMLAudioElement
  >(null);
  let { events } = useScrollOnDrag(containerRef);
  const { libraryService } = useContext(GlobalStateContext);
  const settings = useSelector(libraryService, (state) => {
    return state.context.settings;
  });

  const item = useSelector(
    libraryService,
    (state) => {
      const library = filter(
        state.context.libraryLoadId,
        state.context.textFilter,
        state.context.library,
        state.context.settings.filters,
        state.context.settings.sortBy
      );
      return library
        ? getValueWithCycling(library, state.context.cursor + offset)
        : null;
    },
    (a, b) => a?.path === b?.path && a?.timeStamp === b?.timeStamp
  ) as Item;

  if (settings.controlMode === 'touchpad') {
    events = {};
  }

  const { drop } = useTagDrop(item, 'DETAIL');

  const handleScroll = (e: WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    e.preventDefault();
    //If wheel up.
    if (e.deltaY < 0) {
      libraryService.send('DECREMENT_CURSOR');
    }
    //If wheel down.
    if (e.deltaY > 0) {
      libraryService.send('INCREMENT_CURSOR');
    }
  };

  useEffect(() => {
    if (containerRef.current === null || settings.controlMode !== 'mouse') {
      return;
    }
    const container = containerRef.current;
    if (container) {
      container.addEventListener('wheel', handleScroll, { passive: false });
    }

    return () => {
      if (container) {
        container.removeEventListener('wheel', handleScroll);
      }
    };
  }, [containerRef.current, settings.controlMode]);

  function handleClick(e: React.MouseEvent<HTMLDivElement, MouseEvent>) {
    if (e.clientX < containerRef.current!.clientWidth / 2) {
      libraryService.send('DECREMENT_CURSOR');
    } else {
      libraryService.send('INCREMENT_CURSOR');
    }
  }

  const { orientation } = useMediaDimensions(
    mediaRef as React.RefObject<HTMLImageElement | HTMLVideoElement>
  );
  const [coverSize, setCoverSize] = useState({ width: 0, height: 0 });

  const handleResize = () => {
    const parentDiv = containerRef.current;
    const media = mediaRef.current;

    // Calculate the center of the image relative to the parent div
    if (media === null || parentDiv === null) {
      return;
    }

    // Audio elements don't need visual resizing
    if (media instanceof HTMLAudioElement) {
      return;
    }

    const parentWidth = parentDiv.clientWidth;
    const parentHeight = parentDiv.clientHeight;
    const offsetX = (media.clientWidth - parentWidth) / 2;
    const offsetY = (media.clientHeight - parentHeight) / 2;
    // Set the scroll position of the parent div to the center of the image
    parentDiv.scrollLeft = offsetX;
    parentDiv.scrollTop = offsetY;

    // If type is Video use videoheight and videowidth
    // If type is Image use naturalHeight and naturalWidth
    const nativeHeight =
      media instanceof HTMLVideoElement
        ? media.videoHeight
        : media.naturalHeight;
    const nativeWidth =
      media instanceof HTMLVideoElement ? media.videoWidth : media.naturalWidth;

    const mediaCoverSize = resizeToCover(
      parentWidth,
      parentHeight,
      nativeWidth,
      nativeHeight
    );
    setCoverSize(mediaCoverSize);
  };

  const handleLoad = () => {
    const media = mediaRef.current;

    handleResize();

    if (media instanceof HTMLVideoElement) {
      libraryService.send('SET_VIDEO_LENGTH', {
        videoLength: media.duration,
      });
      libraryService.send('SET_PLAYING_STATE', {
        playing: true,
      });
      libraryService.send('LOOP_VIDEO', {
        loopStartTime: 0,
        loopLength: 0,
      });
      if (item.timeStamp) {
        media.currentTime = item.timeStamp;
      }
    } else if (media instanceof HTMLAudioElement) {
      libraryService.send('SET_VIDEO_LENGTH', {
        videoLength: media.duration,
      });
      libraryService.send('SET_PLAYING_STATE', {
        playing: true,
      });
      libraryService.send('LOOP_VIDEO', {
        loopStartTime: 0,
        loopLength: 0,
      });
      if (item.timeStamp) {
        media.currentTime = item.timeStamp;
      }
    }
  };

  useEffect(() => {
    const divElement = containerRef.current;

    if (!divElement) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        handleResize();
      }
    });

    resizeObserver.observe(divElement);

    // Clean up function
    return () => resizeObserver.unobserve(divElement);
  }, [containerRef.current]); // Re-run effect if `onResize` changes

  if (!item) {
    return null;
  }

  // Use effect that registers a counter that fires a empty function every 1 second.

  return (
    <div
      ref={drop}
      className={[
        'DetailContainer',
        settings.comicMode ? 'comicMode' : '',
        settings.controlMode === 'mouse' ? 'grabbable' : '',
      ].join(' ')}
    >
      {settings.battleMode ? <BattleMode item={item} offset={offset} /> : null}
      <div
        className="Detail"
        onContextMenu={(e) => {
          e.preventDefault();
          libraryService.send('SHOW_COMMAND_PALETTE', {
            position: { x: e.clientX, y: e.clientY },
          });
        }}
        onClick={settings.controlMode === 'touchpad' ? handleClick : undefined}
        ref={containerRef}
        {...events}
      >
        {getPlayer(
          item.path,
          mediaRef,
          handleLoad,
          settings,
          orientation,
          coverSize,
          settings.detailImageCache,
          item.timeStamp
        )}
      </div>
      {!settings.showControls &&
        (getFileType(item.path) === 'video' ||
          getFileType(item.path) === 'audio') && (
          <div className="videoControls">
            <VideoControls />
          </div>
        )}
      {settings.showTags === 'all' || settings.showTags === 'detail' ? (
        <div className="controls">
          <Tags item={item} enableTagGeneration={false} />
        </div>
      ) : null}
      {settings.showFileInfo === 'all' || settings.showFileInfo === 'detail' ? (
        <div className="item-info">
          <span
            className="file-path"
            onClick={() => {
              console.log('SET_FILE', { path: item.path });
              libraryService.send('SET_FILE', { path: item.path });
            }}
          >
            {item.path}
          </span>
        </div>
      ) : null}
    </div>
  );
}

function getValueWithCycling<T>(arr: T[], index: number): T | null {
  if (arr.length === 0) {
    return null;
  }

  // Use the modulo operator to cycle through the array
  const cycledIndex = index % arr.length;

  // Ensure the index is non-negative
  const normalizedIndex =
    cycledIndex < 0 ? cycledIndex + arr.length : cycledIndex;

  return arr[normalizedIndex];
}
