import { useContext, useEffect, useLayoutEffect, useRef, useState } from 'react';
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
import DescriptionOverlay from './description-overlay';

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
  startTime = 0,
  version = 0
) {
  if (getFileType(path, Boolean(imageCache)) === FileTypes.Video) {
    return (
      <Video
        key={`${path}-${version}`}
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
        version={version}
        useHLS={settings.useHLS}
      />
    );
  }
  if (getFileType(path) === FileTypes.Audio) {
    return (
      <Audio
        key={`${path}-${version}`}
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
          version={version}
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
        version={version}
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

  // Version counter used to bust browser HTTP cache when a file is updated on
  // disk (e.g. save task overwrites the file). Incremented in response to the
  // `loki-media-updated` custom event dispatched by the SSE handler.
  const [mediaVersion, setMediaVersion] = useState(0);

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

  // Listen for media-updated SSE events to reload the current file in real time.
  useEffect(() => {
    const handler = (e: Event) => {
      const paths: string[] = (e as CustomEvent).detail?.paths || [];
      if (item && paths.includes(item.path)) {
        setMediaVersion((v) => v + 1);
      }
    };
    window.addEventListener('loki-media-updated', handler);
    return () => window.removeEventListener('loki-media-updated', handler);
  }, [item?.path]);

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
    // Key on `item?.path` (not `containerRef.current`): the container is only
    // rendered once `item` is truthy (see the `if (!item) return null` early
    // return), and ref assignment doesn't trigger a re-render. Depending on the
    // ref meant the effect ran once while the ref was still null (on the initial
    // mount before the library loaded) and never re-ran when the container
    // mounted — so the wheel listener never attached until an unrelated re-render
    // (e.g. toggling controlMode) happened to re-run it.
  }, [item?.path, settings.controlMode]);

  // Touchpad mode: clicks advance the cursor, so the native onDoubleClick
  // the mouse mode uses for the list/detail toggle can't be bound (it would
  // fire after two advances). Instead a very fast second click IS the
  // toggle: each advance is deferred by a tight window, and a second click
  // inside it cancels the advance and toggles the view. The window is
  // deliberately short — rapid click-click-click browsing sits well above
  // it; only a deliberate double-click lands under it.
  const TOGGLE_DOUBLE_CLICK_MS = 200;
  const pendingAdvanceRef = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (pendingAdvanceRef.current !== null) {
        window.clearTimeout(pendingAdvanceRef.current);
      }
    },
    []
  );

  function handleClick(e: React.MouseEvent<HTMLDivElement, MouseEvent>) {
    if (pendingAdvanceRef.current !== null) {
      // Second click within the window — this is a toggle, not two steps.
      window.clearTimeout(pendingAdvanceRef.current);
      pendingAdvanceRef.current = null;
      // The panel refs live in Layout; same custom-event channel as
      // 'loki-media-updated'.
      window.dispatchEvent(new Event('loki-toggle-list-detail'));
      return;
    }
    const rect = containerRef.current!.getBoundingClientRect();
    const decrement = e.clientX - rect.left < rect.width / 2;
    pendingAdvanceRef.current = window.setTimeout(() => {
      pendingAdvanceRef.current = null;
      libraryService.send(decrement ? 'DECREMENT_CURSOR' : 'INCREMENT_CURSOR');
    }, TOGGLE_DOUBLE_CLICK_MS);
  }

  const { orientation } = useMediaDimensions(
    mediaRef as React.RefObject<HTMLImageElement | HTMLVideoElement>
  );
  const [coverSize, setCoverSize] = useState({ width: 0, height: 0 });

  // Relative pan position as a fraction of the scrollable range on each axis
  // (0 = top/left, 0.5 = centered, 1 = bottom/right). Kept across media
  // changes so panning to a corner stays in that corner on the next item;
  // same-sized media restores the identical pixel offset, differently-sized
  // media lands at the same relative spot.
  const panFractionRef = useRef({ x: 0.5, y: 0.5 });

  const applyPan = () => {
    const container = containerRef.current;
    if (container === null) {
      return;
    }
    const maxX = container.scrollWidth - container.clientWidth;
    const maxY = container.scrollHeight - container.clientHeight;
    if (maxX > 0) {
      container.scrollLeft = panFractionRef.current.x * maxX;
    }
    if (maxY > 0) {
      container.scrollTop = panFractionRef.current.y * maxY;
    }
  };

  // Record the pan fraction from actual scrolls (drag-pan and touchpad
  // scrolling both land here). Only record axes that currently overflow —
  // when the media collapses between items the browser clamps scroll to 0,
  // and recording that would wipe the remembered position.
  useEffect(() => {
    const container = containerRef.current;
    if (container === null) {
      return;
    }
    const recordPan = () => {
      const maxX = container.scrollWidth - container.clientWidth;
      const maxY = container.scrollHeight - container.clientHeight;
      if (maxX > 0) {
        panFractionRef.current.x = container.scrollLeft / maxX;
      }
      if (maxY > 0) {
        panFractionRef.current.y = container.scrollTop / maxY;
      }
    };
    container.addEventListener('scroll', recordPan, { passive: true });
    return () => container.removeEventListener('scroll', recordPan);
    // Keyed on `item?.path` for the same mount-timing reason as the wheel
    // listener effect below.
  }, [item?.path]);

  // Re-apply the pan after a coverSize change resizes the media element.
  // handleResize applies it at load time, but the element only takes its new
  // dimensions on the re-render that setCoverSize triggers; a layout effect
  // runs after that layout and before paint, so no repositioning is visible.
  useLayoutEffect(() => {
    applyPan();
  }, [coverSize.width, coverSize.height]);

  const handleResize = () => {
    const parentDiv = containerRef.current;
    const media = mediaRef.current;

    if (media === null || parentDiv === null) {
      return;
    }

    // Audio elements don't need visual resizing
    if (media instanceof HTMLAudioElement) {
      return;
    }

    const parentWidth = parentDiv.clientWidth;
    const parentHeight = parentDiv.clientHeight;
    // Restore the remembered relative pan instead of recentering; for
    // media the same size as the previous item this writes the value the
    // scroll already has, so nothing moves.
    applyPan();

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
        settings.comicMode || settings.battleMode ? 'dualPane' : '',
        settings.controlMode === 'mouse' ? 'grabbable' : '',
      ].join(' ')}
    >
      {settings.battleMode ? <BattleMode item={item} offset={offset} /> : null}
      <div
        className="Detail"
        onContextMenu={(e) => {
          e.preventDefault();
          e.stopPropagation();
          if (e.shiftKey) {
            libraryService.send('SHOW_CONTEXT_PALETTE', {
              position: { x: e.clientX, y: e.clientY },
              target: item ? { type: 'file', path: item.path } : { type: 'library' },
            });
          } else {
            libraryService.send('SHOW_COMMAND_PALETTE', {
              position: { x: e.clientX, y: e.clientY },
            });
          }
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
          item.timeStamp,
          mediaVersion
        )}
      </div>
      {settings.showDescriptionOverlay ? (
        <DescriptionOverlay
          path={item.path}
          fontSize={settings.descriptionOverlaySize}
          sidePadding={settings.descriptionOverlayPadding}
        />
      ) : null}
      {!settings.showControls &&
        (getFileType(item.path) === 'video' ||
          getFileType(item.path) === 'audio') && (
          <div className="videoControls">
            <VideoControls
              mediaRef={mediaRef as React.RefObject<HTMLMediaElement>}
            />
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
