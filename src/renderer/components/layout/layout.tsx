import { useContext, useState, useEffect, useMemo } from 'react';
import {
  Group as PanelGroup,
  Panel,
  Separator as PanelResizeHandle,
  usePanelRef,
  useDefaultLayout,
  type PanelProps,
  type PanelImperativeHandle,
} from 'react-resizable-panels';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { Detail } from '../detail/detail';
import filter from 'renderer/filter';
import { List } from '../list/list';
import './panels.css';
import Taxonomy from '../taxonomy/taxonomy';
import Metadata from '../metadata/metadata';
import CommandPalette from '../controls/command-palette';

function VerticalHandle() {
  return (
    <PanelResizeHandle className="handle vertical">
      <div className={'inner-handle'}></div>
    </PanelResizeHandle>
  );
}

function HorizontalHandle() {
  return (
    <PanelResizeHandle className="handle horizontal">
      <div className={'inner-handle'}></div>
    </PanelResizeHandle>
  );
}

function CollapsiblePanel({
  imperativeRef,
  renderId,
  ...props
}: PanelProps & { imperativeRef: React.RefObject<PanelImperativeHandle | null>; renderId: number }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted && imperativeRef?.current) {
      try {
        setCollapsed(imperativeRef.current.isCollapsed());
      } catch {
        // Panel not yet registered, ignore
      }
    }
  }, [mounted, renderId]);

  return (
    <Panel
      minSize="10%"
      collapsedSize="0%"
      className="panel"
      collapsible
      panelRef={imperativeRef}
      onResize={(size) => {
        setCollapsed(size.asPercentage === 0);
      }}
      {...props}
    >
      {!collapsed ? props.children : null}
    </Panel>
  );
}

const Layout = () => {
  const [renderID, setRenderID] = useState(0);
  const { libraryService } = useContext(GlobalStateContext);

  const { library } = useSelector(
    libraryService,
    (state) => {
      return {
        filters: state.context.settings.filters,
        sortBy: state.context.settings.sortBy,
        library: filter(
          state.context.libraryLoadId,
          state.context.textFilter,
          state.context.library,
          state.context.settings.filters,
          state.context.settings.sortBy
        ),
        libraryLoadId: state.context.libraryLoadId,
      };
    },
    (a, b) =>
      a.libraryLoadId === b.libraryLoadId &&
      a.filters === b.filters &&
      a.sortBy === b.sortBy
  );

  const comicMode = useSelector(
    libraryService,
    (state) => state.context.settings.comicMode
  );

  const battleMode = useSelector(
    libraryService,
    (state) => state.context.settings.battleMode
  );

  const controlMode = useSelector(
    libraryService,
    (state) => state.context.settings.controlMode,
    (a, b) => a === b
  );

  const libraryLayout = useSelector(
    libraryService,
    (state) => state.context.settings.libraryLayout
  );

  const electronStorage = useMemo(
    () => ({
      getItem(name: string) {
        try {
          const stored = window.electron.store.get(name, '') as string;
          if (!stored) return null;
          const parsed = JSON.parse(stored);
          return JSON.stringify(parsed[name] || null);
        } catch (error) {
          console.error(error);
          return null;
        }
      },
      setItem(name: string, value: string) {
        const encoded = JSON.stringify({
          [name]: JSON.parse(value),
        });
        window.electron.store.set(name, encoded);
      },
    }),
    []
  );

  const mainGroupLayout = useDefaultLayout({
    id: 'main-group',
    storage: electronStorage,
  });

  const innerGroupLayout = useDefaultLayout({
    id: 'inner-group',
    storage: electronStorage,
  });

  const detailRef = usePanelRef();
  const listRef = usePanelRef();
  const metaDataRef = usePanelRef();
  const taxonomyRef = usePanelRef();

  function handleListClick() {
    setRenderID((id) => id + 1);
    try {
      listRef.current?.resize('0%');
      detailRef.current?.resize('100%');
    } catch {
      // Panels not yet registered, ignore
    }
  }

  function handleDetailClick() {
    try {
      if (listRef.current?.isCollapsed()) {
        setRenderID((id) => id + 1);
        detailRef.current?.resize('0%');
        listRef.current?.resize('100%');
      } else {
        setRenderID((id) => id + 1);
        listRef.current?.resize('0%');
        detailRef.current?.resize('100%');
      }
    } catch {
      // Panels not yet registered, ignore
    }
  }

  useEffect(() => {
    if (library.length === 1) {
      // Delay to ensure panels are registered
      const timer = setTimeout(() => {
        handleListClick();
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [library]);

  return (
    <>
      <CommandPalette />
      <PanelGroup
        orientation="vertical"
        id="main-group"
        defaultLayout={mainGroupLayout.defaultLayout}
        onLayoutChange={mainGroupLayout.onLayoutChange}
      >
        <Panel className="panel" id="main-panel" defaultSize="100%">
          <PanelGroup
            orientation="horizontal"
            id="inner-group"
            defaultLayout={innerGroupLayout.defaultLayout}
            onLayoutChange={innerGroupLayout.onLayoutChange}
          >
          {libraryLayout === 'left' ? (
            <>
              <CollapsiblePanel
                id="taxonomy-panel"
                defaultSize="20%"
                imperativeRef={taxonomyRef}
                renderId={renderID}
              >
                <Taxonomy />
              </CollapsiblePanel>
              <VerticalHandle />
            </>
          ) : null}
          <CollapsiblePanel
            id="list-panel"
            className="panel"
            defaultSize="50%"
            imperativeRef={listRef}
            renderId={renderID}
          >
            <div className="panel" onDoubleClick={handleListClick}>
              <List />
            </div>
          </CollapsiblePanel>
          <VerticalHandle />
          <CollapsiblePanel
            id="detail-panel"
            className="panel"
            imperativeRef={detailRef}
            renderId={renderID}
          >
            <div
              className="panel"
              onDoubleClick={
                controlMode === 'mouse' ? handleDetailClick : undefined
              }
            >
              <Detail />
              {comicMode || battleMode ? <Detail offset={1} /> : null}
            </div>
          </CollapsiblePanel>
          <VerticalHandle />
          <CollapsiblePanel
            id="metadata-panel"
            imperativeRef={metaDataRef}
            renderId={renderID}
          >
            <Metadata />
          </CollapsiblePanel>
        </PanelGroup>
      </Panel>
        {libraryLayout === 'bottom' ? (
          <>
            <HorizontalHandle />
            <CollapsiblePanel
              id="taxonomy-panel-bottom"
              defaultSize="20%"
              imperativeRef={taxonomyRef}
              renderId={renderID}
            >
              <Taxonomy />
            </CollapsiblePanel>
          </>
        ) : null}
      </PanelGroup>
    </>
  );
};

export default Layout;
