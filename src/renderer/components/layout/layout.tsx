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
import { store } from '../../platform';
import { Detail } from '../detail/detail';
import filter from 'renderer/filter';
import { List } from '../list/list';
import './panels.css';
import Taxonomy from '../taxonomy/taxonomy';
import Metadata from '../metadata/metadata';
import CommandPalette from '../controls/command-palette';
import ContextPalette from '../controls/context-palette';
import RegionSelect from '../controls/region-select';

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
      panelRef={imperativeRef as React.Ref<PanelImperativeHandle>}
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
          const stored = store.get(name, '') as string;
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
        store.set(name, encoded);
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

  // Collapse only the list panel — leave detail and metadata as-is. Used when
  // navigating to a single file: the list isn't useful with one item, but the
  // metadata panel (and its path tree, used to navigate) must stay open.
  function collapseList() {
    setRenderID((id) => id + 1);
    try {
      listRef.current?.resize('0%');
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

  // Touchpad mode's list/detail toggle: the detail view can't bind
  // onDoubleClick (single clicks advance the cursor there), so it detects a
  // fast double-click itself and asks for the toggle via this event.
  useEffect(() => {
    const handler = () => handleDetailClick();
    window.addEventListener('loki-toggle-list-detail', handler);
    return () => window.removeEventListener('loki-toggle-list-detail', handler);
    // handleDetailClick only touches stable refs/setters, so the first
    // render's closure stays valid.
  }, []);

  useEffect(() => {
    if (library.length === 1) {
      // Collapse the list to focus the single file. Only the list collapses;
      // the metadata panel stays open so its path tree remains usable for
      // navigation (and doesn't flash-and-vanish on each click).
      const timer = setTimeout(() => {
        collapseList();
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [library]);

  return (
    <>
      <CommandPalette />
      <ContextPalette />
      <RegionSelect />
      {/* defaultSize props below only apply on first run (or after clearing
          stored layouts) — once the user resizes anything, useDefaultLayout
          restores their persisted layout instead. First-run layout: list and
          detail split the width evenly, metadata starts collapsed, taxonomy
          gets ~20% (bottom strip or left column depending on libraryLayout). */}
      <PanelGroup
        orientation="vertical"
        id="main-group"
        defaultLayout={mainGroupLayout.defaultLayout}
        onLayoutChange={mainGroupLayout.onLayoutChange}
      >
        <Panel
          className="panel"
          id="main-panel"
          defaultSize={libraryLayout === 'bottom' ? '80%' : '100%'}
        >
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
            defaultSize={libraryLayout === 'left' ? '40%' : '50%'}
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
            defaultSize={libraryLayout === 'left' ? '40%' : '50%'}
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
            defaultSize="0%"
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
