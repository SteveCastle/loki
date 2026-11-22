import { useContext, useState, useEffect, useMemo } from 'react';
import {
  Group as PanelGroup,
  Panel,
  Separator as PanelResizeHandle,
  usePanelRef,
  useGroupRef,
  useDefaultLayout,
  type PanelProps,
  type PanelImperativeHandle,
} from 'react-resizable-panels';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { appArgs, store } from '../../platform';
import { Detail } from '../detail/detail';
import filter from 'renderer/filter';
import { List } from '../list/list';
import './panels.css';
import Taxonomy from '../taxonomy/taxonomy';
import Metadata from '../metadata/metadata';
import CommandPalette from '../controls/command-palette';
import ContextPalette from '../controls/context-palette';
import TransformFlow from '../transform/transform-flow';
import EngineSetupDialog from '../transform/engine-setup-dialog';
import RegionSelect from '../controls/region-select';
import SearchOverlay from './search-overlay';

// Persisted panel layout, read through an in-memory cache.
//
// react-resizable-panels reads this via useSyncExternalStore, which React calls
// on EVERY render pass of the panel tree — and `store.get` is
// `ipcRenderer.sendSync`, a blocking round trip to the main process. So every
// re-render of the layout (opening the command palette, for one) fired several
// synchronous IPCs. That is free while the main process is idle and brutal
// while it isn't: measured at 223ms of blocked renderer during the first
// palette open of a session, when main is still finishing startup work. It is
// also why only the FIRST open felt slow.
//
// This process is the only writer, so the cache can never go stale: setItem
// updates it in lockstep. One IPC per key per session instead of one per render.
const layoutCache = new Map<string, string | null>();

const electronPanelStorage = {
  getItem(name: string): string | null {
    if (layoutCache.has(name)) return layoutCache.get(name) ?? null;
    let value: string | null = null;
    try {
      const stored = store.get(name, '') as string;
      if (stored) {
        const parsed = JSON.parse(stored);
        const layout = parsed[name] || null;
        // A saved layout with neither list nor detail visible shows nothing
        // but side panels and has no double-click to recover from; fall back
        // to the defaults instead.
        const hidesMedia =
          layout &&
          'list-panel' in layout &&
          (layout['list-panel'] || 0) + (layout['detail-panel'] || 0) < 1;
        value = hidesMedia ? null : JSON.stringify(layout);
      }
    } catch (error) {
      console.error(error);
      value = null;
    }
    layoutCache.set(name, value);
    return value;
  },
  setItem(name: string, value: string): void {
    const encoded = JSON.stringify({ [name]: JSON.parse(value) });
    // Cache the shape getItem returns, not the raw wrapper.
    layoutCache.set(name, JSON.stringify(JSON.parse(value) || null));
    store.set(name, encoded);
  },
};

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
  // Starts OPEN, deliberately. Defaulting to collapsed for zero-default panels
  // (the metadata panel) saved one wasted mount per launch, but it made
  // "collapsed" the failure mode: if imperativeRef isn't registered when the
  // effect below runs, or isCollapsed() throws, the panel never opens and a
  // restored layout silently loses it. Starting open means the worst case is a
  // panel that flashes and closes, not one the user can't get back.
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

  // A folder load starts from a one-item library (just the opened file) and
  // streams the rest in, so "one item" only means something once it ends.
  const loadingFromFS = useSelector(libraryService, (state) =>
    state.matches({ library: 'loadingFromFS' })
  );

  const libraryLayout = useSelector(
    libraryService,
    (state) => state.context.settings.libraryLayout
  );

  const electronStorage = useMemo(() => electronPanelStorage, []);

  const savedMainGroupLayout = useDefaultLayout({
    id: 'main-group',
    storage: electronStorage,
  });

  const savedInnerGroupLayout = useDefaultLayout({
    id: 'inner-group',
    storage: electronStorage,
  });

  // A window opened on one file starts with only the detail panel showing,
  // and leaves the saved layout (the main window's) untouched.
  const detailOnly = useMemo(() => {
    if (!appArgs.openedForFile) return null;
    const main: Record<string, number> = { 'main-panel': 100 };
    if (libraryLayout === 'bottom') main['taxonomy-panel-bottom'] = 0;
    const inner: Record<string, number> = {
      'list-panel': 0,
      'detail-panel': 100,
      'metadata-panel': 0,
    };
    if (libraryLayout === 'left') inner['taxonomy-panel'] = 0;
    return { main, inner };
    // Only the first layout matters: it seeds the groups once.
  }, []);
  const mainGroupLayout = detailOnly
    ? { defaultLayout: detailOnly.main, onLayoutChange: undefined }
    : savedMainGroupLayout;
  const innerGroupLayout = detailOnly
    ? { defaultLayout: detailOnly.inner, onLayoutChange: undefined }
    : savedInnerGroupLayout;

  const detailRef = usePanelRef();
  const listRef = usePanelRef();
  const metaDataRef = usePanelRef();
  const taxonomyRef = usePanelRef();
  const innerGroupRef = useGroupRef();

  // Give the list + detail share of the row to one of them and collapse the
  // other. Taxonomy and metadata keep their sizes: resizing a single panel to
  // 100% would squeeze every other panel in the group shut.
  function showOnly(show: 'list-panel' | 'detail-panel') {
    const hide = show === 'list-panel' ? 'detail-panel' : 'list-panel';
    const layout = innerGroupRef.current?.getLayout();
    if (!layout || !(show in layout)) return;
    const sides = Object.keys(layout).filter((id) => id !== show && id !== hide);
    const sidesTotal = sides.reduce((sum, id) => sum + (layout[id] ?? 0), 0);
    let total = (layout[show] ?? 0) + (layout[hide] ?? 0);
    // Both collapsed (side panels fill the row): split the row with them
    // rather than leave nothing to look at.
    if (total < 1) total = sidesTotal > 0 ? 50 : 100;
    const rest = 100 - total;
    const next: Record<string, number> = { [show]: total, [hide]: 0 };
    for (const id of sides) {
      next[id] = sidesTotal > 0 ? ((layout[id] ?? 0) / sidesTotal) * rest : 0;
    }
    innerGroupRef.current?.setLayout(next);
  }

  function handleListClick() {
    setRenderID((id) => id + 1);
    try {
      showOnly('detail-panel');
    } catch {
      // Panels not yet registered, ignore
    }
  }

  // Collapse only the list panel — leave taxonomy and metadata as-is. Used
  // when the library holds a single file: the list isn't useful with one
  // item, but the metadata panel (and its path tree, used to navigate) must
  // stay open. Goes through showOnly so the list's width lands on the detail
  // panel: a bare resize('0%') hands it to whichever neighbour the library
  // picks, which is the taxonomy panel when detail is collapsed.
  function collapseList() {
    try {
      if (listRef.current?.isCollapsed()) return;
      setRenderID((id) => id + 1);
      showOnly('detail-panel');
    } catch {
      // Panels not yet registered, ignore
    }
  }

  function handleDetailClick() {
    try {
      setRenderID((id) => id + 1);
      showOnly(listRef.current?.isCollapsed() ? 'list-panel' : 'detail-panel');
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

  // Touch devices can't double-click a list item to open it full-screen;
  // each item shows a corner control instead (list-item.tsx) that asks for
  // the same list-collapse/detail-expand through this event.
  useEffect(() => {
    const handler = () => handleListClick();
    window.addEventListener('loki-open-detail', handler);
    return () => window.removeEventListener('loki-open-detail', handler);
  }, []);

  useEffect(() => {
    if (library.length === 1 && !loadingFromFS) {
      // Collapse the list to focus the single file. Only the list collapses;
      // the metadata panel stays open so its path tree remains usable for
      // navigation (and doesn't flash-and-vanish on each click).
      const timer = setTimeout(() => {
        collapseList();
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [library, loadingFromFS]);

  return (
    <>
      <CommandPalette />
      <ContextPalette />
      <TransformFlow />
      <EngineSetupDialog />
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
            groupRef={innerGroupRef}
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
            <div
              className="panel search-overlay-host"
              onDoubleClick={handleListClick}
            >
              <List />
              <SearchOverlay />
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
              className="panel search-overlay-host"
              onDoubleClick={
                controlMode === 'mouse' ? handleDetailClick : undefined
              }
            >
              <Detail />
              {comicMode || battleMode ? <Detail offset={1} /> : null}
              <SearchOverlay minimal />
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
