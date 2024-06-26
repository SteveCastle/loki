import { useRef, useContext, useMemo, useState, useEffect } from 'react';
import {
  PanelGroup,
  Panel,
  PanelResizeHandle,
  ImperativePanelHandle,
  PanelProps,
} from 'react-resizable-panels';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { Detail } from '../detail/detail';

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
  panelRef,
  renderId,
  ...props
}: PanelProps & { panelRef: any; renderId: number }) {
  const [collapsed, setCollapsed] = useState(panelRef?.current?.getCollapsed());

  useEffect(() => {
    if (panelRef?.current) {
      setCollapsed(panelRef.current?.getCollapsed());
    }
  }, [panelRef.current, renderId]);

  function handleCollapse() {
    setCollapsed(true);
  }

  return (
    <Panel
      minSize={10}
      collapsedSize={0}
      className="panel"
      collapsible
      ref={panelRef}
      onCollapse={handleCollapse}
      onResize={() => {
        setCollapsed(false);
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
  const state = useSelector(
    libraryService,
    (state) => state,
    (a, b) => {
      return a.matches(b);
    }
  );

  const comicMode = useSelector(
    libraryService,
    (state) => state.context.settings.comicMode
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
          const parsed = JSON.parse(
            window.electron.store.get(name, '') as string
          );
          return parsed[name] || '';
        } catch (error) {
          console.error(error);
          return '';
        }
      },
      setItem(name: string, value: any) {
        const encoded = JSON.stringify({
          [name]: value,
        });
        window.electron.store.set(name, encoded);
      },
    }),
    []
  );

  const detailRef = useRef<ImperativePanelHandle>(null);
  const listRef = useRef<ImperativePanelHandle>(null);
  const metaDataRef = useRef<ImperativePanelHandle>(null);
  const taxonomyRef = useRef<ImperativePanelHandle>(null);

  function handleListClick() {
    setRenderID((id) => id + 1);
    listRef.current?.resize(0);
    detailRef.current?.resize(100);
  }

  function handleDetailClick() {
    if (listRef.current?.getCollapsed()) {
      setRenderID((id) => id + 1);
      detailRef.current?.resize(0);
      listRef.current?.resize(100);
    } else {
      setRenderID((id) => id + 1);
      listRef.current?.resize(0);
      detailRef.current?.resize(100);
    }
  }

  return (
    <PanelGroup
      direction="vertical"
      disablePointerEventsDuringResize
      autoSaveId="panels"
      storage={electronStorage}
    >
      <CommandPalette />
      <Panel className="panel" order={0}>
        <PanelGroup
          direction="horizontal"
          disablePointerEventsDuringResize
          autoSaveId="nestedPanels"
          storage={electronStorage}
        >
          {libraryLayout === 'left' ? (
            <>
              <CollapsiblePanel
                defaultSize={20}
                order={0}
                panelRef={taxonomyRef}
                renderId={renderID}
              >
                <Taxonomy />
              </CollapsiblePanel>
              <VerticalHandle />
            </>
          ) : null}
          {!state.matches({ library: 'loadingFromFS' }) && (
            <>
              <CollapsiblePanel
                className="panel"
                defaultSize={50}
                order={1}
                panelRef={listRef}
                renderId={renderID}
              >
                <div className="panel" onDoubleClick={handleListClick}>
                  <List />
                </div>
              </CollapsiblePanel>
              <VerticalHandle />
            </>
          )}
          <CollapsiblePanel
            className="panel"
            order={2}
            panelRef={detailRef}
            renderId={renderID}
          >
            <div
              className="panel"
              onDoubleClick={
                controlMode === 'mouse' ? handleDetailClick : undefined
              }
            >
              <Detail />
              {comicMode ? <Detail offset={1} /> : null}
            </div>
          </CollapsiblePanel>
          <VerticalHandle />
          <CollapsiblePanel
            order={3}
            panelRef={metaDataRef}
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
            defaultSize={20}
            order={1}
            panelRef={taxonomyRef}
            renderId={renderID}
          >
            <Taxonomy />
          </CollapsiblePanel>
        </>
      ) : null}
    </PanelGroup>
  );
};

export default Layout;
