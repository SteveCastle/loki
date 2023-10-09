import { useRef, memo, useState } from 'react';
import { GlobalStateProvider } from '../../state';
import {
  PanelGroup,
  Panel,
  PanelResizeHandle,
  ImperativePanelHandle,
  PanelProps,
} from 'react-resizable-panels';
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

function CollapsiblePanel(props: PanelProps) {
  const ref = useRef<ImperativePanelHandle>(null);
  const [collapsed, setCollapsed] = useState(false);

  function handleResize() {
    if (ref.current) {
      const size = ref.current.getSize();
      if (size < 5) {
        setCollapsed(true);
      } else {
        setCollapsed(false);
      }
    }
  }

  return (
    <Panel
      {...props}
      className="panel"
      collapsible
      minSize={0}
      ref={ref}
      onResize={handleResize}
    >
      {!collapsed ? props.children : null}
    </Panel>
  );
}

const DetailOnly = () => {
  return (
    <GlobalStateProvider>
      <CommandPalette />
      <Panel className="panel">
        <Detail />
      </Panel>
    </GlobalStateProvider>
  );
};

export default memo(DetailOnly);
