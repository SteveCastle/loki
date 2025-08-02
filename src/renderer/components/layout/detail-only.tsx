import { memo } from 'react';
import { GlobalStateProvider } from '../../state';
import { Panel } from 'react-resizable-panels';
import { Detail } from '../detail/detail';

import './panels.css';
import CommandPalette from '../controls/command-palette';

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
