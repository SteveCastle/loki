import React, { useContext } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from './state';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { SetupWizard } from './components/controls/setup-wizard';
import { Panels } from './components/layout/panels';
import './App.css';
import { Loader } from './components/layout/loader';
import HotKeyController from './components/controls/hotkey-controller';
import { ToastSystem } from './components/controls/toast-system';
import AutoPlayController from './components/controls/autoplay-controller';

const queryClient = new QueryClient();

export default function App(): JSX.Element {
  const { libraryService } = useContext(GlobalStateContext);
  const state = useSelector(
    libraryService,
    (state) => state,
    (a, b) => {
      return a.matches(b);
    }
  );
  if (state.matches({ library: 'manualSetup' })) return <SetupWizard />;

  const autoPlay = useSelector(
    libraryService,
    (state) => state.context.settings.autoPlay,
    (a, b) => {
      return a === b;
    }
  );

  return (
    <QueryClientProvider client={queryClient}>
      <DndProvider backend={HTML5Backend}>
        <Panels />
        {state.matches({ library: 'boot' }) ||
        state.matches({ library: 'selectingDB' }) ||
        state.matches({ library: 'loadingFromFS' }) ||
        state.matches({ library: 'loadingDB' }) ? (
          <Loader />
        ) : null}
        <HotKeyController />
        {autoPlay && <AutoPlayController />}
        <ToastSystem />
      </DndProvider>
    </QueryClientProvider>
  );
}
