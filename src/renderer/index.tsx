import { createRoot } from 'react-dom/client';
import invariant from 'tiny-invariant';
import { GlobalStateProvider } from './state';
import App from './App';

const container = document.getElementById('root');
invariant(container, 'root element not found');
const root = createRoot(container);
root.render(
  <GlobalStateProvider>
    <App />
  </GlobalStateProvider>
);
