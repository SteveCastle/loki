import { useContext } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../state';

// False for anonymous visitors on a public-access web server; components
// hide their write/job-launching UI when this is false. Always true in
// Electron and for signed-in users.
export function useCanWrite(): boolean {
  const { libraryService } = useContext(GlobalStateContext);
  return useSelector(libraryService, (s) => s.context.canWrite);
}
