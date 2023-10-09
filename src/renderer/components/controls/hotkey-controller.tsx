/* eslint-disable @typescript-eslint/no-empty-function */
import { useEffect, useState, useContext } from 'react';
import { useSelector } from '@xstate/react';
import filter from '../../filter';
import { GlobalStateContext, Item } from '../../state';
import { usePrevious } from '@react-hooks-library/core';

type KeyState = {
  [key: string]: boolean;
};

type Action = {
  down: (arg: Event) => void;
  up: (arg: Event) => void;
};

type ActionMap = {
  [key: string]: Action;
};

export default function HotKeyController() {
  const { libraryService } = useContext(GlobalStateContext);

  const { library, libraryLoadId, textFilter, activeTag, hotKeys } =
    useSelector(
      libraryService,
      (state) => ({
        library: state.context.library,
        libraryLoadId: state.context.libraryLoadId,
        textFilter: state.context.textFilter,
        activeTag: state.context.dbQuery.tags[0],
        settings: state.context.settings,
        hotKeys: state.context.hotKeys,
      }),
      (a, b) => a.libraryLoadId === b.libraryLoadId
    );

  const cursor = useSelector(
    libraryService,
    (state) => state.context.cursor,
    (a, b) => a === b
  );

  const filteredLibrary = filter(libraryLoadId, textFilter, library);

  const item = filteredLibrary[cursor];
  const [keyState, setKeyState] = useState<KeyState>({});
  const previousKeyState = usePrevious<KeyState>(keyState);

  const actions: ActionMap = {
    toggleTagPreview: {
      down: () =>
        libraryService.send({
          type: 'CHANGE_SETTING',
          data: { applyTagPreview: true },
        }),
      up: () =>
        libraryService.send({
          type: 'CHANGE_SETTING',
          data: { applyTagPreview: false },
        }),
    },
    toggleTagAll: {
      down: () =>
        libraryService.send({
          type: 'CHANGE_SETTING',
          data: { applyTagToAll: true },
        }),
      up: () =>
        libraryService.send({
          type: 'CHANGE_SETTING',
          data: { applyTagToAll: false },
        }),
    },
    incrementCursor: {
      down: () => libraryService.send({ type: 'INCREMENT_CURSOR', data: {} }),
      up: () => {},
    },
    decrementCursor: {
      down: () => libraryService.send({ type: 'DECREMENT_CURSOR', data: {} }),
      up: () => {},
    },
    toggleSound: {
      down: () => libraryService.send({ type: 'CHANGE_SETTING', data: {} }),
      up: () => {},
    },
    toggleControls: {
      down: () => libraryService.send({ type: 'CHANGE_SETTING', data: {} }),
      up: () => {},
    },
    shuffle: {
      down: () => libraryService.send({ type: 'SHUFFLE', data: {} }),
      up: () => {},
    },
    moveToTop: {
      down: () => {
        async function updateAssignmentWeight() {
          // New weight should be the number half way between 0 and the first item in the libraries weight.
          const newWeight = (library[0]?.weight || 1) / 2;
          await window.electron.ipcRenderer.invoke('update-assignment-weight', [
            item.path,
            activeTag,
            newWeight,
          ]);
          console.log(`set weight for ${item.path} to ${newWeight}`);
          libraryService.send({ type: 'SORTED_WEIGHTS' });
        }
        console.log(item);
        if (item && item.path) {
          console.log('move to top');
          updateAssignmentWeight();
        }
      },
      up: () => {},
    },
    moveToEnd: {
      down: () => {
        async function updateAssignmentWeight() {
          // New weight should be the number half way between 0 and the first item in the libraries weight.
          const newWeight = (library[library.length - 1]?.weight || 100) + 0.5;
          await window.electron.ipcRenderer.invoke('update-assignment-weight', [
            item.path,
            activeTag,
            newWeight,
          ]);
          console.log(`set weight for ${item.path} to ${newWeight}`);
          libraryService.send({ type: 'SORTED_WEIGHTS' });
        }
        if (item && item.path) {
          console.log('move to top');
          updateAssignmentWeight();
        }
      },
      up: () => {},
    },
    minimize: {
      down: (e) => {
        e.preventDefault();
        window.electron.ipcRenderer.sendMessage('minimize', []);
      },
      up: () => {},
    },
    copyFile: {
      down: (e) => {
        e.preventDefault();
        const copyContent = async (paths: string[]) => {
          try {
            console.log('ITEM PATH', paths);
            await window.electron.ipcRenderer.invoke(
              'copy-file-into-clipboard',
              [paths]
            );
            console.log('Content copied to clipboard');
          } catch (err) {
            console.error('Failed to copy: ', err);
          }
        };
        copyContent([item.path]);
      },
      up: () => {},
    },
    copyAllSelectedFiles: {
      down: (e) => {
        e.preventDefault();
        const copyContent = async (paths: string[]) => {
          try {
            console.log('ITEM PATH', paths);
            await window.electron.ipcRenderer.invoke(
              'copy-file-into-clipboard',
              [paths]
            );
            console.log('Content copied to clipboard');
          } catch (err) {
            console.error('Failed to copy: ', err);
          }
        };
        copyContent(filteredLibrary.map((item: Item) => item.path));
      },
      up: () => {},
    },
  };
  const actionByHotkey: ActionMap = Object.entries(hotKeys).reduce(
    (acc, [key, hotKey]) => {
      if (actions[key]) {
        acc[hotKey] = actions[key];
      }
      return acc;
    },
    {} as ActionMap
  );

  const handleKeyDown = (e: KeyboardEvent) => {
    const lowerCaseKey = e.key.toLowerCase();
    if (keyState[lowerCaseKey] === true) return;
    setKeyState((prev) => ({ ...prev, [lowerCaseKey]: true }));
  };

  const handleKeyUp = (e: KeyboardEvent) => {
    const lowerCaseKey = e.key.toLowerCase();
    setKeyState((prev) => ({ ...prev, [lowerCaseKey]: false }));
  };

  const handleBlur = () => {
    setKeyState({});
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [keyState, cursor, activeTag]);

  useEffect(() => {
    window.addEventListener('blur', handleBlur);

    return () => {
      window.addEventListener('blur', handleBlur);
    };
  }, []);

  useEffect(() => {
    // Construct a string of all active keys joined by a plus sign.
    const activeKeys = Object.entries(keyState)
      .filter(([key, value]) => value === true)
      .sort(([keyA], [keyB]) => keyA.localeCompare(keyB))
      .map(([key]) => key)
      .join('+');

    if (!previousKeyState) {
      actionByHotkey[activeKeys]?.down?.(new Event('down'));
    }

    if (previousKeyState) {
      const previousKeys = Object.entries(previousKeyState)
        .filter(([key, value]) => value === true)
        .sort(([keyA], [keyB]) => keyA.localeCompare(keyB))
        .map(([key]) => key)
        .join('+');

      if (activeKeys.length > previousKeys.length) {
        actionByHotkey[activeKeys]?.down?.(new Event('down'));
      }
      //Call the up action for any keys that were active before but are no longer active.
      actionByHotkey[previousKeys]?.up?.(new Event('up'));
    }
    // If the key state has just become true for a key, call the down action.
  }, [keyState, previousKeyState]);

  return null;
}
