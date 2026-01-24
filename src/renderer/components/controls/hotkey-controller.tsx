/* eslint-disable @typescript-eslint/no-empty-function */
import { useEffect, useState, useContext } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useSelector } from '@xstate/react';
import filter from '../../filter';
import { GlobalStateContext, Item } from '../../state';

type Action = {
  down: (arg: Event) => void;
  up: (arg: Event) => void;
};

type ActionMap = {
  [key: string]: Action;
};

export default function HotKeyController() {
  const { libraryService } = useContext(GlobalStateContext);
  const queryClient = useQueryClient();
  const [keysPressed, setKeysPressed] = useState<Set<string>>(new Set());
  const { library, libraryLoadId, textFilter, activeTag, hotKeys } =
    useSelector(
      libraryService,
      (state) => ({
        library: state.context.library,
        libraryLoadId: state.context.libraryLoadId,
        textFilter: state.context.textFilter,
        activeTag: state.context.dbQuery.tags[0],
        hotKeys: state.context.hotKeys,
      }),
      (a, b) => a.libraryLoadId === b.libraryLoadId
    );

  const settings = useSelector(
    libraryService,
    (state) => state.context.settings
  );

  const activeCategory = useSelector(
    libraryService,
    (state) => state.context.activeCategory,
    (a, b) => a === b
  );

  const storedCategories = useSelector(
    libraryService,
    (state) => state.context.storedCategories,
    (a, b) => a === b
  );

  const storedTags = useSelector(
    libraryService,
    (state) => state.context.storedTags,
    (a, b) => a === b
  );

  const mostRecentTag = useSelector(
    libraryService,
    (state) => state.context.mostRecentTag,
    (a, b) => a === b
  );

  const mostRecentCategory = useSelector(
    libraryService,
    (state) => state.context.mostRecentCategory,
    (a, b) => a === b
  );

  const activeTags = useSelector(
    libraryService,
    (state) => state.context.dbQuery.tags,
    (a, b) => JSON.stringify(a) === JSON.stringify(b)
  );

  const cursor = useSelector(
    libraryService,
    (state) => state.context.cursor,
    (a, b) => a === b
  );

  const filteredLibrary = filter(
    libraryLoadId,
    textFilter,
    library,
    settings.filters,
    settings.sortBy
  );

  const item = filteredLibrary[cursor];

  // Helper function to create assignments
  const createAssignments = async (tags: string[], itemPath: string) => {
    for (const tag of tags) {
      await window.electron.ipcRenderer.invoke('create-assignment', [
        [itemPath],
        tag,
        mostRecentCategory,
        null,
        false,
      ]);
      queryClient.invalidateQueries({
        queryKey: ['taxonomy', 'tag', tag],
      });
    }
    queryClient.invalidateQueries({ queryKey: ['metadata'] });
    queryClient.invalidateQueries({
      queryKey: ['tags-by-path', item.path],
    });
  };

  // Helper function to apply multiple tags
  const createApplyTagAction = (position: string) => ({
    down: (e: Event) => {
      e.preventDefault();
      e.stopPropagation();
      const tags = storedTags[position];
      if (tags && tags.length > 0 && item?.path) {
        createAssignments(tags, item.path);
      }
    },
    up: () => {},
  });

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
    applyMostRecentTag: {
      down: (e) => {
        e.preventDefault();
        e.stopPropagation();
        async function createAssignment() {
          console.log('creating assignment', item.path, mostRecentTag);
          await window.electron.ipcRenderer.invoke('create-assignment', [
            [item.path],
            mostRecentTag,
            mostRecentCategory,
            null,
            false,
          ]);
          queryClient.invalidateQueries({ queryKey: ['metadata'] });
          queryClient.invalidateQueries({
            queryKey: ['taxonomy', 'tag', mostRecentTag],
          });
          queryClient.invalidateQueries({
            queryKey: ['tags-by-path', item.path],
          });
        }
        createAssignment();
      },
      up: () => {},
    },
    storeCategory1: {
      down: () =>
        libraryService.send({
          type: 'STORE_CATEGORY',
          data: { category: activeCategory, position: '1' },
        }),
      up: () => {},
    },
    tagCategory1: {
      down: () => {
        libraryService.send({
          type: 'SET_ACTIVE_CATEGORY',
          data: { category: storedCategories['1'] },
        });
      },
      up: () => {},
    },
    storeCategory2: {
      down: () =>
        libraryService.send({
          type: 'STORE_CATEGORY',
          data: { category: activeCategory, position: '2' },
        }),
      up: () => {},
    },
    tagCategory2: {
      down: () => {
        libraryService.send({
          type: 'SET_ACTIVE_CATEGORY',
          data: { category: storedCategories['2'] },
        });
      },
      up: () => {},
    },
    storeCategory3: {
      down: () =>
        libraryService.send({
          type: 'STORE_CATEGORY',
          data: { category: activeCategory, position: '3' },
        }),
      up: () => {},
    },
    tagCategory3: {
      down: () => {
        libraryService.send({
          type: 'SET_ACTIVE_CATEGORY',
          data: { category: storedCategories['3'] },
        });
      },
      up: () => {},
    },
    storeCategory4: {
      down: () =>
        libraryService.send({
          type: 'STORE_CATEGORY',
          data: { category: activeCategory, position: '4' },
        }),
      up: () => {},
    },
    tagCategory4: {
      down: () => {
        libraryService.send({
          type: 'SET_ACTIVE_CATEGORY',
          data: { category: storedCategories['4'] },
        });
      },
      up: () => {},
    },
    storeCategory5: {
      down: () =>
        libraryService.send({
          type: 'STORE_CATEGORY',
          data: { category: activeCategory, position: '5' },
        }),
      up: () => {},
    },
    tagCategory5: {
      down: () => {
        libraryService.send({
          type: 'SET_ACTIVE_CATEGORY',
          data: { category: storedCategories['5'] },
        });
      },
      up: () => {},
    },
    storeCategory6: {
      down: () =>
        libraryService.send({
          type: 'STORE_CATEGORY',
          data: { category: activeCategory, position: '6' },
        }),
      up: () => {},
    },
    tagCategory6: {
      down: () => {
        libraryService.send({
          type: 'SET_ACTIVE_CATEGORY',
          data: { category: storedCategories['6'] },
        });
      },
      up: () => {},
    },
    storeCategory7: {
      down: () =>
        libraryService.send({
          type: 'STORE_CATEGORY',
          data: { category: activeCategory, position: '7' },
        }),
      up: () => {},
    },
    tagCategory7: {
      down: () => {
        libraryService.send({
          type: 'SET_ACTIVE_CATEGORY',
          data: { category: storedCategories['7'] },
        });
      },
      up: () => {},
    },
    storeCategory8: {
      down: () =>
        libraryService.send({
          type: 'STORE_CATEGORY',
          data: { category: activeCategory, position: '8' },
        }),
      up: () => {},
    },
    tagCategory8: {
      down: () => {
        libraryService.send({
          type: 'SET_ACTIVE_CATEGORY',
          data: { category: storedCategories['8'] },
        });
      },
      up: () => {},
    },
    storeCategory9: {
      down: () =>
        libraryService.send({
          type: 'STORE_CATEGORY',
          data: { category: activeCategory, position: '9' },
        }),
      up: () => {},
    },
    tagCategory9: {
      down: () => {
        libraryService.send({
          type: 'SET_ACTIVE_CATEGORY',
          data: { category: storedCategories['9'] },
        });
      },
      up: () => {},
    },
    storeTag1: {
      down: () =>
        libraryService.send({
          type: 'STORE_TAG',
          data: { tags: activeTags, position: '1' },
        }),
      up: () => {},
    },
    storeTag2: {
      down: () =>
        libraryService.send({
          type: 'STORE_TAG',
          data: { tags: activeTags, position: '2' },
        }),
      up: () => {},
    },
    storeTag3: {
      down: () =>
        libraryService.send({
          type: 'STORE_TAG',
          data: { tags: activeTags, position: '3' },
        }),
      up: () => {},
    },
    storeTag4: {
      down: () =>
        libraryService.send({
          type: 'STORE_TAG',
          data: { tags: activeTags, position: '4' },
        }),
      up: () => {},
    },
    storeTag5: {
      down: () =>
        libraryService.send({
          type: 'STORE_TAG',
          data: { tags: activeTags, position: '5' },
        }),
      up: () => {},
    },
    storeTag6: {
      down: () =>
        libraryService.send({
          type: 'STORE_TAG',
          data: { tags: activeTags, position: '6' },
        }),
      up: () => {},
    },
    storeTag7: {
      down: () =>
        libraryService.send({
          type: 'STORE_TAG',
          data: { tags: activeTags, position: '7' },
        }),
      up: () => {},
    },
    storeTag8: {
      down: () =>
        libraryService.send({
          type: 'STORE_TAG',
          data: { tags: activeTags, position: '8' },
        }),
      up: () => {},
    },
    storeTag9: {
      down: () =>
        libraryService.send({
          type: 'STORE_TAG',
          data: { tags: activeTags, position: '9' },
        }),
      up: () => {},
    },
    applyTag1: createApplyTagAction('1'),
    applyTag2: createApplyTagAction('2'),
    applyTag3: createApplyTagAction('3'),
    applyTag4: createApplyTagAction('4'),
    applyTag5: createApplyTagAction('5'),
    applyTag6: createApplyTagAction('6'),
    applyTag7: createApplyTagAction('7'),
    applyTag8: createApplyTagAction('8'),
    applyTag9: createApplyTagAction('9'),
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
    deleteFile: {
      down: (e) => {
        e.preventDefault();
        libraryService.send({
          type: 'DELETE_FILE',
          data: { path: item.path },
        });
      },
      up: () => {},
    },
    togglePlayPause: {
      down: (e) => {
        e.preventDefault();
        e.stopPropagation();
        libraryService.send({
          type: 'TOGGLE_PLAY_PAUSE',
        });
      },
      up: () => {},
    },
    refreshLibrary: {
      down: (e) => {
        e.preventDefault();
        e.stopPropagation();
        libraryService.send({
          type: 'REFRESH_LIBRARY',
        });
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
    let mainKey = e.key.toLowerCase();

    // Use e.code for digits to ensure consistent behavior across different keyboard layouts
    // and when modifiers like Shift are pressed
    if (e.code.startsWith('Digit')) {
      mainKey = e.code.slice(5);
    }

    // Handle modifier keys separately for toggle actions
    if (['shift', 'control'].includes(mainKey)) {
      const newKeysPressed = new Set(keysPressed);
      newKeysPressed.add(mainKey);
      setKeysPressed(newKeysPressed);

      // Only trigger toggle actions if no other modifier keys are pressed
      const isShiftOnly =
        mainKey === 'shift' && !e.ctrlKey && !e.altKey && !e.metaKey;
      const isControlOnly =
        mainKey === 'control' && !e.shiftKey && !e.altKey && !e.metaKey;

      if (isShiftOnly && hotKeys.toggleTagPreview === 'shift') {
        actions.toggleTagPreview.down(e);
      }
      if (isControlOnly && hotKeys.toggleTagAll === 'control') {
        actions.toggleTagAll.down(e);
      }
      return;
    }

    // If any non-modifier key is pressed while shift/ctrl are held, turn off their toggles
    if (keysPressed.has('shift') && hotKeys.toggleTagPreview === 'shift') {
      actions.toggleTagPreview.up(e);
    }
    if (keysPressed.has('control') && hotKeys.toggleTagAll === 'control') {
      actions.toggleTagAll.up(e);
    }

    // Don't process other modifier keys as main keys
    if (['alt', 'meta'].includes(mainKey)) {
      return;
    }

    // Check if we're currently focused on any input element
    const target = e.target as HTMLElement;
    const isInputElement =
      target &&
      (target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.contentEditable === 'true' ||
        target.classList.contains('time-input') ||
        target.classList.contains('text-input'));

    // For most keys, if we're in an input element, let the input handle it
    // Exception: Allow Ctrl+combinations and Escape to still work for shortcuts
    if (isInputElement && !e.ctrlKey && !e.metaKey && mainKey !== 'escape') {
      return; // Let the input handle the key
    }

    const keys: string[] = [mainKey === 'space' ? ' ' : mainKey];

    // Add modifier keys in the format expected by the existing system
    // Order must match the hotkey configuration format: key+alt+control+shift
    if (e.altKey) keys.push('alt');
    if (e.ctrlKey || e.metaKey) keys.push('control'); // Handle both Ctrl and Cmd
    if (e.shiftKey) keys.push('shift');

    const keyCombo = keys.join('+');

    // Check if this combination has an action and execute it immediately
    if (actionByHotkey[keyCombo]) {
      e.preventDefault();
      e.stopPropagation();
      actionByHotkey[keyCombo].down(e);
    }
  };

  const handleKeyUp = (e: KeyboardEvent) => {
    const mainKey = e.key.toLowerCase();

    // Handle modifier key releases for toggle actions
    if (['shift', 'control'].includes(mainKey)) {
      const newKeysPressed = new Set(keysPressed);
      newKeysPressed.delete(mainKey);
      setKeysPressed(newKeysPressed);

      // Always turn off toggle actions when the modifier key is released
      if (mainKey === 'shift' && hotKeys.toggleTagPreview === 'shift') {
        actions.toggleTagPreview.up(e);
      }
      if (mainKey === 'control' && hotKeys.toggleTagAll === 'control') {
        actions.toggleTagAll.up(e);
      }
      return;
    }
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [actionByHotkey]); // Only depend on actionByHotkey changes

  // Key state processing is now handled directly in handleKeyDown for better reliability

  return null;
}
