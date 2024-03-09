import React, { useRef, useContext, useState } from 'react';
import { useSelector } from '@xstate/react';
import { Tooltip } from 'react-tooltip';
import useComponentSize from '@rehooks/component-size';
import { GlobalStateContext } from '../../state';
import filter from '../../filter';
import ProgressBar from './progress-bar';
import HotKeyOptions from './hotkey-options';
import useOnClickOutside from '../../hooks/useOnClickOutside';
import sound from '../../../../assets/sound-high.svg';
import gear from '../../../../assets/settings-3.svg';
import shuffle from '../../../../assets/shuffle.svg';
import db from '../../../../assets/database.svg';
import keyboard from '../../../../assets/keyboard.svg';
import grid from '../../../../assets/view-grid.svg';
import image from '../../../../assets/image-2-fill.svg';
import noSound from '../../../../assets/sound-off.svg';
import recursiveIcon from '../../../../assets/recursive.svg';
import folder from '../../../../assets/folder-open-fill.svg';

import Setting from './setting';
import { SETTINGS, SettingKey } from 'settings';

import './command-palette.css';
import DbPathWidget from './db-path';
import GridSizePicker from './gridsize-picker';
import CacheSetting from './cache-setting';
import { getFileType } from 'file-types';

function getDirectory(path: string): string {
  // Use JavaScript's built-in path-separation character as a regex pattern
  const separator = /[\\/]/;

  // Split the path into components
  const components = path.split(separator);

  // Remove the last component (the file)
  components.pop();

  // Reconstruct the path
  const directory = components.join('/');

  return directory;
}

export default function CommandPalette() {
  const { libraryService } = useContext(GlobalStateContext);
  const { display, position } = useSelector(
    libraryService,
    (state) => state.context.commandPalette
  );

  const tags = useSelector(
    libraryService,
    (state) => state.context.dbQuery.tags
  );

  const ref = useRef(null);
  const [tab, setTab] = useState('imageOptions');
  const { width, height } = useComponentSize(ref);
  const windowWidth = window.innerWidth;
  const windowHeight = window.innerHeight;

  const library = useSelector(libraryService, (state) =>
    filter(
      state.context.libraryLoadId,
      state.context.textFilter,
      state.context.library,
      state.context.settings.filters,
      state.context.settings.sortBy
    )
  );
  const cursor = useSelector(libraryService, (state) => state.context.cursor);
  const item = library[cursor];
  const fileType = item?.path ? getFileType(item.path) : '';
  const activeDirectory = useSelector(
    libraryService,
    (state) => state.context.initialFile
  );
  const { playSound, recursive, showControls } = useSelector(
    libraryService,
    (state) => state.context.settings
  );

  const isLoading = useSelector(libraryService, (state) =>
    state.matches('loadingFromFS')
  );

  const getMenuPosition = (x: number, y: number) => {
    const xOverlap = x + width - windowWidth;
    const yOverlap = y + height - windowHeight;
    return {
      left: xOverlap > 0 ? x - xOverlap : x,
      top: yOverlap > 0 ? y - yOverlap : y,
    };
  };

  useOnClickOutside(ref, () => {
    libraryService.send('HIDE_COMMAND_PALETTE');
  });

  return display ? (
    <div
      className="CommandPalette"
      ref={ref}
      tabIndex={-1}
      style={getMenuPosition(position.x, position.y)}
    >
      <div className="menuBar">
        <div className="windowControls">
          <span
            className="closeControl"
            onClick={() =>
              window.electron.ipcRenderer.sendMessage('shutdown', [])
            }
          />
          <span
            className="windowedControl"
            onClick={() => {
              window.electron.ipcRenderer.sendMessage('minimize', []);
            }}
          />
          <span
            className="fullScreenControl"
            onClick={() =>
              window.electron.ipcRenderer.sendMessage('toggle-fullscreen', [])
            }
          />
        </div>
        <div className="menuBarRight">
          <button
            className="menuIconButton"
            onClick={() => libraryService.send('SELECT_FILE')}
          >
            <img src={folder} />
          </button>
          <button
            data-tooltip-delay-show={500}
            data-tooltip-offset={20}
            data-tooltip-id="recursive"
            className={`menuIconButton ${recursive ? 'selected' : ''}`}
            onClick={() =>
              libraryService.send('CHANGE_SETTING_AND_RELOAD', {
                data: { recursive: !recursive },
              })
            }
          >
            <img src={recursiveIcon} />
          </button>
          <button
            data-tooltip-id="shuffle"
            data-tooltip-delay-show={500}
            data-tooltip-offset={20}
            className={`menuIconButton ${recursive ? 'selected' : ''}`}
            onClick={() => libraryService.send('SHUFFLE')}
          >
            <img src={shuffle} />
          </button>
          <button
            data-tooltip-id="sound"
            data-tooltip-delay-show={500}
            data-tooltip-offset={20}
            className="menuIconButton"
            onClick={() =>
              libraryService.send('CHANGE_SETTING', {
                data: { playSound: !playSound },
              })
            }
          >
            <img src={playSound ? sound : noSound} />
          </button>
        </div>
      </div>
      <div className="menuArea">
        <div className="menuContent">
          <span className="listContext">
            {Array.isArray(tags) && tags.length > 0
              ? tags.join(', ')
              : `${getDirectory(activeDirectory)}`}
          </span>
          <ProgressBar
            value={cursor}
            total={library.length}
            isLoading={isLoading}
            setCursor={(c) => {
              libraryService.send('SET_CURSOR', { idx: c });
            }}
          />
          {tab === 'imageOptions' && (
            <div className="tabContent">
              {Object.keys(SETTINGS)
                .filter((k) => SETTINGS[k as SettingKey].display === 'image')
                .map((settingKey) => (
                  <Setting
                    settingKey={settingKey as SettingKey}
                    key={settingKey}
                    reload={SETTINGS[settingKey as SettingKey].reload}
                    resetCursor={SETTINGS[settingKey as SettingKey].resetCursor}
                    currentItem={item}
                  />
                ))}
            </div>
          )}
          {tab === 'listViewOptions' && (
            <div className="tabContent">
              <GridSizePicker />
            </div>
          )}
          {tab === 'dbOptions' && (
            <div className="tabContent">
              <DbPathWidget />
              <CacheSetting />
            </div>
          )}
          {tab === 'hotKeyOptions' && <HotKeyOptions />}
          {tab === 'generalOptions' && (
            <div className="tabContent">
              <p>v2.1.0</p>
              {Object.keys(SETTINGS)
                .filter((k) => SETTINGS[k as SettingKey].display === 'general')
                .map((settingKey) => (
                  <Setting
                    settingKey={settingKey as SettingKey}
                    key={settingKey}
                    reload={SETTINGS[settingKey as SettingKey].reload}
                    resetCursor={SETTINGS[settingKey as SettingKey].resetCursor}
                    currentItem={item}
                  />
                ))}
            </div>
          )}
          <button
            data-tooltip-id="donate-buttons"
            data-tooltip-offset={20}
            data-tooltip-delay-show={500}
            className="donateButton"
            onClick={() => {
              window.electron.ipcRenderer.sendMessage('open-external', [
                'https://www.buymeacoffee.com/lowkeyviewer',
              ]);
            }}
          >
            Donate
          </button>
          <button
            data-tooltip-id="donate-buttons"
            data-tooltip-offset={20}
            data-tooltip-delay-show={500}
            className="patreonButton"
            onClick={() => {
              window.electron.ipcRenderer.sendMessage('open-external', [
                'https://www.patreon.com/lowkeyviewer',
              ]);
            }}
          >
            Patreon
          </button>
        </div>
        <div className="tabs">
          <button
            className={tab === 'imageOptions' ? 'active' : ''}
            onClick={() => setTab('imageOptions')}
          >
            <img src={image} />
          </button>
          <button
            className={tab === 'listViewOptions' ? 'active' : ''}
            onClick={() => setTab('listViewOptions')}
          >
            <img src={grid} />
          </button>
          <button
            className={tab === 'dbOptions' ? 'active' : ''}
            onClick={() => setTab('dbOptions')}
          >
            <img src={db} />
          </button>
          <button
            className={tab === 'hotKeyOptions' ? 'active' : ''}
            onClick={() => setTab('hotKeyOptions')}
          >
            <img src={keyboard} />
          </button>
          <button
            className={tab === 'generalOptions' ? 'active' : ''}
            onClick={() => setTab('generalOptions')}
          >
            <img src={gear} />
          </button>
        </div>
      </div>
      <Tooltip
        id="recursive"
        content={`Include files from all subdirectories.`}
        place="top"
      />
      <Tooltip
        id="shuffle"
        content={`Shuffle images in the list.`}
        place="top"
      />
      <Tooltip id="sound" content={`Play video audio.`} place="top" />
      <Tooltip
        id="donate-buttons"
        content={`Your donations make Lowkey Media Viewer possible!`}
        place="top"
      />
    </div>
  ) : null;
}
