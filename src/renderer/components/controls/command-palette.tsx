import React, {
  useRef,
  useContext,
  useState,
  useCallback,
  useMemo,
} from 'react';
import { useSelector } from '@xstate/react';
import { Tooltip } from 'react-tooltip';
import useComponentSize from '@rehooks/component-size';

// State & Hooks
import { GlobalStateContext } from '../../state';
import useOnClickOutside from '../../hooks/useOnClickOutside';
import filter from '../../filter';

// Child Components
import ProgressBar from './progress-bar';
import HotKeyOptions from './hotkey-options';
import Setting from './setting';
import DbPathWidget from './db-path';
import GridSizePicker from './gridsize-picker';
import CacheSetting from './cache-setting';

// Assets (Icons)
import soundIcon from '../../../../assets/sound-high.svg';
import gearIcon from '../../../../assets/settings-3.svg';
import shuffleIcon from '../../../../assets/shuffle.svg';
import dbIcon from '../../../../assets/database.svg';
import keyboardIcon from '../../../../assets/keyboard.svg';
import autoplayIcon from '../../../../assets/autoplay.svg'; // Note: Used for 'autoPlayOptions' tab, but content missing
import gridIcon from '../../../../assets/view-grid.svg';
import imageIcon from '../../../../assets/image-2-fill.svg';
import noSoundIcon from '../../../../assets/sound-off.svg';
import recursiveIcon from '../../../../assets/recursive.svg';
import folderIcon from '../../../../assets/folder-open-fill.svg';
import lockIcon from '../../../../assets/lock-fill.svg';
import tag from '../../../../assets/tag.svg';

// Settings & Types
import { SETTINGS, SettingKey, clampVolume } from 'settings'; // Assuming SETTINGS is an object and SettingKey is a type

// Styles
import './command-palette.css';

// --- Helper Functions ---

/**
 * Extracts the directory path from a full file path.
 * If the path is a file (has extension), returns the parent directory.
 * If the path is a directory (no extension), returns the path as-is.
 * @param path The full file path or directory path.
 * @returns The directory path.
 */
function getDirectory(path: string): string {
  if (!path) return '';
  const separator = /[\\/]/;
  const components = path.split(separator);
  const lastComponent = components[components.length - 1];

  // Check if the last component has a file extension
  // A file extension is indicated by a dot followed by alphanumeric characters
  const hasExtension = /\.[a-zA-Z0-9]+$/.test(lastComponent);

  if (hasExtension) {
    // It's a file, remove the filename to get the directory
    components.pop();
  }
  // If no extension, assume it's already a directory path

  return components.join('/');
}

// --- Types ---

type TabType =
  | 'imageOptions'
  | 'listViewOptions'
  | 'dbOptions'
  | 'autoPlayOptions' // Added for completeness, though content is missing
  | 'hotKeyOptions'
  | 'generalOptions'
  | 'storedItems';

// eslint-disable-next-line @typescript-eslint/no-empty-interface
interface CommandPaletteProps {}
interface WindowControlsProps {
  onClose: () => void;
  onMinimize: () => void;
  onToggleFullscreen: () => void;
}
interface ActionButtonProps {
  icon: string;
  tooltipId?: string;
  tooltipContent?: string; // Pass content directly if preferred
  onClick: () => void;
  className?: string;
  isSelected?: boolean; // For highlighting active states like 'recursive'
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
}
interface ActionButtonsProps {
  libraryService: any; // Consider more specific type if possible
  recursive: boolean;
  playSound: boolean;
  showControls: boolean;
  alwaysOnTop: boolean;
}
interface MenuBarProps extends ActionButtonsProps {
  windowControlsProps: WindowControlsProps;
}
interface ListContextDisplayProps {
  textFilter: string;
  tags: string[];
  activeDirectory: string;
}
interface SettingsListProps {
  filterType: 'image' | 'general' | 'autoplay';
  battleMode: boolean;
  currentItem: any; // Type according to your item structure
}
interface MenuContentAreaProps extends SettingsListProps {
  activeTab: TabType;
  cursor: number;
  libraryLength: number;
  isLoading: boolean;
  listContextProps: ListContextDisplayProps;
  libraryService: any; // Consider specific type
  storedCategories: { [key: string]: string };
  storedTags: { [key: string]: string[] };
}
interface TabSelectorProps {
  activeTab: TabType;
  onTabSelect: (tab: TabType) => void;
}

// --- Child Components ---

const WindowControls: React.FC<WindowControlsProps> = React.memo(
  ({ onClose, onMinimize, onToggleFullscreen }) => (
    <div className="windowControls">
      <span className="closeControl" onClick={onClose} />
      <span className="windowedControl" onClick={onMinimize} />
      <span className="fullScreenControl" onClick={onToggleFullscreen} />
    </div>
  )
);
WindowControls.displayName = 'WindowControls'; // Add display name

const ActionButton: React.FC<ActionButtonProps> = React.memo(
  ({
    icon,
    tooltipId,
    onClick,
    className = '',
    isSelected = false,
    onMouseEnter,
    onMouseLeave,
  }) => (
    <button
      data-tooltip-id={tooltipId}
      data-tooltip-delay-show={500}
      data-tooltip-offset={20}
      className={`menuIconButton ${className} ${
        isSelected ? 'selected' : ''
      }`.trim()}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <img src={icon} alt={tooltipId || 'action button'} />
    </button>
  )
);
ActionButton.displayName = 'ActionButton'; // Add display name

const ActionButtons: React.FC<ActionButtonsProps> = React.memo(
  ({ libraryService, recursive, playSound, showControls, alwaysOnTop }) => {
    const [showVolumeControl, setShowVolumeControl] = useState(false);
    const volumeRef = useRef<HTMLInputElement>(null);
    const volumeContainerRef = useRef<HTMLDivElement>(null);
    const showTimeoutRef = useRef<NodeJS.Timeout>();
    const hideTimeoutRef = useRef<NodeJS.Timeout>();

    const volume = useSelector(
      libraryService,
      (state: any) => state.context.settings.volume
    );

    const handleSettingChange = useCallback(
      (key: SettingKey, value: any, reload = false) => {
        const eventType = reload
          ? 'CHANGE_SETTING_AND_RELOAD'
          : 'CHANGE_SETTING';
        libraryService.send(eventType, { data: { [key]: value } });
      },
      [libraryService]
    );

    const handlePatreonClick = useCallback(() => {
      window.electron.ipcRenderer.sendMessage('open-external', [
        'https://www.patreon.com/lowkeyviewer',
      ]);
    }, []);

    const handleVolumeContainerMouseEnter = useCallback(() => {
      if (hideTimeoutRef.current) clearTimeout(hideTimeoutRef.current);
      if (playSound) {
        showTimeoutRef.current = setTimeout(
          () => setShowVolumeControl(true),
          300
        ); // Delay showing
      }
    }, [playSound]);

    const handleVolumeContainerMouseLeave = useCallback(() => {
      if (showTimeoutRef.current) clearTimeout(showTimeoutRef.current);
      hideTimeoutRef.current = setTimeout(
        () => setShowVolumeControl(false),
        200
      ); // Delay hiding
    }, []);

    const handleSoundToggle = useCallback(() => {
      const newPlaySound = !playSound;
      handleSettingChange('playSound', newPlaySound);

      // If turning sound on, show volume control immediately
      if (newPlaySound) {
        if (hideTimeoutRef.current) clearTimeout(hideTimeoutRef.current);
        setShowVolumeControl(true);
      }
    }, [playSound, handleSettingChange]);

    return (
      <div className="menuBarRight">
        <ActionButton
          icon={imageIcon}
          onClick={() => libraryService.send('SELECT_FILE')}
          tooltipId="select-file"
        />
        <ActionButton
          icon={folderIcon}
          onClick={() => libraryService.send('SELECT_DIRECTORY')}
          tooltipId="select-directory"
        />
        <ActionButton
          icon={recursiveIcon}
          onClick={() => handleSettingChange('recursive', !recursive, true)}
          isSelected={recursive}
          tooltipId="recursive"
        />
        <ActionButton
          icon={shuffleIcon}
          onClick={() => libraryService.send('SHUFFLE')}
          tooltipId="shuffle"
        />
        <ActionButton
          icon={lockIcon}
          onClick={() => handleSettingChange('alwaysOnTop', !alwaysOnTop)}
          isSelected={alwaysOnTop}
          tooltipId="always-on-top"
        />
        <div
          ref={volumeContainerRef}
          className="volumeButtonContainer"
          onMouseEnter={handleVolumeContainerMouseEnter}
          onMouseLeave={handleVolumeContainerMouseLeave}
        >
          <ActionButton
            icon={playSound ? soundIcon : noSoundIcon}
            onClick={handleSoundToggle}
            tooltipId={showVolumeControl ? undefined : 'sound'}
          />
          {showVolumeControl && playSound && (
            <div className="volumeControlHover">
              <div className="volumeLabel">{Math.round(volume * 100)}%</div>
              <input
                ref={volumeRef}
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={volume}
                onChange={(e) => {
                  const newVolume = clampVolume(parseFloat(e.target.value));
                  handleSettingChange('volume', newVolume);
                }}
                className="volumeSliderHover"
                aria-label="Volume"
              />
            </div>
          )}
        </div>
        <button
          data-tooltip-id="patreon-button"
          data-tooltip-offset={20}
          data-tooltip-delay-show={500}
          className="patreonButton"
          onClick={handlePatreonClick}
        >
          Patreon
        </button>
      </div>
    );
  }
);
ActionButtons.displayName = 'ActionButtons'; // Add display name

const MenuBar: React.FC<MenuBarProps> = React.memo(
  ({ windowControlsProps, ...actionButtonsProps }) => (
    <div className="menuBar">
      <WindowControls {...windowControlsProps} />
      <ActionButtons {...actionButtonsProps} />
    </div>
  )
);
MenuBar.displayName = 'MenuBar'; // Add display name

const ListContextDisplay: React.FC<ListContextDisplayProps> = React.memo(
  ({ textFilter, tags, activeDirectory }) => {
    const displayContent = useMemo(() => {
      if (textFilter) return textFilter;
      if (Array.isArray(tags) && tags.length > 0) return tags.join(', ');
      return getDirectory(activeDirectory);
    }, [textFilter, tags, activeDirectory]);

    return (
      <span className="listContext">{displayContent || 'No Context'}</span>
    );
  }
);
ListContextDisplay.displayName = 'ListContextDisplay'; // Add display name

const SettingsList: React.FC<SettingsListProps> = React.memo(
  ({ filterType, battleMode, currentItem }) => {
    const settingKeys = useMemo(() => {
      return Object.keys(SETTINGS).filter((key) => {
        const settingKey = key as SettingKey;
        const setting = SETTINGS[settingKey];
        // Special case: Exclude 'comicMode' if 'battleMode' is on and we are filtering for 'image'
        if (
          filterType === 'image' &&
          battleMode &&
          settingKey === 'comicMode'
        ) {
          return false;
        }
        return setting.display === filterType;
      }) as SettingKey[];
    }, [filterType, battleMode]);

    return (
      <>
        {settingKeys.map((settingKey) => {
          const setting = SETTINGS[settingKey];
          return (
            <Setting
              settingKey={settingKey}
              key={settingKey}
              reload={setting.reload}
              resetCursor={setting.resetCursor}
              currentItem={currentItem}
            />
          );
        })}
      </>
    );
  }
);
SettingsList.displayName = 'SettingsList'; // Add display name

const StoredItemsView: React.FC<{
  storedCategories: { [key: string]: string };
  storedTags: { [key: string]: string[] };
}> = React.memo(({ storedCategories, storedTags }) => {
  return (
    <div className="tabContent">
      <div className="storedItemsContainer">
        <div className="storedItemsSection">
          <h3 className="sectionTitle">Stored Categories</h3>
          <div className="storedItemsList">
            {Object.entries(storedCategories).length === 0 ? (
              <div className="emptyState">No categories stored yet</div>
            ) : (
              Object.entries(storedCategories).map(([position, category]) => (
                <div key={position} className="storedItem">
                  <div className="itemRow">
                    <span className="hotKeyIndicator">Shift+{position}</span>
                    <span className="itemValue">{category || 'Empty'}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="storedItemsSection">
          <h3 className="sectionTitle">Stored Tags</h3>
          <div className="storedItemsList">
            {Object.entries(storedTags).length === 0 ? (
              <div className="emptyState">No tags stored yet</div>
            ) : (
              Object.entries(storedTags).map(([position, tags]) => (
                <div key={position} className="storedItem">
                  <div className="itemRow">
                    <span className="hotKeyIndicator">{position}</span>
                    <div className="itemValue">
                      {tags && tags.length > 0 ? tags.join(', ') : 'Empty'}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="storedItemsHelp">
          <p>
            <strong>Usage:</strong>
          </p>
          <ul>
            <li>
              <strong>Alt+1-9:</strong> Store current category
            </li>
            <li>
              <strong>Shift+1-9:</strong> Switch to stored category
            </li>
            <li>
              <strong>Ctrl+1-9:</strong> Store currently active tags
            </li>
            <li>
              <strong>1-9:</strong> Apply stored tags to current item
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
});
StoredItemsView.displayName = 'StoredItemsView';

const MenuContentArea: React.FC<MenuContentAreaProps> = React.memo(
  ({
    activeTab,
    cursor,
    libraryLength,
    isLoading,
    listContextProps,
    libraryService,
    battleMode,
    currentItem,
    storedCategories,
    storedTags,
  }) => {
    const handleSetCursor = useCallback(
      (c: number) => {
        libraryService.send('SET_CURSOR', { idx: c });
      },
      [libraryService]
    );

    const renderTabContent = () => {
      switch (activeTab) {
        case 'imageOptions':
          return (
            <div className="tabContent">
              <SettingsList
                filterType="image"
                battleMode={battleMode}
                currentItem={currentItem}
              />
            </div>
          );
        case 'listViewOptions':
          return (
            <div className="tabContent">
              <GridSizePicker />
            </div>
          );
        case 'dbOptions':
          return (
            <div className="tabContent">
              <DbPathWidget />
              <CacheSetting />
            </div>
          );
        case 'hotKeyOptions':
          // Assuming HotKeyOptions doesn't need specific props from here
          return <HotKeyOptions />;
        case 'generalOptions':
          return (
            <div className="tabContent">
              <p>v2.6.3</p> {/* Consider making version dynamic */}
              <SettingsList
                filterType="general"
                battleMode={battleMode}
                currentItem={currentItem}
              />
            </div>
          );
        case 'autoPlayOptions':
          return (
            <div className="tabContent">
              <SettingsList
                filterType="autoplay"
                battleMode={battleMode}
                currentItem={currentItem}
              />
            </div>
          );
        case 'storedItems':
          return (
            <StoredItemsView
              storedCategories={storedCategories}
              storedTags={storedTags}
            />
          );
        default:
          return null;
      }
    };

    return (
      <div className="menuContent">
        <ListContextDisplay {...listContextProps} />
        <ProgressBar
          value={cursor}
          total={libraryLength}
          isLoading={isLoading}
          setCursor={handleSetCursor}
        />

        {renderTabContent()}
      </div>
    );
  }
);
MenuContentArea.displayName = 'MenuContentArea'; // Add display name

const TabSelector: React.FC<TabSelectorProps> = React.memo(
  ({ activeTab, onTabSelect }) => {
    const tabs: { id: TabType; icon: string }[] = [
      { id: 'imageOptions', icon: imageIcon },
      { id: 'listViewOptions', icon: gridIcon },
      { id: 'dbOptions', icon: dbIcon },
      { id: 'autoPlayOptions', icon: autoplayIcon },
      { id: 'hotKeyOptions', icon: keyboardIcon },
      { id: 'storedItems', icon: tag },
      { id: 'generalOptions', icon: gearIcon },
    ];

    return (
      <div className="tabs">
        {tabs.map((tabInfo) => (
          <button
            key={tabInfo.id}
            className={activeTab === tabInfo.id ? 'active' : ''}
            onClick={() => onTabSelect(tabInfo.id)}
            aria-label={`${tabInfo.id} tab`} // Accessibility
          >
            <img src={tabInfo.icon} alt={`${tabInfo.id} options`} />
          </button>
        ))}
      </div>
    );
  }
);
TabSelector.displayName = 'TabSelector'; // Add display name

const CommandPaletteTooltips: React.FC = React.memo(() => (
  <>
    <Tooltip
      id="recursive"
      content="Include files from all subdirectories."
      place="top"
    />
    <Tooltip id="shuffle" content="Shuffle items in the list." place="top" />
    <Tooltip
      id="always-on-top"
      content="Keep window always on top."
      place="top"
    />
    <Tooltip id="select-file" content="Select a file to view." place="top" />
    <Tooltip
      id="select-directory"
      content="Select a directory to browse."
      place="top"
    />
    <Tooltip
      id="patreon-button"
      content="Support development on Patreon"
      place="top"
    />
    {/* Add tooltips for other buttons if needed */}
  </>
));
CommandPaletteTooltips.displayName = 'CommandPaletteTooltips'; // Add display name

// --- Main Component ---

const CommandPalette: React.FC<CommandPaletteProps> = () => {
  const { libraryService } = useContext(GlobalStateContext);

  // Selectors for Command Palette State
  const { display, position } = useSelector(
    libraryService,
    (state) => state.context.commandPalette
  );

  // Selectors for Library and Settings State
  const tags = useSelector(
    libraryService,
    (state) => state.context.dbQuery.tags
  );
  const textFilter = useSelector(
    libraryService,
    (state) => state.context.textFilter
  );
  const activeDirectory = useSelector(
    libraryService,
    (state) => state.context.initialFile
  );
  const cursor = useSelector(libraryService, (state) => state.context.cursor);
  const settings = useSelector(
    libraryService,
    (state) => state.context.settings
  );
  const isLoading = useSelector(libraryService, (state) =>
    state.matches('loadingFromFS')
  );
  const libraryLoadId = useSelector(
    libraryService,
    (state) => state.context.libraryLoadId
  );
  const rawLibrary = useSelector(
    libraryService,
    (state) => state.context.library
  );
  const storedCategories = useSelector(
    libraryService,
    (state) => state.context.storedCategories
  );
  const storedTags = useSelector(
    libraryService,
    (state) => state.context.storedTags
  );

  // Derived State
  const library = useMemo(
    () =>
      filter(
        libraryLoadId,
        textFilter,
        rawLibrary,
        settings.filters,
        settings.sortBy
      ),
    [libraryLoadId, textFilter, rawLibrary, settings.filters, settings.sortBy]
  );
  const currentItem = useMemo(() => library[cursor], [library, cursor]);
  // const fileType = useMemo(() => currentItem?.path ? getFileType(currentItem.path) : '', [currentItem]); // Uncomment if needed

  // Local State
  const [activeTab, setActiveTab] = useState<TabType>('imageOptions');
  const paletteRef = useRef<HTMLDivElement>(null);
  const { width, height } = useComponentSize(paletteRef); // Size of the palette itself

  // Positioning Logic
  const getMenuPosition = useCallback(
    (x: number, y: number) => {
      if (!paletteRef.current) return { left: x, top: y }; // Default if ref not ready

      const windowWidth = window.innerWidth;
      const windowHeight = window.innerHeight;
      const paletteWidth = Math.max(width, 400); // Use measured or min width
      const paletteHeight = Math.max(height, 200); // Use measured or min height

      const xOverlap = x + paletteWidth - windowWidth;
      const yOverlap = y + paletteHeight - windowHeight;

      return {
        left: xOverlap > 0 ? Math.max(0, x - xOverlap - 10) : x, // Adjust slightly off edge
        top: yOverlap > 0 ? Math.max(0, y - yOverlap - 10) : y, // Adjust slightly off edge
      };
    },
    [width, height]
  ); // Recalculate only when size changes

  // Close on Click Outside
  useOnClickOutside(paletteRef, () => {
    libraryService.send('HIDE_COMMAND_PALETTE');
  });

  // Window Control Callbacks (using useCallback)
  const handleClose = useCallback(
    () => window.electron.ipcRenderer.sendMessage('shutdown', []),
    []
  );
  const handleMinimize = useCallback(
    () => window.electron.ipcRenderer.sendMessage('minimize', []),
    []
  );
  const handleToggleFullscreen = useCallback(
    () => window.electron.ipcRenderer.sendMessage('toggle-fullscreen', []),
    []
  );

  // Prepare props for child components
  const windowControlsProps: WindowControlsProps = {
    onClose: handleClose,
    onMinimize: handleMinimize,
    onToggleFullscreen: handleToggleFullscreen,
  };

  const actionButtonsProps: ActionButtonsProps = {
    libraryService,
    recursive: settings.recursive,
    playSound: settings.playSound,
    showControls: settings.showControls,
    alwaysOnTop: settings.alwaysOnTop,
  };

  const listContextProps: ListContextDisplayProps = {
    textFilter,
    tags,
    activeDirectory,
  };

  const menuContentAreaProps: Omit<
    MenuContentAreaProps,
    | 'activeTab'
    | 'cursor'
    | 'libraryLength'
    | 'isLoading'
    | 'listContextProps'
    | 'libraryService'
  > = {
    battleMode: settings.battleMode,
    currentItem: currentItem,
    filterType: 'image', // Default, but MenuContentArea uses activeTab to select SettingsList props
    storedCategories,
    storedTags,
  };

  // --- Render Logic ---
  if (!display) {
    return null;
  }

  const style = getMenuPosition(position.x, position.y);

  return (
    <div
      className="CommandPalette"
      ref={paletteRef}
      tabIndex={-1} // Make it focusable if needed, e.g., for keyboard events
      style={style}
    >
      <MenuBar
        windowControlsProps={windowControlsProps}
        {...actionButtonsProps}
      />

      <div className="menuArea">
        <MenuContentArea
          activeTab={activeTab}
          cursor={cursor}
          libraryLength={library.length}
          isLoading={isLoading}
          listContextProps={listContextProps}
          libraryService={libraryService}
          {...menuContentAreaProps} // Pass battleMode and currentItem
        />
        <TabSelector activeTab={activeTab} onTabSelect={setActiveTab} />
      </div>

      <CommandPaletteTooltips />
    </div>
  );
};
// Add display name for the main component too for good measure
CommandPalette.displayName = 'CommandPalette';

export default CommandPalette;
