import React, { useState, useEffect, useRef } from "react";
import * as R from "ramda";

import ReactDOM from "react-dom";
import "./style.css";
import HotKeyController from "./HotKeyController";
import { getFolder, saveCurrentSettings } from "./fsTools";
import { HotKeySetter } from "./HotKeySetter";
import HotCorner from "./HotCorner";
import Detail from "./Detail";
import List from "./List";
import Spinner from "./Spinner";
// NODE IMPORTS
const electron = window.require("electron");
var shuffle = window.require("shuffle-array");
const settings = window.require("electron-settings");
const { decode } = window.require("js-base64");

import {
  SORT,
  FILTER,
  SIZE,
  VIEW,
  CONTROL_MODE,
  HOT_KEY_DEFAULTS,
  getNext,
  LIST_SIZE,
} from "./constants";
import loadImageList from "./loadImageList";
import CommandPalette from "./CommandPalette";
import About from "./About";

function App() {
  const [dragging, setDragging] = useState(false);
  const [tab, setTab] = useState("fileOptions");
  const [settingHotKey, setSettingHotKey] = useState(false);
  const [view, setView] = useState(VIEW.DETAIL);
  const [filePath, setPath] = useState(
    decode(window.location.search.substr(1))
  );
  const [loading, setLoading] = useState(false);
  const [about, setAbout] = useState(settings.get("settings.starts") % 5 === 0);

  const [shuffles, setShuffles] = useState(true);
  const [firstLoadCleared, setFirstLoadCleared] = useState(false);

  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [items, setItems] = useState([]);
  const [cursor, setCursor] = useState(0);
  const [useBucket, setUseBucket] = useState(false);

  const [controlMode, setControlMode] = useState(
    CONTROL_MODE[settings.get("settings.controlMode")]
  );
  const [audio, setAudio] = useState(settings.get("settings.audio"));
  const [videoControls, setVideoControls] = useState(
    settings.get("settings.videoControls")
  );

  const [volume, setVolume] = useState(1);
  const [sort, setSort] = useState(SORT[settings.get("settings.defaultSort")]);
  const [filter, setFilter] = useState(
    FILTER[settings.get("settings.defaultFilter")]
  );
  const [size, setSize] = useState(SIZE[settings.get("settings.scaleMode")]);
  const [listSize, setListSize] = useState(LIST_SIZE.OVERSCAN);

  const [recursive, setRecursive] = useState(false);

  const [isAlwaysOnTop, setIsAlwaysOnTop] = useState(
    electron.remote.getCurrentWindow().isAlwaysOnTop()
  );

  const [isFullScreen, setIsFullScreen] = useState(
    electron.remote.getCurrentWindow().isFullScreen()
  );

  // Sync window always on top value with state.
  useEffect(() => {
    if (electron.remote.getCurrentWindow().isAlwaysOnTop() !== isAlwaysOnTop) {
      electron.remote.getCurrentWindow().setAlwaysOnTop(isAlwaysOnTop);
      electron.remote.getCurrentWindow().focus();
    }
  }, [isAlwaysOnTop]);

  // Sync isFullScreen with state.
  useEffect(() => {
    if (electron.remote.getCurrentWindow().isFullScreen() !== isFullScreen) {
      electron.remote.getCurrentWindow().setFullScreen(isFullScreen);
      electron.remote.getCurrentWindow().focus();
    }
  }, [isFullScreen]);

  function changePath() {
    console.log("Changing from current path of:", filePath);
    electron.remote.dialog
      .showOpenDialog(electron.remote.getCurrentWindow(), {
        properties: ["openFile"],
        defaultPath: getFolder(filePath),
      })
      .then((files) => {
        if (files.filePaths.length > 0) {
          setPath(files.filePaths[0]);
        }
      });
  }
  // Initialize State from settings.
  useEffect(() => {
    // Uncomment to open dev tools on load.
    // electron.remote.getCurrentWindow().webContents.openDevTools();
    if (settings.has("settings.scaleMode")) {
      setSize(SIZE[settings.get("settings.scaleMode")]);
    }
  }, []);
  // Reload data from image provider if directory, filter, or recursive setting changes
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      const data = await loadImageList(
        getFolder(filePath),
        filter.value,
        sort,
        recursive,
        useBucket
      );
      setItems(data.items);
      let cursor = data.items.findIndex((item) => {
        return item.fileName === filePath;
      });
      if (cursor < 0) {
        cursor = 0;
      }
      setCursor(cursor);
      setLoading(false);
    }
    if (filePath.length > 1) {
      fetchData();
    } else {
      changePath();
    }
  }, [filePath, useBucket, sort, filter, recursive]);

  function nextImage() {
    setCursor(cursor === items.length - 1 ? cursor : cursor + 1);
  }

  function previousImage() {
    setCursor(cursor === 0 ? cursor : cursor - 1);
  }
  function handleClick(e) {
    e.preventDefault();
    // If click is on the left decrease the cursor, if it is on the left increase it.
    e.pageX > window.innerWidth / 2 ? nextImage() : previousImage();
  }

  function handleScroll(e) {
    e.stopPropagation();
    // If delta y is positive increase cursor value, otherwise decrease.
    e.deltaY > 0 ? nextImage() : previousImage();
  }
  const windowRef = electron.remote.getCurrentWindow();
  const commands = {
    fileOptions: {
      changeFile: {
        action: () => {
          changePath();
        },
      },
      toggleRecursion: {
        action: () => {
          setRecursive(!recursive);
        },
      },
      nextImage: {
        action: () => {
          nextImage();
        },
      },
      previousImage: {
        action: () => {
          previousImage();
        },
      },
      shuffle: {
        action: () => {
          setItems(shuffle(items));
          setShuffles(!shuffles);
        },
      },
    },
    listOptions: {
      toggleSortOrder: {
        action: () => {
          setSort(getNext(SORT, sort.key));
        },
      },
      showALL: {
        action: () => {
          setFilter(FILTER.ALL);
        },
      },
      showSTATIC: {
        action: () => {
          setFilter(FILTER.STATIC);
        },
      },
      showVIDEO: {
        action: () => {
          setFilter(FILTER.VIDEO);
        },
      },
      showGIF: {
        action: () => {
          setFilter(FILTER.GIF);
        },
      },
      showMOTION: {
        action: () => {
          setFilter(FILTER.MOTION);
        },
      },
    },
    imageOptions: {
      toggleSizing: {
        action: () => {
          setSize(getNext(SIZE, size.key));
        },
      },
      sizeOVERSCAN: {
        action: () => {
          setSize(SIZE.OVERSCAN);
        },
      },
      sizeACTUAL: {
        action: () => {
          setSize(SIZE.ACTUAL);
        },
      },
      sizeFIT: {
        action: () => {
          setSize(SIZE.FIT);
        },
      },
      sizeCOVER: {
        action: () => {
          setSize(SIZE.COVER);
        },
      },
      toggleAudio: {
        action: () => {
          setAudio(!audio);
        },
      },
      toggleVideoControls: {
        action: () => {
          setVideoControls(!videoControls);
        },
      },
    },
    controlOptions: {
      toggleControls: {
        action: () => {
          setControlMode(getNext(CONTROL_MODE, controlMode.key));
        },
      },
    },
    windowOptions: {
      minimize: {
        action: () => {
          windowRef.minimize();
        },
      },
      toggleFullscreen: {
        action: () => {
          setIsFullScreen(!isFullScreen);
        },
      },
      toggleAlwaysOnTop: {
        action: () => {
          setIsAlwaysOnTop(!isAlwaysOnTop);
        },
      },
      openDevTools: {
        action: () => {
          windowRef.webContents.openDevTools();
        },
      },
    },
  };
  const [hotKeys, setHotKeys] = useState(settings.get("settings.hotKeys"));

  function handleKeyPress(e) {
    console.log("KEY PRESSED:", e.key);
    if (!hotKeys[e.key]) {
      return;
    }
    e.preventDefault();
    const command = R.path(hotKeys[e.key].split("."))(commands);
    command.action();
  }

  if (settingHotKey) {
    return (
      <HotKeySetter
        action={settingHotKey}
        setHotKeys={setHotKeys}
        handleComplete={() => setSettingHotKey(false)}
      />
    );
  }

  if (loading) {
    return (
      <React.Fragment>
        {commandPaletteOpen && (
          <CommandPalette
            status={{
              fileName: "",
              cursor,
              hotKeys,
              tab,
              filePath,
              sort,
              filter,
              size,
              listSize,
              audio,
              videoControls,
              controlMode,
              recursive,
              items,
              isAlwaysOnTop,
              isFullScreen,
              useBucket
            }}
            controls={{
              setSettingHotKey,
              changePath,
              setPath,
              setHotKeys,
              setAudio,
              setVideoControls,
              setSort,
              setTab,
              setFilter,
              setSize,
              setListSize,
              setControlMode,
              setRecursive,
              setCursor,
              setIsAlwaysOnTop,
              setIsFullScreen,
              setUseBucket
            }}
            setAbout={setAbout}
            position={commandPaletteOpen}
            setCommandPaletteOpen={setCommandPaletteOpen}
          />
        )}
        <div
          tabIndex="0"
          onKeyPress={handleKeyPress}
          className="loadingContainer"
          onContextMenu={(e) => {
            setCommandPaletteOpen({ x: e.clientX, y: e.clientY });
          }}
        >
          <Spinner />
        </div>
      </React.Fragment>
    );
  }

  if (items.length === 0) {
    return (
      <React.Fragment>
        {!about && <HotKeyController handleKeyPress={handleKeyPress} />}
        <div className="dragArea"></div>
        <div className="dragAreaHover"></div>

        {commandPaletteOpen && (
          <CommandPalette
            status={{
              fileName: "",
              cursor,
              hotKeys,
              tab,
              filePath,
              sort,
              filter,
              size,
              listSize,
              audio,
              videoControls,
              controlMode,
              recursive,
              items,
              isAlwaysOnTop,
              isFullScreen,
              useBucket
            }}
            controls={{
              setSettingHotKey,
              changePath,
              setPath,
              setHotKeys,
              setAudio,
              setVideoControls,
              setSort,
              setTab,
              setFilter,
              setSize,
              setListSize,
              setControlMode,
              setRecursive,
              setCursor,
              setIsAlwaysOnTop,
              setIsFullScreen,
              setUseBucket
            }}
            setAbout={setAbout}
            position={commandPaletteOpen}
            setCommandPaletteOpen={setCommandPaletteOpen}
          />
        )}
        <div
          tabIndex="0"
          onKeyPress={handleKeyPress}
          className="noItemsContainer"
          onContextMenu={(e) => {
            setCommandPaletteOpen({ x: e.clientX, y: e.clientY });
          }}
        >
          <span className="noItemsMessage">No Images Found</span>
        </div>
      </React.Fragment>
    );
  }

  return (
    <React.Fragment>
      {!false && <HotKeyController handleKeyPress={handleKeyPress} />}
      {settings.get("settings.starts") === 1 && !firstLoadCleared && (
        <div className="firstLoadContainer">
          <div className="firstLoadMenu">
            <div
              className="mouse option"
              onClick={(e) => {
                setControlMode(CONTROL_MODE.MOUSE);
                saveCurrentSettings({ controlMode: CONTROL_MODE.MOUSE.key });
                setFirstLoadCleared(true);
              }}
            >
              <div className="iconContainer">
                <div className="iconScroll" />
              </div>
              <span>
                I am using a mouse with a scroll wheel. Scroll to change images.
                Click and drag to pan.
              </span>
            </div>
            <div
              className="trackpad option"
              onClick={(e) => {
                setControlMode(CONTROL_MODE.TRACK_PAD);
                saveCurrentSettings({
                  controlMode: CONTROL_MODE.TRACK_PAD.key,
                });

                setFirstLoadCleared(true);
              }}
            >
              <div className="iconContainer">
                <div className="trackPadScroll" />
              </div>
              <span>
                I am using a laptop trackpad or Apple Magic Mouse. Tap left or
                right side to flip images. Drag 2 fingers to pan.
              </span>
            </div>
            <div className="changeLaterContainer">
              <span className="changeLaterMessage">
                You Can change this later in the settings by hovering over the
                bottom left corner of the screen.
              </span>
            </div>
          </div>
        </div>
      )}
      {about && <About setAbout={setAbout} />}
      <div className="dragArea"></div>
      <div
        className="dragAreaHover"
        onMouseEnter={() => console.log("entered drag area")}
        onMouseLeave={() => setDragging(false)}
        style={dragging ? { opacity: 1 } : { opacity: 0 }}
      ></div>
      {commandPaletteOpen && (
        <CommandPalette
          status={{
            fileName: items[cursor].fileName,
            cursor,
            hotKeys,
            tab,
            filePath,
            sort,
            filter,
            size,
            listSize,
            audio,
            videoControls,
            controlMode,
            recursive,
            items,
            isAlwaysOnTop,
            isFullScreen,
            useBucket
          }}
          controls={{
            setSettingHotKey,
            changePath,
            setPath,
            setHotKeys,
            setAudio,
            setVideoControls,
            setSort,
            setTab,
            setFilter,
            setSize,
            setListSize,
            setControlMode,
            setRecursive,
            setCursor,
            setIsAlwaysOnTop,
            setIsFullScreen,
            setUseBucket
          }}
          setAbout={setAbout}
          position={commandPaletteOpen}
          setCommandPaletteOpen={setCommandPaletteOpen}
        />
      )}

      {view === VIEW.DETAIL ? (
        <React.Fragment>
          {controlMode.key === CONTROL_MODE.TRACK_PAD.key && (
            <HotCorner handleClick={() => setView(VIEW.LIST)} />
          )}
          <Detail
            fileName={items[cursor].fileName}
            size={size}
            audio={audio}
            volume={volume}
            videoControls={videoControls}
            setVideoControls={setVideoControls}
            setAudio={setAudio}
            setVolume={setVolume}
            handleClick={handleClick}
            handleScroll={handleScroll}
            controlMode={controlMode}
            useBucket={useBucket}
            handleDoubleClick={() => {
              if (controlMode.key === CONTROL_MODE.MOUSE.key) {
                setView(VIEW.LIST);
              }
            }}
            handleRightClick={(e) => {
              setCommandPaletteOpen({ x: e.clientX, y: e.clientY });
            }}
          />
        </React.Fragment>
      ) : (
        <List
          filter={filter}
          fileList={items}
          shuffles={shuffles}
          size={listSize}
          cursor={cursor}
          useBucket={useBucket}
          controlMode={controlMode}
          columns={3}
          setPath={setPath}
          handleSelection={(i) => {
            setCursor(i);
            setView(VIEW.DETAIL);
          }}
          handleRightClick={(e) => {
            setCommandPaletteOpen({ x: e.clientX, y: e.clientY });
          }}
        />
      )}
    </React.Fragment>
  );
}
ReactDOM.render(<App />, document.getElementById("root"));
