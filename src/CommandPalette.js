import React, { useState, useEffect, useRef } from "react";
const electron = window.require("electron");
const settings = window.require("electron-settings");
import currentFolder from "./assets/folder-5-fill.svg";
import currentFile from "./assets/file-fill.svg";
import file from "./assets/file.svg";
import list from "./assets/file-list-2-fill.svg";
import image from "./assets/image-2-fill.svg";
import gear from "./assets/settings-3.svg";
import lock from "./assets/lock-fill.svg";
import database from "./assets/database.svg";
import recursive from "./assets/recursive.svg";
import save from "./assets/save-3-line.svg";
import parentDirectory from "./assets/folder-upload-fill.svg";
import keyboard from "./assets/keyboard.svg";

var path = window.require("path");

import question from "./assets/question-fill.svg";
import ProgressBar from "./ProgressBar";
import { SORT, FILTER, SIZE, CONTROL_MODE, getNext } from "./constants";
import { getFolder, getFile, saveCurrentSettings } from "./fsTools";
import "./CommandPalette.css";

function useOnClickOutside(ref, handler, isLocked) {
  useEffect(
    () => {
      const listener = (event) => {
        // Do nothing if clicking ref's element or descendent elements
        console.log(event);
        if (
          isLocked ||
          !ref.current ||
          ref.current.contains(event.target) ||
          event.button === 2
        ) {
          return;
        }
        handler(event);
      };
      document.addEventListener("mousedown", listener);
      document.addEventListener("touchstart", listener);
      return () => {
        document.removeEventListener("mousedown", listener);
        document.removeEventListener("touchstart", listener);
      };
    },
    // Add ref and handler to effect dependencies
    // It's worth noting that because passed in handler is a new ...
    // ... function on every render that will cause this effect ...
    // ... callback/cleanup to run every render. It's not a big deal ...
    // ... but to optimize you can wrap handler in useCallback before ...
    // ... passing it into this hook.
    [ref, handler]
  );
}

function CommandPallete({
  status = {},
  controls = {},
  setAbout,
  setCommandPaletteOpen,
  position,
}) {
  const [isLocked, setIsLocked] = useState(false);

  const [isAlwaysOnTop, setIsAlwaysOnTop] = useState(
    electron.remote.getCurrentWindow().isAlwaysOnTop()
  );

  const [isFullScreen, setIsFullScreen] = useState(
    electron.remote.getCurrentWindow().isFullScreen()
  );

  const ref = useRef();
  // State for our modal
  // Call hook passing in the ref and a function to call on outside click
  useOnClickOutside(ref, () => setCommandPaletteOpen(false), isLocked);

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

  return (
    <div
      className="CommandPalette"
      tabIndex="-1"
      ref={ref}
      style={{ top: position.y, left: position.x }}
    >
      <div className="menuBar">
        <div className="windowControls">
          <span
            className="closeControl"
            onClick={(e) => electron.remote.getCurrentWindow().close()}
            disabled="disabled"
            tabIndex="-1"
          />
          <span
            className="windowedControl"
            onClick={(e) => electron.remote.getCurrentWindow().minimize()}
            disabled="disabled"
            tabIndex="-1"
          />
          <span
            className="fullScreenControl"
            onClick={(e) => setIsFullScreen(!isFullScreen)}
            disabled="disabled"
            tabIndex="-1"
          />
        </div>
        <div className="menuBarRight">
          <button
            className="saveSettingsButton"
            onClick={() => setIsLocked(!isLocked)}
            style={{ opacity: isLocked ? 1 : 0.6 }}
          >
            <img src={lock} />
          </button>
          <button
            className="saveSettingsButton"
            onClick={() =>
              saveCurrentSettings({
                controlMode: status.controlMode.key,
                defaultSort: status.sort.key,
                scaleMode: status.size.key,
                defaultFilter: status.filter.key,
                audio: status.audio,
                videoControls: status.videoControls,
                isAlwaysOnTop,
                isFullScreen,
              })
            }
          >
            <img src={save} />
          </button>
          <button className="aboutButton" onClick={() => setAbout(true)}>
            <img src={question} />
          </button>
        </div>
      </div>
      <div className="paletteBody">
        <div className="options">
          {status.tab === "fileOptions" && (
            <div className="listOptions">
              <div className="optionSection">
                <div className="optionSet">
                  <label>
                    Directory ({status.items.length} Items)
                    <button
                      onClick={() => setAbout(true)}
                      onClick={() => controls.setRecursive(!status.recursive)}
                      style={{ opacity: status.recursive ? 1 : 0.5 }}
                    >
                      <img src={recursive} />
                    </button>
                  </label>
                  <div className="optionButton" onClick={controls.changePath}>
                    <div className="primary">{getFolder(status.filePath)}</div>
                    <div className="actions">
                      <button
                        className="action"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          controls.setPath(path.dirname(status.filePath));
                        }}
                      >
                        <img src={parentDirectory} />
                      </button>
                      <button className="action">
                        <img src={keyboard} />
                      </button>
                    </div>
                  </div>
                </div>
                <div className="optionSet">
                  <label>
                    Open File {status.cursor + 1} of {status.items.length}
                  </label>
                  <ProgressBar
                    value={status.cursor + 1}
                    total={status.items.length}
                    setCursor={controls.setCursor}
                  />
                  <div
                    className="optionData"
                    onClick={() => controls.setPath(status.fileName)}
                  >
                    <img src={currentFolder} className="icon" />
                    {getFolder(status.fileName).substring(
                      getFolder(status.filePath).length
                    ) || "\\"}
                  </div>
                  <div className="optionData" onClick={controls.changePath}>
                    <img src={currentFile} className="icon" />

                    {getFile(status.fileName)}
                  </div>
                </div>
              </div>
            </div>
          )}
          {status.tab === "listOptions" && (
            <div className="listOptions">
              <div className="optionSection">
                <div className="optionSet">
                  <label>Sort Order</label>
                  {Object.entries(SORT).map(([key, value]) => {
                    return (
                      <div
                        key={key}
                        className={[
                          "optionButton",
                          key === status.sort.key ? "active" : null,
                        ].join(" ")}
                        onClick={() => controls.setSort(value)}
                      >
                        <div className="primary">{value.title}</div>
                        <div className="actions">
                          <button className="action">
                            <img src={keyboard} />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className="optionSection">
                <div className="optionSet">
                  <label>File Type Filter</label>
                  {Object.entries(FILTER).map(([key, value]) => {
                    return (
                      <div
                        key={key}
                        className={[
                          "optionButton",
                          key === status.filter.key ? "active" : null,
                        ].join(" ")}
                        onClick={() => controls.setFilter(value)}
                      >
                        <div className="primary">{value.title}</div>
                        <div className="actions">
                          <button className="action">
                            <img src={keyboard} />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="optionSet">
                  <label>Recursive</label>
                  <div
                    className={[
                      "optionButton",
                      status.recursive ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setRecursive(true)}
                  >
                    <div className="primary">On</div>
                    <div className="actions">
                      <button className="action">
                        <img src={keyboard} />
                      </button>
                    </div>
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !status.recursive ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setRecursive(false)}
                  >
                    <div className="primary">Off</div>
                    <div className="actions">
                      <button className="action">
                        <img src={keyboard} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          {status.tab === "imageOptions" && (
            <div className="imageOptions">
              <div className="optionSection">
                <div className="optionSet">
                  <label>Image Scaling</label>
                  {Object.entries(SIZE).map(([key, value]) => {
                    return (
                      <div
                        key={key}
                        className={[
                          "optionButton",
                          key === status.size.key ? "active" : null,
                        ].join(" ")}
                        onClick={() => controls.setSize(value)}
                      >
                        <div className="primary">{value.title}</div>
                        <div className="actions">
                          <button className="action">
                            <img src={keyboard} />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="optionSet">
                  <label>Video Audio</label>
                  <div
                    className={[
                      "optionButton",
                      status.audio ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setAudio(true)}
                  >
                    <div className="primary">On</div>
                    <div className="actions">
                      <button className="action">
                        <img src={keyboard} />
                      </button>
                    </div>
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !status.audio ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setAudio(false)}
                  >
                    <div className="primary">Off</div>
                    <div className="actions">
                      <button className="action">
                        <img src={keyboard} />
                      </button>
                    </div>
                  </div>
                </div>
                <div className="optionSet">
                  <label>Video Controls</label>
                  <div
                    className={[
                      "optionButton",
                      status.videoControls ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setVideoControls(true)}
                  >
                    <div className="primary">On</div>
                    <div className="actions">
                      <button className="action">
                        <img src={keyboard} />
                      </button>
                    </div>
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !status.videoControls ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setVideoControls(false)}
                  >
                    <div className="primary">Off</div>
                    <div className="actions">
                      <button className="action">
                        <img src={keyboard} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          {status.tab === "controlOptions" && (
            <div className="controlOptions">
              <div className="optionSection">
                <div className="optionSet">
                  <label>Control Mode</label>
                  {Object.entries(CONTROL_MODE).map(([key, value]) => {
                    return (
                      <div
                        key={key}
                        className={[
                          "optionButton",
                          key === status.controlMode.key ? "active" : null,
                        ].join(" ")}
                        onClick={() => controls.setControlMode(value)}
                      >
                        <div className="primary">{value.title}</div>
                        <div className="actions">
                          <button className="action">
                            <img src={keyboard} />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="optionSet">
                  <label>Always on Top</label>
                  <div
                    className={[
                      "optionButton",
                      isAlwaysOnTop ? "active" : null,
                    ].join(" ")}
                    onClick={() => setIsAlwaysOnTop(true)}
                  >
                    <div className="primary">On</div>
                    <div className="actions">
                      <button className="action">
                        <img src={keyboard} />
                      </button>
                    </div>
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !isAlwaysOnTop ? "active" : null,
                    ].join(" ")}
                    onClick={() => setIsAlwaysOnTop(false)}
                  >
                    <div className="primary">Off</div>
                    <div className="actions">
                      <button className="action">
                        <img src={keyboard} />
                      </button>
                    </div>
                  </div>
                </div>
                <div className="optionSet">
                  <label>Fullscreen</label>
                  <div
                    className={[
                      "optionButton",
                      isFullScreen ? "active" : null,
                    ].join(" ")}
                    onClick={() => setIsFullScreen(true)}
                  >
                    <div className="primary">On</div>
                    <div className="actions">
                      <button className="action">
                        <img src={keyboard} />
                      </button>
                    </div>
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !isFullScreen ? "active" : null,
                    ].join(" ")}
                    onClick={() => setIsFullScreen(false)}
                  >
                    <div className="primary">Off</div>
                    <div className="actions">
                      <button className="action">
                        <img src={keyboard} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          {status.tab === "databaseOptions" && (
            <div className="controlOptions">
              <div className="optionSection">
                <div className="optionSet">
                  <label>Database Mode</label>
                  <div className="optionButton">Activate</div>
                </div>
              </div>
            </div>
          )}
        </div>
        <div className="tabs">
          <button
            className={status.tab === "fileOptions" ? "active" : null}
            onClick={() => controls.setTab("fileOptions")}
          >
            <img src={file} />
          </button>
          <button
            className={status.tab === "listOptions" ? "active" : null}
            onClick={() => controls.setTab("listOptions")}
          >
            <img src={list} />
          </button>
          <button
            className={status.tab === "imageOptions" ? "active" : null}
            onClick={() => controls.setTab("imageOptions")}
          >
            <img src={image} />
          </button>
          <button
            className={status.tab === "controlOptions" ? "active" : null}
            onClick={() => controls.setTab("controlOptions")}
          >
            <img src={gear} />
          </button>
          <button
            className={status.tab === "databaseOptions" ? "active" : null}
            onClick={() => controls.setTab("databaseOptions")}
          >
            <img src={database} />
          </button>
        </div>
      </div>
    </div>
  );
}

export default CommandPallete;
