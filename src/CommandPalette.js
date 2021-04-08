import React, { useState, useEffect, useRef } from "react";
const electron = window.require("electron");
const settings = window.require("electron-settings");
import options from "./assets/file-list-2-fill.svg";
import image from "./assets/image-2-fill.svg";
import gear from "./assets/settings-3.svg";
import database from "./assets/database.svg";
import save from "./assets/save-3-line.svg";
import question from "./assets/question-fill.svg";

import { SORT, FILTER, SIZE, CONTROL_MODE, getNext } from "./constants";
import { getFolder, saveCurrentSettings } from "./fsTools";
import "./CommandPalette.css";

function useOnClickOutside(ref, handler) {
  useEffect(
    () => {
      const listener = (event) => {
        // Do nothing if clicking ref's element or descendent elements
        console.log(event);
        if (
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
  const [tab, setTab] = useState("fileOptions");

  const [isAlwaysOnTop, setIsAlwaysOnTop] = useState(
    electron.remote.getCurrentWindow().isAlwaysOnTop()
  );

  const [isFullScreen, setIsFullScreen] = useState(
    electron.remote.getCurrentWindow().isFullScreen()
  );

  const ref = useRef();
  // State for our modal
  // Call hook passing in the ref and a function to call on outside click
  useOnClickOutside(ref, () => setCommandPaletteOpen(false));

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
          {tab === "fileOptions" && (
            <div className="fileOptions">
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
                        {value.title}
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
                        {value.title}
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
                    On
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !status.recursive ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setRecursive(false)}
                  >
                    Off
                  </div>
                </div>
              </div>
            </div>
          )}
          {tab === "imageOptions" && (
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
                        {value.title}
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
                    On
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !status.audio ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setAudio(false)}
                  >
                    Off
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
                    On
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !status.videoControls ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setVideoControls(false)}
                  >
                    Off
                  </div>
                </div>
              </div>
            </div>
          )}
          {tab === "controlOptions" && (
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
                        {value.title}
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
                    On
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !isAlwaysOnTop ? "active" : null,
                    ].join(" ")}
                    onClick={() => setIsAlwaysOnTop(false)}
                  >
                    Off
                  </div>
                </div>
              </div>
            </div>
          )}
          {tab === "databaseOptions" && (
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
            className={tab === "fileOptions" ? "active" : null}
            onClick={() => setTab("fileOptions")}
          >
            <img src={options} />
          </button>
          <button
            className={tab === "imageOptions" ? "active" : null}
            onClick={() => setTab("imageOptions")}
          >
            <img src={image} />
          </button>
          <button
            className={tab === "controlOptions" ? "active" : null}
            onClick={() => setTab("controlOptions")}
          >
            <img src={gear} />
          </button>
          <button
            className={tab === "databaseOptions" ? "active" : null}
            onClick={() => setTab("databaseOptions")}
          >
            <img src={database} />
          </button>
        </div>
      </div>
    </div>
  );
}

export default CommandPallete;
