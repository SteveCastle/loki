import React, { useState, useEffect, useRef } from "react";
const electron = window.require("electron");
import useComponentSize from "@rehooks/component-size";
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

  const ref = useRef();
  // State for our modal
  // Call hook passing in the ref and a function to call on outside click
  useOnClickOutside(ref, () => setCommandPaletteOpen(false), isLocked);

  const { hotKeys } = status;
  const hotKeysByAction = Object.entries(hotKeys).reduce(
    (acc, [index, value]) => ({ ...acc, [value]: index }),
    {}
  );

  let { width, height } = useComponentSize(ref);
  const windowWidth = window.innerWidth;
  const windowHeight = window.innerHeight;

  const getMenuPosition = (x, y) => {
    const xOverlap = x + width - windowWidth;
    const yOverlap = y + height - windowHeight;
    return {
      left: xOverlap > 0 ? x - xOverlap : x,
      top: yOverlap > 0 ? y - yOverlap : y,
    };
  };

  return (
    <div
      className="CommandPalette"
      tabIndex="-1"
      ref={ref}
      style={getMenuPosition(position.x, position.y)}
    >
      <div className="menuBar">
        <div className="windowControls">
          <span
            className="closeControl"
            onClick={() => electron.remote.getCurrentWindow().close()}
            disabled="disabled"
            tabIndex="-1"
          />
          <span
            className="windowedControl"
            onClick={() => electron.remote.getCurrentWindow().minimize()}
            disabled="disabled"
            tabIndex="-1"
          />
          <span
            className="fullScreenControl"
            onClick={() => controls.setIsFullScreen(!status.isFullScreen)}
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
                isAlwaysOnTop: status.isAlwaysOnTop,
                isFullScreen: status.isFullScreen,
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
            <div className="fileOptions">
              <div className="optionSection">
                <div className="optionSet">
                  <label>
                    Directory ({status.items.length} Items)
                    <button
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
                      <button
                        className="action"
                        onClick={(e) => {
                          e.stopPropagation();
                          controls.setSettingHotKey("fileOptions.changeFile");
                        }}
                      >
                        <img src={keyboard} />
                        <span className="currentHotKey">
                          {hotKeysByAction["fileOptions.changeFile"]}
                        </span>
                      </button>
                    </div>
                  </div>
                </div>
                <div className="optionSet">
                  <label>
                    Open File{" "}
                    {status.cursor + (status.fileName.length > 0 ? 1 : 0)} of{" "}
                    {status.items.length}
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
                  <label>
                    Sort Order
                    <button
                      className="action"
                      onClick={(e) => {
                        e.stopPropagation();
                        controls.setSettingHotKey(
                          "listOptions.toggleSortOrder"
                        );
                      }}
                    >
                      <img src={keyboard} />
                      <span className="currentHotKey">
                        {hotKeysByAction["listOptions.toggleSortOrder"]}
                      </span>
                    </button>
                  </label>
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
                        <div className="actions"></div>
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
                          <button
                            className="action"
                            onClick={(e) => {
                              e.stopPropagation();
                              controls.setSettingHotKey(
                                `listOptions.show${value.key}`
                              );
                            }}
                          >
                            <img src={keyboard} />
                            <span className="currentHotKey">
                              {hotKeysByAction[`listOptions.show${value.key}`]}
                            </span>
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="optionSet">
                  <label>
                    Recursive Mode (Scan SubDirectories)
                    <button
                      className="action"
                      onClick={(e) => {
                        e.stopPropagation();
                        controls.setSettingHotKey(
                          `fileOptions.toggleRecursion`
                        );
                      }}
                    >
                      <img src={keyboard} />
                      <span className="currentHotKey">
                        {hotKeysByAction[`fileOptions.toggleRecursion`]}
                      </span>
                    </button>
                  </label>
                  <div
                    className={[
                      "optionButton",
                      status.recursive ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setRecursive(true)}
                  >
                    <div className="primary">On</div>
                    <div className="actions"></div>
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !status.recursive ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setRecursive(false)}
                  >
                    <div className="primary">Off</div>
                    <div className="actions"></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          {status.tab === "imageOptions" && (
            <div className="imageOptions">
              <div className="optionSection">
                <div className="optionSet">
                  <label>
                    Image Scaling
                    <button
                      className="action"
                      onClick={(e) => {
                        e.stopPropagation();
                        controls.setSettingHotKey(`imageOptions.toggleSizing`);
                      }}
                    >
                      <img src={keyboard} />
                      <span className="currentHotKey">
                        {hotKeysByAction[`imageOptions.toggleSizing`]}
                      </span>
                    </button>
                  </label>
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
                          <button
                            className="action"
                            onClick={(e) => {
                              e.stopPropagation();
                              controls.setSettingHotKey(
                                `imageOptions.size${value.key}`
                              );
                            }}
                          >
                            <img src={keyboard} />
                            <span className="currentHotKey">
                              {hotKeysByAction[`imageOptions.size${value.key}`]}
                            </span>
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="optionSet">
                  <label>
                    Video Audio
                    <button
                      className="action"
                      onClick={(e) => {
                        e.stopPropagation();
                        controls.setSettingHotKey(`imageOptions.toggleAudio`);
                      }}
                    >
                      <img src={keyboard} />
                      <span className="currentHotKey">
                        {hotKeysByAction[`imageOptions.toggleAudio`]}
                      </span>
                    </button>
                  </label>
                  <div
                    className={[
                      "optionButton",
                      status.audio ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setAudio(true)}
                  >
                    <div className="primary">On</div>
                    <div className="actions"></div>
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !status.audio ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setAudio(false)}
                  >
                    <div className="primary">Off</div>
                    <div className="actions"></div>
                  </div>
                </div>
                <div className="optionSet">
                  <label>
                    Video Controls
                    <button
                      className="action"
                      onClick={(e) => {
                        e.stopPropagation();
                        controls.setSettingHotKey(
                          `imageOptions.toggleVideoControls`
                        );
                      }}
                    >
                      <img src={keyboard} />
                      <span className="currentHotKey">
                        {hotKeysByAction[`imageOptions.toggleVideoControls`]}
                      </span>
                    </button>
                  </label>
                  <div
                    className={[
                      "optionButton",
                      status.videoControls ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setVideoControls(true)}
                  >
                    <div className="primary">On</div>
                    <div className="actions"></div>
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !status.videoControls ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setVideoControls(false)}
                  >
                    <div className="primary">Off</div>
                    <div className="actions"></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          {status.tab === "controlOptions" && (
            <div className="controlOptions">
              <div className="optionSection">
                <div className="optionSet">
                  <label>
                    Control Mode
                    <button
                      className="action"
                      onClick={(e) => {
                        e.stopPropagation();
                        controls.setSettingHotKey(
                          `controlOptions.toggleControls`
                        );
                      }}
                    >
                      <img src={keyboard} />
                      <span className="currentHotKey">
                        {hotKeysByAction[`controlOptions.toggleControls`]}
                      </span>
                    </button>
                  </label>
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
                        <div className="actions"></div>
                      </div>
                    );
                  })}
                </div>
                <div className="optionSet">
                  <label>
                    Always on Top
                    <button
                      className="action"
                      onClick={(e) => {
                        e.stopPropagation();
                        controls.setSettingHotKey(
                          `windowOptions.toggleAlwaysOnTop`
                        );
                      }}
                    >
                      <img src={keyboard} />
                      <span className="currentHotKey">
                        {hotKeysByAction[`windowOptions.toggleAlwaysOnTop`]}
                      </span>
                    </button>
                  </label>
                  <div
                    className={[
                      "optionButton",
                      status.isAlwaysOnTop ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setIsAlwaysOnTop(true)}
                  >
                    <div className="primary">On</div>
                    <div className="actions"></div>
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !status.isAlwaysOnTop ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setIsAlwaysOnTop(false)}
                  >
                    <div className="primary">Off</div>
                    <div className="actions"></div>
                  </div>
                </div>
                <div className="optionSet">
                  <label>
                    Fullscreen
                    <button
                      className="action"
                      onClick={(e) => {
                        e.stopPropagation();
                        controls.setSettingHotKey(
                          `windowOptions.toggleFullscreen`
                        );
                      }}
                    >
                      <img src={keyboard} />
                      <span className="currentHotKey">
                        {hotKeysByAction[`windowOptions.toggleFullscreen`]}
                      </span>
                    </button>
                  </label>
                  <div
                    className={[
                      "optionButton",
                      status.isFullScreen ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setIsFullScreen(true)}
                  >
                    <div className="primary">On</div>
                    <div className="actions"></div>
                  </div>
                  <div
                    className={[
                      "optionButton",
                      !status.isFullScreen ? "active" : null,
                    ].join(" ")}
                    onClick={() => controls.setIsFullScreen(false)}
                  >
                    <div className="primary">Off</div>
                    <div className="actions"></div>
                  </div>
                </div>
                <div className="optionSet">
                  <label>Action HotKeys</label>
                  <div className="optionButton">
                    <div className="primary">Minimize Window</div>
                    <div className="actions">
                      <button
                        className="action"
                        onClick={(e) => {
                          e.stopPropagation();
                          controls.setSettingHotKey(`windowOptions.minimize`);
                        }}
                      >
                        <img src={keyboard} />
                        <span className="currentHotKey">
                          {hotKeysByAction[`windowOptions.minimize`]}
                        </span>
                      </button>
                    </div>
                  </div>
                  <div className="optionButton">
                    <div className="primary">Next Image</div>
                    <div className="actions">
                      <button
                        className="action"
                        onClick={(e) => {
                          e.stopPropagation();
                          controls.setSettingHotKey(`fileOptions.nextImage`);
                        }}
                      >
                        <img src={keyboard} />
                        <span className="currentHotKey">
                          {hotKeysByAction[`fileOptions.nextImage`]}
                        </span>
                      </button>
                    </div>
                  </div>
                  <div className="optionButton">
                    <div className="primary">Previous Image</div>
                    <div className="actions">
                      <button
                        className="action"
                        onClick={(e) => {
                          e.stopPropagation();
                          controls.setSettingHotKey(
                            `fileOptions.previousImage`
                          );
                        }}
                      >
                        <img src={keyboard} />
                        <span className="currentHotKey">
                          {hotKeysByAction[`fileOptions.previousImage`]}
                        </span>
                      </button>
                    </div>
                  </div>
                  <div className="optionButton">
                    <div className="primary">Shuffle Images</div>
                    <div className="actions">
                      <button
                        className="action"
                        onClick={(e) => {
                          e.stopPropagation();
                          controls.setSettingHotKey(`fileOptions.shuffle`);
                        }}
                      >
                        <img src={keyboard} />
                        <span className="currentHotKey">
                          {hotKeysByAction[`fileOptions.shuffle`]}
                        </span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          {status.tab === "tagOptions" && (
            <div className="controlOptions">
              <div className="optionSection">
                <div className="optionSet">
                  <label>Tagging</label>
                  <div className="optionButton">
                    <div className="primary">Quick Tag</div>
                    <div className="actions">
                      <button
                        className="action"
                        onClick={(e) => {
                          e.stopPropagation();
                          controls.setSettingHotKey(`tagActions.addTag`);
                        }}
                      >
                        <img src={keyboard} />
                        <span className="currentHotKey">
                          {hotKeysByAction[`tagActions.addTag`]}
                        </span>
                      </button>
                    </div>
                  </div>
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
            className={status.tab === "tagOptions" ? "active" : null}
            onClick={() => controls.setTab("tagOptions")}
          >
            <img src={database} />
          </button>
        </div>
      </div>
    </div>
  );
}

export default CommandPallete;
