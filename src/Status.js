import React, { useState, useEffect } from "react";
const electron = window.require("electron");
const settings = window.require("electron-settings");

import { SORT, FILTER, SIZE, CONTROL_MODE, getNext } from "./constants";
import { getFolder, saveCurrentSettings } from "./fsTools";
function Status({ status = {}, controls = {}, setAbout }) {
  const [isAlwaysOnTop, setIsAlwaysOnTop] = useState(
    electron.remote.getCurrentWindow().isAlwaysOnTop()
  );

  const [isFullScreen, setIsFullScreen] = useState(
    electron.remote.getCurrentWindow().isFullScreen()
  );
  // Sync window always on top value with state.
  useEffect(() => {
    electron.remote.getCurrentWindow().setAlwaysOnTop(isAlwaysOnTop);
  }, [isAlwaysOnTop]);

  // Sync isFullScreen with state.
  useEffect(() => {
    electron.remote.getCurrentWindow().setFullScreen(isFullScreen);
  }, [isFullScreen]);

  return (
    <div className={`statusContainer`} tabIndex="-1">
      <div className="windowControls">
        <span
          className="closeControl"
          onClick={(e) => electron.remote.getCurrentWindow().close()}
          disabled="disabled"
          tabIndex="-1"
        />
        <span
          className="windowedControl"
          onClick={(e) => setIsFullScreen(false)}
          disabled="disabled"
          tabIndex="-1"
        />
        <span
          className="fullScreenControl"
          onClick={(e) => setIsFullScreen(true)}
          disabled="disabled"
          tabIndex="-1"
        />
      </div>
      <div className="statusToast">
        <span className="statusLabel">
          Path<span className="itemCount">({status.items.length} Items)</span>
        </span>

        <span className="statusValue" onClick={controls.changePath}>
          {getFolder(status.filePath)}
        </span>
      </div>
      <div className="statusToast">
        <span className="statusLabel">
          Sort Order <strong>(S)</strong>
        </span>
        <span
          className="statusValue"
          onClick={() => controls.setSort(getNext(SORT, status.sort))}
        >
          {status.sort}
        </span>
      </div>
      <div className="statusToast">
        <span className="statusLabel">
          Image Scaling <strong>(C)</strong>
        </span>
        <span
          className="statusValue"
          onClick={() => controls.setSize(getNext(SIZE, status.size.key))}
        >
          {status.size.title}
        </span>
      </div>
      <div className="statusToast">
        <span className="statusLabel">
          Filter <strong>(A, J, V, G)</strong>
        </span>
        <span
          className="statusValue"
          onClick={() => controls.setFilter(getNext(FILTER, status.filter.key))}
        >
          {status.filter.title}
        </span>
      </div>
      <div className="statusToast">
        <span className="statusLabel">
          Control Mode <strong>(M)</strong>
        </span>
        <span
          className="statusValue"
          onClick={() =>
            controls.setControlMode(getNext(CONTROL_MODE, status.controlMode))
          }
        >
          {status.controlMode}
        </span>
      </div>
      <div className="statusToast">
        <span className="statusLabel">Recursive (R)</span>
        <span
          className="statusValue"
          onClick={() => controls.setRecursive(!status.recursive)}
        >
          {status.recursive ? "Recursive" : "Not Recursive"}
        </span>
      </div>
      <div className="statusToast">
        <span className="statusLabel">Always on Top</span>
        <span
          className="statusValue"
          onClick={(e) => setIsAlwaysOnTop(!isAlwaysOnTop)}
        >
          {isAlwaysOnTop ? "Yes" : "No"}
        </span>
      </div>
      <button
        className="saveSettingsButton"
        onClick={() =>
          saveCurrentSettings({
            controlMode: status.controlMode,
            defaultSort: status.sort.key,
            scaleMode: status.size.key,
            defaultFilter: status.filter.key,
            isAlwaysOnTop,
            isFullScreen,
          })
        }
      >
        Save Settings
      </button>
      <button className="aboutButton" onClick={() => setAbout(true)}>
        About
      </button>
    </div>
  );
}

export default Status;
