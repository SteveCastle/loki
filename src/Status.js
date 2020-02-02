import React from "react";
const electron = window.require("electron");
import HotKeyController from "./HotKeyController";

import { SORT, FILTER, SIZE, CONTROL_MODE, getNext } from "./constants";
import { getFolder, saveCurrentSettings } from "./fsTools";
function Status({ status = {}, controls = {} }, visible) {
  return (
    <div
      className={`statusContainer ${!visible ? "hidden" : ""}`}
      tabIndex="-1"
    >
      <div className="windowControls">
        <span
          className="closeControl"
          onClick={e => electron.remote.getCurrentWindow().close()}
          disabled="disabled"
          tabIndex="-1"
        />
        <span
          className="windowedControl"
          onClick={e => electron.remote.getCurrentWindow().setFullScreen(false)}
          disabled="disabled"
          tabIndex="-1"
        />
        <span
          className="fullScreenControl"
          onClick={e => electron.remote.getCurrentWindow().setFullScreen(true)}
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
      <button
        className="saveSettingsButton"
        onClick={() =>
          saveCurrentSettings({
            controlMode: status.controlMode,
            defaultSort: status.sort.key,
            scaleMode: status.size.key,
            defaultFilter: status.filter.key
          })
        }
      >
        Save Settings
      </button>
    </div>
  );
}

export default Status;
