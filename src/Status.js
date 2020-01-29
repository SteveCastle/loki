import React from "react";
const electron = window.require("electron");

import { SORT, FILTER, SIZE, VIEW, CONTROL_MODE, getNext } from "./constants";

function Status({ status = {}, controls = {} }) {
  return (
    <div className="statusContainer">
      <div className="windowControls">
        <span
          className="closeControl"
          onClick={e => electron.remote.getCurrentWindow().close()}
        />
        <span
          className="windowedControl"
          onClick={e => electron.remote.getCurrentWindow().setFullScreen(false)}
        />
        <span
          className="fullScreenControl"
          onClick={e => electron.remote.getCurrentWindow().setFullScreen(true)}
        />
      </div>
      <div className="statusToast">
        <span className="statusLabel">Path</span>
        <span className="statusValue">{status.filePath}</span>
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
          onClick={() => controls.setSize(getNext(SIZE, status.size))}
        >
          {status.size}
        </span>
      </div>
      <div className="statusToast">
        <span className="statusLabel">
          Filter <strong>(S, V, G)</strong>
        </span>
        <span className="statusValue">{status.filter.toString()}</span>
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
    </div>
  );
}

export default Status;