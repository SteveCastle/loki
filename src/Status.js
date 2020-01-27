import React from "react";

function Status({ status = {} }) {
  return (
    <div className="statusContainer">
      <div className="statusToast">
        <span className="statusLabel">Path</span>
        <span className="statusValue">{status.filePath}</span>
      </div>
      <div className="statusToast">
        <span className="statusLabel">Sort Order</span>
        <span className="statusValue">{status.sort}</span>
      </div>
      <div className="statusToast">
        <span className="statusLabel">Image Scaling</span>
        <span className="statusValue">{status.size}</span>
      </div>
      <div className="statusToast">
        <span className="statusLabel">Filter</span>
        <span className="statusValue">{status.filter.toString()}</span>
      </div>
      <div className="statusToast">
        <span className="statusLabel">Recursive</span>
        <span className="statusValue">
          {status.recursive ? "Recursive" : "Not Recursive"}
        </span>
      </div>
    </div>
  );
}

export default Status;
