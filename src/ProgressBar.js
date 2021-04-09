import React, { useRef } from "react";
import "./ProgressBar.css";

function mapRange(value, in_min, in_max, out_min, out_max) {
  return ((value - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min;
}

export default function ProgressBar({ total, value, setCursor }) {
  const ref = useRef();
  function handleClick(e) {
    setCursor(
      Math.floor(mapRange(e.nativeEvent.offsetX, 0, 350, 0, total - 1))
    );
  }
  return (
    <div className="ProgressBar" onClick={handleClick} ref={ref}>
      <div
        style={{ width: `${Math.floor((value / total) * 100)}%` }}
        className="progress"
      ></div>
    </div>
  );
}
