import React from "react";
import "./ProgressBar.css";
export default function ProgressBar({ total, value }) {
  return (
    <div
      style={{ width: `${Math.floor((value / total) * 100)}%` }}
      className="ProgressBar"
    ></div>
  );
}
