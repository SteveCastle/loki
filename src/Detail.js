import React from "react";
const url = window.require("url");
const path = window.require("path");

import { SIZE, EXTENSIONS, VIEW } from "./constants";

function Detail({ fileName, size, handleClick }) {
  return (
    <div className="container" onClick={handleClick} tabIndex="0">
      {EXTENSIONS.img.includes(path.extname(fileName).toLowerCase()) && (
        <img
          key={fileName}
          src={url.format({
            protocol: "file",
            pathname: fileName
          })}
          className={`${size === SIZE.OVERSCAN ? "overscan" : null}`}
        />
      )}
      {EXTENSIONS.video.includes(path.extname(fileName).toLowerCase()) && (
        <video
          className={`${size === SIZE.OVERSCAN ? "overscan" : null}`}
          src={url.format({
            protocol: "file",
            pathname: fileName
          })}
          loop
          autoPlay
          muted
          controls
        />
      )}
    </div>
  );
}

export default Detail;
